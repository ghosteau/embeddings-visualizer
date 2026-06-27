"""Embedding analysis: the :class:`LoadedModel` value object.

A :class:`LoadedModel` is an immutable-after-construction snapshot of one
transformer's token embedding space, together with derived metadata and an
LRU cache of UMAP projections. All of the geometric queries the API exposes
(neighbors, comparisons, search, statistics) are methods here, operating purely
on NumPy arrays — there is no global state and no HTTP awareness, which makes
the maths trivial to unit-test and safe to share across concurrent requests.

The split of responsibilities is deliberate:

* :class:`LoadedModel` — *what* we know about an already-loaded model.
* :class:`~app.core.model_manager.ModelManager` — *when* models are loaded,
  cached, and evicted.
"""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from typing import Any, Optional

import numpy as np

from app.schemas import DistanceMetric, TokenType

# Regex matching any character that is not a letter, digit, or whitespace.
# Used to flag tokens containing punctuation/symbols.
_SPECIAL_CHAR_RE = re.compile(r"[^a-zA-Z0-9\s]")


def _classify_token(token: str) -> TokenType:
    """Assign a coarse lexical category to a decoded token string."""
    if token.startswith("<") and token.endswith(">"):
        return TokenType.special
    if token.isdigit():
        return TokenType.number
    if token.isalpha():
        return TokenType.word
    return TokenType.mixed


class LoadedModel:
    """A loaded model's embedding space plus derived, queryable metadata.

    Args:
        name: The Hugging Face identifier the model was loaded from.
        embeddings: The full ``(vocab_size, dim)`` raw input-embedding matrix.
        tokenizer: The model's tokenizer, used to decode token ids to strings.
        top_n: How many of the most-frequent tokens to prepare for analysis.
        max_cached_projections: Upper bound on cached UMAP projections.

    Token *frequency* is approximated by vocabulary index: most tokenizers order
    their vocabulary roughly by frequency, so the first ``top_n`` ids are a good
    proxy for "the most common tokens". This mirrors the original design while
    making the assumption explicit.
    """

    def __init__(
        self,
        name: str,
        embeddings: np.ndarray,
        tokenizer: Any,
        top_n: int,
        max_cached_projections: int = 8,
    ) -> None:
        self.name = name
        # Record the full-vocabulary dimensions for reporting, but only retain
        # the analysed subset below — holding the entire embedding matrix as
        # well would roughly double memory for large models.
        self.vocabulary_size = int(embeddings.shape[0])
        self.embedding_dimension = int(embeddings.shape[1])

        self._max_cached_projections = max_cached_projections
        # Cache of UMAP projections keyed by their config; bounded LRU.
        self._projections: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._projection_lock = threading.Lock()
        # The projection most recently built, surfaced as token x/y/z.
        self._active_projection: Optional[np.ndarray] = None

        count = min(top_n, self.vocabulary_size)
        self.selected_indices = list(range(count))
        # A contiguous, owned float32 copy of the subset we analyse against. The
        # caller drops its reference to the full matrix afterwards, freeing it.
        self.selected_embeddings = np.ascontiguousarray(
            embeddings[:count].astype(np.float32)
        )

        self._prepare_tokens(tokenizer)

    # ------------------------------------------------------------------ setup --
    def _prepare_tokens(self, tokenizer: Any) -> None:
        """Decode the selected tokens and precompute their metadata.

        Runs once at construction. Precomputing embedding norms here means
        repeated neighbor/comparison queries never recompute them.
        """
        # Precompute L2 norms once; reused by every cosine/euclidean query.
        self._norms = np.linalg.norm(self.selected_embeddings, axis=1)
        # Guard against division-by-zero for any all-zero embedding rows.
        self._safe_norms = np.where(self._norms == 0, 1e-12, self._norms)

        self.tokens: list[str] = []
        self.token_types: list[TokenType] = []
        self.lengths: list[int] = []
        self.has_special: list[bool] = []
        self.is_upper: list[bool] = []
        self.is_digit: list[bool] = []

        for idx in self.selected_indices:
            try:
                decoded = tokenizer.decode([idx]) if tokenizer is not None else ""
                token = decoded.strip() or f"<TOKEN_{idx}>"
            except Exception:
                # A handful of vocabulary ids decode to invalid byte sequences;
                # represent them with a stable placeholder rather than failing.
                token = f"<UNK_{idx}>"

            self.tokens.append(token)
            self.token_types.append(_classify_token(token))
            self.lengths.append(len(token))
            self.has_special.append(bool(_SPECIAL_CHAR_RE.search(token)))
            self.is_upper.append(token.isupper())
            self.is_digit.append(token.isdigit())

        self.token_count = len(self.tokens)

    @property
    def embedding_norms(self) -> np.ndarray:
        """L2 norms of every selected embedding (precomputed at construction)."""
        return self._norms

    # ------------------------------------------------------------- internals --
    def _require_index(self, index: int) -> None:
        """Raise ``IndexError`` if ``index`` is outside the analysed range."""
        if index < 0 or index >= self.token_count:
            raise IndexError(
                f"Token index {index} out of range (0..{self.token_count - 1})."
            )

    def _cosine_similarities(self, target: np.ndarray) -> np.ndarray:
        """Cosine similarity of ``target`` against every selected embedding.

        Fully vectorized: a single matrix-vector product plus a norm division,
        which is dramatically faster than the original per-token Python loop.
        """
        target_norm = float(np.linalg.norm(target)) or 1e-12
        dots = self.selected_embeddings @ target
        return dots / (self._safe_norms * target_norm)

    # ----------------------------------------------------------- projections --
    def reduce_dimensions(self, config) -> np.ndarray:
        """Return a UMAP projection for ``config``, computing and caching it.

        UMAP is imported lazily so merely importing this module (e.g. in tests
        that exercise the pure-NumPy methods) does not pull in the heavy
        dependency. Identical configs return the cached projection instantly.
        """
        key = (
            config.n_components,
            config.n_neighbors,
            config.min_dist,
            config.metric.value,
            self.token_count,
        )

        with self._projection_lock:
            cached = self._projections.get(key)
            if cached is not None:
                self._projections.move_to_end(key)  # mark most-recently-used
                self._active_projection = cached
                return cached

        # Compute outside the lock: UMAP can take seconds and we must not block
        # other models' projection lookups while it runs.
        import umap  # local import: heavy, optional at import time

        reducer = umap.UMAP(
            n_neighbors=min(config.n_neighbors, max(2, self.token_count - 1)),
            min_dist=config.min_dist,
            metric=config.metric.value,
            n_components=config.n_components,
            random_state=42,  # deterministic projections for reproducibility
        )
        projection = reducer.fit_transform(self.selected_embeddings).astype(np.float32)

        with self._projection_lock:
            self._projections[key] = projection
            self._projections.move_to_end(key)
            # Evict least-recently-used projections beyond the cap.
            while len(self._projections) > self._max_cached_projections:
                self._projections.popitem(last=False)
            self._active_projection = projection

        return projection

    # ------------------------------------------------------------- queries --
    def get_token_details(
        self,
        index: int,
        include_distance: bool = False,
        metric: DistanceMetric = DistanceMetric.euclidean,
    ) -> dict[str, Any]:
        """Return metadata for one token, optionally with origin distance."""
        self._require_index(index)
        details: dict[str, Any] = {
            "token": self.tokens[index],
            "index": index,
            "length": self.lengths[index],
            "type": self.token_types[index],
            "frequency_rank": self.selected_indices[index],
            "embedding_norm": float(self._norms[index]),
            "has_special_chars": self.has_special[index],
            "is_uppercase": self.is_upper[index],
            "is_digit": self.is_digit[index],
        }

        # Attach projected coordinates if a visualization has been built.
        if self._active_projection is not None:
            proj = self._active_projection
            details["x"] = float(proj[index, 0])
            details["y"] = float(proj[index, 1])
            details["z"] = float(proj[index, 2]) if proj.shape[1] > 2 else 0.0

        if include_distance:
            if metric == DistanceMetric.euclidean:
                # Distance from the origin is simply the embedding's norm.
                details["distance_to_origin"] = float(self._norms[index])
            else:
                # Cosine distance to the origin is undefined (the zero vector
                # has no direction); report 1.0 by convention for completeness.
                details["distance_to_origin"] = 1.0
            details["distance_metric"] = metric

        return details

    def find_neighbors(
        self,
        index: int,
        n_neighbors: int = 10,
        metric: DistanceMetric = DistanceMetric.euclidean,
    ) -> list[dict[str, Any]]:
        """Return the ``n_neighbors`` closest tokens to ``index``.

        Both distance and cosine similarity are returned for every neighbor so
        the frontend can display either without a second request.
        """
        self._require_index(index)
        target = self.selected_embeddings[index]

        # Cosine similarity is always computed (cheap and informative).
        similarities = self._cosine_similarities(target)

        if metric == DistanceMetric.cosine:
            distances = 1.0 - similarities
        else:  # euclidean
            distances = np.linalg.norm(self.selected_embeddings - target, axis=1)

        # argsort ascending; drop self (always distance 0 to itself).
        order = np.argsort(distances)
        order = order[order != index][:n_neighbors]

        return [
            {
                "token": self.tokens[i],
                "index": int(i),
                "distance": float(distances[i]),
                "similarity": float(similarities[i]),
            }
            for i in order
        ]

    def search_tokens(self, query: str, max_results: int = 50) -> list[dict[str, Any]]:
        """Find tokens whose text contains ``query`` (case-insensitive).

        Exact matches are surfaced first, then substring matches, so the most
        relevant results appear at the top of the list.
        """
        q = query.lower()
        exact: list[dict[str, Any]] = []
        contains: list[dict[str, Any]] = []

        for i, token in enumerate(self.tokens):
            lowered = token.lower()
            if lowered == q:
                exact.append({"token": token, "index": i, "match_type": "exact"})
            elif q in lowered:
                contains.append({"token": token, "index": i, "match_type": "contains"})
            if len(exact) + len(contains) >= max_results * 2:
                break

        return (exact + contains)[:max_results]

    def _resolve_token(self, name: str) -> Optional[int]:
        """Return the index of the first token exactly equal to ``name``."""
        try:
            return self.tokens.index(name)
        except ValueError:
            return None

    def compare_by_index(self, index1: int, index2: int) -> dict[str, Any]:
        """Compare two tokens (by index) on raw embeddings."""
        self._require_index(index1)
        self._require_index(index2)
        return self._compare(index1, index2)

    def compare_by_name(self, name1: str, name2: str) -> dict[str, Any]:
        """Compare two tokens (by text) on raw embeddings.

        Raises:
            KeyError: if either token cannot be found; the message names the
                missing token so the API can return a precise 404.
        """
        idx1 = self._resolve_token(name1)
        if idx1 is None:
            raise KeyError(name1)
        idx2 = self._resolve_token(name2)
        if idx2 is None:
            raise KeyError(name2)
        return self._compare(idx1, idx2)

    def _compare(self, idx1: int, idx2: int) -> dict[str, Any]:
        """Shared comparison maths for the by-index / by-name variants."""
        emb1 = self.selected_embeddings[idx1]
        emb2 = self.selected_embeddings[idx2]
        denom = (self._safe_norms[idx1] * self._safe_norms[idx2])
        cosine = float(np.dot(emb1, emb2) / denom)
        euclidean = float(np.linalg.norm(emb1 - emb2))
        return {
            "token1": self.tokens[idx1],
            "token2": self.tokens[idx2],
            "token1_index": idx1,
            "token2_index": idx2,
            "cosine_similarity": cosine,
            "euclidean_distance": euclidean,
        }

    def batch_similarity(self, names: list[str]) -> dict[str, Any]:
        """Compute a cosine-similarity matrix over the resolved tokens."""
        resolved: list[tuple[str, int]] = []
        not_found: list[str] = []
        seen: set[int] = set()
        for name in names:
            idx = self._resolve_token(name)
            if idx is None:
                not_found.append(name)
            elif idx not in seen:
                seen.add(idx)
                resolved.append((name, idx))

        if len(resolved) < 2:
            # Caller (API layer) turns this into a 400 with a helpful message.
            raise ValueError("Need at least two resolvable tokens to compare.")

        idxs = [idx for _, idx in resolved]
        vectors = self.selected_embeddings[idxs]
        norms = np.linalg.norm(vectors, axis=1)
        norms = np.where(norms == 0, 1e-12, norms)
        normalized = vectors / norms[:, None]
        matrix = normalized @ normalized.T

        return {
            "tokens": [name for name, _ in resolved],
            "similarity_matrix": matrix.astype(float).tolist(),
            "not_found": not_found,
        }

    def get_embedding_vector(self, index: int) -> list[float]:
        """Return the raw embedding vector for a token as a plain list."""
        self._require_index(index)
        return self.selected_embeddings[index].astype(float).tolist()

    def type_distribution(self) -> dict[str, int]:
        """Count tokens by lexical type."""
        counts: dict[str, int] = {}
        for t in self.token_types:
            counts[t.value] = counts.get(t.value, 0) + 1
        return counts

    def statistics(self) -> dict[str, Any]:
        """Compute aggregate statistics over the analysed token subset."""
        lengths = np.asarray(self.lengths)
        return {
            "model_info": {
                "name": self.name,
                "vocabulary_size": self.vocabulary_size,
                "embedding_dimension": self.embedding_dimension,
                "tokens_loaded": self.token_count,
            },
            "token_distribution": {
                "by_type": self.type_distribution(),
                "by_length": {
                    "min": int(lengths.min()),
                    "max": int(lengths.max()),
                    "mean": float(lengths.mean()),
                    "median": float(np.median(lengths)),
                },
            },
            "embedding_statistics": {
                "norm": {
                    "min": float(self._norms.min()),
                    "max": float(self._norms.max()),
                    "mean": float(self._norms.mean()),
                    "std": float(self._norms.std()),
                }
            },
            "special_characteristics": {
                "has_special_chars": int(sum(self.has_special)),
                "is_uppercase": int(sum(self.is_upper)),
                "is_digit": int(sum(self.is_digit)),
            },
        }
