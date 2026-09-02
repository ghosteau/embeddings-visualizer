"""Model lifecycle management: loading, caching, and eviction.

The :class:`ModelManager` is the single owner of loaded models. It exists to
make the service safe and economical when many users hit it at once:

* **Shared by name.** Two users asking for ``gpt2`` share one in-memory copy.
* **LRU-bounded.** At most ``max_cached_models`` live at once; the least
  recently used is evicted when a new one is loaded, capping memory use.
* **Concurrency-safe.** A per-model async lock ensures a model is loaded only
  once even under a burst of simultaneous requests; cache mutations are guarded.
* **Time-bounded.** Loads run in a worker thread under :func:`asyncio.wait_for`,
  so a slow or stuck download cannot hang a request indefinitely.

The heavy ``transformers`` import is deferred to load time so the module (and
the tests around it) stay lightweight.
"""

from __future__ import annotations

import asyncio
import gc
from collections import OrderedDict
from typing import Optional

import numpy as np

from app.config import Settings
from app.core.exceptions import (
    ModelLoadError,
    ModelLoadTimeoutError,
    ModelNotAllowedError,
    ModelNotLoadedError,
)
from app.core.visualizer import LoadedModel
from app.schemas import LoadState


def _friendly_hf_error(name: str, exc: Exception, what: str) -> str:
    """Turn a raw Hugging Face exception into a clear, user-facing message.

    ``what`` is "tokenizer" or "model" to indicate which stage failed.
    """
    text = str(exc).lower()
    if "trust_remote_code" in text:
        return (
            f"'{name}' ships custom code that must be trusted to run. For safety "
            f"this server only loads standard architectures."
        )
    if any(s in text for s in ("401", "403", "gated", "authentication", "is not authorized")):
        return f"'{name}' is private or gated and can't be loaded without credentials."
    if any(s in text for s in ("404", "not found", "does not appear", "repository not found")):
        return f"Model '{name}' was not found on the Hugging Face Hub. Check the id."
    if any(s in text for s in ("connection", "offline", "couldn't reach", "timed out", "proxy")):
        return f"Couldn't reach the Hugging Face Hub to download '{name}'. Check your connection."
    # Fall back to a trimmed version of the underlying error.
    detail = str(exc).splitlines()[0][:200]
    return f"Could not load the {what} for '{name}': {detail}"


def _extract_input_embeddings(model: object, name: str) -> np.ndarray:
    """Pull the token input-embedding matrix out of a loaded model.

    Not every architecture exposes a usable token embedding table (vision
    models, audio models, some custom heads, encoder-only setups with tied or
    absent input embeddings). Rather than letting an ``AttributeError`` bubble
    up as an opaque 500, we validate and raise a clear, user-facing message.
    """
    getter = getattr(model, "get_input_embeddings", None)
    layer = getter() if callable(getter) else None
    weight = getattr(layer, "weight", None) if layer is not None else None
    if weight is None:
        raise ModelLoadError(
            f"'{name}' does not expose a token embedding table, so its embedding "
            f"space can't be visualized. Try a text model such as gpt2 or bert-base-uncased."
        )

    matrix = weight.data.detach().cpu().numpy()
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ModelLoadError(
            f"'{name}' has an unexpected embedding shape {tuple(matrix.shape)}; "
            f"it can't be visualized."
        )
    return matrix


class _ModelSlot:
    """Mutable per-model bookkeeping held in the cache.

    Tracks loading state and progress separately from the heavy
    :class:`LoadedModel`, so a model can advertise ``loading``/``error`` status
    before (or instead of) holding a fully built model object.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.state: LoadState = LoadState.not_loaded
        self.progress: Optional[str] = None
        self.error: Optional[str] = None
        self.model: Optional[LoadedModel] = None
        # Guards loading so concurrent callers don't trigger duplicate loads.
        self.lock = asyncio.Lock()


class ModelManager:
    """Owns the set of loaded models and orchestrates their lifecycle."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # Insertion-ordered cache; order is maintained as an LRU list.
        self._slots: "OrderedDict[str, _ModelSlot]" = OrderedDict()
        # Guards structural mutations of ``_slots`` (insert/evict/reorder).
        self._cache_lock = asyncio.Lock()

    # -------------------------------------------------------------- helpers --
    def _check_allowed(self, name: str) -> None:
        """Enforce the optional allow-list of loadable model ids."""
        allowed = self._settings.allowed_models
        if allowed and name not in allowed:
            raise ModelNotAllowedError(
                f"Model '{name}' is not permitted on this server. "
                f"Allowed models: {', '.join(allowed)}."
            )

    async def _get_or_create_slot(self, name: str) -> _ModelSlot:
        """Return the slot for ``name``, creating and registering it if new."""
        async with self._cache_lock:
            slot = self._slots.get(name)
            if slot is None:
                slot = _ModelSlot(name)
                self._slots[name] = slot
            self._slots.move_to_end(name)  # most-recently-used
            return slot

    async def _evict_if_needed(self) -> None:
        """Evict least-recently-used *loaded* models above the cache cap."""
        async with self._cache_lock:
            while len(self._slots) > self._settings.max_cached_models:
                # Find the oldest slot that is safe to drop (not mid-load).
                evicted = None
                for key, slot in self._slots.items():
                    if slot.state != LoadState.loading:
                        evicted = key
                        break
                if evicted is None:
                    break  # everything is loading; nothing safe to evict yet
                self._slots.pop(evicted)

    # --------------------------------------------------------------- public --
    async def ensure_loaded(self, name: str) -> LoadState:
        """Ensure ``name`` is loaded, loading it (once) if necessary.

        Returns the resulting :class:`LoadState`. This is safe to call
        concurrently: the per-slot lock collapses a burst of callers into a
        single load, and everyone observes the final state.
        """
        self._check_allowed(name)
        slot = await self._get_or_create_slot(name)

        async with slot.lock:
            if slot.state == LoadState.loaded:
                return slot.state

            slot.state = LoadState.loading
            slot.progress = "Initializing…"
            slot.error = None
            try:
                model = await asyncio.wait_for(
                    asyncio.to_thread(self._blocking_load, slot),
                    timeout=self._settings.model_load_timeout_seconds,
                )
            except asyncio.TimeoutError:
                slot.state = LoadState.error
                slot.error = (
                    f"Loading timed out after "
                    f"{self._settings.model_load_timeout_seconds}s."
                )
                raise ModelLoadTimeoutError(slot.error)
            except ModelLoadError as exc:
                # Already a precise, user-facing error — record it on the slot
                # (so /status reflects the failure) and re-raise unchanged.
                slot.state = LoadState.error
                slot.error = exc.message
                raise
            except Exception as exc:  # pragma: no cover - defensive catch-all
                slot.state = LoadState.error
                slot.error = str(exc)
                raise ModelLoadError(f"Failed to load '{name}': {exc}") from exc

            slot.model = model
            slot.state = LoadState.loaded
            slot.progress = "Loaded."

        await self._evict_if_needed()
        return slot.state

    def _blocking_load(self, slot: _ModelSlot) -> LoadedModel:
        """Synchronously download and prepare a model (runs in a thread).

        Kept deliberately small and self-contained so it can be wrapped by
        :func:`asyncio.wait_for`. Heavy imports happen here, off the event loop.
        """
        from transformers import AutoModel, AutoTokenizer

        name = slot.name
        slot.progress = "Downloading tokenizer…"
        try:
            tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=False)
        except Exception as exc:
            raise ModelLoadError(_friendly_hf_error(name, exc, "tokenizer")) from exc

        # GPT-2 family ships without a pad token; align it with EOS so the
        # tokenizer is well-formed even though we don't pad during analysis.
        if getattr(tokenizer, "pad_token", None) is None and getattr(
            tokenizer, "eos_token", None
        ):
            tokenizer.pad_token = tokenizer.eos_token

        slot.progress = "Downloading model weights…"
        try:
            model = AutoModel.from_pretrained(
                name,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
            )
        except Exception as exc:
            raise ModelLoadError(_friendly_hf_error(name, exc, "model")) from exc

        slot.progress = "Extracting embeddings…"
        embeddings = _extract_input_embeddings(model, name)

        slot.progress = "Preparing tokens…"
        loaded = LoadedModel(
            name=name,
            embeddings=embeddings,
            tokenizer=tokenizer,
            top_n=self._settings.default_top_n,
            max_cached_projections=self._settings.max_cached_projections,
        )
        # Release full weights immediately; the service only retains the
        # compact analysed embedding subset owned by ``loaded``.
        del embeddings
        del model
        gc.collect()
        return loaded

    def get(self, name: str) -> LoadedModel:
        """Return a loaded model or raise :class:`ModelNotLoadedError`.

        Touches the LRU ordering so actively-queried models resist eviction.
        """
        slot = self._slots.get(name)
        if slot is None or slot.model is None or slot.state != LoadState.loaded:
            raise ModelNotLoadedError(
                f"Model '{name}' is not loaded. Load it via POST /api/models/load first."
            )
        self._slots.move_to_end(name)
        return slot.model

    def status(self, name: str) -> dict[str, object]:
        """Return a status snapshot for ``name`` (without raising)."""
        slot = self._slots.get(name)
        if slot is None:
            return {"model": name, "state": LoadState.not_loaded, "progress": None, "error": None}
        return {
            "model": name,
            "state": slot.state,
            "progress": slot.progress,
            "error": slot.error,
        }

    def all_statuses(self) -> list[dict[str, object]]:
        """Return status snapshots for every model the manager knows about."""
        return [self.status(name) for name in self._slots]

    async def unload(self, name: str) -> bool:
        """Evict a model from the cache. Returns True if one was removed."""
        async with self._cache_lock:
            return self._slots.pop(name, None) is not None
