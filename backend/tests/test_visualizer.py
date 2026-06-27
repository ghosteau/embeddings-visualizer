"""Unit tests for the pure-NumPy embedding analysis in :class:`LoadedModel`."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.visualizer import LoadedModel, _classify_token
from app.schemas import DistanceMetric, TokenType
from tests.conftest import VOCAB


def test_token_preparation(loaded_model: LoadedModel) -> None:
    assert loaded_model.token_count == len(VOCAB)
    assert loaded_model.tokens[0] == "the"
    # Norms are precomputed and finite.
    assert loaded_model.embedding_norms.shape == (len(VOCAB),)
    assert np.all(np.isfinite(loaded_model.embedding_norms))


@pytest.mark.parametrize(
    "token,expected",
    [
        ("cat", TokenType.word),
        ("42", TokenType.number),
        ("<eos>", TokenType.special),
        ("@home", TokenType.mixed),
    ],
)
def test_classify_token(token: str, expected: TokenType) -> None:
    assert _classify_token(token) == expected


def test_neighbors_exclude_self_and_are_sorted(loaded_model: LoadedModel) -> None:
    neighbors = loaded_model.find_neighbors(0, n_neighbors=5, metric=DistanceMetric.euclidean)
    assert len(neighbors) == 5
    assert all(n["index"] != 0 for n in neighbors)
    distances = [n["distance"] for n in neighbors]
    assert distances == sorted(distances)  # ascending
    # Cosine similarity is always reported, even for the euclidean metric.
    assert all("similarity" in n for n in neighbors)


def test_cosine_neighbors_distance_matches_similarity(loaded_model: LoadedModel) -> None:
    neighbors = loaded_model.find_neighbors(1, n_neighbors=3, metric=DistanceMetric.cosine)
    for n in neighbors:
        assert n["distance"] == pytest.approx(1.0 - n["similarity"], abs=1e-5)


def test_compare_consistency(loaded_model: LoadedModel) -> None:
    by_id = loaded_model.compare_by_index(0, 1)
    by_name = loaded_model.compare_by_name(VOCAB[0], VOCAB[1])
    assert by_id["cosine_similarity"] == pytest.approx(by_name["cosine_similarity"])
    assert by_id["euclidean_distance"] == pytest.approx(by_name["euclidean_distance"])
    # Self-comparison: cosine similarity 1, euclidean distance 0.
    same = loaded_model.compare_by_index(2, 2)
    assert same["cosine_similarity"] == pytest.approx(1.0, abs=1e-5)
    assert same["euclidean_distance"] == pytest.approx(0.0, abs=1e-5)


def test_compare_by_name_missing_raises_keyerror(loaded_model: LoadedModel) -> None:
    with pytest.raises(KeyError):
        loaded_model.compare_by_name("the", "definitely-not-a-token")


def test_search_ranks_exact_first(loaded_model: LoadedModel) -> None:
    results = loaded_model.search_tokens("run", max_results=10)
    tokens = [r["token"] for r in results]
    assert "run" in tokens and "running" in tokens
    # Exact match "run" should precede the substring match "running".
    assert tokens.index("run") < tokens.index("running")
    assert results[0]["match_type"] == "exact"


def test_batch_similarity_matrix_is_symmetric(loaded_model: LoadedModel) -> None:
    result = loaded_model.batch_similarity(["cat", "dog", "house", "nope"])
    matrix = np.array(result["similarity_matrix"])
    assert matrix.shape == (3, 3)
    assert "nope" in result["not_found"]
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-5)
    np.testing.assert_allclose(np.diag(matrix), np.ones(3), atol=1e-5)


def test_batch_similarity_requires_two_tokens(loaded_model: LoadedModel) -> None:
    with pytest.raises(ValueError):
        loaded_model.batch_similarity(["cat", "missing"])


def test_token_details_distance(loaded_model: LoadedModel) -> None:
    details = loaded_model.get_token_details(
        3, include_distance=True, metric=DistanceMetric.euclidean
    )
    # Euclidean distance from origin equals the embedding norm.
    assert details["distance_to_origin"] == pytest.approx(
        float(loaded_model.embedding_norms[3]), abs=1e-5
    )


def test_index_out_of_range_raises(loaded_model: LoadedModel) -> None:
    with pytest.raises(IndexError):
        loaded_model.get_token_details(9999)


def test_projection_is_cached(loaded_model: LoadedModel, monkeypatch) -> None:
    """reduce_dimensions should compute once per config then serve from cache."""
    from app.schemas import VisualizationConfig

    calls = {"n": 0}

    class FakeUMAP:
        def __init__(self, **kwargs):
            pass

        def fit_transform(self, X):
            calls["n"] += 1
            return np.zeros((X.shape[0], 3), dtype=np.float32)

    import sys
    import types

    fake_umap = types.ModuleType("umap")
    fake_umap.UMAP = FakeUMAP
    monkeypatch.setitem(sys.modules, "umap", fake_umap)

    config = VisualizationConfig(n_components=3)
    first = loaded_model.reduce_dimensions(config)
    second = loaded_model.reduce_dimensions(config)
    assert calls["n"] == 1  # second call hit the cache
    assert first.shape == (len(VOCAB), 3)
    np.testing.assert_array_equal(first, second)
