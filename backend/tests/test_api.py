"""Integration tests for the HTTP API using FastAPI's TestClient."""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from tests.conftest import VOCAB

MODEL = "fake-model"


def test_root_and_health(client: TestClient) -> None:
    assert client.get("/").status_code == 200
    health = client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "healthy"
    assert any(m["model"] == MODEL for m in body["cached_models"])


def test_list_models(client: TestClient) -> None:
    body = client.get("/api/models").json()
    assert any(p["id"] == "gpt2" for p in body["presets"])
    assert "load_timeout_seconds" in body


def test_query_without_loaded_model_returns_409(client: TestClient) -> None:
    resp = client.get("/api/tokens/0", params={"model": "not-loaded"})
    assert resp.status_code == 409
    assert resp.json()["error"] == "ModelNotLoadedError"


def test_token_details_and_neighbors(client: TestClient) -> None:
    details = client.get("/api/tokens/0", params={"model": MODEL}).json()
    assert details["token"] == VOCAB[0]

    neighbors = client.get(
        "/api/tokens/0/neighbors", params={"model": MODEL, "n_neighbors": 5}
    ).json()
    assert len(neighbors) == 5
    assert all(n["index"] != 0 for n in neighbors)


def test_token_out_of_range_returns_404(client: TestClient) -> None:
    resp = client.get("/api/tokens/9999", params={"model": MODEL})
    assert resp.status_code == 404


def test_search(client: TestClient) -> None:
    results = client.get(
        "/api/tokens/search", params={"model": MODEL, "query": "run"}
    ).json()
    tokens = [r["token"] for r in results]
    assert "run" in tokens and "running" in tokens


def test_compare_endpoints(client: TestClient) -> None:
    by_name = client.post(
        "/api/analysis/compare",
        params={"model": MODEL},
        json={"token1": "cat", "token2": "dog"},
    )
    assert by_name.status_code == 200
    assert "cosine_similarity" in by_name.json()

    missing = client.post(
        "/api/analysis/compare",
        params={"model": MODEL},
        json={"token1": "cat", "token2": "nope"},
    )
    assert missing.status_code == 404


def test_batch_analysis(client: TestClient) -> None:
    resp = client.post(
        "/api/analysis/batch",
        params={"model": MODEL},
        json={"tokens": ["cat", "dog", "house"]},
    )
    assert resp.status_code == 200
    assert len(resp.json()["similarity_matrix"]) == 3


def test_statistics(client: TestClient) -> None:
    body = client.get("/api/analysis/statistics", params={"model": MODEL}).json()
    assert body["model_info"]["tokens_loaded"] == len(VOCAB)
    assert "by_type" in body["token_distribution"]


def test_visualization_endpoint(client: TestClient, monkeypatch) -> None:
    """The projection endpoint should return parallel coordinate/token arrays."""
    import sys
    import types

    class FakeUMAP:
        def __init__(self, **kwargs):
            self.n = kwargs.get("n_components", 3)

        def fit_transform(self, X):
            return np.zeros((X.shape[0], self.n), dtype=np.float32)

    fake_umap = types.ModuleType("umap")
    fake_umap.UMAP = FakeUMAP
    monkeypatch.setitem(sys.modules, "umap", fake_umap)

    resp = client.post(
        "/api/visualization",
        params={"model": MODEL},
        json={"n_components": 3},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["coordinates"]) == len(body["tokens"]) == len(VOCAB)
    assert body["statistics"]["reduced_dimension"] == 3


def test_validation_error_on_bad_metric(client: TestClient) -> None:
    resp = client.get("/api/tokens/0", params={"model": MODEL, "metric": "manhattan"})
    assert resp.status_code == 422
