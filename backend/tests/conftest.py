"""Shared pytest fixtures.

All fixtures are fully offline: a synthetic embedding matrix and a fake
tokenizer stand in for a real Hugging Face model, so the suite runs fast and
deterministically with no network access or heavy downloads.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.core.model_manager import ModelManager, _ModelSlot
from app.core.visualizer import LoadedModel
from app.main import create_app
from app.schemas import LoadState

# A small, fixed vocabulary mixing words, numbers, and symbols so token
# classification and search have something meaningful to exercise.
VOCAB = [
    "the", "cat", "dog", "house", "run", "running", "Python", "CODE",
    "42", "7", "hello", "world", "embedding", "vector", "model", "token",
    "<eos>", "!", "@home", "ML",
]


class FakeTokenizer:
    """Minimal tokenizer that maps vocabulary ids back to known strings."""

    pad_token = "<pad>"
    eos_token = "<eos>"

    def decode(self, ids: list[int]) -> str:
        idx = ids[0]
        return VOCAB[idx] if idx < len(VOCAB) else f"tok{idx}"


@pytest.fixture
def settings() -> Settings:
    """Test settings with a small cache and short, offline-friendly limits."""
    return Settings(
        environment="development",
        max_cached_models=2,
        default_top_n=len(VOCAB),
        model_load_timeout_seconds=30,
    )


@pytest.fixture
def embeddings() -> np.ndarray:
    """A reproducible synthetic embedding matrix."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((len(VOCAB), 16)).astype(np.float32)


@pytest.fixture
def loaded_model(embeddings: np.ndarray) -> LoadedModel:
    """A fully constructed :class:`LoadedModel` over the synthetic data."""
    return LoadedModel(
        name="fake-model",
        embeddings=embeddings,
        tokenizer=FakeTokenizer(),
        top_n=len(VOCAB),
        max_cached_projections=4,
    )


@pytest.fixture
def client(settings: Settings, loaded_model: LoadedModel) -> TestClient:
    """A TestClient whose manager already has the fake model loaded.

    The model is injected directly into the manager's cache so API tests run
    without touching the network or the real loading path.
    """
    app = create_app(settings)
    with TestClient(app) as test_client:
        manager: ModelManager = app.state.manager
        slot = _ModelSlot(loaded_model.name)
        slot.model = loaded_model
        slot.state = LoadState.loaded
        manager._slots[loaded_model.name] = slot
        yield test_client
