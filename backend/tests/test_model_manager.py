"""Tests for the ModelManager cache, eviction, and load orchestration.

These avoid the real (network-bound) load path by monkeypatching
``_blocking_load`` to return a synthetic :class:`LoadedModel`. ``asyncio.run``
drives the async API so no extra async-test plugin is required.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from app.config import Settings
from app.core.exceptions import (
    ModelLoadError,
    ModelLoadTimeoutError,
    ModelNotAllowedError,
    ModelNotLoadedError,
)
from app.core.model_manager import ModelManager, _extract_input_embeddings
from app.core.visualizer import LoadedModel
from app.schemas import LoadState
from tests.conftest import VOCAB, FakeTokenizer


def _fake_loaded(name: str) -> LoadedModel:
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    emb = rng.standard_normal((len(VOCAB), 8)).astype(np.float32)
    return LoadedModel(name, emb, FakeTokenizer(), top_n=len(VOCAB))


def _patch_load(manager: ModelManager) -> None:
    """Replace the heavy blocking load with an instant synthetic one."""
    manager._blocking_load = lambda slot: _fake_loaded(slot.name)  # type: ignore[assignment]


def test_ensure_loaded_and_get(settings: Settings) -> None:
    manager = ModelManager(settings)
    _patch_load(manager)
    state = asyncio.run(manager.ensure_loaded("alpha"))
    assert state == LoadState.loaded
    assert manager.get("alpha").name == "alpha"


def test_get_unloaded_raises(settings: Settings) -> None:
    manager = ModelManager(settings)
    with pytest.raises(ModelNotLoadedError):
        manager.get("nope")


def test_lru_eviction(settings: Settings) -> None:
    # Cache cap is 2 (from the settings fixture).
    manager = ModelManager(settings)
    _patch_load(manager)

    async def scenario() -> None:
        await manager.ensure_loaded("a")
        await manager.ensure_loaded("b")
        manager.get("a")  # touch "a" so "b" becomes least-recently-used
        await manager.ensure_loaded("c")  # should evict "b"

    asyncio.run(scenario())
    assert manager.get("a").name == "a"
    assert manager.get("c").name == "c"
    with pytest.raises(ModelNotLoadedError):
        manager.get("b")


def test_allow_list_enforced() -> None:
    settings = Settings(allowed_models=["gpt2"])
    manager = ModelManager(settings)
    _patch_load(manager)
    with pytest.raises(ModelNotAllowedError):
        asyncio.run(manager.ensure_loaded("roberta-base"))
    # An allowed model still loads.
    assert asyncio.run(manager.ensure_loaded("gpt2")) == LoadState.loaded


def test_timeout_surfaces_error(settings: Settings) -> None:
    manager = ModelManager(settings)

    def slow_load(slot):
        import time

        time.sleep(5)
        return _fake_loaded(slot.name)

    manager._blocking_load = slow_load  # type: ignore[assignment]
    # Force an immediate timeout (bypass field validation via object.__setattr__).
    object.__setattr__(settings, "model_load_timeout_seconds", 0)

    with pytest.raises(ModelLoadTimeoutError):
        asyncio.run(manager.ensure_loaded("slowpoke"))
    assert manager.status("slowpoke")["state"] == LoadState.error


# --- graceful handling of models without usable embeddings ----------------
class _NoEmbeddingsModel:
    def get_input_embeddings(self):
        return None


class _NoWeightModel:
    def get_input_embeddings(self):
        return object()  # has no .weight


def test_extract_embeddings_missing_raises() -> None:
    with pytest.raises(ModelLoadError):
        _extract_input_embeddings(_NoEmbeddingsModel(), "vision-model")
    with pytest.raises(ModelLoadError):
        _extract_input_embeddings(_NoWeightModel(), "weird-model")


def test_load_failure_sets_error_state_and_raises(settings: Settings) -> None:
    manager = ModelManager(settings)

    def boom(slot):
        raise ModelLoadError("does not expose a token embedding table")

    manager._blocking_load = boom  # type: ignore[assignment]
    with pytest.raises(ModelLoadError):
        asyncio.run(manager.ensure_loaded("no-embeddings-model"))
    status = manager.status("no-embeddings-model")
    assert status["state"] == LoadState.error
    assert "embedding" in (status["error"] or "")
