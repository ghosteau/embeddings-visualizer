"""Settings parsing, exercised through the environment.

These tests read the environment the way a deployed container does. That
matters: the list fields are the ones most likely to be set in production and
least likely to be set in development, so a parsing bug in them survives every
local run and only appears on the server.
"""

from __future__ import annotations

import pytest

from app.config import Settings


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate each test from the developer's real environment and .env file."""
    for name in (
        "CORS_ORIGINS",
        "ALLOWED_MODELS",
        "MAX_EMBEDDING_PARAMS",
        "MAX_DOWNLOAD_BYTES",
        "ENVIRONMENT",
    ):
        monkeypatch.delenv(name, raising=False)


def test_comma_separated_cors_origins(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CSV env value must parse, not be treated as JSON.

    pydantic-settings decodes "complex" fields (list, dict) inside the settings
    source, which runs before field validators. Without NoDecode on the field,
    a comma-separated value raised SettingsError at import time and the
    container never started.
    """
    monkeypatch.setenv("CORS_ORIGINS", "https://a.example.com,https://b.example.com")
    assert Settings(_env_file=None).cors_origins == [
        "https://a.example.com",
        "https://b.example.com",
    ]


def test_single_cors_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    """The one-value case is the production case, and is not valid JSON."""
    monkeypatch.setenv("CORS_ORIGINS", "https://embeddings.example.com")
    assert Settings(_env_file=None).cors_origins == ["https://embeddings.example.com"]


def test_comma_separated_allowed_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """ALLOWED_MODELS is the other list field and shares the same validator."""
    monkeypatch.setenv("ALLOWED_MODELS", "distilgpt2,gpt2,bert-base-uncased")
    assert Settings(_env_file=None).allowed_models == [
        "distilgpt2",
        "gpt2",
        "bert-base-uncased",
    ]


def test_list_fields_tolerate_whitespace_and_blanks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Values wrap across lines in compose files; padding must not survive."""
    monkeypatch.setenv("ALLOWED_MODELS", " distilgpt2 , gpt2 ,, ")
    assert Settings(_env_file=None).allowed_models == ["distilgpt2", "gpt2"]


def test_list_fields_default_when_unset() -> None:
    """Unset values keep the development defaults."""
    settings = Settings(_env_file=None)
    assert settings.allowed_models == []
    assert "http://localhost:5173" in settings.cors_origins


def test_json_list_still_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    """A JSON array keeps working, so existing deployments do not break."""
    monkeypatch.setenv("ALLOWED_MODELS", '["distilgpt2", "gpt2"]')
    assert Settings(_env_file=None).allowed_models == ["distilgpt2", "gpt2"]


def test_production_compose_values_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact environment docker-compose.yml sets must construct cleanly."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("CORS_ORIGINS", "https://embeddings.mannymcgrail.com")
    monkeypatch.setenv(
        "ALLOWED_MODELS",
        "distilgpt2,gpt2,distilbert-base-uncased,bert-base-uncased,roberta-base",
    )
    monkeypatch.setenv("MAX_EMBEDDING_PARAMS", "60000000")
    monkeypatch.setenv("MAX_DOWNLOAD_BYTES", "2000000000")

    settings = Settings(_env_file=None)

    assert settings.is_production
    assert settings.cors_origins == ["https://embeddings.mannymcgrail.com"]
    assert len(settings.allowed_models) == 5
    assert settings.max_embedding_params == 60_000_000
