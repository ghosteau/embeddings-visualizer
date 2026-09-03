"""Application configuration.

Settings are read from environment variables (and an optional ``.env`` file)
so the same image can be deployed to any environment without code changes.
See ``.env.example`` for the full list of knobs and their defaults.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings.

    Every field can be overridden with an environment variable of the same
    name (case-insensitive), optionally prefixed via ``.env``. Validation runs
    at startup so misconfiguration fails fast and loudly.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ API --
    app_name: str = "Vector Embedding Visualizer API"
    app_version: str = "1.0.0"
    environment: str = Field(
        default="development",
        description="One of 'development' or 'production'. Controls docs exposure.",
    )

    # --------------------------------------------------------------- Network --
    host: str = "0.0.0.0"
    port: int = 8000

    # When set, FastAPI serves the compiled Vite application from this folder.
    # The production container uses this for a same-origin, single-service deploy.
    static_dir: Path | None = None

    # Projection responses are highly compressible JSON. GZip materially cuts
    # transfer size when the API and frontend are on different hosts.
    gzip_minimum_size: int = Field(default=1000, ge=0, le=1_000_000)

    # CORS: comma-separated list of allowed origins. Defaults to the common
    # Vite dev-server origins. In production set this to your real frontend URL.
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    )

    # ----------------------------------------------------------- Model cache --
    # How many distinct models may be resident in memory at once. When the
    # cache is full, the least-recently-used model is evicted. Transformer
    # embedding matrices are large, so keep this small on modest hosts.
    max_cached_models: int = Field(default=2, ge=1, le=16)

    # Hard ceiling on how long a single model load may take before it is
    # cancelled. Protects the service from huge/slow downloads hanging workers.
    model_load_timeout_seconds: int = Field(default=180, ge=10, le=1800)

    # Number of most-frequent tokens to analyze per model. The full vocabulary
    # can be tens of thousands of tokens; UMAP on all of them is slow, so we
    # prepare and project the most-frequent subset. This is the single source of
    # truth for how many points exist; the frontend decides how many to *show*.
    # 6000 covers the common research vocabulary (man/woman/dog/cat/science/king…)
    # while keeping the one-time UMAP projection reasonably fast. The frontend
    # renders a lighter default subset for smoothness; all of these stay
    # searchable / inspectable / comparable.
    default_top_n: int = Field(default=6000, ge=10, le=50000)

    # Cap on the number of cached UMAP projections (keyed by config) per model.
    max_cached_projections: int = Field(default=8, ge=1, le=64)

    # ------------------------------------------------------------- Preflight --
    # A public deployment accepts arbitrary Hugging Face ids, so a visitor can
    # name a model far larger than the host can survive. These two ceilings are
    # checked against the Hub's metadata *before* a single weight byte is
    # downloaded, turning "fill the disk, then OOM" into an immediate, clear
    # rejection. Both are generous for the text models this tool is built for:
    # the largest preset (gpt2-medium) has a 51M-parameter embedding table and
    # a ~1.5 GB download.
    #
    # Embedding rows x columns. 250M params is ~1 GB as float32, which is the
    # matrix we actually keep resident.
    max_embedding_params: int = Field(default=250_000_000, ge=1_000_000)

    # Total size of the repository's weight files. Guards disk and bandwidth,
    # which the embedding ceiling alone does not: a sharded 600 GB checkpoint
    # can carry a perfectly ordinary embedding table.
    max_download_bytes: int = Field(default=6_000_000_000, ge=100_000_000)

    # ---------------------------------------------------------- Allowed models --
    # Optional allow-list. When non-empty, only these Hugging Face model ids may
    # be loaded — important for a public deployment so visitors cannot trigger
    # arbitrary multi-gigabyte downloads. Empty list = allow any model.
    allowed_models: list[str] = Field(default_factory=list)

    @field_validator("cors_origins", "allowed_models", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Allow these list fields to be supplied as a comma-separated string.

        ``pydantic-settings`` reads env vars as strings; this lets
        ``CORS_ORIGINS="https://a.com,https://b.com"`` work as expected.
        """
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("static_dir", mode="before")
    @classmethod
    def _empty_static_dir_is_none(cls, value: object) -> object:
        """Treat a blank environment value as disabled, never as ``Path('.')``."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def is_production(self) -> bool:
        """True when running in a production-like environment."""
        return self.environment.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance.

    Cached so the ``.env`` file and environment are parsed exactly once and the
    same object is shared everywhere (and is trivially overridable in tests).
    """
    return Settings()
