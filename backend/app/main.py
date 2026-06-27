"""Application factory and process entry point.

``create_app`` assembles the FastAPI application: settings, the model manager,
CORS, exception handling, and all routers. Keeping construction in a factory
(rather than at import time) makes the app trivially configurable in tests and
avoids global side effects on import.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import analysis, health, models, tokens, visualization
from app.config import Settings, get_settings
from app.core.exceptions import VisualizerError
from app.core.model_manager import ModelManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage process-wide resources for the app's lifetime.

    On startup we construct the single :class:`ModelManager` and attach it to
    ``app.state``. On shutdown we drop the reference so cached models (and their
    embedding matrices) are released promptly.
    """
    settings: Settings = get_settings()
    app.state.manager = ModelManager(settings)
    try:
        yield
    finally:
        app.state.manager = None


def register_exception_handlers(app: FastAPI) -> None:
    """Translate domain exceptions into consistent JSON error responses."""

    @app.exception_handler(VisualizerError)
    async def _handle_visualizer_error(
        request: Request, exc: VisualizerError
    ) -> JSONResponse:
        # Every domain error carries the HTTP status it should map to, so the
        # API surface stays consistent and the core stays HTTP-agnostic.
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message, "error": exc.__class__.__name__},
        )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and return a configured FastAPI application instance."""
    settings = settings or get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Explore the token embedding spaces of transformer language models: "
            "load a model, project its embeddings to 2D/3D with UMAP, and inspect "
            "neighbors, similarities, and statistics."
        ),
        lifespan=lifespan,
        # Hide interactive docs in production unless explicitly desired.
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    # Mount routers. Order is cosmetic (affects docs grouping), not behavioral.
    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(visualization.router)
    app.include_router(tokens.router)
    app.include_router(analysis.router)

    return app


# The ASGI application object referenced by uvicorn (``app.main:app``).
app = create_app()


def main() -> None:
    """Run the development server (``python -m app.main``)."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=not settings.is_production,
        log_level="info",
    )


if __name__ == "__main__":
    main()
