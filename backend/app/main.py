"""Application factory and process entry point.

``create_app`` assembles the FastAPI application: settings, the model manager,
CORS, exception handling, and all routers. Keeping construction in a factory
(rather than at import time) makes the app trivially configurable in tests and
avoids global side effects on import.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import analysis, health, models, tokens, visualization
from app.config import Settings, get_settings
from app.core.exceptions import VisualizerError
from app.core.model_manager import ModelManager


def build_lifespan(settings: Settings):
    """Bind one validated settings object to the application's lifetime."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.manager = ModelManager(settings)
        try:
            yield
        finally:
            app.state.manager = None

    return lifespan


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
        lifespan=build_lifespan(settings),
        # Hide interactive docs in production unless explicitly desired.
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Accept"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=settings.gzip_minimum_size)

    @app.middleware("http")
    async def add_operational_headers(request: Request, call_next):
        """Add trace and timing headers without exposing framework details."""
        request_id = uuid4().hex
        started = perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{(perf_counter() - started) * 1000:.1f}"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    register_exception_handlers(app)

    # Mount routers. Order is cosmetic (affects docs grouping), not behavioral.
    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(visualization.router)
    app.include_router(tokens.router)
    app.include_router(analysis.router)

    # A production build can be served by this process after every API route.
    # Keeping API registration first ensures the SPA mount never shadows it.
    if settings.static_dir and settings.static_dir.is_dir():
        app.mount("/", StaticFiles(directory=settings.static_dir, html=True), name="frontend")

    # Route dependencies receive the exact same settings instance as lifespan
    # and middleware, including in tests that call create_app(custom_settings).
    app.dependency_overrides[get_settings] = lambda: settings

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
