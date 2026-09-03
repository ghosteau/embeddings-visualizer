"""Vector Embedding Visualizer — backend application package.

A FastAPI service for loading transformer models, extracting their token
embedding spaces, reducing them to 2D/3D with UMAP, and exploring the
geometry of those spaces (neighbors, similarity, comparisons, statistics).

The package is organised as:

* :mod:`app.config`         — environment-driven settings.
* :mod:`app.schemas`        — Pydantic request/response contracts.
* :mod:`app.core`           — model management and the embedding math.
* :mod:`app.api`            — HTTP routers grouped by concern.
* :mod:`app.main`           — the application factory (``create_app``).
"""

__version__ = "1.0.0"
