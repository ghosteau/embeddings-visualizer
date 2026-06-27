"""Typed domain exceptions.

The API layer translates each of these into a clean HTTP response (see
``app.main.register_exception_handlers``). Raising a specific exception from
the core keeps the business logic free of HTTP concerns while still producing
precise, actionable error messages for clients.
"""

from __future__ import annotations


class VisualizerError(Exception):
    """Base class for all application-specific errors.

    Attributes:
        message: Human-readable description, surfaced to the client.
        status_code: The HTTP status the API layer should respond with.
    """

    status_code: int = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ModelNotLoadedError(VisualizerError):
    """Raised when an operation needs a model that is not resident in memory."""

    status_code = 409  # Conflict: the resource is not in the required state.


class ModelNotAllowedError(VisualizerError):
    """Raised when a requested model id is not on the configured allow-list."""

    status_code = 403


class ModelLoadError(VisualizerError):
    """Raised when a model fails to download or initialise."""

    status_code = 502  # Bad gateway: the upstream (HF hub / model) failed.


class ModelLoadTimeoutError(ModelLoadError):
    """Raised when loading a model exceeds the configured timeout."""

    status_code = 504  # Gateway timeout.


class TokenNotFoundError(VisualizerError):
    """Raised when a token index or name cannot be resolved."""

    status_code = 404


class InvalidRequestError(VisualizerError):
    """Raised for semantically invalid requests that pass schema validation."""

    status_code = 400
