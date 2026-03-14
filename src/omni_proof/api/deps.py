"""FastAPI dependency injection helpers."""

from fastapi import Request

from omni_proof.config.settings import Settings


def get_settings(request: Request) -> Settings:
    """Get application settings from request state."""
    return request.app.state.settings
