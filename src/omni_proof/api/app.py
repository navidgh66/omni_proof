"""FastAPI application for the Causal-Multimodal Engine."""

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from omni_proof.api.routes import brand, causal, compliance, generative, insights
from omni_proof.config.settings import Settings

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    app = FastAPI(
        title="OmniProof",
        description="Causal-Multimodal Engine for Creative Performance Attribution",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["content-type", "authorization"],
    )
    app.include_router(brand.router, prefix="/api/v1/brand", tags=["brand"])
    app.include_router(causal.router, prefix="/api/v1/causal", tags=["causal"])
    app.include_router(compliance.router, prefix="/api/v1/compliance", tags=["compliance"])
    app.include_router(insights.router, prefix="/api/v1/insights", tags=["insights"])
    app.include_router(generative.router, prefix="/api/v1/generative", tags=["generative"])

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("unhandled_error", path=request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app
