"""FastAPI application for the Causal-Multimodal Engine."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from omni_proof.api.routes import causal, compliance, insights, generative


def create_app() -> FastAPI:
    app = FastAPI(
        title="OmniProof",
        description="Causal-Multimodal Engine for Creative Performance Attribution",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(causal.router, prefix="/api/v1/causal", tags=["causal"])
    app.include_router(compliance.router, prefix="/api/v1/compliance", tags=["compliance"])
    app.include_router(insights.router, prefix="/api/v1/insights", tags=["insights"])
    app.include_router(generative.router, prefix="/api/v1/generative", tags=["generative"])

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app
