"""Causal effects API routes."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from omni_proof.api.deps import get_settings
from omni_proof.config.settings import Settings

router = APIRouter()


class AnalyzeRequest(BaseModel):
    treatment: str = Field(..., min_length=1, max_length=100, pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    outcome: str = Field(..., min_length=1, max_length=100, pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    confounders: list[str] = Field(..., min_length=1, max_length=50)


@router.get("/effects")
async def list_effects():
    return {"effects": [], "message": "No effects computed yet"}


@router.get("/effects/{treatment_name}")
async def get_effect(treatment_name: str):
    return {"treatment": treatment_name, "segments": {}, "message": "Not found"}


@router.post("/analyze")
async def analyze(request: AnalyzeRequest, settings: Settings = Depends(get_settings)):
    return {
        "status": "not_configured",
        "message": "Connect a data source to run causal analysis",
        "treatment": request.treatment,
        "outcome": request.outcome,
        "confounders": request.confounders,
    }
