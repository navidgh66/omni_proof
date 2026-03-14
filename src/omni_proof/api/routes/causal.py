"""Causal effects API routes."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class AnalyzeRequest(BaseModel):
    treatment: str
    outcome: str
    confounders: list[str]


@router.get("/effects")
async def list_effects():
    return {"effects": [], "message": "No effects computed yet"}


@router.get("/effects/{treatment_name}")
async def get_effect(treatment_name: str):
    return {"treatment": treatment_name, "segments": {}, "message": "Not found"}


@router.post("/analyze")
async def analyze(request: AnalyzeRequest):
    return {
        "job_id": "pending",
        "treatment": request.treatment,
        "outcome": request.outcome,
        "status": "queued",
    }
