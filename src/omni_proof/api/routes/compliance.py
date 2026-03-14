"""Brand compliance API routes."""

from fastapi import APIRouter, Depends, UploadFile

from omni_proof.api.deps import get_settings
from omni_proof.config.settings import Settings

router = APIRouter()


@router.post("/check")
async def check_compliance(file: UploadFile, settings: Settings = Depends(get_settings)):
    return {
        "asset_id": file.filename,
        "status": "not_configured",
        "message": "Configure embedding provider and brand guidelines to run compliance checks",
    }


@router.get("/reports")
async def list_reports(campaign_id: str | None = None):
    return {"reports": [], "campaign_id": campaign_id}
