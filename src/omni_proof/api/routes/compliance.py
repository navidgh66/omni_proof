"""Brand compliance API routes."""

import re
from uuid import uuid4

from fastapi import APIRouter, Depends, UploadFile

from omni_proof.api.deps import get_settings
from omni_proof.config.settings import Settings

router = APIRouter()


def _sanitize_filename(filename: str | None) -> str:
    """Sanitize uploaded filename to prevent injection attacks."""
    if not filename:
        return str(uuid4())
    # Strip path components and keep only safe characters
    name = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    return name or str(uuid4())


@router.post("/check")
async def check_compliance(file: UploadFile, settings: Settings = Depends(get_settings)):
    return {
        "asset_id": _sanitize_filename(file.filename),
        "status": "not_configured",
        "message": "Configure embedding provider and brand guidelines to run compliance checks",
    }


@router.get("/reports")
async def list_reports(campaign_id: str | None = None):
    return {"reports": [], "campaign_id": campaign_id}
