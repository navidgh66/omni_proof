"""Brand extraction API routes."""

import re

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from omni_proof.api.deps import get_settings
from omni_proof.config.settings import Settings

router = APIRouter()

# Only allow safe filename characters in paths — no traversal
_SAFE_PATH_RE = re.compile(r"^[a-zA-Z0-9_./ -]+$")


class ExtractRequest(BaseModel):
    brand_name: str = Field(..., min_length=1, max_length=200)
    asset_paths: list[str] = Field(..., min_length=1, max_length=50)

    @field_validator("asset_paths")
    @classmethod
    def validate_paths(cls, paths: list[str]) -> list[str]:
        for p in paths:
            if ".." in p or not _SAFE_PATH_RE.match(p):
                raise ValueError(f"Invalid asset path: {p}")
        return paths


@router.post("/extract")
async def extract_brand(
    request: ExtractRequest,
    settings: Settings = Depends(get_settings),
) -> dict:
    return {
        "status": "not_configured",
        "message": "Configure embedding provider and Gemini client to run brand extraction",
        "brand_name": request.brand_name,
        "asset_count": len(request.asset_paths),
    }


@router.get("/profile/{profile_id}")
async def get_profile(profile_id: str) -> dict:
    if not profile_id.isalnum() and "-" not in profile_id:
        raise HTTPException(status_code=400, detail="Invalid profile ID")
    return {
        "status": "not_configured",
        "message": "Configure storage to retrieve brand profiles",
        "profile_id": profile_id,
    }


@router.post("/update/{profile_id}")
async def update_brand(
    profile_id: str,
    request: ExtractRequest,
    settings: Settings = Depends(get_settings),
) -> dict:
    if not profile_id.isalnum() and "-" not in profile_id:
        raise HTTPException(status_code=400, detail="Invalid profile ID")
    return {
        "status": "not_configured",
        "message": "Configure embedding provider and Gemini client to update brand profiles",
        "profile_id": profile_id,
        "brand_name": request.brand_name,
        "new_asset_count": len(request.asset_paths),
    }
