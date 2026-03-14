"""Brand extraction API routes."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from omni_proof.api.deps import get_settings
from omni_proof.config.settings import Settings

router = APIRouter()


class ExtractRequest(BaseModel):
    brand_name: str
    asset_paths: list[str]


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
    return {
        "status": "not_configured",
        "message": "Configure embedding provider and Gemini client to update brand profiles",
        "profile_id": profile_id,
        "brand_name": request.brand_name,
        "new_asset_count": len(request.asset_paths),
    }
