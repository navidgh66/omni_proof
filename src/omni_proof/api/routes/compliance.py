"""Brand compliance API routes."""

from fastapi import APIRouter, UploadFile

router = APIRouter()


@router.post("/check")
async def check_compliance(file: UploadFile):
    return {
        "asset_id": file.filename,
        "passed": True,
        "violations": [],
        "message": "Compliance check stub",
    }


@router.get("/reports")
async def list_reports(campaign_id: str | None = None):
    return {"reports": [], "campaign_id": campaign_id}
