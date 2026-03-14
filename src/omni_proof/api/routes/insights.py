"""Insights API routes."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/briefs")
async def list_briefs():
    return {"briefs": []}


@router.get("/segments")
async def get_segments(segment: str | None = None):
    return {"segment": segment, "effects": []}
