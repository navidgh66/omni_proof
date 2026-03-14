"""Generative loop API routes."""

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from omni_proof.api.generative_loop import GenerativePromptBuilder

logger = structlog.get_logger()

router = APIRouter()


class PromptRequest(BaseModel):
    target_segment: str = Field(..., min_length=1, max_length=200)
    objective: str = Field(..., min_length=1, max_length=200)
    constraints: list[str] = Field(default=[], max_length=20)


@router.post("/prompt")
async def generate_prompt(request: PromptRequest):
    try:
        builder = GenerativePromptBuilder()
        prompt = builder.build_prompt(
            cate_insights=[],
            brand_rules=[],
            target_segment=request.target_segment,
            objective=request.objective,
            constraints=request.constraints,
        )
        return {"prompt": prompt}
    except Exception:
        logger.exception("prompt_generation_failed")
        raise HTTPException(status_code=500, detail="Prompt generation failed")
