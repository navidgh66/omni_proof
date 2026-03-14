"""Generative loop API routes."""

from fastapi import APIRouter
from pydantic import BaseModel

from omni_proof.api.generative_loop import GenerativePromptBuilder

router = APIRouter()
_builder = GenerativePromptBuilder()


class PromptRequest(BaseModel):
    target_segment: str
    objective: str
    constraints: list[str] = []


@router.post("/prompt")
async def generate_prompt(request: PromptRequest):
    prompt = _builder.build_prompt(
        cate_insights=[],
        brand_rules=[],
        target_segment=request.target_segment,
        objective=request.objective,
        constraints=request.constraints,
    )
    return {"prompt": prompt}
