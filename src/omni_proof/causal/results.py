"""Pydantic models for causal estimation results."""

from pydantic import BaseModel, Field


class EffectEstimate(BaseModel):
    effect: float
    ci_lower: float
    ci_upper: float
    p_value: float = Field(default=0.0)


class ATEResult(BaseModel):
    treatment: str
    outcome: str
    ate: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n_samples: int


class CATEResult(BaseModel):
    treatment: str
    outcome: str
    segments: dict[str, EffectEstimate] = Field(default_factory=dict)
    refutation_passed: bool = False


class RefutationResult(BaseModel):
    test_name: str
    original_effect: float
    new_effect: float
    passed: bool
    p_value: float = 0.0
