"""Pydantic models for orchestration layer."""

from pydantic import BaseModel, Field


class Violation(BaseModel):
    rule_type: str = Field(description="concrete or semantic")
    severity: str = Field(description="critical, warning, or info")
    description: str
    evidence: str = Field(default="")


class ComplianceReport(BaseModel):
    asset_id: str
    passed: bool
    violations: list[Violation] = Field(default_factory=list)
    evidence_sources: list[str] = Field(default_factory=list)
    score: float = Field(default=1.0)


class DesignBrief(BaseModel):
    treatment: str
    finding: str
    segment_breakdown: dict[str, str] = Field(default_factory=dict)
    recommendation: str
    confidence: str = Field(default="unknown")
