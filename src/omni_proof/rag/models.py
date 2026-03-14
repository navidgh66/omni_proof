"""Pydantic models for brand RAG system."""

from pydantic import BaseModel, Field


class BrandAsset(BaseModel):
    asset_id: str
    source_type: str = Field(description="guideline, approved_creative, or palette")
    section_type: str = Field(default="", description="logo_rules, color_palette, typography, tone")
    page_number: int = Field(default=0)
    tags: list[str] = Field(default_factory=list)
    content_summary: str = Field(default="")
    score: float = Field(default=0.0)


class BrandRule(BaseModel):
    rule_id: str
    section_type: str
    description: str
    source_page: int = 0
    hex_codes: list[str] = Field(default_factory=list)
    min_clear_space_px: int = Field(default=0)
    approved_fonts: list[str] = Field(default_factory=list)
    tone_keywords: list[str] = Field(default_factory=list)


class ComplianceResult(BaseModel):
    passed: bool
    violations: list[dict] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    score: float = Field(default=1.0, description="1.0 = fully compliant, 0.0 = fully non-compliant")
