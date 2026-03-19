"""Data models for brand identity extraction pipeline."""

from datetime import datetime

from pydantic import BaseModel, Field

from omni_proof.rag.models import BrandRule


class BrandColorInfo(BaseModel):
    hex_codes: list[str]
    palette_mood: str


class BrandTypographyInfo(BaseModel):
    font_styles: list[str]
    font_names: list[str]
    text_hierarchy: str


class BrandToneInfo(BaseModel):
    formality: str
    emotional_register: str
    key_phrases: list[str]
    vocabulary_themes: list[str]


class BrandVisualInfo(BaseModel):
    layout_pattern: str
    motion_intensity: str
    dominant_objects: list[str]


class BrandAssetMetadata(BaseModel):
    asset_description: str
    colors: BrandColorInfo
    typography: BrandTypographyInfo
    tone: BrandToneInfo
    visual: BrandVisualInfo
    logo_detected: bool
    media_type_detected: str


class AssetExtraction(BaseModel):
    asset_path: str
    media_type: str
    embedding: list[float]
    structured_metadata: BrandAssetMetadata
    extracted_at: datetime


class BrandVoice(BaseModel):
    formality: str
    emotional_register: str
    vocabulary_themes: list[str]
    sentence_style: str
    confidence: float


class BrandVisualStyle(BaseModel):
    dominant_colors: list[str]
    color_consistency: float
    typography_styles: list[str]
    layout_patterns: list[str]
    motion_style: str
    confidence: float


class BrandConflict(BaseModel):
    dimension: str
    existing_value: str
    new_value: str
    source_assets: list[str]
    severity: str


class BrandProfile(BaseModel):
    profile_id: str
    brand_name: str
    rules: list[BrandRule]
    voice: BrandVoice
    visual_style: BrandVisualStyle
    visual_fingerprint: list[float]
    source_assets: list[str]
    extractions: list[AssetExtraction] = Field(default_factory=list)
    confidence_scores: dict[str, float] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
