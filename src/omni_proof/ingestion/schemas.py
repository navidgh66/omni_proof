"""Pydantic schemas for structured creative metadata extraction via Gemini."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from omni_proof.ingestion.enums import (
    AudioGenre,
    BackgroundSetting,
    CTAType,
    EmotionalTone,
    TypographyStyle,
    VoiceoverDemographic,
)


class VisualElements(BaseModel):
    objects_detected: list[str] = Field(description="Objects detected in the frame")
    logo_screen_ratio: float = Field(ge=0.0, le=1.0, description="Logo area as fraction of screen")
    background_setting: BackgroundSetting
    dominant_colors: list[str] = Field(description="Dominant hex color codes")
    contrast_ratio: float = Field(ge=0.0, description="WCAG contrast ratio")
    faces_detected: int = Field(ge=0, default=0)


class TemporalPacing(BaseModel):
    scene_transitions: int = Field(ge=0, description="Number of scene cuts")
    time_to_first_logo: float = Field(ge=0.0, description="Seconds until logo appears")
    product_exposure_seconds: float = Field(ge=0.0, description="Total seconds product is visible")
    motion_intensity: float = Field(ge=0.0, le=1.0, description="0=static, 1=maximum motion")
    total_duration_seconds: float = Field(ge=0.0)


class TextualElements(BaseModel):
    text_density: float = Field(ge=0.0, le=1.0, description="Fraction of screen covered by text")
    cta_type: CTAType
    promotional_text: str = Field(default="", description="Extracted promotional copy")
    typography_style: TypographyStyle


class AuditoryElements(BaseModel):
    audio_genre: AudioGenre
    voiceover_demographic: VoiceoverDemographic
    emotional_tone: EmotionalTone
    music_tempo_bpm: int = Field(ge=0, le=300, default=0)


class CreativeMetadata(BaseModel):
    asset_id: UUID
    campaign_id: UUID
    platform: str
    timestamp: datetime
    visual: VisualElements
    temporal: TemporalPacing
    textual: TextualElements
    auditory: AuditoryElements
