"""Tests for creative metadata Pydantic schemas."""

from datetime import datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omni_proof.ingestion.enums import (
    AudioGenre,
    BackgroundSetting,
    CTAType,
    EmotionalTone,
    TypographyStyle,
    VoiceoverDemographic,
)
from omni_proof.ingestion.schemas import (
    AuditoryElements,
    CreativeMetadata,
    TemporalPacing,
    TextualElements,
    VisualElements,
)


def _sample_visual():
    return VisualElements(
        objects_detected=["product", "person", "logo"],
        logo_screen_ratio=0.15,
        background_setting=BackgroundSetting.OUTDOOR,
        dominant_colors=["#FF6B35", "#004E89", "#FFFFFF"],
        contrast_ratio=4.5,
        faces_detected=1,
    )


def _sample_temporal():
    return TemporalPacing(
        scene_transitions=8,
        time_to_first_logo=2.5,
        product_exposure_seconds=12.0,
        motion_intensity=0.75,
        total_duration_seconds=30.0,
    )


def _sample_textual():
    return TextualElements(
        text_density=0.3,
        cta_type=CTAType.URGENCY,
        promotional_text="Limited time offer!",
        typography_style=TypographyStyle.SANS_SERIF,
    )


def _sample_auditory():
    return AuditoryElements(
        audio_genre=AudioGenre.POP,
        voiceover_demographic=VoiceoverDemographic.FEMALE_YOUNG,
        emotional_tone=EmotionalTone.ENERGETIC,
        music_tempo_bpm=128,
    )


class TestVisualElements:
    def test_valid_construction(self):
        v = _sample_visual()
        assert v.logo_screen_ratio == 0.15
        assert v.background_setting == BackgroundSetting.OUTDOOR

    def test_logo_ratio_above_1_rejected(self):
        with pytest.raises(ValidationError):
            VisualElements(
                objects_detected=[],
                logo_screen_ratio=1.5,
                background_setting=BackgroundSetting.INDOOR,
                dominant_colors=[],
                contrast_ratio=1.0,
                faces_detected=0,
            )


class TestTemporalPacing:
    def test_valid_construction(self):
        t = _sample_temporal()
        assert t.scene_transitions == 8

    def test_negative_duration_rejected(self):
        with pytest.raises(ValidationError):
            TemporalPacing(
                scene_transitions=5,
                time_to_first_logo=1.0,
                product_exposure_seconds=-1.0,
                motion_intensity=0.5,
                total_duration_seconds=30.0,
            )


class TestCreativeMetadata:
    def test_composite_model(self):
        asset_id = uuid4()
        meta = CreativeMetadata(
            asset_id=asset_id,
            campaign_id=uuid4(),
            platform="instagram",
            timestamp=datetime.now(),
            visual=_sample_visual(),
            temporal=_sample_temporal(),
            textual=_sample_textual(),
            auditory=_sample_auditory(),
        )
        assert meta.asset_id == asset_id
        assert meta.platform == "instagram"

    def test_json_roundtrip(self):
        meta = CreativeMetadata(
            asset_id=uuid4(),
            campaign_id=uuid4(),
            platform="youtube",
            timestamp=datetime.now(),
            visual=_sample_visual(),
            temporal=_sample_temporal(),
            textual=_sample_textual(),
            auditory=_sample_auditory(),
        )
        json_str = meta.model_dump_json()
        restored = CreativeMetadata.model_validate_json(json_str)
        assert restored.asset_id == meta.asset_id

    def test_json_schema_export(self):
        schema = CreativeMetadata.model_json_schema()
        assert "properties" in schema
        for key in ["asset_id", "visual", "temporal", "textual", "auditory"]:
            assert key in schema["properties"]
