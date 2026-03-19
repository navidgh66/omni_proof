"""Tests for brand extraction data models."""

from datetime import UTC, datetime

from omni_proof.brand_extraction.models import (
    AssetExtraction,
    BrandAssetMetadata,
    BrandColorInfo,
    BrandConflict,
    BrandProfile,
    BrandToneInfo,
    BrandTypographyInfo,
    BrandVisualInfo,
    BrandVisualStyle,
    BrandVoice,
)
from omni_proof.rag.models import BrandRule


def _make_metadata() -> BrandAssetMetadata:
    return BrandAssetMetadata(
        asset_description="A social media ad with blue branding",
        colors=BrandColorInfo(hex_codes=["#004E89", "#FFFFFF"], palette_mood="cool"),
        typography=BrandTypographyInfo(
            font_styles=["sans-serif"],
            font_names=["Helvetica"],
            text_hierarchy="strong_hierarchy",
        ),
        tone=BrandToneInfo(
            formality="formal",
            emotional_register="authoritative",
            key_phrases=["Trust the process"],
            vocabulary_themes=["reliability", "innovation"],
        ),
        visual=BrandVisualInfo(
            layout_pattern="centered",
            motion_intensity="static",
            dominant_objects=["logo", "product"],
        ),
        logo_detected=True,
        media_type_detected="image",
    )


class TestBrandAssetMetadata:
    def test_valid_construction(self):
        meta = _make_metadata()
        assert meta.colors.hex_codes == ["#004E89", "#FFFFFF"]
        assert meta.logo_detected is True

    def test_json_roundtrip(self):
        meta = _make_metadata()
        data = meta.model_dump()
        restored = BrandAssetMetadata(**data)
        assert restored == meta


class TestAssetExtraction:
    def test_valid_construction(self):
        ext = AssetExtraction(
            asset_path="/tmp/ad.jpg",
            media_type="image",
            embedding=[0.1] * 3072,
            structured_metadata=_make_metadata(),
            extracted_at=datetime.now(UTC),
        )
        assert ext.media_type == "image"
        assert len(ext.embedding) == 3072


class TestBrandProfile:
    def test_valid_construction(self):
        now = datetime.now(UTC)
        profile = BrandProfile(
            profile_id="bp-1",
            brand_name="TestBrand",
            rules=[
                BrandRule(
                    rule_id="r1",
                    section_type="color_palette",
                    description="Use blue and white",
                    hex_codes=["#004E89", "#FFFFFF"],
                )
            ],
            voice=BrandVoice(
                formality="formal",
                emotional_register="authoritative",
                vocabulary_themes=["reliability"],
                sentence_style="short_punchy",
                confidence=0.85,
            ),
            visual_style=BrandVisualStyle(
                dominant_colors=["#004E89", "#FFFFFF"],
                color_consistency=0.9,
                typography_styles=["sans-serif"],
                layout_patterns=["centered"],
                motion_style="static",
                confidence=0.8,
            ),
            visual_fingerprint=[0.1] * 3072,
            source_assets=["/tmp/ad1.jpg", "/tmp/ad2.jpg"],
            extractions=[],
            confidence_scores={"color": 0.9, "typography": 0.8},
            created_at=now,
            updated_at=now,
        )
        assert profile.brand_name == "TestBrand"
        assert len(profile.rules) == 1


class TestBrandConflict:
    def test_valid_construction(self):
        conflict = BrandConflict(
            dimension="color_palette",
            existing_value="#004E89",
            new_value="#FF6B35",
            source_assets=["/tmp/new_ad.jpg"],
            severity="major",
        )
        assert conflict.severity == "major"
