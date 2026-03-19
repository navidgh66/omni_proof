"""Tests for BrandExtractor orchestrator."""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from omni_proof.brand_extraction.extractor import BrandExtractor
from omni_proof.brand_extraction.models import (
    AssetExtraction,
    BrandAssetMetadata,
    BrandColorInfo,
    BrandProfile,
    BrandToneInfo,
    BrandTypographyInfo,
    BrandVisualInfo,
    BrandVisualStyle,
    BrandVoice,
)
from omni_proof.rag.models import BrandRule


def _make_extraction(path="/tmp/ad.jpg"):
    return AssetExtraction(
        asset_path=path,
        media_type="image",
        embedding=[0.1] * 10,
        structured_metadata=BrandAssetMetadata(
            asset_description="Test",
            colors=BrandColorInfo(hex_codes=["#004E89"], palette_mood="cool"),
            typography=BrandTypographyInfo(
                font_styles=["sans-serif"], font_names=["Helvetica"], text_hierarchy="flat"
            ),
            tone=BrandToneInfo(
                formality="formal",
                emotional_register="authoritative",
                key_phrases=[],
                vocabulary_themes=["reliability"],
            ),
            visual=BrandVisualInfo(
                layout_pattern="centered", motion_intensity="static", dominant_objects=["logo"]
            ),
            logo_detected=True,
            media_type_detected="image",
        ),
        extracted_at=datetime.now(UTC),
    )


@pytest.fixture
def mock_deps():
    embedding = AsyncMock()
    embedding.generate_embedding = AsyncMock(return_value=[0.1] * 10)
    gemini = AsyncMock()
    store = AsyncMock()
    store.upsert = AsyncMock()
    return embedding, gemini, store


class TestBrandExtractorExtract:
    @pytest.mark.asyncio
    async def test_extract_returns_brand_profile(self, mock_deps):
        embedding, gemini, store = mock_deps
        extractor = BrandExtractor(embedding, gemini, store)
        with patch.object(extractor, "_processor") as mock_proc:
            mock_proc.process_batch = AsyncMock(
                return_value=[
                    _make_extraction("/tmp/a.jpg"),
                    _make_extraction("/tmp/b.jpg"),
                ]
            )
            profile = await extractor.extract("TestBrand", [Path("/tmp/a.jpg"), Path("/tmp/b.jpg")])
        assert profile.brand_name == "TestBrand"
        assert len(profile.source_assets) == 2
        assert len(profile.rules) > 0
        assert len(profile.extractions) == 2

    @pytest.mark.asyncio
    async def test_extract_indexes_into_vector_store(self, mock_deps):
        embedding, gemini, store = mock_deps
        extractor = BrandExtractor(embedding, gemini, store)
        with patch.object(extractor, "_processor") as mock_proc:
            mock_proc.process_batch = AsyncMock(return_value=[_make_extraction()])
            await extractor.extract("TestBrand", [Path("/tmp/a.jpg")])
        assert store.upsert.await_count > 0


class TestBrandExtractorUpdate:
    @pytest.mark.asyncio
    async def test_update_detects_conflicts(self, mock_deps):
        embedding, gemini, store = mock_deps
        extractor = BrandExtractor(embedding, gemini, store)
        now = datetime.now(UTC)
        existing = BrandProfile(
            profile_id="bp-1",
            brand_name="TestBrand",
            rules=[
                BrandRule(
                    rule_id="r1",
                    section_type="color_palette",
                    description="Blue",
                    hex_codes=["#004E89"],
                )
            ],
            voice=BrandVoice(
                formality="formal",
                emotional_register="authoritative",
                vocabulary_themes=["reliability"],
                sentence_style="mixed",
                confidence=0.9,
            ),
            visual_style=BrandVisualStyle(
                dominant_colors=["#004E89"],
                color_consistency=0.9,
                typography_styles=["sans-serif"],
                layout_patterns=["centered"],
                motion_style="static",
                confidence=0.8,
            ),
            visual_fingerprint=[0.1] * 10,
            source_assets=["/tmp/old.jpg"],
            extractions=[_make_extraction("/tmp/old.jpg")],
            confidence_scores={"color": 0.9},
            created_at=now,
            updated_at=now,
        )
        # Create 2 new extractions with different tone to outvote the old one
        new_ext1 = _make_extraction("/tmp/new1.jpg")
        new_ext1.structured_metadata.tone.formality = "casual"
        new_ext1.structured_metadata.tone.emotional_register = "playful"
        new_ext2 = _make_extraction("/tmp/new2.jpg")
        new_ext2.structured_metadata.tone.formality = "casual"
        new_ext2.structured_metadata.tone.emotional_register = "playful"
        with patch.object(extractor, "_processor") as mock_proc:
            mock_proc.process_batch = AsyncMock(return_value=[new_ext1, new_ext2])
            updated, conflicts = await extractor.update(
                existing, [Path("/tmp/new1.jpg"), Path("/tmp/new2.jpg")]
            )
        assert updated.brand_name == "TestBrand"
        assert len(updated.source_assets) == 3
        assert len(updated.extractions) == 3
        assert len(conflicts) > 0
