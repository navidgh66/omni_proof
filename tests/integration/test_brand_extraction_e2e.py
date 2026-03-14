"""End-to-end test for brand extraction pipeline."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from omni_proof.brand_extraction.extractor import BrandExtractor
from omni_proof.brand_extraction.models import (
    BrandAssetMetadata,
    BrandColorInfo,
    BrandToneInfo,
    BrandTypographyInfo,
    BrandVisualInfo,
)
from omni_proof.storage.memory_store import InMemoryVectorStore


def _make_mock_metadata(**overrides):
    defaults = dict(
        asset_description="Brand ad",
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
    defaults.update(overrides)
    return BrandAssetMetadata(**defaults)


@pytest.fixture
def mock_gemini():
    client = AsyncMock()
    client.extract_metadata = AsyncMock(return_value=_make_mock_metadata())
    return client


@pytest.fixture
def mock_embedding():
    provider = AsyncMock()
    provider.generate_embedding = AsyncMock(return_value=[0.1] * 3072)
    return provider


@pytest.fixture
def memory_store():
    return InMemoryVectorStore()


class TestBrandExtractionE2E:
    @pytest.mark.asyncio
    async def test_full_extract_and_update_cycle(self, mock_gemini, mock_embedding, memory_store):
        extractor = BrandExtractor(mock_embedding, mock_gemini, memory_store)

        # Extract from initial assets
        profile = await extractor.extract(
            "AcmeCorp",
            [Path("/tmp/ad1.jpg"), Path("/tmp/ad2.jpg"), Path("/tmp/guide.pdf")],
        )

        assert profile.brand_name == "AcmeCorp"
        assert len(profile.rules) > 0
        assert "#004E89" in profile.visual_style.dominant_colors
        assert profile.voice.formality == "formal"
        assert len(profile.extractions) == 3

        # Verify indexed into vector store
        results = await memory_store.search([0.1] * 3072, top_k=10, namespace="brand_assets")
        assert len(results) > 0

        # Update with same brand patterns — no conflicts
        updated, conflicts = await extractor.update(profile, [Path("/tmp/new_ad.jpg")])
        assert len(updated.extractions) == 4
        assert len(updated.source_assets) == 4
        assert len(conflicts) == 0

    @pytest.mark.asyncio
    async def test_update_detects_conflicts(self, mock_gemini, mock_embedding, memory_store):
        extractor = BrandExtractor(mock_embedding, mock_gemini, memory_store)

        # Use 3 initial assets to establish a clear "formal" baseline
        profile = await extractor.extract(
            "AcmeCorp",
            [Path("/tmp/ad1.jpg"), Path("/tmp/ad2.jpg"), Path("/tmp/ad3.jpg")],
        )
        assert profile.voice.formality == "formal"

        # New 3 assets with conflicting voice — majority shifts to casual
        mock_gemini.extract_metadata = AsyncMock(
            return_value=_make_mock_metadata(
                tone=BrandToneInfo(
                    formality="casual",
                    emotional_register="playful",
                    key_phrases=["Let's go!"],
                    vocabulary_themes=["fun", "adventure"],
                ),
            )
        )

        updated, conflicts = await extractor.update(
            profile,
            [Path("/tmp/r1.jpg"), Path("/tmp/r2.jpg"), Path("/tmp/r3.jpg"), Path("/tmp/r4.jpg")],
        )
        # With 3 formal + 4 casual, majority is casual — conflict with original profile
        assert len(conflicts) > 0
