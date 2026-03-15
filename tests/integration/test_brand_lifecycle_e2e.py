"""E2E test: brand extraction full lifecycle — extract, index, update, conflict."""

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
from omni_proof.rag.brand_retriever import BrandRetriever
from omni_proof.storage.memory_store import InMemoryVectorStore


def _make_metadata(**overrides):
    defaults = dict(
        asset_description="Brand asset",
        colors=BrandColorInfo(hex_codes=["#FF6B35", "#004E89"], palette_mood="warm"),
        typography=BrandTypographyInfo(
            font_styles=["sans-serif"], font_names=["Inter"], text_hierarchy="clear"
        ),
        tone=BrandToneInfo(
            formality="professional",
            emotional_register="confident",
            key_phrases=["Innovation"],
            vocabulary_themes=["technology", "trust"],
        ),
        visual=BrandVisualInfo(
            layout_pattern="asymmetric",
            motion_intensity="moderate",
            dominant_objects=["product", "logo"],
        ),
        logo_detected=True,
        media_type_detected="image",
    )
    defaults.update(overrides)
    return BrandAssetMetadata(**defaults)


@pytest.fixture
def store():
    return InMemoryVectorStore()


@pytest.fixture
def mock_embedding():
    provider = AsyncMock()
    counter = {"n": 0}

    async def varied_embedding(*args, **kwargs):
        counter["n"] += 1
        emb = [0.0] * 3072
        emb[counter["n"] % 3072] = 1.0
        emb[0] = 0.5  # common component for similarity
        return emb

    provider.generate_embedding = varied_embedding
    return provider


@pytest.fixture
def mock_gemini():
    client = AsyncMock()
    client.extract_metadata = AsyncMock(return_value=_make_metadata())
    return client


class TestBrandLifecycleE2E:
    @pytest.mark.asyncio
    async def test_extract_index_retrieve(self, mock_gemini, mock_embedding, store):
        """Extract brand → index in store → retrieve via RAG."""
        extractor = BrandExtractor(mock_embedding, mock_gemini, store)
        profile = await extractor.extract(
            "TechCorp", [Path("/tmp/guide.pdf"), Path("/tmp/logo.png"), Path("/tmp/ad.jpg")]
        )
        assert profile.brand_name == "TechCorp"
        assert len(profile.rules) >= 2

        # Verify assets are indexed and retrievable
        retriever = BrandRetriever(gemini_client=mock_embedding, vector_store=store)
        results = await retriever.search_by_text("brand colors")
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_update_compatible_merges(self, mock_gemini, mock_embedding, store):
        """Update with same-style assets should merge without conflicts."""
        extractor = BrandExtractor(mock_embedding, mock_gemini, store)
        profile = await extractor.extract(
            "TechCorp", [Path("/tmp/a.jpg"), Path("/tmp/b.jpg")]
        )
        original_count = len(profile.extractions)

        updated, conflicts = await extractor.update(profile, [Path("/tmp/c.jpg")])
        assert len(updated.extractions) == original_count + 1
        assert len(conflicts) == 0

    @pytest.mark.asyncio
    async def test_update_conflicting_colors(self, mock_gemini, mock_embedding, store):
        """Update with dramatically different colors triggers conflict."""
        extractor = BrandExtractor(mock_embedding, mock_gemini, store)
        profile = await extractor.extract(
            "TechCorp", [Path(f"/tmp/a{i}.jpg") for i in range(3)]
        )

        # Switch to completely different colors
        mock_gemini.extract_metadata = AsyncMock(
            return_value=_make_metadata(
                colors=BrandColorInfo(hex_codes=["#00FF00", "#FF00FF"], palette_mood="neon"),
            )
        )
        updated, conflicts = await extractor.update(
            profile, [Path(f"/tmp/new{i}.jpg") for i in range(4)]
        )
        assert len(conflicts) > 0

    @pytest.mark.asyncio
    async def test_multiple_asset_types(self, mock_gemini, mock_embedding, store):
        """Extract from image, PDF, and audio asset types."""
        call_count = {"n": 0}

        async def varied_metadata(*args, **kwargs):
            call_count["n"] += 1
            media_type = ["image", "document", "audio"][call_count["n"] % 3]
            return _make_metadata(media_type_detected=media_type)

        mock_gemini.extract_metadata = varied_metadata

        extractor = BrandExtractor(mock_embedding, mock_gemini, store)
        profile = await extractor.extract(
            "MultiCorp",
            [Path("/tmp/ad.jpg"), Path("/tmp/guide.pdf"), Path("/tmp/jingle.mp3")],
        )
        assert len(profile.extractions) == 3
        media_types = {e.structured_metadata.media_type_detected for e in profile.extractions}
        assert len(media_types) >= 2  # At least 2 different types
