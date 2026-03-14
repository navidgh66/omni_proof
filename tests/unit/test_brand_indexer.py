"""Tests for brand indexer."""

from unittest.mock import AsyncMock

import pytest

from omni_proof.rag.brand_indexer import BrandIndexer


@pytest.fixture
def mock_gemini():
    client = AsyncMock()
    client.generate_embedding = AsyncMock(return_value=[0.1] * 3072)
    return client


@pytest.fixture
def mock_store():
    store = AsyncMock()
    store.upsert = AsyncMock()
    return store


@pytest.fixture
def indexer(mock_gemini, mock_store):
    return BrandIndexer(gemini_client=mock_gemini, vector_store=mock_store)


class TestBrandIndexer:
    @pytest.mark.asyncio
    async def test_index_brand_guide_page(self, indexer, mock_store):
        await indexer.index_brand_guide_page(
            page_id="page-1", page_content_path="/tmp/page1.pdf",
            section_type="logo_rules", page_number=1,
        )
        mock_store.upsert.assert_awaited_once()
        call_kwargs = mock_store.upsert.call_args.kwargs
        assert call_kwargs["namespace"] == "brand_assets"
        assert call_kwargs["metadata"]["source_type"] == "guideline"
        assert call_kwargs["metadata"]["section_type"] == "logo_rules"

    @pytest.mark.asyncio
    async def test_index_approved_creative(self, indexer, mock_store):
        await indexer.index_approved_creative(
            asset_id="creative-1", asset_path="/tmp/ad.jpg", tags=["summer", "lifestyle"],
        )
        call_kwargs = mock_store.upsert.call_args.kwargs
        assert call_kwargs["metadata"]["source_type"] == "approved_creative"
        assert call_kwargs["metadata"]["tags"] == ["summer", "lifestyle"]

    @pytest.mark.asyncio
    async def test_index_color_palette(self, indexer, mock_store, mock_gemini):
        await indexer.index_color_palette(
            palette_id="pal-1", hex_codes=["#FF6B35", "#004E89"], palette_name="Primary",
        )
        mock_gemini.generate_embedding.assert_awaited_once()
        call_kwargs = mock_store.upsert.call_args.kwargs
        assert call_kwargs["metadata"]["source_type"] == "palette"
        assert call_kwargs["metadata"]["hex_codes"] == ["#FF6B35", "#004E89"]
