"""Tests for brand retriever."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from omni_proof.rag.brand_retriever import BrandRetriever


@pytest.fixture
def mock_gemini():
    client = AsyncMock()
    client.generate_embedding = AsyncMock(return_value=[0.1] * 3072)
    return client


@pytest.fixture
def mock_store():
    store = AsyncMock()
    store.search = AsyncMock(
        return_value=[
            {
                "id": "asset-1",
                "score": 0.95,
                "metadata": {
                    "source_type": "guideline",
                    "section_type": "logo_rules",
                    "page_number": 2,
                },
            },
            {
                "id": "asset-2",
                "score": 0.88,
                "metadata": {"source_type": "approved_creative", "tags": ["summer"]},
            },
        ]
    )
    return store


@pytest.fixture
def retriever(mock_gemini, mock_store):
    return BrandRetriever(gemini_client=mock_gemini, vector_store=mock_store)


class TestBrandRetriever:
    @pytest.mark.asyncio
    async def test_search_by_text(self, retriever):
        results = await retriever.search_by_text("warm contemporary photography")
        assert len(results) == 2
        assert results[0].asset_id == "asset-1"
        assert results[0].score == 0.95
        assert results[0].source_type == "guideline"

    @pytest.mark.asyncio
    async def test_search_by_image(self, retriever):
        results = await retriever.search_by_image(Path("/tmp/test.jpg"))
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_get_guidelines_filters_by_type(self, retriever, mock_store):
        mock_store.search = AsyncMock(
            return_value=[
                {
                    "id": "g-1",
                    "score": 0.92,
                    "metadata": {
                        "source_type": "guideline",
                        "section_type": "typography",
                        "page_number": 5,
                    },
                },
            ]
        )
        results = await retriever.get_guidelines_for_asset(Path("/tmp/ad.mp4"))
        assert len(results) == 1
        assert results[0].source_type == "guideline"
        # Verify filter was passed
        call_kwargs = mock_store.search.call_args.kwargs
        assert call_kwargs["filters"] == {"source_type": {"$eq": "guideline"}}
