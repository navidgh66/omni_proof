"""Tests for Gemini API client wrapper."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omni_proof.ingestion.gemini_client import GeminiClient


@pytest.fixture
def client():
    with patch("omni_proof.ingestion.gemini_client.GeminiClient._create_client"):
        c = GeminiClient(api_key="test-key")
    return c


class TestEmbeddingGeneration:
    @pytest.mark.asyncio
    async def test_default_dims(self, client):
        mock_response = MagicMock()
        mock_response.embeddings = [MagicMock(values=[0.1] * 3072)]
        client._client = MagicMock()
        client._client.aio.models.embed_content = AsyncMock(return_value=mock_response)

        result = await client.generate_embedding(Path("test.jpg"))
        assert len(result) == 3072

    @pytest.mark.asyncio
    async def test_truncated_dims(self, client):
        mock_response = MagicMock()
        mock_response.embeddings = [MagicMock(values=[0.1] * 768)]
        client._client = MagicMock()
        client._client.aio.models.embed_content = AsyncMock(return_value=mock_response)

        result = await client.generate_embedding(Path("test.jpg"), dimensions=768)
        assert len(result) == 768

    @pytest.mark.asyncio
    async def test_invalid_dims_rejected(self, client):
        with pytest.raises(ValueError, match="dimensions must be one of"):
            await client.generate_embedding(Path("test.jpg"), dimensions=999)


class TestMetadataExtraction:
    @pytest.mark.asyncio
    async def test_returns_parsed(self, client):
        mock_parsed = {"key": "value"}
        mock_response = MagicMock()
        mock_response.parsed = mock_parsed
        client._client = MagicMock()
        client._client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        result = await client.extract_metadata(Path("test.mp4"), dict)
        assert result == {"key": "value"}
