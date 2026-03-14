"""Tests for the ingestion pipeline orchestrator."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from omni_proof.ingestion.pipeline import IngestPipeline


class TestIngestPipeline:
    @pytest.mark.asyncio
    async def test_ingest_single(self):
        mock_gemini = AsyncMock()
        mock_gemini.extract_metadata = AsyncMock(return_value={"asset_id": "123"})
        mock_gemini.generate_embedding = AsyncMock(return_value=[0.1] * 3072)

        pipeline = IngestPipeline(gemini_client=mock_gemini)
        metadata, embedding = await pipeline.ingest(Path("test.mp4"), dict)

        assert metadata == {"asset_id": "123"}
        assert len(embedding) == 3072

    @pytest.mark.asyncio
    async def test_ingest_batch_skips_failures(self):
        mock_gemini = AsyncMock()
        mock_gemini.extract_metadata = AsyncMock(side_effect=[{"id": "1"}, Exception("fail"), {"id": "3"}])
        mock_gemini.generate_embedding = AsyncMock(return_value=[0.1] * 3072)

        pipeline = IngestPipeline(gemini_client=mock_gemini)
        results = await pipeline.ingest_batch(
            [Path("a.mp4"), Path("b.mp4"), Path("c.mp4")], dict
        )

        assert len(results) == 2
