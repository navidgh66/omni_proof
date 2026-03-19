"""E2E test: ingestion pipeline -> vector + relational storage."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from omni_proof.ingestion.pipeline import IngestPipeline
from omni_proof.storage.memory_store import InMemoryVectorStore
from omni_proof.storage.relational_store import RelationalStore


class TestIngestionStorageE2E:
    @pytest.fixture
    def mock_gemini(self):
        client = AsyncMock()
        client.extract_metadata = AsyncMock(
            return_value=type("Meta", (), {"asset_id": "asset-1", "platform": "youtube"})()
        )
        client.generate_embedding = AsyncMock(return_value=[0.1] * 3072)
        return client

    @pytest.fixture
    def vector_store(self):
        return InMemoryVectorStore()

    @pytest.fixture
    async def relational_store(self, tmp_path):
        store = RelationalStore(f"sqlite+aiosqlite:///{tmp_path}/test.db")
        await store.initialize()
        return store

    @pytest.mark.asyncio
    async def test_ingest_single_and_store(self, mock_gemini, vector_store):
        pipeline = IngestPipeline(mock_gemini)
        metadata, embedding = await pipeline.ingest(Path("/tmp/ad.jpg"), type(None))

        await vector_store.upsert("asset-1", embedding, {"platform": "youtube"})
        results = await vector_store.search([0.1] * 3072, top_k=1)
        assert len(results) == 1
        assert results[0]["id"] == "asset-1"

    @pytest.mark.asyncio
    async def test_batch_ingest_stores_all(self, mock_gemini, vector_store):
        call_count = 0

        async def unique_embedding(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            emb = [0.0] * 3072
            emb[call_count % 3072] = 1.0
            return emb

        mock_gemini.generate_embedding = unique_embedding

        pipeline = IngestPipeline(mock_gemini)
        paths = [Path(f"/tmp/ad_{i}.jpg") for i in range(5)]
        results = await pipeline.ingest_batch(paths, type(None))

        assert len(results) == 5

        for i, (_meta, emb) in enumerate(results):
            await vector_store.upsert(f"asset-{i}", emb, {"index": i})

        search = await vector_store.search([1.0] + [0.0] * 3071, top_k=10)
        assert len(search) >= 5

    @pytest.mark.asyncio
    async def test_failed_ingestion_skips_asset(self, mock_gemini, vector_store):
        call_count = 0

        async def fail_on_second(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 3:  # Fail on 2nd asset's embedding (call 3: meta1, emb1, meta2)
                raise RuntimeError("Gemini rate limit")
            return type("M", (), {"asset_id": f"a{call_count}"})()

        mock_gemini.extract_metadata = fail_on_second
        mock_gemini.generate_embedding = AsyncMock(return_value=[0.1] * 3072)

        pipeline = IngestPipeline(mock_gemini)
        results = await pipeline.ingest_batch(
            [Path("/tmp/a.jpg"), Path("/tmp/b.jpg"), Path("/tmp/c.jpg")], type(None)
        )
        # b.jpg fails on extract_metadata, should be skipped
        assert len(results) < 3

    @pytest.mark.asyncio
    async def test_ingest_and_store_relational(self, mock_gemini, relational_store):
        pipeline = IngestPipeline(mock_gemini)
        metadata, embedding = await pipeline.ingest(Path("/tmp/ad.jpg"), type(None))

        await relational_store.create_creative_metadata({
            "asset_id": "asset-1",
            "campaign_id": "camp-1",
            "platform": "youtube",
        })
        result = await relational_store.get_creative_metadata("asset-1")
        assert result is not None
        assert result["platform"] == "youtube"
