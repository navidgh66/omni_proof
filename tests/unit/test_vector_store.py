"""Tests for Pinecone vector store implementation."""

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from omni_proof.storage.vector_store import PineconeVectorStore


@pytest.fixture
def mock_index():
    index = MagicMock()
    index.upsert = MagicMock()
    index.query = MagicMock(return_value=MagicMock(matches=[]))
    index.delete = MagicMock()
    return index


@pytest.fixture
def store(mock_index):
    return PineconeVectorStore(index=mock_index)


class TestPineconeUpsert:
    @pytest.mark.asyncio
    async def test_upsert_calls_index(self, store, mock_index):
        asset_id = str(uuid4())
        embedding = [0.1] * 3072
        metadata = {"campaign_id": "camp1", "platform": "youtube"}

        await store.upsert(asset_id=asset_id, embedding=embedding, metadata=metadata)

        mock_index.upsert.assert_called_once()
        call_args = mock_index.upsert.call_args
        vectors = call_args.kwargs["vectors"]
        assert vectors[0][0] == asset_id
        assert len(vectors[0][1]) == 3072
        assert vectors[0][2] == metadata
        assert call_args.kwargs["namespace"] == "creatives"

    @pytest.mark.asyncio
    async def test_upsert_custom_namespace(self, store, mock_index):
        await store.upsert("id1", [0.1] * 10, {}, namespace="brand_assets")
        call_args = mock_index.upsert.call_args
        assert call_args.kwargs["namespace"] == "brand_assets"


class TestPineconeSearch:
    @pytest.mark.asyncio
    async def test_search_returns_formatted_results(self, store, mock_index):
        mock_index.query.return_value = MagicMock(
            matches=[
                MagicMock(id="id1", score=0.95, metadata={"platform": "youtube"}),
                MagicMock(id="id2", score=0.87, metadata={"platform": "instagram"}),
            ]
        )

        results = await store.search(query_embedding=[0.1] * 3072, top_k=5)

        assert len(results) == 2
        assert results[0]["id"] == "id1"
        assert results[0]["score"] == 0.95
        assert results[0]["metadata"]["platform"] == "youtube"

    @pytest.mark.asyncio
    async def test_search_with_filters(self, store, mock_index):
        mock_index.query.return_value = MagicMock(matches=[])
        filters = {"platform": {"$eq": "youtube"}}

        await store.search([0.1] * 10, top_k=5, filters=filters)

        call_args = mock_index.query.call_args
        assert call_args.kwargs["filter"] == filters

    @pytest.mark.asyncio
    async def test_search_empty_results(self, store, mock_index):
        mock_index.query.return_value = MagicMock(matches=[])
        results = await store.search([0.1] * 10)
        assert results == []


class TestPineconeDelete:
    @pytest.mark.asyncio
    async def test_delete_by_id(self, store, mock_index):
        await store.delete(asset_id="asset-123")
        mock_index.delete.assert_called_once_with(ids=["asset-123"], namespace="creatives")


class TestPineconeBatchUpsert:
    @pytest.mark.asyncio
    async def test_batch_upsert(self, store, mock_index):
        vectors = [("id1", [0.1] * 10, {"k": "v"}), ("id2", [0.2] * 10, {"k": "v2"})]
        await store.upsert_batch(vectors, batch_size=50)
        mock_index.upsert.assert_called_once_with(
            vectors=vectors, namespace="creatives", batch_size=50
        )
