"""Tests for async relational store CRUD operations."""

from datetime import date, datetime
from uuid import uuid4

import pytest

from omni_proof.storage.relational_store import RelationalStore


@pytest.fixture
async def store():
    s = RelationalStore(database_url="sqlite+aiosqlite:///:memory:")
    await s.initialize()
    return s


class TestCreativeMetadataCRUD:
    @pytest.mark.asyncio
    async def test_create_and_get(self, store):
        asset_id = str(uuid4())
        await store.create_creative_metadata(
            {
                "asset_id": asset_id,
                "campaign_id": str(uuid4()),
                "platform": "youtube",
                "logo_screen_ratio": 0.15,
                "background_setting": "outdoor",
                "created_at": datetime.now(),
            }
        )
        result = await store.get_creative_metadata(asset_id)
        assert result is not None
        assert result["platform"] == "youtube"
        assert result["logo_screen_ratio"] == 0.15

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self, store):
        result = await store.get_creative_metadata("nonexistent")
        assert result is None


class TestPerformanceRecordCRUD:
    @pytest.mark.asyncio
    async def test_create_and_query(self, store):

        asset_id = str(uuid4())
        await store.create_performance_record(
            {
                "id": str(uuid4()),
                "asset_id": asset_id,
                "impressions": 100000,
                "clicks": 5000,
                "conversions": 250,
                "roas": 3.5,
                "ctr": 0.05,
                "audience_segment": "18-24",
                "date": date.today(),
                "platform": "youtube",
            }
        )
        results = await store.get_performance_by_asset(asset_id)
        assert len(results) == 1
        assert results[0]["roas"] == 3.5


class TestCausalDataMatrix:
    @pytest.mark.asyncio
    async def test_join_metadata_with_performance(self, store):

        asset_id = str(uuid4())
        await store.create_creative_metadata(
            {
                "asset_id": asset_id,
                "campaign_id": str(uuid4()),
                "platform": "youtube",
                "logo_screen_ratio": 0.15,
                "cta_type": "urgency",
                "created_at": datetime.now(),
            }
        )
        await store.create_performance_record(
            {
                "id": str(uuid4()),
                "asset_id": asset_id,
                "impressions": 100000,
                "clicks": 5000,
                "conversions": 250,
                "roas": 3.5,
                "ctr": 0.05,
                "audience_segment": "18-24",
                "date": date.today(),
                "platform": "youtube",
            }
        )
        matrix = await store.get_causal_data_matrix()
        assert len(matrix) == 1
        assert matrix[0]["roas"] == 3.5
        assert matrix[0]["logo_screen_ratio"] == 0.15
