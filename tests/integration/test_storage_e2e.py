"""E2E test: relational store with synthetic data."""

from datetime import date, datetime
from uuid import uuid4

import pytest

from omni_proof.storage.relational_store import RelationalStore


class TestStorageE2E:
    @pytest.fixture
    async def store(self):
        s = RelationalStore(database_url="sqlite+aiosqlite:///:memory:")
        await s.initialize()
        return s

    @pytest.mark.asyncio
    async def test_full_data_flow(self, store):
        """Insert metadata + performance records, then query causal data matrix."""
        asset_ids = [str(uuid4()) for _ in range(5)]
        campaign_id = str(uuid4())

        # Insert creative metadata
        for i, aid in enumerate(asset_ids):
            await store.create_creative_metadata({
                "asset_id": aid,
                "campaign_id": campaign_id,
                "platform": "youtube" if i % 2 == 0 else "instagram",
                "logo_screen_ratio": 0.1 + i * 0.05,
                "cta_type": "urgency",
                "scene_transitions": 5 + i,
                "created_at": datetime.now(),
            })

        # Insert performance records
        for i, aid in enumerate(asset_ids):
            await store.create_performance_record({
                "id": str(uuid4()),
                "asset_id": aid,
                "impressions": 100000 + i * 10000,
                "clicks": 5000 + i * 500,
                "conversions": 200 + i * 50,
                "roas": 2.0 + i * 0.5,
                "ctr": 0.05 + i * 0.01,
                "audience_segment": "18-24" if i < 3 else "25-34",
                "date": date.today(),
                "platform": "youtube" if i % 2 == 0 else "instagram",
            })

        # Query causal data matrix (JOIN)
        matrix = await store.get_causal_data_matrix()
        assert len(matrix) == 5

        # Verify joined data has both metadata and performance fields
        row = matrix[0]
        assert "logo_screen_ratio" in row  # from metadata
        assert "roas" in row  # from performance
        assert "ctr" in row
        assert "platform" in row

    @pytest.mark.asyncio
    async def test_query_performance_by_asset(self, store):
        asset_id = str(uuid4())
        await store.create_creative_metadata({
            "asset_id": asset_id,
            "campaign_id": str(uuid4()),
            "platform": "tiktok",
            "created_at": datetime.now(),
        })
        for i in range(3):
            await store.create_performance_record({
                "id": str(uuid4()),
                "asset_id": asset_id,
                "impressions": 50000,
                "clicks": 2500,
                "roas": 3.0,
                "ctr": 0.05,
                "audience_segment": ["18-24", "25-34", "35-44"][i],
                "date": date.today(),
                "platform": "tiktok",
            })
        results = await store.get_performance_by_asset(asset_id)
        assert len(results) == 3
