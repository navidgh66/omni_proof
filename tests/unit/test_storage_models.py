"""Tests for SQLAlchemy ORM models."""

from datetime import date, datetime
from uuid import uuid4

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from omni_proof.storage.models import (
    Base,
    CampaignRecord,
    CreativeMetadataRecord,
    PerformanceRecord,
)


def _engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


class TestCreativeMetadataRecord:
    def test_create_and_read(self):
        engine = _engine()
        asset_id = str(uuid4())
        with Session(engine) as session:
            record = CreativeMetadataRecord(
                asset_id=asset_id,
                campaign_id=str(uuid4()),
                logo_screen_ratio=0.15,
                background_setting="outdoor",
                platform="instagram",
                created_at=datetime.now(),
            )
            session.add(record)
            session.commit()
            fetched = session.get(CreativeMetadataRecord, asset_id)
            assert fetched is not None
            assert fetched.logo_screen_ratio == 0.15
            assert fetched.platform == "instagram"


class TestPerformanceRecord:
    def test_create_and_read(self):
        engine = _engine()
        record_id = str(uuid4())
        with Session(engine) as session:
            record = PerformanceRecord(
                id=record_id,
                asset_id=str(uuid4()),
                impressions=100000,
                clicks=5000,
                conversions=250,
                roas=3.5,
                ctr=0.05,
                audience_segment="18-24",
                date=date.today(),
                platform="instagram",
            )
            session.add(record)
            session.commit()
            fetched = session.get(PerformanceRecord, record_id)
            assert fetched is not None
            assert fetched.roas == 3.5


class TestCampaignRecord:
    def test_create_and_read(self):
        engine = _engine()
        campaign_id = str(uuid4())
        with Session(engine) as session:
            record = CampaignRecord(
                campaign_id=campaign_id,
                name="Q3 Summer Campaign",
                start_date=date(2026, 7, 1),
                end_date=date(2026, 9, 30),
                budget=50000.00,
                target_demographics={"age": "18-34", "region": "US"},
            )
            session.add(record)
            session.commit()
            fetched = session.get(CampaignRecord, campaign_id)
            assert fetched is not None
            assert fetched.name == "Q3 Summer Campaign"
