"""Async SQLAlchemy relational store for structured performance data."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from omni_proof.storage.models import Base, CreativeMetadataRecord, PerformanceRecord


class RelationalStore:
    def __init__(self, database_url: str):
        self._engine = create_async_engine(database_url)
        self._session_factory = sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )

    async def initialize(self):
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def create_creative_metadata(self, data: dict) -> None:
        async with self._session_factory() as session:
            record = CreativeMetadataRecord(**data)
            session.add(record)
            await session.commit()

    async def get_creative_metadata(self, asset_id: str) -> dict | None:
        async with self._session_factory() as session:
            result = await session.get(CreativeMetadataRecord, asset_id)
            if result is None:
                return None
            return {c.name: getattr(result, c.name) for c in result.__table__.columns}

    async def create_performance_record(self, data: dict) -> None:
        async with self._session_factory() as session:
            record = PerformanceRecord(**data)
            session.add(record)
            await session.commit()

    async def get_performance_by_asset(self, asset_id: str) -> list[dict]:
        async with self._session_factory() as session:
            stmt = select(PerformanceRecord).where(PerformanceRecord.asset_id == asset_id)
            results = await session.execute(stmt)
            rows = results.scalars().all()
            return [{c.name: getattr(r, c.name) for c in r.__table__.columns} for r in rows]

    async def get_causal_data_matrix(self) -> list[dict]:
        async with self._session_factory() as session:
            stmt = select(CreativeMetadataRecord, PerformanceRecord).join(
                PerformanceRecord,
                CreativeMetadataRecord.asset_id == PerformanceRecord.asset_id,
            )
            results = await session.execute(stmt)
            rows = results.all()
            matrix = []
            for cm, pr in rows:
                row = {c.name: getattr(cm, c.name) for c in cm.__table__.columns}
                row.update({c.name: getattr(pr, c.name) for c in pr.__table__.columns})
                matrix.append(row)
            return matrix
