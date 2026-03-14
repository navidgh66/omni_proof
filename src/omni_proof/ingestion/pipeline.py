"""Orchestrates the full ingestion flow: preprocess -> extract metadata -> embed."""

import asyncio
from pathlib import Path

import structlog

from omni_proof.ingestion.gemini_client import GeminiClient

logger = structlog.get_logger()


class IngestPipeline:
    """Orchestrates asset ingestion through Gemini extraction and embedding."""

    def __init__(self, gemini_client: GeminiClient):
        self._gemini = gemini_client

    async def ingest(self, asset_path: Path, schema: type) -> tuple:
        logger.info("ingesting_asset", path=str(asset_path))
        metadata, embedding = await asyncio.gather(
            self._gemini.extract_metadata(asset_path, schema),
            self._gemini.generate_embedding(asset_path),
        )
        logger.info("ingestion_complete", asset_id=str(getattr(metadata, "asset_id", "unknown")))
        return metadata, embedding

    async def ingest_batch(self, asset_paths: list[Path], schema: type) -> list[tuple]:
        results = []
        for path in asset_paths:
            try:
                result = await self.ingest(path, schema)
                results.append(result)
            except Exception:
                logger.exception("ingestion_failed", path=str(path))
        return results
