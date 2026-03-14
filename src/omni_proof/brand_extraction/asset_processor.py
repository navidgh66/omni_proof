"""Processes individual assets through Gemini Flash and Embedding 2."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import structlog

from omni_proof.brand_extraction.models import AssetExtraction, BrandAssetMetadata
from omni_proof.core.exceptions import IngestionError
from omni_proof.core.interfaces import EmbeddingProvider
from omni_proof.ingestion.gemini_client import GeminiClient

logger = structlog.get_logger()

MEDIA_TYPE_MAP = {
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".gif": "image",
    ".webp": "image",
    ".bmp": "image",
    ".mp4": "video",
    ".avi": "video",
    ".mov": "video",
    ".webm": "video",
    ".mp3": "audio",
    ".wav": "audio",
    ".ogg": "audio",
    ".flac": "audio",
    ".pdf": "pdf",
}


class AssetProcessor:
    def __init__(self, embedding_provider: EmbeddingProvider, gemini_client: GeminiClient):
        self._embedding = embedding_provider
        self._gemini = gemini_client

    def _detect_media_type(self, path: Path) -> str:
        return MEDIA_TYPE_MAP.get(path.suffix.lower(), "image")

    async def process(self, asset_path: Path) -> AssetExtraction:
        media_type = self._detect_media_type(asset_path)
        metadata, embedding = await asyncio.gather(
            self._gemini.extract_metadata(asset_path, BrandAssetMetadata),
            self._embedding.generate_embedding(asset_path, task_type="SEMANTIC_SIMILARITY"),
        )
        return AssetExtraction(
            asset_path=str(asset_path),
            media_type=media_type,
            embedding=embedding,
            structured_metadata=metadata,
            extracted_at=datetime.now(timezone.utc),
        )

    async def process_batch(self, assets: list[Path]) -> list[AssetExtraction]:
        results: list[AssetExtraction] = []
        for path in assets:
            try:
                result = await self.process(path)
                results.append(result)
            except Exception:
                logger.exception("brand_asset_processing_failed", path=str(path))
        if not results:
            raise IngestionError(f"All {len(assets)} assets failed during brand extraction")
        return results
