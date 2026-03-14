"""Async wrapper around Gemini API for metadata extraction and embedding generation."""

import asyncio
from pathlib import Path
from typing import Any

import structlog

from omni_proof.config.constants import (
    DEFAULT_EMBEDDING_DIMS,
    GEMINI_EMBEDDING_MODEL,
    GEMINI_VISION_MODEL,
    MATRYOSHKA_DIMS,
)
from omni_proof.core.exceptions import EmbeddingError, MetadataExtractionError
from omni_proof.core.interfaces import EmbeddingProvider

logger = structlog.get_logger()


class GeminiClient(EmbeddingProvider):
    """Wraps Gemini API calls with retry logic."""

    def __init__(self, api_key: str, max_retries: int = 3):
        self._api_key = api_key
        self._max_retries = max_retries
        self._client = self._create_client()

    def _create_client(self):
        from google import genai

        return genai.Client(api_key=self._api_key)

    async def generate_embedding(
        self,
        content: str | Path,
        dimensions: int = DEFAULT_EMBEDDING_DIMS,
        task_type: str | None = None,
    ) -> list[float]:
        if dimensions not in MATRYOSHKA_DIMS:
            raise ValueError(f"dimensions must be one of {MATRYOSHKA_DIMS}, got {dimensions}")

        config = {"output_dimensionality": dimensions}
        if task_type:
            config["task_type"] = task_type

        for attempt in range(self._max_retries):
            try:
                response = await self._client.aio.models.embed_content(
                    model=GEMINI_EMBEDDING_MODEL,
                    contents=str(content),
                    config=config,
                )
                return response.embeddings[0].values
            except Exception as e:
                wait = 2**attempt
                logger.warning("embed_retry", attempt=attempt, wait=wait, error=str(e))
                if attempt == self._max_retries - 1:
                    raise EmbeddingError(
                        f"Embedding failed after {self._max_retries} retries"
                    ) from e
                await asyncio.sleep(wait)

    async def extract_metadata(self, asset_path: Path, schema: type) -> Any:
        for attempt in range(self._max_retries):
            try:
                response = await self._client.aio.models.generate_content(
                    model=GEMINI_VISION_MODEL,
                    contents=str(asset_path),
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": schema,
                    },
                )
                return response.parsed
            except Exception as e:
                wait = 2**attempt
                logger.warning("extract_retry", attempt=attempt, wait=wait, error=str(e))
                if attempt == self._max_retries - 1:
                    raise MetadataExtractionError(
                        f"Metadata extraction failed after {self._max_retries} retries"
                    ) from e
                await asyncio.sleep(wait)
