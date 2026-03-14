"""Indexes brand guideline documents into Pinecone for RAG retrieval."""

import structlog

from omni_proof.ingestion.gemini_client import GeminiClient
from omni_proof.storage.vector_store import VectorStore

logger = structlog.get_logger()


class BrandIndexer:
    """Ingests brand guidelines, approved creatives, and palettes into vector store."""

    def __init__(self, gemini_client: GeminiClient, vector_store: VectorStore):
        self._gemini = gemini_client
        self._store = vector_store

    async def index_brand_guide_page(
        self, page_id: str, page_content_path, section_type: str, page_number: int
    ) -> None:
        embedding = await self._gemini.generate_embedding(page_content_path)
        metadata = {
            "source_type": "guideline",
            "section_type": section_type,
            "page_number": page_number,
        }
        await self._store.upsert(
            asset_id=page_id, embedding=embedding, metadata=metadata, namespace="brand_assets"
        )
        logger.info("indexed_brand_page", page_id=page_id, section=section_type)

    async def index_approved_creative(
        self, asset_id: str, asset_path, tags: list[str]
    ) -> None:
        embedding = await self._gemini.generate_embedding(asset_path)
        metadata = {
            "source_type": "approved_creative",
            "tags": tags,
        }
        await self._store.upsert(
            asset_id=asset_id, embedding=embedding, metadata=metadata, namespace="brand_assets"
        )
        logger.info("indexed_approved_creative", asset_id=asset_id)

    async def index_color_palette(
        self, palette_id: str, hex_codes: list[str], palette_name: str
    ) -> None:
        text_description = f"Brand color palette '{palette_name}': {', '.join(hex_codes)}"
        embedding = await self._gemini.generate_embedding(text_description)
        metadata = {
            "source_type": "palette",
            "section_type": "color_palette",
            "hex_codes": hex_codes,
            "palette_name": palette_name,
        }
        await self._store.upsert(
            asset_id=palette_id, embedding=embedding, metadata=metadata, namespace="brand_assets"
        )
        logger.info("indexed_palette", palette_id=palette_id)
