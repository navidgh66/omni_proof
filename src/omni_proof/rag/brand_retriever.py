"""Retrieves brand guidelines and assets from Pinecone for compliance checking."""

import structlog

from omni_proof.ingestion.gemini_client import GeminiClient
from omni_proof.rag.models import BrandAsset
from omni_proof.storage.vector_store import VectorStore

logger = structlog.get_logger()


class BrandRetriever:
    """Cross-modal search over brand assets stored in Pinecone."""

    def __init__(self, gemini_client: GeminiClient, vector_store: VectorStore):
        self._gemini = gemini_client
        self._store = vector_store

    async def search_by_text(self, query: str, top_k: int = 10) -> list[BrandAsset]:
        embedding = await self._gemini.generate_embedding(query)
        results = await self._store.search(
            query_embedding=embedding, top_k=top_k, namespace="brand_assets"
        )
        return [
            BrandAsset(
                asset_id=r["id"],
                source_type=r["metadata"].get("source_type", ""),
                section_type=r["metadata"].get("section_type", ""),
                page_number=r["metadata"].get("page_number", 0),
                tags=r["metadata"].get("tags", []),
                score=r["score"],
            )
            for r in results
        ]

    async def search_by_image(self, image_path, top_k: int = 10) -> list[BrandAsset]:
        embedding = await self._gemini.generate_embedding(image_path)
        results = await self._store.search(
            query_embedding=embedding, top_k=top_k, namespace="brand_assets"
        )
        return [
            BrandAsset(
                asset_id=r["id"],
                source_type=r["metadata"].get("source_type", ""),
                section_type=r["metadata"].get("section_type", ""),
                tags=r["metadata"].get("tags", []),
                score=r["score"],
            )
            for r in results
        ]

    async def get_guidelines_for_asset(
        self, asset_path, top_k: int = 10
    ) -> list[BrandAsset]:
        embedding = await self._gemini.generate_embedding(asset_path)
        results = await self._store.search(
            query_embedding=embedding,
            top_k=top_k,
            filters={"source_type": {"$eq": "guideline"}},
            namespace="brand_assets",
        )
        return [
            BrandAsset(
                asset_id=r["id"],
                source_type="guideline",
                section_type=r["metadata"].get("section_type", ""),
                page_number=r["metadata"].get("page_number", 0),
                score=r["score"],
            )
            for r in results
        ]
