"""E2E test: RAG indexing -> retrieval -> compliance checking with real vector store."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from omni_proof.orchestration.compliance_chain import ComplianceChain
from omni_proof.orchestration.models import Violation
from omni_proof.rag.brand_indexer import BrandIndexer
from omni_proof.rag.brand_retriever import BrandRetriever
from omni_proof.storage.memory_store import InMemoryVectorStore


@pytest.fixture
def store():
    return InMemoryVectorStore()


@pytest.fixture
def mock_embedding():
    """Returns embeddings based on content for basic similarity."""
    provider = AsyncMock()

    async def content_based_embedding(content, *args, **kwargs):
        content_str = str(content)
        emb = [0.0] * 3072
        # Create a deterministic but content-dependent embedding
        for _i, c in enumerate(content_str):
            emb[ord(c) % 3072] += 0.1
        # Normalize
        norm = sum(x**2 for x in emb) ** 0.5
        if norm > 0:
            emb = [x / norm for x in emb]
        return emb

    provider.generate_embedding = content_based_embedding
    return provider


class TestRAGComplianceE2E:
    @pytest.mark.asyncio
    async def test_index_and_retrieve_guidelines(self, mock_embedding, store):
        """Index brand guidelines, then retrieve them via text query."""
        indexer = BrandIndexer(gemini_client=mock_embedding, vector_store=store)

        await indexer.index_brand_guide_page(
            "page-1", "Brand color palette: blue #004E89, accent orange #FF6B35",
            section_type="color_palette", page_number=1,
        )
        await indexer.index_brand_guide_page(
            "page-2", "Typography: Inter for body, Playfair Display for headers",
            section_type="typography", page_number=2,
        )
        await indexer.index_color_palette(
            "palette-1", ["#004E89", "#FF6B35"], "primary"
        )

        retriever = BrandRetriever(gemini_client=mock_embedding, vector_store=store)
        results = await retriever.search_by_text("color palette blue orange")
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_compliance_with_real_retriever(self, mock_embedding, store):
        """Full chain: index guidelines → retrieve → compliance check."""
        indexer = BrandIndexer(gemini_client=mock_embedding, vector_store=store)
        await indexer.index_brand_guide_page(
            "rule-1", "Brand must use blue #004E89 as primary color",
            section_type="color_palette", page_number=1,
        )

        retriever = BrandRetriever(gemini_client=mock_embedding, vector_store=store)
        chain = ComplianceChain(gemini_client=mock_embedding, brand_retriever=retriever)

        report = await chain.check_compliance("test-asset", Path("/tmp/ad.jpg"))
        assert report.asset_id == "test-asset"
        assert report.passed is True
        assert report.score == 1.0

    @pytest.mark.asyncio
    async def test_compliance_with_violations(self, mock_embedding, store):
        """Compliance with a custom evaluator that finds violations."""
        indexer = BrandIndexer(gemini_client=mock_embedding, vector_store=store)
        await indexer.index_brand_guide_page(
            "rule-1", "Logo must appear in first 3 seconds",
            section_type="logo_rules", page_number=1,
        )

        retriever = BrandRetriever(gemini_client=mock_embedding, vector_store=store)

        async def mock_evaluator(asset_path, guidelines):
            return [Violation(
                rule_type="semantic", severity="warning",
                description="Logo appears late at 5s",
            )]

        chain = ComplianceChain(
            gemini_client=mock_embedding,
            brand_retriever=retriever,
            evaluator=mock_evaluator,
        )
        report = await chain.check_compliance("test-asset", Path("/tmp/ad.jpg"))
        assert report.passed is True  # warning, not critical
        assert len(report.violations) == 1
        assert report.score < 1.0

    @pytest.mark.asyncio
    async def test_compliance_no_guidelines_found(self, mock_embedding, store):
        """When no guidelines are indexed, compliance should still return a valid report."""
        retriever = BrandRetriever(gemini_client=mock_embedding, vector_store=store)
        chain = ComplianceChain(gemini_client=mock_embedding, brand_retriever=retriever)

        report = await chain.check_compliance("orphan-asset", Path("/tmp/ad.jpg"))
        assert report.passed is True
        assert report.evidence_sources == []
