"""Brand compliance checking pipeline."""

from collections.abc import Awaitable, Callable
from pathlib import Path

import structlog

from omni_proof.core.interfaces import EmbeddingProvider
from omni_proof.orchestration.models import ComplianceReport, Violation
from omni_proof.rag.brand_retriever import BrandRetriever

logger = structlog.get_logger()


class ComplianceChain:
    """Checks new creatives against brand guidelines retrieved from RAG."""

    def __init__(
        self,
        gemini_client: EmbeddingProvider,
        brand_retriever: BrandRetriever,
        evaluator: Callable[[str | Path, list], Awaitable[list[Violation]]] | None = None,
    ):
        self._gemini = gemini_client
        self._retriever = brand_retriever
        self._evaluator = evaluator

    async def check_compliance(self, asset_id: str, asset_path: str | Path) -> ComplianceReport:
        # Step 1: Retrieve relevant brand guidelines
        guidelines = await self._retriever.get_guidelines_for_asset(asset_path)
        evidence_sources = [g.asset_id for g in guidelines]

        # Step 2: Evaluate compliance via Gemini
        # In production, this sends the asset + guidelines to Gemini for evaluation
        # For now, we return the structure with retrieved guidelines
        violations = await self._evaluate(asset_path, guidelines)

        passed = all(v.severity != "critical" for v in violations)
        score = max(0.0, 1.0 - (len(violations) * 0.2))

        report = ComplianceReport(
            asset_id=asset_id,
            passed=passed,
            violations=violations,
            evidence_sources=evidence_sources,
            score=score,
        )
        logger.info(
            "compliance_check", asset_id=asset_id, passed=passed, violations=len(violations)
        )
        return report

    async def _evaluate(self, asset_path: str | Path, guidelines) -> list[Violation]:
        """Evaluate asset against retrieved guidelines using the configured evaluator."""
        if self._evaluator is None:
            return []
        return await self._evaluator(asset_path, guidelines)
