"""E2E test: brand compliance checking pipeline."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from omni_proof.orchestration.compliance_chain import ComplianceChain
from omni_proof.orchestration.models import Violation
from omni_proof.rag.brand_retriever import BrandRetriever
from omni_proof.rag.models import BrandAsset


class TestBrandComplianceE2E:
    @pytest.fixture
    def compliant_chain(self):
        mock_gemini = AsyncMock()
        mock_retriever = AsyncMock(spec=BrandRetriever)
        mock_retriever.get_guidelines_for_asset = AsyncMock(return_value=[
            BrandAsset(asset_id="guide-1", source_type="guideline", section_type="color_palette", score=0.95),
            BrandAsset(asset_id="guide-2", source_type="guideline", section_type="logo_rules", score=0.90),
        ])
        return ComplianceChain(gemini_client=mock_gemini, brand_retriever=mock_retriever)

    @pytest.mark.asyncio
    async def test_compliant_creative_passes(self, compliant_chain):
        report = await compliant_chain.check_compliance("asset-ok", Path("/tmp/compliant.jpg"))
        assert report.passed is True
        assert report.violations == []
        assert len(report.evidence_sources) == 2

    @pytest.mark.asyncio
    async def test_compliance_report_structure(self, compliant_chain):
        report = await compliant_chain.check_compliance("asset-1", Path("/tmp/test.jpg"))
        assert hasattr(report, "passed")
        assert hasattr(report, "violations")
        assert hasattr(report, "evidence_sources")
        assert hasattr(report, "score")
        assert 0.0 <= report.score <= 1.0
