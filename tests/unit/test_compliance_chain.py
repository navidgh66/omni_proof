"""Tests for brand compliance pipeline."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from omni_proof.orchestration.compliance_chain import ComplianceChain
from omni_proof.orchestration.models import Violation
from omni_proof.rag.models import BrandAsset


@pytest.fixture
def mock_gemini():
    return AsyncMock()


@pytest.fixture
def mock_retriever():
    retriever = AsyncMock()
    retriever.get_guidelines_for_asset = AsyncMock(
        return_value=[
            BrandAsset(
                asset_id="guide-1", source_type="guideline", section_type="logo_rules", score=0.95
            ),
            BrandAsset(
                asset_id="guide-2",
                source_type="guideline",
                section_type="color_palette",
                score=0.88,
            ),
        ]
    )
    return retriever


class TestComplianceChain:
    @pytest.mark.asyncio
    async def test_check_compliance_passes_with_no_violations(self, mock_gemini, mock_retriever):
        chain = ComplianceChain(gemini_client=mock_gemini, brand_retriever=mock_retriever)
        report = await chain.check_compliance("asset-1", Path("/tmp/ad.jpg"))
        assert report.passed is True
        assert report.violations == []
        assert report.asset_id == "asset-1"
        assert "guide-1" in report.evidence_sources

    @pytest.mark.asyncio
    async def test_check_compliance_retrieves_guidelines(self, mock_gemini, mock_retriever):
        chain = ComplianceChain(gemini_client=mock_gemini, brand_retriever=mock_retriever)
        await chain.check_compliance("asset-1", Path("/tmp/ad.jpg"))
        mock_retriever.get_guidelines_for_asset.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_check_compliance_with_evaluator(self, mock_gemini, mock_retriever):
        async def mock_evaluator(asset_path, guidelines):
            return [
                Violation(rule_type="concrete", severity="warning", description="Logo too small"),
            ]

        chain = ComplianceChain(
            gemini_client=mock_gemini,
            brand_retriever=mock_retriever,
            evaluator=mock_evaluator,
        )
        report = await chain.check_compliance("asset-1", Path("/tmp/ad.jpg"))
        assert report.passed is True  # warning, not critical
        assert len(report.violations) == 1
        assert report.violations[0].description == "Logo too small"
        assert report.score == 0.8  # 1.0 - 0.2

    @pytest.mark.asyncio
    async def test_critical_violation_fails_compliance(self, mock_gemini, mock_retriever):
        async def mock_evaluator(asset_path, guidelines):
            return [
                Violation(
                    rule_type="concrete", severity="critical", description="Wrong brand colors"
                ),
            ]

        chain = ComplianceChain(
            gemini_client=mock_gemini,
            brand_retriever=mock_retriever,
            evaluator=mock_evaluator,
        )
        report = await chain.check_compliance("asset-1", Path("/tmp/ad.jpg"))
        assert report.passed is False
        assert len(report.violations) == 1
