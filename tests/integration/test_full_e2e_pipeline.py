"""Full end-to-end pipeline: ingest -> store -> causal -> compliance -> generate."""

from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest

from omni_proof.api.generative_loop import GenerativePromptBuilder
from omni_proof.brand_extraction.extractor import BrandExtractor
from omni_proof.brand_extraction.models import (
    BrandAssetMetadata,
    BrandColorInfo,
    BrandToneInfo,
    BrandTypographyInfo,
    BrandVisualInfo,
)
from omni_proof.causal.estimator import DMLEstimator
from omni_proof.orchestration.compliance_chain import ComplianceChain
from omni_proof.orchestration.insight_synthesizer import InsightSynthesizer
from omni_proof.rag.brand_indexer import BrandIndexer
from omni_proof.rag.brand_retriever import BrandRetriever
from omni_proof.storage.memory_store import InMemoryVectorStore
from tests.fixtures.synthetic_ads import generate_synthetic_dataset


def _make_metadata():
    return BrandAssetMetadata(
        asset_description="Brand ad",
        colors=BrandColorInfo(hex_codes=["#004E89", "#FF6B35"], palette_mood="bold"),
        typography=BrandTypographyInfo(
            font_styles=["sans-serif"], font_names=["Inter"], text_hierarchy="clear"
        ),
        tone=BrandToneInfo(
            formality="professional",
            emotional_register="confident",
            key_phrases=["Innovate"],
            vocabulary_themes=["tech"],
        ),
        visual=BrandVisualInfo(
            layout_pattern="centered",
            motion_intensity="static",
            dominant_objects=["logo"],
        ),
        logo_detected=True,
        media_type_detected="image",
    )


def _encode_categoricals(df, cols):
    df = df.copy()
    for col in cols:
        if df[col].dtype == object:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True).astype(float)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
    return df


@pytest.fixture
def store():
    return InMemoryVectorStore()


@pytest.fixture
def mock_embedding():
    provider = AsyncMock()
    provider.generate_embedding = AsyncMock(return_value=[0.1] * 3072)
    return provider


@pytest.fixture
def mock_gemini():
    client = AsyncMock()
    client.extract_metadata = AsyncMock(return_value=_make_metadata())
    return client


class TestFullE2EPipeline:
    @pytest.mark.asyncio
    async def test_happy_path_all_stages(self, mock_gemini, mock_embedding, store):
        """Full pipeline: brand extract → index → causal → compliance → generate."""
        # Stage 1: Brand extraction
        extractor = BrandExtractor(mock_embedding, mock_gemini, store)
        profile = await extractor.extract(
            "AcmeCorp", [Path("/tmp/a.jpg"), Path("/tmp/b.jpg"), Path("/tmp/c.jpg")]
        )
        assert profile.brand_name == "AcmeCorp"

        # Stage 2: Index brand guidelines
        indexer = BrandIndexer(gemini_client=mock_embedding, vector_store=store)
        await indexer.index_brand_guide_page(
            "guide-1", "Use blue #004E89 as primary", "color_palette", 1
        )

        # Stage 3: Causal analysis
        data = generate_synthetic_dataset(n=300, seed=42)
        encoded = _encode_categoricals(data, ["platform", "audience_segment"])
        encoded["audience_segment"] = data["audience_segment"].values
        confounders = [
            c for c in encoded.columns if c.startswith(("platform_", "audience_segment_"))
        ] + ["budget"]
        estimator = DMLEstimator(cv=3)
        cate = estimator.estimate_cate(
            encoded, "logo_in_first_3s", "ctr", confounders, "audience_segment"
        )
        cate.refutation_passed = True

        # Stage 4: Synthesize insights
        synthesizer = InsightSynthesizer()
        brief = synthesizer.synthesize(cate)
        assert brief.confidence == "HIGH"

        # Stage 5: Compliance check
        retriever = BrandRetriever(gemini_client=mock_embedding, vector_store=store)
        chain = ComplianceChain(gemini_client=mock_embedding, brand_retriever=retriever)
        report = await chain.check_compliance("new-ad", Path("/tmp/new_ad.jpg"))
        assert report.passed is True

        # Stage 6: Creative generation
        builder = GenerativePromptBuilder()
        prompt = builder.build_prompt(
            cate_insights=[{"treatment": brief.treatment, "effect": 0.5}],
            brand_rules=[{"description": r.description} for r in profile.rules],
            target_segment="18-24",
            objective="conversion",
        )
        assert "18-24" in prompt
        assert "conversion" in prompt

    @pytest.mark.asyncio
    async def test_empty_data_pipeline(self, mock_gemini, mock_embedding, store):
        """Pipeline with no causal data — should produce neutral insights."""
        rng = np.random.RandomState(42)
        n = 100
        data = pd.DataFrame({
            "treatment": rng.binomial(1, 0.5, n).astype(float),
            "outcome": rng.randn(n) * 0.01,  # Near-zero variance
            "conf": rng.randn(n),
            "segment": rng.choice(["A", "B"], n),
        })
        estimator = DMLEstimator(cv=2)
        cate = estimator.estimate_cate(data, "treatment", "outcome", ["conf"], "segment")
        cate.refutation_passed = False

        synthesizer = InsightSynthesizer()
        brief = synthesizer.synthesize(cate)
        assert brief.confidence == "LOW"

    @pytest.mark.asyncio
    async def test_round_trip_store_and_retrieve(self, mock_embedding, store):
        """Store data in vector store, retrieve via RAG, verify match."""
        indexer = BrandIndexer(gemini_client=mock_embedding, vector_store=store)
        await indexer.index_brand_guide_page(
            "rule-1", "Primary color is blue #004E89", "color_palette", 1
        )
        await indexer.index_approved_creative("creative-1", Path("/tmp/approved.jpg"), ["hero"])

        retriever = BrandRetriever(gemini_client=mock_embedding, vector_store=store)

        # Search in brand_assets namespace
        all_results = await retriever.search_by_text("brand assets")
        assert len(all_results) >= 1

        # Guidelines-only filter
        guideline_results = await retriever.get_guidelines_for_asset(Path("/tmp/test.jpg"))
        for r in guideline_results:
            assert r.source_type == "guideline"
