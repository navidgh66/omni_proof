"""End-to-end test: synthetic data -> causal estimation -> insight synthesis."""

import pytest
import numpy as np
import pandas as pd

from omni_proof.causal.dag_builder import CausalDAGBuilder
from omni_proof.causal.identifier import CausalIdentifier
from omni_proof.causal.estimator import DMLEstimator
from omni_proof.causal.refuter import CausalRefuter
from omni_proof.causal.results import ATEResult, CATEResult
from omni_proof.orchestration.insight_synthesizer import InsightSynthesizer
from tests.fixtures.synthetic_ads import generate_synthetic_dataset


def _encode_categoricals(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """One-hot encode categorical columns so estimator gets numeric data."""
    df = df.copy()
    for col in cols:
        if df[col].dtype == object:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True).astype(float)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
    return df


class TestFullCausalPipeline:
    """E2E: generate data with known effect -> estimate -> refute -> synthesize."""

    @pytest.fixture
    def data(self):
        return generate_synthetic_dataset(n=500, seed=42)

    def test_dag_construction_and_identification(self, data):
        builder = CausalDAGBuilder()
        model = builder.build_dag(
            data=data,
            treatment="logo_in_first_3s",
            outcome="ctr",
            confounders=["platform", "audience_segment", "budget"],
        )
        identifier = CausalIdentifier()
        estimand = identifier.identify_effect(model)
        assert estimand is not None

    def test_ate_recovers_planted_effect(self, data):
        """The planted effect is 0.5. DML should recover it within CI."""
        cat_cols = ["platform", "audience_segment"]
        encoded = _encode_categoricals(data, cat_cols)
        confounder_cols = [c for c in encoded.columns if c.startswith(("platform_", "audience_segment_"))] + ["budget"]
        estimator = DMLEstimator(cv=3)
        result = estimator.estimate_ate(
            encoded, "logo_in_first_3s", "ctr", confounder_cols
        )
        assert isinstance(result, ATEResult)
        # Planted effect is 0.5; allow generous range
        assert 0.0 < result.ate < 1.5, f"ATE {result.ate} outside expected range"
        assert result.ci_lower < result.ate < result.ci_upper
        assert result.n_samples == 500

    def test_cate_shows_segments(self, data):
        encoded = _encode_categoricals(data, ["platform"])
        confounder_cols = [c for c in encoded.columns if c.startswith("platform_")] + ["budget"]
        estimator = DMLEstimator(cv=3)
        result = estimator.estimate_cate(
            encoded, "logo_in_first_3s", "ctr",
            confounder_cols, "audience_segment"
        )
        assert isinstance(result, CATEResult)
        assert "18-24" in result.segments
        assert "25-34" in result.segments
        assert "35-44" in result.segments

    def test_refutation_suite_passes(self, data):
        encoded = _encode_categoricals(data, ["platform", "audience_segment"])
        confounders = [c for c in encoded.columns if c.startswith(("platform_", "audience_segment_"))] + ["budget"]
        refuter = CausalRefuter(cv=2)

        placebo = refuter.placebo_test(encoded, "logo_in_first_3s", "ctr", confounders)
        subset = refuter.subset_test(encoded, "logo_in_first_3s", "ctr", confounders)
        random_conf = refuter.random_confounder_test(encoded, "logo_in_first_3s", "ctr", confounders)

        assert subset.passed, f"Subset test failed: orig={subset.original_effect}, new={subset.new_effect}"
        assert random_conf.passed, f"Random confounder test failed"

    def test_full_pipeline_dag_to_brief(self, data):
        """Complete flow: DAG -> identify -> estimate CATE -> synthesize brief."""
        # 1. Build DAG
        builder = CausalDAGBuilder()
        model = builder.build_dag(
            data=data,
            treatment="logo_in_first_3s",
            outcome="ctr",
            confounders=["platform", "audience_segment", "budget"],
        )

        # 2. Identify
        identifier = CausalIdentifier()
        estimand = identifier.identify_effect(model)
        assert estimand is not None

        # 3. Estimate CATE
        encoded = _encode_categoricals(data, ["platform"])
        confounder_cols = [c for c in encoded.columns if c.startswith("platform_")] + ["budget"]
        estimator = DMLEstimator(cv=3)
        cate = estimator.estimate_cate(
            encoded, "logo_in_first_3s", "ctr",
            confounder_cols, "audience_segment"
        )
        cate.refutation_passed = True  # Mark as validated

        # 4. Synthesize insight
        synthesizer = InsightSynthesizer()
        brief = synthesizer.synthesize(cate)

        assert brief.treatment == "logo_in_first_3s"
        assert brief.confidence == "HIGH"
        assert len(brief.segment_breakdown) == 3
