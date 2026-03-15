"""Edge case tests for boundary conditions across modules."""

import numpy as np
import pandas as pd
import pytest

from omni_proof.causal.estimator import DMLEstimator
from omni_proof.causal.results import CATEResult, EffectEstimate
from omni_proof.orchestration.insight_synthesizer import InsightSynthesizer
from omni_proof.storage.memory_store import InMemoryVectorStore


class TestDMLEstimatorEdgeCases:
    def test_constant_outcome(self):
        """All-zero outcome should not crash."""
        rng = np.random.RandomState(42)
        n = 100
        data = pd.DataFrame({
            "treatment": rng.binomial(1, 0.5, n).astype(float),
            "outcome": np.zeros(n),
            "conf": rng.randn(n),
        })
        estimator = DMLEstimator(cv=2)
        result = estimator.estimate_ate(data, "treatment", "outcome", ["conf"])
        assert result.n_samples == n

    def test_single_confounder(self):
        """Estimator works with just one confounder."""
        rng = np.random.RandomState(42)
        n = 200
        conf = rng.randn(n)
        t = (conf > 0).astype(float)
        y = 0.3 * t + 0.5 * conf + rng.randn(n) * 0.2
        data = pd.DataFrame({"treatment": t, "outcome": y, "conf": conf})
        estimator = DMLEstimator(cv=2)
        result = estimator.estimate_ate(data, "treatment", "outcome", ["conf"])
        assert result.ci_lower < result.ate < result.ci_upper


class TestInsightSynthesizerEdgeCases:
    def test_no_significant_segments(self):
        """All segments have high p-values — should return neutral finding."""
        cate = CATEResult(
            treatment="test_treatment",
            outcome="ctr",
            segments={
                "seg_a": EffectEstimate(effect=0.01, ci_lower=-0.1, ci_upper=0.12, p_value=0.8),
                "seg_b": EffectEstimate(effect=-0.02, ci_lower=-0.15, ci_upper=0.11, p_value=0.9),
            },
            refutation_passed=True,
        )
        synth = InsightSynthesizer()
        brief = synth.synthesize(cate)
        assert "No statistically significant" in brief.finding


class TestInMemoryVectorStoreEdgeCases:
    @pytest.mark.asyncio
    async def test_search_empty_store(self):
        store = InMemoryVectorStore()
        results = await store.search([0.1] * 10, top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_search_different_namespace_returns_empty(self):
        store = InMemoryVectorStore()
        await store.upsert("id1", [1.0, 0.0, 0.0], {"type": "test"}, namespace="ns_a")
        results = await store.search([1.0, 0.0, 0.0], top_k=5, namespace="ns_b")
        assert results == []

    @pytest.mark.asyncio
    async def test_delete_nonexistent_no_error(self):
        store = InMemoryVectorStore()
        await store.delete("nonexistent-id")  # Should not raise
