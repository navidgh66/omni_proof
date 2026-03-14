"""Tests for DML estimator with synthetic data."""

import numpy as np
import pandas as pd
import pytest

from omni_proof.causal.estimator import DMLEstimator
from omni_proof.causal.results import ATEResult, CATEResult


@pytest.fixture
def synthetic_data():
    """Synthetic data with known treatment effect of ~0.5 on outcome."""
    np.random.seed(42)
    n = 500
    confounder1 = np.random.randn(n)
    confounder2 = np.random.randn(n)
    treatment = (confounder1 + np.random.randn(n) > 0).astype(float)
    outcome = 0.5 * treatment + 0.3 * confounder1 + 0.2 * confounder2 + np.random.randn(n) * 0.5
    segment = np.random.choice(["18-24", "25-34"], n)
    return pd.DataFrame(
        {
            "treatment": treatment,
            "outcome": outcome,
            "conf1": confounder1,
            "conf2": confounder2,
            "segment": segment,
        }
    )


class TestDMLEstimator:
    def test_ate_recovers_effect(self, synthetic_data):
        est = DMLEstimator(cv=3)
        result = est.estimate_ate(synthetic_data, "treatment", "outcome", ["conf1", "conf2"])
        assert isinstance(result, ATEResult)
        assert result.n_samples == 500
        # Known effect is ~0.5; should be in rough range
        assert -0.5 < result.ate < 1.5
        assert result.ci_lower < result.ci_upper

    def test_cate_returns_segments(self, synthetic_data):
        est = DMLEstimator(cv=3)
        result = est.estimate_cate(
            synthetic_data, "treatment", "outcome", ["conf1", "conf2"], "segment"
        )
        assert isinstance(result, CATEResult)
        assert "18-24" in result.segments
        assert "25-34" in result.segments
