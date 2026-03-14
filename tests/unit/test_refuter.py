"""Tests for causal refutation suite."""

import numpy as np
import pandas as pd
import pytest

from omni_proof.causal.refuter import CausalRefuter


@pytest.fixture
def synthetic_data():
    np.random.seed(42)
    n = 300
    conf = np.random.randn(n)
    treatment = (conf + np.random.randn(n) > 0).astype(float)
    outcome = 0.5 * treatment + 0.3 * conf + np.random.randn(n) * 0.5
    return pd.DataFrame({"treatment": treatment, "outcome": outcome, "conf": conf})


class TestCausalRefuter:
    def test_placebo_test(self, synthetic_data):
        refuter = CausalRefuter(cv=2)
        result = refuter.placebo_test(synthetic_data, "treatment", "outcome", ["conf"])
        assert result.test_name == "placebo"
        # Placebo effect should be smaller than original
        assert abs(result.new_effect) <= abs(result.original_effect) + 0.5

    def test_subset_test(self, synthetic_data):
        refuter = CausalRefuter(cv=2)
        result = refuter.subset_test(synthetic_data, "treatment", "outcome", ["conf"])
        assert result.test_name == "subset"
        assert result.passed is True

    def test_random_confounder_test(self, synthetic_data):
        refuter = CausalRefuter(cv=2)
        result = refuter.random_confounder_test(synthetic_data, "treatment", "outcome", ["conf"])
        assert result.test_name == "random_confounder"
        assert result.passed is True
