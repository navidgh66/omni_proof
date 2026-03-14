"""Tests for visual DML estimator."""

import numpy as np
import pytest

from omni_proof.causal.dice_dml.visual_estimator import VisualDMLEstimator
from omni_proof.causal.results import ATEResult


class TestVisualDMLEstimator:
    def test_estimate_returns_ate_result(self):
        rng = np.random.RandomState(42)
        n = 300
        d = 50  # small dims for speed

        # Create embeddings with known treatment direction
        fp = rng.randn(d)
        fp = fp / np.linalg.norm(fp)

        treatment = rng.binomial(1, 0.5, n).astype(float)
        confounders = rng.randn(n, d) * 0.5
        # Embeddings = confounders + treatment * fingerprint (entangled)
        embeddings = confounders + np.outer(treatment, fp) * 2.0
        outcome = 0.5 * treatment + confounders[:, 0] * 0.3 + rng.randn(n) * 0.3

        est = VisualDMLEstimator(cv=2)
        result = est.estimate_visual_ate(embeddings, treatment, outcome, fp)

        assert isinstance(result, ATEResult)
        assert result.n_samples == n
        assert result.ci_lower < result.ci_upper
