"""E2E tests for advanced causal scenarios: multi-treatment, DICE-DML, edge cases."""

import numpy as np
import pandas as pd

from omni_proof.causal.dice_dml.disentangler import TreatmentDisentangler
from omni_proof.causal.dice_dml.visual_estimator import VisualDMLEstimator
from omni_proof.causal.estimator import DMLEstimator
from omni_proof.causal.results import ATEResult, CATEResult


def _make_two_treatment_data(n=500, seed=42):
    """Synthetic data with two independent treatments and known effects."""
    rng = np.random.RandomState(seed)
    conf = rng.randn(n)
    t1 = (conf + rng.randn(n) > 0).astype(float)  # treatment 1
    t2 = (rng.randn(n) > 0.2).astype(float)  # treatment 2, weakly confounded
    y = 0.5 * t1 + 0.3 * t2 + 0.4 * conf + rng.randn(n) * 0.3
    return pd.DataFrame({"t1": t1, "t2": t2, "conf": conf, "outcome": y})


class TestMultiTreatmentCausal:
    def test_two_independent_treatments(self):
        data = _make_two_treatment_data(n=500)
        estimator = DMLEstimator(cv=3)

        ate1 = estimator.estimate_ate(data, "t1", "outcome", ["conf", "t2"])
        ate2 = estimator.estimate_ate(data, "t2", "outcome", ["conf", "t1"])

        # t1 has planted effect ~0.5, t2 has ~0.3
        assert isinstance(ate1, ATEResult)
        assert isinstance(ate2, ATEResult)
        assert ate1.ate > ate2.ate  # t1 effect is larger

    def test_cate_with_multiple_segments(self):
        rng = np.random.RandomState(42)
        n = 600
        segment = rng.choice(["A", "B", "C", "D"], n)
        conf = rng.randn(n)
        t = (conf + rng.randn(n) > 0).astype(float)
        # Heterogeneous effect: strong for A, weak for others
        effect = np.where(segment == "A", 0.8, 0.1)
        y = effect * t + 0.3 * conf + rng.randn(n) * 0.2
        data = pd.DataFrame({"treatment": t, "outcome": y, "conf": conf, "segment": segment})
        estimator = DMLEstimator(cv=3)
        cate = estimator.estimate_cate(data, "treatment", "outcome", ["conf"], "segment")
        assert isinstance(cate, CATEResult)
        assert len(cate.segments) == 4
        # Segment A should have the strongest effect
        assert cate.segments["A"].effect > cate.segments["B"].effect


class TestRefutationCatchesSpurious:
    def test_random_treatment_has_near_zero_effect(self):
        """A purely random treatment should have near-zero estimated effect."""
        rng = np.random.RandomState(42)
        n = 500
        conf = rng.randn(n)
        random_treatment = rng.binomial(1, 0.5, n).astype(float)
        outcome = 0.5 * conf + rng.randn(n) * 0.3
        data = pd.DataFrame({"treatment": random_treatment, "outcome": outcome, "conf": conf})
        estimator = DMLEstimator(cv=3)
        result = estimator.estimate_ate(data, "treatment", "outcome", ["conf"])
        # Effect should be close to zero since treatment is random
        assert abs(result.ate) < 0.3, f"ATE {result.ate} too large for random treatment"


class TestDICEDMLE2E:
    def test_disentangle_and_estimate(self):
        """Full DICE-DML: fingerprint extraction -> projection -> estimation."""
        rng = np.random.RandomState(42)
        n = 100
        dim = 128

        # Simulate embeddings where treatment signal is in first 10 dims
        treatment_signal = rng.randn(n, 10)
        confounder_signal = rng.randn(n, dim - 10)
        original_embs = np.hstack([treatment_signal, confounder_signal])

        # Counterfactuals: same confounder, different treatment
        cf_treatment = rng.randn(n, 10)
        counterfactual_embs = np.hstack([cf_treatment, confounder_signal])

        # Extract fingerprints and disentangle
        disentangler = TreatmentDisentangler()
        fingerprints = []
        projected = []
        for i in range(n):
            fp = disentangler.extract_treatment_fingerprint(
                original_embs[i], counterfactual_embs[i]
            )
            fingerprints.append(fp)
            proj = disentangler.orthogonal_projection(original_embs[i], fp)
            projected.append(proj)

        fingerprints = np.array(fingerprints)
        projected = np.array(projected)

        # Verify orthogonality: projected vectors should have low correlation with fingerprints
        for i in range(min(10, n)):
            dot = np.dot(projected[i], fingerprints[i])
            assert abs(dot) < 1e-6, f"Projected vector not orthogonal to fingerprint: dot={dot}"

        # Run visual DML estimation
        treatment = rng.binomial(1, 0.5, n).astype(float)
        outcome = 0.4 * treatment + rng.randn(n) * 0.2
        # Use mean fingerprint as treatment_fingerprint for batch disentangle
        mean_fp = np.mean(fingerprints, axis=0)
        estimator = VisualDMLEstimator(cv=2)
        result = estimator.estimate_visual_ate(
            embeddings=original_embs, treatment=treatment, outcome=outcome,
            treatment_fingerprint=mean_fp,
        )
        assert isinstance(result, ATEResult)
        assert result.n_samples == n
