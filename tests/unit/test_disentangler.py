"""Tests for treatment disentanglement via orthogonal projection."""

import numpy as np
import pytest

from omni_proof.causal.dice_dml.disentangler import TreatmentDisentangler


@pytest.fixture
def disentangler():
    return TreatmentDisentangler()


class TestTreatmentFingerprint:
    def test_fingerprint_is_normalized(self, disentangler):
        orig = np.array([1.0, 2.0, 3.0])
        cf = np.array([1.0, 2.0, 2.0])
        fp = disentangler.extract_treatment_fingerprint(orig, cf)
        assert np.linalg.norm(fp) == pytest.approx(1.0, abs=1e-7)

    def test_fingerprint_direction(self, disentangler):
        orig = np.array([1.0, 0.0, 1.0])
        cf = np.array([1.0, 0.0, 0.0])
        fp = disentangler.extract_treatment_fingerprint(orig, cf)
        # Difference is in 3rd dimension only
        assert fp[2] == pytest.approx(1.0, abs=1e-7)
        assert fp[0] == pytest.approx(0.0, abs=1e-7)


class TestOrthogonalProjection:
    def test_projected_vector_is_orthogonal_to_fingerprint(self, disentangler):
        emb = np.array([3.0, 4.0, 5.0])
        fp = np.array([0.0, 0.0, 1.0])
        projected = disentangler.orthogonal_projection(emb, fp)
        dot = np.dot(projected, fp)
        assert dot == pytest.approx(0.0, abs=1e-7)

    def test_projection_preserves_orthogonal_components(self, disentangler):
        emb = np.array([3.0, 4.0, 5.0])
        fp = np.array([0.0, 0.0, 1.0])
        projected = disentangler.orthogonal_projection(emb, fp)
        assert projected[0] == pytest.approx(3.0)
        assert projected[1] == pytest.approx(4.0)
        assert projected[2] == pytest.approx(0.0, abs=1e-7)


class TestBatchDisentangle:
    def test_batch_all_orthogonal(self, disentangler):
        rng = np.random.RandomState(42)
        embeddings = rng.randn(50, 3072)
        fp = rng.randn(3072)
        fp = fp / np.linalg.norm(fp)

        result = disentangler.disentangle_batch(embeddings, fp)

        # Each projected vector should be orthogonal to fingerprint
        dots = result @ fp
        assert np.allclose(dots, 0.0, atol=1e-7)
        assert result.shape == (50, 3072)
