"""Disentangles treatment signal from confounders in embedding space."""

import numpy as np
import structlog

logger = structlog.get_logger()


class TreatmentDisentangler:
    """Extracts treatment fingerprints and projects embeddings to confounder-only space."""

    def extract_treatment_fingerprint(
        self, original_emb: np.ndarray, counterfactual_emb: np.ndarray,
    ) -> np.ndarray:
        """Vector subtraction isolates the pure treatment signal."""
        fingerprint = original_emb - counterfactual_emb
        norm = np.linalg.norm(fingerprint)
        if norm > 0:
            fingerprint = fingerprint / norm
        return fingerprint

    def orthogonal_projection(
        self, embedding: np.ndarray, treatment_fingerprint: np.ndarray,
    ) -> np.ndarray:
        """Project out treatment component, leaving pure confounder representation."""
        # Project embedding onto fingerprint direction
        dot = np.dot(embedding, treatment_fingerprint)
        treatment_component = dot * treatment_fingerprint
        # Subtract to get orthogonal (confounder-only) representation
        return embedding - treatment_component

    def disentangle_batch(
        self, embeddings: np.ndarray, treatment_fingerprint: np.ndarray,
    ) -> np.ndarray:
        """Apply orthogonal projection to a batch of embeddings."""
        # embeddings: (n, d), fingerprint: (d,)
        dots = embeddings @ treatment_fingerprint  # (n,)
        projections = np.outer(dots, treatment_fingerprint)  # (n, d)
        return embeddings - projections
