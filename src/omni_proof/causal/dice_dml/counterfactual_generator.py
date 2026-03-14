"""Generates deepfake counterfactual pairs for visual causal inference."""

import numpy as np
import structlog

from omni_proof.ingestion.gemini_client import GeminiClient

logger = structlog.get_logger()


class CounterfactualPair:
    """Holds original and counterfactual embeddings for a single treatment change."""

    def __init__(self, original_emb: np.ndarray, counterfactual_emb: np.ndarray, treatment_attr: str):
        self.original_emb = original_emb
        self.counterfactual_emb = counterfactual_emb
        self.treatment_attr = treatment_attr

    @property
    def background_similarity(self) -> float:
        """Cosine similarity between original and counterfactual (background should be ~1.0)."""
        dot = np.dot(self.original_emb, self.counterfactual_emb)
        norm = np.linalg.norm(self.original_emb) * np.linalg.norm(self.counterfactual_emb)
        if norm == 0:
            return 0.0
        return float(dot / norm)


class CounterfactualGenerator:
    """Creates paired embeddings where only the treatment attribute differs."""

    def __init__(self, gemini_client: GeminiClient):
        self._gemini = gemini_client

    async def generate(
        self, original_path, counterfactual_path, treatment_attr: str,
    ) -> CounterfactualPair:
        original_emb = np.array(await self._gemini.generate_embedding(original_path))
        cf_emb = np.array(await self._gemini.generate_embedding(counterfactual_path))

        pair = CounterfactualPair(original_emb, cf_emb, treatment_attr)

        similarity = pair.background_similarity
        logger.info(
            "counterfactual_generated",
            treatment=treatment_attr,
            background_similarity=f"{similarity:.4f}",
        )
        if similarity < 0.90:
            logger.warning("low_background_similarity", similarity=similarity)

        return pair
