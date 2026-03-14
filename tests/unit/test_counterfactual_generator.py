"""Tests for counterfactual pair generation."""

import numpy as np
import pytest
from unittest.mock import AsyncMock

from omni_proof.causal.dice_dml.counterfactual_generator import (
    CounterfactualGenerator,
    CounterfactualPair,
)


class TestCounterfactualPair:
    def test_background_similarity_identical(self):
        emb = np.array([1.0, 0.0, 0.0])
        pair = CounterfactualPair(emb, emb, "color")
        assert pair.background_similarity == pytest.approx(1.0)

    def test_background_similarity_orthogonal(self):
        pair = CounterfactualPair(
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            "color",
        )
        assert pair.background_similarity == pytest.approx(0.0, abs=1e-7)

    def test_background_similarity_similar(self):
        orig = np.random.RandomState(42).randn(3072)
        # Small perturbation = high similarity
        cf = orig + np.random.RandomState(43).randn(3072) * 0.01
        pair = CounterfactualPair(orig, cf, "typography")
        assert pair.background_similarity > 0.95


class TestCounterfactualGenerator:
    @pytest.mark.asyncio
    async def test_generate_returns_pair(self):
        mock_gemini = AsyncMock()
        mock_gemini.generate_embedding = AsyncMock(
            side_effect=[list(np.random.randn(3072)), list(np.random.randn(3072) * 0.01 + np.random.randn(3072))]
        )
        gen = CounterfactualGenerator(gemini_client=mock_gemini)
        pair = await gen.generate("/tmp/orig.jpg", "/tmp/cf.jpg", "color")
        assert isinstance(pair, CounterfactualPair)
        assert pair.treatment_attr == "color"
        assert len(pair.original_emb) == 3072
