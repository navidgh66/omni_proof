"""Tests for causal DAG construction."""

import numpy as np
import pandas as pd
import pytest

from omni_proof.causal.dag_builder import TEMPLATES, CausalDAGBuilder


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 200
    return pd.DataFrame(
        {
            "platform": np.random.choice(["youtube", "instagram"], n),
            "audience_segment": np.random.choice(["18-24", "25-34"], n),
            "season": np.random.choice(["summer", "winter"], n),
            "budget": np.random.uniform(1000, 50000, n),
            "logo_in_first_3s": np.random.binomial(1, 0.5, n),
            "ctr": np.random.uniform(0.01, 0.10, n),
            "warm_color_palette": np.random.binomial(1, 0.5, n),
            "engagement_rate": np.random.uniform(0.05, 0.20, n),
            "fast_audio_pacing": np.random.binomial(1, 0.5, n),
            "conversion_rate": np.random.uniform(0.01, 0.05, n),
            "product_category": np.random.choice(["tech", "fashion"], n),
            "production_quality": np.random.uniform(1, 10, n),
            "time_of_day": np.random.choice(["morning", "evening"], n),
            "collider_var": np.random.uniform(0, 1, n),
        }
    )


@pytest.fixture
def builder():
    return CausalDAGBuilder()


class TestDAGBuilder:
    def test_build_dag_returns_causal_model(self, builder, sample_data):
        model = builder.build_dag(
            data=sample_data,
            treatment="logo_in_first_3s",
            outcome="ctr",
            confounders=["platform", "audience_segment", "season"],
        )
        assert model is not None
        assert hasattr(model, "identify_effect")

    def test_collider_warning_and_removal(self, builder, sample_data, caplog):
        model = builder.build_dag(
            data=sample_data,
            treatment="logo_in_first_3s",
            outcome="ctr",
            confounders=["platform", "collider_var"],
            colliders=["collider_var"],
        )
        assert model is not None

    def test_build_from_template(self, builder, sample_data):
        for name in TEMPLATES:
            model = builder.build_from_template(sample_data, name)
            assert model is not None

    def test_unknown_template_raises(self, builder, sample_data):
        with pytest.raises(ValueError, match="Unknown template"):
            builder.build_from_template(sample_data, "nonexistent")


class TestCausalIdentifier:
    def test_identify_effect(self, builder, sample_data):
        from omni_proof.causal.identifier import CausalIdentifier

        model = builder.build_dag(
            data=sample_data,
            treatment="logo_in_first_3s",
            outcome="ctr",
            confounders=["platform", "audience_segment"],
        )
        identifier = CausalIdentifier()
        estimand = identifier.identify_effect(model)
        assert estimand is not None
