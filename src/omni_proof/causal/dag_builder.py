"""Causal DAG construction using DoWhy for creative performance attribution."""

import warnings

import structlog
from dowhy import CausalModel

logger = structlog.get_logger()

# Common creative attribution DAG templates
TEMPLATES = {
    "logo_timing": {
        "treatment": "logo_in_first_3s",
        "outcome": "ctr",
        "confounders": ["platform", "audience_segment", "season", "budget"],
    },
    "color_temperature": {
        "treatment": "warm_color_palette",
        "outcome": "engagement_rate",
        "confounders": ["product_category", "production_quality", "platform"],
    },
    "audio_pacing": {
        "treatment": "fast_audio_pacing",
        "outcome": "conversion_rate",
        "confounders": ["audience_segment", "time_of_day", "platform", "budget"],
    },
}


class CausalDAGBuilder:
    """Constructs DoWhy causal models from treatment-confounder-outcome specifications."""

    def build_dag(
        self,
        data,
        treatment: str,
        outcome: str,
        confounders: list[str],
        effect_modifiers: list[str] | None = None,
        colliders: list[str] | None = None,
    ) -> CausalModel:
        if colliders:
            overlap = set(colliders) & set(confounders)
            if overlap:
                logger.warning(
                    "collider_in_confounders",
                    colliders=list(overlap),
                    msg="Colliders in adjustment set cause M-bias. Removing them.",
                )
                confounders = [c for c in confounders if c not in overlap]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = CausalModel(
                data=data,
                treatment=treatment,
                outcome=outcome,
                common_causes=confounders,
                effect_modifiers=effect_modifiers or [],
            )
        return model

    def build_from_template(self, data, template_name: str) -> CausalModel:
        if template_name not in TEMPLATES:
            raise ValueError(f"Unknown template: {template_name}. Available: {list(TEMPLATES)}")
        t = TEMPLATES[template_name]
        return self.build_dag(
            data=data,
            treatment=t["treatment"],
            outcome=t["outcome"],
            confounders=t["confounders"],
        )
