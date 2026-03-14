"""Causal effect identification using backdoor criterion."""

import warnings
from typing import Any

import structlog

logger = structlog.get_logger()


class CausalIdentifier:
    """Identifies causal estimands from DAGs using backdoor criterion."""

    def identify_effect(self, causal_model: Any) -> Any:
        # Uses Any to avoid tight coupling with DoWhy's IdentifiedEstimand
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimand = causal_model.identify_effect(proceed_when_unidentifiable=True)
        logger.info(
            "effect_identified",
            estimand_type=str(estimand.estimand_type)
            if hasattr(estimand, "estimand_type")
            else "unknown",
        )
        return estimand
