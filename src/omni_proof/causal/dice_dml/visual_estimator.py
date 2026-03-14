"""Visual DML estimator using disentangled embeddings."""

import warnings

import numpy as np
import structlog
from econml.dml import LinearDML
from lightgbm import LGBMRegressor, LGBMClassifier

from omni_proof.causal.dice_dml.disentangler import TreatmentDisentangler
from omni_proof.causal.results import ATEResult

logger = structlog.get_logger()


class VisualDMLEstimator:
    """DML estimation on disentangled visual embeddings (DICE-DML approach)."""

    def __init__(self, cv: int = 3):
        self._cv = cv
        self._disentangler = TreatmentDisentangler()

    def estimate_visual_ate(
        self,
        embeddings: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        treatment_fingerprint: np.ndarray,
    ) -> ATEResult:
        # Disentangle: remove treatment signal from embeddings
        W = self._disentangler.disentangle_batch(embeddings, treatment_fingerprint)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est = LinearDML(
                model_y=LGBMRegressor(n_estimators=50, verbose=-1),
                model_t=LGBMClassifier(n_estimators=50, verbose=-1),
                cv=self._cv,
                random_state=42,
                discrete_treatment=True,
            )
            est.fit(outcome, treatment, W=W)

        ate = float(est.ate())
        ci = est.ate_interval(alpha=0.05)

        return ATEResult(
            treatment="visual_treatment",
            outcome="visual_outcome",
            ate=ate,
            ci_lower=float(ci[0]),
            ci_upper=float(ci[1]),
            p_value=0.0,
            n_samples=len(outcome),
        )
