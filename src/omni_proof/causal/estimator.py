"""Double Machine Learning estimator using EconML."""

import warnings

import numpy as np
import structlog
from econml.dml import LinearDML, CausalForestDML
from lightgbm import LGBMRegressor, LGBMClassifier

from omni_proof.causal.results import ATEResult, CATEResult, EffectEstimate

logger = structlog.get_logger()


class DMLEstimator:
    """Wraps EconML DML estimators for causal effect estimation."""

    def __init__(self, cv: int = 5):
        self._cv = cv

    def estimate_ate(
        self, data, treatment_col: str, outcome_col: str, confounder_cols: list[str],
    ) -> ATEResult:
        Y = data[outcome_col].values.astype(float)
        T = data[treatment_col].values.astype(float)
        W = data[confounder_cols].values.astype(float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est = LinearDML(
                model_y=LGBMRegressor(n_estimators=50, verbose=-1),
                model_t=LGBMClassifier(n_estimators=50, verbose=-1),
                discrete_treatment=True,
                cv=self._cv,
                random_state=42,
            )
            est.fit(Y, T, W=W)

        ate = float(est.ate())
        ci = est.ate_interval(alpha=0.05)
        inference = est.effect_inference()
        p_val = float(inference.pvalue().mean()) if hasattr(inference, 'pvalue') else 0.0

        return ATEResult(
            treatment=treatment_col,
            outcome=outcome_col,
            ate=ate,
            ci_lower=float(ci[0]),
            ci_upper=float(ci[1]),
            p_value=p_val,
            n_samples=len(data),
        )

    def estimate_cate(
        self, data, treatment_col: str, outcome_col: str,
        confounder_cols: list[str], segment_col: str,
    ) -> CATEResult:
        Y = data[outcome_col].values.astype(float)
        T = data[treatment_col].values.astype(float)
        W = data[confounder_cols].values.astype(float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est = CausalForestDML(
                model_y=LGBMRegressor(n_estimators=50, verbose=-1),
                model_t=LGBMClassifier(n_estimators=50, verbose=-1),
                discrete_treatment=True,
                cv=self._cv,
                random_state=42,
                n_estimators=100,
            )
            est.fit(Y, T, X=W, W=W)

        segments = {}
        for segment_value in data[segment_col].unique():
            mask = data[segment_col] == segment_value
            segment_W = W[mask]
            effects = est.effect(X=segment_W)
            ci = est.effect_interval(X=segment_W, alpha=0.05)
            mean_effect = float(np.mean(effects))
            segments[str(segment_value)] = EffectEstimate(
                effect=mean_effect,
                ci_lower=float(np.mean(ci[0])),
                ci_upper=float(np.mean(ci[1])),
            )

        return CATEResult(
            treatment=treatment_col,
            outcome=outcome_col,
            segments=segments,
        )
