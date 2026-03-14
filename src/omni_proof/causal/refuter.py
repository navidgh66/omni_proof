"""Robustness checks for causal estimates."""

import warnings

import numpy as np
import structlog
from econml.dml import LinearDML
from lightgbm import LGBMRegressor, LGBMClassifier

from omni_proof.causal.results import RefutationResult

logger = structlog.get_logger()


class CausalRefuter:
    """Implements placebo, subset, and random confounder robustness tests."""

    def __init__(self, cv: int = 3):
        self._cv = cv

    def _fit_and_ate(self, Y, T, W) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est = LinearDML(
                model_y=LGBMRegressor(n_estimators=30, verbose=-1),
                model_t=LGBMClassifier(n_estimators=30, verbose=-1),
                discrete_treatment=True,
                cv=self._cv,
                random_state=42,
            )
            est.fit(Y, T, W=W)
        return float(est.ate())

    def placebo_test(
        self, data, treatment_col: str, outcome_col: str, confounder_cols: list[str],
    ) -> RefutationResult:
        Y = data[outcome_col].values.astype(float)
        T = data[treatment_col].values.astype(float)
        W = data[confounder_cols].values.astype(float)

        original_ate = self._fit_and_ate(Y, T, W)

        rng = np.random.RandomState(42)
        T_placebo = rng.permutation(T)
        placebo_ate = self._fit_and_ate(Y, T_placebo, W)

        passed = abs(placebo_ate) < abs(original_ate) * 0.5
        logger.info("placebo_test", original=original_ate, placebo=placebo_ate, passed=passed)

        return RefutationResult(
            test_name="placebo",
            original_effect=original_ate,
            new_effect=placebo_ate,
            passed=passed,
        )

    def subset_test(
        self, data, treatment_col: str, outcome_col: str, confounder_cols: list[str],
        drop_fraction: float = 0.1,
    ) -> RefutationResult:
        Y = data[outcome_col].values.astype(float)
        T = data[treatment_col].values.astype(float)
        W = data[confounder_cols].values.astype(float)

        original_ate = self._fit_and_ate(Y, T, W)

        rng = np.random.RandomState(42)
        n = len(Y)
        keep_idx = rng.choice(n, size=int(n * (1 - drop_fraction)), replace=False)
        subset_ate = self._fit_and_ate(Y[keep_idx], T[keep_idx], W[keep_idx])

        # Check if subset ATE is within reasonable range of original
        tolerance = abs(original_ate) * 1.0 + 0.01  # generous tolerance
        passed = abs(subset_ate - original_ate) < tolerance
        logger.info("subset_test", original=original_ate, subset=subset_ate, passed=passed)

        return RefutationResult(
            test_name="subset",
            original_effect=original_ate,
            new_effect=subset_ate,
            passed=passed,
        )

    def random_confounder_test(
        self, data, treatment_col: str, outcome_col: str, confounder_cols: list[str],
    ) -> RefutationResult:
        Y = data[outcome_col].values.astype(float)
        T = data[treatment_col].values.astype(float)
        W = data[confounder_cols].values.astype(float)

        original_ate = self._fit_and_ate(Y, T, W)

        rng = np.random.RandomState(42)
        noise = rng.randn(len(Y), 1)
        W_with_noise = np.hstack([W, noise])
        noisy_ate = self._fit_and_ate(Y, T, W_with_noise)

        tolerance = abs(original_ate) * 0.5 + 0.01
        passed = abs(noisy_ate - original_ate) < tolerance
        logger.info("random_confounder_test", original=original_ate, noisy=noisy_ate, passed=passed)

        return RefutationResult(
            test_name="random_confounder",
            original_effect=original_ate,
            new_effect=noisy_ate,
            passed=passed,
        )
