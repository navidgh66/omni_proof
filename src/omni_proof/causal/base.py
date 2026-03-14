"""Abstract base class for causal estimators."""

from abc import ABC, abstractmethod

from omni_proof.causal.results import ATEResult, CATEResult


class Estimator(ABC):
    """Abstract base class for causal effect estimators."""

    @abstractmethod
    def estimate_ate(
        self,
        data,
        treatment_col: str,
        outcome_col: str,
        confounder_cols: list[str],
    ) -> ATEResult:
        """Estimate Average Treatment Effect (ATE).

        Args:
            data: DataFrame containing treatment, outcome, and confounders
            treatment_col: Name of the treatment column
            outcome_col: Name of the outcome column
            confounder_cols: List of confounder column names

        Returns:
            ATEResult with effect estimate, confidence interval, and p-value
        """
        ...

    @abstractmethod
    def estimate_cate(
        self,
        data,
        treatment_col: str,
        outcome_col: str,
        confounder_cols: list[str],
        segment_col: str,
    ) -> CATEResult:
        """Estimate Conditional Average Treatment Effect (CATE) by segment.

        Args:
            data: DataFrame containing treatment, outcome, confounders, and segments
            treatment_col: Name of the treatment column
            outcome_col: Name of the outcome column
            confounder_cols: List of confounder column names
            segment_col: Name of the segmentation column

        Returns:
            CATEResult with segment-specific effect estimates
        """
        ...
