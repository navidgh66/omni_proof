"""Converts quantitative causal estimates into natural language design briefs."""

import structlog

from omni_proof.causal.results import CATEResult
from omni_proof.orchestration.models import DesignBrief

logger = structlog.get_logger()


def _classify_effect(effect: float, p_value: float) -> str:
    if p_value > 0.05:
        return "NEUTRAL"
    if effect > 0.05:
        return "RECOMMENDED"
    if effect > 0:
        return "CONSIDER"
    return "AVOID"


class InsightSynthesizer:
    """Translates CATE results into actionable design briefs."""

    def synthesize(self, cate_result: CATEResult) -> DesignBrief:
        segments = cate_result.segments
        breakdown = {}
        best_segment = None
        best_effect = float("-inf")

        for seg_name, estimate in segments.items():
            classification = _classify_effect(estimate.effect, estimate.p_value)
            pct = f"{estimate.effect * 100:+.1f}%"
            breakdown[seg_name] = f"{pct} — {classification}"
            if estimate.effect > best_effect and estimate.p_value <= 0.05:
                best_effect = estimate.effect
                best_segment = seg_name

        treatment_label = cate_result.treatment.replace("_", " ").title()

        if best_segment:
            finding = (
                f"{treatment_label} causes a {best_effect * 100:+.1f}% uplift "
                f"for the {best_segment} demographic"
            )
            recommendation = (
                f"Apply {treatment_label.lower()} for campaigns targeting {best_segment}."
            )
        else:
            finding = f"No statistically significant effect found for {treatment_label}."
            recommendation = "No action recommended based on current data."

        confidence = "HIGH" if cate_result.refutation_passed else "LOW"

        brief = DesignBrief(
            treatment=cate_result.treatment,
            finding=finding,
            segment_breakdown=breakdown,
            recommendation=recommendation,
            confidence=confidence,
        )
        logger.info("synthesized_brief", treatment=cate_result.treatment, confidence=confidence)
        return brief
