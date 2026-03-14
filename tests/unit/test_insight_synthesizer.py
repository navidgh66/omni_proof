"""Tests for insight synthesizer."""

from omni_proof.causal.results import CATEResult, EffectEstimate
from omni_proof.orchestration.insight_synthesizer import InsightSynthesizer


class TestInsightSynthesizer:
    def test_synthesize_with_significant_effects(self):
        cate = CATEResult(
            treatment="fast_pacing_first_3s",
            outcome="ctr",
            segments={
                "18-24": EffectEstimate(effect=0.12, ci_lower=0.08, ci_upper=0.16, p_value=0.001),
                "25-34": EffectEstimate(effect=0.05, ci_lower=0.01, ci_upper=0.09, p_value=0.02),
                "35-44": EffectEstimate(effect=-0.02, ci_lower=-0.06, ci_upper=0.02, p_value=0.31),
            },
            refutation_passed=True,
        )
        synthesizer = InsightSynthesizer()
        brief = synthesizer.synthesize(cate)

        assert brief.treatment == "fast_pacing_first_3s"
        assert "18-24" in brief.finding
        assert "+12.0%" in brief.finding
        assert brief.confidence == "HIGH"
        assert "RECOMMENDED" in brief.segment_breakdown["18-24"]
        assert "NEUTRAL" in brief.segment_breakdown["35-44"]

    def test_synthesize_no_significant_effects(self):
        cate = CATEResult(
            treatment="blue_background",
            outcome="ctr",
            segments={
                "18-24": EffectEstimate(effect=0.01, ci_lower=-0.03, ci_upper=0.05, p_value=0.6),
            },
            refutation_passed=False,
        )
        synthesizer = InsightSynthesizer()
        brief = synthesizer.synthesize(cate)

        assert "No statistically significant" in brief.finding
        assert brief.confidence == "LOW"

    def test_synthesize_segment_breakdown_format(self):
        cate = CATEResult(
            treatment="urgency_cta",
            outcome="conversion",
            segments={
                "18-24": EffectEstimate(effect=0.08, ci_lower=0.04, ci_upper=0.12, p_value=0.001),
            },
            refutation_passed=True,
        )
        brief = InsightSynthesizer().synthesize(cate)
        assert "18-24" in brief.segment_breakdown
        assert "+" in brief.segment_breakdown["18-24"]

    def test_synthesize_with_custom_thresholds(self):
        cate = CATEResult(
            treatment="custom_treatment",
            outcome="ctr",
            segments={
                "segment_a": EffectEstimate(
                    effect=0.03, ci_lower=0.01, ci_upper=0.05, p_value=0.02
                ),
                "segment_b": EffectEstimate(
                    effect=0.08, ci_lower=0.04, ci_upper=0.12, p_value=0.15
                ),
            },
            refutation_passed=True,
        )
        synthesizer = InsightSynthesizer(p_value_threshold=0.1, recommend_threshold=0.07)
        brief = synthesizer.synthesize(cate)

        # segment_a: effect=0.03 < recommend_threshold=0.07, p_value=0.02 <= 0.1 -> CONSIDER
        assert "CONSIDER" in brief.segment_breakdown["segment_a"]
        # segment_b: effect=0.08 > recommend_threshold=0.07, but p_value=0.15 > 0.1 -> NEUTRAL
        assert "NEUTRAL" in brief.segment_breakdown["segment_b"]
        # segment_a is best segment since it's significant (p<=0.1) and has positive effect
        assert "segment_a" in brief.finding
        assert "+3.0%" in brief.finding
