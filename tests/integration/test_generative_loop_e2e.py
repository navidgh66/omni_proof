"""E2E test: causal insights -> generative prompt."""

from omni_proof.api.generative_loop import GenerativePromptBuilder
from omni_proof.causal.results import CATEResult, EffectEstimate
from omni_proof.orchestration.insight_synthesizer import InsightSynthesizer
from tests.fixtures.synthetic_ads import SAMPLE_BRAND_GUIDE


class TestGenerativeLoopE2E:
    def test_cate_to_prompt_pipeline(self):
        """Full flow: CATE result -> design brief -> generative prompt."""
        # 1. Create CATE result (simulating estimation output)
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

        # 2. Synthesize brief
        synthesizer = InsightSynthesizer()
        brief = synthesizer.synthesize(cate)
        assert "18-24" in brief.finding
        assert brief.confidence == "HIGH"

        # 3. Build generative prompt with causal insights + brand rules
        builder = GenerativePromptBuilder()
        prompt = builder.build_prompt(
            cate_insights=[
                {"treatment": "fast pacing (first 3s)", "effect": 0.12},
                {"treatment": "urgency CTA", "effect": 0.06},
            ],
            brand_rules=SAMPLE_BRAND_GUIDE["rules"],
            target_segment="18-24",
            objective="conversion",
            constraints=["16:9 aspect ratio", "Max 30 seconds"],
        )

        # 4. Verify prompt contains all expected components
        assert "18-24" in prompt
        assert "conversion" in prompt
        assert "+12.0%" in prompt
        assert "Primary palette" in prompt or "#FF6B35" in prompt
        assert "16:9" in prompt
        assert "30 seconds" in prompt

    def test_prompt_without_significant_effects(self):
        """When no significant effects, prompt should still be valid."""
        builder = GenerativePromptBuilder()
        prompt = builder.build_prompt(
            cate_insights=[],
            brand_rules=SAMPLE_BRAND_GUIDE["rules"],
            target_segment="35-44",
            objective="brand_awareness",
        )
        assert "35-44" in prompt
        assert "brand_awareness" in prompt
        assert "Brand guidelines" in prompt
