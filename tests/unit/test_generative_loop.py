"""Tests for generative prompt builder."""

from omni_proof.api.generative_loop import GenerativePromptBuilder


class TestGenerativePromptBuilder:
    def test_basic_prompt(self):
        builder = GenerativePromptBuilder()
        prompt = builder.build_prompt(
            cate_insights=[],
            brand_rules=[],
            target_segment="18-24",
            objective="engagement",
        )
        assert "18-24" in prompt
        assert "engagement" in prompt

    def test_prompt_with_insights(self):
        builder = GenerativePromptBuilder()
        prompt = builder.build_prompt(
            cate_insights=[
                {"treatment": "fast_pacing", "effect": 0.12},
                {"treatment": "warm_colors", "effect": 0.08},
            ],
            brand_rules=[],
            target_segment="25-34",
        )
        assert "+12.0%" in prompt
        assert "fast_pacing" in prompt.lower() or "fast pacing" in prompt.lower()

    def test_prompt_with_brand_rules(self):
        builder = GenerativePromptBuilder()
        prompt = builder.build_prompt(
            cate_insights=[],
            brand_rules=[{"description": "Use hex #FF6B35 as primary color"}],
            target_segment="18-24",
        )
        assert "#FF6B35" in prompt

    def test_prompt_with_constraints(self):
        builder = GenerativePromptBuilder()
        prompt = builder.build_prompt(
            cate_insights=[],
            brand_rules=[],
            target_segment="18-24",
            constraints=["16:9 aspect ratio", "Max 30 seconds"],
        )
        assert "16:9" in prompt
        assert "30 seconds" in prompt
