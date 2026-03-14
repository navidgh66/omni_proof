"""Generative prompt builder — NextAds closed-loop system."""

import structlog

logger = structlog.get_logger()


class GenerativePromptBuilder:
    """Combines causal insights + brand rules into parameterized creative prompts."""

    def build_prompt(
        self,
        cate_insights: list[dict],
        brand_rules: list[dict],
        target_segment: str,
        objective: str = "engagement",
        constraints: list[str] | None = None,
    ) -> str:
        lines = [f"Generate a creative asset optimized for {objective}."]
        lines.append(f"Target audience: {target_segment}")
        lines.append("")

        if cate_insights:
            lines.append("Causal insights (statistically validated):")
            for insight in cate_insights:
                effect = insight.get("effect", 0)
                treatment = insight.get("treatment", "unknown")
                lines.append(f"- {treatment}: {effect * 100:+.1f}% impact on {objective}")
            lines.append("")

        if brand_rules:
            lines.append("Brand guidelines:")
            for rule in brand_rules:
                lines.append(f"- {rule.get('description', '')}")
            lines.append("")

        if constraints:
            lines.append("Additional constraints:")
            for c in constraints:
                lines.append(f"- {c}")

        prompt = "\n".join(lines)
        logger.info("prompt_built", segment=target_segment, lines=len(lines))
        return prompt
