"""Generate synthetic ad metadata with planted treatment effects for E2E testing."""

from datetime import date, datetime
from uuid import uuid4

import numpy as np
import pandas as pd


def generate_synthetic_dataset(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Create synthetic ad data with known treatment effect of ~0.5 on CTR."""
    rng = np.random.RandomState(seed)

    platforms = rng.choice(["youtube", "instagram", "tiktok"], n)
    segments = rng.choice(["18-24", "25-34", "35-44"], n)
    seasons = rng.choice(["summer", "winter", "spring", "fall"], n)
    budgets = rng.uniform(1000, 50000, n)

    # Treatment: logo in first 3 seconds (binary)
    # Confounded by platform and budget
    treatment_prob = 0.3 + 0.2 * (platforms == "youtube").astype(float) + 0.1 * (budgets > 25000).astype(float)
    logo_in_first_3s = rng.binomial(1, np.clip(treatment_prob, 0, 1), n).astype(float)

    # Outcome: CTR with TRUE causal effect of 0.5 from treatment
    # Plus confounder effects
    ctr = (
        0.5 * logo_in_first_3s  # TRUE treatment effect
        + 0.3 * (platforms == "youtube").astype(float)
        + 0.2 * (budgets / 50000)
        + 0.1 * (segments == "18-24").astype(float)
        + rng.randn(n) * 0.3
    )

    return pd.DataFrame({
        "asset_id": [str(uuid4()) for _ in range(n)],
        "campaign_id": [str(uuid4()) for _ in range(n)],
        "platform": platforms,
        "audience_segment": segments,
        "season": seasons,
        "budget": budgets,
        "logo_in_first_3s": logo_in_first_3s,
        "ctr": ctr,
        "logo_screen_ratio": rng.uniform(0.05, 0.3, n),
        "scene_transitions": rng.randint(2, 15, n),
        "cta_type": rng.choice(["urgency", "passive", "inquisitive"], n),
        "audio_genre": rng.choice(["pop", "electronic", "ambient"], n),
        "impressions": rng.randint(10000, 500000, n),
        "clicks": (ctr * rng.randint(10000, 500000, n)).astype(int),
        "conversions": rng.randint(10, 1000, n),
        "roas": rng.uniform(1.0, 8.0, n),
        "date": [date(2026, rng.randint(1, 13), rng.randint(1, 29)) for _ in range(n)],
    })


SAMPLE_BRAND_GUIDE = {
    "brand_name": "OmniCorp",
    "primary_colors": ["#FF6B35", "#004E89", "#FFFFFF"],
    "approved_fonts": ["Inter", "Playfair Display"],
    "logo_min_clear_space_px": 24,
    "tone": "warm, contemporary, professional",
    "rules": [
        {"section": "color_palette", "description": "Primary palette: #FF6B35 (accent), #004E89 (base), #FFFFFF (background)"},
        {"section": "typography", "description": "Headlines: Playfair Display Bold. Body: Inter Regular 16px minimum"},
        {"section": "logo_rules", "description": "Minimum clear space of 24px around logo. Never distort aspect ratio."},
        {"section": "tone", "description": "Warm, contemporary aesthetic. Avoid cold or clinical imagery."},
    ],
}
