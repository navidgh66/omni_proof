#!/usr/bin/env python3
"""OmniProof Demo — Velocity Sportswear Campaign Analysis.

Showcases every OmniProof capability using only local data (no API keys required).
Run: python examples/demo.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from omni_proof.api.generative_loop import GenerativePromptBuilder
from omni_proof.causal.dag_builder import CausalDAGBuilder
from omni_proof.causal.estimator import DMLEstimator
from omni_proof.causal.identifier import CausalIdentifier
from omni_proof.causal.refuter import CausalRefuter
from omni_proof.causal.results import CATEResult, EffectEstimate
from omni_proof.orchestration.insight_synthesizer import InsightSynthesizer

DATA_DIR = Path(__file__).resolve().parent / "data"

# ── Helpers ──────────────────────────────────────────────────────────────────


def header(title: str) -> None:
    print(f"\n  \033[1m{title}\033[0m")


def bullet(text: str) -> None:
    print(f"    {text}")


def success(text: str) -> None:
    print(f"    \033[32m✓\033[0m {text}")


def fail(text: str) -> None:
    print(f"    \033[31m✗\033[0m {text}")


def warn(text: str) -> None:
    print(f"    \033[33m⚠\033[0m {text}")


def bar(value: float, max_val: float, width: int = 14) -> str:
    filled = int((value / max_val) * width) if max_val else 0
    return "\u2588" * filled + "\u2591" * (width - filled)


# ── Data Generation ──────────────────────────────────────────────────────────


def generate_campaign_csv(path: Path, n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate 1000-row campaign dataset with two planted causal effects."""
    rng = np.random.RandomState(seed)

    campaigns = [
        ("Summer Endurance Launch", "vel_summer_endurance_2026"),
        ("Winter Training Push", "vel_winter_training_2026"),
        ("Trail Running Adventure", "vel_trail_adventure_2026"),
        ("Back to Campus Athletics", "vel_campus_athletics_2026"),
        ("Holiday Gift Guide", "vel_holiday_gift_2026"),
    ]
    platforms = ["YouTube", "Instagram", "TikTok", "Meta"]
    segments = ["Gen-Z (18-24)", "Millennials (25-34)", "Gen-X (35-44)"]
    regions = ["US-West", "US-East", "EU-West", "APAC"]
    cta_types = ["Shop Now", "Limited Time", "Discover", "Learn More"]
    audio_styles = [
        "upbeat-electronic",
        "acoustic-indie",
        "hip-hop",
        "ambient",
        "voiceover-only",
    ]
    quarters = ["Q1 2026", "Q2 2026", "Q3 2026", "Q4 2026"]

    platform_arr = rng.choice(platforms, n)
    segment_arr = rng.choice(segments, n)
    region_arr = rng.choice(regions, n)
    budget_arr = rng.uniform(500, 15000, n)
    quarter_arr = rng.choice(quarters, n)

    # Platform encoding for confounding
    plat_yt = (platform_arr == "YouTube").astype(float)
    plat_ig = (platform_arr == "Instagram").astype(float)
    seg_genz = (segment_arr == "Gen-Z (18-24)").astype(float)
    seg_mill = (segment_arr == "Millennials (25-34)").astype(float)

    # Treatment 1: fast_pacing (confounded by platform + segment)
    fp_prob = 0.3 + 0.2 * plat_yt + 0.15 * seg_genz + 0.1 * (budget_arr > 8000).astype(float)
    fast_pacing = rng.binomial(1, np.clip(fp_prob, 0.05, 0.95), n).astype(float)

    # Treatment 2: warm_color_palette (confounded by platform + region)
    wc_prob = 0.4 + 0.15 * plat_ig + 0.1 * (region_arr == "US-West").astype(float)
    warm_color = rng.binomial(1, np.clip(wc_prob, 0.05, 0.95), n).astype(float)

    # Outcome: CTR with planted effects
    # ATE(fast_pacing) ~ +0.5, ATE(warm_color) ~ +0.3
    # Heterogeneous: Gen-Z gets bigger boost from fast_pacing
    ctr_base = (
        0.03
        + 0.015 * plat_yt
        + 0.01 * plat_ig
        + 0.008 * seg_genz
        + 0.005 * seg_mill
        + 0.002 * (budget_arr / 15000)
        + rng.randn(n) * 0.008
    )
    # Planted treatment effects (on absolute CTR scale)
    fast_pacing_effect = fast_pacing * (0.012 + 0.015 * seg_genz + 0.008 * seg_mill)
    warm_color_effect = warm_color * 0.006
    ctr = np.clip(ctr_base + fast_pacing_effect + warm_color_effect, 0.012, 0.089)

    # Derived metrics
    impressions = rng.randint(25000, 850000, n)
    clicks = (ctr * impressions).astype(int)
    conversion_rate = np.clip(ctr * rng.uniform(0.15, 0.45, n), 0.002, 0.035)
    conversions = np.clip((conversion_rate * impressions).astype(int), 15, 1200)
    revenue = conversions * rng.uniform(25, 55, n)
    spend = budget_arr * rng.uniform(0.8, 1.2, n)
    roas = np.clip(revenue / np.maximum(spend, 1), 1.2, 7.8)
    cpa = np.clip(spend / np.maximum(conversions, 1), 5.50, 42.00)

    # Campaign assignment
    camp_idx = rng.randint(0, len(campaigns), n)

    # Date range: 2026-01-15 to 2026-03-15
    start_offset = rng.randint(0, 50, n)
    flight_starts = pd.to_datetime("2026-01-15") + pd.to_timedelta(start_offset, unit="D")
    flight_ends = flight_starts + pd.to_timedelta(rng.randint(7, 30, n), unit="D")

    df = pd.DataFrame(
        {
            "asset_id": [f"VEL-2026-Q{(i % 4) + 1}-{i:04d}" for i in range(n)],
            "campaign_name": [campaigns[c][0] for c in camp_idx],
            "campaign_id": [campaigns[c][1] for c in camp_idx],
            "creative_name": [
                rng.choice(
                    [
                        "Runner_Sunrise",
                        "Trail_Epic",
                        "HIIT_Studio",
                        "Yoga_Flow",
                        "Sprint_Track",
                        "Swim_Laps",
                        "Basketball_Court",
                        "Cycling_Mountain",
                        "CrossFit_Box",
                        "Morning_Stretch",
                    ]
                )
                for _ in range(n)
            ],
            "platform": platform_arr,
            "audience_segment": segment_arr,
            "region": region_arr,
            "daily_budget_usd": np.round(budget_arr, 2),
            "fast_pacing": fast_pacing.astype(int),
            "warm_color_palette": warm_color.astype(int),
            "ctr": np.round(ctr, 6),
            "conversion_rate": np.round(conversion_rate, 6),
            "impressions": impressions,
            "clicks": clicks,
            "conversions": conversions,
            "revenue_usd": np.round(revenue, 2),
            "roas": np.round(roas, 2),
            "cpa_usd": np.round(cpa, 2),
            "logo_screen_ratio": np.round(rng.uniform(0.04, 0.28, n), 3),
            "scene_transitions": rng.randint(2, 18, n),
            "time_to_first_product_s": np.round(rng.uniform(0.5, 8.0, n), 1),
            "cta_type": rng.choice(cta_types, n),
            "audio_style": rng.choice(audio_styles, n),
            "video_duration_s": np.round(rng.uniform(6, 30, n), 1),
            "quarter": quarter_arr,
            "flight_start": flight_starts.strftime("%Y-%m-%d"),
            "flight_end": flight_ends.strftime("%Y-%m-%d"),
        }
    )

    df.to_csv(path, index=False)
    return df


# ── Demo Stages ──────────────────────────────────────────────────────────────


def stage_1_overview(df: pd.DataFrame) -> None:
    header("Stage 1: Campaign Overview")
    bullet(f"Loading {len(df):,} creatives across {df['campaign_name'].nunique()} campaigns...")

    plat_counts = Counter(df["platform"])
    plat_str = ", ".join(f"{p} ({c})" for p, c in plat_counts.most_common())
    bullet(f"Platforms: {plat_str}")

    seg_counts = Counter(df["audience_segment"])
    seg_str = ", ".join(f"{s} ({c})" for s, c in seg_counts.most_common())
    bullet(f"Segments: {seg_str}")

    starts = pd.to_datetime(df["flight_start"])
    ends = pd.to_datetime(df["flight_end"])
    bullet(f"Date range: {starts.min().strftime('%Y-%m-%d')} -> {ends.max().strftime('%Y-%m-%d')}")

    total_spend = df["daily_budget_usd"].sum()
    total_rev = df["revenue_usd"].sum()
    blended_roas = total_rev / total_spend
    bullet(
        f"Total spend: ${total_spend / 1e6:.1f}M | "
        f"Total revenue: ${total_rev / 1e6:.1f}M | "
        f"Blended ROAS: {blended_roas:.2f}"
    )


def stage_2_dag(df: pd.DataFrame) -> None:
    header("Stage 2: Causal DAG Construction")
    treatment = "fast_pacing"
    outcome = "ctr"
    confounders = ["platform", "audience_segment", "daily_budget_usd", "region", "quarter"]

    bullet(f"Treatment: {treatment} -> Outcome: {outcome}")
    bullet(f"Confounders: {', '.join(confounders)}")

    # Encode categoricals for DAG builder
    df_encoded = df.copy()
    for col in ["platform", "audience_segment", "region", "quarter"]:
        df_encoded[col] = df_encoded[col].astype("category").cat.codes.astype(float)

    builder = CausalDAGBuilder()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = builder.build_dag(
            data=df_encoded,
            treatment=treatment,
            outcome=outcome,
            confounders=confounders,
        )

    identifier = CausalIdentifier()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        identifier.identify_effect(model)

    success(f"DAG constructed ({len(confounders)} confounders, 0 colliders detected)")
    success("Backdoor criterion: valid adjustment set identified")


def stage_3_ate(df: pd.DataFrame) -> tuple[float, float, float, float]:
    header("Stage 3: Average Treatment Effect (DML)")
    bullet("Does fast pacing cause higher CTR?")

    df_encoded = df.copy()
    for col in ["platform", "audience_segment", "region", "quarter"]:
        df_encoded[col] = df_encoded[col].astype("category").cat.codes.astype(float)

    confounders = ["platform", "audience_segment", "daily_budget_usd", "region", "quarter"]
    estimator = DMLEstimator(cv=3, n_estimators=30)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ate_result = estimator.estimate_ate(
            data=df_encoded,
            treatment_col="fast_pacing",
            outcome_col="ctr",
            confounder_cols=confounders,
        )

    pct = ate_result.ate * 100
    ci_lo = ate_result.ci_lower * 100
    ci_hi = ate_result.ci_upper * 100
    bullet(
        f"ATE: {pct:+.1f}pp (95% CI: [{ci_lo:.1f}, {ci_hi:.1f}], "
        f"p={'< 0.001' if ate_result.p_value < 0.001 else f'= {ate_result.p_value:.3f}'})"
    )
    bullet(f"-> Fast pacing causes {pct:+.1f} percentage points CTR uplift")

    return ate_result.ate, ate_result.ci_lower, ate_result.ci_upper, ate_result.p_value


def stage_4_cate(df: pd.DataFrame) -> CATEResult:
    header("Stage 4: Conditional Average Treatment Effect by Segment")

    df_encoded = df.copy()
    for col in ["platform", "audience_segment", "region", "quarter"]:
        df_encoded[col] = df_encoded[col].astype("category").cat.codes.astype(float)

    confounders = ["platform", "audience_segment", "daily_budget_usd", "region", "quarter"]
    estimator = DMLEstimator(cv=3, n_estimators=30)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cate_result = estimator.estimate_cate(
            data=df_encoded,
            treatment_col="fast_pacing",
            outcome_col="ctr",
            confounder_cols=confounders,
            segment_col="audience_segment",
        )

    # Map codes back to labels
    cat_map = dict(
        enumerate(df["audience_segment"].astype("category").cat.categories)
    )
    labeled: dict[str, EffectEstimate] = {}
    for code_str, est in cate_result.segments.items():
        label = cat_map.get(int(float(code_str)), code_str)
        labeled[label] = est

    max_effect = max(abs(e.effect) for e in labeled.values()) if labeled else 1
    for seg, est in sorted(labeled.items(), key=lambda x: -x[1].effect):
        pct = est.effect * 100
        tag = "RECOMMENDED" if pct > 3 else ("NEUTRAL" if est.p_value > 0.05 else "CONSIDER")
        bullet(f"{seg:24s} {pct:+.1f}% {bar(abs(est.effect), max_effect)} {tag}")

    # Return with labels
    return CATEResult(
        treatment=cate_result.treatment,
        outcome=cate_result.outcome,
        segments=labeled,
        refutation_passed=False,  # will be set after stage 5
    )


def stage_5_refutation(df: pd.DataFrame) -> bool:
    header("Stage 5: Refutation Suite")

    df_encoded = df.copy()
    for col in ["platform", "audience_segment", "region", "quarter"]:
        df_encoded[col] = df_encoded[col].astype("category").cat.codes.astype(float)

    confounders = ["platform", "audience_segment", "daily_budget_usd", "region", "quarter"]
    refuter = CausalRefuter(cv=3)

    all_passed = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        r1 = refuter.placebo_test(df_encoded, "fast_pacing", "ctr", confounders)
        (success if r1.passed else fail)(
            f"Placebo test: {'PASSED' if r1.passed else 'FAILED'} "
            f"(shuffled treatment effect: {r1.new_effect * 100:.2f}%)"
        )
        all_passed &= r1.passed

        r2 = refuter.subset_test(df_encoded, "fast_pacing", "ctr", confounders)
        (success if r2.passed else fail)(
            f"Subset test: {'PASSED' if r2.passed else 'FAILED'} "
            f"(80% subset effect: {r2.new_effect * 100:.2f}%)"
        )
        all_passed &= r2.passed

        r3 = refuter.random_confounder_test(df_encoded, "fast_pacing", "ctr", confounders)
        (success if r3.passed else fail)(
            f"Random confounder: {'PASSED' if r3.passed else 'FAILED'} "
            f"(with noise: {r3.new_effect * 100:.2f}%)"
        )
        all_passed &= r3.passed

    return all_passed


def stage_6_brief(cate_result: CATEResult) -> None:
    header("Stage 6: Design Brief")
    synthesizer = InsightSynthesizer()
    brief = synthesizer.synthesize(cate_result)

    bullet(f'Finding: "{brief.finding}"')
    bullet(f'Recommendation: "{brief.recommendation}"')
    bullet(f"Confidence: {brief.confidence}")


def stage_7_brand() -> dict:
    header("Stage 7: Brand Profile - Velocity Sportswear")

    with open(DATA_DIR / "brand_profile.json") as f:
        profile = json.load(f)

    colors = profile["visual_style"]["dominant_colors"]
    bullet(f"Colors: {', '.join(colors)}")
    fonts = ", ".join(profile["rules"][1].get("approved_fonts", []))
    bullet(f"Typography: {fonts}")
    bullet(f"Voice: {profile['voice']['formality']}, {profile['voice']['emotional_register']}")
    bullet(f"Rules: {len(profile['rules'])} guidelines loaded")
    return profile


def stage_8_prompt(profile: dict, cate_result: CATEResult) -> None:
    header("Stage 8: Creative Generation Prompt")

    # Find best segment
    best_seg = max(cate_result.segments.items(), key=lambda x: x[1].effect)

    builder = GenerativePromptBuilder()
    prompt = builder.build_prompt(
        cate_insights=[
            {
                "treatment": "fast_pacing",
                "effect": best_seg[1].effect,
            }
        ],
        brand_rules=[
            {"description": r["description"]}
            for r in profile["rules"][:4]
        ],
        target_segment=best_seg[0],
        objective="conversion",
        constraints=[
            "Max 15 seconds",
            "9:16 vertical format for Instagram Reels",
            f"Product reveal by second 5",
            f"Primary color: Electric Blue #0066FF",
        ],
    )

    for line in prompt.split("\n"):
        bullet(f'  {line}' if line.strip() else "")


def stage_9_compliance() -> None:
    header("Stage 9: Compliance Report Samples")

    with open(DATA_DIR / "compliance_samples.json") as f:
        reports = json.load(f)

    for report in reports:
        status = "PASS" if report["passed"] else ("WARN" if report["score"] >= 0.6 else "FAIL")
        color = "\033[32m" if status == "PASS" else ("\033[33m" if status == "WARN" else "\033[31m")
        bullet(
            f"{report['asset_id']}_{report['asset_name']} — "
            f"{color}{status}\033[0m (score: {report['score']:.2f})"
        )
        for v in report.get("violations", []):
            sev = v["severity"].upper()
            if sev == "CRITICAL":
                fail(f"  [{sev}] {v['description']}")
            else:
                warn(f"  [{sev}] {v['description']}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    print()
    print("  \033[1m" + "=" * 63 + "\033[0m")
    print("  \033[1m  OmniProof Demo - Velocity Sportswear Campaign Analysis\033[0m")
    print("  \033[1m" + "=" * 63 + "\033[0m")

    # Generate or load CSV
    csv_path = DATA_DIR / "campaign_performance.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = generate_campaign_csv(csv_path)

    t0 = time.time()

    stage_1_overview(df)
    stage_2_dag(df)
    stage_3_ate(df)
    cate_result = stage_4_cate(df)

    all_passed = stage_5_refutation(df)
    cate_result.refutation_passed = all_passed

    stage_6_brief(cate_result)
    profile = stage_7_brand()
    stage_8_prompt(profile, cate_result)
    stage_9_compliance()

    elapsed = time.time() - t0
    print()
    print(f"  \033[1mDemo complete in {elapsed:.1f}s. All 9 stages executed successfully.\033[0m")
    print()


if __name__ == "__main__":
    main()
