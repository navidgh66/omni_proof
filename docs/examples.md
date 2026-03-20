# Examples & Tutorials

## Demo Data

OmniProof ships with a complete demo dataset for a fictional brand, **Velocity Sportswear** (DTC activewear).

### Files

| Path | Description |
|------|-------------|
| `examples/creatives/` | 10 PNG images + 4 MP4 A/B video variants |
| `examples/data/campaign_performance.csv` | 1,000 rows of synthetic campaign data |
| `examples/data/brand_profile.json` | Brand identity (colors, typography, voice, rules) |
| `examples/data/brand_guidelines.json` | 12 brand rules (logo, color, typography, tone) |
| `examples/data/compliance_samples.json` | 5 sample compliance reports |
| `examples/data/creative_metadata_samples.json` | 14 creative metadata records (10 images + 4 videos) |

### Creative Assets

**Images** (10 PNGs): `runner_sunrise`, `trail_epic`, `hiit_studio`, `yoga_flow`, `sprint_track`, `swim_laps`, `basketball_court`, `cycling_mountain`, `crossfit_box`, `morning_stretch`

**Videos** (4 MP4s): A/B variants of `runner_sunrise` and `basketball_court` with fast/slow pacing treatment.

### Planted Causal Effects

The CSV has two deliberate causal effects for testing:

| Treatment | Outcome | True ATE | Heterogeneity |
|-----------|---------|----------|---------------|
| `fast_pacing` | `ctr` | ~+1.9 percentage points | Strong for Gen-Z, moderate for Millennials, weak for Gen-X |
| `warm_color_palette` | `ctr` | ~+0.6 percentage points | Homogeneous across segments |

---

## Tutorial 1: Offline Demo (No API Keys)

```bash
python examples/demo.py
```

Runs 9 stages in ~4 seconds. See [Getting Started](getting-started.md#first-run-offline-demo) for stage descriptions.

---

## Tutorial 2: Causal Analysis

Estimate treatment effects from the demo CSV:

```python
import pandas as pd
from omni_proof import DMLEstimator
from omni_proof.causal.dag_builder import CausalDAGBuilder
from omni_proof.causal.identifier import CausalIdentifier
from omni_proof.causal.refuter import CausalRefuter

df = pd.read_csv("examples/data/campaign_performance.csv")

# 1. Build causal DAG
builder = CausalDAGBuilder()
model = builder.build_dag(
    data=df,
    treatment="fast_pacing",
    outcome="ctr",
    confounders=["platform", "audience_segment", "daily_budget_usd", "region", "quarter"],
)

# 2. Identify causal effect
identifier = CausalIdentifier()
estimand = identifier.identify_effect(model)

# 3. Estimate ATE
estimator = DMLEstimator(cv=5, n_estimators=50)
ate = estimator.estimate_ate(
    df, "fast_pacing", "ctr",
    ["platform", "audience_segment", "daily_budget_usd", "region", "quarter"],
)
print(f"ATE: {ate.ate:+.4f}, p={ate.p_value:.4f}")

# 4. Estimate CATE by audience segment
cate = estimator.estimate_cate(
    df, "fast_pacing", "ctr",
    ["platform", "daily_budget_usd", "region", "quarter"],
    segment_col="audience_segment",
)
for seg, eff in cate.segments.items():
    print(f"  {seg}: {eff.effect:+.4f} (p={eff.p_value:.4f})")

# 5. Refute
refuter = CausalRefuter()
confounders = ["platform", "audience_segment", "daily_budget_usd", "region", "quarter"]
for test_fn in [refuter.placebo_test, refuter.subset_test, refuter.random_confounder_test]:
    result = test_fn(df, "fast_pacing", "ctr", confounders)
    print(f"  {result.test_name}: {'PASS' if result.passed else 'FAIL'}")
```

---

## Tutorial 3: Design Brief Generation

Convert causal results into actionable recommendations:

```python
from omni_proof import InsightSynthesizer

synth = InsightSynthesizer(p_value_threshold=0.05, recommend_threshold=0.05)
brief = synth.synthesize(cate)

print(f"Treatment: {brief.treatment}")
print(f"Finding: {brief.finding}")
print(f"Confidence: {brief.confidence}")
print()
for segment, label in brief.segment_breakdown.items():
    print(f"  {segment}: {label}")
print()
print(f"Recommendation: {brief.recommendation}")
```

---

## Tutorial 4: Brand Compliance (Requires API Keys)

```python
import asyncio
import json
from pathlib import Path
from omni_proof import GeminiClient, InMemoryVectorStore, ComplianceChain, Settings
from omni_proof.rag.brand_indexer import BrandIndexer
from omni_proof.rag.brand_retriever import BrandRetriever

settings = Settings()

async def main():
    gemini = GeminiClient(api_key=settings.gemini_api_key)
    store = InMemoryVectorStore()

    # Index brand guidelines
    indexer = BrandIndexer(gemini, store)
    with open("examples/data/brand_guidelines.json") as f:
        guidelines = json.load(f)
    for rule in guidelines["rules"]:
        await indexer.index_brand_guide_page(
            page_id=rule["rule_id"],
            page_content_path=rule["description"],  # text content
            section_type=rule["section_type"],
            page_number=rule.get("source_page", 0),
        )

    # Check compliance
    retriever = BrandRetriever(gemini, store)
    chain = ComplianceChain(gemini, retriever)
    report = await chain.check_compliance(
        asset_id="test-yoga",
        asset_path=Path("examples/creatives/yoga_flow.png"),
    )

    print(f"Passed: {report.passed}")
    print(f"Score: {report.score:.2f}")
    for v in report.violations:
        print(f"  [{v.severity}] {v.description}")

asyncio.run(main())
```

---

## Tutorial 5: DICE-DML with Video Pairs (Requires API Keys)

Use counterfactual video pairs to estimate the causal effect of pacing on embeddings:

```python
import asyncio
import numpy as np
from pathlib import Path
from omni_proof import GeminiClient, Settings
from omni_proof.causal.dice_dml.counterfactual_generator import CounterfactualGenerator
from omni_proof.causal.dice_dml.disentangler import TreatmentDisentangler
from omni_proof.causal.dice_dml.visual_estimator import VisualDMLEstimator

settings = Settings()

async def main():
    gemini = GeminiClient(api_key=settings.gemini_api_key)

    # Generate counterfactual pair
    generator = CounterfactualGenerator(gemini)
    pair = await generator.generate(
        original_path=Path("examples/creatives/runner_sunrise_fast_pacing.mp4"),
        counterfactual_path=Path("examples/creatives/runner_sunrise_slow_pacing.mp4"),
        treatment_attr="fast_pacing",
    )
    print(f"Background similarity: {pair.background_similarity:.4f}")

    # Extract treatment fingerprint
    disentangler = TreatmentDisentangler()
    fingerprint = disentangler.extract_treatment_fingerprint(
        pair.original_emb, pair.counterfactual_emb,
    )

    # (In production, you'd have embeddings for all creatives)
    # Here we simulate with the pair + some noise
    all_embeddings = np.random.randn(100, 3072).astype(np.float32)
    treatment = np.random.binomial(1, 0.5, 100)
    outcome = treatment * 0.02 + np.random.randn(100) * 0.01  # planted effect

    # Estimate visual ATE
    estimator = VisualDMLEstimator(cv=3)
    ate = estimator.estimate_visual_ate(
        embeddings=all_embeddings,
        treatment=treatment,
        outcome=outcome,
        treatment_fingerprint=fingerprint,
    )
    print(f"Visual ATE: {ate.ate:+.4f} (p={ate.p_value:.4f})")

asyncio.run(main())
```

---

## Tutorial 6: Creative Prompt Generation

Combine causal insights with brand rules to generate optimized creative prompts:

```python
from omni_proof.api.generative_loop import GenerativePromptBuilder

builder = GenerativePromptBuilder()
prompt = builder.build_prompt(
    cate_insights=[
        {"treatment": "fast_pacing", "effect": "+3.2%", "segment": "Gen-Z 18-24"},
        {"treatment": "warm_color_palette", "effect": "+0.6%", "segment": "all"},
    ],
    brand_rules=[
        {"type": "color", "description": "Primary: #FF5733, Secondary: #1A1A2E"},
        {"type": "typography", "description": "Montserrat Bold for headlines"},
        {"type": "tone", "description": "Energetic, action-oriented"},
    ],
    target_segment="Gen-Z 18-24",
    objective="engagement",
    constraints=["15-second format", "Instagram Reels"],
)
print(prompt)
```

---

## Jupyter Playground

For a full interactive walkthrough including Pinecone integration and DICE-DML:

```bash
jupyter notebook examples/playground.ipynb
```

The notebook covers 10 steps including embedding generation, vector storage, causal analysis, and the full DICE-DML pipeline. A pre-run version with saved outputs is at `examples/playground_output.ipynb`.
