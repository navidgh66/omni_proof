# OmniProof Demo — Velocity Sportswear

Interactive demo showcasing all OmniProof capabilities using a fictional DTC activewear brand.

## Quick Start

```bash
# From the repo root (requires the package to be installed)
pip install -e ".[dev]"
python examples/demo.py
```

No API keys required — the demo uses local DML estimation, pre-built brand profiles, and sample data.

## What It Demonstrates

| Stage | Capability | Module Used |
|-------|-----------|-------------|
| 1 | Campaign data loading & overview | pandas |
| 2 | Causal DAG construction | `CausalDAGBuilder` + DoWhy |
| 3 | Average Treatment Effect (ATE) via DML | `DMLEstimator` (LinearDML) |
| 4 | Conditional ATE by audience segment | `DMLEstimator` (CausalForestDML) |
| 5 | Robustness refutation suite | `CausalRefuter` (placebo, subset, random confounder) |
| 6 | Natural language design brief | `InsightSynthesizer` |
| 7 | Brand profile loading | `BrandProfile` schema |
| 8 | Generative creative prompt | `GenerativePromptBuilder` |
| 9 | Brand compliance reports | `ComplianceReport` schema |

## Data Files

| File | Description |
|------|-------------|
| `data/campaign_performance.csv` | 1,000-row synthetic campaign dataset with planted causal effects |
| `data/brand_profile.json` | Velocity Sportswear brand profile (matches `BrandProfile` schema) |
| `data/brand_guidelines.json` | 12 brand guidelines for RAG retrieval |
| `data/compliance_samples.json` | 5 sample compliance reports (PASS/WARN/FAIL) |
| `data/creative_metadata_samples.json` | 20 creative metadata records (matches `CreativeMetadata` schema) |

## Planted Causal Effects

The campaign dataset has two treatments with known ground-truth effects:

- **`fast_pacing`**: True ATE ~ +1.2-1.5pp on CTR, heterogeneous by segment (Gen-Z gets larger effect)
- **`warm_color_palette`**: True ATE ~ +0.6pp on CTR

Both are confounded by platform, audience segment, budget, and region — DML correctly recovers the effects.
