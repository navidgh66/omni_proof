# Spec: Layer 4 — Orchestration & RAG Execution

## Overview
LangGraph orchestrates two main workflows: brand compliance checking and insight synthesis.

## Brand Compliance Pipeline (LangGraph)

### State Graph
```
[embed_asset] -> [retrieve_guidelines] -> [evaluate_compliance] -> [generate_report]
```

### Node Details

**embed_asset:**
- Input: uploaded creative (image/video)
- Action: generate Gemini embedding
- Output: dense vector + extracted metadata

**retrieve_guidelines:**
- Input: dense vector from previous node
- Action: query `brand_assets` collection in vector DB (top_k=10)
- Output: list of relevant brand rules with source references

**evaluate_compliance:**
- Input: creative + retrieved brand rules
- Action: send both to Gemini with compliance evaluation prompt
- Checks:
  - Concrete: hex color codes match palette, logo clear space >= N pixels, font family matches approved list
  - Semantic: aesthetic tone aligns with brand description ("warm, contemporary")
- Output: list of violations with severity (critical/warning/info)

**generate_report:**
- Input: violations list
- Output: `ComplianceResult` with pass/fail, violations, evidence citations from brand guide

## Insight Synthesis

### Input
```python
CATEResult(
    treatment="fast_pacing_first_3s",
    segments={
        "18-24": EffectEstimate(effect=0.12, ci=(0.08, 0.16), p=0.001),
        "25-34": EffectEstimate(effect=0.05, ci=(0.01, 0.09), p=0.02),
        "35-44": EffectEstimate(effect=-0.02, ci=(-0.06, 0.02), p=0.31),
    },
    refutation_passed=True
)
```

### Output (DesignBrief)
```
CAUSAL INSIGHT: Fast Pacing in Opening 3 Seconds

FINDING: Increasing video pacing in the first 3 seconds causes a
+12% uplift in CTR for the 18-24 demographic (CI: 8-16%, p<0.001),
independent of production quality, platform, and seasonal timing.

SEGMENT BREAKDOWN:
- 18-24: Strong positive (+12%) — RECOMMENDED
- 25-34: Moderate positive (+5%) — CONSIDER
- 35-44: No significant effect — NEUTRAL

RECOMMENDATION: Apply fast-paced opening for campaigns targeting
18-34 demographics. Maintain current pacing for 35+ audiences.

CONFIDENCE: HIGH (all refutation checks passed)
```
