# Causal Methodology

OmniProof uses causal inference — not correlations — to determine what creative elements actually drive performance. This document explains the statistical methods, why they matter, and how they're implemented.

## Why Causal Inference?

Standard A/B testing answers: *"Did version A outperform version B?"*

OmniProof answers: *"Does fast pacing **cause** higher CTR, controlling for platform, budget, audience, and seasonality — and for which segments?"*

The difference: A/B tests require controlled experiments. Causal inference extracts the same rigor from observational data you already have.

## The Pipeline

```
1. DAG Construction    →  Encode domain knowledge as a causal graph
2. Identification      →  Find valid adjustment sets (backdoor criterion)
3. Estimation (DML)    →  Estimate ATE and CATE via Double Machine Learning
4. Refutation          →  Stress-test the estimate with robustness checks
5. Synthesis           →  Translate into actionable design briefs
```

## Step 1: Directed Acyclic Graph (DAG)

A causal DAG encodes assumptions about which variables affect which. OmniProof builds DAGs programmatically:

```python
from omni_proof.causal.dag_builder import CausalDAGBuilder

builder = CausalDAGBuilder()
model = builder.build_dag(
    data=df,
    treatment="fast_pacing",
    outcome="ctr",
    confounders=["platform", "audience_segment", "daily_budget_usd", "region"],
)
```

**Example DAG:**

```
platform ──────────┐
audience_segment ──┤
daily_budget_usd ──┤──→ fast_pacing ──→ ctr
region ────────────┘         │
                             └──→ ctr
```

Confounders (platform, budget, etc.) affect both the treatment *and* the outcome. Failing to control for them produces biased estimates.

### Built-in Templates

For common creative questions, use preset templates:

```python
model = builder.build_from_template(data=df, template_name="logo_timing")
# Treatment: logo_in_first_3s → Outcome: ctr
# Confounders: platform, audience_segment, season, budget
```

## Step 2: Identification

The `CausalIdentifier` applies the **backdoor criterion** to find the minimal set of variables that blocks all confounding paths:

```python
from omni_proof.causal.identifier import CausalIdentifier

identifier = CausalIdentifier()
estimand = identifier.identify_effect(model)
```

If a valid adjustment set exists, the effect is identifiable from observational data.

## Step 3: Estimation (Double Machine Learning)

### Average Treatment Effect (ATE)

DML uses a two-stage procedure to estimate unbiased treatment effects:

1. **First stage:** Fit flexible ML models (LightGBM) to predict treatment and outcome from confounders
2. **Second stage:** Estimate the treatment effect from the residuals (what confounders can't explain)

```python
from omni_proof import DMLEstimator

estimator = DMLEstimator(cv=5, n_estimators=50)
ate = estimator.estimate_ate(
    data=df,
    treatment_col="fast_pacing",
    outcome_col="ctr",
    confounder_cols=["platform", "audience_segment", "daily_budget_usd"],
)

print(f"ATE: {ate.ate:+.4f}")           # e.g., +0.0189
print(f"95% CI: [{ate.ci_lower:.4f}, {ate.ci_upper:.4f}]")
print(f"p-value: {ate.p_value:.4f}")
print(f"Samples: {ate.n_samples}")
```

**Why DML over regression?** DML is *doubly robust*: if either the treatment model or the outcome model is correctly specified, the estimate is consistent. Simple regression assumes a known functional form and fails with complex confounding.

### Conditional Average Treatment Effect (CATE)

CATE estimates *heterogeneous* effects — how the treatment effect varies by segment:

```python
cate = estimator.estimate_cate(
    data=df,
    treatment_col="fast_pacing",
    outcome_col="ctr",
    confounder_cols=["platform", "daily_budget_usd"],
    segment_col="audience_segment",
)

for segment, effect in cate.segments.items():
    print(f"  {segment}: {effect.effect:+.4f} (p={effect.p_value:.4f})")
# Gen-Z 18-24:    +0.032 (p=0.001) — strong positive effect
# Millennials:     +0.015 (p=0.023) — moderate positive effect
# Gen-X 35-44:    +0.003 (p=0.412) — no significant effect
```

This is what makes OmniProof actionable: the same treatment can help one segment and hurt another.

## Step 4: Refutation

Estimates are only useful if they're robust. OmniProof runs three standard robustness tests:

### Placebo Test

Randomly shuffle the treatment column and re-estimate. If the original effect was real, the placebo effect should be near zero.

```python
from omni_proof.causal.refuter import CausalRefuter

refuter = CausalRefuter()
placebo = refuter.placebo_test(df, "fast_pacing", "ctr", confounders)
print(f"Original: {placebo.original_effect:+.4f}")
print(f"Placebo:  {placebo.new_effect:+.4f}")
print(f"Passed:   {placebo.passed}")   # True if placebo ≈ 0
```

### Subset Test

Drop 10% of observations and re-estimate. A real effect should be stable:

```python
subset = refuter.subset_test(df, "fast_pacing", "ctr", confounders, drop_fraction=0.1)
print(f"Passed: {subset.passed}")  # True if ATE is stable
```

### Random Confounder Test

Add a random noise variable as an additional confounder. If the estimate was already unconfounded, the noise should change nothing:

```python
random = refuter.random_confounder_test(df, "fast_pacing", "ctr", confounders)
print(f"Passed: {random.passed}")  # True if ATE unchanged
```

## Step 5: DICE-DML (Visual Embeddings)

Standard DML works on tabular features. But creative performance is driven by *visual* properties that live in embedding space. DICE-DML (Disentangled Causal Embeddings via DML) extends causal inference to visual embeddings.

### The Problem

A 3072-dim embedding captures everything about a creative: colors, composition, pacing, branding, *and* the treatment. To estimate the causal effect of, say, "fast pacing," we need to disentangle the pacing signal from everything else.

### The Solution

**1. Generate counterfactual pairs**

Create two versions of the same creative that differ only in the treatment (e.g., fast vs. slow pacing):

```python
from omni_proof.causal.dice_dml.counterfactual_generator import CounterfactualGenerator

generator = CounterfactualGenerator(gemini_client)
pair = await generator.generate(
    original_path=Path("runner_sunrise_fast.mp4"),
    counterfactual_path=Path("runner_sunrise_slow.mp4"),
    treatment_attr="fast_pacing",
)
print(f"Background similarity: {pair.background_similarity:.4f}")  # Should be ~0.95+
```

**2. Extract treatment fingerprint**

The difference between the two embeddings isolates the pure treatment signal:

```python
from omni_proof.causal.dice_dml.disentangler import TreatmentDisentangler

disentangler = TreatmentDisentangler()
fingerprint = disentangler.extract_treatment_fingerprint(
    pair.original_emb,
    pair.counterfactual_emb,
)
# fingerprint is L2-normalized, shape (3072,)
```

**3. Project out treatment from all embeddings**

Orthogonal projection removes the treatment component, leaving confounder-only representations:

```python
clean_embeddings = disentangler.disentangle_batch(all_embeddings, fingerprint)
# clean_embeddings: (n, 3072) — treatment signal removed
```

**4. Estimate on disentangled space**

```python
from omni_proof.causal.dice_dml.visual_estimator import VisualDMLEstimator

visual_estimator = VisualDMLEstimator(cv=3)
ate = visual_estimator.estimate_visual_ate(
    embeddings=all_embeddings,
    treatment=treatment_array,
    outcome=outcome_array,
    treatment_fingerprint=fingerprint,
)
```

### Why This Works

- **Counterfactual pairs** ensure the fingerprint captures only the treatment change, not background variation
- **Orthogonal projection** is a linear operation that cleanly separates treatment from confounders in embedding space
- **DML on the projected space** gives an unbiased estimate because confounders in the embeddings no longer correlate with treatment

## Design Brief Synthesis

The `InsightSynthesizer` converts statistical results into natural-language recommendations:

```python
from omni_proof import InsightSynthesizer

synth = InsightSynthesizer(p_value_threshold=0.05, recommend_threshold=0.05)
brief = synth.synthesize(cate_result)

print(brief.finding)
# "fast_pacing has a significant positive effect on ctr"

print(brief.segment_breakdown)
# {"Gen-Z 18-24": "+3.2% — RECOMMENDED", "Gen-X 35-44": "+0.3% — NEUTRAL"}

print(brief.recommendation)
# "Apply fast_pacing for Gen-Z 18-24 and Millennials 25-34 segments"
```

**Classification rules:**

| Condition | Label |
|-----------|-------|
| p-value > threshold | NEUTRAL (insufficient evidence) |
| Effect > recommend_threshold | RECOMMENDED |
| Effect > 0 | CONSIDER |
| Effect <= 0 | AVOID |

## References

- Chernozhukov, V., et al. (2018). "Double/debiased machine learning for treatment and structural parameters." *The Econometrics Journal*, 21(1), C1-C68.
- Dowhy: An end-to-end library for causal inference. [github.com/py-why/dowhy](https://github.com/py-why/dowhy)
- EconML: A Python Package for ML-Based Heterogeneous Treatment Effects Estimation. [github.com/py-why/EconML](https://github.com/py-why/EconML)
