# Spec: Layer 3 — Causal Discovery & Estimation Core

## Overview
Four-stage causal pipeline (Model -> Identify -> Estimate -> Refute) using DoWhy + EconML. Includes DICE-DML for visual embeddings.

## DAG Construction (DoWhy)

### Variables
- **Treatment (T):** Discrete creative feature under test (e.g., `logo_in_first_3s: bool`)
- **Outcome (Y):** Business metric (`roas`, `ctr`, `conversions`)
- **Confounders (W):** Variables affecting both T and Y — `season`, `audience_segment`, `platform`, `production_quality`, `budget`
- **Mediators (M):** Variables on the causal path T->M->Y (exclude from adjustment)
- **Colliders (C):** Variables caused by both T and Y (MUST exclude — causes M-bias)

### Example DAG
```
season ──> logo_placement ──> ctr
  |                            ^
  └────────────────────────────┘
platform ──> logo_placement
platform ──> ctr
audience ──> logo_placement
audience ──> ctr
```

## Double Machine Learning (EconML)

### Stage 1: Residualization
```python
# Propensity model: predict treatment from confounders
T_hat = LightGBM.fit(W).predict(W)
T_residual = T - T_hat

# Outcome model: predict outcome from confounders
Y_hat = LightGBM.fit(W).predict(W)
Y_residual = Y - Y_hat
```

### Stage 2: Orthogonalization
Residuals are now orthogonal to confounder space. Pure treatment/outcome variance remains.

### Stage 3: Estimation
```python
# ATE: Average Treatment Effect
ate = LinearDML(model_y=LightGBM(), model_t=LightGBM(), cv=5)
ate.fit(Y, T, X=heterogeneity_features, W=confounders)

# CATE: Conditional Average Treatment Effect (per segment)
cate = CausalForestDML(model_y=LightGBM(), model_t=LightGBM(), cv=5)
cate.fit(Y, T, X=heterogeneity_features, W=confounders)
cate.effect(X_test)  # segment-level effects
```

## DICE-DML for Visual Causality

### Problem
Raw Gemini embeddings entangle treatment + confounders in one vector. Standard DML on these vectors produces biased estimates.

### Solution
1. **Generate counterfactual pairs:** Alter ONLY treatment attribute (e.g., CTA typography) via generative AI. Background identical.
2. **Extract treatment fingerprint:** `fingerprint = emb(original) - emb(counterfactual)`
3. **Orthogonal projection:** Project embeddings onto subspace orthogonal to fingerprint → pure confounder representation
4. **Feed disentangled vectors into DML pipeline**

### Expected Performance
- 73-97% RMSE reduction vs. naive embedding approach (per DICE-DML paper)

## Refutation Suite

| Test | Method | Pass Criterion |
|:---|:---|:---|
| Placebo | Randomize treatment column | Effect becomes non-significant (p > 0.05) |
| Subset | Drop 10% random data | Effect stays within original CI bounds |
| Random Confounder | Add noise variable as confounder | Effect magnitude unchanged |

If placebo test shows significant effect → flag insight as **SPURIOUS**.
