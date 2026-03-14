# Spec: Layer 5 — Dashboard API & Generative Loop

## Overview
FastAPI serves causal insights via REST endpoints. The "NextAds" generative loop feeds proven causal parameters back into content generation.

## API Endpoints

### Causal Effects
```
GET  /api/v1/causal/effects
  -> list of all estimated treatment effects with confidence levels

GET  /api/v1/causal/effects/{treatment_name}
  -> detailed CATE breakdown by segment for specific treatment

POST /api/v1/causal/analyze
  Body: { treatment: str, outcome: str, confounders: list[str] }
  -> triggers new causal analysis, returns job_id

GET  /api/v1/causal/jobs/{job_id}
  -> analysis status and results when complete
```

### Brand Compliance
```
POST /api/v1/compliance/check
  Body: multipart/form-data (creative file)
  -> ComplianceResult (pass/fail, violations, evidence)

GET  /api/v1/compliance/reports?campaign_id=X
  -> historical compliance reports for campaign
```

### Insights
```
GET  /api/v1/insights/briefs
  -> latest design briefs generated from causal data

GET  /api/v1/insights/segments?segment=18-24
  -> all effects filtered by audience segment
```

### Generative Loop
```
POST /api/v1/generative/prompt
  Body: { target_segment: str, objective: str, constraints: list[str] }
  -> optimized creative prompt parameterized by causal data + brand rules
```

## Generative Prompt Builder

### Logic
1. Query top CATE results for target segment (effects with p < 0.05, refutation passed)
2. Retrieve active brand rules from RAG
3. Combine into parameterized prompt

### Example Output
```
Generate a 16:9 lifestyle image with the following specifications:
- Visual tone: warm, contemporary (per Brand Guide v3.2, Section 4.1)
- Product placement: upper-right quadrant (CATE: +8% engagement for 25-34)
- Opening pacing: fast cuts, < 1.5s per scene (CATE: +12% CTR for 18-24)
- Color palette: #FF6B35, #004E89, #FFFFFF (per Brand Guide v3.2, Section 2.3)
- Logo: visible within first 2 seconds, minimum 15% screen area
- CTA style: urgency-based ("Limited time") (CATE: +6% conversion for 18-34)
- Audio: upbeat, 120+ BPM, conversational voiceover
```

## Dashboard Data Models

```python
class CausalEffectSummary(BaseModel):
    treatment: str
    outcome: str
    ate: float
    ci_lower: float
    ci_upper: float
    p_value: float
    refutation_passed: bool
    segments: dict[str, SegmentEffect]

class SegmentEffect(BaseModel):
    segment: str
    effect: float
    ci_lower: float
    ci_upper: float
    p_value: float
    recommendation: Literal["recommended", "consider", "neutral", "avoid"]
```
