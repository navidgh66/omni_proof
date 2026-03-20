# REST API Reference

Base URL: `http://localhost:8000`

## Health Check

```
GET /health
```

**Response:**

```json
{"status": "ok"}
```

---

## Brand Extraction

### Extract Brand Profile

```
POST /api/v1/brand/extract
```

Extract brand identity from a set of creative assets.

**Request Body:**

```json
{
  "brand_name": "Velocity Sportswear",
  "asset_paths": [
    "examples/creatives/runner_sunrise.png",
    "examples/creatives/yoga_flow.png"
  ]
}
```

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `brand_name` | `string` | 1-200 chars | Brand name |
| `asset_paths` | `string[]` | No `..`, no unsafe chars | Paths to creative assets |

**Response:** `BrandProfile` object (see [API Reference](api-reference.md#brandprofile)).

---

### Get Brand Profile

```
GET /api/v1/brand/profile/{profile_id}
```

**Parameters:**

| Name | In | Type | Description |
|------|-----|------|-------------|
| `profile_id` | path | `string` | Profile ID returned from extract |

---

### Update Brand Profile

```
POST /api/v1/brand/update/{profile_id}
```

Update an existing profile with new assets. Returns the updated profile and any detected conflicts.

**Request Body:** Same as extract.

**Response:**

```json
{
  "profile": { ... },
  "conflicts": [
    {
      "dimension": "color",
      "existing_value": "#FF5733",
      "new_value": "#00FF00",
      "source_assets": ["new_asset.png"],
      "severity": "major"
    }
  ]
}
```

---

## Causal Analysis

### Analyze Treatment Effect

```
POST /api/v1/causal/analyze
```

**Request Body:**

```json
{
  "treatment": "fast_pacing",
  "outcome": "ctr",
  "confounders": ["platform", "audience_segment", "daily_budget_usd"]
}
```

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `treatment` | `string` | `^[a-zA-Z_][a-zA-Z0-9_]*$` | Treatment column name |
| `outcome` | `string` | `^[a-zA-Z_][a-zA-Z0-9_]*$` | Outcome column name |
| `confounders` | `string[]` | — | Confounder column names |

**Response:**

```json
{
  "ate": {
    "treatment": "fast_pacing",
    "outcome": "ctr",
    "ate": 0.0189,
    "ci_lower": 0.0102,
    "ci_upper": 0.0276,
    "p_value": 0.0001,
    "n_samples": 1000
  }
}
```

---

### List Effects

```
GET /api/v1/causal/effects
```

Returns all previously computed causal effects.

---

### Get Effect by Treatment

```
GET /api/v1/causal/effects/{treatment_name}
```

---

## Compliance

### Check Compliance

```
POST /api/v1/compliance/check
Content-Type: multipart/form-data
```

Upload a creative asset to check against brand guidelines.

| Field | Type | Description |
|-------|------|-------------|
| `file` | `file` | Creative asset (image, video, PDF) |

**Response:**

```json
{
  "asset_id": "uploaded-001",
  "passed": false,
  "violations": [
    {
      "rule_type": "concrete",
      "severity": "critical",
      "description": "Logo clear space below minimum 20px requirement",
      "evidence": "Detected 8px clear space around logo"
    },
    {
      "rule_type": "semantic",
      "severity": "warning",
      "description": "Tone deviates from brand voice guidelines",
      "evidence": "Detected 'casual' tone, expected 'energetic'"
    }
  ],
  "evidence_sources": ["brand_guide_page_3", "brand_guide_page_7"],
  "score": 0.65
}
```

Filenames are sanitized: path components and unsafe characters are stripped.

---

### List Compliance Reports

```
GET /api/v1/compliance/reports
GET /api/v1/compliance/reports?campaign_id=camp-001
```

---

## Insights

### List Design Briefs

```
GET /api/v1/insights/briefs
```

**Response:**

```json
[
  {
    "treatment": "fast_pacing",
    "finding": "fast_pacing has a significant positive effect on ctr",
    "segment_breakdown": {
      "Gen-Z 18-24": "+3.2% — RECOMMENDED",
      "Millennials 25-34": "+1.5% — CONSIDER",
      "Gen-X 35-44": "+0.3% — NEUTRAL"
    },
    "recommendation": "Apply fast_pacing for Gen-Z 18-24 and Millennials 25-34",
    "confidence": "HIGH"
  }
]
```

---

### Get Segment Details

```
GET /api/v1/insights/segments
GET /api/v1/insights/segments?segment=Gen-Z+18-24
```

---

## Generative

### Generate Creative Prompt

```
POST /api/v1/generative/prompt
```

**Request Body:**

```json
{
  "target_segment": "Gen-Z 18-24",
  "objective": "engagement",
  "constraints": ["15-second format", "Instagram Reels", "include product close-up"]
}
```

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `target_segment` | `string` | 1-200 chars | Target audience segment |
| `objective` | `string` | 1-200 chars | Campaign objective |
| `constraints` | `string[]` | Max 20 items | Creative constraints |

**Response:**

```json
{
  "prompt": "Create a 15-second Instagram Reel targeting Gen-Z 18-24...\n\nCAUSAL INSIGHTS:\n- Use fast pacing (proven +3.2% CTR for this segment)\n- Use warm color palette (proven +0.6% CTR overall)\n\nBRAND RULES:\n- Primary colors: #FF5733, #1A1A2E\n- Typography: Montserrat Bold for headlines\n- Tone: Energetic, action-oriented\n\nCONSTRAINTS:\n- 15-second format\n- Include product close-up\n..."
}
```

---

## Error Handling

All errors return structured JSON. Stack traces are never exposed.

```json
{
  "detail": "Treatment column 'invalid!name' contains disallowed characters"
}
```

**Status Codes:**

| Code | Meaning |
|------|---------|
| `200` | Success |
| `400` | Validation error (bad input) |
| `404` | Resource not found |
| `422` | Unprocessable entity (Pydantic validation failure) |
| `500` | Internal server error (no stack trace) |

## CORS

Allowed origins (configurable via `OMNI_PROOF_CORS_ALLOWED_ORIGINS`):

- `http://localhost:3000`
- `http://localhost:8000`

Allowed methods: `GET`, `POST` only.
