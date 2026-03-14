# Spec: Layer 2 — Vector & Relational Storage

## Overview

Dual storage: **Pinecone** (serverless vector DB) for semantic search, relational DB (PostgreSQL) for structured performance data. Linked by `asset_id`.

## Vector Store (Pinecone)

### SDK Pattern

```python
from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion

pc = Pinecone(api_key="PINECONE_API_KEY")

# Create index (3072 dims to match Gemini Embedding 2 output)
pc.create_index(
    name="creative-embeddings",
    dimension=3072,
    metric="cosine",
    spec=ServerlessSpec(cloud=CloudProvider.AWS, region=AwsRegion.US_EAST_1),
)
index = pc.Index(host="creative-embeddings-xxx.svc.pinecone.io")
```

### Index Schema

```
Index: "creative-embeddings"
  - dimension: 3072 (Gemini Embedding 2 default)
  - metric: cosine
  - namespace: "creatives" | "brand_assets"
  - metadata per vector:
    - asset_id: str (UUID)
    - campaign_id: str
    - platform: str
    - timestamp: str (ISO 8601)
    - asset_type: str (video/image/audio/pdf)

Index: "brand-assets" (or namespace "brand_assets" in same index)
  - metadata per vector:
    - asset_id: str
    - source_type: str (guideline/approved_creative/palette)
    - section_type: str (logo_rules/color_palette/typography/tone)
    - page_number: int (for PDFs)
    - tags: list[str]
```

### Operations

```python
# Upsert — tuple format (id, values, metadata)
index.upsert(
    vectors=[
        (asset_id, embedding, {"campaign_id": "camp1", "platform": "youtube", "asset_type": "video"}),
    ],
    namespace="creatives",
)

# Query — with metadata filtering
results = index.query(
    vector=query_embedding,
    top_k=10,
    include_metadata=True,
    filter={"platform": {"$eq": "youtube"}, "campaign_id": {"$eq": "camp1"}},
    namespace="creatives",
)
for match in results.matches:
    print(match.id, match.score, match.metadata)

# Delete — by IDs
index.delete(ids=[asset_id], namespace="creatives")

# Batch upsert — for large datasets
index.upsert(vectors=large_list, namespace="creatives", batch_size=100, show_progress=True)
```

### Metadata Filter Operators

| Operator | Example | Description |
|:---------|:--------|:------------|
| `$eq` | `{"platform": {"$eq": "youtube"}}` | Exact match |
| `$in` | `{"platform": {"$in": ["youtube", "instagram"]}}` | Match any |
| `$gte` / `$lte` | `{"year": {"$gte": 2026}}` | Range |
| `$exists` | `{"tags": {"$exists": true}}` | Field exists |

## Relational Store (PostgreSQL)

### Tables

```sql
creative_metadata (
  asset_id UUID PRIMARY KEY,
  campaign_id UUID FK,
  -- Visual
  objects_detected JSONB,
  logo_screen_ratio FLOAT,
  background_setting VARCHAR,
  dominant_colors JSONB,
  contrast_ratio FLOAT,
  -- Temporal
  scene_transitions INT,
  time_to_first_logo FLOAT,
  product_exposure_seconds FLOAT,
  motion_intensity FLOAT,
  -- Textual
  text_density FLOAT,
  cta_type VARCHAR,
  promotional_text TEXT,
  typography_style VARCHAR,
  -- Auditory
  audio_genre VARCHAR,
  voiceover_tone VARCHAR,
  music_tempo_bpm INT,
  -- Meta
  created_at TIMESTAMP,
  platform VARCHAR
)

performance_records (
  id UUID PRIMARY KEY,
  asset_id UUID FK -> creative_metadata,
  impressions BIGINT,
  clicks BIGINT,
  conversions INT,
  roas FLOAT,
  ctr FLOAT,
  audience_segment VARCHAR,
  date DATE,
  platform VARCHAR
)

campaigns (
  campaign_id UUID PRIMARY KEY,
  name VARCHAR,
  start_date DATE,
  end_date DATE,
  budget DECIMAL,
  target_demographics JSONB
)
```

## Join Pattern

```sql
SELECT cm.*, pr.roas, pr.ctr, pr.audience_segment
FROM creative_metadata cm
JOIN performance_records pr ON cm.asset_id = pr.asset_id
WHERE pr.date BETWEEN :start AND :end
```

This produces the data matrix for causal modeling.
