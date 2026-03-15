<p align="center">
  <img src="assets/logo.png" width="320" alt="OmniProof" />
</p>

<p align="center">
  <strong>From correlation to causation.<br>Upload creatives, discover <em>why</em> they perform.</strong>
</p>

<p align="center">
  <a href="https://github.com/navidgh66/omni_proof/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/navidgh66/omni_proof/ci.yml?branch=main&style=for-the-badge&label=CI" alt="CI"></a>
  <a href="https://pypi.org/project/omni-proof/"><img src="https://img.shields.io/pypi/v/omni-proof?style=for-the-badge&color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/omni-proof/"><img src="https://img.shields.io/pypi/pyversions/omni-proof?style=for-the-badge" alt="Python"></a>
  <a href="https://github.com/navidgh66/omni_proof/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="MIT License"></a>
</p>

<p align="center">
  <a href="#installation">Installation</a> &middot;
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#how-it-works">How It Works</a> &middot;
  <a href="#hands-on-playground">Playground</a> &middot;
  <a href="#features">Features</a> &middot;
  <a href="#api-reference">API</a> &middot;
  <a href="#contributing">Contributing</a>
</p>

---

**OmniProof** is an open-source Python engine that answers **why** creative assets perform differently. It replaces gut-feel marketing analytics with rigorous causal inference -- moving from *"ads with blue backgrounds got more clicks"* to *"blue backgrounds **cause** a +12% CTR uplift for the 18-24 segment, controlling for platform, budget, and seasonality."*

It combines **Gemini Embedding 2** for native multimodal understanding, **Double Machine Learning** for causal estimation, and **RAG-based brand compliance** into a single, modular pipeline.

## Highlights

- **[Causal Engine](https://github.com/navidgh66/omni_proof#causal-methodology)** -- DML + refutation tests isolate true treatment effects from confounders. Not correlations.
- **[Multimodal Embeddings](https://github.com/navidgh66/omni_proof#gemini-embedding-2)** -- Gemini Embedding 2 maps video, images, audio, PDFs, and text into a shared 3072-dim space.
- **[Brand Intelligence](https://github.com/navidgh66/omni_proof#advanced-usage)** -- Extract structured brand guidelines from any asset, then auto-check new creatives for compliance.
- **[DICE-DML](https://github.com/navidgh66/omni_proof#causal-methodology)** -- Disentangle visual confounders from treatment signals using counterfactual embedding pairs.
- **[Creative Generation](https://github.com/navidgh66/omni_proof#advanced-usage)** -- Causal insights feed directly into optimized creative prompts.
- **[REST API](https://github.com/navidgh66/omni_proof#api-reference)** -- 12 endpoints covering brand extraction, compliance, causal analysis, and generation.
- **[Modular](https://github.com/navidgh66/omni_proof#architecture)** -- Use the full pipeline or any layer independently as a library.

## Installation

```bash
pip install omni-proof
```

Or install from source:

```bash
git clone https://github.com/navidgh66/omni_proof.git
cd omni_proof
pip install -e ".[dev]"
```

> Requires Python 3.11+. For the full pipeline you'll need a [Gemini API key](https://aistudio.google.com/apikey) and a [Pinecone](https://app.pinecone.io/) account. The causal analysis layer works with local data only -- no API keys needed.

## Quick Start

```python
import pandas as pd
from pathlib import Path
from omni_proof import BrandExtractor, ComplianceChain, DMLEstimator, GeminiClient, Settings
from omni_proof.storage.memory_store import InMemoryVectorStore
from omni_proof.rag.brand_retriever import BrandRetriever

settings = Settings(gemini_api_key="AIza...", pinecone_api_key="pcsk_...",
                    pinecone_index_host="https://my-index.svc.pinecone.io")
client = GeminiClient(api_key=settings.gemini_api_key)
store = InMemoryVectorStore()

# 1. Extract brand identity from assets
extractor = BrandExtractor(embedding_provider=client, gemini_client=client, vector_store=store)
profile = await extractor.extract("AcmeCorp", [Path("brand_guide.pdf"), Path("logo.png")])

# 2. Check a new creative for brand compliance
retriever = BrandRetriever(gemini_client=client, vector_store=store)
chain = ComplianceChain(gemini_client=client, brand_retriever=retriever)
report = await chain.check_compliance("ad_001", Path("new_ad.jpg"))
print(f"Compliant: {report.passed} (score: {report.score})")

# 3. Estimate causal effect of a creative feature (no API keys needed)
data = pd.read_csv("campaign_data.csv")
estimator = DMLEstimator(cv=5, n_estimators=50)
ate = estimator.estimate_ate(data, "fast_pacing", "ctr", ["platform", "audience_segment", "budget"])
print(f"ATE: {ate.ate:+.3f} (p={ate.p_value:.4f})")
```

Or start the API server:

```bash
uvicorn omni_proof.api.app:create_app --factory --reload
curl localhost:8000/health  # {"status": "ok"}
```

## How It Works

```
  Upload creatives (video, image, PDF, audio)
          |
          v
  +---------------------------+
  |  Gemini Embedding 2       |  3072-dim multimodal embeddings
  |  Gemini 3.1 Flash Lite         |  Structured feature extraction
  +------------+--------------+
               |
        +------+------+
        v             v
    Pinecone       SQL DB
    (vectors)      (metadata + outcomes)
        |             |
        +------+------+
               v
  +---------------------------+
  |  Causal Engine             |  DAG -> Identify -> DML -> Refute
  |  (DoWhy + EconML)          |  DICE-DML for visual embeddings
  +------------+--------------+
               |
        +------+-----------+
        v                  v
  Brand Compliance     Creative Generation
  (RAG retrieval)      (causal-informed prompts)
```

## Configuration

Set environment variables with the `OMNI_PROOF_` prefix, or pass them programmatically:

```bash
OMNI_PROOF_GEMINI_API_KEY=AIza...
OMNI_PROOF_PINECONE_API_KEY=pcsk_...
OMNI_PROOF_PINECONE_INDEX_HOST=https://my-index-abc123.svc.pinecone.io
OMNI_PROOF_DATABASE_URL=sqlite+aiosqlite:///./omni_proof.db  # default
```

| Variable | Required For | Where to Get |
|:---------|:-------------|:-------------|
| `OMNI_PROOF_GEMINI_API_KEY` | Embeddings + extraction | [Google AI Studio](https://aistudio.google.com/apikey) |
| `OMNI_PROOF_PINECONE_API_KEY` | Vector storage | [Pinecone Console](https://app.pinecone.io/) |
| `OMNI_PROOF_PINECONE_INDEX_HOST` | Vector storage | Pinecone Console |
| `OMNI_PROOF_DATABASE_URL` | Relational storage | PostgreSQL or SQLite URI |

## Architecture

OmniProof is organized into five layers. Each can be used independently:

| Layer | Module | Key Classes |
|:------|:-------|:------------|
| **Ingestion** | `omni_proof.ingestion` | `GeminiClient`, `AssetPreprocessor`, `IngestPipeline` |
| **Storage** | `omni_proof.storage` | `PineconeVectorStore`, `InMemoryVectorStore`, `RelationalStore` |
| **Causal** | `omni_proof.causal` | `CausalDAGBuilder`, `DMLEstimator`, `CausalRefuter`, `VisualDMLEstimator` |
| **Orchestration** | `omni_proof.orchestration` | `ComplianceChain`, `InsightSynthesizer`, `BrandExtractor` |
| **API** | `omni_proof.api` | FastAPI app, routes, `GenerativePromptBuilder` |

### Key Abstractions

| Interface | Purpose | Implementations |
|:----------|:--------|:----------------|
| `EmbeddingProvider` | Generate embeddings from any content | `GeminiClient` |
| `VectorStore` | Store and search vectors | `PineconeVectorStore`, `InMemoryVectorStore` |
| `Estimator` | Estimate causal effects | `DMLEstimator` |

## Advanced Usage

<details>
<summary><strong>Causal pipeline with DAG + refutation</strong></summary>

```python
from omni_proof.causal.dag_builder import CausalDAGBuilder
from omni_proof.causal.identifier import CausalIdentifier
from omni_proof.causal.refuter import CausalRefuter

dag = CausalDAGBuilder()
model = dag.build_dag(data, treatment="fast_pacing", outcome="ctr",
                      confounders=["platform", "audience_segment", "budget"])

estimand = CausalIdentifier().identify_effect(model)
ate = DMLEstimator().estimate_ate(data, "fast_pacing", "ctr",
                                  ["platform", "audience_segment", "budget"])

refuter = CausalRefuter()
placebo = refuter.placebo_test(data, "fast_pacing", "ctr",
                               ["platform", "audience_segment", "budget"])
print(f"Effect: {ate.ate:+.3f}, Placebo passed: {placebo.passed}")
```
</details>

<details>
<summary><strong>Brand extraction with conflict detection</strong></summary>

```python
profile = await extractor.extract("AcmeCorp", [
    Path("brand_guide.pdf"), Path("approved_ad.jpg"),
    Path("brand_video.mp4"), Path("jingle.mp3"),
])
print(f"Colors: {profile.visual_style.dominant_colors}")
print(f"Voice: {profile.voice.formality}, {profile.voice.emotional_register}")
print(f"Rules: {len(profile.rules)}")

# Update with new assets -- detects conflicts
updated, conflicts = await extractor.update(profile, [Path("new_campaign.jpg")])
for c in conflicts:
    print(f"  CONFLICT [{c.severity}] {c.dimension}: {c.existing_value} -> {c.new_value}")
```
</details>

<details>
<summary><strong>Creative generation from causal insights</strong></summary>

```python
from omni_proof.api.generative_loop import GenerativePromptBuilder

builder = GenerativePromptBuilder()
prompt = builder.build_prompt(
    cate_insights=[{"treatment": "fast_pacing", "effect": 0.12}],
    brand_rules=[{"description": "Use blue (#004E89) as primary color"}],
    target_segment="18-24",
    objective="conversion",
    constraints=["16:9 aspect ratio", "max 15 seconds"],
)
```
</details>

<details>
<summary><strong>CATE by segment + insight synthesis</strong></summary>

```python
cate = estimator.estimate_cate(data, "fast_pacing", "ctr",
                                confounder_cols=["platform", "budget"],
                                segment_col="audience_segment")
for segment, effect in cate.segments.items():
    print(f"  {segment}: {effect.effect:+.3f} (CI: {effect.ci_lower:.3f} to {effect.ci_upper:.3f})")

from omni_proof import InsightSynthesizer
brief = InsightSynthesizer(p_value_threshold=0.05).synthesize(cate)
print(f"{brief.finding} -> {brief.recommendation}")
```
</details>

## Hands-On Playground

The `examples/` directory ships with a complete dataset for **Velocity Sportswear** — a fictional DTC activewear brand with 10 image creatives, 4 A/B video variants, and 1,000 rows of campaign performance data. Everything is wired together so you can exercise every OmniProof module end-to-end.

### Run the offline demo (no API keys)

```bash
python examples/demo.py
```

This runs all 9 stages — DAG construction, DML estimation, CATE by segment, refutation, design briefs, brand profile loading, prompt generation, and compliance review — using only local data in ~4 seconds.

### What's in `examples/`

```
examples/
  creatives/
    runner_sunrise_*.png          # 10 image creatives (one per concept)
    trail_epic_*.png
    hiit_studio_*.png
    ...
    runner_sunrise_fast_pacing_A.mp4   # A/B video variants
    runner_sunrise_slow_pacing_B.mp4   #   (fast_pacing treatment)
    basketball_court_fast_pacing_A.mp4
    basketball_court_slow_pacing_B.mp4
  data/
    campaign_performance.csv      # 1,000 rows with planted causal effects
    brand_profile.json            # Velocity Sportswear brand profile
    brand_guidelines.json         # 12 brand rules for RAG
    compliance_samples.json       # 5 compliance reports (PASS/WARN/FAIL)
    creative_metadata_samples.json # 14 records (10 images + 4 videos)
```

### With API keys: full pipeline

Set your keys and explore each capability interactively:

```bash
export OMNI_PROOF_GEMINI_API_KEY=AIza...
export OMNI_PROOF_PINECONE_API_KEY=pcsk_...
export OMNI_PROOF_PINECONE_INDEX_HOST=https://your-index.svc.pinecone.io
```

<details>
<summary><strong>1. Generate embeddings for the example creatives</strong></summary>

```python
import asyncio
from pathlib import Path
from omni_proof.ingestion.gemini_client import GeminiClient
from omni_proof.storage.memory_store import InMemoryVectorStore

client = GeminiClient(api_key="AIza...")
store = InMemoryVectorStore()

async def embed_creatives():
    creative_dir = Path("examples/creatives")
    for img in sorted(creative_dir.glob("*.png")):
        embedding = await client.generate_embedding(img)
        await store.upsert(
            asset_id=img.stem,
            embedding=embedding,
            metadata={"file": img.name, "type": "image"},
            namespace="creatives",
        )
        print(f"  Embedded {img.name} -> {len(embedding)} dims")

    # Embed video variants too
    for vid in sorted(creative_dir.glob("*.mp4")):
        embedding = await client.generate_embedding(vid)
        await store.upsert(
            asset_id=vid.stem,
            embedding=embedding,
            metadata={"file": vid.name, "type": "video"},
            namespace="creatives",
        )
        print(f"  Embedded {vid.name} -> {len(embedding)} dims")

asyncio.run(embed_creatives())
```
</details>

<details>
<summary><strong>2. Extract structured metadata from a creative</strong></summary>

```python
import asyncio
from pathlib import Path
from omni_proof.ingestion.gemini_client import GeminiClient
from omni_proof.ingestion.pipeline import IngestPipeline
from omni_proof.ingestion.schemas import CreativeMetadata

client = GeminiClient(api_key="AIza...")
pipeline = IngestPipeline(gemini_client=client)

async def extract():
    asset = Path("examples/creatives/runner_sunrise_1773607608677.png")
    metadata, embedding = await pipeline.ingest(asset, CreativeMetadata)
    print(f"Objects: {metadata.visual.objects_detected}")
    print(f"CTA: {metadata.textual.cta_type} — '{metadata.textual.promotional_text}'")
    print(f"Embedding: {len(embedding)} dims")

asyncio.run(extract())
```
</details>

<details>
<summary><strong>3. Extract brand identity from the example creatives</strong></summary>

```python
import asyncio
from pathlib import Path
from omni_proof.brand_extraction.extractor import BrandExtractor
from omni_proof.ingestion.gemini_client import GeminiClient
from omni_proof.storage.memory_store import InMemoryVectorStore

client = GeminiClient(api_key="AIza...")
store = InMemoryVectorStore()
extractor = BrandExtractor(embedding_provider=client, gemini_client=client, vector_store=store)

async def extract_brand():
    assets = sorted(Path("examples/creatives").glob("*.png"))
    profile = await extractor.extract("Velocity Sportswear", assets)
    print(f"Brand: {profile.brand_name}")
    print(f"Colors: {profile.visual_style.dominant_colors}")
    print(f"Voice: {profile.voice.formality}, {profile.voice.emotional_register}")
    print(f"Rules extracted: {len(profile.rules)}")
    for rule in profile.rules:
        print(f"  [{rule.section_type}] {rule.description[:80]}...")

asyncio.run(extract_brand())
```
</details>

<details>
<summary><strong>4. Check a creative for brand compliance (RAG)</strong></summary>

```python
import asyncio
from pathlib import Path
from omni_proof.ingestion.gemini_client import GeminiClient
from omni_proof.storage.memory_store import InMemoryVectorStore
from omni_proof.rag.brand_retriever import BrandRetriever
from omni_proof.orchestration.compliance_chain import ComplianceChain

client = GeminiClient(api_key="AIza...")
store = InMemoryVectorStore()

async def check_compliance():
    # First, index the brand guidelines as embeddings
    import json
    with open("examples/data/brand_guidelines.json") as f:
        guidelines = json.load(f)
    for g in guidelines:
        emb = await client.generate_embedding(g["description"], task_type="RETRIEVAL_DOCUMENT")
        await store.upsert(
            asset_id=g["guideline_id"],
            embedding=emb,
            metadata={"source_type": "guideline", "section": g["section"]},
            namespace="brand_assets",
        )
    print(f"Indexed {len(guidelines)} brand guidelines")

    # Now check a creative
    retriever = BrandRetriever(gemini_client=client, vector_store=store)
    chain = ComplianceChain(gemini_client=client, brand_retriever=retriever)
    report = await chain.check_compliance(
        "VEL-2026-Q1-0012",
        Path("examples/creatives/runner_sunrise_1773607608677.png"),
    )
    print(f"Passed: {report.passed} (score: {report.score})")
    print(f"Guidelines retrieved: {report.evidence_sources}")

asyncio.run(check_compliance())
```
</details>

<details>
<summary><strong>5. Run causal analysis on the campaign data</strong></summary>

```python
import pandas as pd
from omni_proof.causal.dag_builder import CausalDAGBuilder
from omni_proof.causal.identifier import CausalIdentifier
from omni_proof.causal.estimator import DMLEstimator
from omni_proof.causal.refuter import CausalRefuter

data = pd.read_csv("examples/data/campaign_performance.csv")

# Encode categoricals
for col in ["platform", "audience_segment", "region", "quarter"]:
    data[col] = data[col].astype("category").cat.codes.astype(float)

confounders = ["platform", "audience_segment", "daily_budget_usd", "region", "quarter"]

# Build DAG + identify
dag = CausalDAGBuilder()
model = dag.build_dag(data, treatment="fast_pacing", outcome="ctr", confounders=confounders)
CausalIdentifier().identify_effect(model)

# Estimate ATE
estimator = DMLEstimator(cv=3, n_estimators=30)
ate = estimator.estimate_ate(data, "fast_pacing", "ctr", confounders)
print(f"ATE: {ate.ate*100:+.1f}pp (p={ate.p_value:.4f})")

# CATE by segment
cate = estimator.estimate_cate(data, "fast_pacing", "ctr", confounders, "audience_segment")
for seg, eff in cate.segments.items():
    print(f"  Segment {seg}: {eff.effect*100:+.1f}pp")

# Refutation
refuter = CausalRefuter(cv=3)
placebo = refuter.placebo_test(data, "fast_pacing", "ctr", confounders)
subset = refuter.subset_test(data, "fast_pacing", "ctr", confounders)
print(f"Placebo: {'PASS' if placebo.passed else 'FAIL'}")
print(f"Subset:  {'PASS' if subset.passed else 'FAIL'}")
```
</details>

<details>
<summary><strong>6. Synthesize design brief + generate creative prompt</strong></summary>

```python
from omni_proof.orchestration.insight_synthesizer import InsightSynthesizer
from omni_proof.api.generative_loop import GenerativePromptBuilder

# Using the cate result from step 5
brief = InsightSynthesizer().synthesize(cate)
print(f"Finding: {brief.finding}")
print(f"Recommendation: {brief.recommendation}")
print(f"Confidence: {brief.confidence}")

# Generate a creative prompt combining causal insights + brand rules
import json
with open("examples/data/brand_profile.json") as f:
    profile = json.load(f)

builder = GenerativePromptBuilder()
prompt = builder.build_prompt(
    cate_insights=[{"treatment": "fast_pacing", "effect": 0.025}],
    brand_rules=[{"description": r["description"]} for r in profile["rules"]],
    target_segment="Gen-Z (18-24)",
    objective="conversion",
    constraints=["9:16 vertical", "max 15 seconds", "product reveal by second 5"],
)
print(prompt)
```
</details>

<details>
<summary><strong>7. Compare A/B video embeddings (DICE-DML prep)</strong></summary>

```python
import asyncio
import numpy as np
from pathlib import Path
from omni_proof.ingestion.gemini_client import GeminiClient

client = GeminiClient(api_key="AIza...")

async def compare_ab():
    creatives = Path("examples/creatives")

    # Embed both variants
    emb_a = await client.generate_embedding(creatives / "runner_sunrise_fast_pacing_A.mp4")
    emb_b = await client.generate_embedding(creatives / "runner_sunrise_slow_pacing_B.mp4")

    a, b = np.array(emb_a), np.array(emb_b)
    cosine_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    treatment_fingerprint = a - b  # isolates the fast_pacing signal

    print(f"Variant A dims: {len(emb_a)}")
    print(f"Variant B dims: {len(emb_b)}")
    print(f"Cosine similarity: {cosine_sim:.4f}")
    print(f"Treatment fingerprint norm: {np.linalg.norm(treatment_fingerprint):.4f}")
    print("(This fingerprint feeds into VisualDMLEstimator for DICE-DML)")

asyncio.run(compare_ab())
```
</details>

<details>
<summary><strong>8. Start the API server with example data</strong></summary>

```bash
# Start the server
uvicorn omni_proof.api.app:create_app --factory --reload

# Health check
curl localhost:8000/health

# Trigger causal analysis
curl -X POST localhost:8000/api/v1/causal/analyze \
  -H "Content-Type: application/json" \
  -d '{"treatment": "fast_pacing", "outcome": "ctr",
       "confounders": ["platform", "audience_segment", "daily_budget_usd"]}'

# Generate a creative prompt
curl -X POST localhost:8000/api/v1/generative/prompt \
  -H "Content-Type: application/json" \
  -d '{"target_segment": "Gen-Z (18-24)", "objective": "conversion"}'
```
</details>

## API Reference

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/brand/extract` | Extract brand profile from uploaded assets |
| `POST` | `/api/v1/brand/update/{id}` | Update brand profile with new assets |
| `GET` | `/api/v1/brand/profile/{id}` | Retrieve a brand profile |
| `POST` | `/api/v1/compliance/check` | Check creative for brand compliance |
| `GET` | `/api/v1/compliance/reports` | Historical compliance reports |
| `POST` | `/api/v1/causal/analyze` | Trigger causal analysis |
| `GET` | `/api/v1/causal/effects` | List estimated causal effects |
| `GET` | `/api/v1/causal/effects/{treatment}` | CATE breakdown by segment |
| `GET` | `/api/v1/insights/briefs` | Design briefs from causal data |
| `GET` | `/api/v1/insights/segments` | Effects by audience segment |
| `POST` | `/api/v1/generative/prompt` | Generate optimized creative prompt |

## Causal Methodology

OmniProof implements a four-stage causal pipeline:

1. **Model** -- Build a DAG mapping treatments, outcomes, and confounders
2. **Identify** -- Apply the backdoor criterion to find valid adjustment sets
3. **Estimate** -- Double Machine Learning (Neyman orthogonalization) via EconML
4. **Refute** -- Placebo tests, subset validation, and random confounder checks

For visual embeddings where treatment and confounders are entangled, **DICE-DML** generates counterfactual pairs, isolates treatment fingerprints via vector subtraction, and applies orthogonal projection before estimation.

## Gemini Embedding 2

All modalities map to the same 3072-dimensional semantic space via [Gemini Embedding 2](https://ai.google.dev/gemini-api/docs/embeddings):

| Modality | Limit |
|:---------|:------|
| Text | 8,192 tokens |
| Images | 6 per request |
| Video | 80s (with audio) / 120s (without) |
| Audio | 80s |
| PDF | 1 document, 6 pages |
| Output | 3,072 dims (Matryoshka: truncate to 1536 / 768 / 128) |

## Tech Stack

| Component | Technology |
|:----------|:-----------|
| Embeddings | Gemini Embedding 2 |
| Structured extraction | Gemini 3.1 Flash Lite |
| Vector DB | Pinecone Serverless |
| Relational DB | PostgreSQL / SQLite |
| Causal inference | DoWhy + EconML |
| API | FastAPI |
| ML models | LightGBM |
| Schemas | Pydantic v2 + SQLAlchemy 2.0 |

## Testing

```bash
pytest tests/unit/ -v               # 150 unit tests
pytest tests/integration/ -v        # 53 integration tests
pytest tests/ -v                    # All 203 tests
ruff check src/ tests/              # Lint
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=navidgh66/omni_proof&type=Date)](https://star-history.com/#navidgh66/omni_proof&Date)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and PR guidelines.

## License

[MIT](LICENSE) -- OmniProof Contributors
