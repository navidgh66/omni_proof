<p align="center">
  <h1 align="center">OmniProof</h1>
  <p align="center">
    <strong>Causal-Multimodal Engine for Creative Performance Attribution</strong>
  </p>
  <p align="center">
    <a href="#installation">Installation</a> &middot;
    <a href="#quick-start">Quick Start</a> &middot;
    <a href="#usage-modes">Usage Modes</a> &middot;
    <a href="#api-reference">API Reference</a> &middot;
    <a href="#contributing">Contributing</a>
  </p>
</p>

---

OmniProof is an open-source engine that answers **why** creative assets perform differently. It moves marketing analytics from correlation ("ads with blue backgrounds got more clicks") to causation ("blue backgrounds *cause* a +12% CTR uplift for the 18-24 segment, controlling for platform, budget, and seasonality").

It combines **Google Gemini Embedding 2** (native multimodal embeddings), **Double Machine Learning** (causal inference), and **RAG-based brand compliance** into a modular Python library and API.

## Features

- **Causal Performance Attribution** -- Estimate true causal effects of creative features (pacing, colors, CTA type) on outcomes (CTR, conversions) using DML, not correlations
- **Multimodal Brand Extraction** -- Upload brand assets (PDFs, images, videos, audio) and automatically extract structured brand guidelines, visual identity, and voice profile
- **Brand Compliance Checking** -- Verify new creatives against extracted or existing brand guidelines via multimodal RAG
- **Creative Generation** -- Generate optimized creative prompts parameterized by proven causal insights and brand rules
- **DICE-DML Visual Causality** -- Disentangle visual confounders from treatment signals using counterfactual embedding pairs
- **Modular Architecture** -- Use any capability independently or combine them

## Installation

### From source

```bash
git clone https://github.com/navidgh66/omni_proof.git
cd omni_proof
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### Requirements

- Python 3.11+
- A [Google AI Studio](https://aistudio.google.com/apikey) API key (for Gemini Embedding 2 + Flash)
- A [Pinecone](https://app.pinecone.io/) account (for vector storage)
- PostgreSQL (production) or SQLite (development, default)

## Configuration

OmniProof uses environment variables with the `OMNI_PROOF_` prefix. Create a `.env` file in the project root:

```bash
# Required for embedding and metadata extraction
OMNI_PROOF_GEMINI_API_KEY=AIza...

# Required for vector storage
OMNI_PROOF_PINECONE_API_KEY=pcsk_...
OMNI_PROOF_PINECONE_INDEX_HOST=https://my-index-abc123.svc.pinecone.io

# Optional (defaults shown)
OMNI_PROOF_DATABASE_URL=sqlite+aiosqlite:///./omni_proof.db
OMNI_PROOF_EMBEDDING_DIMENSIONS=3072
OMNI_PROOF_LOG_LEVEL=INFO
OMNI_PROOF_CORS_ALLOWED_ORIGINS=["*"]
```

### API Keys

| Variable | Required For | Where to Get |
|:---------|:-------------|:-------------|
| `OMNI_PROOF_GEMINI_API_KEY` | Embeddings + metadata extraction | [Google AI Studio](https://aistudio.google.com/apikey) |
| `OMNI_PROOF_PINECONE_API_KEY` | Vector storage and retrieval | [Pinecone Console](https://app.pinecone.io/) |
| `OMNI_PROOF_PINECONE_INDEX_HOST` | Vector storage and retrieval | Pinecone Console (index details page) |
| `OMNI_PROOF_DATABASE_URL` | Relational storage | Your PostgreSQL or SQLite connection string |

> **Note:** The causal analysis layer (`DMLEstimator`, `CausalDAGBuilder`, `CausalRefuter`) requires **no API keys** -- it works entirely with local data using scikit-learn, EconML, and DoWhy.

### Programmatic configuration

When using OmniProof as a library, you can pass settings directly:

```python
from omni_proof import Settings, GeminiClient

settings = Settings(
    gemini_api_key="AIza...",
    pinecone_api_key="pcsk_...",
    pinecone_index_host="https://my-index-abc123.svc.pinecone.io",
)
client = GeminiClient(api_key=settings.gemini_api_key)
```

## Quick Start

### Start the API server

```bash
uvicorn omni_proof.api.app:create_app --factory --reload
```

```bash
curl localhost:8000/health
# {"status": "ok"}
```

### Using Docker

```bash
docker-compose up -d    # Starts API + PostgreSQL
curl localhost:8000/health
```

## Usage Modes

OmniProof is a modular toolkit. Each capability works independently:

| Mode | What It Does | API Keys Needed |
|:-----|:-------------|:----------------|
| **Performance Insights** | Causal analysis of creative features vs outcomes | None |
| **Brand Extraction** | Extract brand guidelines from uploaded assets | Gemini |
| **Brand Compliance** | Check creatives against brand guidelines | Gemini + Pinecone |
| **Creative Generation** | Generate optimized creative prompts | None (uses stored results) |
| **Full Pipeline** | All of the above combined | Gemini + Pinecone |

### Mode 1: Causal Performance Insights (no API keys needed)

Estimate the true causal effect of creative features on performance:

```python
import pandas as pd
from omni_proof import DMLEstimator

# Load your creative performance data
data = pd.read_csv("campaign_data.csv")

estimator = DMLEstimator(cv=5, n_estimators=50)

# Average Treatment Effect: does fast pacing cause higher CTR?
ate = estimator.estimate_ate(
    data,
    treatment_col="fast_pacing",
    outcome_col="ctr",
    confounder_cols=["platform", "audience_segment", "budget"],
)
print(f"ATE: {ate.ate:+.3f} (CI: {ate.ci_lower:.3f} to {ate.ci_upper:.3f}, p={ate.p_value:.4f})")

# Conditional ATE: how does the effect vary by segment?
cate = estimator.estimate_cate(
    data,
    treatment_col="fast_pacing",
    outcome_col="ctr",
    confounder_cols=["platform", "budget"],
    segment_col="audience_segment",
)
for segment, effect in cate.segments.items():
    print(f"  {segment}: {effect.effect:+.3f} (CI: {effect.ci_lower:.3f} to {effect.ci_upper:.3f})")
```

**Full causal pipeline with DAG construction and refutation:**

```python
from omni_proof.causal.dag_builder import CausalDAGBuilder
from omni_proof.causal.identifier import CausalIdentifier
from omni_proof.causal.refuter import CausalRefuter

# 1. Build causal DAG
dag = CausalDAGBuilder()
model = dag.build_dag(data, treatment="fast_pacing", outcome="ctr",
                      confounders=["platform", "audience_segment", "budget"])

# 2. Identify estimand via backdoor criterion
identifier = CausalIdentifier()
estimand = identifier.identify_effect(model)

# 3. Estimate effect
estimator = DMLEstimator()
ate = estimator.estimate_ate(data, "fast_pacing", "ctr", ["platform", "audience_segment", "budget"])

# 4. Refute -- reject spurious findings
refuter = CausalRefuter()
placebo = refuter.placebo_test(data, "fast_pacing", "ctr", ["platform", "audience_segment", "budget"])
subset = refuter.subset_test(data, "fast_pacing", "ctr", ["platform", "audience_segment", "budget"])
print(f"Placebo passed: {placebo.passed}, Subset passed: {subset.passed}")
```

### Mode 2: Brand Extraction (requires Gemini key)

Extract structured brand guidelines from a collection of assets:

```python
from pathlib import Path
from omni_proof import BrandExtractor, GeminiClient, Settings
from omni_proof.storage.memory_store import InMemoryVectorStore

settings = Settings(gemini_api_key="AIza...")
client = GeminiClient(api_key=settings.gemini_api_key)
store = InMemoryVectorStore()

extractor = BrandExtractor(
    embedding_provider=client,
    gemini_client=client,
    vector_store=store,
)

# Extract brand profile from assets
profile = await extractor.extract("AcmeCorp", [
    Path("brand_guide.pdf"),
    Path("approved_ad_1.jpg"),
    Path("approved_ad_2.mp4"),
    Path("brand_jingle.mp3"),
])

print(f"Brand: {profile.brand_name}")
print(f"Voice: {profile.voice.formality}, {profile.voice.emotional_register}")
print(f"Colors: {profile.visual_style.dominant_colors}")
print(f"Rules extracted: {len(profile.rules)}")
for rule in profile.rules:
    print(f"  [{rule.section_type}] {rule.description}")

# Later, update with new assets (detects conflicts)
updated, conflicts = await extractor.update(profile, [Path("new_campaign.jpg")])
for conflict in conflicts:
    print(f"  CONFLICT [{conflict.severity}] {conflict.dimension}: {conflict.existing_value} -> {conflict.new_value}")
```

### Mode 3: Brand Compliance (requires Gemini + Pinecone)

Check creatives against brand guidelines:

```python
from omni_proof import ComplianceChain
from omni_proof.rag.brand_retriever import BrandRetriever

retriever = BrandRetriever(gemini_client=client, vector_store=store)
chain = ComplianceChain(gemini_client=client, brand_retriever=retriever)

report = await chain.check_compliance("new_ad_001", Path("new_ad.jpg"))
print(f"Passed: {report.passed}, Score: {report.score}")
for violation in report.violations:
    print(f"  [{violation.severity}] {violation.description}")
```

### Mode 4: Creative Generation

Generate optimized creative prompts from causal insights:

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
print(prompt)
```

### Mode 5: Translate Causal Results to Design Briefs

```python
from omni_proof import InsightSynthesizer

synthesizer = InsightSynthesizer(p_value_threshold=0.05, recommend_threshold=0.05)
brief = synthesizer.synthesize(cate_result)

print(f"Finding: {brief.finding}")
print(f"Recommendation: {brief.recommendation}")
print(f"Confidence: {brief.confidence}")
```

## Architecture

```
src/omni_proof/
  config/              Settings, constants
  core/                EmbeddingProvider ABC, exception hierarchy
  ingestion/           GeminiClient, AssetPreprocessor, IngestPipeline
  storage/             PineconeVectorStore, InMemoryVectorStore, RelationalStore
  causal/              CausalDAGBuilder, DMLEstimator, CausalRefuter
    dice_dml/          CounterfactualGenerator, Disentangler, VisualDMLEstimator
  brand_extraction/    BrandExtractor, PatternAggregator, ConflictDetector
  rag/                 BrandIndexer, BrandRetriever
  orchestration/       ComplianceChain, InsightSynthesizer
  api/                 FastAPI app, routes, GenerativePromptBuilder
```

### Key Abstractions

| Interface | Purpose | Implementations |
|:----------|:--------|:----------------|
| `EmbeddingProvider` | Generate embeddings from any content | `GeminiClient` (swap in OpenAI, local models, etc.) |
| `VectorStore` | Store and search vector embeddings | `PineconeVectorStore`, `InMemoryVectorStore` |
| `Estimator` | Estimate causal effects | `DMLEstimator` (add PropensityScore, Metalearners, etc.) |

## API Reference

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| `GET` | `/health` | Health check |
| `GET` | `/api/v1/causal/effects` | List all estimated causal effects |
| `GET` | `/api/v1/causal/effects/{treatment}` | CATE breakdown by segment |
| `POST` | `/api/v1/causal/analyze` | Trigger new causal analysis |
| `POST` | `/api/v1/compliance/check` | Upload creative for brand compliance review |
| `GET` | `/api/v1/compliance/reports` | Historical compliance reports |
| `GET` | `/api/v1/insights/briefs` | Latest design briefs from causal data |
| `GET` | `/api/v1/insights/segments` | Effects filtered by audience segment |
| `POST` | `/api/v1/generative/prompt` | Generate optimized creative prompt |
| `POST` | `/api/v1/brand/extract` | Extract brand profile from uploaded assets |
| `POST` | `/api/v1/brand/update/{id}` | Update brand profile with new assets |
| `GET` | `/api/v1/brand/profile/{id}` | Retrieve a brand profile |

## Causal Methodology

OmniProof uses a four-stage causal pipeline:

1. **Model** -- Construct a Directed Acyclic Graph (DAG) mapping treatments, outcomes, and confounders
2. **Identify** -- Apply the backdoor criterion to find valid adjustment sets
3. **Estimate** -- Double Machine Learning (Neyman Orthogonalization) via EconML to isolate true effects
4. **Refute** -- Placebo tests, subset validation, and random confounder checks to reject spurious findings

For visual embeddings where treatment and confounders are entangled, **DICE-DML** generates counterfactual pairs, extracts treatment fingerprints via vector subtraction, and applies orthogonal projection to disentangle the representation before estimation.

## Gemini Embedding 2

OmniProof leverages [Gemini Embedding 2](https://ai.google.dev/gemini-api/docs/embeddings) for native multimodal embeddings. All modalities (text, images, video, audio, PDF) map to the same 3072-dimensional semantic space.

| Modality | Limit |
|:---------|:------|
| Text | 8,192 tokens |
| Images | 6 per request |
| Video | 80s (with audio) / 120s (without) |
| Audio | 80s max |
| PDF | 1 document, 6 pages |
| Output | 3,072 dimensions (Matryoshka: truncate to 1536 / 768 / 128) |

## Testing

```bash
pytest tests/unit/ -v               # 142 unit tests
pytest tests/integration/ -v        # 15 integration tests
pytest tests/ -v                    # All 157 tests

ruff check src/ tests/              # Lint
ruff format src/ tests/             # Format
```

## Tech Stack

| Component | Technology |
|:----------|:-----------|
| Embeddings | Google Gemini Embedding 2 (`gemini-embedding-2-preview`) |
| Structured extraction | Google Gemini 2.0 Flash |
| Vector database | Pinecone Serverless / InMemoryVectorStore (dev) |
| Relational database | PostgreSQL (prod) / SQLite + aiosqlite (dev) |
| Causal inference | DoWhy + EconML (LinearDML, CausalForestDML) |
| Visual causality | DICE-DML (orthogonal projection) |
| API framework | FastAPI |
| ML models | LightGBM (first-stage nuisance models) |
| Schemas | Pydantic v2 + SQLAlchemy 2.0 async |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and PR guidelines.

## License

[MIT](LICENSE) -- OmniProof Contributors
