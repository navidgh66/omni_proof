<p align="center">
  <img src="https://raw.githubusercontent.com/navidgh66/omni_proof/main/assets/logo.png" width="320" alt="OmniProof" />
</p>

<p align="center">
  <strong>The engine that sees your creatives (Gemini Embedding 2), extracts their DNA automatically, and proves what causes performance.<br>Not vibes. Not correlations. <em>Proof.</em></strong>
</p>

<p align="center">
  <a href="https://github.com/navidgh66/omni_proof/releases"><img src="https://img.shields.io/github/v/release/navidgh66/omni_proof?style=for-the-badge&label=Release&color=blue" alt="Release"></a>
  <a href="https://github.com/navidgh66/omni_proof"><img src="https://img.shields.io/badge/python-3.11+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://github.com/navidgh66/omni_proof/stargazers"><img src="https://img.shields.io/github/stars/navidgh66/omni_proof?style=for-the-badge&color=yellow" alt="Stars"></a>
  <a href="https://github.com/navidgh66/omni_proof/pulls"><img src="https://img.shields.io/github/issues-pr/navidgh66/omni_proof?style=for-the-badge&color=purple&label=PRs" alt="Pull Requests"></a>
  <a href="https://github.com/navidgh66/omni_proof/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="MIT License"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#hands-on-playground">Playground</a> &middot;
  <a href="#how-it-works">How It Works</a> &middot;
  <a href="#features">Features</a> &middot;
  <a href="#documentation">Docs</a> &middot;
  <a href="#api-reference">API</a> &middot;
  <a href="#contributing">Contributing</a>
</p>

---

Everyone knows *which* creatives performed. Nobody knows *why*. **OmniProof** does.

Upload any creative — video, image, PDF. OmniProof embeds it (**Gemini Embedding 2**), extracts structured metadata automatically, estimates causal effects with **Double Machine Learning**, and checks brand compliance via **RAG**. The output isn't a dashboard of correlations. It's proof: *"blue backgrounds **cause** a +12% CTR uplift for 18-24, controlling for platform, budget, and seasonality."*

## Quick Start

```bash
pip install omni-proof
```

<details><summary>Or install from source</summary>

```bash
git clone https://github.com/navidgh66/omni_proof.git
cd omni_proof
pip install -e ".[dev]"
```

</details>

> Requires Python 3.11+. The causal analysis layer works with local data only — no API keys needed. For the full pipeline you'll need a [Gemini API key](https://aistudio.google.com/apikey) and a [Pinecone](https://app.pinecone.io/) account.

**Try it now — zero setup, zero API keys:**

```bash
python examples/demo.py   # 9-stage pipeline demo in ~4 seconds
```

**Or use it as a library:**

```python
import pandas as pd
from omni_proof import DMLEstimator

data = pd.read_csv("examples/data/campaign_performance.csv")
estimator = DMLEstimator(cv=5, n_estimators=50)

# Does fast pacing *cause* higher CTR?
ate = estimator.estimate_ate(data, "fast_pacing", "ctr",
    ["platform", "audience_segment", "daily_budget_usd"])
print(f"ATE: {ate.ate:+.4f} (p={ate.p_value:.4f})")
# ATE: +0.0189 (p=0.0001)

# For which segments?
cate = estimator.estimate_cate(data, "fast_pacing", "ctr",
    ["platform", "daily_budget_usd"], segment_col="audience_segment")
for seg, eff in cate.segments.items():
    print(f"  {seg}: {eff.effect:+.4f} (p={eff.p_value:.4f})")
# Gen-Z 18-24:    +0.032 (p=0.001)  — strong
# Millennials:     +0.015 (p=0.023)  — moderate
# Gen-X 35-44:    +0.003 (p=0.412)  — not significant
```

<details><summary>Full pipeline example (requires API keys)</summary>

```python
import asyncio
from pathlib import Path
from omni_proof import (
    BrandExtractor, ComplianceChain, GeminiClient, Settings,
)
from omni_proof.storage.memory_store import InMemoryVectorStore
from omni_proof.rag.brand_retriever import BrandRetriever

settings = Settings()  # reads OMNI_PROOF_* env vars
client = GeminiClient(api_key=settings.gemini_api_key)
store = InMemoryVectorStore()

async def main():
    # Extract brand identity from assets
    extractor = BrandExtractor(client, client, store)
    profile = await extractor.extract("AcmeCorp",
        [Path("brand_guide.pdf"), Path("logo.png")])

    # Check a new creative for brand compliance
    retriever = BrandRetriever(client, store)
    chain = ComplianceChain(client, retriever)
    report = await chain.check_compliance("ad_001", Path("new_ad.jpg"))
    print(f"Compliant: {report.passed} (score: {report.score})")

asyncio.run(main())
```

</details>

## Hands-On Playground

The [`examples/playground.ipynb`](examples/playground.ipynb) notebook walks through every OmniProof capability interactively — from embedding creatives into Pinecone, to causal analysis, to DICE-DML on A/B video pairs.

| Walkthrough | What it shows | API keys? |
|:--|:--|:--|
| 0. Offline Demo | All 9 pipeline stages in ~4s | No |
| 1. Embed into Pinecone | Gemini Embedding 2 → Pinecone storage | Yes |
| 2. Extract Metadata | Structured creative metadata via Flash Lite | Yes |
| 3. Brand Identity | Multimodal brand profile extraction | Yes |
| 4. Brand Compliance | RAG: embed guidelines → retrieve → evaluate | Yes |
| 5. Causal Analysis | DAG → DML → ATE/CATE → refutation | No |
| 6. Embeddings + Causal | Merge Pinecone embeddings with campaign data | Yes |
| 7. DICE-DML | Counterfactual pairs → treatment fingerprint → visual ATE | Yes |
| 8. Design Brief | Causal insights → creative prompt generation | No |
| 9. API Server | FastAPI endpoints for all capabilities | Yes |

```bash
pip install -e ".[dev]"
jupyter notebook examples/playground.ipynb
```

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

## Highlights

- **[Causal Engine](#causal-methodology)** -- DML + refutation tests isolate true treatment effects from confounders. Not correlations.
- **[Multimodal Embeddings](#gemini-embedding-2)** -- Gemini Embedding 2 maps video, images, audio, PDFs, and text into a shared 3072-dim space.
- **[Brand Intelligence](#hands-on-playground)** -- Extract structured brand guidelines from any asset, then auto-check new creatives for compliance.
- **[DICE-DML](#causal-methodology)** -- Disentangle visual confounders from treatment signals using counterfactual embedding pairs.
- **[Creative Generation](#api-reference)** -- Causal insights feed directly into optimized creative prompts.
- **[REST API](#api-reference)** -- 12 endpoints covering brand extraction, compliance, causal analysis, and generation.
- **[Modular](#architecture)** -- Use the full pipeline or any layer independently as a library.

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

Start the server:

```bash
uvicorn omni_proof.api.app:create_app --factory --reload
curl localhost:8000/health  # {"status": "ok"}
```

## Documentation

For in-depth guides, see the [`docs/`](docs/) directory:

| Document | Description |
|:---------|:------------|
| [Getting Started](docs/getting-started.md) | Installation, environment setup, and first run |
| [Architecture](docs/architecture.md) | System design, layer overview, and data flow |
| [API Reference](docs/api-reference.md) | Complete class and method reference |
| [Configuration](docs/configuration.md) | Environment variables, settings, and constants |
| [Causal Methodology](docs/causal-methodology.md) | DML, CATE, DICE-DML, and refutation explained |
| [REST API](docs/rest-api.md) | HTTP endpoints, request/response schemas |
| [Examples & Tutorials](docs/examples.md) | Walkthrough of demo, playground, and common workflows |

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
