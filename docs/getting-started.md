# Getting Started

## Installation

### From PyPI

```bash
pip install omni-proof
```

### From Source

```bash
git clone https://github.com/navidgh66/omni_proof.git
cd omni_proof
pip install -e ".[dev]"
```

### Verify Installation

```python
import omni_proof
print(omni_proof.__version__)
```

## Environment Setup

OmniProof uses environment variables prefixed with `OMNI_PROOF_`. Set them in your shell or a `.env` file:

```bash
# Required for embedding & metadata extraction
OMNI_PROOF_GEMINI_API_KEY=AIza...

# Required for Pinecone vector storage
OMNI_PROOF_PINECONE_API_KEY=pcsk_...
OMNI_PROOF_PINECONE_INDEX_HOST=https://your-index-abc123.svc.pinecone.io

# Optional — defaults to local SQLite
OMNI_PROOF_DATABASE_URL=sqlite+aiosqlite:///./omni_proof.db
```

> **No API keys?** The causal analysis layer, design brief generation, and offline demo work entirely locally. Only the ingestion (Gemini) and vector storage (Pinecone) layers require API keys.

## First Run: Offline Demo

The fastest way to see OmniProof in action — no API keys needed:

```bash
python examples/demo.py
```

This runs a 9-stage demo in ~4 seconds:

1. **Campaign Overview** — loads 1,000 synthetic creatives across 5 campaigns
2. **DAG Construction** — builds a causal graph (treatment: `fast_pacing`, outcome: `ctr`)
3. **ATE Estimation** — estimates Average Treatment Effect via DML
4. **CATE Estimation** — segment-level effects (Gen-Z, Millennials, Gen-X)
5. **Refutation Suite** — placebo, subset, and random confounder tests
6. **Design Brief** — translates causal findings into actionable recommendations
7. **Brand Profile** — displays Velocity Sportswear brand identity
8. **Creative Prompt** — generates a parameterized prompt from insights + brand rules
9. **Compliance Reports** — shows sample PASS/WARN/FAIL reports

## Using OmniProof as a Library

### Causal Analysis (No API Keys)

```python
import pandas as pd
from omni_proof import DMLEstimator

data = pd.read_csv("examples/data/campaign_performance.csv")

estimator = DMLEstimator(cv=5, n_estimators=50)

# Average Treatment Effect
ate = estimator.estimate_ate(
    data,
    treatment_col="fast_pacing",
    outcome_col="ctr",
    confounder_cols=["platform", "audience_segment", "daily_budget_usd"],
)
print(f"ATE: {ate.ate:.4f} (p={ate.p_value:.4f})")
print(f"95% CI: [{ate.ci_lower:.4f}, {ate.ci_upper:.4f}]")

# Conditional Average Treatment Effect by segment
cate = estimator.estimate_cate(
    data,
    treatment_col="fast_pacing",
    outcome_col="ctr",
    confounder_cols=["platform", "daily_budget_usd"],
    segment_col="audience_segment",
)
for segment, effect in cate.segments.items():
    print(f"  {segment}: {effect.effect:+.4f} (p={effect.p_value:.4f})")
```

### Full Pipeline (Requires API Keys)

```python
import asyncio
from pathlib import Path
from omni_proof import (
    GeminiClient,
    PineconeVectorStore,
    BrandExtractor,
    ComplianceChain,
    Settings,
)
from omni_proof.rag.brand_retriever import BrandRetriever

settings = Settings()  # reads OMNI_PROOF_* env vars

async def main():
    gemini = GeminiClient(api_key=settings.gemini_api_key)

    # Initialize Pinecone (see Configuration docs for setup)
    from pinecone import Pinecone
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(host=settings.pinecone_index_host)
    store = PineconeVectorStore(index)

    # Extract brand identity from assets
    extractor = BrandExtractor(gemini, gemini, store)
    profile = await extractor.extract(
        brand_name="Velocity Sportswear",
        assets=[Path("examples/creatives/runner_sunrise.png")],
    )
    print(f"Brand: {profile.brand_name}")
    print(f"Rules: {len(profile.rules)}")

    # Check compliance
    retriever = BrandRetriever(gemini, store)
    chain = ComplianceChain(gemini, retriever)
    report = await chain.check_compliance(
        asset_id="test-001",
        asset_path=Path("examples/creatives/yoga_flow.png"),
    )
    print(f"Compliant: {report.passed} (score: {report.score:.2f})")

asyncio.run(main())
```

## Running the API Server

```bash
# With Docker
docker-compose up -d

# Without Docker
uvicorn omni_proof.api.app:create_app --factory --reload --port 8000
```

The API serves at `http://localhost:8000`. Check health:

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

See [REST API Reference](rest-api.md) for all endpoints.

## Running Tests

```bash
# All tests (203)
pytest -v --tb=short

# Unit tests only (150)
pytest tests/unit/ -v

# Integration tests only (53)
pytest tests/integration/ -v

# Lint & format
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy src/omni_proof/ --ignore-missing-imports
```

## Next Steps

- [Architecture](architecture.md) — understand the 5-layer system design
- [API Reference](api-reference.md) — full class and method documentation
- [Causal Methodology](causal-methodology.md) — the statistical theory behind OmniProof
- [Examples & Tutorials](examples.md) — hands-on walkthroughs
