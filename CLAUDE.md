# OmniProof — AI Assistant Context

## Project
Causal-Multimodal Engine for Creative Performance Attribution using Gemini Embedding 2.

## Tech Stack
- Python 3.11+, src-layout, pyproject.toml (hatchling)
- Gemini Embedding 2 (gemini-embedding-2-preview) + Gemini 2.0 Flash
- Pinecone serverless (vector DB)
- SQLAlchemy 2.0 async + Alembic (relational DB)
- DoWhy + EconML (causal inference)
- FastAPI (API layer)

## Commands
```bash
pip install -e ".[dev]"                          # Install
pytest -v --tb=short                             # Run all tests (157)
pytest tests/unit/ -v                            # Unit tests (140)
pytest tests/integration/ -v                     # Integration tests (17)
ruff check src/ tests/                           # Lint
ruff format src/ tests/                          # Format
mypy src/omni_proof/ --ignore-missing-imports    # Type check
python -m build                                  # Build wheel + sdist
docker-compose up -d                             # Start API + PostgreSQL
```

## Environment Setup
Required env vars (prefix `OMNI_PROOF_`):
```bash
OMNI_PROOF_GEMINI_API_KEY=AIza...              # Gemini API
OMNI_PROOF_PINECONE_API_KEY=pcsk_...           # Pinecone
OMNI_PROOF_PINECONE_INDEX_HOST=https://...     # Pinecone index URL
OMNI_PROOF_DATABASE_URL=sqlite+aiosqlite:///./omni_proof.db  # Default
```
Causal analysis layer needs no API keys — works with local data only.

## Architecture (5 Layers)
1. **Ingestion** (`src/omni_proof/ingestion/`): AssetPreprocessor + GeminiClient + IngestPipeline
2. **Storage** (`src/omni_proof/storage/`): PineconeVectorStore + InMemoryVectorStore + RelationalStore
3. **Causal** (`src/omni_proof/causal/`): DAG builder + DML estimator + refuter + DICE-DML
4. **Brand & Orchestration** (`src/omni_proof/brand_extraction/`, `src/omni_proof/orchestration/`): BrandExtractor + ComplianceChain + InsightSynthesizer
5. **API** (`src/omni_proof/api/`): FastAPI routes + GenerativePromptBuilder

## Key Patterns
- All Gemini calls go through GeminiClient (retry + rate limiting)
- Vector store uses abstract VectorStore interface (Pinecone + InMemory implementations)
- Causal pipeline: DAG -> Identify -> Estimate (DML) -> Refute
- DICE-DML for visual embeddings: counterfactual pairs -> disentangle -> estimate
- Brand compliance: embed asset -> retrieve guidelines (RAG) -> evaluate -> report
- BrandExtractor: extract structured brand identity from multimodal assets with conflict detection

## Gotchas
- Use `async_sessionmaker` (not `sessionmaker`) for SQLAlchemy async — mypy rejects the sync overload with AsyncEngine
- `Counter` variables need explicit type annotations (`Counter[str]`) for mypy
- Gemini `generate_embedding` config dict must be typed `dict[str, Any]` because it mixes `int` and `str` values
- DAG template values from dicts are `Sequence[str]` — cast with `str()` / `list()` before passing to typed methods
- `generate_embedding` retry loop needs an explicit `raise` after the loop for mypy return-type satisfaction

## Gemini Embedding 2 Limits
- Video: 80s (with audio) / 120s (without)
- Audio: 80s max
- Images: 6 per request
- PDF: 1 doc, 6 pages
- Output: 3072 dims (Matryoshka: 128/768/1536/3072)
