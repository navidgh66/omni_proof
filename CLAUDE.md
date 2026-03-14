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
pip install -e ".[dev]"    # Install
pytest -v --tb=short       # Run all tests
pytest tests/unit/ -v      # Unit tests only
pytest tests/integration/  # Integration tests
ruff check src/ tests/     # Lint
ruff format src/ tests/    # Format
```

## Architecture (5 Layers)
1. **Ingestion** (`src/omni_proof/ingestion/`): Preprocessor + GeminiClient + IngestPipeline
2. **Storage** (`src/omni_proof/storage/`): PineconeVectorStore + RelationalStore
3. **Causal** (`src/omni_proof/causal/`): DAG builder + DML estimator + refuter + DICE-DML
4. **Orchestration** (`src/omni_proof/orchestration/`): ComplianceChain + InsightSynthesizer
5. **API** (`src/omni_proof/api/`): FastAPI routes + GenerativePromptBuilder

## Key Patterns
- All Gemini calls go through GeminiClient (retry + rate limiting)
- Vector store uses abstract VectorStore interface (Pinecone implementation)
- Causal pipeline: DAG -> Identify -> Estimate (DML) -> Refute
- DICE-DML for visual embeddings: counterfactual pairs -> disentangle -> estimate
- Brand compliance: embed asset -> retrieve guidelines (RAG) -> evaluate -> report

## Gemini Embedding 2 Limits
- Video: 80s (with audio) / 120s (without)
- Audio: 80s max
- Images: 6 per request
- PDF: 1 doc, 6 pages
- Output: 3072 dims (Matryoshka: 128/768/1536/3072)
