# OmniProof — Agent Context

This file provides context for AI coding assistants working on this project.

## Project Overview
Causal-Multimodal Engine for Creative Performance Attribution using Gemini Embedding 2 + Pinecone + DoWhy/EconML.

## Tech Stack
- Python 3.11+, src-layout with pyproject.toml (hatchling)
- Gemini Embedding 2 (`gemini-embedding-2-preview`) — 3072 dims, Matryoshka
- Gemini 3.1 Flash Lite (`gemini-3.1-flash-lite-preview`) for structured extraction
- Pinecone serverless — namespaces: "creatives", "brand_assets"
- SQLAlchemy 2.0 async (`async_sessionmaker`, NOT `sessionmaker`) + Alembic, aiosqlite for dev
- DoWhy + EconML (LinearDML, CausalForestDML) for causal inference
- FastAPI for API layer
- pytest + pytest-asyncio for testing

## Version & Release
- Current version: `0.0.4` (in pyproject.toml, single source of truth via `importlib.metadata`)
- Published to PyPI as `omni-proof` (yanked — was test publish)
- Release workflow: `.github/workflows/release.yml` — triggers on `v*` tags, verifies tag is on main
- PyPI trusted publisher configured: owner `navidgh66`, repo `omni_proof`, workflow `release.yml`, env `pypi`

## Test Suite
- **203 tests total** (150 unit + 53 integration), all passing
- Run: `.venv/bin/pytest tests/ -v --tb=short`
- Lint: `.venv/bin/ruff check src/ tests/`
- Type check: `.venv/bin/mypy src/omni_proof/ --ignore-missing-imports`
- Build: `.venv/bin/python -m build`

## Examples & Demo Data
- **Brand**: Velocity Sportswear (fictional DTC activewear)
- `examples/creatives/` — 10 PNG images + 4 MP4 A/B video variants
  - Images: runner_sunrise, trail_epic, hiit_studio, yoga_flow, sprint_track, swim_laps, basketball_court, cycling_mountain, crossfit_box, morning_stretch
  - Videos: runner_sunrise + basketball_court with fast_pacing A/B variants
  - Images generated with Nano Banana, videos generated from images via Nano Banana
- `examples/data/` — campaign_performance.csv (1000 rows), brand_profile.json, brand_guidelines.json (12 rules), compliance_samples.json (5 reports), creative_metadata_samples.json (14 records: 10 images + 4 videos)
- `examples/demo.py` — 9-stage offline demo (no API keys), runs in ~4s
- Creative names are consistent across CSV, metadata JSON, and actual files (e.g. `Runner_Sunrise`)
- CSV has two planted causal effects: `fast_pacing` (ATE ~+1.9pp CTR, heterogeneous by segment) and `warm_color_palette` (ATE ~+0.6pp)
- All JSON data validates against Pydantic schemas (BrandProfile, CreativeMetadata, ComplianceReport)
- README "Hands-On Playground" section: 8 collapsible walkthroughs (offline demo + 7 with API keys)

## Security Hardening
- CORS: restricted to localhost origins (no wildcard), methods limited to GET/POST
- Path traversal: `field_validator` on `asset_paths` rejects `..` and unsafe chars
- Upload filenames: `_sanitize_filename()` strips path components + unsafe chars
- Input validation: `Field(min_length, max_length, pattern)` on all API request models
- Global exception handler: no stack traces leaked to clients
- Causal route: treatment/outcome validated with `^[a-zA-Z_][a-zA-Z0-9_]*$` pattern

## Key Gotchas
- Use `async_sessionmaker` (not `sessionmaker`) for SQLAlchemy async — mypy rejects sync overload
- `Counter` variables need `Counter[str]` type annotation for mypy
- Gemini config dict must be `dict[str, Any]` (mixes int + str values)
- DAG template values are `Sequence[str]` — cast with `str()`/`list()` for typed methods
- `generate_embedding` retry loop needs explicit `raise` after loop for mypy return type
- `langchain`/`langgraph` were removed — never imported, were dead dependencies

## Architecture (5 Layers)
1. **Ingestion** (`src/omni_proof/ingestion/`): AssetPreprocessor + GeminiClient + IngestPipeline
2. **Storage** (`src/omni_proof/storage/`): PineconeVectorStore + InMemoryVectorStore + RelationalStore
3. **Causal** (`src/omni_proof/causal/`): DAG builder + DML estimator + refuter + DICE-DML
4. **Brand & Orchestration** (`src/omni_proof/brand_extraction/`, `src/omni_proof/orchestration/`): BrandExtractor + ComplianceChain + InsightSynthesizer
5. **API** (`src/omni_proof/api/`): FastAPI routes + GenerativePromptBuilder
