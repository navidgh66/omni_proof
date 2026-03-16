# OmniProof — AI Assistant Context

## Project
Causal-Multimodal Engine for Creative Performance Attribution using Gemini Embedding 2.

## Tech Stack
- Python 3.11+, src-layout, pyproject.toml (hatchling)
- Gemini Embedding 2 (gemini-embedding-2-preview) + Gemini 3.1 Flash Lite
- Pinecone serverless (vector DB)
- SQLAlchemy 2.0 async + Alembic (relational DB)
- DoWhy + EconML (causal inference)
- FastAPI (API layer)

## Commands
```bash
pip install -e ".[dev]"                          # Install
pytest -v --tb=short                             # Run all tests (203)
pytest tests/unit/ -v                            # Unit tests (150)
pytest tests/integration/ -v                     # Integration tests (53)
ruff check src/ tests/                           # Lint
ruff format src/ tests/                          # Format
mypy src/omni_proof/ --ignore-missing-imports    # Type check
python -m build                                  # Build wheel + sdist
docker-compose up -d                             # Start API + PostgreSQL
python examples/demo.py                          # Run offline demo (~4s, no API keys)
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

## Examples & Demo Data
- **Brand**: Velocity Sportswear (fictional DTC activewear)
- `examples/creatives/` — 10 PNG images + 4 MP4 A/B video variants (fast_pacing treatment)
- `examples/data/` — campaign_performance.csv (1000 rows, 2 planted causal effects), brand_profile.json, brand_guidelines.json (12 rules), compliance_samples.json, creative_metadata_samples.json (14 records)
- `examples/demo.py` — 9-stage offline demo: DAG, ATE, CATE, refutation, brief, brand, prompt, compliance
- `examples/playground.ipynb` — 10-walkthrough notebook using Pinecone (not InMemory), includes DICE-DML + embeddings-to-causal merge
- `examples/playground_output.ipynb` — same notebook with saved outputs (visual showcase for README)
- Creative names consistent across CSV, metadata JSON, and actual files (e.g. `Runner_Sunrise`)
- CSV planted effects: `fast_pacing` (ATE ~+1.9pp CTR, heterogeneous by segment), `warm_color_palette` (ATE ~+0.6pp)
- All JSON validates against Pydantic schemas (BrandProfile, CreativeMetadata)

## Security
- CORS: restricted to localhost origins, methods limited to GET/POST
- Path traversal: `field_validator` on `asset_paths` rejects `..` and unsafe chars
- Upload filenames: `_sanitize_filename()` strips path components + unsafe chars
- Input validation: `Field(min_length, max_length, pattern)` on all API request models
- Global exception handler: no stack traces leaked to clients
- Causal route: treatment/outcome validated with `^[a-zA-Z_][a-zA-Z0-9_]*$` pattern

## Data Generation Rules
- When generating demo data across multiple files (CSV, JSON, images), validate all cross-references in a single pass before declaring done
- Creative names must be consistent across: CSV `creative_name` column, metadata JSON `creative_name` field, and actual filenames in `examples/creatives/`
- Always validate generated JSON against Pydantic schemas: `BrandProfile`, `CreativeMetadata`, `ComplianceReport`
- Run `python examples/demo.py` after any example data changes to verify end-to-end

## Workflow Rules
- When reviewing PRs, always fetch fresh diff data from remote — never rely on cached data
- When a skill or plugin applies, invoke it immediately — do not begin work without checking first
- When asked about a specific PR/branch/issue number, scope all actions to that exact item only
- Repo is private — shields.io badges fail. Use GitHub-native badge URLs or static badges instead.
- Git local config uses `navidgh66` account — do not change global git config

## Automations

### Hooks (`.claude/settings.local.json`)
- **Auto-lint**: `ruff check --fix` + `ruff format` runs automatically on every `.py` file after Edit/Write
- **Block secrets**: Edits to `.env`, `.secret`, `.key`, `credentials` files are blocked pre-Edit/Write

### Skills
- `/validate-examples` — validates all example data cross-consistency, Pydantic schemas, and demo run
- `/release-check` — pre-release checklist: version bump, changelog, tests, lint, types, build, branch, clean tree

### Subagents
- `security-reviewer` (`.claude/agents/security-reviewer.md`) — reviews code for OWASP top 10 + regressions against existing security hardening

## Gotchas
- Use `async_sessionmaker` (not `sessionmaker`) for SQLAlchemy async — mypy rejects the sync overload with AsyncEngine
- `Counter` variables need explicit type annotations (`Counter[str]`) for mypy
- Gemini `generate_embedding` config dict must be typed `dict[str, Any]` because it mixes `int` and `str` values
- DAG template values from dicts are `Sequence[str]` — cast with `str()` / `list()` before passing to typed methods
- `generate_embedding` retry loop needs an explicit `raise` after the loop for mypy return-type satisfaction
- `langchain`/`langgraph` were removed — never imported, were dead dependencies
- `.ipynb` files: Edit tool is blocked; use NotebookEdit for cell edits or Bash with `python -c` for JSON-level changes (e.g. replacing strings across all cells)

## Releasing to PyPI
Tags must be on the `main` branch — the release workflow rejects tags on other branches.
```bash
# 1. Update version in pyproject.toml
# 2. Update CHANGELOG.md (change "Unreleased" to date)
# 3. Commit, push to main
git tag v0.1.0
git push origin v0.1.0
```
This triggers `.github/workflows/release.yml` which:
1. Verifies the tag is on `main`
2. Builds wheel + sdist
3. Publishes to PyPI (trusted publisher — no token needed)
4. Creates a GitHub Release with auto-generated notes

**Prerequisites:** Configure [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/adding-a-publisher/) with owner `navidgh66`, repo `omni_proof`, workflow `release.yml`, environment `pypi`. Also create a `pypi` environment in GitHub repo settings.

## Gemini Embedding 2 Limits
- Video: 80s (with audio) / 120s (without)
- Audio: 80s max
- Images: 6 per request
- PDF: 1 doc, 6 pages
- Output: 3072 dims (Matryoshka: 128/768/1536/3072)
