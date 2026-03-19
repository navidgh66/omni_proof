# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.4] - 2026-03-19

### Added
- `CODE_OF_CONDUCT.md` (Contributor Covenant v2.1) — later removed, enforcement via GitHub
- `SECURITY.md` with vulnerability reporting process (moved to `.github/`)
- GitHub issue/PR templates and Dependabot config
- `InMemoryVectorStore` exported at top level for testing without Pinecone
- Subpackage exports: `ingestion`, `orchestration`, `rag` `__init__.py` populated
- CLI entry point `omniproof-demo` for running the demo
- Optional dependency groups: `api`, `docs`
- `[tool.mypy]` config in pyproject.toml

### Changed
- Fixed hardcoded version in `api/app.py` — now uses dynamic `__version__`
- Fixed clone URL in `CONTRIBUTING.md` (`your-org` → `navidgh66`)
- Moved `CONTRIBUTING.md` and `SECURITY.md` to `.github/`
- Added upper-bound constraints on major dependencies
- Expanded ruff rules: `UP`, `B`, `SIM`, `RUF` (migrated enums to `StrEnum`, fixed `raise from`)
- Updated `AGENTS.md` version from stale `0.0.1` to current

## [0.0.3] - 2026-03-17

### Changed
- Revamped README description and fixed logo URL for PyPI rendering

## [0.0.2] - 2026-03-16

### Added
- Playground notebook (`examples/playground.ipynb`) showcasing full pipeline with Pinecone
- Example creatives (10 PNG images + 4 MP4 A/B video variants) and demo data
- 9-stage offline demo script (`examples/demo.py`) — no API keys required
- Security reviewer agent and release check / example validation skills
- Integration tests for brand lifecycle, RAG compliance, API error/validation, and full pipeline
- AGENTS.md for AI coding assistant context

### Changed
- Expanded test suite from 157 to 203 tests (150 unit + 53 integration)
- Updated README with demo instructions, security section, and example data documentation

## [0.0.1] - 2026-03-15

### Added
- Initial project structure with src-layout and pyproject.toml configuration
- Multimodal ingestion pipeline with Gemini Embedding 2 support
  - Preprocessing for images, video, audio, PDF, and text
  - GeminiClient with retry logic and rate limiting
  - IngestPipeline for end-to-end asset processing
- Vector storage layer with Pinecone integration
  - Abstract VectorStore interface
  - PineconeVectorStore implementation with namespaces
- Relational storage with SQLAlchemy 2.0 async
  - Campaign, creative, and performance models
  - Alembic migrations support
- Causal inference pipeline using DoWhy and EconML
  - DAG builder for causal graph construction
  - DML estimator (LinearDML, CausalForestDML)
  - Refutation tests for sensitivity analysis
  - DICE-DML for visual embedding counterfactuals
- Orchestration layer
  - ComplianceChain for brand guideline validation
  - InsightSynthesizer for generating actionable recommendations
- FastAPI-based dashboard API
  - Campaign ingestion endpoint
  - Brand compliance endpoints
  - Causal analysis endpoints
  - Generative prompt builder integration
- Comprehensive test suite
  - 140 unit tests and 17 integration tests with pytest and pytest-asyncio
  - Integration test scaffolding
- Development tooling
  - Ruff for linting and formatting
  - Type hints throughout codebase
  - Structured logging with structlog

### Dependencies
- google-genai for Gemini API access
- pinecone for vector storage
- dowhy and econml for causal inference
- fastapi and uvicorn for API layer
- sqlalchemy and alembic for relational storage
