# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
