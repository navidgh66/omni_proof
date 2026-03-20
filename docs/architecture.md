# Architecture

OmniProof is organized into five layers, each independently usable. Data flows top-to-bottom through the pipeline, but any layer can be used standalone.

```
                    ┌─────────────────────────────┐
                    │        REST API (FastAPI)     │  Layer 5
                    │  /brand  /causal  /compliance │
                    │  /insights     /generative    │
                    └──────────────┬────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                     │
   ┌──────────▼──────────┐ ┌──────▼──────┐ ┌───────────▼───────────┐
   │   Brand Extraction   │ │ Orchestration│ │     RAG System        │  Layer 4
   │   BrandExtractor     │ │ Compliance   │ │  BrandIndexer         │
   │   ConflictDetector   │ │ InsightSynth │ │  BrandRetriever       │
   └──────────┬───────────┘ └──────┬──────┘ └───────────┬───────────┘
              │                    │                     │
              └────────────────────┼─────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │       Causal Engine          │  Layer 3
                    │  DAGBuilder → Identifier     │
                    │  DMLEstimator → Refuter      │
                    │  DICE-DML (Visual Estimator) │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │         Storage              │  Layer 2
                    │  PineconeVectorStore         │
                    │  InMemoryVectorStore         │
                    │  RelationalStore (SQLAlchemy) │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │        Ingestion             │  Layer 1
                    │  GeminiClient (Embedding 2)  │
                    │  AssetPreprocessor           │
                    │  IngestPipeline              │
                    └─────────────────────────────┘
```

## Layer 1: Ingestion

**Module:** `src/omni_proof/ingestion/`

Responsible for converting raw creative assets into embeddings and structured metadata.

| Component | Role |
|-----------|------|
| `GeminiClient` | Wraps Gemini API with retry logic and rate limiting. Implements `EmbeddingProvider` interface. Generates 3072-dim embeddings (Gemini Embedding 2) and extracts structured metadata (Gemini 3.1 Flash Lite). |
| `AssetPreprocessor` | Handles Gemini API limits: batches images (max 6), segments PDFs (max 6 pages), chunks video (80s/120s) and audio (80s). |
| `IngestPipeline` | Orchestrates parallel metadata extraction + embedding generation. Supports batch ingestion with graceful failure handling. |

**Key abstractions:**

- `EmbeddingProvider` (abstract) — any embedding backend can be swapped in
- `CreativeMetadata` (Pydantic) — structured schema for visual, temporal, textual, and auditory elements

**Data flow:**

```
Raw Asset (PNG/MP4/PDF)
  → AssetPreprocessor (chunk/batch if needed)
  → GeminiClient.generate_embedding() → [3072-dim float vector]
  → GeminiClient.extract_metadata()   → CreativeMetadata (Pydantic)
```

## Layer 2: Storage

**Module:** `src/omni_proof/storage/`

Dual storage: vectors for similarity search, relational for structured queries.

| Component | Role |
|-----------|------|
| `VectorStore` (abstract) | Interface for vector upsert, search, delete, batch operations. |
| `PineconeVectorStore` | Production implementation using Pinecone serverless. Two namespaces: `creatives` and `brand_assets`. |
| `InMemoryVectorStore` | Development/testing implementation using numpy cosine similarity. Same interface as Pinecone. |
| `RelationalStore` | Async SQLAlchemy 2.0 store for `CreativeMetadataRecord`, `PerformanceRecord`, and `CampaignRecord`. Uses `async_sessionmaker` with `AsyncSession`. |

**Namespaces:**

- `creatives` — embeddings of creative assets for similarity search
- `brand_assets` — embeddings of brand guidelines, approved creatives, and color palettes for RAG

## Layer 3: Causal Engine

**Module:** `src/omni_proof/causal/`

The statistical core. Estimates treatment effects using Double Machine Learning (DML) with rigorous robustness testing.

| Component | Role |
|-----------|------|
| `CausalDAGBuilder` | Constructs DoWhy causal models from treatment-confounder-outcome specs. Includes preset templates (`logo_timing`, `color_temperature`, `audio_pacing`). |
| `CausalIdentifier` | Applies the backdoor criterion to identify valid adjustment sets. |
| `DMLEstimator` | Wraps EconML's `LinearDML` and `CausalForestDML`. Estimates ATE (average) and CATE (segment-level) effects. |
| `CausalRefuter` | Three robustness tests: placebo (shuffle treatment), subset (drop 10%), random confounder (add noise). |

**DICE-DML sub-module** (`causal/dice_dml/`):

| Component | Role |
|-----------|------|
| `CounterfactualGenerator` | Creates embedding pairs where only the treatment attribute differs (e.g., fast vs. slow pacing of the same creative). |
| `TreatmentDisentangler` | Extracts treatment fingerprints via vector subtraction, then orthogonally projects embeddings to isolate confounder-only representations. |
| `VisualDMLEstimator` | Runs DML on disentangled embeddings to estimate visual treatment effects free of confounding. |

**Causal pipeline:**

```
Data (DataFrame)
  → CausalDAGBuilder.build_dag()     → CausalModel (DoWhy)
  → CausalIdentifier.identify_effect() → IdentifiedEstimand
  → DMLEstimator.estimate_ate()      → ATEResult
  → DMLEstimator.estimate_cate()     → CATEResult
  → CausalRefuter.placebo_test()     → RefutationResult
  → CausalRefuter.subset_test()      → RefutationResult
  → CausalRefuter.random_confounder_test() → RefutationResult
```

## Layer 4: Brand & Orchestration

**Module:** `src/omni_proof/brand_extraction/`, `src/omni_proof/orchestration/`, `src/omni_proof/rag/`

Three sub-systems that connect the engine to business workflows.

### Brand Extraction

| Component | Role |
|-----------|------|
| `BrandExtractor` | Orchestrates multi-asset brand identity extraction. Produces a `BrandProfile` with rules, voice, visual style, and visual fingerprint. |
| `AssetProcessor` | Processes individual assets through Gemini Flash (metadata) and Gemini Embedding 2 (vector) in parallel. |
| `PatternAggregator` | Aggregates patterns across multiple extractions. Detects outliers (cosine similarity < 0.7). Computes confidence scores per dimension. |
| `ConflictDetector` | Compares new extractions against an existing `BrandProfile`. Reports conflicts by dimension with severity (`major`/`minor`). |

### RAG System

| Component | Role |
|-----------|------|
| `BrandIndexer` | Ingests brand guidelines, approved creatives, and color palettes into the `brand_assets` vector namespace. |
| `BrandRetriever` | Cross-modal search: query by text or image, filter by `source_type` (guideline, approved_creative, palette). |

### Orchestration

| Component | Role |
|-----------|------|
| `ComplianceChain` | Embeds a new creative → retrieves relevant guidelines via RAG → evaluates compliance → produces a `ComplianceReport` with violations and score. |
| `InsightSynthesizer` | Converts `CATEResult` segment-level effects into a natural-language `DesignBrief` with recommendations (RECOMMENDED / CONSIDER / NEUTRAL / AVOID). |
| `GenerativePromptBuilder` | Combines causal insights + brand rules into parameterized prompts for generative models. |

## Layer 5: REST API

**Module:** `src/omni_proof/api/`

FastAPI application with 5 route modules:

| Route Prefix | Endpoints | Layer |
|-------------|-----------|-------|
| `/api/v1/brand` | `POST /extract`, `GET /profile/{id}`, `POST /update/{id}` | Brand Extraction |
| `/api/v1/causal` | `POST /analyze`, `GET /effects`, `GET /effects/{name}` | Causal Engine |
| `/api/v1/compliance` | `POST /check`, `GET /reports` | Orchestration |
| `/api/v1/insights` | `GET /briefs`, `GET /segments` | Orchestration |
| `/api/v1/generative` | `POST /prompt` | Generative |
| `/health` | `GET /health` | — |

See [REST API Reference](rest-api.md) for full request/response schemas.

## Key Design Decisions

### Abstract Interfaces

`EmbeddingProvider` and `VectorStore` are abstract base classes. This allows:

- Swapping Gemini for another embedding provider
- Using `InMemoryVectorStore` in tests and `PineconeVectorStore` in production
- Custom implementations for other vector databases

### Async Throughout

All I/O operations are async. The storage layer uses `async_sessionmaker` (SQLAlchemy 2.0), and the ingestion layer uses `asyncio.gather` for parallel embedding + metadata extraction.

### Pydantic Everywhere

Every data structure crossing a boundary is a Pydantic `BaseModel`:

- Ingestion schemas (`CreativeMetadata`, `VisualElements`, etc.)
- Causal results (`ATEResult`, `CATEResult`, `RefutationResult`)
- Brand models (`BrandProfile`, `BrandConflict`, `AssetExtraction`)
- Orchestration models (`ComplianceReport`, `Violation`, `DesignBrief`)
- API request/response models

### Exception Hierarchy

```
OmniProofError
├── IngestionError
│   ├── EmbeddingError
│   └── MetadataExtractionError
├── StorageError
│   ├── VectorStoreError
│   └── RelationalStoreError
├── CausalError
│   ├── DAGConstructionError
│   ├── EstimationError
│   └── RefutationError
└── ComplianceError
```

All exceptions inherit from `OmniProofError` for easy catch-all handling. The API layer catches these and returns structured error responses without leaking stack traces.

## Security

- **CORS**: Restricted to localhost origins; methods limited to GET/POST
- **Path traversal**: `field_validator` on `asset_paths` rejects `..` and unsafe characters
- **Upload filenames**: `_sanitize_filename()` strips path components and unsafe characters
- **Input validation**: `Field(min_length, max_length, pattern)` on all API request models
- **Causal route**: Treatment/outcome column names validated with `^[a-zA-Z_][a-zA-Z0-9_]*$`
- **Global exception handler**: No stack traces leaked to clients
