# Blueprint: Causal-Multimodal Engine for Creative Performance Attribution

**Objective:** Build an enterprise-grade system that leverages Gemini Embedding 2 to (1) extract structured metadata from multimodal ad creatives, (2) generate dense embeddings, (3) perform causal inference via Double Machine Learning and DICE-DML, (4) enforce brand compliance via multimodal RAG, and (5) present actionable insights through dashboards with a closed-loop generative feedback system.

**Repository:** `navidgh66/omni_proof`
**Base branch:** `main`
**Mode:** Full branch/PR workflow (git + gh available)

---

## Dependency Graph

```
Step 1 (Project Scaffold)
    |
    v
Step 2 (Pydantic Schemas) ──────> Step 3 (Gemini Ingestion Pipeline)
    |                                      |
    v                                      v
Step 4 (Vector DB + Relational Storage) <──┘
    |
    ├──> Step 5 (Causal DAG + DoWhy Core) ──> Step 7 (DML + EconML Estimation)
    |                                                    |
    ├──> Step 6 (Brand RAG Knowledge Base)               v
    |         |                              Step 8 (DICE-DML Visual Causality)
    |         v
    |    Step 9 (Orchestration + RAG Pipeline)
    |         |
    v         v
Step 10 (Dashboard + Generative Loop)
    |
    v
Step 11 (Integration Tests + E2E Validation)
    |
    v
Step 12 (Documentation + Deployment Config)
```

**Parallel groups:**
- Steps 5 & 6 can run in parallel (no shared files)
- Steps 7 & 9 can overlap once their dependencies complete

---

## Step 1: Project Scaffold and Environment Setup

**Branch:** `feat/project-scaffold`
**PR into:** `main`
**Model tier:** default
**Estimated scope:** 1 PR

### Context Brief
This is a greenfield Python project. No code exists yet — only research documents in `research/`. Set up the monorepo structure, dependency management, and configuration foundation.

### Tasks
- [ ] Initialize Python project with `pyproject.toml` (Python 3.11+)
- [ ] Create directory structure:
  ```
  src/
    omni_proof/
      __init__.py
      config/
        settings.py          # Pydantic Settings for env vars
        constants.py          # Model names, dimension defaults
      ingestion/
      storage/
      causal/
      rag/
      orchestration/
      api/
  tests/
    unit/
    integration/
  scripts/
  ```
- [ ] Add core dependencies to `pyproject.toml`:
  - `google-genai` (Gemini SDK)
  - `pydantic` + `pydantic-settings`
  - `dowhy`, `econml`
  - `langchain`, `langgraph`
  - `pinecone` (Pinecone Python SDK)
  - `sqlalchemy`, `alembic`
  - `pytest`, `pytest-asyncio`
  - `httpx`, `structlog`
- [ ] Create `.env.example` with placeholder keys (GEMINI_API_KEY, PINECONE_API_KEY, DATABASE_URL)
- [ ] Add `.gitignore` for Python, `.env`, `__pycache__`, etc.
- [ ] Create `Makefile` with targets: `install`, `test`, `lint`, `format`

### Verification
```bash
pip install -e ".[dev]" && python -c "import omni_proof; print('OK')" && make test
```

### Exit Criteria
- `pip install -e .` succeeds
- `import omni_proof` works
- Directory structure matches spec
- All dependencies resolve

### Rollback
```bash
git checkout main && git branch -D feat/project-scaffold
```

---

## Step 2: Pydantic Schemas for Creative Metadata Extraction

**Branch:** `feat/pydantic-schemas`
**PR into:** `main` (after Step 1 merged)
**Depends on:** Step 1
**Model tier:** default

### Context Brief
The causal engine requires structured metadata from raw ad creatives. Gemini's Structured Output feature enforces deterministic extraction using developer-defined schemas. These schemas define the "digital twin" of every creative asset — the discrete variables that become treatment vectors and covariates in the causal models.

### Tasks
- [ ] Create `src/omni_proof/ingestion/schemas.py`:
  - `VisualElements` — object detection results, spatial ratios (logo screen %), background setting enum, dominant hex codes, contrast ratios
  - `TemporalPacing` — scene transition frequency, time-to-first-logo (seconds), product exposure duration, motion intensity score
  - `TextualElements` — on-screen text density, CTA classification enum (urgency/passive/inquisitive), promotional phrasing, typography style
  - `AuditoryElements` — audio genre, voiceover demographic estimation, emotional tone enum (authoritative/conversational/energetic), music tempo BPM
  - `CreativeMetadata` — composite model combining all four modality categories + asset_id, timestamp, platform, campaign_id
- [ ] Create `src/omni_proof/ingestion/enums.py` — all enum types (BackgroundSetting, CTAType, EmotionalTone, AudioGenre, etc.)
- [ ] Create `src/omni_proof/storage/models.py` — SQLAlchemy ORM models mirroring the Pydantic schemas for relational storage
- [ ] Write unit tests in `tests/unit/test_schemas.py`:
  - Validate all enum constraints
  - Test serialization/deserialization round-trips
  - Test schema export for Gemini API compatibility

### Verification
```bash
pytest tests/unit/test_schemas.py -v
```

### Exit Criteria
- All Pydantic models validate correctly with sample data
- Schemas export to JSON Schema format compatible with Gemini Structured Output
- SQLAlchemy models mirror Pydantic structure

---

## Step 3: Gemini Ingestion Pipeline (Embedding + Metadata Extraction)

**Branch:** `feat/gemini-ingestion`
**PR into:** `main` (after Step 2 merged)
**Depends on:** Steps 1, 2
**Model tier:** strongest (complex API integration)

### Context Brief
This is the foundational data transformation layer. Raw ad creatives (video up to 80s with audio / 120s without, images up to 6 per request, audio up to 80s, PDFs up to 6 pages) are processed through two Gemini endpoints: (1) Gemini 2.0 with Structured Output for metadata extraction using the Pydantic schemas from Step 2, and (2) `gemini-embedding-2-preview` for dense vector generation (default 3072 dims, truncatable to 1536/768/128 via Matryoshka Representation Learning).

### Tasks
- [ ] Create `src/omni_proof/ingestion/gemini_client.py`:
  - Async wrapper around `google-genai` SDK
  - `extract_metadata(asset_path, schema) -> CreativeMetadata` — sends asset + Pydantic schema to Gemini 2.0, returns structured output
  - `generate_embedding(asset_path, dimensions=3072) -> list[float]` — calls `gemini-embedding-2-preview`, supports Matryoshka truncation
  - Rate limiting and retry logic with exponential backoff
- [ ] Create `src/omni_proof/ingestion/preprocessor.py`:
  - Video chunking: split videos > 80s (with audio) or > 120s (without) into segments at keyframes
  - PDF segmentation: split > 6 pages into batches
  - Image batching: group images into batches of 6
- [ ] Create `src/omni_proof/ingestion/pipeline.py`:
  - `IngestPipeline` class orchestrating: preprocess -> extract metadata -> generate embedding -> return `(CreativeMetadata, embedding_vector)`
  - Support batch processing of asset directories
  - Structured logging for each stage
- [ ] Write tests:
  - `tests/unit/test_preprocessor.py` — video chunking logic, PDF splitting
  - `tests/integration/test_gemini_client.py` — mock Gemini API responses, validate schema enforcement

### Verification
```bash
pytest tests/unit/test_preprocessor.py tests/integration/test_gemini_client.py -v
```

### Exit Criteria
- Pipeline processes sample video/image/audio and returns valid `CreativeMetadata` + embedding vector
- Matryoshka truncation produces correct dimension counts
- Rate limiting handles 429 errors gracefully

---

## Step 4: Vector Database + Relational Storage Layer

**Branch:** `feat/storage-layer`
**PR into:** `main` (after Step 3 merged)
**Depends on:** Steps 2, 3
**Model tier:** default

### Context Brief
Embeddings go into a vector database (Pinecone serverless for scalability and metadata filtering) for semantic search. Structured performance data (impressions, clicks, conversions) and extracted metadata go into a relational database via SQLAlchemy. Both stores share `asset_id` as the join key.

### Tasks
- [ ] Create `src/omni_proof/storage/vector_store.py`:
  - `VectorStore` abstract interface with `upsert(asset_id, embedding, metadata)`, `search(query_embedding, top_k, filters)`, `delete(asset_id)`
  - `PineconeVectorStore` implementation using `pinecone` SDK
  - Support metadata payload filtering (by campaign_id, date range, platform)
  - Collection creation with configurable vector dimensions
- [ ] Create `src/omni_proof/storage/relational_store.py`:
  - SQLAlchemy async engine setup
  - `RelationalStore` with CRUD for `CreativeMetadata`, `PerformanceData`, `CampaignData`
  - Alembic migration setup in `alembic/`
- [ ] Create `src/omni_proof/storage/performance_models.py`:
  - `PerformanceRecord` — asset_id, impressions, clicks, conversions, roas, ctr, platform, audience_segment, date
  - `CampaignRecord` — campaign_id, name, start_date, end_date, budget, target_demographics
- [ ] Create initial Alembic migration
- [ ] Write tests:
  - `tests/unit/test_vector_store.py` — mock Pinecone client
  - `tests/unit/test_relational_store.py` — SQLite in-memory tests

### Verification
```bash
pytest tests/unit/test_vector_store.py tests/unit/test_relational_store.py -v
```

### Exit Criteria
- Vector upsert/query/delete works with mock Pinecone client
- Relational CRUD operations work against SQLite
- Alembic migration generates and applies cleanly

---

## Step 5: Causal DAG Construction with DoWhy

**Branch:** `feat/causal-dag`
**PR into:** `main` (after Step 4 merged)
**Depends on:** Step 4
**Model tier:** strongest (complex causal modeling)
**Parallel with:** Step 6

### Context Brief
The causal engine formalizes assumptions using Directed Acyclic Graphs (DAGs) via the DoWhy library. The DAG maps: treatments (creative features like "logo in first 3 seconds"), outcomes (ROAS, CTR), and confounders (seasonal trends, demographics, platform, production quality). The system must satisfy the unconfoundedness assumption and correctly handle colliders/mediators.

### Tasks
- [ ] Create `src/omni_proof/causal/__init__.py`
- [ ] Create `src/omni_proof/causal/dag_builder.py`:
  - `CausalDAGBuilder` class that constructs DoWhy causal models
  - `build_dag(treatment_col, outcome_col, confounders, mediators=None, colliders=None) -> CausalModel`
  - Pre-built DAG templates for common creative attribution scenarios:
    - Logo placement timing -> CTR (confounded by platform, audience, season)
    - Color temperature -> Engagement (confounded by product category, production quality)
    - Audio pacing -> Conversion (confounded by demographic, time of day)
  - Validation: warn if potential colliders are included in adjustment set
- [ ] Create `src/omni_proof/causal/identifier.py`:
  - `identify_effect(causal_model) -> IdentifiedEstimand`
  - Implements backdoor criterion identification
  - Returns the mathematical formula for the target estimand
- [ ] Write tests:
  - `tests/unit/test_dag_builder.py` — DAG construction, confounder mapping
  - `tests/unit/test_identifier.py` — backdoor criterion identification on sample DAGs

### Verification
```bash
pytest tests/unit/test_dag_builder.py tests/unit/test_identifier.py -v
```

### Exit Criteria
- DAGs correctly represent treatment-confounder-outcome relationships
- Backdoor criterion correctly identifies valid adjustment sets
- Collider warnings fire when appropriate

---

## Step 6: Brand RAG Knowledge Base Construction

**Branch:** `feat/brand-rag`
**PR into:** `main` (after Step 4 merged)
**Depends on:** Step 4
**Model tier:** default
**Parallel with:** Step 5

### Context Brief
The system serves as a multimodal RAG for brand compliance. Brand guidelines (PDFs, images, color palettes, font files, approved creatives) are embedded via Gemini Embedding 2 into the same vector store. Cross-modal retrieval enables text->image, image->text, and image->image queries against brand assets.

### Tasks
- [ ] Create `src/omni_proof/rag/__init__.py`
- [ ] Create `src/omni_proof/rag/brand_indexer.py`:
  - `BrandIndexer` class that ingests brand guideline documents
  - `index_brand_guide(pdf_path)` — segments PDF, generates embeddings per page/section, stores with metadata (section_type, page_number)
  - `index_approved_creative(asset_path, tags)` — embeds approved creatives with brand tags
  - `index_color_palette(hex_codes, names)` — stores color palette as searchable vectors
- [ ] Create `src/omni_proof/rag/brand_retriever.py`:
  - `BrandRetriever` class for cross-modal search
  - `search_by_text(query, top_k) -> list[BrandAsset]` — text-to-image/doc retrieval
  - `search_by_image(image_path, top_k) -> list[BrandAsset]` — image-to-image retrieval
  - `get_guidelines_for_asset(asset_path) -> list[BrandRule]` — find relevant brand rules for a given creative
- [ ] Create `src/omni_proof/rag/models.py`:
  - `BrandAsset`, `BrandRule`, `ComplianceResult` Pydantic models
- [ ] Write tests:
  - `tests/unit/test_brand_indexer.py`
  - `tests/unit/test_brand_retriever.py`

### Verification
```bash
pytest tests/unit/test_brand_indexer.py tests/unit/test_brand_retriever.py -v
```

### Exit Criteria
- Brand PDFs are segmented and embedded correctly
- Cross-modal retrieval returns relevant results (text->image, image->text)
- Brand rules are associated with correct asset types

---

## Step 7: Double Machine Learning + EconML Estimation

**Branch:** `feat/dml-estimation`
**PR into:** `main` (after Step 5 merged)
**Depends on:** Step 5
**Model tier:** strongest (complex econometric modeling)

### Context Brief
The estimation core implements the Partially Linear Model via Double Machine Learning (Neyman Orthogonalization). Two ML models (LightGBM recommended) handle the first-stage residualization: (1) propensity model predicts treatment from confounders, (2) outcome model predicts performance from confounders. Residuals are orthogonalized, then Causal Forests (EconML) estimate Conditional Average Treatment Effects (CATE) across audience segments.

### Tasks
- [ ] Create `src/omni_proof/causal/estimator.py`:
  - `DMLEstimator` class wrapping EconML's `LinearDML` and `CausalForestDML`
  - `estimate_ate(data, treatment_col, outcome_col, confounders) -> ATEResult` — Average Treatment Effect
  - `estimate_cate(data, treatment_col, outcome_col, confounders, heterogeneity_cols) -> CATEResult` — Conditional ATE across segments
  - Configurable first-stage models (LightGBM default, XGBoost option)
  - Cross-fitting with configurable folds (default 5)
- [ ] Create `src/omni_proof/causal/refuter.py`:
  - `CausalRefuter` class implementing robustness checks
  - `placebo_test(model, data) -> RefutationResult` — randomize treatment, check if effect persists (should NOT persist)
  - `subset_validation(model, data, drop_fraction=0.1) -> RefutationResult` — drop random subsets, check stability
  - `add_random_confounder(model, data) -> RefutationResult` — add noise variable, effect should not change
  - Auto-flag insights as spurious if placebo test shows significance
- [ ] Create `src/omni_proof/causal/results.py`:
  - `ATEResult`, `CATEResult`, `RefutationResult` Pydantic models
  - Confidence intervals, p-values, effect sizes
  - Segment-level breakdowns for CATE
- [ ] Write tests:
  - `tests/unit/test_estimator.py` — synthetic data with known treatment effects
  - `tests/unit/test_refuter.py` — verify placebo test catches spurious correlations

### Verification
```bash
pytest tests/unit/test_estimator.py tests/unit/test_refuter.py -v
```

### Exit Criteria
- DML recovers known treatment effects from synthetic data within CI bounds
- Placebo test correctly flags random treatment as non-significant
- CATE estimates show heterogeneity across synthetic segments
- Refutation suite passes all three checks on valid estimates

---

## Step 8: DICE-DML Visual Causal Inference

**Branch:** `feat/dice-dml`
**PR into:** `main` (after Step 7 merged)
**Depends on:** Steps 3, 7
**Model tier:** strongest (cutting-edge causal representation learning)

### Context Brief
Standard DML fails on raw visual embeddings because treatment attributes and confounders are entangled in the dense vector. DICE-DML solves this via: (1) generating deepfake counterfactuals where only the treatment attribute changes, (2) subtracting original vs. deepfake embeddings to isolate the "treatment fingerprint", (3) training the encoder adversarially to become invariant to the treatment fingerprint via orthogonal projection. The resulting disentangled vectors enable unbiased causal estimation on visual data (73-97% RMSE reduction vs. naive approaches).

### Tasks
- [ ] Create `src/omni_proof/causal/dice_dml/__init__.py`
- [ ] Create `src/omni_proof/causal/dice_dml/counterfactual_generator.py`:
  - `CounterfactualGenerator` class that creates paired images
  - `generate_counterfactual(image_path, treatment_attribute, new_value) -> CounterfactualPair`
  - Uses Gemini or external generative model to alter only the specified treatment attribute
  - Validates that background elements remain unchanged (cosine similarity check)
- [ ] Create `src/omni_proof/causal/dice_dml/disentangler.py`:
  - `TreatmentDisentangler` class
  - `extract_treatment_fingerprint(original_emb, counterfactual_emb) -> np.ndarray` — vector subtraction to isolate treatment signal
  - `orthogonal_projection(embedding, treatment_fingerprint) -> np.ndarray` — project out treatment component to get pure confounder representation
  - Training loop for adversarial invariance learning
- [ ] Create `src/omni_proof/causal/dice_dml/visual_estimator.py`:
  - `VisualDMLEstimator` extending the base `DMLEstimator`
  - Integrates disentangled representations into the DML pipeline
  - `estimate_visual_cate(embeddings, treatment_fingerprints, outcome, segments) -> CATEResult`
- [ ] Write tests:
  - `tests/unit/test_counterfactual_generator.py`
  - `tests/unit/test_disentangler.py` — verify orthogonality of projected vectors
  - `tests/unit/test_visual_estimator.py` — synthetic visual data with known effects

### Verification
```bash
pytest tests/unit/test_counterfactual_generator.py tests/unit/test_disentangler.py tests/unit/test_visual_estimator.py -v
```

### Exit Criteria
- Counterfactual pairs differ only in treatment attribute (cosine similarity > 0.95 for background)
- Orthogonal projection produces vectors with near-zero correlation to treatment fingerprint
- Visual DML estimates outperform naive embedding approach on synthetic benchmark

---

## Step 9: Orchestration Layer + Brand Compliance Pipeline

**Branch:** `feat/orchestration`
**PR into:** `main` (after Steps 6, 7 merged)
**Depends on:** Steps 6, 7
**Model tier:** default

### Context Brief
LangChain/LangGraph orchestrates two workflows: (1) Brand compliance — new creatives are embedded, matched against brand guidelines via RAG, then evaluated by Gemini for compliance (hex code checks, logo clear space, aesthetic tone). (2) Insight synthesis — causal estimates are translated into natural language design briefs by an LLM.

### Tasks
- [ ] Create `src/omni_proof/orchestration/__init__.py`
- [ ] Create `src/omni_proof/orchestration/compliance_chain.py`:
  - LangGraph state graph for brand compliance review
  - Nodes: `embed_asset` -> `retrieve_guidelines` -> `evaluate_compliance` -> `generate_report`
  - `evaluate_compliance` sends asset + retrieved guidelines to Gemini for rule-checking
  - Output: `ComplianceResult` with pass/fail, violations list, evidence from brand guide
- [ ] Create `src/omni_proof/orchestration/insight_synthesizer.py`:
  - `InsightSynthesizer` class
  - `synthesize(cate_results, audience_segments) -> DesignBrief`
  - Converts quantitative causal estimates into actionable natural language briefs
  - Example output: "Increasing video pacing in the first 3 seconds causes +12% CTR for 18-24 demographic"
- [ ] Create `src/omni_proof/orchestration/models.py`:
  - `ComplianceResult`, `Violation`, `DesignBrief` Pydantic models
- [ ] Write tests:
  - `tests/unit/test_compliance_chain.py` — mock LangGraph execution
  - `tests/unit/test_insight_synthesizer.py` — verify brief generation from sample CATE data

### Verification
```bash
pytest tests/unit/test_compliance_chain.py tests/unit/test_insight_synthesizer.py -v
```

### Exit Criteria
- Compliance pipeline correctly flags brand violations with evidence
- Insight synthesizer produces readable, actionable briefs from CATE results
- LangGraph state transitions work correctly

---

## Step 10: Dashboard API + Generative Feedback Loop

**Branch:** `feat/dashboard-api`
**PR into:** `main` (after Step 9 merged)
**Depends on:** Steps 8, 9
**Model tier:** default

### Context Brief
The application layer exposes REST/WebSocket endpoints for dashboard visualization of causal lift data. It also implements the "NextAds" closed-loop: proven causal insights + brand rules automatically parameterize content generation prompts, feeding them back into generative AI tools.

### Tasks
- [ ] Create `src/omni_proof/api/__init__.py`
- [ ] Create `src/omni_proof/api/app.py`:
  - FastAPI application with CORS, auth middleware stubs
- [ ] Create `src/omni_proof/api/routes/`:
  - `causal.py` — `GET /causal/effects` (list all estimated effects), `GET /causal/effects/{treatment}` (specific treatment CATE), `POST /causal/analyze` (trigger new analysis)
  - `compliance.py` — `POST /compliance/check` (upload creative for brand review), `GET /compliance/reports`
  - `insights.py` — `GET /insights/briefs` (latest design briefs), `GET /insights/segments` (segment-level breakdowns)
  - `generative.py` — `POST /generative/prompt` (generate optimized creative prompt from causal data + brand rules)
- [ ] Create `src/omni_proof/api/generative_loop.py`:
  - `GenerativePromptBuilder` class
  - `build_prompt(top_cate_results, brand_rules, target_segment) -> str`
  - Outputs parameterized prompts like: "Generate a 16:9 lifestyle image, warm tone, product in upper-right quadrant"
- [ ] Write tests:
  - `tests/unit/test_api_routes.py` — FastAPI TestClient tests
  - `tests/unit/test_generative_loop.py`

### Verification
```bash
pytest tests/unit/test_api_routes.py tests/unit/test_generative_loop.py -v
```

### Exit Criteria
- All API endpoints return correct response schemas
- Generative prompt builder produces valid, parameterized prompts
- Compliance endpoint correctly invokes the compliance chain

---

## Step 11: Integration Tests + End-to-End Validation

**Branch:** `feat/integration-tests`
**PR into:** `main` (after Step 10 merged)
**Depends on:** Step 10
**Model tier:** strongest (validation quality)

### Context Brief
Full pipeline validation: ingest a sample creative -> extract metadata -> generate embedding -> store -> run causal analysis -> generate insights -> check brand compliance -> produce generative prompt. Uses synthetic data with known ground-truth causal effects to validate the entire chain.

### Tasks
- [ ] Create `tests/integration/test_full_pipeline.py`:
  - End-to-end test with synthetic ad dataset (known treatment effects)
  - Validate: ingestion -> storage -> causal estimation -> insight synthesis
  - Assert recovered treatment effects are within expected CI bounds
- [ ] Create `tests/integration/test_brand_compliance_e2e.py`:
  - Upload compliant + non-compliant creatives
  - Verify correct pass/fail classifications
- [ ] Create `tests/integration/test_generative_loop_e2e.py`:
  - Feed known CATE results + brand rules
  - Verify generated prompts contain correct parameters
- [ ] Create `tests/fixtures/`:
  - `synthetic_ads.py` — generate synthetic ad metadata with planted treatment effects
  - `sample_brand_guide.json` — mock brand guidelines
- [ ] Performance benchmarks:
  - Embedding generation latency
  - Causal estimation time for N=1000, 10000, 100000 records

### Verification
```bash
pytest tests/integration/ -v --tb=long
```

### Exit Criteria
- Full pipeline recovers planted treatment effects from synthetic data
- Brand compliance correctly classifies 100% of test cases
- All integration tests pass
- Performance benchmarks documented

---

## Step 12: Documentation + Deployment Configuration

**Branch:** `feat/docs-deploy`
**PR into:** `main` (after Step 11 merged)
**Depends on:** Step 11
**Model tier:** default

### Tasks
- [ ] Create `README.md` — project overview, architecture diagram (Mermaid), quick start, API docs link
- [ ] Create `docs/architecture.md` — detailed 5-layer architecture description
- [ ] Create `docs/causal-methodology.md` — explanation of DML, DICE-DML, DAG construction for non-technical stakeholders
- [ ] Create `Dockerfile` + `docker-compose.yml`:
  - API service
  - Pinecone vector DB
  - PostgreSQL for relational storage
- [ ] Create `CLAUDE.md` — project conventions, testing commands, architecture summary for AI assistants
- [ ] Create `.github/workflows/ci.yml` — lint, type-check, unit tests, integration tests

### Verification
```bash
docker-compose build && docker-compose up -d && curl http://localhost:8000/health && docker-compose down
```

### Exit Criteria
- Docker build succeeds
- Health endpoint responds
- CI workflow runs all test suites
- Documentation covers all 5 architectural layers

---

## Anti-Pattern Checklist

| Anti-Pattern | Mitigation |
|:---|:---|
| Treating embeddings as causal features directly | DICE-DML disentangles treatment from confounders in Step 8 |
| Including colliders in adjustment set | DAG builder warns about colliders in Step 5 |
| Skipping robustness checks | Refuter runs 3 checks automatically in Step 7 |
| Using Gemini structured output without schema validation | Pydantic enforces types in Step 2 |
| Entangled vector search for brand compliance | Separate brand knowledge base with metadata filtering in Step 6 |
| Optimizing on correlations instead of causal effects | DML + refutation gates all insights before dashboard in Steps 7-8 |

---

## Plan Mutation Protocol

To modify this plan:
1. **Split a step:** Create Step N.1, N.2 with updated dependency edges
2. **Insert a step:** Add between existing steps, update all downstream dependencies
3. **Skip a step:** Mark as `SKIPPED` with reason, verify no downstream breakage
4. **Reorder:** Only if dependency graph allows — verify no unmet dependencies
5. **Abandon:** Mark as `ABANDONED`, document reason, ensure partial state is clean

All mutations must be documented with date and reason in this file.
