# API Reference

Complete reference for all public classes and methods in OmniProof.

## Public Exports

```python
from omni_proof import (
    BrandExtractor,
    BrandProfile,
    ComplianceChain,
    DMLEstimator,
    EmbeddingProvider,
    Estimator,
    GeminiClient,
    InMemoryVectorStore,
    InsightSynthesizer,
    PineconeVectorStore,
    Settings,
    VectorStore,
)
```

---

## Core Interfaces

### `EmbeddingProvider`

Abstract base class for embedding backends.

```python
class EmbeddingProvider(ABC):
    async def generate_embedding(
        self,
        content: str | Path,
        dimensions: int = 3072,
        task_type: str | None = None,
    ) -> list[float]
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `content` | `str \| Path` | — | Text string or path to a media file (image, video, audio, PDF) |
| `dimensions` | `int` | `3072` | Output dimensionality. Matryoshka options: 128, 768, 1536, 3072 |
| `task_type` | `str \| None` | `None` | Gemini task type (e.g., `"SEMANTIC_SIMILARITY"`, `"RETRIEVAL_DOCUMENT"`) |

**Returns:** `list[float]` — embedding vector of the specified dimensionality.

**Raises:** `ValueError` if dimensions not in supported set.

---

### `VectorStore`

Abstract base class for vector storage backends.

```python
class VectorStore(ABC):
    async def upsert(
        self,
        asset_id: str,
        embedding: list[float],
        metadata: dict,
        namespace: str | None = None,
    ) -> None

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict | None = None,
        namespace: str | None = None,
    ) -> list[dict]

    async def delete(
        self,
        asset_id: str,
        namespace: str | None = None,
    ) -> None

    async def upsert_batch(
        self,
        vectors: list[tuple[str, list[float], dict]],
        namespace: str | None = None,
        batch_size: int = 100,
    ) -> None
```

**`search` return format:**

```python
[
    {"id": "asset-001", "score": 0.95, "metadata": {...}},
    {"id": "asset-002", "score": 0.89, "metadata": {...}},
]
```

---

### `Estimator`

Abstract base class for causal effect estimators.

```python
class Estimator(ABC):
    def estimate_ate(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        confounder_cols: list[str],
    ) -> ATEResult

    def estimate_cate(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        confounder_cols: list[str],
        segment_col: str,
    ) -> CATEResult
```

---

## Ingestion Layer

### `GeminiClient`

Wraps Gemini API calls with retry logic and rate limiting. Implements `EmbeddingProvider`.

```python
class GeminiClient(EmbeddingProvider):
    def __init__(self, api_key: str, max_retries: int = 3)

    async def generate_embedding(
        self,
        content: str | Path,
        dimensions: int = 3072,
        task_type: str | None = None,
    ) -> list[float]

    async def extract_metadata(
        self,
        asset_path: Path,
        schema: type,
    ) -> Any
```

| Method | Description |
|--------|-------------|
| `generate_embedding` | Generate embedding via Gemini Embedding 2 (`gemini-embedding-2-preview`). Retries up to `max_retries` on transient failures. |
| `extract_metadata` | Extract structured metadata via Gemini 3.1 Flash Lite (`gemini-3.1-flash-lite-preview`). Returns an instance of the provided Pydantic `schema` type. |

---

### `AssetPreprocessor`

Handles Gemini API input limits by batching and chunking.

```python
class AssetPreprocessor:
    def batch_images(
        self,
        image_paths: list[Path],
        batch_size: int = 6,
    ) -> list[list[Path]]

    def segment_pdf_by_count(
        self,
        total_pages: int,
        max_pages: int = 6,
    ) -> list[tuple[int, int]]

    def compute_video_chunks(
        self,
        duration_seconds: float,
        has_audio: bool = True,
    ) -> list[tuple[float, float]]

    def compute_audio_chunks(
        self,
        duration_seconds: float,
        max_seconds: int = 80,
    ) -> list[tuple[float, float]]
```

**Gemini Embedding 2 limits:**

| Modality | Limit |
|----------|-------|
| Video (with audio) | 80 seconds |
| Video (without audio) | 120 seconds |
| Audio | 80 seconds |
| Images | 6 per request |
| PDF | 1 document, 6 pages |
| Output dimensions | 128, 768, 1536, or 3072 (Matryoshka) |

---

### `IngestPipeline`

Orchestrates parallel metadata extraction and embedding generation.

```python
class IngestPipeline:
    def __init__(self, gemini_client: GeminiClient)

    async def ingest(
        self,
        asset_path: Path,
        schema: type,
    ) -> tuple

    async def ingest_batch(
        self,
        asset_paths: list[Path],
        schema: type,
    ) -> list[tuple]
```

`ingest` runs embedding and metadata extraction concurrently via `asyncio.gather`. `ingest_batch` processes multiple assets with graceful failure handling (individual failures don't abort the batch).

---

## Storage Layer

### `PineconeVectorStore`

Production vector store backed by Pinecone serverless.

```python
class PineconeVectorStore(VectorStore):
    def __init__(
        self,
        index,                              # pinecone.Index instance
        default_namespace: str = "creatives",
    )
```

**Namespaces:** `"creatives"` for creative asset embeddings, `"brand_assets"` for brand guidelines and approved creatives.

---

### `InMemoryVectorStore`

Development and testing vector store using numpy cosine similarity.

```python
class InMemoryVectorStore(VectorStore):
    def __init__(self, default_namespace: str = "creatives")
```

Same interface as `PineconeVectorStore`. Stores vectors in memory as numpy arrays.

---

### `RelationalStore`

Async SQLAlchemy 2.0 store for structured performance data.

```python
class RelationalStore:
    def __init__(self, database_url: str)

    async def initialize(self) -> None
    async def create_creative_metadata(self, data: dict) -> None
    async def get_creative_metadata(self, asset_id: str) -> dict | None
    async def create_performance_record(self, data: dict) -> None
    async def get_performance_by_asset(self, asset_id: str) -> list[dict]
    async def get_causal_data_matrix(self) -> list[dict]
```

**ORM models:**

| Model | Table | Key Columns |
|-------|-------|-------------|
| `CreativeMetadataRecord` | `creative_metadata` | `asset_id` (PK), visual/temporal/textual/auditory fields |
| `PerformanceRecord` | `performance_records` | `id` (PK), `asset_id` (FK), impressions, clicks, conversions, CTR, ROAS |
| `CampaignRecord` | `campaigns` | `campaign_id` (PK), name, dates, budget, demographics |

---

## Causal Engine

### `CausalDAGBuilder`

Constructs DoWhy causal models.

```python
class CausalDAGBuilder:
    def build_dag(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        confounders: list[str],
        effect_modifiers: list[str] | None = None,
        colliders: list[str] | None = None,
    ) -> CausalModel

    def build_from_template(
        self,
        data: pd.DataFrame,
        template_name: str,
    ) -> CausalModel
```

**Built-in templates:**

| Template | Treatment | Outcome | Confounders |
|----------|-----------|---------|-------------|
| `logo_timing` | `logo_in_first_3s` | `ctr` | platform, audience_segment, season, budget |
| `color_temperature` | `warm_color_palette` | `engagement_rate` | product_category, production_quality, platform |
| `audio_pacing` | `fast_audio_pacing` | `conversion_rate` | audience_segment, time_of_day, platform, budget |

---

### `DMLEstimator`

Double Machine Learning estimator wrapping EconML.

```python
class DMLEstimator(Estimator):
    def __init__(self, cv: int = 5, n_estimators: int = 50)

    def estimate_ate(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        confounder_cols: list[str],
    ) -> ATEResult

    def estimate_cate(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        confounder_cols: list[str],
        segment_col: str,
    ) -> CATEResult
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `cv` | `int` | `5` | Number of cross-validation folds |
| `n_estimators` | `int` | `50` | Number of trees in the LightGBM first-stage models |

---

### `CausalRefuter`

Robustness tests for causal estimates.

```python
class CausalRefuter:
    def __init__(self, cv: int = 3)

    def placebo_test(
        self, data, treatment_col, outcome_col, confounder_cols,
    ) -> RefutationResult

    def subset_test(
        self, data, treatment_col, outcome_col, confounder_cols,
        drop_fraction: float = 0.1,
    ) -> RefutationResult

    def random_confounder_test(
        self, data, treatment_col, outcome_col, confounder_cols,
    ) -> RefutationResult
```

| Test | What It Does | Pass Condition |
|------|-------------|----------------|
| `placebo_test` | Shuffles treatment, re-estimates ATE | Effect diminishes toward zero |
| `subset_test` | Drops 10% of data, re-estimates | ATE remains stable |
| `random_confounder_test` | Adds random noise variable as confounder | No significant change in ATE |

---

### DICE-DML

#### `CounterfactualGenerator`

```python
class CounterfactualGenerator:
    def __init__(self, gemini_client: EmbeddingProvider)

    async def generate(
        self,
        original_path: Path,
        counterfactual_path: Path,
        treatment_attr: str,
    ) -> CounterfactualPair
```

Returns a `CounterfactualPair` with `original_emb`, `counterfactual_emb`, and `background_similarity` (cosine, should be ~1.0).

#### `TreatmentDisentangler`

```python
class TreatmentDisentangler:
    def extract_treatment_fingerprint(
        self,
        original_emb: np.ndarray,        # (3072,)
        counterfactual_emb: np.ndarray,   # (3072,)
    ) -> np.ndarray                       # (3072,) L2-normalized

    def orthogonal_projection(
        self,
        embedding: np.ndarray,            # (3072,)
        treatment_fingerprint: np.ndarray, # (3072,)
    ) -> np.ndarray                       # (3072,)

    def disentangle_batch(
        self,
        embeddings: np.ndarray,           # (n, 3072)
        treatment_fingerprint: np.ndarray, # (3072,)
    ) -> np.ndarray                       # (n, 3072)
```

#### `VisualDMLEstimator`

```python
class VisualDMLEstimator:
    def __init__(self, cv: int = 3)

    def estimate_visual_ate(
        self,
        embeddings: np.ndarray,             # (n, 3072)
        treatment: np.ndarray,              # (n,)
        outcome: np.ndarray,                # (n,)
        treatment_fingerprint: np.ndarray,  # (3072,)
    ) -> ATEResult
```

---

## Brand Extraction

### `BrandExtractor`

Orchestrates multi-asset brand identity extraction.

```python
class BrandExtractor:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        gemini_client: GeminiClient,
        vector_store: VectorStore,
    )

    async def extract(
        self,
        brand_name: str,
        assets: list[Path],
    ) -> BrandProfile

    async def update(
        self,
        profile: BrandProfile,
        new_assets: list[Path],
    ) -> tuple[BrandProfile, list[BrandConflict]]
```

**`extract`** processes all assets in parallel, aggregates patterns, computes confidence scores, and returns a full `BrandProfile`.

**`update`** processes new assets, merges with existing profile, and returns the updated profile plus any detected conflicts.

---

### `BrandProfile`

```python
class BrandProfile(BaseModel):
    profile_id: str
    brand_name: str
    rules: list[BrandRule]
    voice: BrandVoice
    visual_style: BrandVisualStyle
    visual_fingerprint: list[float]       # L2-normalized mean embedding
    source_assets: list[str]
    extractions: list[AssetExtraction]
    confidence_scores: dict[str, float]   # {"color": 0.9, "typography": 0.8, ...}
    created_at: datetime
    updated_at: datetime
```

---

## Orchestration

### `ComplianceChain`

```python
class ComplianceChain:
    def __init__(
        self,
        gemini_client: EmbeddingProvider,
        brand_retriever: BrandRetriever,
        evaluator: Callable | None = None,
    )

    async def check_compliance(
        self,
        asset_id: str,
        asset_path: str | Path,
    ) -> ComplianceReport
```

**`ComplianceReport`:**

```python
class ComplianceReport(BaseModel):
    asset_id: str
    passed: bool
    violations: list[Violation]
    evidence_sources: list[str]
    score: float                  # 1.0 = compliant, 0.0 = non-compliant
```

**`Violation`:**

```python
class Violation(BaseModel):
    rule_type: str    # "concrete" or "semantic"
    severity: str     # "critical", "warning", "info"
    description: str
    evidence: str
```

---

### `InsightSynthesizer`

```python
class InsightSynthesizer:
    def __init__(
        self,
        p_value_threshold: float = 0.05,
        recommend_threshold: float = 0.05,
    )

    def synthesize(self, cate_result: CATEResult) -> DesignBrief
```

**`DesignBrief`:**

```python
class DesignBrief(BaseModel):
    treatment: str
    finding: str
    segment_breakdown: dict[str, str]   # {"Gen-Z": "+3.2% — RECOMMENDED"}
    recommendation: str
    confidence: str                     # "HIGH" or "LOW"
```

**Classification logic:**

| Condition | Label |
|-----------|-------|
| `p_value > threshold` | NEUTRAL |
| `effect > recommend_threshold` | RECOMMENDED |
| `effect > 0` | CONSIDER |
| `effect <= 0` | AVOID |

---

## RAG System

### `BrandIndexer`

```python
class BrandIndexer:
    def __init__(self, gemini_client: EmbeddingProvider, vector_store: VectorStore)

    async def index_brand_guide_page(
        self, page_id: str, page_content_path: str | Path,
        section_type: str, page_number: int,
    ) -> None

    async def index_approved_creative(
        self, asset_id: str, asset_path: str | Path, tags: list[str],
    ) -> None

    async def index_color_palette(
        self, palette_id: str, hex_codes: list[str], palette_name: str,
    ) -> None
```

### `BrandRetriever`

```python
class BrandRetriever:
    def __init__(self, gemini_client: EmbeddingProvider, vector_store: VectorStore)

    async def search_by_text(self, query: str, top_k: int = 10) -> list[BrandAsset]
    async def search_by_image(self, image_path: Path, top_k: int = 10) -> list[BrandAsset]
    async def get_guidelines_for_asset(self, asset_path: Path, top_k: int = 10) -> list[BrandAsset]
```

---

## Result Types

### Causal Results

```python
class ATEResult(BaseModel):
    treatment: str
    outcome: str
    ate: float          # Average Treatment Effect
    ci_lower: float     # 95% CI lower bound
    ci_upper: float     # 95% CI upper bound
    p_value: float
    n_samples: int

class CATEResult(BaseModel):
    treatment: str
    outcome: str
    segments: dict[str, EffectEstimate]
    refutation_passed: bool

class EffectEstimate(BaseModel):
    effect: float
    ci_lower: float
    ci_upper: float
    p_value: float

class RefutationResult(BaseModel):
    test_name: str
    original_effect: float
    new_effect: float
    passed: bool
    p_value: float
```

### Ingestion Schemas

```python
class CreativeMetadata(BaseModel):
    asset_id: UUID
    campaign_id: UUID
    platform: str
    timestamp: datetime
    visual: VisualElements
    temporal: TemporalPacing
    textual: TextualElements
    auditory: AuditoryElements
```

See `src/omni_proof/ingestion/schemas.py` for the full field reference of each sub-model.

---

## Configuration

### `Settings`

```python
class Settings(BaseSettings):
    gemini_api_key: str = ""
    pinecone_api_key: str = ""
    pinecone_index_host: str = ""
    database_url: str = "sqlite+aiosqlite:///./omni_proof.db"
    embedding_dimensions: int = 3072
    log_level: str = "INFO"
    cors_allowed_origins: list[str] = ["http://localhost:3000", "http://localhost:8000"]

    model_config = {"env_prefix": "OMNI_PROOF_", "env_file": ".env"}
```

See [Configuration Reference](configuration.md) for full details.
