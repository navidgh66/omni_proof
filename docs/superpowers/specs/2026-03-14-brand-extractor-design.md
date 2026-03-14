# BrandExtractor: Multimodal Brand Identity Extraction Pipeline

## Overview

A user-triggered pipeline that extracts structured brand guidelines and identity profiles from collections of multimodal assets (PDFs, images, videos, audio) using Gemini Embedding 2's native multimodal capabilities and Gemini Flash structured extraction.

## Usage Modes

OmniProof is a modular toolkit. Each capability is independently usable:

| Mode | Input | Output | Layers Used |
|------|-------|--------|-------------|
| **Performance Insights** | Creatives + performance data | Causal effects, design briefs | Causal, Orchestration |
| **Creative Generation** | Performance insights (optional) + brand rules (optional) | LLM-generated creative prompts | Orchestration |
| **Brand Extraction** | Brand assets (docs, images, video, audio) | BrandProfile | Brand Extraction |
| **Full Pipeline** | All of the above | Insights + compliance + brand-aware generation | All layers |

Any combination is valid. The API presents all options to the user. No capability assumes another is active.

## Architecture

Three-stage pipeline, user-triggered via two operations: `extract` (fresh) and `update` (incremental with conflict detection).

```
Stage 1: Asset Processing (AssetProcessor)
  User uploads assets (PDFs, images, videos, audio)
    -> AssetPreprocessor (existing) computes chunk boundaries
    -> AssetProcessor handles actual file content preparation per media type
    -> Gemini Flash: structured extraction per asset (BrandAssetMetadata schema)
    -> Gemini Embedding 2: multimodal embedding per asset
       (task_type parameter for brand-optimized embeddings)
    -> Returns list[AssetExtraction]

Stage 2: Pattern Aggregation (PatternAggregator)
  list[AssetExtraction]
    -> Aggregate by dimension (color, typography, voice, visual style)
    -> Compute frequency/consistency scores per dimension
    -> Identify dominant patterns -> BrandRule objects
    -> Compute centroid embedding (visual fingerprint)
    -> Detect outlier assets (far from centroid)
    -> Returns aggregated patterns + rules + fingerprint + confidence scores

Stage 3: Profile Assembly (BrandExtractor orchestrator)
  Aggregated output
    -> Assemble BrandProfile
    -> Index rules + fingerprint into vector store
    -> Persist extractions to relational store (for update flow)
    -> Return BrandProfile

Update flow adds:
  Stage 2.5: Conflict Detection (ConflictDetector)
    -> Compare new patterns against existing BrandProfile
    -> Flag contradictions as BrandConflict objects
    -> Return updated profile + conflicts for user resolution
```

## Prerequisites: GeminiClient Enhancements

The current `GeminiClient` needs two changes before this pipeline works:

### 1. File upload support for multimodal embedding

Current `generate_embedding` sends `contents=str(content)` which just embeds the path string. For multimodal assets (images, video, audio, PDF), the Gemini API requires actual file content. The method must detect if `content` is a file path and upload the file via `client.files.upload()` before embedding.

```python
async def generate_embedding(
    self, content: str | Path, dimensions: int = DEFAULT_EMBEDDING_DIMS,
    task_type: str | None = None,
) -> list[float]:
    # If content is a Path to a file, upload it first
    # Pass task_type to config if provided
```

### 2. Task type parameter for EmbeddingProvider

Add optional `task_type: str | None = None` to both `EmbeddingProvider.generate_embedding` and `GeminiClient.generate_embedding`. This enables Gemini Embedding 2's task-specific optimization (e.g., `"RETRIEVAL_DOCUMENT"`, `"SEMANTIC_SIMILARITY"`).

The `EmbeddingProvider` ABC update:
```python
class EmbeddingProvider(ABC):
    @abstractmethod
    async def generate_embedding(
        self, content: str | Path, dimensions: int = 3072,
        task_type: str | None = None,
    ) -> list[float]: ...
```

These are implementation prerequisites, not part of the brand extraction module itself.

## Data Models

### New module: `src/omni_proof/brand_extraction/models.py`

```python
class BrandColorInfo(BaseModel):
    """Per-asset color extraction from Gemini Flash."""
    hex_codes: list[str]
    palette_mood: str  # "warm", "cool", "neutral", "vibrant"

class BrandTypographyInfo(BaseModel):
    """Per-asset typography extraction."""
    font_styles: list[str]   # "serif", "sans-serif", "script", "monospace"
    font_names: list[str]    # detected font names if identifiable
    text_hierarchy: str      # "strong_hierarchy", "flat", "mixed"

class BrandToneInfo(BaseModel):
    """Per-asset tone/voice extraction."""
    formality: str            # "formal", "casual", "mixed"
    emotional_register: str   # "inspiring", "authoritative", "playful", etc.
    key_phrases: list[str]
    vocabulary_themes: list[str]

class BrandVisualInfo(BaseModel):
    """Per-asset visual style extraction."""
    layout_pattern: str       # "centered", "grid", "asymmetric", "full_bleed"
    motion_intensity: str     # "static", "slow", "dynamic"
    dominant_objects: list[str]

class BrandAssetMetadata(BaseModel):
    """Schema for Gemini Flash structured extraction of brand attributes."""
    asset_description: str
    colors: BrandColorInfo
    typography: BrandTypographyInfo
    tone: BrandToneInfo
    visual: BrandVisualInfo
    logo_detected: bool
    media_type_detected: str

class AssetExtraction(BaseModel):
    """Result of processing a single asset through Stage 1."""
    asset_path: str
    media_type: str           # "image", "video", "audio", "pdf"
    embedding: list[float]
    structured_metadata: BrandAssetMetadata
    extracted_at: datetime

class BrandVoice(BaseModel):
    """Aggregated voice profile across all assets."""
    formality: str
    emotional_register: str
    vocabulary_themes: list[str]
    sentence_style: str       # "short_punchy", "long_descriptive", "mixed"
    confidence: float

class BrandVisualStyle(BaseModel):
    """Aggregated visual style across all assets."""
    dominant_colors: list[str]
    color_consistency: float
    typography_styles: list[str]
    layout_patterns: list[str]
    motion_style: str
    confidence: float

class BrandConflict(BaseModel):
    """A detected conflict between new assets and existing profile."""
    dimension: str            # "color_palette", "typography", "tone"
    existing_value: str
    new_value: str
    source_assets: list[str]
    severity: str             # "major" (contradicts), "minor" (extends)

class BrandProfile(BaseModel):
    """The complete extracted brand identity."""
    profile_id: str
    brand_name: str
    rules: list[BrandRule]    # reuses existing rag.models.BrandRule
    voice: BrandVoice
    visual_style: BrandVisualStyle
    visual_fingerprint: list[float]   # centroid embedding
    source_assets: list[str]
    extractions: list[AssetExtraction]  # persisted for update flow
    confidence_scores: dict[str, float]  # per-dimension
    created_at: datetime
    updated_at: datetime
```

Note: `BrandProfile.extractions` stores the original per-asset extractions. This is needed for the `update` flow — when new assets arrive, the aggregator re-aggregates over all extractions (old + new) to produce updated patterns. The full `BrandProfile` including extractions is serialized to the relational store as JSON.

## Components

### 5 files in `src/omni_proof/brand_extraction/`

### 1. `asset_processor.py` — AssetProcessor

Processes individual assets through Gemini Flash + Embedding 2.

```python
class AssetProcessor:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        gemini_client: GeminiClient,  # needs extract_metadata
    ): ...

    async def process(self, asset_path: Path) -> AssetExtraction:
        """Process single asset: structured extraction + embedding in parallel."""
        ...

    async def process_batch(self, assets: list[Path]) -> list[AssetExtraction]:
        """Process multiple assets, skip failures. Raises IngestionError if ALL fail."""
        ...
```

- Uses existing `AssetPreprocessor` to compute chunk boundaries
- Handles file content preparation per media type:
  - Images: pass path directly (Gemini handles images natively)
  - PDFs: if >6 pages, process first 6 pages (Gemini limit), log warning
  - Video: if >80s with audio / >120s without, process first chunk only, log warning
  - Audio: if >80s, process first 80s only, log warning
- Runs `extract_metadata(asset, BrandAssetMetadata)` and `generate_embedding(asset, task_type="SEMANTIC_SIMILARITY")` in parallel per asset
- Detects media type from file extension
- If ALL assets in a batch fail, raises `IngestionError` rather than returning empty list

### 2. `pattern_aggregator.py` — PatternAggregator

Pure logic, no external calls.

```python
class PatternAggregator:
    def aggregate(
        self, extractions: list[AssetExtraction],
    ) -> tuple[list[BrandRule], BrandVoice, BrandVisualStyle, list[float], dict[str, float]]:
        """Returns (rules, voice, visual_style, fingerprint, confidence_scores).

        Raises ValueError if extractions is empty.
        """
        ...
```

Aggregation logic per dimension:

- **Colors**: Frequency-count hex codes across all assets. Top N become dominant. Consistency = fraction of assets containing dominant colors.
- **Typography**: Collect font styles/names, find most common. Generate BrandRule with approved fonts.
- **Voice**: Mode of formality/emotional_register, union of vocabulary themes. Consistency = agreement ratio.
- **Visual fingerprint**: Mean of all asset embeddings, L2-normalized. This is the brand's "vibe" vector.
- **Confidence scores**: Per dimension, `agreements / total_assets`. High = consistent brand, low = fragmented.
- **Outlier detection**: Assets whose embedding cosine similarity to centroid < 0.7 are flagged in logs.

Each detected pattern becomes a `BrandRule` with:
- `rule_id`: auto-generated UUID
- `section_type`: the dimension ("color_palette", "typography", "tone", "visual_style")
- `description`: human-readable rule text generated from the aggregated data
- `hex_codes`, `approved_fonts`, `tone_keywords`: populated from aggregation

### 3. `conflict_detector.py` — ConflictDetector

Pure logic. Compares new patterns against existing profile.

```python
class ConflictDetector:
    def detect(
        self,
        existing_profile: BrandProfile,
        new_rules: list[BrandRule],
        new_voice: BrandVoice,
        new_visual_style: BrandVisualStyle,
    ) -> list[BrandConflict]:
        ...
```

Conflict detection rules:
- **Color**: New dominant colors not in existing `dominant_colors` -> major if completely different palette, minor if additive
- **Typography**: New font styles not in existing `typography_styles` -> major
- **Voice**: Formality or emotional register changed -> major. New vocabulary themes -> minor.
- **Visual**: Layout pattern shift -> minor. Motion style change -> minor.

### 4. `extractor.py` — BrandExtractor

Orchestrator with two public methods.

```python
class BrandExtractor:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        gemini_client: GeminiClient,
        vector_store: VectorStore,
    ): ...

    async def extract(
        self,
        brand_name: str,
        assets: list[Path],
    ) -> BrandProfile:
        """Fresh extraction: process all assets, aggregate, assemble profile, index."""
        ...

    async def update(
        self,
        profile: BrandProfile,
        new_assets: list[Path],
    ) -> tuple[BrandProfile, list[BrandConflict]]:
        """Incremental update: process new assets, detect conflicts, return for user review."""
        ...
```

`extract` flow:
1. `AssetProcessor.process_batch(assets)` -> extractions
2. `PatternAggregator.aggregate(extractions)` -> rules, voice, visual_style, fingerprint, confidence
3. Assemble `BrandProfile` (with extractions stored for future updates)
4. Index each rule into vector store (namespace `"brand_assets"`) using embedding_provider
5. Return profile

`update` flow:
1. `AssetProcessor.process_batch(new_assets)` -> new_extractions
2. Combine: `all_extractions = profile.extractions + new_extractions`
3. `PatternAggregator.aggregate(all_extractions)` -> updated patterns
4. `ConflictDetector.detect(profile, updated_patterns)` -> conflicts
5. Assemble updated `BrandProfile` (new source_assets appended, updated_at refreshed, all extractions stored)
6. Return (updated_profile, conflicts) — user reviews conflicts, then calls a commit method to index

### 5. API Routes — `src/omni_proof/api/routes/brand.py`

```python
router = APIRouter()

@router.post("/extract")
async def extract_brand(
    brand_name: str = Form(...),
    files: list[UploadFile] = File(...),
    settings: Settings = Depends(get_settings),
) -> BrandProfileResponse:
    """Upload assets and extract a new BrandProfile."""
    ...

@router.post("/update/{profile_id}")
async def update_brand(
    profile_id: str,
    files: list[UploadFile] = File(...),
    settings: Settings = Depends(get_settings),
) -> BrandUpdateResponse:
    """Upload new assets and get updated profile + conflicts."""
    ...

@router.get("/profile/{profile_id}")
async def get_profile(profile_id: str) -> BrandProfileResponse:
    """Retrieve an existing BrandProfile."""
    ...
```

Response models:
```python
class BrandProfileResponse(BaseModel):
    profile: BrandProfile
    outlier_assets: list[str]  # assets flagged as inconsistent

class BrandUpdateResponse(BaseModel):
    updated_profile: BrandProfile
    conflicts: list[BrandConflict]
    outlier_assets: list[str]
```

The API route saves uploaded files to a temp directory, passes paths to `BrandExtractor`, and serializes the `BrandProfile` to the relational store as JSON.

## Gemini Embedding 2 Integration

Key capabilities leveraged:

- **Native multimodal**: PDF, image, video, audio all embedded into the same 3072-dim space. No modality-specific handling needed for embeddings.
- **Task type parameter**: Use Gemini's task type optimization for brand-relevant similarity matching.
- **Cross-modal comparison**: A PDF brand guide page and a video ad produce comparable vectors. Compliance checking can compare any modality against the brand fingerprint.
- **Matryoshka dimensions**: Use full 3072 for brand fingerprint (maximum fidelity), potentially 768 for quick similarity screening.

## Integration with Existing Layers

### Compliance checking
`ComplianceChain` can use `BrandProfile.visual_fingerprint` for fast cosine similarity pre-screening before running detailed rule-based checks via the evaluator.

### Creative generation
`GenerativePromptBuilder` receives `BrandProfile.rules` as `brand_rules` parameter and `BrandProfile.voice` to inform tone. No changes to its interface needed.

### Causal analysis
Fully independent. Does not use brand extraction at all unless the user combines them.

### Vector store
Brand rules are indexed into the `"brand_assets"` namespace. The visual fingerprint is stored as a special vector with `source_type: "brand_fingerprint"`. The full `BrandProfile` (including extractions) is serialized as JSON to the relational store for persistence and update flow support.

## Error Handling

- `AssetProcessor.process_batch`: skips individual asset failures, raises `IngestionError` if ALL assets fail
- `PatternAggregator.aggregate`: raises `ValueError` if empty extractions list
- `BrandExtractor.extract/update`: propagates errors from sub-components, wraps in `OmniProofError` subtypes
- API routes: return appropriate HTTP status codes (400 for bad input, 422 for empty batch, 500 for internal errors)

## Testing Strategy

- **Unit tests**: Each component tested independently with mock data
  - `AssetProcessor`: mock GeminiClient, verify extraction structure, verify batch failure handling
  - `PatternAggregator`: feed synthetic AssetExtractions, verify aggregation logic, confidence scores, outlier detection, empty input handling
  - `ConflictDetector`: feed existing profile + conflicting new patterns, verify conflict detection and severity classification
  - `BrandExtractor`: mock all dependencies, verify orchestration flow for both extract and update
- **Integration test**: End-to-end with `InMemoryVectorStore`, mock Gemini, verify full extract -> index -> retrieve cycle, and update -> conflict detection cycle

## File Structure

```
src/omni_proof/brand_extraction/
    __init__.py
    models.py              # BrandProfile, BrandVoice, BrandVisualStyle, etc.
    asset_processor.py     # AssetProcessor
    pattern_aggregator.py  # PatternAggregator
    conflict_detector.py   # ConflictDetector
    extractor.py           # BrandExtractor (orchestrator)

src/omni_proof/api/routes/
    brand.py               # API routes for brand extraction

tests/unit/
    test_asset_processor.py
    test_pattern_aggregator.py
    test_conflict_detector.py
    test_brand_extractor.py

tests/integration/
    test_brand_extraction_e2e.py
```
