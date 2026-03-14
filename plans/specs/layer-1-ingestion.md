# Spec: Layer 1 — Multimodal Ingestion & Transformation

## Overview

Transforms raw ad creatives (video, image, audio, PDF) into structured metadata + dense embeddings using Gemini 2.0 and Gemini Embedding 2.

## API Contracts

### GeminiClient

```python
class GeminiClient:
    async def extract_metadata(
        self, asset_path: Path, schema: type[BaseModel]
    ) -> CreativeMetadata:
        """Send asset + Pydantic schema to Gemini 2.0 Structured Output."""

    async def generate_embedding(
        self, asset_path: Path, dimensions: int = 3072, task: str | None = None
    ) -> list[float]:
        """Call gemini-embedding-2-preview. Supports Matryoshka truncation via output_dimensionality."""
```

### Preprocessor

```python
class AssetPreprocessor:
    def chunk_video(self, video_path: Path, has_audio: bool = True) -> list[Path]:
        """Split at keyframes. Max 80s with audio, 120s without."""

    def segment_pdf(self, pdf_path: Path, max_pages: int = 6) -> list[Path]:
        """Split PDF into <= 6-page segments. 1 doc per request."""

    def batch_images(self, image_paths: list[Path], batch_size: int = 6) -> list[list[Path]]:
        """Group images into batches of 6."""
```

### IngestPipeline

```python
class IngestPipeline:
    async def ingest(self, asset_path: Path) -> tuple[CreativeMetadata, list[float]]:
        """Full pipeline: preprocess -> extract -> embed."""

    async def ingest_batch(self, asset_dir: Path) -> list[tuple[CreativeMetadata, list[float]]]:
        """Process all assets in directory."""
```

## Gemini Embedding 2 API Constraints

**Model ID:** `gemini-embedding-2-preview`
**Region:** `us-central1` only
**Pricing:** Standard PayGo only (no provisioned throughput)

| Modality | Constraint |
|:---------|:-----------|
| Text | Up to 8,192 input tokens |
| Images | Up to 6 per prompt; `image/png`, `image/jpeg` |
| Video | **80 seconds (with audio)** or **120 seconds (without audio)**; 1 per prompt; `video/mpeg`, `video/mp4` |
| Audio | **80 seconds max**; 1 file per prompt; `audio/mp3`, `audio/wav` |
| PDF | 1 document; 6 pages max |
| Output | 3,072 default dimensions (Matryoshka: truncate via `output_dimensionality`) |

**Key features:**
- Custom task instructions (e.g., `task:code retrieval`)
- Document OCR capability
- Audio track extraction from video inputs
- Native multimodal — no intermediate transcription needed

## Data Flow

```
Raw Asset (mp4/jpg/mp3/pdf)
  |
  v
Preprocessor (chunk/segment/batch)
  |
  ├──> Gemini 2.0 + Pydantic Schema ──> CreativeMetadata (JSON)
  |
  └──> gemini-embedding-2-preview ──> Dense Vector (float[3072])
                                        |
                                        v
                                   Pinecone Index (upsert)
```

## Error Handling

- 429 Rate Limit: exponential backoff (1s, 2s, 4s, 8s, max 60s)
- 400 Invalid Input: log asset path + error, skip asset, continue batch
- 500 Server Error: retry up to 3 times, then fail with structured error
