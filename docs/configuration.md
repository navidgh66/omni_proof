# Configuration Reference

## Environment Variables

All environment variables use the `OMNI_PROOF_` prefix. They can be set in your shell, a `.env` file, or any secrets manager.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OMNI_PROOF_GEMINI_API_KEY` | For ingestion | `""` | Google Gemini API key |
| `OMNI_PROOF_PINECONE_API_KEY` | For vector storage | `""` | Pinecone API key |
| `OMNI_PROOF_PINECONE_INDEX_HOST` | For vector storage | `""` | Pinecone index URL (e.g., `https://your-index-abc123.svc.pinecone.io`) |
| `OMNI_PROOF_DATABASE_URL` | No | `sqlite+aiosqlite:///./omni_proof.db` | SQLAlchemy async connection string |
| `OMNI_PROOF_EMBEDDING_DIMENSIONS` | No | `3072` | Embedding output dimensions |
| `OMNI_PROOF_LOG_LEVEL` | No | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `OMNI_PROOF_CORS_ALLOWED_ORIGINS` | No | `["http://localhost:3000", "http://localhost:8000"]` | Allowed CORS origins (JSON array) |

## Settings Class

Settings are loaded via Pydantic's `BaseSettings`, which reads from environment variables and `.env` files automatically:

```python
from omni_proof import Settings

# Reads OMNI_PROOF_* from environment / .env
settings = Settings()

# Or override programmatically
settings = Settings(
    gemini_api_key="AIza...",
    database_url="postgresql+asyncpg://user:pass@localhost:5432/omni_proof",
)
```

## Database Configuration

### SQLite (Default, Development)

```bash
OMNI_PROOF_DATABASE_URL=sqlite+aiosqlite:///./omni_proof.db
```

### PostgreSQL (Production)

```bash
OMNI_PROOF_DATABASE_URL=postgresql+asyncpg://user:password@host:5432/omni_proof
```

Requires `asyncpg` to be installed:

```bash
pip install asyncpg
```

## Pinecone Setup

1. Create a Pinecone account at [pinecone.io](https://www.pinecone.io/)
2. Create a serverless index:
   - **Dimensions:** 3072 (matching Gemini Embedding 2 output)
   - **Metric:** cosine
   - **Cloud/Region:** any (e.g., `aws/us-east-1`)
3. Set the environment variables:

```bash
OMNI_PROOF_PINECONE_API_KEY=pcsk_...
OMNI_PROOF_PINECONE_INDEX_HOST=https://your-index-abc123.svc.pinecone.io
```

The store uses two namespaces within the same index:

| Namespace | Content |
|-----------|---------|
| `creatives` | Creative asset embeddings for similarity search |
| `brand_assets` | Brand guidelines, approved creatives, and palettes for RAG |

## Gemini API Setup

1. Get an API key from [Google AI Studio](https://aistudio.google.com/apikey)
2. Set the environment variable:

```bash
OMNI_PROOF_GEMINI_API_KEY=AIza...
```

OmniProof uses two Gemini models:

| Model | Purpose | Used By |
|-------|---------|---------|
| `gemini-embedding-2-preview` | 3072-dim multimodal embeddings | `GeminiClient.generate_embedding()` |
| `gemini-3.1-flash-lite-preview` | Structured metadata extraction | `GeminiClient.extract_metadata()` |

## Constants

Defined in `src/omni_proof/config/constants.py`:

| Constant | Value | Description |
|----------|-------|-------------|
| `DEFAULT_EMBEDDING_DIMS` | `3072` | Default embedding output dimensions |
| `MATRYOSHKA_DIMS` | `[128, 768, 1536, 3072]` | Valid Matryoshka dimension options |
| `MAX_VIDEO_SECONDS_WITH_AUDIO` | `80` | Gemini video limit (with audio) |
| `MAX_VIDEO_SECONDS_NO_AUDIO` | `120` | Gemini video limit (without audio) |
| `MAX_AUDIO_SECONDS` | `80` | Gemini audio limit |
| `MAX_IMAGES_PER_REQUEST` | `6` | Gemini images per embedding request |
| `MAX_PDF_PAGES` | `6` | Gemini PDF pages per request |
| `MAX_TEXT_TOKENS` | `8192` | Gemini text token limit |

## Docker Compose

The included `docker-compose.yml` starts the API server with PostgreSQL:

```bash
docker-compose up -d
```

Set environment variables in a `.env` file or pass them directly:

```bash
OMNI_PROOF_GEMINI_API_KEY=AIza... \
OMNI_PROOF_PINECONE_API_KEY=pcsk_... \
docker-compose up -d
```
