"""Application settings loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    gemini_api_key: str = ""
    pinecone_api_key: str = ""
    pinecone_index_host: str = ""
    database_url: str = "sqlite+aiosqlite:///./omni_proof.db"
    embedding_dimensions: int = 3072
    log_level: str = "INFO"

    model_config = {"env_prefix": "OMNI_PROOF_", "env_file": ".env"}
