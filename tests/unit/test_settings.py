"""Tests for Settings configuration."""



from omni_proof.config.settings import Settings


class TestSettingsDefaults:
    def test_default_values(self):
        s = Settings(gemini_api_key="", pinecone_api_key="", pinecone_index_host="")
        assert s.gemini_api_key == ""
        assert s.database_url == "sqlite+aiosqlite:///./omni_proof.db"
        assert s.embedding_dimensions == 3072
        assert s.log_level == "INFO"

    def test_cors_default_is_not_wildcard(self):
        s = Settings()
        assert "*" not in s.cors_allowed_origins

    def test_custom_values(self):
        s = Settings(
            gemini_api_key="test-key",
            pinecone_api_key="pcsk_test",
            pinecone_index_host="https://test.svc.pinecone.io",
            database_url="postgresql+asyncpg://localhost/omni",
            embedding_dimensions=768,
            cors_allowed_origins=["https://myapp.com"],
        )
        assert s.gemini_api_key == "test-key"
        assert s.embedding_dimensions == 768
        assert s.cors_allowed_origins == ["https://myapp.com"]

    def test_env_var_loading(self, monkeypatch):
        monkeypatch.setenv("OMNI_PROOF_GEMINI_API_KEY", "from-env")
        monkeypatch.setenv("OMNI_PROOF_LOG_LEVEL", "DEBUG")
        s = Settings()
        assert s.gemini_api_key == "from-env"
        assert s.log_level == "DEBUG"
