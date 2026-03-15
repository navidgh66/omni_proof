"""E2E tests for API error handling and CORS."""

import pytest
from fastapi.testclient import TestClient

from omni_proof.api.app import create_app
from omni_proof.config.settings import Settings


@pytest.fixture
def client():
    return TestClient(create_app())


class TestErrorHandling:
    def test_health_always_works(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_404_on_unknown_route(self, client):
        resp = client.get("/api/v1/nonexistent")
        assert resp.status_code == 404

    def test_method_not_allowed(self, client):
        resp = client.delete("/api/v1/causal/effects")
        assert resp.status_code == 405


class TestCORS:
    def test_cors_allowed_origin(self):
        settings = Settings(cors_allowed_origins=["https://myapp.com"])
        client = TestClient(create_app(settings))
        resp = client.options(
            "/health",
            headers={"Origin": "https://myapp.com", "Access-Control-Request-Method": "GET"},
        )
        assert resp.headers.get("access-control-allow-origin") == "https://myapp.com"

    def test_cors_disallowed_origin(self):
        settings = Settings(cors_allowed_origins=["https://myapp.com"])
        client = TestClient(create_app(settings))
        resp = client.options(
            "/health",
            headers={"Origin": "https://evil.com", "Access-Control-Request-Method": "GET"},
        )
        assert resp.headers.get("access-control-allow-origin") != "https://evil.com"
