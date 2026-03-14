"""E2E test: FastAPI application endpoints."""

import pytest
from fastapi.testclient import TestClient

from omni_proof.api.app import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


class TestAPIE2E:
    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_causal_analyze_flow(self, client):
        # Submit analysis
        resp = client.post(
            "/api/v1/causal/analyze",
            json={
                "treatment": "logo_in_first_3s",
                "outcome": "ctr",
                "confounders": ["platform", "audience_segment", "budget"],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "not_configured"
        assert resp.json()["treatment"] == "logo_in_first_3s"

    def test_generative_prompt_flow(self, client):
        resp = client.post(
            "/api/v1/generative/prompt",
            json={
                "target_segment": "18-24",
                "objective": "conversion",
                "constraints": ["16:9 aspect ratio", "warm color palette"],
            },
        )
        assert resp.status_code == 200
        prompt = resp.json()["prompt"]
        assert "18-24" in prompt
        assert "conversion" in prompt
        assert "16:9" in prompt

    def test_all_get_endpoints_respond(self, client):
        endpoints = [
            "/api/v1/causal/effects",
            "/api/v1/causal/effects/test_treatment",
            "/api/v1/compliance/reports",
            "/api/v1/insights/briefs",
            "/api/v1/insights/segments",
        ]
        for ep in endpoints:
            resp = client.get(ep)
            assert resp.status_code == 200, f"Failed: {ep}"
