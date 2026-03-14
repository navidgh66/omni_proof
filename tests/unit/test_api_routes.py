"""Tests for FastAPI routes."""

import pytest
from fastapi.testclient import TestClient

from omni_proof.api.app import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestCausalRoutes:
    def test_list_effects(self, client):
        resp = client.get("/api/v1/causal/effects")
        assert resp.status_code == 200
        assert "effects" in resp.json()

    def test_get_effect(self, client):
        resp = client.get("/api/v1/causal/effects/logo_timing")
        assert resp.status_code == 200
        assert resp.json()["treatment"] == "logo_timing"

    def test_analyze(self, client):
        resp = client.post("/api/v1/causal/analyze", json={
            "treatment": "fast_pacing",
            "outcome": "ctr",
            "confounders": ["platform", "audience"],
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "queued"


class TestComplianceRoutes:
    def test_list_reports(self, client):
        resp = client.get("/api/v1/compliance/reports")
        assert resp.status_code == 200

    def test_list_reports_with_campaign(self, client):
        resp = client.get("/api/v1/compliance/reports?campaign_id=camp1")
        assert resp.status_code == 200
        assert resp.json()["campaign_id"] == "camp1"


class TestInsightsRoutes:
    def test_list_briefs(self, client):
        resp = client.get("/api/v1/insights/briefs")
        assert resp.status_code == 200

    def test_get_segments(self, client):
        resp = client.get("/api/v1/insights/segments?segment=18-24")
        assert resp.status_code == 200
        assert resp.json()["segment"] == "18-24"


class TestGenerativeRoutes:
    def test_generate_prompt(self, client):
        resp = client.post("/api/v1/generative/prompt", json={
            "target_segment": "18-24",
            "objective": "conversion",
            "constraints": ["16:9 aspect ratio"],
        })
        assert resp.status_code == 200
        prompt = resp.json()["prompt"]
        assert "18-24" in prompt
        assert "conversion" in prompt
