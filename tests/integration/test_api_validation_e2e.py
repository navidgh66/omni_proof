"""E2E tests for API input validation and security boundaries."""

import pytest
from fastapi.testclient import TestClient

from omni_proof.api.app import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


class TestBrandValidation:
    def test_path_traversal_rejected(self, client):
        resp = client.post(
            "/api/v1/brand/extract",
            json={"brand_name": "Test", "asset_paths": ["../../../etc/passwd"]},
        )
        assert resp.status_code == 422

    def test_empty_asset_paths_rejected(self, client):
        resp = client.post(
            "/api/v1/brand/extract",
            json={"brand_name": "Test", "asset_paths": []},
        )
        assert resp.status_code == 422

    def test_brand_name_too_long(self, client):
        resp = client.post(
            "/api/v1/brand/extract",
            json={"brand_name": "x" * 201, "asset_paths": ["logo.png"]},
        )
        assert resp.status_code == 422

    def test_empty_brand_name_rejected(self, client):
        resp = client.post(
            "/api/v1/brand/extract",
            json={"brand_name": "", "asset_paths": ["logo.png"]},
        )
        assert resp.status_code == 422

    def test_valid_paths_accepted(self, client):
        resp = client.post(
            "/api/v1/brand/extract",
            json={"brand_name": "TestBrand", "asset_paths": ["assets/logo.png", "guide.pdf"]},
        )
        assert resp.status_code == 200


class TestCausalValidation:
    def test_treatment_with_spaces_rejected(self, client):
        resp = client.post(
            "/api/v1/causal/analyze",
            json={"treatment": "has space", "outcome": "ctr", "confounders": ["x"]},
        )
        assert resp.status_code == 422

    def test_treatment_with_special_chars_rejected(self, client):
        resp = client.post(
            "/api/v1/causal/analyze",
            json={"treatment": "drop;table", "outcome": "ctr", "confounders": ["x"]},
        )
        assert resp.status_code == 422

    def test_empty_confounders_rejected(self, client):
        resp = client.post(
            "/api/v1/causal/analyze",
            json={"treatment": "fast_pacing", "outcome": "ctr", "confounders": []},
        )
        assert resp.status_code == 422

    def test_valid_analyze_accepted(self, client):
        resp = client.post(
            "/api/v1/causal/analyze",
            json={"treatment": "fast_pacing", "outcome": "ctr", "confounders": ["platform"]},
        )
        assert resp.status_code == 200


class TestComplianceValidation:
    def test_malicious_filename_sanitized(self, client):
        files = {"file": ("../../etc/passwd", b"data", "image/jpeg")}
        resp = client.post("/api/v1/compliance/check", files=files)
        assert resp.status_code == 200
        # Filename should be sanitized — no path separators
        asset_id = resp.json()["asset_id"]
        assert "/" not in asset_id
        assert "\\" not in asset_id
        assert ".." not in asset_id


class TestGenerativeValidation:
    def test_empty_segment_rejected(self, client):
        resp = client.post(
            "/api/v1/generative/prompt",
            json={"target_segment": "", "objective": "conversion"},
        )
        assert resp.status_code == 422

    def test_empty_objective_rejected(self, client):
        resp = client.post(
            "/api/v1/generative/prompt",
            json={"target_segment": "18-24", "objective": ""},
        )
        assert resp.status_code == 422
