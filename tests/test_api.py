import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    app.dependency_overrides = {}
    with patch("app.auth.verify_api_key", return_value="test-key"):
        yield TestClient(app)
    app.dependency_overrides = {}


def test_root():
    c = TestClient(app)
    res = c.get("/")
    assert res.status_code == 200


def test_health():
    c = TestClient(app)
    res = c.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data


def test_upload_rejects_non_pdf(client):
    res = client.post(
        "/upload",
        files={"file": ("test.txt", b"hello", "text/plain")},
    )
    assert res.status_code == 400


def test_query_requires_auth():
    c = TestClient(app)
    res = c.post("/query", json={"question": "test"})
    assert res.status_code in (401, 403)


def test_query_empty_question(client):
    res = client.post("/query", json={"question": ""})
    assert res.status_code == 400


def test_cache_clear(client):
    res = client.post("/cache/clear")
    assert res.status_code == 200
    assert "Cache cleared" in res.json()["message"]
