import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("app.auth.verify_api_key", return_value="test-key"):
        from app.main import app
        yield TestClient(app)


def test_root(client):
    res = client.get("/")
    assert res.status_code == 200


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert "postgres" in data
    assert "neo4j" in data


def test_upload_rejects_non_pdf(client):
    res = client.post(
        "/upload",
        headers={"X-API-Key": "test-key"},
        files={"file": ("test.txt", b"hello", "text/plain")},
    )
    assert res.status_code == 400


def test_query_requires_auth():
    from app.main import app
    c = TestClient(app)
    res = c.post("/query", json={"question": "test"})
    assert res.status_code == 401


def test_query_empty_question(client):
    res = client.post(
        "/query",
        json={"question": ""},
        headers={"X-API-Key": "test-key"},
    )
    assert res.status_code == 400


def test_cache_clear(client):
    res = client.post("/cache/clear", headers={"X-API-Key": "test-key"})
    assert res.status_code == 200
    assert "Cache cleared" in res.json()["message"]
