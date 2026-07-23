import os
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

os.environ["API_KEY"] = "test-key"

from app.main import app

client = TestClient(app)
HEADERS = {"X-API-Key": "test-key"}


def test_root():
    res = client.get("/")
    assert res.status_code == 200


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data


def test_upload_rejects_non_pdf():
    res = client.post("/upload", headers=HEADERS, files={"file": ("test.txt", b"hello", "text/plain")})
    assert res.status_code == 400


def test_query_requires_auth():
    res = client.post("/query", json={"question": "test"})
    assert res.status_code in (401, 403)


def test_query_empty_question():
    res = client.post("/query", json={"question": ""}, headers=HEADERS)
    assert res.status_code == 400


def test_cache_clear():
    res = client.post("/cache/clear", headers=HEADERS)
    assert res.status_code == 200
    assert "Cache cleared" in res.json()["message"]
