import os

os.environ["SIEVE_BACKEND"] = "mock"

from fastapi.testclient import TestClient

from app.server import app

client = TestClient(app)


def test_health_and_index():
    assert client.get("/api/health").json()["backend"] == "mock"
    assert "Strict JSON" in client.get("/").text


def test_samples_and_extract():
    samples = client.get("/api/samples").json()
    assert samples and {"text", "label"} <= set(samples[0])
    s = samples[0]
    r = client.post("/api/extract", json={"text": s["text"], "gold": s["label"]}).json()
    assert r["finetuned"]["strict"] and r["finetuned"]["schema_valid"]
    assert not r["base"]["strict"] and r["base"]["json_valid"]  # fenced, but parseable
    assert set(r["base"]["fields_correct"]) == set(s["label"])


def test_rejects_tiny_input():
    assert client.post("/api/extract", json={"text": "hi"}).status_code == 422
