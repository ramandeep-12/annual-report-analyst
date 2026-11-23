import json

import pytest
from fastapi.testclient import TestClient

from rag import api
from rag.limits import RateLimiter
from tests.conftest import FakeLLM


@pytest.fixture
def client(store, fake_sec, monkeypatch):
    monkeypatch.setattr(api, "store", lambda: store)
    monkeypatch.setattr(api, "sec", lambda: fake_sec)
    monkeypatch.setattr(api, "llm", lambda: FakeLLM("Sales rose [1]."))
    monkeypatch.setattr(api, "limiter", RateLimiter(0, 0))
    return TestClient(api.app)


def events(response):
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def test_chat_page_and_health(client):
    assert "<title>10-K Analyst</title>" in client.get("/").text
    assert client.get("/health").json() == {"status": "ok"}


def test_chat_streams_status_sources_answer(client):
    evs = events(client.post("/chat", json={"question": "How did sales change?", "ticker": "acme"}))
    types = [e["type"] for e in evs]
    assert types[0] == "status"  # first question about ACME downloads its filings
    assert "sources" in types and types[-1] == "done"
    answer = "".join(e["text"] for e in evs if e["type"] == "delta")
    assert answer.strip() == "Sales rose [1]."
    source = next(e for e in evs if e["type"] == "sources")["sources"][0]
    assert source["number"] == 1 and source["company"] == "Acme Corp"


def test_second_question_skips_download_status(client):
    client.post("/chat", json={"question": "First question", "ticker": "ACME"})
    evs = events(client.post("/chat", json={"question": "Second question", "ticker": "ACME"}))
    assert evs[0]["type"] == "sources"


def test_chat_unknown_ticker(client):
    evs = events(client.post("/chat", json={"question": "Risks?", "ticker": "NOPE"}))
    assert evs == [{"type": "error", "text": "Couldn't find a US-listed company with ticker NOPE."}]


def test_ask_returns_answer_and_sources(client):
    body = client.post("/ask", json={"question": "How did sales change?", "ticker": "ACME"}).json()
    assert body["answer"] == "Sales rose [1]."
    assert body["sources"][0]["fiscal_year"] in (2024, 2025, 2026)


def test_ask_unknown_ticker_is_404(client):
    assert client.post("/ask", json={"question": "Risks?", "ticker": "NOPE"}).status_code == 404


def test_invalid_request_is_rejected(client):
    assert client.post("/chat", json={"question": "Hi", "ticker": "AAPL!"}).status_code == 422


def test_companies_search_marks_indexed(client):
    assert client.get("/companies", params={"q": "acm"}).json() == [
        {"ticker": "ACME", "name": "Acme Corp", "indexed": False}
    ]
    client.post("/ask", json={"question": "Anything new?", "ticker": "ACME"})
    assert client.get("/companies", params={"q": "acme"}).json()[0]["indexed"] is True


def test_rate_limit_returns_429(client, monkeypatch):
    monkeypatch.setattr(api, "limiter", RateLimiter(per_hour=1, per_day_total=0))
    client.post("/ask", json={"question": "First one", "ticker": "ACME"})
    r = client.post("/ask", json={"question": "Second one", "ticker": "ACME"})
    assert r.status_code == 429 and "per hour" in r.json()["detail"]
