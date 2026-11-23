import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.rag import RAG, Article, HelpdeskRAG, tokenize

client = TestClient(app)


def test_tokenize_removes_stopwords_and_stems():
    assert tokenize("How do I import the Tickets?") == ["import", "ticket"]


def test_password_question_cites_reset_article():
    result = RAG.answer("I forgot my password")
    assert result["citations"] == ["kb-001"]
    assert "[kb-001]" in result["answer"]
    assert result["mode"] == "extractive"


def test_unrelated_question_returns_no_match():
    result = RAG.answer("What is the capital of France?")
    assert result["citations"] == []
    assert result["mode"] == "no_match"


def test_retrieval_eval_meets_quality_bar():
    report = RAG.evaluate()
    assert report["questions"] >= 10
    assert report["top1_accuracy"] >= 0.85, report["failures"]
    assert report["top3_accuracy"] >= 0.9


def test_health_and_demo_page():
    assert client.get("/health").json()["status"] == "ok"
    page = client.get("/")
    assert page.status_code == 200
    assert "FlowDesk Support Assistant" in page.text


def test_ask_endpoint_and_validation():
    r = client.post("/ask", json={"question": "What is the API rate limit?"})
    assert r.status_code == 200
    assert r.json()["citations"] == ["kb-007"]
    assert client.post("/ask", json={"question": "hi"}).status_code == 422


def test_eval_endpoint():
    r = client.get("/eval")
    assert r.status_code == 200
    assert r.json()["top3_accuracy"] >= 0.9


@pytest.fixture
def fresh_rag(monkeypatch):
    """Isolated RAG instance so ingestion tests don't leak into other tests."""
    rag = HelpdeskRAG()
    monkeypatch.setattr("app.main.RAG", rag)
    return rag


def test_ingested_article_is_retrievable(fresh_rag):
    fresh_rag.add_article(
        Article(
            id="kb-csv",
            category="data",
            title="Exporting reports to CSV",
            content="Open Reports, choose a date range and click Export CSV to download ticket metrics.",
        )
    )
    assert fresh_rag.answer("How do I export CSV?")["citations"] == ["kb-csv"]


def test_post_articles_endpoint(fresh_rag):
    payload = {
        "id": "kb-015",
        "category": "workspace",
        "title": "Setting business hours",
        "content": "Go to Settings > Business hours to define working days so SLA timers pause overnight.",
    }
    r = client.post("/articles", json=payload)
    assert r.status_code == 201
    assert client.post("/ask", json={"question": "set business hours"}).json()["citations"] == ["kb-015"]
    assert client.post("/articles", json={**payload, "id": "Bad ID!"}).status_code == 422


def test_llm_failure_falls_back_to_extractive(fresh_rag, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "invalid-test-key")

    def boom(*args, **kwargs):
        raise RuntimeError("network disabled in tests")

    monkeypatch.setattr("app.rag.httpx.post", boom)
    result = fresh_rag.answer("How do I reset my password?")
    assert result["mode"] == "extractive"
    assert result["citations"] == ["kb-001"]


def test_latency_header_present():
    r = client.get("/health")
    assert float(r.headers["X-Process-Time-Ms"]) >= 0
