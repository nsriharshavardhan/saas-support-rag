from fastapi.testclient import TestClient

from app.main import app
from app.rag import RAG, tokenize

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

