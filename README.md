# SaaS Support RAG

[![CI](https://github.com/nsriharshavardhan/saas-support-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/nsriharshavardhan/saas-support-rag/actions/workflows/ci.yml)

**Live demo:** https://saas-support-rag.onrender.com

A help centre assistant for a fictional B2B SaaS product (**FlowDesk**). It answers customer questions
**only from knowledge-base articles** and cites them (e.g. `[kb-007]`). If nothing relevant is found, it says so
instead of making something up.

## Features

- **BM25 retrieval written from scratch.** No vector DB needed, deterministic, and easy to explain.
- **Citation-grounded answers** with a minimum-score "no match" guard against hallucinated replies.
- **Built-in evaluation.** `GET /eval` reports top-1 and top-3 retrieval accuracy on a labelled question set.
  The test suite fails if top-1 accuracy drops below 85%.
- **Delivery:** pytest suite, GitHub Actions CI (tests + Docker build), Dockerfile, and a one-click `render.yaml`.

## API

| Method | Path        | Description                                   |
|--------|-------------|-----------------------------------------------|
| GET    | `/`         | Browser demo UI                               |
| POST   | `/ask`      | `{"question": "...", "top_k": 3}` → answer + citations + sources |
| GET    | `/eval`     | Retrieval accuracy report                     |
| GET    | `/health`   | Health check                                  |

```bash
curl -X POST https://saas-support-rag.onrender.com/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the API rate limit?"}'
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
uvicorn app.main:app --reload   # open http://127.0.0.1:8000
```

Or with Docker:

```bash
docker build -t saas-support-rag . && docker run -p 8000:8000 saas-support-rag
```

## Deploy (Render free tier)

1. Sign in to [Render](https://render.com) with GitHub, then choose **New → Blueprint** and pick this repo.
2. Render reads `render.yaml`, installs dependencies and starts `uvicorn`.

## Design notes

- **Why BM25 instead of embeddings?** For a few hundred support articles, lexical retrieval is fast and free.
  You can also debug it by hand, and the eval endpoint shows it performs well on this data.
  The `HelpdeskRAG.search` interface is deliberately small, so it can be swapped for a pgvector or hybrid retriever later.
- **Why a score threshold?** It's better to hand off to a human agent than give a confident wrong answer.
