"""Helpdesk retrieval-augmented answering.

A small, dependency-free BM25 retriever over a JSON knowledge base. Answers are
always grounded in retrieved articles and returned with ``[kb-id]`` citations.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "helpdesk_kb.json"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "get", "how", "i", "if", "in", "is", "it", "my", "of", "on", "or",
    "our", "so", "that", "the", "this", "to", "we", "what", "when", "where",
    "which", "will", "with", "you", "your", "me",
}

# Below this BM25 score we say "I don't know" instead of guessing.
MIN_SCORE = 1.0


def _stem(token: str) -> str:
    """Very small suffix stripper so 'tickets'/'ticket' and 'importing'/'import' match."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [_stem(t) for t in tokens if t not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


@dataclass
class Article:
    id: str
    category: str
    title: str
    content: str

    @property
    def text(self) -> str:
        # Title is repeated to give it extra weight in BM25.
        return f"{self.title} {self.title} {self.content}"


@dataclass
class BM25Index:
    k1: float = 1.5
    b: float = 0.75
    docs: list[list[str]] = field(default_factory=list)
    doc_freq: Counter = field(default_factory=Counter)
    avg_len: float = 0.0

    def build(self, texts: list[str]) -> None:
        self.docs = [tokenize(t) for t in texts]
        self.doc_freq = Counter()
        for doc in self.docs:
            self.doc_freq.update(set(doc))
        self.avg_len = sum(len(d) for d in self.docs) / max(len(self.docs), 1)

    def idf(self, term: str) -> float:
        n = len(self.docs)
        df = self.doc_freq.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def score(self, query: str) -> list[float]:
        q_terms = tokenize(query)
        scores = []
        for doc in self.docs:
            tf = Counter(doc)
            s = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                f = tf[term]
                denom = f + self.k1 * (1 - self.b + self.b * len(doc) / self.avg_len)
                s += self.idf(term) * f * (self.k1 + 1) / denom
            scores.append(s)
        return scores


class HelpdeskRAG:
    def __init__(self, data_path: Path = DATA_PATH) -> None:
        raw = json.loads(Path(data_path).read_text(encoding="utf-8"))
        self.product: str = raw.get("product", "the product")
        self.articles: list[Article] = [Article(**a) for a in raw["articles"]]
        self.eval_set: list[dict] = raw.get("eval", [])
        self.index = BM25Index()
        self.rebuild_index()

    # ---- indexing -------------------------------------------------------
    def rebuild_index(self) -> None:
        self.index.build([a.text for a in self.articles])

    # ---- retrieval ------------------------------------------------------
    def search(self, query: str, top_k: int = 3) -> list[tuple[Article, float]]:
        scores = self.index.score(query)
        ranked = sorted(zip(self.articles, scores), key=lambda x: x[1], reverse=True)
        return [(a, round(s, 3)) for a, s in ranked[:top_k] if s > 0]

    # ---- answering ------------------------------------------------------
    def _extractive_answer(self, query: str, hits: list[tuple[Article, float]]) -> str:
        """Pick the 2 most query-relevant sentences from the best article."""
        best = hits[0][0]
        q_terms = set(tokenize(query))
        sentences = split_sentences(best.content)
        ranked = sorted(
            enumerate(sentences),
            key=lambda pair: len(q_terms & set(tokenize(pair[1]))),
            reverse=True,
        )
        keep = sorted(ranked[:2])  # restore original order for readability
        return " ".join(s for _, s in keep) + f" [{best.id}]"

    def answer(self, query: str, top_k: int = 3) -> dict:
        hits = self.search(query, top_k=top_k)
        if not hits or hits[0][1] < MIN_SCORE:
            return {
                "answer": "Sorry, I couldn't find that in the help centre. "
                "A support agent will follow up.",
                "citations": [],
                "sources": [],
                "mode": "no_match",
            }

        return {
            "answer": self._extractive_answer(query, hits),
            "citations": [hits[0][0].id],
            "sources": [
                {"id": a.id, "title": a.title, "category": a.category, "score": s}
                for a, s in hits
            ],
            "mode": "extractive",
        }

    # ---- evaluation -----------------------------------------------------
    def evaluate(self) -> dict:
        """Top-1 and top-3 retrieval accuracy on the labelled eval questions."""
        top1 = top3 = 0
        failures = []
        for item in self.eval_set:
            ids = [a.id for a, _ in self.search(item["question"], top_k=3)]
            top1 += bool(ids) and ids[0] == item["expected"]
            top3 += item["expected"] in ids
            if not ids or ids[0] != item["expected"]:
                failures.append({**item, "got": ids[:1]})
        n = max(len(self.eval_set), 1)
        return {
            "questions": len(self.eval_set),
            "top1_accuracy": round(top1 / n, 3),
            "top3_accuracy": round(top3 / n, 3),
            "failures": failures,
        }


RAG = HelpdeskRAG()
