"""FastAPI service for the FlowDesk helpdesk RAG assistant."""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.rag import RAG, Article

app = FastAPI(
    title="SaaS Support RAG",
    description="Citation-grounded helpdesk answers over a SaaS knowledge base (BM25 + optional LLM).",
    version="1.1.0",
)


class AskIn(BaseModel):
    question: str = Field(..., min_length=3, max_length=500, examples=["How do I reset my password?"])
    top_k: int = Field(3, ge=1, le=10)


class ArticleIn(BaseModel):
    id: str = Field(..., pattern=r"^[a-z0-9-]{3,40}$", examples=["kb-015"])
    category: str = Field(..., min_length=2, max_length=40)
    title: str = Field(..., min_length=3, max_length=200)
    content: str = Field(..., min_length=20, max_length=5000)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "articles": len(RAG.articles)}


@app.post("/ask")
def ask(body: AskIn) -> dict:
    return RAG.answer(body.question, top_k=body.top_k)


@app.get("/articles")
def list_articles() -> list[dict]:
    return [{"id": a.id, "category": a.category, "title": a.title} for a in RAG.articles]


@app.post("/articles", status_code=201)
def add_article(body: ArticleIn) -> dict:
    RAG.add_article(Article(**body.model_dump()))
    return {"status": "indexed", "id": body.id, "articles": len(RAG.articles)}


@app.get("/eval")
def evaluate() -> dict:
    return RAG.evaluate()


DEMO_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FlowDesk Support Assistant</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:760px;margin:40px auto;padding:0 16px;color:#1f2937}
 h1{margin-bottom:4px} p.sub{color:#6b7280;margin-top:0}
 form{display:flex;gap:8px} input{flex:1;padding:10px;border:1px solid #d1d5db;border-radius:8px;font-size:15px}
 button{padding:10px 16px;border:0;border-radius:8px;background:#2563eb;color:#fff;font-size:15px;cursor:pointer}
 .chips button{background:#eef2ff;color:#3730a3;margin:8px 6px 0 0;padding:6px 10px;font-size:13px}
 .card{border:1px solid #e5e7eb;border-radius:10px;padding:16px;margin-top:20px}
 .src{font-size:13px;color:#4b5563} .meta{font-size:12px;color:#9ca3af;margin-top:8px}
</style></head><body>
<h1>FlowDesk Support Assistant</h1>
<p class="sub">Ask a question about the FlowDesk help centre. Answers are grounded in knowledge-base articles with citations.</p>
<form id="f"><input id="q" placeholder="e.g. How do I verify webhook signatures?" required><button>Ask</button></form>
<div class="chips">
 <button type="button">How do I reset my password?</button>
 <button type="button">What is the API rate limit?</button>
 <button type="button">Can I get a refund on an annual plan?</button>
</div>
<div id="out"></div>
<p class="meta"><a href="/docs">API docs</a> · <a href="/eval">Retrieval eval</a> · <a href="/articles">Articles</a></p>
<script>
const out=document.getElementById('out'), q=document.getElementById('q');
async function ask(text){
  out.innerHTML='<div class="card">Searching…</div>';
  const t0=performance.now();
  const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:text})});
  const d=await r.json();
  const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
  const srcs=(d.sources||[]).map(s=>`<li>[${esc(s.id)}] ${esc(s.title)} <span class="meta">score ${s.score}</span></li>`).join('');
  out.innerHTML=`<div class="card"><div>${esc(d.answer||d.detail)}</div>
    ${srcs?`<p class="src"><b>Sources</b></p><ul class="src">${srcs}</ul>`:''}
    <div class="meta">mode: ${esc(d.mode||'-')} · ${Math.round(performance.now()-t0)} ms round trip</div></div>`;
}
document.getElementById('f').onsubmit=e=>{e.preventDefault();ask(q.value)};
document.querySelectorAll('.chips button').forEach(b=>b.onclick=()=>{q.value=b.textContent;ask(b.textContent)});
</script></body></html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def demo() -> str:
    return DEMO_HTML
