# 10-K Analyst

Ask questions about any US public company's annual reports (SEC 10-K filings) and get answers with citations back to the exact filing and section.

Filings are fetched **on demand**: the first question about a company downloads and indexes its last three 10-Ks from SEC EDGAR; after that, answers come straight from the vector store.

## How it works

```
question + ticker
      │
      ▼
ensure_company ──(not stored yet?)──► SEC EDGAR API ─► parse HTML ─► split by 10-K Item ─► embed ─► Chroma
      │
      ▼
vector search (filtered by ticker / year) ─► top-k excerpts ─► Claude ─► answer with [n] citations
```

| Module | Role |
|---|---|
| `src/rag/sec_client.py` | Ticker → CIK lookup, list recent 10-Ks, download filings |
| `src/rag/parsing.py` | HTML → clean text (tables kept as rows), split into 10-K sections, chunk |
| `src/rag/store.py` | Chroma vector store + registry of ingested filings |
| `src/rag/ingest.py` | On-demand ingestion with per-ticker locking and weekly refresh |
| `src/rag/answer.py` | Retrieval + prompt + Claude call |
| `src/rag/api.py` | FastAPI app (`/ask`, `/filings`, `/health`) |
| `src/rag/cli.py` | `rag ingest`, `rag ask`, `rag list` |

## Setup

```bash
uv sync
cp .env.example .env   # then fill in SEC_NAME, SEC_EMAIL, ANTHROPIC_API_KEY
```

## Usage

```bash
# CLI
uv run rag ingest AAPL MSFT
uv run rag ask AAPL "What are Apple's main risk factors?"
uv run rag ask TSLA "How did R&D spending change?" --years 2024 2025

# API (docs at http://localhost:8000/docs)
uv run uvicorn rag.api:app --reload
```

## Roadmap

- [x] On-demand ingestion from SEC EDGAR
- [x] Section-aware chunking and cited answers
- [ ] Web UI
- [ ] Hybrid search (BM25 + vectors) and reranking
- [ ] Evaluation set with SEC XBRL figures as ground truth
- [ ] Multi-company comparison questions
- [ ] Docker + deployment
