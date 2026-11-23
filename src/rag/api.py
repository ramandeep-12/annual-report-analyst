"""HTTP API and chat page.  Run locally with:  uv run uvicorn rag.api:app --reload

Then open http://127.0.0.1:8000 for the chat, or /docs for the raw API.
"""

import json
import logging
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from openai import OpenAI
from pydantic import BaseModel, Field

from rag import config
from rag.answer import answer_question, retrieve, stream_answer
from rag.ingest import ensure_company
from rag.limits import RateLimiter
from rag.sec_client import SECClient, UnknownTickerError, pretty_name
from rag.store import FilingStore, Hit

logging.basicConfig(level=logging.INFO)
for _noisy in ("httpx", "httpx2", "openai"):  # hide per-request HTTP logs
    logging.getLogger(_noisy).setLevel(logging.WARNING)
log = logging.getLogger(__name__)
app = FastAPI(title="10-K Analyst", version="0.2.0")

STATIC_DIR = Path(__file__).parent / "static"
limiter = RateLimiter(config.RATE_LIMIT_PER_HOUR, config.DAILY_QUESTION_LIMIT)


def _enforce_limits(request: Request) -> None:
    # Behind a hosting proxy the visitor's address is in X-Forwarded-For
    forwarded = request.headers.get("x-forwarded-for", "")
    visitor = forwarded.split(",")[0].strip() or (request.client.host if request.client else "?")
    if message := limiter.check(visitor):
        raise HTTPException(status_code=429, detail=message)


@lru_cache
def store() -> FilingStore:
    return FilingStore()


@lru_cache
def sec() -> SECClient:
    return SECClient(config.sec_user_agent())


@lru_cache
def llm() -> OpenAI:
    return OpenAI()  # reads OPENAI_API_KEY


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=20000)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    ticker: str = Field(pattern=r"^[A-Za-z.\-]{1,10}$", examples=["AAPL"])
    years: list[int] | None = Field(default=None, examples=[None])
    history: list[Turn] = Field(
        default_factory=list,
        max_length=40,
        description="Earlier messages in the conversation, oldest first, for follow-up questions",
    )


class Source(BaseModel):
    number: int
    company: str
    fiscal_year: int
    section: str
    url: str
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


def _sources(hits: list[Hit]) -> list[Source]:
    return [
        Source(
            number=i,
            company=h.company,
            fiscal_year=h.fiscal_year,
            section=h.section,
            url=h.url,
            excerpt=h.text[:600],
        )
        for i, h in enumerate(hits, start=1)
    ]


@app.get("/", include_in_schema=False)
def chat_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# Shown in the company picker before the user types anything
POPULAR_TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "NFLX",
    "JPM",
    "V",
    "WMT",
    "KO",
    "NKE",
    "DIS",
    "SBUX",
    "MCD",
]


@app.get("/companies")
def companies(q: str = "", limit: int = 8) -> list[dict]:
    """Search US-listed companies by name or ticker (empty query = popular companies)."""
    indexed = {f["ticker"] for f in store().filings()}
    if q.strip():
        matches = sec().search(q, limit=min(limit, 25))
    else:
        matches = []
        for t in POPULAR_TICKERS:
            try:
                matches.append((t, pretty_name(sec().lookup(t)[1])))
            except UnknownTickerError:
                pass
    return [{"ticker": t, "name": name, "indexed": t in indexed} for t, name in matches]


@app.get("/filings")
def filings(ticker: str | None = None) -> list[dict]:
    """Filings already ingested (optionally for one ticker)."""
    return store().filings(ticker)


@app.post("/ask")
def ask(req: AskRequest, request: Request) -> AskResponse:
    """Ask a question and get the whole answer at once."""
    _enforce_limits(request)
    # Sync endpoint: FastAPI runs it in a thread pool, so slow ingestion doesn't block others
    try:
        ensure_company(store(), sec(), req.ticker)
    except UnknownTickerError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None

    history = [t.model_dump() for t in req.history]
    result = answer_question(store(), llm(), req.question, req.ticker, req.years, history)
    return AskResponse(answer=result.text, sources=_sources(result.sources))


def _event(type_: str, **data) -> str:
    return json.dumps({"type": type_, **data}) + "\n"


@app.post("/chat")
def chat(req: AskRequest, request: Request) -> StreamingResponse:
    """Same as /ask, but streams the answer as it's written.

    Response is newline-delimited JSON events:
      {"type": "status",  "text": "..."}        progress while filings download
      {"type": "sources", "sources": [...]}     excerpts the answer will cite
      {"type": "delta",   "text": "..."}        next piece of the answer
      {"type": "error",   "text": "..."}        something went wrong
      {"type": "done"}
    """
    _enforce_limits(request)
    ticker = req.ticker.upper()
    history = [t.model_dump() for t in req.history]

    def events() -> Iterator[str]:
        try:
            if not store().filings(ticker):
                sec().lookup(ticker)  # fail fast on unknown tickers, before promising a download
                yield _event(
                    "status",
                    text=f"First question about {ticker}: downloading and indexing its "
                    "10-K filings from SEC EDGAR. This takes about a minute.",
                )
            ensure_company(store(), sec(), ticker)

            hits = retrieve(store(), req.question, ticker, req.years, history)
            if not hits:
                yield _event("error", text=f"No 10-K filings found for {ticker}.")
                return
            yield _event("sources", sources=[s.model_dump() for s in _sources(hits)])

            for text in stream_answer(llm(), req.question, hits, history):
                yield _event("delta", text=text)
            yield _event("done")
        except UnknownTickerError:
            yield _event("error", text=f"Couldn't find a US-listed company with ticker {ticker}.")
        except Exception as e:  # report to the user instead of silently cutting the stream
            log.exception("chat failed")
            yield _event("error", text=f"Something went wrong: {e}")

    return StreamingResponse(events(), media_type="application/x-ndjson")
