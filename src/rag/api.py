"""HTTP API.  Run locally with:  uv run uvicorn rag.api:app --reload"""

import logging
from functools import lru_cache

import anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag import config
from rag.answer import answer_question
from rag.ingest import ensure_company
from rag.sec_client import SECClient, UnknownTickerError
from rag.store import FilingStore

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="10-K Analyst", version="0.1.0")


@lru_cache
def store() -> FilingStore:
    return FilingStore()


@lru_cache
def sec() -> SECClient:
    return SECClient(config.sec_user_agent())


@lru_cache
def llm() -> anthropic.Anthropic:
    return anthropic.Anthropic()


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    ticker: str = Field(pattern=r"^[A-Za-z.\-]{1,10}$", examples=["AAPL"])
    years: list[int] | None = Field(default=None, examples=[[2024, 2025]])


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


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/filings")
def filings(ticker: str | None = None) -> list[dict]:
    """Filings already ingested (optionally for one ticker)."""
    return store().filings(ticker)


@app.post("/ask")
def ask(req: AskRequest) -> AskResponse:
    # Sync endpoint: FastAPI runs it in a thread pool, so slow ingestion doesn't block others
    try:
        ensure_company(store(), sec(), req.ticker)
    except UnknownTickerError as e:
        raise HTTPException(status_code=404, detail=str(e))

    result = answer_question(store(), llm(), req.question, req.ticker, req.years)
    return AskResponse(
        answer=result.text,
        sources=[
            Source(
                number=i,
                company=h.company,
                fiscal_year=h.fiscal_year,
                section=h.section,
                url=h.url,
                excerpt=h.text[:500],
            )
            for i, h in enumerate(result.sources, start=1)
        ],
    )
