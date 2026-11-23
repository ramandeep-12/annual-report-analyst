"""Shared fakes, so tests run offline: no SEC, no OpenAI, no embedding-model download."""

import hashlib
import re
from types import SimpleNamespace

import numpy as np
import pytest
from chromadb.api.types import EmbeddingFunction

from rag.sec_client import Filing, SECClient
from rag.store import FilingStore


class WordHashEmbedding(EmbeddingFunction):
    """Tiny deterministic bag-of-words embedding: texts sharing words end up close."""

    def __init__(self) -> None:
        pass

    def __call__(self, input):
        vectors = []
        for text in input:
            v = np.zeros(128, dtype=np.float32)
            for word in re.findall(r"[a-z]+", text.lower()):
                v[int(hashlib.md5(word.encode()).hexdigest(), 16) % 128] += 1
            vectors.append(v / (np.linalg.norm(v) or 1))
        return vectors

    @staticmethod
    def name() -> str:
        return "word-hash-test"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config):
        return WordHashEmbedding()


@pytest.fixture
def store(tmp_path) -> FilingStore:
    return FilingStore(data_dir=tmp_path, embedding_function=WordHashEmbedding())


def make_filing(year: int, ticker: str = "ACME") -> Filing:
    return Filing(
        ticker=ticker,
        company="Acme Corp",
        cik=123,
        accession=f"0000000123-{year % 100}-000001",
        filing_date=f"{year}-11-01",
        report_date=f"{year}-09-30",
        primary_doc=f"acme-{year}.htm",
    )


class FakeSEC:
    """Stands in for SECClient: serves canned filings, counts downloads."""

    def __init__(self, filings: list[Filing], html: bytes):
        self.filings = filings
        self.html = html
        self.downloads = 0

    def lookup(self, ticker):
        if ticker.upper() != "ACME":
            from rag.sec_client import UnknownTickerError

            raise UnknownTickerError(f"Unknown ticker: {ticker}")
        return 123, "Acme Corp"

    def latest_10ks(self, ticker, limit):
        self.lookup(ticker)
        return self.filings[:limit]

    def download(self, filing):
        self.downloads += 1
        return self.html

    def search(self, q, limit=8):
        return [("ACME", "Acme Corp")] if "acme".startswith(q.lower()) else []


class FakeLLM:
    """Stands in for openai.OpenAI(): records requests, returns a canned answer."""

    def __init__(self, answer: str = "Revenue grew [1].", refusal: str | None = None):
        self.requests: list[dict] = []
        self.answer = answer
        self.refusal = refusal
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        if kwargs.get("stream"):
            words = self.answer.split(" ")
            return [
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=w + " ", refusal=None))]
                )
                for w in words
            ]
        message = SimpleNamespace(
            content=None if self.refusal else self.answer, refusal=self.refusal
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


FILING_HTML = b"""<html><body>
<p>Item 1. Business</p>
<p>Acme designs and sells rocket powered roller skates and anvils to customers worldwide.
We operate two segments, Skates and Anvils, and sell mainly through our online store.</p>
<p>Item 1A. Risk Factors</p>
<p>Our business depends on a single supplier of rocket engines. Any disruption at this
supplier could delay shipments of roller skates and reduce revenue significantly.</p>
<p>Item 7. Management's Discussion and Analysis</p>
<p>Total net sales increased to $120 million from $100 million, driven by strong anvil demand.</p>
<table><tr><td>Net sales</td><td>$120</td><td>$100</td></tr></table>
</body></html>"""


@pytest.fixture
def fake_sec() -> FakeSEC:
    return FakeSEC([make_filing(2026), make_filing(2025), make_filing(2024)], FILING_HTML)


def offline_sec_client(companies: list[tuple[str, int, str]]) -> SECClient:
    """A real SECClient whose ticker list is preloaded instead of downloaded."""
    client = SECClient("test test@example.com")
    client._companies = companies
    client._tickers = {t: (cik, name) for t, cik, name in companies}
    return client
