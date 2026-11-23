"""Vector store (Chroma) plus a small registry of which filings are ingested."""

import json
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from rag import config
from rag.parsing import Chunk
from rag.sec_client import Filing


@dataclass
class Hit:
    text: str
    ticker: str
    company: str
    fiscal_year: int
    section: str
    url: str
    score: float


class FilingStore:
    def __init__(self, data_dir: Path | None = None, embedding_function=None):
        data_dir = data_dir or config.DATA_DIR
        data_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(data_dir / "chroma"))
        self.collection = client.get_or_create_collection(
            "filings",
            # Default: Chroma's built-in local model (all-MiniLM-L6-v2): free, no API key
            embedding_function=embedding_function or embedding_functions.DefaultEmbeddingFunction(),
            metadata={"hnsw:space": "cosine"},
        )
        self._registry_path = data_dir / "registry.json"
        self._lock = threading.Lock()

    # ---------- registry: which filings are already embedded ----------

    def _read_registry(self) -> dict:
        if self._registry_path.exists():
            return json.loads(self._registry_path.read_text())
        return {"filings": {}, "checked": {}}

    def _write_registry(self, registry: dict) -> None:
        tmp = self._registry_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(registry, indent=2))
        tmp.replace(self._registry_path)

    def has_filing(self, accession: str) -> bool:
        return accession in self._read_registry()["filings"]

    def last_checked(self, ticker: str) -> datetime | None:
        value = self._read_registry()["checked"].get(ticker.upper())
        return datetime.fromisoformat(value) if value else None

    def mark_checked(self, ticker: str) -> None:
        with self._lock:
            registry = self._read_registry()
            registry["checked"][ticker.upper()] = datetime.now(UTC).isoformat()
            self._write_registry(registry)

    def filings(self, ticker: str | None = None) -> list[dict]:
        rows = self._read_registry()["filings"].values()
        if ticker:
            rows = [r for r in rows if r["ticker"] == ticker.upper()]
        return sorted(rows, key=lambda r: (r["ticker"], -r["fiscal_year"]))

    # ---------- vectors ----------

    def add_filing(self, filing: Filing, chunks: list[Chunk]) -> None:
        batch = 128
        for start in range(0, len(chunks), batch):
            part = chunks[start : start + batch]
            self.collection.upsert(
                ids=[f"{filing.accession}-{start + i}" for i in range(len(part))],
                documents=[c.text for c in part],
                metadatas=[
                    {
                        "ticker": filing.ticker,
                        "company": filing.company,
                        "fiscal_year": filing.fiscal_year,
                        "accession": filing.accession,
                        "section": c.section,
                        "url": filing.url,
                    }
                    for c in part
                ],
            )
        # Only register after all vectors are stored, so a crash mid-way gets retried
        with self._lock:
            registry = self._read_registry()
            registry["filings"][filing.accession] = {
                **asdict(filing),
                "fiscal_year": filing.fiscal_year,
                "url": filing.url,
                "chunks": len(chunks),
            }
            self._write_registry(registry)

    def search(self, question: str, ticker: str, years: list[int] | None, k: int) -> list[Hit]:
        where: dict = {"ticker": ticker.upper()}
        if years:
            where = {"$and": [where, {"fiscal_year": {"$in": years}}]}
        result = self.collection.query(query_texts=[question], n_results=k, where=where)
        return [
            Hit(
                text=doc,
                ticker=meta["ticker"],
                company=meta["company"],
                fiscal_year=meta["fiscal_year"],
                section=meta["section"],
                url=meta["url"],
                score=1 - dist,
            )
            for doc, meta, dist in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True
            )
        ]
