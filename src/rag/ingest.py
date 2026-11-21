"""On-demand ingestion: make sure a company's recent 10-Ks are in the vector store.

The first question about a company downloads and embeds its filings (slow, ~1 min).
After that, every user gets answers straight from the store (fast).
"""

import logging
import threading
from datetime import datetime, timedelta, timezone

from rag import config
from rag.parsing import chunk_filing
from rag.sec_client import Filing, SECClient
from rag.store import FilingStore

log = logging.getLogger(__name__)

# One lock per ticker, so two users asking about NVDA at once don't ingest it twice
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(ticker: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(ticker, threading.Lock())


def ensure_company(store: FilingStore, sec: SECClient, ticker: str) -> list[Filing]:
    """Ingest any of the company's recent 10-Ks that aren't stored yet.

    Returns the filings that were newly ingested (empty if already up to date).
    """
    ticker = ticker.upper()
    with _lock_for(ticker):
        checked = store.last_checked(ticker)
        fresh = timedelta(days=config.REFRESH_AFTER_DAYS)
        if checked and datetime.now(timezone.utc) - checked < fresh:
            return []

        new = []
        for filing in sec.latest_10ks(ticker, limit=config.YEARS_PER_COMPANY):
            if store.has_filing(filing.accession):
                continue
            log.info("Ingesting %s FY%s (%s)", ticker, filing.fiscal_year, filing.accession)
            chunks = chunk_filing(sec.download(filing))
            store.add_filing(filing, chunks)
            log.info("  stored %d chunks", len(chunks))
            new.append(filing)

        store.mark_checked(ticker)
        return new
