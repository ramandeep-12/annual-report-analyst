"""Minimal client for SEC EDGAR's free JSON API.

Docs: https://www.sec.gov/search-filings/edgar-application-programming-interfaces
Fair-access rules: identify yourself in the User-Agent, max 10 requests/second.
"""

import time
from dataclasses import dataclass

import requests

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}"


class UnknownTickerError(ValueError):
    pass


# Brand names people search for that differ from the SEC's official company name
ALIASES = {
    "google": "GOOGL", "youtube": "GOOGL", "facebook": "META", "instagram": "META",
    "whatsapp": "META", "coke": "KO", "chase": "JPM", "jp morgan": "JPM",
    "walmart": "WMT", "costco": "COST", "home depot": "HD",
}


def _normalize(text: str) -> str:
    """Lowercase and drop punctuation, so "coca-cola" matches "COCA COLA CO"."""
    text = text.lower().replace("'", "").replace(".", "")
    return " ".join(text.replace("-", " ").replace(",", " ").replace("&", " & ").split())


def pretty_name(name: str) -> str:
    """'MICROSOFT CORP' -> 'Microsoft Corp' (names already in mixed case are kept)."""
    return name.title() if name.isupper() else name


@dataclass(frozen=True)
class Filing:
    ticker: str
    company: str
    cik: int
    accession: str  # unique filing id, e.g. 0000320193-24-000123
    filing_date: str  # when it was filed, YYYY-MM-DD
    report_date: str  # fiscal period end, YYYY-MM-DD
    primary_doc: str  # the main HTML document's filename

    @property
    def fiscal_year(self) -> int:
        return int(self.report_date[:4])

    @property
    def url(self) -> str:
        return ARCHIVE_URL.format(
            cik=self.cik, accession=self.accession.replace("-", ""), doc=self.primary_doc
        )


class SECClient:
    def __init__(self, user_agent: str):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self._tickers: dict[str, tuple[int, str]] | None = None
        self._companies: list[tuple[str, int, str]] = []  # (ticker, cik, name), SEC's order
        self._last_request = 0.0

    def _get(self, url: str) -> requests.Response:
        # Stay well under the 10 requests/second limit
        wait = 0.15 - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        resp = self.session.get(url, timeout=30)
        self._last_request = time.monotonic()
        resp.raise_for_status()
        return resp

    def _load_tickers(self) -> dict[str, tuple[int, str]]:
        if self._tickers is None:
            rows = list(self._get(TICKERS_URL).json().values())
            self._companies = [(r["ticker"].upper(), r["cik_str"], r["title"]) for r in rows]
            self._tickers = {t: (cik, name) for t, cik, name in self._companies}
        return self._tickers

    def lookup(self, ticker: str) -> tuple[int, str]:
        """Ticker -> (CIK number, company name)."""
        try:
            return self._load_tickers()[ticker.upper()]
        except KeyError:
            raise UnknownTickerError(f"Unknown ticker: {ticker}") from None

    def search(self, query: str, limit: int = 8) -> list[tuple[str, str]]:
        """Find companies by name or ticker -> [(ticker, name)], best matches first.

        SEC lists companies roughly by size, so among equal matches bigger companies come first.
        """
        self._load_tickers()
        q = _normalize(query)
        if not q:
            return []
        ranked: list[tuple[int, int, str, str]] = []
        seen_ciks: set[int] = set()
        for alias, ticker in ALIASES.items():
            if ticker in self._tickers and alias.startswith(q) and len(q) >= 3:
                ranked.append((0, -1, ticker, self._tickers[ticker][1]))
        for pos, (ticker, cik, name) in enumerate(self._companies):
            t, n = ticker.lower(), _normalize(name)
            if t == q:
                rank = 0
            elif n.startswith(q):
                rank = 1
            elif t.startswith(q):
                rank = 2
            elif f" {q}" in f" {n}":  # a word in the name starts with the query
                rank = 3
            else:
                continue
            ranked.append((rank, pos, ticker, name))
        ranked.sort()
        results = []
        for rank, _, ticker, name in ranked:
            cik = self._tickers[ticker][0]
            if cik in seen_ciks and rank != 0:  # one row per company (e.g. GOOGL, not also GOOG)
                continue
            seen_ciks.add(cik)
            results.append((ticker, pretty_name(name)))
            if len(results) >= limit:
                break
        return results

    def latest_10ks(self, ticker: str, limit: int) -> list[Filing]:
        """Most recent 10-K filings for a company, newest first."""
        cik, company = self.lookup(ticker)
        recent = self._get(SUBMISSIONS_URL.format(cik=cik)).json()["filings"]["recent"]
        filings = []
        for i, form in enumerate(recent["form"]):
            if form != "10-K":
                continue
            filings.append(
                Filing(
                    ticker=ticker.upper(),
                    company=company,
                    cik=cik,
                    accession=recent["accessionNumber"][i],
                    filing_date=recent["filingDate"][i],
                    report_date=recent["reportDate"][i],
                    primary_doc=recent["primaryDocument"][i],
                )
            )
            if len(filings) >= limit:
                break
        return filings

    def download(self, filing: Filing) -> bytes:
        return self._get(filing.url).content
