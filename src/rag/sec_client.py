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

    def lookup(self, ticker: str) -> tuple[int, str]:
        """Ticker -> (CIK number, company name)."""
        if self._tickers is None:
            rows = self._get(TICKERS_URL).json().values()
            self._tickers = {r["ticker"].upper(): (r["cik_str"], r["title"]) for r in rows}
        try:
            return self._tickers[ticker.upper()]
        except KeyError:
            raise UnknownTickerError(f"Unknown ticker: {ticker}") from None

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
