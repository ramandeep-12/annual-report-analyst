import pytest

from rag.sec_client import SECClient, UnknownTickerError, pretty_name
from tests.conftest import offline_sec_client

COMPANIES = [
    ("AAPL", 1, "Apple Inc."),
    ("MSFT", 2, "MICROSOFT CORP"),
    ("GOOGL", 3, "Alphabet Inc."),
    ("GOOG", 3, "Alphabet Inc."),
    ("KO", 4, "COCA COLA CO"),
    ("MCD", 5, "MCDONALDS CORP"),
    ("APLE", 6, "Apple Hospitality REIT, Inc."),
]


@pytest.fixture
def sec() -> SECClient:
    return offline_sec_client(COMPANIES)


def tickers(results):
    return [t for t, _ in results]


def test_lookup_known_and_unknown(sec):
    assert sec.lookup("aapl") == (1, "Apple Inc.")
    with pytest.raises(UnknownTickerError):
        sec.lookup("NOPE")


def test_search_by_name_prefers_bigger_company(sec):
    assert tickers(sec.search("apple")) == ["AAPL", "APLE"]


def test_search_exact_ticker(sec):
    assert tickers(sec.search("ko"))[0] == "KO"


def test_search_ignores_punctuation(sec):
    assert tickers(sec.search("coca-cola")) == ["KO"]
    assert tickers(sec.search("McDonald's")) == ["MCD"]


def test_search_brand_alias(sec):
    assert tickers(sec.search("google")) == ["GOOGL"]


def test_search_returns_one_row_per_company(sec):
    assert tickers(sec.search("alphabet")) == ["GOOGL"]


def test_search_empty_and_no_match(sec):
    assert sec.search("  ") == []
    assert sec.search("zzzz") == []


def test_pretty_name():
    assert pretty_name("MICROSOFT CORP") == "Microsoft Corp"
    assert pretty_name("Apple Inc.") == "Apple Inc."


def test_latest_10ks_keeps_only_10k_newest_first(sec, monkeypatch):
    recent = {
        "form": ["10-Q", "10-K", "8-K", "10-K", "10-K"],
        "accessionNumber": ["a0", "a1", "a2", "a3", "a4"],
        "filingDate": ["2026-12-01", "2026-11-01", "2026-10-01", "2025-11-01", "2024-11-01"],
        "reportDate": ["2026-11-30", "2026-09-30", "2026-10-01", "2025-09-30", "2024-09-30"],
        "primaryDocument": ["q.htm", "k26.htm", "e.htm", "k25.htm", "k24.htm"],
    }

    class Resp:
        def json(self):
            return {"filings": {"recent": recent}}

    monkeypatch.setattr(sec, "_get", lambda url: Resp())
    filings = sec.latest_10ks("AAPL", limit=2)
    assert [f.fiscal_year for f in filings] == [2026, 2025]
    assert filings[0].url.endswith("/1/a1/k26.htm")
