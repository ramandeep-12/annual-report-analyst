from rag.parsing import Chunk
from tests.conftest import make_filing


def add(store, year, texts):
    store.add_filing(make_filing(year), [Chunk(text=t, section="Item 7: MD&A") for t in texts])


def test_registry_tracks_ingested_filings(store):
    assert store.filings() == []
    add(store, 2025, ["Net sales grew because of anvils."])
    assert store.has_filing(make_filing(2025).accession)
    [row] = store.filings("acme")
    assert row["fiscal_year"] == 2025 and row["chunks"] == 1


def test_search_finds_relevant_chunk_and_filters_by_ticker(store):
    add(store, 2025, ["Net sales grew because of anvils.", "The board met four times."])
    hits = store.search("why did net sales grow", "ACME", None, k=1)
    assert "anvils" in hits[0].text
    assert store.search("net sales", "OTHER", None, k=3) == []


def test_search_filters_by_year(store):
    add(store, 2025, ["Net sales in 2025 were strong."])
    add(store, 2024, ["Net sales in 2024 were weak."])
    hits = store.search("net sales", "ACME", [2024], k=5)
    assert {h.fiscal_year for h in hits} == {2024}


def test_last_checked_roundtrip(store):
    assert store.last_checked("ACME") is None
    store.mark_checked("acme")
    assert store.last_checked("ACME") is not None
