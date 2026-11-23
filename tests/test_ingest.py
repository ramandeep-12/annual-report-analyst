from rag.ingest import ensure_company


def test_ingests_new_filings_once(store, fake_sec):
    first = ensure_company(store, fake_sec, "acme")
    assert len(first) == 3
    assert fake_sec.downloads == 3
    assert len(store.filings("ACME")) == 3

    # Checked recently: no SEC calls at all the second time
    assert ensure_company(store, fake_sec, "ACME") == []
    assert fake_sec.downloads == 3


def test_only_missing_filings_are_downloaded_after_refresh(store, fake_sec, monkeypatch):
    ensure_company(store, fake_sec, "ACME")
    monkeypatch.setattr(store, "last_checked", lambda ticker: None)  # pretend a week passed
    assert ensure_company(store, fake_sec, "ACME") == []
    assert fake_sec.downloads == 3
