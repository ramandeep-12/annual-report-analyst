from rag.answer import answer_question, retrieve, stream_answer
from rag.ingest import ensure_company
from tests.conftest import FakeLLM


def test_answer_cites_retrieved_excerpts(store, fake_sec):
    ensure_company(store, fake_sec, "ACME")
    llm = FakeLLM()
    result = answer_question(store, llm, "What are the supplier risks?", "ACME")

    assert result.text == "Revenue grew [1]."
    assert result.sources
    prompt = llm.requests[0]["messages"][-1]["content"]
    assert "[1] Acme Corp 10-K" in prompt and "Question: What are the supplier risks?" in prompt


def test_no_filings_means_no_llm_call(store):
    llm = FakeLLM()
    result = answer_question(store, llm, "Anything?", "ACME")
    assert "No filings" in result.text and llm.requests == []


def test_refusal_is_reported(store, fake_sec):
    ensure_company(store, fake_sec, "ACME")
    result = answer_question(store, FakeLLM(refusal="no"), "question?", "ACME")
    assert result.sources == [] and "can't answer" in result.text


def test_follow_up_sends_history_and_searches_with_previous_question(store, fake_sec, monkeypatch):
    ensure_company(store, fake_sec, "ACME")
    queries = []
    real_search = store.search
    monkeypatch.setattr(
        store, "search", lambda q, *a, **k: queries.append(q) or real_search(q, *a, **k)
    )

    history = [
        {"role": "user", "content": "What are the supplier risks?"},
        {"role": "assistant", "content": "One supplier [1]."},
    ]
    llm = FakeLLM()
    answer_question(store, llm, "And how bad could it get?", "ACME", history=history)

    assert queries == ["What are the supplier risks?\nAnd how bad could it get?"]
    roles = [m["role"] for m in llm.requests[0]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]


def test_stream_answer_yields_pieces(store, fake_sec):
    ensure_company(store, fake_sec, "ACME")
    hits = retrieve(store, "net sales", "ACME")
    assert (
        "".join(stream_answer(FakeLLM("Sales rose [2]."), "net sales", hits)).strip()
        == "Sales rose [2]."
    )
