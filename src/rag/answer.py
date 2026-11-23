"""Retrieve relevant chunks and ask the LLM (OpenAI) to answer with numbered citations."""

from collections.abc import Iterator
from dataclasses import dataclass

from openai import OpenAI

from rag import config
from rag.store import FilingStore, Hit

SYSTEM_PROMPT = """You are a financial analyst answering questions about companies' SEC 10-K filings.

Answer using only the numbered excerpts provided with the latest question. Cite each claim with the excerpt number in square brackets, like [2] or [1][4]. If the excerpts do not contain the answer, say so plainly rather than relying on outside knowledge, especially for figures. Keep answers concise; use a small table when comparing years.

Earlier turns of the conversation are there so you understand follow-up questions. Citation numbers in earlier answers referred to earlier excerpts, so only cite the excerpts given with the latest question."""

# How many earlier messages (user + assistant) to send along for follow-up questions
HISTORY_TURNS = 6


@dataclass
class Answer:
    text: str
    sources: list[Hit]


def _format_excerpts(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{i}] {h.company} 10-K, fiscal year {h.fiscal_year}, {h.section}\n{h.text}"
        for i, h in enumerate(hits, start=1)
    )


def retrieve(
    store: FilingStore,
    question: str,
    ticker: str,
    years: list[int] | None = None,
    history: list[dict] | None = None,
) -> list[Hit]:
    # A follow-up like "what about 2024?" means little on its own, so search with
    # the previous question too.
    previous = [t["content"] for t in history or [] if t["role"] == "user"]
    query = f"{previous[-1]}\n{question}" if previous else question
    return store.search(query, ticker, years, k=config.TOP_K)


def _messages(question: str, hits: list[Hit], history: list[dict] | None) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or [])[-HISTORY_TURNS:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append(
        {
            "role": "user",
            "content": f"<excerpts>\n{_format_excerpts(hits)}\n</excerpts>\n\nQuestion: {question}",
        }
    )
    return messages


def answer_question(
    store: FilingStore,
    llm: OpenAI,
    question: str,
    ticker: str,
    years: list[int] | None = None,
    history: list[dict] | None = None,
) -> Answer:
    hits = retrieve(store, question, ticker, years, history)
    if not hits:
        return Answer(text=f"No filings are stored for {ticker.upper()} yet.", sources=[])

    response = llm.chat.completions.create(
        model=config.LLM_MODEL,
        max_completion_tokens=16000,
        messages=_messages(question, hits, history),
    )

    message = response.choices[0].message
    if message.refusal:
        return Answer(text="Sorry, I can't answer that question.", sources=[])
    return Answer(text=message.content or "", sources=hits)


def stream_answer(
    llm: OpenAI, question: str, hits: list[Hit], history: list[dict] | None = None
) -> Iterator[str]:
    """Yield the answer text piece by piece as the model writes it."""
    stream = llm.chat.completions.create(
        model=config.LLM_MODEL,
        max_completion_tokens=16000,
        messages=_messages(question, hits, history),
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content
        elif getattr(delta, "refusal", None):
            yield delta.refusal
