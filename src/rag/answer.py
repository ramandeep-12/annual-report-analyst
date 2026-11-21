"""Retrieve relevant chunks and ask Claude to answer with numbered citations."""

from dataclasses import dataclass

import anthropic

from rag import config
from rag.store import FilingStore, Hit

SYSTEM_PROMPT = """You are a financial analyst answering questions about companies' SEC 10-K filings.

Answer using only the numbered excerpts provided. Cite each claim with the excerpt number in square brackets, like [2] or [1][4]. If the excerpts do not contain the answer, say so plainly rather than relying on outside knowledge, especially for figures. Keep answers concise; use a small table when comparing years."""


@dataclass
class Answer:
    text: str
    sources: list[Hit]


def _format_excerpts(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{i}] {h.company} 10-K, fiscal year {h.fiscal_year}, {h.section}\n{h.text}"
        for i, h in enumerate(hits, start=1)
    )


def answer_question(
    store: FilingStore,
    llm: anthropic.Anthropic,
    question: str,
    ticker: str,
    years: list[int] | None = None,
) -> Answer:
    hits = store.search(question, ticker, years, k=config.TOP_K)
    if not hits:
        return Answer(text=f"No filings are stored for {ticker.upper()} yet.", sources=[])

    response = llm.beta.messages.create(
        model=config.LLM_MODEL,
        max_tokens=16000,
        output_config={"effort": "medium"},
        # If the model declines a request, the API retries it on a fallback model
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"<excerpts>\n{_format_excerpts(hits)}\n</excerpts>\n\nQuestion: {question}",
            }
        ],
    )

    if response.stop_reason == "refusal":
        return Answer(text="Sorry, I can't answer that question.", sources=[])

    text = "".join(block.text for block in response.content if block.type == "text")
    return Answer(text=text, sources=hits)
