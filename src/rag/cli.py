"""Command line interface.

    uv run rag ingest AAPL MSFT        # pre-load companies
    uv run rag ask AAPL "What are the main risk factors?"
    uv run rag list
"""

import argparse
import logging

from openai import OpenAI

from rag import config
from rag.answer import answer_question
from rag.ingest import ensure_company
from rag.sec_client import SECClient
from rag.store import FilingStore


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag", description="10-K Analyst")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="download + embed recent 10-Ks")
    p_ingest.add_argument("tickers", nargs="+")

    p_ask = sub.add_parser("ask", help="ask a question about a company")
    p_ask.add_argument("ticker")
    p_ask.add_argument("question")
    p_ask.add_argument("--years", type=int, nargs="*")

    sub.add_parser("list", help="show ingested filings")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    store = FilingStore()

    if args.command == "list":
        for f in store.filings():
            print(f"{f['ticker']:6} FY{f['fiscal_year']}  {f['chunks']:5} chunks  {f['url']}")
        return

    sec = SECClient(config.sec_user_agent())
    if args.command == "ingest":
        for ticker in args.tickers:
            new = ensure_company(store, sec, ticker)
            print(f"{ticker.upper()}: {len(new)} new filing(s) ingested")
    elif args.command == "ask":
        ensure_company(store, sec, args.ticker)
        result = answer_question(store, OpenAI(), args.question, args.ticker, args.years)
        print("\n" + result.text + "\n\nSources:")
        for i, h in enumerate(result.sources, start=1):
            print(f"  [{i}] {h.company} FY{h.fiscal_year} — {h.section}\n      {h.url}")
