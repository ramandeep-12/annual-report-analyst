"""App settings, read from environment variables (or a local .env file).

Locally: copy .env.example to .env and fill it in.
Deployed: set the same variables in your hosting platform's "Secrets" page.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# Where the vector DB and filing registry live. Point this at a persistent
# volume when deployed, otherwise data is lost on every restart.
DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data"))

# OpenAI model used to write answers
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-5.4-mini")

# Retrieval / chunking
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
TOP_K = 8  # chunks sent to the LLM per question

# Limits for a public demo (0 = no limit)
RATE_LIMIT_PER_HOUR = int(os.environ.get("RATE_LIMIT_PER_HOUR", "0"))  # per visitor
DAILY_QUESTION_LIMIT = int(os.environ.get("DAILY_QUESTION_LIMIT", "0"))  # across everyone

# How many recent 10-Ks to keep per company, and how often to re-check SEC for new ones
YEARS_PER_COMPANY = 3
REFRESH_AFTER_DAYS = 7


def sec_user_agent() -> str:
    """The SEC requires every request to identify the app operator (name + email)."""
    name = os.environ.get("SEC_NAME")
    email = os.environ.get("SEC_EMAIL")
    if not name or not email:
        raise RuntimeError("Set SEC_NAME and SEC_EMAIL (see .env.example).")
    return f"{name} {email}"
