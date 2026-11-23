FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Dependencies first, so code changes don't reinstall them
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH" DATA_DIR=/data

# Bake Chroma's embedding model into the image so the first question isn't slow
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction as E; E()(['warm up'])"

# Filings + vectors live here; mount a persistent volume to keep them across restarts
VOLUME /data
EXPOSE 8000
CMD ["sh", "-c", "uvicorn rag.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
