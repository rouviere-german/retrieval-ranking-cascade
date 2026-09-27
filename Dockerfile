# syntax=docker/dockerfile:1
#
# Multi-stage, non-root image for the cascade service. Base images are pinned by digest (resolved
# 2026-09-13); Dependabot's docker ecosystem proposes digest bumps. python:3.12-slim is multi-arch,
# so the index digest works on amd64 and arm64.
#
# faiss-cpu and LightGBM ship OpenMP-linked wheels, so the runtime stage installs libgomp1. Without
# it the image builds and then fails at import time, which is the worst possible place to find out.

ARG PYTHON_IMAGE=python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

FROM ghcr.io/astral-sh/uv:0.12.5@sha256:e85be844203885286c60ffad8a858d48afb6c5a5c237ca0e67f12e74b8f174b1 AS uv

FROM ${PYTHON_IMAGE} AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3.12 \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /build
# Dependencies first, so source edits do not invalidate the dependency layer.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM ${PYTHON_IMAGE} AS runtime
LABEL org.opencontainers.image.title="retrieval-ranking-cascade" \
      org.opencontainers.image.description="Two-tower retrieval, HNSW recall and LambdaMART reranking under per-stage latency budgets." \
      org.opencontainers.image.licenses="MIT"
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --home-dir /nonexistent \
       --shell /usr/sbin/nologin app
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    # The bundle lives on a mounted volume; the image never bakes a model in.
    APP_BUNDLE_DIR=/var/lib/cascade/bundle \
    APP_AUDIT_LOG_PATH=/var/lib/cascade/audit/admin.jsonl \
    APP_DATA_DIR=/var/lib/cascade \
    # One faiss thread per search: the server already has many request threads, and letting each
    # search fan out oversubscribes the CPU and hurts tail latency.
    APP_FAISS_THREADS=1 \
    OMP_NUM_THREADS=1
WORKDIR /app
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-m", "retrieval_ranking_cascade.core.healthcheck"]
ENTRYPOINT ["python", "-m", "retrieval_ranking_cascade"]
CMD ["serve"]
