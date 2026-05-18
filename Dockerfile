# syntax=docker/dockerfile:1.7

# ---------- Stage 1: builder ----------
# Compile wheels once; the runtime image then only has to copy them.
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build deps for native wheels (bcrypt, lxml-ish stuff in some langchain
# transitive deps, etc.). Drop them in the runtime stage.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt requirements-test.txt requirements-worker.txt ./

# Bake all wheels into a directory so the runtime stage installs from
# them without touching PyPI. `--prefer-binary` keeps the image small
# by preferring manylinux wheels over source builds.
RUN pip wheel --wheel-dir=/wheels --prefer-binary \
        -r requirements.txt \
        -r requirements-worker.txt

# ---------- Stage 2: runtime ----------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# Non-root runtime user — never run the API as root in a container.
RUN groupadd -r lexmind && useradd -r -g lexmind -d /app -s /sbin/nologin lexmind

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY requirements.txt requirements-worker.txt ./
RUN pip install --no-index --find-links=/wheels \
        -r requirements.txt \
        -r requirements-worker.txt \
    && rm -rf /wheels

# Copy the application. .dockerignore should exclude `frontend/`,
# `chatbot/data/`, `docs/images/`, `.git/`, `__pycache__/`, etc.
COPY backend ./backend
COPY chatbot ./chatbot

USER lexmind

# The API listens on 8000 by default. Override CMD to launch the
# worker instead:
#   docker run … lexmind/backend arq chatbot.workers.pdf_worker.WorkerSettings
EXPOSE 8000

# `--proxy-headers` makes Uvicorn trust X-Forwarded-* from the
# reverse proxy in front. Always run behind one in production.
CMD ["uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--proxy-headers", \
     "--forwarded-allow-ips=*"]
