# Hugging Face Spaces (Docker SDK) image for the FastAPI backend.
# Spaces serve on port 7860 and run the container as UID 1000.
# Local Docker Compose uses Dockerfile.api instead (Postgres on :8000).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System libraries for WeasyPrint (PDF export) + psycopg2 + build of wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        libpango-1.0-0 \
        libpangocairo-1.0-0 \
        libgdk-pixbuf-2.0-0 \
        libcairo2 \
        libffi-dev \
        shared-mime-info \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces run as a non-root user; give it a home and a writable workdir.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

COPY --chown=user:user requirements-deploy.txt /app/requirements-deploy.txt
RUN pip install --no-cache-dir -r /app/requirements-deploy.txt

COPY --chown=user:user . /app

# /app is created by WORKDIR as root; make it (and runtime cache/log dirs)
# writable by the unprivileged HF user so settings can mkdir at startup.
RUN mkdir -p /app/data/cache/prices /app/logs \
    && chown -R user:user /app

USER user

# Cache + logs are created under /app at runtime (owned by user via --chown).
EXPOSE 7860

# Run migrations on each boot, then serve. Port is fixed to 7860 for HF Spaces.
CMD alembic upgrade head && exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}
