#!/bin/sh
set -e

echo "Running database migrations..."
alembic upgrade head

echo "Starting API..."
exec uvicorn api.main:app --host 0.0.0.0 --port 8000
