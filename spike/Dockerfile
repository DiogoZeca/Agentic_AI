# ── Spike Prediction API — Production Image ──────────────────────────────────
#
#   Build context: repo root (Agentic_AI/)
#
#   Build:
#     docker compose up --build
#
#   The model artifacts directory (data/full_run/) is mounted at runtime, not
#   baked into the image — this allows model updates without a rebuild.
#
#   Environment variable:
#     MODEL_DIR  path inside the container to the spike model artefacts
#                (default: /app/data/full_run/models/spike)
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim

# libgomp1 is required by XGBoost (OpenMP runtime) — not present in slim base
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Non-root user — best practice for production containers
RUN useradd --no-log-init --create-home appuser

WORKDIR /app

# Layer-cache dependencies separately from source code.
# Rebuilds only trigger a pip re-install when requirements change.
COPY spike/requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

# Install the spike package in editable mode
COPY pyproject.toml .
COPY spike/ ./spike/
RUN pip install --no-cache-dir -e .

# Artifact mount point — bind-mounted from host at runtime (see docker-compose.yml)
RUN mkdir -p data/full_run/models/spike \
 && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "spike.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
