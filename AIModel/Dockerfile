# ── Build stage ────────────────────────────────────────────────────────────────
# Python 3.11 required — Prophet/TimesFM dependency constraint
FROM python:3.11-slim

# System deps required by Prophet (cmdstanpy needs a C++ compiler)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache: only re-runs on requirements change)
COPY requirements-service.txt .
RUN pip install --no-cache-dir -r requirements-service.txt

# Pre-compile cmdstan so startup is fast (runs once at build time)
RUN python -c "import cmdstanpy; cmdstanpy.install_cmdstan(progress=False)"

# Copy application code
COPY api.py data_generator.py data_loader.py prophet_model.py anomaly_detector.py ./

# Data directory — mount a volume here to persist / pre-load CSV
RUN mkdir -p data

EXPOSE 8000

# Uvicorn with a single worker (model state lives in-process)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
