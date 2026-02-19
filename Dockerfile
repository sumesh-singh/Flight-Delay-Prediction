# ============================================================================
# Flight Delay Prediction System - Dockerfile
# Addresses IEEE Reproducibility Limitation (#2)
# ============================================================================

# Stage 1: Base Python environment
FROM python:3.11-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Application
FROM base AS app

WORKDIR /app

# Copy project files
# Copy project files
COPY config/ config/
COPY src/ src/
COPY experiments/ experiments/
COPY scripts/ scripts/
COPY data/ data/
COPY streamlit_app/ streamlit_app/
COPY train_pipeline.py .
COPY requirements.txt .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/metadata data/external data/cache \
    experiments/logs models/

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default: Run Streamlit UI
CMD ["streamlit", "run", "streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# ============================================================================
# Usage:
#   docker build -t flight-delay-prediction .
#   docker run -p 8501:8501 flight-delay-prediction
#
# To run experiments instead of UI:
#   docker run flight-delay-prediction python experiments/run_multiyear_experiments.py
# ============================================================================
