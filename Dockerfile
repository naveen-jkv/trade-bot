# Multi-Stage / Production Dockerfile for Autonomous Trading Engine
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable immediate log streaming
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRADING_MODE=PAPER \
    CONFIRM_REAL_MONEY_TRADING=FALSE \
    PORT=8000

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Create non-root application user
RUN useradd -m -u 1000 trader && \
    chown -R trader:trader /app

USER trader

# Expose backend port
EXPOSE 8000

# Health check to ensure API server is responsive
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/system/mode || exit 1

# Start FastAPI application
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
