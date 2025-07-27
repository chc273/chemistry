# Base Docker image for QuantChem packages
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Create app directory
WORKDIR /app

# Copy workspace configuration
COPY pyproject.toml ./
COPY uv.lock ./

# Install dependencies
RUN uv sync --frozen

# Development stage
FROM base as dev
RUN uv sync --frozen --extra dev

# Production stage
FROM base as prod
COPY packages/ ./packages/
COPY shared/ ./shared/
RUN uv build

EXPOSE 8000
CMD ["python", "-m", "quantum.cli"]