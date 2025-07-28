# Multi-stage Dockerfile for quantum chemistry package
# Stage 1: Base scientific computing environment
FROM continuumio/miniconda3:latest as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CONDA_ALWAYS_YES="true"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libhdf5-dev \
    libopenmpi-dev \
    openmpi-bin \
    git \
    curl \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Create conda environment with scientific packages
RUN conda create -n quantum-chemistry python=3.11 && \
    conda activate quantum-chemistry

# Activate environment in shell
SHELL ["conda", "run", "-n", "quantum-chemistry", "/bin/bash", "-c"]

# Install scientific computing packages
RUN conda install -c conda-forge \
    numpy>=1.24.0 \
    scipy>=1.10.0 \
    pyscf>=2.3.0 \
    pyyaml>=6.0 \
    pydantic>=2.0.0 \
    h5py \
    matplotlib \
    jupyter \
    ipython \
    pytest>=7.4.0 \
    && conda clean -a

# Install additional Python packages
RUN pip install --no-cache-dir \
    ruff>=0.0.285 \
    mypy>=1.5.0 \
    black \
    pre-commit

# Stage 2: Development environment
FROM base as development

# Set working directory
WORKDIR /opt/quantum-chemistry

# Copy package files
COPY . .

# Install the package in development mode
RUN pip install -e .

# Install development dependencies
RUN pip install -e .[dev]

# Set up Jupyter configuration
RUN jupyter --generate-config && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.port = 8888" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> ~/.jupyter/jupyter_notebook_config.py

# Create directories for data and notebooks
RUN mkdir -p /opt/data /opt/notebooks /opt/results

# Stage 3: Testing environment
FROM base as testing

WORKDIR /opt/quantum-chemistry

# Copy package and tests
COPY . .

# Install package and test dependencies
RUN pip install -e .[dev]

# Run tests by default
CMD ["python", "-m", "pytest", "tests/", "-v"]

# Stage 4: Production environment
FROM base as production

WORKDIR /opt/quantum-chemistry

# Copy only necessary files
COPY src/ src/
COPY pyproject.toml README.md CHANGES.md ./

# Install package in production mode
RUN pip install .

# Create non-root user
RUN useradd -m -u 1000 quantum && \
    chown -R quantum:quantum /opt/quantum-chemistry

USER quantum

# Default command
CMD ["python", "-c", "import quantum.chemistry; print('Quantum Chemistry Package Ready')"]

# Stage 5: Jupyter environment for interactive development
FROM development as jupyter

WORKDIR /opt/quantum-chemistry

# Expose Jupyter port
EXPOSE 8888

# Copy example notebooks
COPY examples/ /opt/notebooks/examples/
COPY docs/ /opt/notebooks/docs/

# Start Jupyter by default
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/opt/notebooks"]