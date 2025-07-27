# Machine Learning package container
FROM ghcr.io/quantum/base:latest

# Install ML dependencies
RUN uv add torch tensorflow scikit-learn

# Copy ML package
COPY packages/quantum-ml/ ./packages/quantum-ml/

# Install the package
RUN uv pip install -e ./packages/quantum-ml/

# Set entrypoint for ML tasks
ENTRYPOINT ["python", "-m", "quantum.ml"]