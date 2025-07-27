# Quantum Chemistry package container
FROM ghcr.io/quantum/base:latest

# Install additional quantum chemistry dependencies
RUN uv add pyscf openfermion

# Copy quantum chemistry package
COPY packages/quantum-chemistry/ ./packages/quantum-chemistry/

# Install the package
RUN uv pip install -e ./packages/quantum-chemistry/

# Set entrypoint for QC calculations
ENTRYPOINT ["python", "-m", "quantum.chemistry"]