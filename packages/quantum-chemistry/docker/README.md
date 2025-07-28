# Quantum Chemistry External Methods Docker Setup

This directory contains Docker configurations and installation scripts for external quantum chemistry packages used by the multireference framework.

## Overview

The external methods integration provides containerized access to:

- **Block2**: DMRG (Density Matrix Renormalization Group) calculations
- **OpenMolcas**: CASPT2/MS-CASPT2 calculations  
- **Dice**: SHCI (Semistochastic Heat-bath Configuration Interaction)
- **Quantum Package**: CIPSI (Configuration Interaction using Perturbative Selection)
- **ipie**: AF-QMC (Auxiliary Field Quantum Monte Carlo) - installed via Python

## Quick Start

### 1. Build All Containers

```bash
cd docker
docker-compose -f docker-compose.external.yml build
```

### 2. Test Individual Packages

```bash
# Test Block2 DMRG
docker-compose -f docker-compose.external.yml run block2

# Test OpenMolcas CASPT2
docker-compose -f docker-compose.external.yml run openmolcas

# Test Dice SHCI
docker-compose -f docker-compose.external.yml run dice

# Test Quantum Package CIPSI
docker-compose -f docker-compose.external.yml run quantum-package
```

### 3. Use Combined Environment

```bash
# Production environment with all methods
docker-compose -f docker-compose.external.yml run quantum-chemistry-full

# Development environment with source code mounted
docker-compose -f docker-compose.external.yml run quantum-chemistry-dev
```

## Directory Structure

```
docker/
├── base/
│   └── Dockerfile.scientific-base       # Scientific computing base image
├── external-packages/
│   ├── Dockerfile.block2                # Block2 DMRG container
│   ├── Dockerfile.openmolcas            # OpenMolcas CASPT2 container
│   ├── Dockerfile.dice                  # Dice SHCI container
│   └── Dockerfile.quantum-package       # Quantum Package CIPSI container
├── install-scripts/
│   ├── install-block2.sh               # Native Block2 installation
│   ├── install-openmolcas.sh           # Native OpenMolcas installation
│   ├── install-dice.sh                 # Native Dice installation
│   └── install-quantum-package.sh      # Native Quantum Package installation
├── docker-compose.external.yml         # Docker Compose configuration
├── Dockerfile.combined                 # Unified container with all methods
└── README.md                           # This file
```

## Installation Options

### Option 1: Docker Containers (Recommended)

Docker provides isolated, reproducible environments with all dependencies pre-configured.

**Advantages:**
- No dependency conflicts
- Consistent across different systems
- Easy deployment on HPC clusters
- Reproducible builds

**Usage:**
```bash
# Build and run specific method
docker build -t qc-block2 -f external-packages/Dockerfile.block2 .
docker run -it qc-block2

# Or use docker-compose for orchestration
docker-compose -f docker-compose.external.yml up quantum-chemistry-full
```

### Option 2: Native Installation Scripts

For systems where Docker is not available or when maximum performance is needed.

**Usage:**
```bash
# Install individual packages
./install-scripts/install-block2.sh
./install-scripts/install-openmolcas.sh
./install-scripts/install-dice.sh
./install-scripts/install-quantum-package.sh

# Source environment after installation
source ~/.bashrc
```

### Option 3: Simple Python Package (ipie)

ipie is installed as a regular Python dependency:

```bash
# Already included in pyproject.toml
uv add 'ipie>=0.1.0' --optional external

# Or with pip
pip install ipie
```

## Package Details

### Block2 DMRG
- **Purpose**: Large active space DMRG calculations
- **Dependencies**: CMake, pybind11, MKL/BLAS, MPI
- **Features**: Complex support, large bond dimensions, GPU acceleration
- **Container**: `quantum-chemistry/block2:latest`

### OpenMolcas CASPT2
- **Purpose**: Production-quality CASPT2/MS-CASPT2 calculations
- **Dependencies**: Fortran compiler, CMake, LAPACK
- **Features**: Multi-state CASPT2, IPEA shifts, intruder state handling
- **Container**: `quantum-chemistry/openmolcas:latest`

### Dice SHCI
- **Purpose**: Semistochastic heat-bath configuration interaction
- **Dependencies**: C++11, Boost, Eigen, MPI
- **Features**: Near-FCI accuracy, parallel execution, PySCF integration
- **Container**: `quantum-chemistry/dice:latest`

### Quantum Package CIPSI
- **Purpose**: Configuration interaction with perturbative selection
- **Dependencies**: OCaml, OPAM, Ninja, IRPF90
- **Features**: Systematic convergence, EZFIO interface, FCI extrapolation
- **Container**: `quantum-chemistry/quantum-package:latest`

### ipie AF-QMC
- **Purpose**: Auxiliary-field quantum Monte Carlo calculations
- **Dependencies**: Python packages (numpy, scipy, cupy for GPU)
- **Features**: Statistical error bars, GPU acceleration, distributed calculations
- **Installation**: Regular Python package via uv/pip

## Environment Variables

### Docker Environment
```bash
# Block2
PYTHONPATH=/opt/block2/build:/opt/block2
OMP_NUM_THREADS=4

# OpenMolcas
MOLCAS=/opt/molcas
MOLCAS_MEM=2000

# Dice
DICE_ROOT=/opt/dice

# Quantum Package
QP_ROOT=/opt/quantum-package

# General
OMP_NUM_THREADS=4
```

### Native Environment
After running installation scripts, these variables are added to `~/.bashrc`:

```bash
# Block2
export PYTHONPATH=$HOME/quantum-software/block2/build:$HOME/quantum-software/block2:$PYTHONPATH

# OpenMolcas  
export MOLCAS=$HOME/quantum-software/molcas/install
export MOLCAS_MEM=2000

# Dice
export DICE_ROOT=$HOME/quantum-software/dice

# Quantum Package
export QP_ROOT=$HOME/quantum-software/quantum-package
source $QP_ROOT/quantum_package.rc
```

## Performance Considerations

### Docker Performance
- Use bind mounts for data directories to avoid container storage overhead
- Set appropriate `OMP_NUM_THREADS` based on available CPU cores
- For GPU calculations, use `--gpus all` flag with Docker

### Memory Requirements
- **Block2**: Scales with bond dimension and active space size
- **OpenMolcas**: Set `MOLCAS_MEM` based on available RAM (MB)
- **Dice**: Memory usage depends on determinant count
- **Quantum Package**: Automatic memory management
- **ipie**: Scales with number of walkers and system size

### Parallelization
All packages support OpenMP threading. MPI parallelization is available for:
- Block2 (with `-DMPI=ON`)
- Dice (built with MPI compilers)
- OpenMolcas (with MPI-enabled build)

## Integration with Quantum Chemistry Framework

The external methods integrate seamlessly with the main framework:

```python
from quantum.chemistry.multireference.external import (
    DMRGMethod, AFQMCMethod, SelectedCIMethod
)
from quantum.chemistry.multireference.external.openmolcas import CASPT2Method

# Use methods with automatic containerized execution
dmrg = DMRGMethod(bond_dimension=1000)
result = dmrg.calculate(scf_obj, active_space)
```

## Troubleshooting

### Common Issues

1. **Container build failures**: Check Docker daemon and available disk space
2. **Permission errors**: Ensure Docker user has appropriate permissions
3. **Memory errors**: Increase Docker memory limits or container memory settings
4. **Path issues**: Verify environment variables are set correctly

### Testing Installation

Each container includes a test script:

```bash
# Test individual containers
docker run quantum-chemistry/block2:latest
docker run quantum-chemistry/openmolcas:latest
docker run quantum-chemistry/dice:latest
docker run quantum-chemistry/quantum-package:latest

# Test combined environment
docker run quantum-chemistry/full:latest
```

### Debug Mode

For debugging, run containers interactively:

```bash
docker run -it --entrypoint /bin/bash quantum-chemistry/block2:latest
```

## Contributing

When adding new external methods:

1. Create a new Dockerfile in `external-packages/`
2. Add the service to `docker-compose.external.yml`
3. Create a native installation script in `install-scripts/`
4. Update the combined Dockerfile to include the new method
5. Add integration tests

## Support

For issues with specific packages, consult their official documentation:
- [Block2 Documentation](https://block2.readthedocs.io/)
- [OpenMolcas Manual](https://molcas.gitlab.io/OpenMolcas/)
- [Dice Documentation](https://sanshar.github.io/Dice/)
- [Quantum Package Documentation](https://quantum-package.readthedocs.io/)
- [ipie Documentation](https://github.com/JoonhoLee-Group/ipie)