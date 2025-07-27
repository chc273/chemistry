# Quantum

A comprehensive Python monorepo for quantum chemistry, materials science, and quantum computing with integrated machine learning capabilities. Built with a modular architecture using uv workspaces.

## Architecture

This project uses a **monorepo structure** with multiple specialized packages:

- **`quantum-core`**: Base classes, interfaces, and shared utilities
- **`quantum-chemistry`**: Quantum chemistry calculations (Hartree-Fock, DFT, Coupled Cluster)
- **`quantum-materials`**: Materials science and crystal structure analysis
- **`quantum-computing`**: Quantum algorithms and circuit construction
- **`quantum-ml`**: Machine learning for molecular and materials properties
- **`quantum-cli`**: Command-line interface for all functionality

## Features

### 🧬 Quantum Chemistry (`quantum-chemistry`)

- **Multiple calculation methods**: Hartree-Fock, DFT, Coupled Cluster
- **Flexible basis sets**: Support for standard quantum chemistry basis sets
- **Property calculations**: Energies, molecular orbitals, dipole moments, polarizabilities
- **Geometry optimization**: Structure optimization with various algorithms
- **PySCF integration**: Seamless integration with PySCF backend

### 🔬 Materials Science (`quantum-materials`)

- **Crystal structure handling**: Create, manipulate, and analyze crystal structures
- **PyMatGen integration**: Full integration with Materials Project ecosystem
- **Electronic structure calculations**: Band structures, density of states
- **Phonon calculations**: Vibrational analysis for crystals
- **ASE compatibility**: Works with Atomic Simulation Environment

### ⚛️ Quantum Computing (`quantum-computing`)

- **Variational algorithms**: VQE, QAOA implementations
- **Quantum circuit building**: Flexible quantum circuit construction
- **Multiple backends**: Support for Qiskit, PennyLane, and classical simulators
- **Noise modeling**: Realistic quantum device simulation

### 🤖 Machine Learning (`quantum-ml`)

- **Property prediction**: ML models for molecular and materials properties
- **Feature extraction**: Multiple molecular and crystal descriptors
- **Model types**: Random Forest, Neural Networks, with uncertainty quantification
- **Quantum ML**: Integration of quantum computing with machine learning

### 🖥️ Core Infrastructure (`quantum-core`)

- **Base classes**: `Molecule`, `Crystal`, `ComputationEngine`
- **Data validation**: Pydantic-based models for type safety
- **Converters**: Seamless conversion between different format standards
- **Shared utilities**: Common functionality across all packages

## Installation

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended package manager)

### Using uv (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/yourusername/quantum.git
cd quantum

# Install all packages in development mode
uv sync --dev

# Activate the virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

### Using pip

```bash
git clone https://github.com/yourusername/quantum.git
cd quantum

# Install the main package (installs all sub-packages)
pip install -e .

# Or install specific packages
pip install -e packages/quantum-core
pip install -e packages/quantum-chemistry
```

### Docker Development

```bash
# Build all services
docker-compose build

# Run development environment
docker-compose up dev

# Run specific services
docker-compose up quantum-chemistry
docker-compose up quantum-ml
```

## Quick Start

### Basic Molecular Calculation

```python
import numpy as np
from quantum.core import Molecule, ComputationEngine
from quantum.chemistry import HartreeFockCalculator

# Create a water molecule
water = Molecule(
    name="water",
    atoms=["O", "H", "H"],
    coordinates=np.array([
        [0.0000, 0.0000, 0.1173],
        [0.0000, 0.7572, -0.4692],
        [0.0000, -0.7572, -0.4692],
    ]),
)

# Set up computation engine
engine = ComputationEngine()
engine.register_calculator("hf", HartreeFockCalculator)

# Run calculation
results = engine.run_single_point(water, method="hf", basis_set="sto-3g")
print(f"Energy: {results['energy']:.6f} Hartree")
```

### Machine Learning Workflow

```python
from quantum.ml import MolecularML
from quantum.core import Molecule

# Initialize ML model
ml_model = MolecularML(
    model_type="random_forest",
    feature_type="coulomb_matrix"
)

# Train on your data
results = ml_model.train(molecules, properties)
print(f"Model R²: {results.r2:.4f}")

# Make predictions
predictions = ml_model.predict(new_molecules)
```

### Materials Science

```python
from quantum.materials import Crystal
from quantum.core import MaterialsDatabase

# Create a crystal structure
silicon = Crystal.cubic_cell(
    atoms=["Si", "Si"],
    lattice_parameter=5.43,
    name="Silicon"
)

# Analyze properties
print(f"Volume: {silicon.get_volume():.2f} Ų")
print(f"Density: {silicon.get_density():.2f} g/cm³")

# Create supercell
supercell = silicon.supercell(2, 2, 2)
```

### Quantum Computing

```python
from quantum.computing import VQEOptimizer

# Set up VQE calculation
vqe = VQEOptimizer(
    ansatz="ucc",
    optimizer="cobyla",
    backend="qiskit_simulator"
)

# Run optimization
result = vqe.optimize(molecule)
print(f"Ground state energy: {result.ground_state_energy:.6f}")
```

## Command Line Interface

The quantum CLI provides access to all functionality:

```bash
# Run quantum chemistry calculation
quantum calculate molecule.xyz --method b3lyp --basis 6-31g*

# Train ML model
quantum train-ml data/ --model random_forest --features coulomb_matrix

# Predict properties
quantum predict molecule.xyz model.joblib

# Analyze crystal structure
quantum crystal-info structure.cif --supercell 2,2,2

# Show available methods
quantum --help
```

## Project Structure

```
quantum/
├── packages/                    # Modular packages
│   ├── quantum-core/           # Core functionality
│   │   └── src/quantum/core/
│   ├── quantum-chemistry/      # Quantum chemistry methods
│   │   └── src/quantum/chemistry/
│   ├── quantum-materials/      # Materials science tools
│   │   └── src/quantum/materials/
│   ├── quantum-computing/      # Quantum algorithms
│   │   └── src/quantum/computing/
│   ├── quantum-ml/            # Machine learning models
│   │   └── src/quantum/ml/
│   └── quantum-cli/           # Command-line interface
│       └── src/quantum/cli/
├── examples/                   # Example scripts
├── tests/                     # Test suite
├── docker/                    # Docker configurations
├── docs/                      # Documentation
├── shared/                    # Shared configurations
├── pyproject.toml            # Main project config
└── docker-compose.yml        # Development services
```

## Development

### Workspace Management

This project uses **uv workspaces** for managing multiple packages:

```bash
# Install all packages in development mode
uv sync --dev

# Add dependency to specific package
uv add numpy --package quantum-core

# Run tests for specific package
uv run pytest packages/quantum-chemistry/tests/

# Build specific package
uv build packages/quantum-core/
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=quantum

# Run specific package tests
uv run pytest packages/quantum-chemistry/tests/
uv run pytest packages/quantum-ml/tests/
```

### Code Quality

```bash
# Lint all packages
uv run ruff check packages/

# Format code
uv run ruff format packages/

# Type checking
uv run mypy packages/

# Run pre-commit hooks
uv run pre-commit run --all-files
```

### Docker Development

```bash
# Build all services
docker-compose build

# Run development environment
docker-compose up dev

# Run specific services
docker-compose up quantum-chemistry quantum-ml

# Scale services
docker-compose up --scale quantum-chemistry=3
```

## Package Dependencies

### quantum-core

- **numpy**: Numerical computing
- **scipy**: Scientific computing  
- **ase**: Atomic Simulation Environment
- **pymatgen**: Materials analysis
- **pydantic**: Data validation
- **qcelemental**: Quantum chemistry data

### quantum-chemistry

- **pyscf**: Quantum chemistry calculations
- **openfermion**: Quantum simulation tools

### quantum-materials

- **pymatgen**: Materials Project integration
- **phonopy**: Phonon calculations
- **spglib**: Space group analysis

### quantum-computing

- **qiskit**: Quantum circuits and algorithms
- **pennylane**: Quantum machine learning
- **cirq**: Google's quantum computing library

### quantum-ml

- **scikit-learn**: Classical ML algorithms
- **torch**: Deep learning
- **tensorflow**: Neural networks
- **rdkit**: Chemical informatics

## Contributing

We welcome contributions! This monorepo structure makes it easy to contribute to specific areas:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Choose your focus area** (e.g., `packages/quantum-chemistry/`)
4. **Make your changes** with tests
5. **Ensure quality checks pass**:

   ```bash
   uv run ruff check packages/
   uv run pytest packages/your-package/tests/
   ```

6. **Commit and push** (`git commit -m 'Add amazing feature'`)
7. **Open a Pull Request**

### Package-Specific Contributing

- **quantum-chemistry**: Implement new calculation methods, basis sets
- **quantum-materials**: Add materials property calculators, crystal analysis
- **quantum-computing**: Develop quantum algorithms, circuit optimizations  
- **quantum-ml**: Create new ML models, feature extractors
- **quantum-core**: Improve base classes, add converters, enhance utilities

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Quantum in your research, please cite:

```bibtex
@software{quantum2024,
  title={Quantum: A Modular Python Framework for Quantum Chemistry, Materials Science, and Quantum Computing},
  author={Chemistry Team},
  year={2024},
  url={https://github.com/yourusername/quantum}
}
```

## Acknowledgments

- Built with **uv workspaces** for modern Python package management
- **Docker services** for scalable development and deployment
- Integrated with leading scientific packages: **PySCF**, **PyMatGen**, **ASE**, **Qiskit**
- Thanks to all contributors and the quantum computing community

## Support

- **Documentation**: [quantum.readthedocs.io](https://quantum.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/yourusername/quantum/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/quantum/discussions)
- **Email**: <support@quantum.com>
