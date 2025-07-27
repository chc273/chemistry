# QuantChem

A comprehensive Python package for quantum chemistry, materials science, and quantum computing with integrated machine learning capabilities.

## Features

### 🧬 Quantum Chemistry
- **Multiple calculation methods**: Hartree-Fock, DFT, Coupled Cluster
- **Flexible basis sets**: Support for standard quantum chemistry basis sets
- **Property calculations**: Energies, molecular orbitals, dipole moments, polarizabilities
- **Geometry optimization**: Structure optimization with various algorithms

### 🔬 Materials Science
- **Crystal structure handling**: Create, manipulate, and analyze crystal structures
- **Materials database integration**: Interface with materials databases
- **Electronic structure calculations**: Band structures, density of states
- **Phonon calculations**: Vibrational analysis for crystals

### ⚛️ Quantum Computing
- **Variational algorithms**: VQE, QAOA implementations
- **Quantum circuit building**: Flexible quantum circuit construction
- **Multiple backends**: Support for Qiskit, PennyLane, and classical simulators
- **Noise modeling**: Realistic quantum device simulation

### 🤖 Machine Learning Integration
- **Property prediction**: ML models for molecular and materials properties
- **Feature extraction**: Multiple molecular and crystal descriptors
- **Model types**: Random Forest, Neural Networks, with uncertainty quantification
- **Quantum ML**: Integration of quantum computing with machine learning

## Installation

### Using uv (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/yourusername/quantchem.git
cd quantchem
uv sync

# Install in development mode
uv pip install -e .
```

### Using pip

```bash
git clone https://github.com/yourusername/quantchem.git
cd quantchem
pip install -e .
```

## Quick Start

### Basic Molecular Calculation

```python
import numpy as np
from quantchem import Molecule, ComputationEngine, HartreeFockCalculator

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
from quantchem.ml import MolecularML

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
from quantchem import Crystal, MaterialsDatabase

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
from quantchem.quantum_computing import VQEOptimizer

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

QuantChem provides a convenient CLI for common tasks:

```bash
# Run quantum chemistry calculation
quantchem calculate molecule.xyz --method b3lyp --basis 6-31g*

# Train ML model
quantchem train-ml data/ --model random_forest --features coulomb_matrix

# Predict properties
quantchem predict molecule.xyz model.joblib

# Analyze crystal structure
quantchem crystal-info structure.cif --supercell 2,2,2
```

## Examples

Check out the `examples/` directory for comprehensive tutorials:

- `01_basic_molecular_calculations.py` - Basic quantum chemistry workflows
- `02_machine_learning_workflow.py` - ML model training and prediction
- `03_materials_science.py` - Crystal structure analysis
- `04_quantum_computing.py` - VQE and quantum algorithms
- `05_integrated_workflow.py` - Combined QC, ML, and QC workflows

## Documentation

Full documentation is available at [quantchem.readthedocs.io](https://quantchem.readthedocs.io).

## Development

### Setting up Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/quantchem.git
cd quantchem

# Install development dependencies
uv sync --dev

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Check code quality
ruff check src/
ruff format src/
mypy src/
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=quantchem

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
```

### Code Style

This project uses:
- **ruff** for linting and formatting (line length: 88)
- **mypy** for type checking
- **pre-commit** for automated code quality checks

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Dependencies

### Core Dependencies
- **numpy**: Numerical computing
- **scipy**: Scientific computing
- **pandas**: Data manipulation
- **pydantic**: Data validation

### Quantum Chemistry
- **pyscf**: Quantum chemistry calculations
- **openfermion**: Quantum simulation tools

### Materials Science
- **pymatgen**: Materials analysis
- **ase**: Atomic simulation environment
- **phonopy**: Phonon calculations

### Quantum Computing
- **qiskit**: Quantum circuits and algorithms
- **pennylane**: Quantum machine learning
- **cirq**: Google's quantum computing library

### Machine Learning
- **scikit-learn**: Classical ML algorithms
- **torch**: Deep learning
- **tensorflow**: Neural networks

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use QuantChem in your research, please cite:

```bibtex
@software{quantchem2024,
  title={QuantChem: A Comprehensive Package for Quantum Chemistry, Materials Science, and Quantum Computing},
  author={Chemistry Team},
  year={2024},
  url={https://github.com/yourusername/quantchem}
}
```

## Acknowledgments

- Built with love for the quantum chemistry and materials science communities
- Inspired by leading packages like PySCF, Qiskit, and PyMatGen
- Thanks to all contributors and users

## Support

- **Documentation**: [quantchem.readthedocs.io](https://quantchem.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/yourusername/quantchem/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/quantchem/discussions)
- **Email**: support@quantchem.com
