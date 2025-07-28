# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a quantum chemistry package that provides a unified interface for quantum chemistry calculations using PySCF as the backend, with a primary focus on automated active space selection for multireference methods. The package implements multiple active space selection algorithms and provides consistent APIs for Hartree-Fock, DFT, and multireference calculations.

## Development Commands

### Package Management
- This project uses `uv` for Python package management
- Install package in development mode: `uv pip install -e .`
- Install with dev dependencies: `uv pip install -e .[dev]`

### Testing
- Run all tests: `pytest`
- Run specific test: `pytest tests/unit/test_active_space.py`
- Run with verbose output: `pytest -v`
- Skip slow tests: `pytest -m "not slow"`
- Skip integration tests: `pytest -m "not integration"`

### Code Quality
- Format code: `ruff format .`
- Lint code: `ruff check .`
- Type checking: `mypy src/quantum/chemistry/`
- Line length limit: 88 characters (configured in pyproject.toml)

## Architecture Overview

### Core Module Structure
- `src/quantum/chemistry/` - Main package directory
  - `__init__.py` - Package exports and version
  - `active_space.py` - Unified active space selection methods (primary module)
  - `hartree_fock.py` - HF method implementations (RHF, UHF, ROHF)
  - `dft.py` - DFT calculators (B3LYP, PBE, M06, wB97X-D)
  - `fcidump.py` - FCIDUMP file creation from active spaces  
  - `avas_defaults.yaml` - Default AVAS orbital configurations
  - `multireference/` - Multireference method implementations
    - `base.py` - Abstract base classes and common interfaces
    - `methods/` - Method implementations (CASSCF, NEVPT2, Selected CI, etc.)
    - `workflows/` - High-level workflow orchestration
    - `external/` - External software interfaces
    - `benchmarking/` - Benchmarking and analysis tools

### Active Space Selection Methods
The package implements 7+ active space selection algorithms:
1. **AVAS** (Atomic Valence Active Space) - overlap-based selection
2. **APC** (Atomic Population Coefficient) - ranked orbital approach
3. **DMET-CAS** - density matrix embedding theory
4. **Natural Orbitals from MP2** - correlation-based selection
5. **Boys/Pipek-Mezey Localization** - spatial localization
6. **IAO/IBO** (Intrinsic Atomic/Bond Orbitals) - chemically meaningful orbitals
7. **Energy Window** - HOMO-LUMO gap based selection

### Multireference Methods
The package provides implementations of major multireference quantum chemistry methods:

#### Available Methods
1. **CASSCF** (Complete Active Space Self-Consistent Field)
   - Implementation: `CASSCFMethod` class
   - Backend: PySCF CASSCF module
   - Use case: Static correlation, method calibration

2. **NEVPT2** (N-Electron Valence State Perturbation Theory)
   - Implementation: `NEVPT2Method` class  
   - Backend: PySCF NEVPT2 module
   - Use case: Dynamic correlation recovery, production calculations

3. **CASPT2** (Complete Active Space Perturbation Theory)
   - Implementation: `CASPT2Method` class (placeholder for external software)
   - Backend: OpenMolcas/ORCA interfaces (planned)
   - Use case: Alternative to NEVPT2 with different systematic errors

#### Planned Methods
- **Selected CI** (SHCI/CIPSI) for near-FCI benchmarks
- **AF-QMC** (Auxiliary Field Quantum Monte Carlo) for transition metals  
- **DMRG** (Density Matrix Renormalization Group) for large active spaces

#### Method Selection and Workflow
- **MethodSelector**: Automated method recommendation based on system properties
- **MultireferenceWorkflow**: High-level orchestration of complete calculations
- **Method Comparison**: Cross-validation between different approaches

### Key Classes and Functions

#### Active Space Selection
- `UnifiedActiveSpaceFinder` - Central class for method comparison and selection
- `ActiveSpaceResult` - Standard result container for all methods
- `ActiveSpaceMethod` - Enum defining available methods
- `auto_find_active_space()` - Automatic method selection with target sizing

#### Multireference Methods  
- `MultireferenceMethod` - Abstract base class for all MR methods
- `MultireferenceResult` - Unified result container for MR calculations
- `CASSCFMethod`, `NEVPT2Method`, `CASPT2Method` - Method implementations
- `MultireferenceWorkflow` - High-level calculation orchestration
- `MethodSelector` - Automated method recommendation system

### Calculation Flow
1. Perform mean-field calculation (HF/DFT)
2. Select active space using one of the implemented methods
3. `ActiveSpaceResult` provides orbital coefficients and metadata
4. **Multireference calculation** using automated method selection or manual choice
5. `MultireferenceResult` contains energies, properties, and analysis data
6. Optional FCIDUMP file generation for external programs
7. Automated benchmarking and cross-method validation

### Common Usage Patterns

#### Complete Automated Workflow
```python
from quantum.chemistry import MultireferenceWorkflow

workflow = MultireferenceWorkflow()
results = workflow.run_calculation(
    scf_obj,
    active_space_method="auto",
    mr_method="auto", 
    target_accuracy="standard"
)
```

#### Manual Method Selection
```python
from quantum.chemistry import CASSCFMethod, find_active_space_avas

active_space = find_active_space_avas(scf_obj, threshold=0.2)
casscf = CASSCFMethod()
result = casscf.calculate(scf_obj, active_space)
```

#### Method Comparison
```python
comparison = workflow.compare_methods(
    scf_obj, active_space, 
    methods=["casscf", "nevpt2"]
)
```

## Testing Architecture

### Test Organization
- `tests/conftest.py` - Shared fixtures (H2, H2O, Fe complex molecules)
- `tests/unit/` - Unit tests for individual modules
  - `test_active_space.py` - Active space selection tests
  - `test_fcidump.py` - FCIDUMP generation tests  
  - `test_multireference.py` - Multireference method tests
- Test molecules: H2 (simple), H2O (molecular), Fe complex (transition metal)

### Testing Multireference Methods
- Run multireference tests: `pytest tests/unit/test_multireference.py`
- Test individual methods: `pytest tests/unit/test_multireference.py::TestCASSCFMethod`
- All MR tests use existing molecular fixtures and active space selections

### Common Test Patterns
- All tests use PySCF `gto.Mole` objects from fixtures
- SCF calculations are pre-converged in fixtures
- Active space tests verify orbital counts and method-specific properties
- Use `sto-3g` and `minao` basis sets for fast testing

## Dependencies and Integration

### Core Dependencies
- `pyscf>=2.3.0` - Quantum chemistry backend
- `quantum-core` - Internal workspace dependency
- `numpy`, `scipy` - Numerical computing
- `pydantic>=2.0.0` - Data validation and models
- `pyyaml` - Configuration file parsing

### PySCF Integration Points
- Direct use of `pyscf.mcscf.avas` and `pyscf.mcscf.apc` modules
- Integration with `pyscf.scf` objects for mean-field inputs
- Orbital localization via `pyscf.lo` module
- FCIDUMP generation via `pyscf.tools.fcidump`

## Development Guidelines

### Code Organization
- Follow the established pattern: each method has a dedicated function (`find_active_space_*`)
- All methods return `ActiveSpaceResult` objects for consistency
- Use `UnifiedActiveSpaceFinder` for method comparison and automation
- Maintain backward compatibility with existing PySCF workflows

### Active Space Method Implementation
- Each method should handle both RHF and UHF mean-field objects
- Include proper error handling for method-specific requirements
- Provide sensible defaults but allow parameter customization
- Document method-specific parameters and their physical meaning

### Multireference Method Implementation
- Inherit from `MultireferenceMethod` abstract base class
- Implement required methods: `calculate()`, `estimate_cost()`, `_get_method_type()`
- Return `MultireferenceResult` objects with standardized metadata
- Handle both RHF and UHF inputs gracefully
- Provide cost estimation and parameter recommendation methods
- Include convergence monitoring and error handling

### Integration Patterns
- MR methods integrate seamlessly with existing `ActiveSpaceResult` objects
- Use `MultireferenceWorkflow` for high-level orchestration
- Maintain backward compatibility with direct PySCF usage
- Support both automated and manual method selection

### Testing Best Practices
- Use existing molecular fixtures from `conftest.py`
- Test both simple systems (H2) and chemically relevant cases (H2O, transition metals)
- Verify active space sizes and orbital characteristics
- Include edge cases and error conditions
- Test multireference method accuracy and convergence
- Validate integration between active space selection and MR calculations