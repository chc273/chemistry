# Quantum Chemistry Package

This package provides a unified interface for quantum chemistry calculations using PySCF as the backend, with a focus on automated active space selection for multireference methods.

## Features

- **Hartree-Fock Methods**: RHF, UHF, ROHF calculations
- **Density Functional Theory**: Various DFT functionals (B3LYP, PBE, M06, wB97X-D)
- **Active Space Selection**: Comprehensive suite of automated active space selection methods
- **Unified Interface**: Consistent API across all methods for easy comparison and automation

## Active Space Selection Methods

Active space selection is crucial for multireference quantum chemistry methods like CASSCF, CASCI, and DMRG. This package provides a unified interface to multiple automated active space selection algorithms.

### Available Methods

#### 1. AVAS (Atomic Valence Active Space)

**Principle**: AVAS selects active space orbitals based on their overlap with atomic valence orbitals. It projects molecular orbitals onto minimal basis set atomic orbitals to identify orbitals with significant atomic character.

**Mechanism**:

- Projects MOs onto minimal atomic orbital basis (minao)
- Selects orbitals with overlap above a threshold
- Automatically includes valence orbitals for specified atoms

**Usage**:

```python
from quantum.chemistry import find_active_space_avas

# Automatic selection using default valence orbitals
result = find_active_space_avas(mf, system)

# Manual specification of atoms and orbitals
result = find_active_space_avas(
    mf, system, 
    avas_atoms=['Fe', 'C'],  # Element symbols
    ao_labels=['Fe 3d', 'C 2p'],  # Explicit orbital labels
    threshold=0.2
)
```

**Best for**: Transition metal complexes, systems where chemical intuition suggests specific atomic orbitals are important.

**Reference**: [PySCF AVAS Documentation](https://pyscf.org/user/mcscf.html#avas)

#### 2. APC (Approximate Pair Coefficient)

**Principle**: APC uses a ranked-orbital approach that estimates orbital importance through pair-interaction analysis of Hartree-Fock exchange matrix elements and orbital energies. It iteratively promotes virtual orbitals to identify strongly correlated orbitals.

**Mechanism**:

- Calculates approximate pair coefficients from HF exchange matrix
- Computes orbital "entropies" based on pair interactions
- Ranks orbitals by importance and selects top-ranked orbitals
- Uses iterative APC-N approach for improved selection

**Usage**:

```python
from quantum.chemistry import find_active_space_apc

# Basic APC selection
result = find_active_space_apc(
    mf, system,
    max_size=(8, 8),  # (nelec, norb) or max_norb
    n=2  # Number of APC iterations
)
```

**Best for**: General-purpose automated selection, high-throughput studies, systems without clear chemical intuition for active space.

**References**:

- [Ranked-Orbital Approach](https://doi.org/10.1021/acs.jctc.1c00037)
- [Large-Scale Benchmarking](https://doi.org/10.1021/acs.jctc.2c00630)

#### 3. DMET-CAS (Density Matrix Embedding Theory)

**Principle**: Uses density matrix embedding theory to identify orbitals with significant correlation by analyzing the density matrix structure.

**Mechanism**:

- Constructs density matrix from mean-field calculation
- Uses DMET analysis to identify strongly correlated orbitals
- Selects orbitals based on density matrix eigenvalues

**Usage**:

```python
from quantum.chemistry import find_active_space_dmet_cas

result = find_active_space_dmet_cas(
    mf, system,
    target_atoms=['Fe', 'O']  # Focus on specific atoms
)
```

**Best for**: Strongly correlated systems, transition metal complexes.

#### 4. Natural Orbitals from MP2

**Principle**: Uses MP2 natural orbital occupations to identify orbitals with significant correlation effects. Orbitals with occupations significantly different from 0 or 2 indicate strong correlation.

**Mechanism**:

- Performs MP2 calculation to get natural orbitals
- Analyzes natural orbital occupation numbers (NOONs)
- Selects orbitals with occupations between thresholds

**Usage**:

```python
from quantum.chemistry import find_active_space_natural_orbitals

result = find_active_space_natural_orbitals(
    mf, system,
    occupation_threshold=0.02,  # Minimum occupation for selection
    max_orbitals=10
)
```

**Best for**: Systems with clear correlation signatures, basis for other multireference methods.

#### 5. Boys/Pipek-Mezey Localization

**Principle**: Uses orbital localization to identify spatially localized orbitals, then selects those localized on specific atoms or regions.

**Mechanism**:

- Localizes orbitals within an energy window
- Boys: Minimizes orbital spread (maximizes localization)
- Pipek-Mezey: Maximizes orbital populations on atoms
- Selects orbitals with significant character on target atoms

**Usage**:

```python
from quantum.chemistry import UnifiedActiveSpaceFinder

finder = UnifiedActiveSpaceFinder()

# Boys localization
result = finder.find_active_space(
    'boys', mf, system,
    energy_window=(2.0, 2.0),  # HOMO-2 to LUMO+2
    localization_threshold=0.3
)

# Pipek-Mezey localization  
result = finder.find_active_space(
    'pipek_mezey', mf, system,
    energy_window=(2.0, 2.0),
    localization_threshold=0.3
)
```

**Best for**: Systems where spatial localization is important, bond-breaking processes.

#### 6. IAO/IBO (Intrinsic Atomic/Bond Orbitals)

**Principle**: Uses intrinsic atomic orbitals (IAO) and intrinsic bond orbitals (IBO) to provide chemically meaningful orbital representations.

**Mechanism**:

- IAO: Constructs localized atomic orbitals using minimal basis projections
- IBO: Uses IAO to construct bond orbitals with maximum localization
- Selects orbitals based on overlap with target regions

**Usage**:

```python
from quantum.chemistry import find_active_space_iao, find_active_space_ibo

# IAO-based selection
iao_result = find_active_space_iao(
    mf, system,
    target_atoms=['C', 'O'],
    energy_window=(2.0, 2.0)
)

# IBO-based selection
ibo_result = find_active_space_ibo(
    mf, system,
    target_atoms=['C', 'O'],
    energy_window=(2.0, 2.0)
)
```

**Best for**: Chemical analysis, understanding bonding patterns, organic molecules.

#### 7. Energy Window Selection

**Principle**: Selects orbitals within a specified energy range around the HOMO-LUMO gap.

**Mechanism**:

- Defines energy window around Fermi level
- Selects all orbitals within the window
- Simple but effective for many systems

**Usage**:

```python
from quantum.chemistry import find_active_space_energy_window

result = find_active_space_energy_window(
    mf, system,
    energy_window=(2.0, 2.0),  # ±2 eV around HOMO-LUMO
    max_orbitals=12
)
```

**Best for**: Quick initial screening, systems with clear HOMO-LUMO gap structure.

### Unified Interface and Automation

#### Comparing Multiple Methods

```python
from quantum.chemistry import UnifiedActiveSpaceFinder

finder = UnifiedActiveSpaceFinder()

# Compare multiple methods
methods = ['avas', 'apc', 'natural_orbitals', 'energy_window']
results = finder.compare_methods(methods, mf, system)

for method, result in results.items():
    print(f"{method}: ({result.n_active_electrons}, {result.n_active_orbitals})")
```

#### Automatic Selection

```python
from quantum.chemistry import auto_find_active_space

# Automatic selection with target size
result = auto_find_active_space(
    mf, system,
    target_size=(6, 6),  # Target (nelec, norb)
    priority_methods=['avas', 'apc', 'natural_orbitals']
)

print(f"Best method: {result.method}")
print(f"Active space: ({result.n_active_electrons}, {result.n_active_orbitals})")
```

### Method Selection Guidelines

| System Type | Recommended Methods | Notes |
|-------------|-------------------|-------|
| Transition Metal Complexes | AVAS, APC, DMET-CAS | Chemical intuition important |
| Organic Molecules | Natural Orbitals, IAO/IBO, Boys | Focus on bonding orbitals |
| High-Throughput Studies | APC, Energy Window | Automated, robust |
| Strongly Correlated Systems | APC, DMET-CAS, Natural Orbitals | Capture correlation effects |
| π-Systems | AVAS (2pz), Energy Window | Target π-orbitals specifically |
| Bond Breaking | Localization methods, Natural Orbitals | Spatial and correlation analysis |

### Integration with CASSCF/CASCI

All active space selection methods return `ActiveSpaceResult` objects that can be directly used with CASSCF calculations:

```python
from quantum.chemistry import find_active_space_avas
from pyscf import mcscf

# Select active space
result = find_active_space_avas(mf, system, threshold=0.2)

# Use in CASSCF calculation
casscf = mcscf.CASSCF(mf, result.n_active_orbitals, result.n_active_electrons)
casscf.kernel(result.orbital_coefficients)

print(f"CASSCF Energy: {casscf.e_tot}")
```

### Visualization and Analysis

```python
# Export active orbitals for visualization
finder.export_molden(result, mf, "active_orbitals.molden")

# Analyze selection quality
print(f"Method: {result.method}")
print(f"Active space: ({result.n_active_electrons}, {result.n_active_orbitals})")
if result.selection_scores is not None:
    print(f"Score range: {result.selection_scores.min():.3f} - {result.selection_scores.max():.3f}")
```

## Installation

```bash
# Install base package
pip install -e packages/quantum-chemistry

# Install with all dependencies
pip install -e packages/quantum-chemistry[all]
```

## Requirements

- Python ≥ 3.8
- PySCF ≥ 2.0
- NumPy
- PyYAML
- Pydantic

## Examples

See the `examples/` directory for comprehensive examples demonstrating all active space selection methods.

## Contributing

Contributions are welcome! Please see the main repository documentation for guidelines.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

1. King, D. S.; Gagliardi, L. "A Ranked-Orbital Approach to Select Active Spaces for High-Throughput Multireference Computation" *J. Chem. Theory Comput.* **2021**, *17*, 4006-4020. [DOI: 10.1021/acs.jctc.1c00037](https://doi.org/10.1021/acs.jctc.1c00037)

2. King, D. S.; Truhlar, D. G.; Gagliardi, L. "Large-Scale Benchmarking of Multireference Vertical-Excitation Calculations via Automated Active-Space Selection" *J. Chem. Theory Comput.* **2022**, *18*, 6065-6076. [DOI: 10.1021/acs.jctc.2c00630](https://doi.org/10.1021/acs.jctc.2c00630)

3. Sayfutyarova, E. R.; Sun, Q.; Chan, G. K.-L.; Knizia, G. "Automated Construction of Molecular Active Spaces from Atomic Valence Orbitals" *J. Chem. Theory Comput.* **2017**, *13*, 4063-4078.

4. PySCF Documentation: [https://pyscf.org/](https://pyscf.org/)
