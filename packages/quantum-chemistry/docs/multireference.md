# Multireference Methods Benchmarking Protocol

## Overview

This protocol provides systematic instructions for benchmarking multireference (MR) quantum chemistry methods. The goal is to establish reliable accuracy assessments across different MR approaches for various chemical systems.

## 1. Method Selection Matrix

### 1.1 Primary Methods to Benchmark

| Method | Best For | Accuracy Expectation | Computational Cost |
|--------|----------|---------------------|-------------------|
| **CASSCF + NEVPT2/CASPT2** | Vertical excitations, mild MR | MAE ~0.11-0.13 eV | Moderate |
| **Selected CI (SHCI/CIPSI/ASCI)** | Near-FCI benchmarks, small-medium systems | Sub-milli-Hartree vs FCI | Variable |
| **AF-QMC (phaseless)** | Transition metals, strong correlation | Chemical accuracy (≤0.5 kcal/mol) | High |
| **DMRG + NEVPT2/CASPT2** | Large active spaces, challenging systems | High for static+dynamic correlation | High |

### 1.2 System-Specific Recommendations

- **Organic molecules, mild MR**: Start with CASSCF+NEVPT2 using APC active space selection
- **Transition metal complexes**: Use DMRG-CASSCF+NEVPT2 or AF-QMC
- **Bond-breaking, diradicals**: Selected CI → AF-QMC workflow
- **Large conjugated systems**: DMRG-based approaches

## 2. Software Implementation Strategy

### 2.1 Primary Software Stack

| Method | Recommended Software | License | Key Features |
|--------|---------------------|---------|--------------|
| **CASSCF/NEVPT2** | OpenMolcas, ORCA, PySCF | Open/Academic | Production-ready, automated workflows |
| **CASPT2** | OpenMolcas, ORCA | Open/Academic | Multiple variants (MS/XMS, IPEA shifts) |
| **Selected CI** | Quantum Package (CIPSI), Dice (SHCI) | Open source | Excellent parallelization |
| **AF-QMC** | QMCPACK, ipie | Open source | GPU acceleration available |
| **DMRG** | block2, CheMPS2, QCMAQUIS | Open source | Large active space capability |

### 2.2 Integration Setup

1. **Primary workflow**: PySCF + external solvers for maximum flexibility
2. **Production runs**: OpenMolcas for CASSCF/NEVPT2/CASPT2 workflows
3. **High-performance**: Quantum Package for CIPSI, block2 for DMRG
4. **Validation**: Cross-check results between at least two independent codes

## 3. Active Space Selection Protocol

### 3.1 Automated Methods Hierarchy

1. **Start with APC (Automated Pair Counting)**
   - Implementation: PySCF (`pyscf.mcscf.apc`)
   - Good for: Organic systems, excited states, high-throughput
   - Validation: Check energy convergence vs active space size

2. **Use autoCAS for challenging systems**
   - Implementation: Standalone + DMRG interface
   - Good for: Transition metals, strong correlation
   - Method: DMRG-based orbital entanglement analysis

3. **Dipole-moment based selection for state balance**
   - Implementation: Custom protocols in PySCF/OpenMolcas
   - Good for: Excited states, photochemistry

### 3.2 Active Space Validation

```bash
# Example validation workflow
1. Start with minimal active space (APC rank 1-6)
2. Systematically increase active space size
3. Monitor energy convergence: |E(n) - E(n+2)| < threshold
4. Check PT2 correction stability
5. Validate with larger basis set if needed
```

## 4. Systematic Benchmarking Protocol

### 4.1 Reference Data Generation

#### Step 1: Establish sCI References

```bash
# For small-medium systems (< 20 electrons active)
1. Run SHCI (Dice) or CIPSI (Quantum Package)
2. Target: PT2 correction < 1 mH
3. Extrapolate to FCI limit using E vs PT2 plots
4. Use these as "exact" references for method validation
```

#### Step 2: Cross-Method Validation

```bash
# Essential validation checks
1. Compare NEVPT2 vs CASPT2 on same active space
2. Validate DMRG results vs smaller active space sCI
3. Check AF-QMC trial wavefunction dependence
4. Basis set convergence studies
```

### 4.2 Benchmark Test Sets

#### Dataset 1: Vertical Excitations (QUESTDB subset)

- **Systems**: 500+ organic transitions
- **Method**: SA-CASSCF + NEVPT2/CASPT2 with APC
- **Metrics**: MAE, systematic error trends
- **Reference**: PMC9558375, doi:10.1021/acs.jctc.2c00630

#### Dataset 2: Bond Dissociation (sCI benchmarks)

- **Systems**: C₂, N₂, Cr₂ curves
- **Method**: SHCI/CIPSI → near-FCI extrapolation
- **Metrics**: Potential energy surfaces
- **Reference**: doi:10.1021/acs.jctc.6b00407

#### Dataset 3: Transition Metal Properties

- **Systems**: 3d TM diatomics, spin-state gaps
- **Method**: AF-QMC with sCI/DMRG trials
- **Metrics**: Bond dissociation energies, spin splittings
- **Reference**: PMC10413869

### 4.3 Statistical Analysis Requirements

```python
# Required error metrics for each benchmark
1. Mean Absolute Error (MAE)
2. Root Mean Square Error (RMSE)  
3. Maximum Absolute Error
4. Error distributions (histograms)
5. Method-specific systematic trends
6. Basis set dependence analysis
```

## 5. Implementation Workflow

### 5.1 Phase 1: Infrastructure Setup (Week 1-2)

1. **Software Installation**

   ```bash
   # Core packages
   - Install OpenMolcas + DMRG interface
   - Compile PySCF with external solver hooks
   - Set up Quantum Package 2.x
   - Build Dice/SHCI with PySCF interface
   ```

2. **Test System Validation**

   ```bash
   # Run standard test cases
   - H₂O CASSCF(6,4) + NEVPT2
   - C₂ dissociation with SHCI
   - Simple TM complex with DMRG
   ```

### 5.2 Phase 2: Method Calibration (Week 3-4)

1. **Active Space Protocols**

   ```bash
   # Establish thresholds for each system type
   - APC ranking cutoffs
   - autoCAS entanglement thresholds  
   - Convergence criteria for PT2 corrections
   ```

2. **Cross-Code Validation**

   ```bash
   # Compare implementations
   - NEVPT2: OpenMolcas vs ORCA vs PySCF
   - CASPT2: OpenMolcas vs ORCA
   - sCI: SHCI vs CIPSI
   ```

### 5.3 Phase 3: Production Benchmarking (Week 5-8)

1. **Systematic Studies**

   ```bash
   # Run all benchmark sets
   - Vertical excitations with automated APC
   - Bond breaking with sCI extrapolation
   - TM systems with AF-QMC
   ```

2. **Data Analysis Pipeline**

   ```python
   # Automated analysis workflow
   - Parse all output files
   - Generate error statistics
   - Create comparison plots
   - Identify systematic trends
   ```

## 6. Quality Control Checklist

### 6.1 Before Production Runs

- [ ] All software versions documented
- [ ] Test calculations reproduce literature values
- [ ] Active space selection protocols validated
- [ ] Error thresholds established

### 6.2 During Benchmarking

- [ ] Regular convergence monitoring
- [ ] Cross-method spot checks
- [ ] Basis set dependence verification
- [ ] Statistical significance testing

### 6.3 Final Validation

- [ ] All results reproduced independently
- [ ] Error bars and uncertainties reported
- [ ] Method limitations clearly documented
- [ ] Computational cost analysis included

## 7. Expected Deliverables

### 7.1 Technical Reports

1. **Method Accuracy Matrix**: MAE/RMSE for each method vs system type
2. **Computational Cost Analysis**: Wall-time vs system size scaling
3. **Software Recommendations**: Pros/cons of each implementation
4. **Active Space Guidelines**: When to use each selection method

### 7.2 Practical Guidelines

1. **Decision Tree**: Method selection based on system properties
2. **Runtime Estimates**: Expected costs for different system sizes
3. **Failure Mode Analysis**: When methods break down and alternatives

## 8. Success Metrics

- **Accuracy Target**: Reproduce literature benchmarks within ±0.05 eV
- **Coverage Goal**: Test at least 100 systems across all categories
- **Reproducibility**: All results independently verified
- **Documentation**: Complete protocols for team knowledge transfer

## References

Key papers to consult:

- Large-scale NEVPT2 benchmarking: PMC9558375
- Selected CI methods: doi:10.1021/acs.jctc.6b00407
- AF-QMC benchmarks: PMC10413869  
- DMRG applications: doi:10.1063/1.4892418
