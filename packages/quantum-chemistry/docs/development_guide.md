# Development Guide - Research-to-Development Cycle

This guide provides comprehensive instructions for using the research-to-development cycle infrastructure for validating and improving the multireference quantum chemistry implementation.

## Overview

The research-to-development cycle includes:

1. **Enhanced Testing Infrastructure** - Comprehensive test suite with benchmark validation
2. **Literature Validation** - QUESTDB standard benchmarks and cross-code validation  
3. **Containerization** - Docker and Singularity containers for reproducible environments
4. **Automated CI/CD** - Quality gates and continuous validation
5. **Benchmarking Suite** - Systematic accuracy assessment and performance monitoring

## Quick Start

### Using Docker for Development

```bash
# Start development environment
docker-compose up development

# Run interactive Jupyter environment
docker-compose up jupyter
# Access at http://localhost:8888

# Run comprehensive tests
docker-compose up testing

# Run benchmarking suite
docker-compose up benchmarking
```

### Using the Validation Framework

```python
from quantum.chemistry.multireference.benchmarking import (
    create_standard_benchmark_datasets,
    ValidationRunner,
    BenchmarkAnalyzer
)

# Create standard benchmark datasets
datasets = create_standard_benchmark_datasets()

# Run validation on your implementation
runner = ValidationRunner()
report = runner.generate_validation_report(
    datasets['questdb'],
    methods=['casscf', 'nevpt2'],
    max_systems=10
)

# Analyze results
analyzer = BenchmarkAnalyzer(datasets['questdb'])
stats = analyzer.calculate_error_statistics()
comparison = analyzer.compare_methods(['casscf', 'nevpt2'])
```

## Development Workflow

### 1. Code Development

1. **Make changes** to multireference methods or add new features
2. **Run local tests** to ensure basic functionality:
   ```bash
   pytest tests/unit/test_multireference.py -v
   ```

### 2. Validation Testing

3. **Run benchmark validation**:
   ```python
   # Test your changes against literature benchmarks
   from quantum.chemistry.multireference.benchmarking import ValidationRunner
   
   runner = ValidationRunner()
   # Test with small subset first
   datasets = create_standard_benchmark_datasets()
   report = runner.generate_validation_report(
       datasets['questdb'], 
       max_systems=5
   )
   
   print(f"Success rate: {report['validation_results']['summary_statistics']['success_rate']:.2%}")
   ```

### 3. Containerized Testing

4. **Test in clean environment**:
   ```bash
   # Test in Docker container
   docker-compose run testing
   
   # Or build and test production image
   docker build --target production -t quantum-chemistry:test .
   docker run --rm quantum-chemistry:test python -c "import quantum.chemistry; print('OK')"
   ```

### 4. Continuous Integration

5. **Push changes** - CI pipeline automatically runs:
   - Quality gates (linting, type checking)
   - Unit tests across Python versions
   - Integration tests
   - Docker build tests
   - Performance regression tests

## Benchmarking Infrastructure

### Standard Benchmark Datasets

The package includes three standard benchmark datasets:

#### 1. QUESTDB Subset (Vertical Excitations)
```python
questdb = datasets['questdb']
# Contains representative molecules for excited state calculations
# Reference: PMC9558375, doi:10.1021/acs.jctc.2c00630
```

#### 2. Transition Metal Complexes
```python
tm_dataset = datasets['transition_metals'] 
# Iron complexes and other transition metal systems
# Focus on spin-state energetics and strong correlation
```

#### 3. Bond Dissociation Curves
```python
bond_dataset = datasets['bond_dissociation']
# N2, C2 dissociation curves for multireference validation
# Challenge static correlation treatment
```

### Creating Custom Benchmarks

```python
from quantum.chemistry.multireference.benchmarking import BenchmarkDatasetBuilder

# Build custom dataset
builder = BenchmarkDatasetBuilder()
custom_dataset = (builder
    .set_metadata("my_benchmark", "Custom benchmark for X systems")
    .add_questdb_subset()  # Add standard molecules
    .build())

# Add your own molecules
from quantum.chemistry.multireference.benchmarking import BenchmarkMolecule, SystemType

my_molecule = BenchmarkMolecule(
    name="my_system",
    atoms=[("C", 0.0, 0.0, 0.0), ("O", 0.0, 0.0, 1.2)],
    charge=0,
    multiplicity=1,
    basis_set="cc-pVDZ",
    system_type=SystemType.ORGANIC,
    source="my_research",
    theoretical_references={"energy": -113.123456}
)

custom_dataset.entries.append(BenchmarkEntry(
    system=my_molecule,
    method="reference",
    active_space_method="manual",
    n_active_electrons=8,
    n_active_orbitals=6,
    energy=0.0,  # Will be calculated
    reference_energy=-113.123456
))
```

## Validation and Analysis

### Running Comprehensive Validation

```python
# Full validation with analysis
runner = ValidationRunner()

# Generate comprehensive report
report = runner.generate_validation_report(
    dataset=datasets['questdb'],
    methods=['casscf', 'nevpt2'],
    max_systems=None  # Test all systems
)

# Check results
print(f"Overall grade: {report['quality_assessment']['overall_grade']}")
print(f"Success rate: {report['validation_results']['summary_statistics']['success_rate']:.2%}")

# Get recommendations
for rec in report['recommendations']:
    print(f"- {rec}")
```

### Statistical Analysis

```python
# Detailed error analysis
analyzer = BenchmarkAnalyzer(datasets['questdb'])

# Compare methods
comparison = analyzer.compare_methods(['casscf', 'nevpt2'])
for method, stats in comparison.items():
    print(f"{method}: MAE = {stats['mean_absolute_error']:.3f} eV")

# Check for systematic errors
systematic = analyzer.systematic_error_analysis('nevpt2')
if systematic['systematic_bias']['significant_bias']:
    direction = systematic['systematic_bias']['bias_direction']
    print(f"NEVPT2 shows systematic {direction} bias")

# Identify problem cases
problems = analyzer.identify_problem_cases(error_threshold=0.2)  # > 0.2 eV error
for case in problems[:5]:  # Show worst 5
    print(f"Problem: {case['system_name']} - Error: {case['error_ev']:.3f} eV")
```

## Container Usage

### Development with Docker

```bash
# Start development environment
docker-compose up -d development
docker-compose exec development bash

# Inside container:
cd /opt/quantum-chemistry
python -c "
from quantum.chemistry.multireference import MultireferenceWorkflow
workflow = MultireferenceWorkflow()
print('Development environment ready!')
"
```

### HPC with Singularity

```bash  
# Build Singularity container
singularity build quantum-chemistry.sif Singularity.def

# Run on HPC cluster
singularity exec quantum-chemistry.sif python my_calculation.py

# Interactive session
singularity shell quantum-chemistry.sif
```

### Production Deployment

```bash
# Build optimized production image
docker build --target production -t quantum-chemistry:v1.0 .

# Run calculation
docker run --rm -v $(pwd)/data:/opt/data quantum-chemistry:v1.0 \
    python -c "
    from quantum.chemistry.multireference.workflows import MultireferenceWorkflow
    # Your calculation code here
    "
```

## Performance Monitoring

### Benchmarking Performance

```python
import time
from quantum.chemistry.multireference.benchmarking import ValidationRunner

# Performance test
runner = ValidationRunner()
start_time = time.time()

report = runner.generate_validation_report(
    datasets['questdb'],
    methods=['casscf'],
    max_systems=10
)

elapsed = time.time() - start_time
timing = report['validation_results']['summary_statistics'].get('timing', {})

print(f"Total time: {elapsed:.2f}s")
print(f"Mean calculation time: {timing.get('mean_total_time', 0):.2f}s")
print(f"Success rate: {report['validation_results']['summary_statistics']['success_rate']:.2%}")
```

### Continuous Monitoring

The CI pipeline includes automated performance regression tests:

- **Daily benchmarks** (via cron schedule)
- **Performance alerts** if calculations take >300s
- **Success rate monitoring** with alerts if <85%
- **Memory usage tracking** and optimization suggestions

## Quality Gates

### Pre-commit Checks

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Manual check
pre-commit run --all-files
```

### CI/CD Pipeline Stages

1. **Quality Gates**: Linting, type checking, import validation
2. **Unit Tests**: Core functionality across Python 3.10-3.12
3. **Integration Tests**: Full workflow validation
4. **Docker Tests**: Container build and execution
5. **Performance Tests**: Benchmark timing and regression detection
6. **Coverage**: Code coverage reporting

### Success Criteria

- **All tests pass**: 100% success rate required
- **Code coverage**: >90% for multireference methods
- **Performance**: No >20% regression from baseline
- **Validation**: >85% success rate on benchmark datasets

## Troubleshooting

### Common Issues

1. **SCF Convergence Failures**
   ```python
   # Increase SCF cycles and adjust convergence
   runner = ValidationRunner(max_scf_cycles=200)
   ```

2. **Memory Issues**
   ```bash
   # Use Docker with memory limits
   docker run --memory=8g quantum-chemistry:dev
   ```

3. **Container Build Failures**
   ```bash
   # Check system dependencies
   docker build --no-cache --target base .
   ```

### Getting Help

- **Error Analysis**: Use `BenchmarkAnalyzer.identify_problem_cases()`
- **Validation Reports**: Check `quality_assessment` section
- **Performance Issues**: Review timing statistics in reports
- **Container Issues**: Check logs with `docker-compose logs`

## Contributing

When contributing new features:

1. **Add tests** to appropriate test files
2. **Update benchmarks** if adding new methods
3. **Run validation** to ensure accuracy maintained
4. **Update documentation** and examples
5. **Test containerization** works correctly

The research-to-development cycle ensures that all contributions maintain high quality and accuracy standards while providing reproducible environments for development and deployment.