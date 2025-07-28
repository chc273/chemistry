#!/usr/bin/env python3
"""
External Multireference Methods Usage Example

This example demonstrates how to use the external quantum chemistry methods
integrated through Docker containers and native installations.
"""

import numpy as np
from pyscf import gto, scf

# Import the quantum chemistry framework
from quantum.chemistry.active_space import find_active_space_avas
from quantum.chemistry.multireference import MultireferenceWorkflow

# Import external methods (optional - will be available if containers are built)
try:
    from quantum.chemistry.multireference.external import (
        DMRGMethod,
        AFQMCMethod,
        SelectedCIMethod,
    )
    from quantum.chemistry.multireference.external.openmolcas import CASPT2Method
    EXTERNAL_METHODS_AVAILABLE = True
except ImportError as e:
    print(f"External methods not available: {e}")
    EXTERNAL_METHODS_AVAILABLE = False


def setup_test_molecule():
    """Set up a test molecule for calculations."""
    # Create a simple water molecule
    mol = gto.Mole()
    mol.atom = '''
        O  0.0000  0.0000  0.0000
        H  0.7571  0.0000  0.5861
        H -0.7571  0.0000  0.5861
    '''
    mol.basis = 'sto-3g'
    mol.build()
    
    # Run SCF calculation
    mf = scf.RHF(mol)
    mf.kernel()
    
    print(f"SCF energy: {mf.e_tot:.6f} Hartree")
    return mol, mf


def test_core_methods(mf):
    """Test core multireference methods (always available)."""
    print("\n" + "="*50)
    print("Testing Core Multireference Methods")
    print("="*50)
    
    # Find active space using AVAS
    active_space = find_active_space_avas(mf, threshold=0.2)
    print(f"Active space: ({active_space.n_active_electrons}, {active_space.n_active_orbitals})")
    
    # Use the unified workflow
    workflow = MultireferenceWorkflow()
    
    # Test CASSCF
    print("\n--- CASSCF Calculation ---")
    results = workflow.run_calculation(
        mf,
        active_space_method="avas",
        mr_method="casscf",
        target_accuracy="standard"
    )
    print(f"CASSCF Energy: {results['multireference_result'].energy:.6f} Hartree")
    print(f"Correlation Energy: {results['multireference_result'].correlation_energy:.6f} Hartree")
    
    # Test NEVPT2
    print("\n--- NEVPT2 Calculation ---")
    results = workflow.run_calculation(
        mf,
        active_space_method="avas", 
        mr_method="nevpt2",
        target_accuracy="high"
    )
    print(f"NEVPT2 Energy: {results['multireference_result'].energy:.6f} Hartree")
    print(f"Correlation Energy: {results['multireference_result'].correlation_energy:.6f} Hartree")


def test_external_methods(mf):
    """Test external multireference methods (if available)."""
    if not EXTERNAL_METHODS_AVAILABLE:
        print("\n" + "="*50)
        print("External Methods Not Available")
        print("="*50)
        print("To enable external methods:")
        print("1. Build Docker containers: cd docker && docker-compose -f docker-compose.external.yml build")
        print("2. Or install natively using: docker/install-scripts/install-*.sh")
        return
    
    print("\n" + "="*50)
    print("Testing External Multireference Methods")
    print("="*50)
    
    # Find active space
    active_space = find_active_space_avas(mf, threshold=0.2)
    print(f"Active space: ({active_space.n_active_electrons}, {active_space.n_active_orbitals})")
    
    # Test Block2 DMRG
    print("\n--- Block2 DMRG Calculation ---")
    try:
        dmrg = DMRGMethod(
            bond_dimension=500,
            max_sweeps=10,
            post_correction=None
        )
        result = dmrg.calculate(mf, active_space)
        print(f"DMRG Energy: {result.energy:.6f} Hartree")
        print(f"Correlation Energy: {result.correlation_energy:.6f} Hartree")
        print(f"Converged: {result.convergence_info['converged']}")
    except Exception as e:
        print(f"DMRG calculation failed: {e}")
    
    # Test OpenMolcas CASPT2
    print("\n--- OpenMolcas CASPT2 Calculation ---")
    try:
        caspt2 = CASPT2Method(
            ipea_shift=0.0,
            multistate=False
        )
        result = caspt2.calculate(mf, active_space)
        print(f"CASPT2 Energy: {result.energy:.6f} Hartree")
        print(f"Correlation Energy: {result.correlation_energy:.6f} Hartree")
        print(f"Converged: {result.convergence_info['converged']}")
    except Exception as e:
        print(f"CASPT2 calculation failed: {e}")
    
    # Test ipie AF-QMC
    print("\n--- ipie AF-QMC Calculation ---")
    try:
        afqmc = AFQMCMethod(
            backend="ipie",
            n_walkers=50,
            n_steps=100,  # Small for demo
            timestep=0.01
        )
        result = afqmc.calculate(mf, active_space)
        print(f"AF-QMC Energy: {result.energy:.6f} ± {result.active_space_info.get('error_bars', {}).get('energy', 0.0):.6f} Hartree")
        print(f"Correlation Energy: {result.correlation_energy:.6f} Hartree")
        print(f"Statistical Error: {result.convergence_info.get('statistical_error', 'N/A')}")
    except Exception as e:
        print(f"AF-QMC calculation failed: {e}")
    
    # Test Selected CI (SHCI)
    print("\n--- Dice SHCI Calculation ---")
    try:
        shci = SelectedCIMethod(
            method_type="shci",
            pt2_threshold=1e-4,
            max_determinants=10000,  # Small for demo
            extrapolate_fci=False
        )
        result = shci.calculate(mf, active_space)
        print(f"SHCI Energy: {result.energy:.6f} Hartree")
        print(f"Correlation Energy: {result.correlation_energy:.6f} Hartree")
        print(f"N Determinants: {result.active_space_info.get('sci_data', {}).get('n_determinants', 'N/A')}")
    except Exception as e:
        print(f"SHCI calculation failed: {e}")


def test_method_comparison(mf):
    """Compare multiple methods on the same system."""
    print("\n" + "="*50)
    print("Method Comparison")
    print("="*50)
    
    active_space = find_active_space_avas(mf, threshold=0.2)
    
    # Use workflow for automatic method comparison
    workflow = MultireferenceWorkflow()
    
    methods_to_compare = ["casscf", "nevpt2"]
    if EXTERNAL_METHODS_AVAILABLE:
        # Add external methods if available
        methods_to_compare.extend(["dmrg", "caspt2"])
    
    print("Comparing methods:", methods_to_compare)
    print(f"SCF Energy: {mf.e_tot:.6f} Hartree")
    print("-" * 50)
    
    for method in methods_to_compare:
        try:
            results = workflow.run_calculation(
                mf,
                active_space_method="avas",
                mr_method=method,
                target_accuracy="standard"
            )
            energy = results['multireference_result'].energy
            corr_energy = results['multireference_result'].correlation_energy
            print(f"{method.upper():>10}: {energy:.6f} Hartree (ΔE_corr = {corr_energy:.6f})")
        except Exception as e:
            print(f"{method.upper():>10}: Failed ({str(e)[:50]}...)")


def display_installation_info():
    """Display information about available methods and installation."""
    print("\n" + "="*50)
    print("Installation Status")
    print("="*50)
    
    # Check core methods
    print("\n--- Core Methods (Always Available) ---")
    try:
        from quantum.chemistry.multireference import CASSCFMethod, NEVPT2Method
        print("✅ CASSCF - PySCF-based CASSCF calculations")
        print("✅ NEVPT2 - PySCF-based NEVPT2 calculations")
    except ImportError:
        print("❌ Core methods not available - framework installation issue")
    
    # Check external methods
    print("\n--- External Methods ---")
    
    # ipie (simple Python package)
    try:
        import ipie
        print(f"✅ ipie AF-QMC - Version {getattr(ipie, '__version__', 'unknown')}")
    except ImportError:
        print("❌ ipie AF-QMC - Install with: uv add ipie")
    
    # Container-based methods
    external_methods = [
        ("Block2 DMRG", "quantum-chemistry/block2:latest"),
        ("OpenMolcas CASPT2", "quantum-chemistry/openmolcas:latest"), 
        ("Dice SHCI", "quantum-chemistry/dice:latest"),
        ("Quantum Package CIPSI", "quantum-chemistry/quantum-package:latest"),
    ]
    
    import subprocess
    docker_available = bool(shutil.which("docker"))
    
    if not docker_available:
        print("❌ Docker not available - external methods require Docker or native installation")
    else:
        for method_name, container_image in external_methods:
            try:
                result = subprocess.run(
                    ["docker", "image", "inspect", container_image],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    print(f"✅ {method_name} - Container available")
                else:
                    print(f"❌ {method_name} - Container not built")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                print(f"❌ {method_name} - Container check failed")
    
    print("\n--- Installation Commands ---")
    print("Build all Docker containers:")
    print("  cd docker && docker-compose -f docker-compose.external.yml build")
    print("\nNative installation (individual packages):")
    print("  ./docker/install-scripts/install-block2.sh")
    print("  ./docker/install-scripts/install-openmolcas.sh") 
    print("  ./docker/install-scripts/install-dice.sh")
    print("  ./docker/install-scripts/install-quantum-package.sh")


def main():
    """Main example execution."""
    print("Quantum Chemistry External Methods Usage Example")
    print("=" * 50)
    
    # Display installation status
    display_installation_info()
    
    # Set up test system
    mol, mf = setup_test_molecule()
    
    # Test core methods (always available)
    test_core_methods(mf)
    
    # Test external methods (if available)
    test_external_methods(mf)
    
    # Compare methods
    test_method_comparison(mf)
    
    print("\n" + "="*50)
    print("Example completed!")
    print("="*50)


if __name__ == "__main__":
    import shutil
    main()