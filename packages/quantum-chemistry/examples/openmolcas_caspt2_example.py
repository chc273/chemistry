#!/usr/bin/env python3
"""
OpenMolcas CASPT2 Integration Example

This example demonstrates the complete CASPT2 integration with OpenMolcas,
including input generation, output parsing, and cross-method validation.
"""

import numpy as np
from pyscf import gto, scf

from quantum.chemistry.active_space import find_active_space_avas
from quantum.chemistry.multireference.external.openmolcas import (
    CASPT2Method,
    OpenMolcasInputGenerator,
    OpenMolcasOutputParser,
    OpenMolcasValidator,
    OpenMolcasParameters,
)


def main():
    """Demonstrate OpenMolcas CASPT2 integration capabilities."""
    
    print("OpenMolcas CASPT2 Integration Example")
    print("=" * 50)
    
    # Set up a simple molecule (water)
    print("\n1. Setting up water molecule...")
    mol = gto.Mole()
    mol.atom = """
    O  0.0000  0.0000  0.0000
    H  0.7571  0.0000  0.5861
    H -0.7571  0.0000  0.5861
    """
    mol.basis = "sto-3g"
    mol.build()
    
    # Run SCF calculation
    print("2. Running SCF calculation...")
    mf = scf.RHF(mol)
    mf.kernel()
    print(f"   SCF energy: {mf.e_tot:.6f} Hartree")
    
    # Find active space
    print("\n3. Finding active space with AVAS...")
    active_space = find_active_space_avas(mf, threshold=0.2)
    print(f"   Active space: ({active_space.n_active_electrons},{active_space.n_active_orbitals})")
    print(f"   Selection method: {active_space.method}")
    
    # Demonstrate input generation
    print("\n4. Generating OpenMolcas input files...")
    generator = OpenMolcasInputGenerator()
    
    # Generate CASPT2 input
    caspt2_input = generator.generate_input(
        mf, active_space, 
        calculation_type="caspt2",
        ipea_shift=0.0,
        imaginary_shift=0.0
    )
    
    print("   CASPT2 input file generated:")
    print("   " + "─" * 40)
    for i, line in enumerate(caspt2_input.split('\n')[:15]):
        print(f"   {line}")
    print("   ... (truncated)")
    
    # Generate MS-CASPT2 input
    ms_caspt2_input = generator.generate_input(
        mf, active_space,
        calculation_type="ms_caspt2",
        n_states=3,
        multistate=True,
        ipea_shift=0.25
    )
    
    print("\n   MS-CASPT2 input file generated with 3 states")
    
    # Demonstrate parameter validation
    print("\n5. Parameter validation example...")
    try:
        params = OpenMolcasParameters(
            n_active_electrons=active_space.n_active_electrons,
            n_active_orbitals=active_space.n_active_orbitals,
            charge=0,
            spin_multiplicity=1,
            ipea_shift=0.25,
            imaginary_shift=0.1,
            multistate=True,
            n_states=3
        )
        print("   ✓ Parameters validated successfully")
        print(f"   IPEA shift: {params.ipea_shift}")
        print(f"   Imaginary shift: {params.imaginary_shift}")
        print(f"   Multistate: {params.multistate} ({params.n_states} states)")
    except ValueError as e:
        print(f"   ✗ Parameter validation failed: {e}")
    
    # Demonstrate output parsing (with mock output)
    print("\n6. Output parsing demonstration...")
    mock_caspt2_output = """
    Total SCF energy:       -76.026765
    RASSCF root number    1 Total energy:      -76.237891
    CASPT2 Total Energy:    -76.285432
    Reference energy:       -76.237891
    CASPT2 converged in 15 iterations
    Total wall time:        123.45
    Total CPU time:         98.76
    Maximum memory used:    2048.0 MB
    """
    
    parser = OpenMolcasOutputParser()
    results = parser.parse_output(mock_caspt2_output, "caspt2")
    
    print("   Parsed results:")
    print(f"   SCF energy: {results.scf_energy:.6f} Hartree")
    print(f"   CASSCF energy: {results.casscf_energy:.6f} Hartree")
    print(f"   CASPT2 energy: {results.caspt2_energy:.6f} Hartree")
    print(f"   Correlation energy: {results.correlation_energy:.6f} Hartree")
    print(f"   Converged: {results.caspt2_converged}")
    print(f"   Wall time: {results.wall_time:.1f} seconds")
    
    # Demonstrate method initialization and parameter optimization
    print("\n7. CASPT2 method with automatic parameter optimization...")
    
    # Mock the interface to avoid requiring actual OpenMolcas installation
    print("   Note: This would require OpenMolcas installation for actual calculations")
    
    try:
        # This will fail without OpenMolcas, but we can show the method setup
        method = CASPT2Method(
            ipea_shift=None,  # Auto-optimize
            multistate=False,
            auto_optimize_parameters=True
        )
        print("   ✓ CASPT2 method initialized")
    except Exception as e:
        print(f"   Expected error (no OpenMolcas): {type(e).__name__}")
        
        # Show parameter optimization without initialization
        from quantum.chemistry.multireference.external.openmolcas.caspt2_method import CASPT2Method
        
        # Create method instance without interface validation
        class MockCASPT2Method(CASPT2Method):
            def __init__(self, **kwargs):
                # Skip parent initialization to avoid software detection
                self.ipea_shift = kwargs.get('ipea_shift')
                self.multistate = kwargs.get('multistate', False)
                self.n_states = kwargs.get('n_states', 1)
                self.auto_optimize_parameters = kwargs.get('auto_optimize_parameters', True)
        
        method = MockCASPT2Method(auto_optimize_parameters=True)
        
        # Show parameter optimization for water (organic system)
        optimized_params = method._optimize_parameters_for_system(mol, active_space)
        print("   Optimized parameters for H2O (organic system):")
        print(f"   IPEA shift: {optimized_params['ipea_shift']}")
        print(f"   Imaginary shift: {optimized_params['imaginary_shift']}")
        print(f"   Memory: {optimized_params['memory_mb']} MB")
    
    # Demonstrate cost estimation
    print("\n8. Computational cost estimation...")
    
    class MockMethod:
        def __init__(self):
            self.multistate = False
            self.n_states = 1
        
        def estimate_cost(self, n_electrons, n_orbitals, basis_size, **kwargs):
            # Simple cost estimation
            active_memory = (n_orbitals ** 4) * 8e-6
            basis_memory = (basis_size ** 2) * 8e-6
            total_memory = active_memory + basis_memory
            time_estimate = (n_orbitals ** 5) * (basis_size ** 2) * 1e-9
            
            return {
                "memory_mb": total_memory,
                "time_seconds": time_estimate,
                "disk_mb": total_memory * 2.5,
                "scaling_info": {
                    "active_space_scaling": "O(N^5)",
                    "basis_scaling": "O(M^2)",
                    "combined_scaling": "O(N^5 * M^2)"
                }
            }
    
    mock_method = MockMethod()
    cost = mock_method.estimate_cost(
        active_space.n_active_electrons,
        active_space.n_active_orbitals,
        mol.nao_nr()
    )
    
    print(f"   Estimated memory: {cost['memory_mb']:.1f} MB")
    print(f"   Estimated time: {cost['time_seconds']:.2f} seconds")
    print(f"   Estimated disk: {cost['disk_mb']:.1f} MB")
    print(f"   Scaling: {cost['scaling_info']['combined_scaling']}")
    
    # Demonstrate validator capabilities
    print("\n9. Cross-method validation framework...")
    validator = OpenMolcasValidator()
    
    print("   Available benchmarks:")
    for benchmark_name in validator.benchmarks.keys():
        benchmark = validator.benchmarks[benchmark_name]
        print(f"   - {benchmark_name}: {benchmark['caspt2_energy']:.6f} Hartree")
    
    print(f"\n   Validation tolerances:")
    for key, value in validator.tolerances.items():
        print(f"   - {key}: {value}")
    
    # Show templates
    print("\n10. Available input templates...")
    print("    Templates loaded:")
    for template_name in generator.templates.keys():
        print(f"    - {template_name}")
    
    print("\nExample completed successfully!")
    print("\nTo use this implementation:")
    print("1. Install OpenMolcas or build the Docker container")
    print("2. Use CASPT2Method for calculations")
    print("3. Leverage automatic parameter optimization")
    print("4. Validate results with built-in cross-validation")


if __name__ == "__main__":
    main()