"""
Basic molecular calculations example.

This example demonstrates how to:
1. Create molecular systems
2. Run quantum chemistry calculations
3. Analyze results
"""

from ase import Atoms

from quantum.chemistry import HartreeFockCalculator
from quantum.core import ComputationEngine, Molecule


def main():
    """Run basic molecular calculations."""
    print("QuantChem Basic Molecular Calculations Example")
    print("=" * 50)

    # Create a water molecule using ASE Atoms
    water_atoms = Atoms(
        symbols=["O", "H", "H"],
        positions=[
            [0.0000, 0.0000, 0.1173],
            [0.0000, 0.7572, -0.4692],
            [0.0000, -0.7572, -0.4692],
        ],
    )
    water = Molecule(
        atoms=water_atoms,
        name="water",
        charge=0,
        multiplicity=1,
    )

    print(f"Created molecule: {water.name}")
    print(f"Number of atoms: {len(water.atoms)}")
    print(f"Number of electrons: {water.get_num_electrons()}")
    print(f"Nuclear repulsion energy: {water.compute_nuclear_repulsion():.6f} au")
    print()

    # Set up computation engine
    engine = ComputationEngine()

    # Register Hartree-Fock calculator
    engine.register_calculator("hf", HartreeFockCalculator)

    print("Available calculators:", engine.get_available_calculators())
    print()

    # Run Hartree-Fock calculation
    print("Running Hartree-Fock calculation...")
    try:
        results = engine.run_single_point(water, method="hf", basis_set="sto-3g")

        print(f"Total energy: {results['energy']:.6f} Hartree")
        print(f"Converged: {results['converged']}")
        print(f"Iterations: {results['iterations']}")

        if "homo_energy" in results:
            print(f"HOMO energy: {results['homo_energy']:.6f} Hartree")
        if "lumo_energy" in results and results["lumo_energy"] is not None:
            print(f"LUMO energy: {results['lumo_energy']:.6f} Hartree")
            homo_lumo_gap = results["lumo_energy"] - results["homo_energy"]
            print(
                f"HOMO-LUMO gap: {homo_lumo_gap:.6f} Hartree ({homo_lumo_gap * 27.211:.2f} eV)"
            )

    except Exception as e:
        print(f"Calculation failed: {e}")

    print()

    # Calculate molecular properties
    print("Calculating molecular properties...")
    try:
        properties = engine.run_property_calculation(
            water,
            properties=["dipole", "polarizability"],
            method="hf",
            basis_set="sto-3g",
        )

        print(f"Energy: {properties['energy']:.6f} Hartree")
        print(f"Dipole moment: {properties['dipole']} Debye")
        print("Polarizability tensor:")
        print(properties["polarizability"])

    except Exception as e:
        print(f"Property calculation failed: {e}")

    print()

    # Analyze molecular geometry
    print("Molecular geometry analysis:")
    bonds = water.get_bond_lengths()
    for i, j, distance in bonds:
        print(f"Bond {water.atoms[i]}-{water.atoms[j]}: {distance:.4f} Å")

    # Center of mass
    com = water.get_center_of_mass()
    print(f"Center of mass: [{com[0]:.4f}, {com[1]:.4f}, {com[2]:.4f}] Å")

    # Create centered molecule
    centered_water = water.center_at_origin()
    new_com = centered_water.get_center_of_mass()
    print(
        f"New center of mass: [{new_com[0]:.6f}, {new_com[1]:.6f}, {new_com[2]:.6f}] Å"
    )


if __name__ == "__main__":
    main()
