#!/usr/bin/env python3
"""
Example: Unified Active Space Selection

This example demonstrates how to use the unified active space finding module
to automatically select active spaces for CASSCF/CASCI calculations using
various methods including AVAS, APC, DMET-CAS, natural orbitals, localization,
and energy windows.
"""

import numpy as np

from quantum.chemistry import (
    ActiveSpaceMethod,
    B3LYPCalculator,
    HartreeFockCalculator,
    UnifiedActiveSpaceFinder,
    auto_find_active_space,
    find_active_space_avas,
    find_active_space_dmet_cas,
    find_active_space_natural_orbitals,
)
from quantum.core import ComputationEngine, Molecule


def main():
    """Demonstrate active space selection methods."""

    print("=== Unified Active Space Selection Example ===\n")

    # Create a water molecule
    water = Molecule(
        name="water",
        atoms=["O", "H", "H"],
        coordinates=np.array(
            [
                [0.0000, 0.0000, 0.1173],
                [0.0000, 0.7572, -0.4692],
                [0.0000, -0.7572, -0.4692],
            ]
        ),
    )

    # Create a more complex molecule - formaldehyde
    formaldehyde = Molecule(
        name="formaldehyde",
        atoms=["C", "O", "H", "H"],
        coordinates=np.array(
            [
                [0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 1.2091],
                [0.0000, 0.9429, -0.5897],
                [0.0000, -0.9429, -0.5897],
            ]
        ),
    )

    molecules = [water, formaldehyde]

    for molecule in molecules:
        print(f"\n{'='*50}")
        print(f"Active Space Selection for {molecule.name}")
        print(f"{'='*50}")

        # Run SCF calculation first
        print(f"\n1. Running SCF calculation for {molecule.name}...")
        engine = ComputationEngine()
        engine.register_calculator("hf", HartreeFockCalculator)

        hf_results = engine.run_single_point(molecule, method="hf", basis_set="6-31g")

        # Get the PySCF mean field object for active space selection
        hf_calc = engine._calculators["hf"]
        mf = hf_calc._mf

        print(f"   SCF Energy: {hf_results['energy']:.6f} Hartree")
        print(f"   HOMO-LUMO Gap: {hf_results.get('homo_lumo_gap', 'N/A')} Hartree")

        # Initialize the unified active space finder
        finder = UnifiedActiveSpaceFinder()
        print(f"\nAvailable methods: {finder.get_available_methods()}")

        # 2. AVAS Active Space Selection
        print("\n2. AVAS Active Space Selection")
        print("-" * 30)

        try:
            avas_result = find_active_space_avas(
                mf, molecule, threshold=0.2, minao="minao"
            )
            print(f"   Method: {avas_result.method}")
            print(f"   Active electrons: {avas_result.n_active_electrons}")
            print(f"   Active orbitals: {avas_result.n_active_orbitals}")
            print(f"   Active orbital indices: {avas_result.active_orbital_indices}")

            if avas_result.avas_data:
                print(f"   AVAS atoms: {avas_result.avas_data['avas_atoms']}")
                print(f"   AVAS threshold: {avas_result.avas_data['threshold']}")

        except Exception as e:
            print(f"   AVAS failed: {e}")

        # 3. DMET-CAS Active Space Selection
        print("\n3. DMET-CAS Active Space Selection")
        print("-" * 35)

        try:
            dmet_result = find_active_space_dmet_cas(mf, molecule)
            print(f"   Method: {dmet_result.method}")
            print(f"   Active electrons: {dmet_result.n_active_electrons}")
            print(f"   Active orbitals: {dmet_result.n_active_orbitals}")
            print(f"   Active orbital indices: {dmet_result.active_orbital_indices}")

            if dmet_result.metadata:
                print(f"   AO labels: {dmet_result.metadata.get('ao_labels', 'N/A')}")

        except Exception as e:
            print(f"   DMET-CAS failed: {e}")

        # 4. Natural Orbitals from MP2
        print("\n4. Natural Orbitals (MP2) Selection")
        print("-" * 37)

        try:
            natural_result = find_active_space_natural_orbitals(
                mf, molecule, occupation_threshold=0.02, max_orbitals=10
            )
            print(f"   Method: {natural_result.method}")
            print(f"   Active electrons: {natural_result.n_active_electrons}")
            print(f"   Active orbitals: {natural_result.n_active_orbitals}")

            if natural_result.selection_scores is not None:
                print(
                    f"   Occupation range: {np.min(natural_result.selection_scores):.4f} to {np.max(natural_result.selection_scores):.4f}"
                )

        except Exception as e:
            print(f"   Natural orbitals failed: {e}")

        # 5. APC Active Space Selection
        print("\n5. APC (Atomic Population Coefficient) Selection")
        print("-" * 50)

        try:
            apc_result = finder.find_active_space(
                ActiveSpaceMethod.APC,
                mf,
                molecule,
                population_threshold=0.1,
                energy_window=(2, 2),  # HOMO-2 to LUMO+2
            )
            print(f"   Method: {apc_result.method}")
            print(f"   Active electrons: {apc_result.n_active_electrons}")
            print(f"   Active orbitals: {apc_result.n_active_orbitals}")
            print(f"   Active orbital indices: {apc_result.active_orbital_indices}")

            if apc_result.selection_scores is not None:
                avg_score = np.mean(apc_result.selection_scores)
                print(f"   Average population score: {avg_score:.4f}")

        except Exception as e:
            print(f"   APC failed: {e}")

        # 6. Energy Window Selection
        print("\n6. Energy Window Selection")
        print("-" * 30)

        try:
            energy_result = finder.find_active_space(
                ActiveSpaceMethod.ENERGY_WINDOW,
                mf,
                molecule,
                energy_window=(1.5, 1.5),  # ±1.5 eV around HOMO-LUMO
                max_orbitals=10,
            )
            print(f"   Method: {energy_result.method}")
            print(f"   Active electrons: {energy_result.n_active_electrons}")
            print(f"   Active orbitals: {energy_result.n_active_orbitals}")
            print(f"   Energy window: {energy_result.metadata['energy_window']} eV")

            if energy_result.orbital_energies is not None:
                print(
                    f"   Orbital energy range: {np.min(energy_result.orbital_energies):.4f} to {np.max(energy_result.orbital_energies):.4f} Hartree"
                )

        except Exception as e:
            print(f"   Energy window failed: {e}")

        # 7. Boys Localization Selection
        print("\n7. Boys Localization Selection")
        print("-" * 35)

        try:
            boys_result = finder.find_active_space(
                ActiveSpaceMethod.BOYS_LOCALIZATION,
                mf,
                molecule,
                energy_window=(2, 2),
                localization_threshold=0.3,
            )
            print(f"   Method: {boys_result.method}")
            print(f"   Active electrons: {boys_result.n_active_electrons}")
            print(f"   Active orbitals: {boys_result.n_active_orbitals}")

            if boys_result.localization_data:
                print(
                    f"   Localization threshold: {boys_result.localization_data['localization_threshold']}"
                )

        except Exception as e:
            print(f"   Boys localization failed: {e}")

        # 8. IAO and IBO Selection
        print("\n8. IAO/IBO Localization Selection")
        print("-" * 37)

        try:
            iao_result = finder.find_active_space(
                ActiveSpaceMethod.IAO_LOCALIZATION,
                mf,
                molecule,
                energy_window=(2, 2),
                minao_basis="minao",
            )
            print(f"   IAO Method: {iao_result.method}")
            print(f"   Active electrons: {iao_result.n_active_electrons}")
            print(f"   Active orbitals: {iao_result.n_active_orbitals}")

        except Exception as e:
            print(f"   IAO localization failed: {e}")

        try:
            ibo_result = finder.find_active_space(
                ActiveSpaceMethod.IBO_LOCALIZATION,
                mf,
                molecule,
                energy_window=(2, 2),
                minao_basis="minao",
            )
            print(f"   IBO Method: {ibo_result.method}")
            print(f"   Active electrons: {ibo_result.n_active_electrons}")
            print(f"   Active orbitals: {ibo_result.n_active_orbitals}")

        except Exception as e:
            print(f"   IBO localization failed: {e}")

        # 9. Compare Multiple Methods
        print("\n9. Method Comparison")
        print("-" * 20)

        methods_to_compare = [
            ActiveSpaceMethod.AVAS,
            ActiveSpaceMethod.DMET_CAS,
            ActiveSpaceMethod.NATURAL_ORBITALS,
            ActiveSpaceMethod.APC,
            ActiveSpaceMethod.ENERGY_WINDOW,
        ]

        method_kwargs = {
            "avas": {"threshold": 0.2},
            "dmet_cas": {},
            "natural_orbitals": {"occupation_threshold": 0.02, "max_orbitals": 8},
            "apc": {"population_threshold": 0.1, "energy_window": (2, 2)},
            "energy_window": {"energy_window": (1.5, 1.5), "max_orbitals": 8},
        }

        try:
            comparison_results = finder.compare_methods(
                methods_to_compare, mf, molecule, method_kwargs=method_kwargs
            )

            print("   Method comparison summary:")
            for method_name, result in comparison_results.items():
                print(
                    f"     {method_name:18s}: ({result.n_active_electrons:2d} electrons, {result.n_active_orbitals:2d} orbitals)"
                )

        except Exception as e:
            print(f"   Method comparison failed: {e}")

        # 10. Automatic Active Space Selection
        print("\n10. Automatic Selection")
        print("-" * 26)

        try:
            auto_result = auto_find_active_space(
                mf,
                molecule,
                target_size=(6, 6),  # Target 6 electrons in 6 orbitals
            )
            print(f"   Auto-selected method: {auto_result.method}")
            print(f"   Active electrons: {auto_result.n_active_electrons}")
            print(f"   Active orbitals: {auto_result.n_active_orbitals}")
            print(
                f"   Recommended for CASSCF({auto_result.n_active_electrons},{auto_result.n_active_orbitals})"
            )

        except Exception as e:
            print(f"   Automatic selection failed: {e}")

        # 11. Export to Molden (if available)
        print("\n11. Export Options")
        print("-" * 18)

        try:
            # Use the auto-selected result for export
            if "auto_result" in locals():
                finder.export_molden(
                    auto_result, mf, f"{molecule.name}_active_orbitals.molden"
                )
                print(
                    f"   Active orbitals exported to {molecule.name}_active_orbitals.molden"
                )
        except Exception as e:
            print(f"   Molden export failed: {e}")

        # 12. FCIDUMP Export for external quantum chemistry codes
        print("\n12. FCIDUMP Export")
        print("-" * 18)

        try:
            from quantum.chemistry.fcidump import (
                active_space_to_fcidump,
                avas_to_fcidump,
            )

            # Create FCIDUMP with AVAS active space
            avas_file = avas_to_fcidump(
                mf,
                molecule,
                f"{molecule.name}_avas.fcidump",
                approach="effective",
                threshold=0.2,
            )
            print(f"   AVAS FCIDUMP: {avas_file}")

            # Create FCIDUMP with energy window method
            ew_file = active_space_to_fcidump(
                mf,
                molecule,
                f"{molecule.name}_energy_window.fcidump",
                method="energy_window",
                approach="effective",
                energy_window=(2.0, 2.0),
                max_orbitals=6,
            )
            print(f"   Energy Window FCIDUMP: {ew_file}")
            print(
                "   These files can be used with DMRG codes like Block2, CheMPS2, etc."
            )

        except Exception as e:
            print(f"   FCIDUMP export failed: {e}")


def demonstrate_advanced_usage():
    """Demonstrate advanced active space selection features."""

    print(f"\n{'='*60}")
    print("Advanced Active Space Selection Features")
    print(f"{'='*60}")

    # Create benzene molecule for more complex active space
    benzene = Molecule(
        name="benzene",
        atoms=["C"] * 6 + ["H"] * 6,
        coordinates=np.array(
            [
                # Carbon ring
                [1.396, 0.000, 0.000],
                [0.698, 1.209, 0.000],
                [-0.698, 1.209, 0.000],
                [-1.396, 0.000, 0.000],
                [-0.698, -1.209, 0.000],
                [0.698, -1.209, 0.000],
                # Hydrogens
                [2.485, 0.000, 0.000],
                [1.242, 2.152, 0.000],
                [-1.242, 2.152, 0.000],
                [-2.485, 0.000, 0.000],
                [-1.242, -2.152, 0.000],
                [1.242, -2.152, 0.000],
            ]
        ),
    )

    print(f"\nDemonstrating with {benzene.name} (π-system example)")

    # Run DFT calculation for better starting point
    engine = ComputationEngine()
    engine.register_calculator("b3lyp", B3LYPCalculator)

    try:
        dft_results = engine.run_single_point(
            benzene, method="b3lyp", basis_set="6-31g"
        )

        dft_calc = engine._calculators["b3lyp"]
        mf = dft_calc._mf

        print(f"DFT Energy: {dft_results['energy']:.6f} Hartree")

        finder = UnifiedActiveSpaceFinder()

        # Select π-system active space using AVAS
        print("\nπ-system AVAS selection:")
        pi_result = finder.find_active_space(
            ActiveSpaceMethod.AVAS,
            mf,
            benzene,
            avas_atoms=[0, 1, 2, 3, 4, 5],  # Carbon atoms only
            threshold=0.1,
            minao="minao",
        )

        print(
            f"   π-system active space: ({pi_result.n_active_electrons},{pi_result.n_active_orbitals})"
        )
        print(
            f"   Suitable for CASSCF({pi_result.n_active_electrons},{pi_result.n_active_orbitals}) calculation"
        )

        # Compare with energy window approach
        energy_pi_result = finder.find_active_space(
            ActiveSpaceMethod.ENERGY_WINDOW,
            mf,
            benzene,
            energy_window=(2.0, 2.0),
            max_orbitals=12,
        )

        print("\nEnergy window approach:")
        print(
            f"   Active space: ({energy_pi_result.n_active_electrons},{energy_pi_result.n_active_orbitals})"
        )

        # Automatic selection with target
        auto_pi_result = finder.auto_select_active_space(
            mf,
            benzene,
            target_size=(6, 6),  # Target π-system (6π electrons, 6π orbitals)
            priority_methods=[
                ActiveSpaceMethod.AVAS,
                ActiveSpaceMethod.ENERGY_WINDOW,
                ActiveSpaceMethod.APC,
            ],
        )

        print("\nAutomatic π-system selection:")
        print(f"   Best method: {auto_pi_result.method}")
        print(
            f"   Active space: ({auto_pi_result.n_active_electrons},{auto_pi_result.n_active_orbitals})"
        )

    except Exception as e:
        print(f"Advanced demonstration failed: {e}")


if __name__ == "__main__":
    main()
    demonstrate_advanced_usage()

    print(f"\n{'='*60}")
    print("Example completed successfully!")
    print("Next steps:")
    print("- Use the selected active spaces in CASSCF/CASCI calculations")
    print("- Visualize active orbitals with molecular visualization tools")
    print("- Compare active space quality with different methods")
    print(f"{'='*60}")
