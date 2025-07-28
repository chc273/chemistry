"""
High-level workflow orchestration for multireference calculations.

This module provides the main interface for running comprehensive
multireference calculations with automated method selection and
parameter optimization.
"""

from typing import Any, Dict, List, Optional, Union

from pyscf import scf

from quantum.chemistry.active_space import (
    ActiveSpaceResult,
    UnifiedActiveSpaceFinder,
    auto_find_active_space,
)

from ..base import MethodSelector, MultireferenceResult
from ..methods import CASSCFMethod, NEVPT2Method


class MultireferenceWorkflow:
    """
    High-level workflow orchestrator for multireference calculations.

    This class provides a simplified interface for running complete
    multireference calculations, from active space selection to
    method execution and result analysis.
    """

    def __init__(
        self,
        method_selector: Optional[MethodSelector] = None,
        active_space_finder: Optional[UnifiedActiveSpaceFinder] = None,
    ):
        """
        Initialize workflow orchestrator.

        Args:
            method_selector: Method selection engine
            active_space_finder: Active space selection engine
        """
        self.method_selector = method_selector or MethodSelector()
        self.active_space_finder = active_space_finder or UnifiedActiveSpaceFinder()

        # Register available methods
        self._register_methods()

    def _register_methods(self):
        """Register available multireference methods."""
        from ..base import MultireferenceMethodType

        self.method_selector.register_method(
            MultireferenceMethodType.CASSCF,
            CASSCFMethod,
            accuracy="moderate",
            cost="low",
            suitable_for=["organic", "general"],
        )

        self.method_selector.register_method(
            MultireferenceMethodType.NEVPT2,
            NEVPT2Method,
            accuracy="high",
            cost="moderate",
            suitable_for=["organic", "transition_metal"],
        )

    def run_calculation(
        self,
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
        active_space_method: str = "auto",
        mr_method: str = "auto",
        target_accuracy: str = "standard",
        cost_constraint: str = "moderate",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run complete multireference calculation workflow.

        Args:
            scf_obj: Converged SCF object
            active_space_method: Active space selection method ("auto" for automatic)
            mr_method: Multireference method ("auto" for automatic selection)
            target_accuracy: Desired accuracy level
            cost_constraint: Computational cost constraint
            **kwargs: Additional parameters

        Returns:
            Dict containing all calculation results and metadata
        """
        results = {
            "scf_energy": scf_obj.e_tot,
            "molecule": scf_obj.mol,
            "workflow_parameters": {
                "active_space_method": active_space_method,
                "mr_method": mr_method,
                "target_accuracy": target_accuracy,
                "cost_constraint": cost_constraint,
            },
        }

        # Step 1: Active space selection
        if active_space_method == "auto":
            active_space = auto_find_active_space(
                scf_obj,
                target_size=kwargs.get("target_active_space_size"),
                priority_methods=kwargs.get("priority_methods", ["avas", "apc"]),
            )
        else:
            active_space = self.active_space_finder.find_active_space(
                active_space_method, scf_obj, **kwargs
            )

        results["active_space"] = active_space

        # Step 2: Method selection
        if mr_method == "auto":
            recommended_method = self.method_selector.recommend_method(
                scf_obj,
                active_space,
                accuracy_target=target_accuracy,
                cost_constraint=cost_constraint,
            )
        else:
            from ..base import MultireferenceMethodType

            recommended_method = MultireferenceMethodType(mr_method)

        results["selected_method"] = recommended_method.value

        # Step 3: Method execution
        method_class = self.method_selector._method_registry[recommended_method][
            "class"
        ]
        method_instance = method_class(**kwargs)

        mr_result = method_instance.calculate(scf_obj, active_space, **kwargs)
        results["multireference_result"] = mr_result

        # Step 4: Analysis and validation
        analysis = self._analyze_results(scf_obj, active_space, mr_result)
        results["analysis"] = analysis

        return results

    def compare_methods(
        self,
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
        active_space: ActiveSpaceResult,
        methods: List[str],
        **kwargs,
    ) -> Dict[str, MultireferenceResult]:
        """
        Compare multiple multireference methods on the same system.

        Args:
            scf_obj: Converged SCF object
            active_space: Active space selection result
            methods: List of method names to compare
            **kwargs: Additional parameters

        Returns:
            Dict mapping method names to results
        """
        results = {}

        for method_name in methods:
            try:
                from ..base import MultireferenceMethodType

                method_type = MultireferenceMethodType(method_name)

                if method_type in self.method_selector._method_registry:
                    method_class = self.method_selector._method_registry[method_type][
                        "class"
                    ]
                    method_instance = method_class(**kwargs)

                    result = method_instance.calculate(scf_obj, active_space, **kwargs)
                    results[method_name] = result
                else:
                    print(f"Method {method_name} not available")

            except Exception as e:
                print(f"Error calculating {method_name}: {e}")
                continue

        return results

    def _analyze_results(
        self,
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
        active_space: ActiveSpaceResult,
        mr_result: MultireferenceResult,
    ) -> Dict[str, Any]:
        """
        Analyze multireference calculation results.

        Args:
            scf_obj: SCF object
            active_space: Active space result
            mr_result: Multireference result

        Returns:
            Dict with analysis results
        """
        analysis = {}

        # Energy analysis
        correlation_recovery = abs(mr_result.correlation_energy or 0.0)
        analysis["correlation_energy_recovery"] = correlation_recovery

        # Active space analysis
        analysis["active_space_analysis"] = {
            "selection_method": active_space.method,
            "electrons_per_orbital": active_space.n_active_electrons
            / active_space.n_active_orbitals,
            "active_space_size": (
                active_space.n_active_electrons,
                active_space.n_active_orbitals,
            ),
        }

        # Method-specific analysis
        if mr_result.occupation_numbers is not None:
            # Analyze natural orbital occupations
            occ_nums = mr_result.occupation_numbers
            strongly_occupied = sum(occ_nums > 1.8)
            weakly_occupied = sum((occ_nums > 0.2) & (occ_nums < 1.8))

            analysis["natural_orbital_analysis"] = {
                "strongly_occupied": strongly_occupied,
                "weakly_occupied": weakly_occupied,
                "multireference_character": weakly_occupied / len(occ_nums),
            }

        # Convergence analysis
        analysis["convergence_quality"] = {
            "converged": mr_result.convergence_info.get("converged", False),
            "method": mr_result.method,
        }

        return analysis

    def estimate_calculation_cost(
        self,
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
        active_space_size: tuple,
        method: str,
    ) -> Dict[str, float]:
        """
        Estimate computational cost for given calculation parameters.

        Args:
            scf_obj: SCF object for system information
            active_space_size: (n_electrons, n_orbitals) tuple
            method: Method name

        Returns:
            Dict with cost estimates
        """
        try:
            from ..base import MultireferenceMethodType

            method_type = MultireferenceMethodType(method)

            if method_type not in self.method_selector._method_registry:
                return {"error": f"Method {method} not available"}

            method_class = self.method_selector._method_registry[method_type]["class"]
            method_instance = method_class()

            n_electrons, n_orbitals = active_space_size
            basis_size = scf_obj.mol.nao

            cost_estimate = method_instance.estimate_cost(
                n_electrons, n_orbitals, basis_size
            )

            return cost_estimate

        except Exception as e:
            return {"error": str(e)}
