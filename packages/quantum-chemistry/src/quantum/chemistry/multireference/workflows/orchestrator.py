"""
High-level workflow orchestration for multireference calculations.

This module provides the main interface for running comprehensive
multireference calculations with automated method selection,
multireference diagnostics, and parameter optimization.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
from pyscf import scf

from quantum.chemistry.active_space import (
    ActiveSpaceResult,
    UnifiedActiveSpaceFinder,  
    auto_find_active_space,
)
from quantum.chemistry.diagnostics import (
    MultireferenceDiagnostics,
    IntelligentMethodSelector,
    DiagnosticConfig,
    ComprehensiveDiagnosticResult,
)
from quantum.chemistry.diagnostics.decision_tree import (
    ComputationalConstraint,
    AccuracyTarget,
)

from ..base import MethodSelector, MultireferenceResult
from ..methods import CASSCFMethod, NEVPT2Method


class MultireferenceWorkflow:
    """
    High-level workflow orchestrator for multireference calculations.

    This class provides a simplified interface for running complete
    multireference calculations with integrated diagnostics, from
    multireference character assessment to automated method selection
    and result analysis.
    """

    def __init__(
        self,
        method_selector: Optional[MethodSelector] = None,
        active_space_finder: Optional[UnifiedActiveSpaceFinder] = None,
        diagnostics: Optional[MultireferenceDiagnostics] = None,
        intelligent_selector: Optional[IntelligentMethodSelector] = None,
        diagnostic_config: Optional[DiagnosticConfig] = None,
    ):
        """
        Initialize workflow orchestrator.

        Args:
            method_selector: Method selection engine (legacy)
            active_space_finder: Active space selection engine
            diagnostics: Multireference diagnostics system
            intelligent_selector: AI-driven method selector
            diagnostic_config: Configuration for diagnostics
        """
        self.method_selector = method_selector or MethodSelector()
        self.active_space_finder = active_space_finder or UnifiedActiveSpaceFinder()
        
        # New diagnostic capabilities
        self.diagnostic_config = diagnostic_config or DiagnosticConfig()
        self.diagnostics = diagnostics or MultireferenceDiagnostics(self.diagnostic_config)
        self.intelligent_selector = intelligent_selector or IntelligentMethodSelector(self.diagnostic_config)

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
        run_diagnostics: bool = True,
        diagnostic_level: str = "hierarchical",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run complete multireference calculation workflow with integrated diagnostics.

        Args:
            scf_obj: Converged SCF object
            active_space_method: Active space selection method ("auto" for automatic)
            mr_method: Multireference method ("auto" for automatic selection)
            target_accuracy: Desired accuracy level
            cost_constraint: Computational cost constraint
            run_diagnostics: Whether to run multireference diagnostics
            diagnostic_level: Level of diagnostics ("fast", "reference", "hierarchical", "full")
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
                "run_diagnostics": run_diagnostics,
                "diagnostic_level": diagnostic_level,
            },
        }

        # Step 1: Multireference Diagnostics (if enabled)
        diagnostic_result = None
        if run_diagnostics:
            diagnostic_result = self.run_diagnostics(scf_obj, diagnostic_level)
            results["diagnostic_result"] = diagnostic_result
            
            # Print diagnostic summary
            if diagnostic_result:
                print(diagnostic_result.get_summary())

        # Step 2: Intelligent Active Space Selection
        if active_space_method == "auto":
            if diagnostic_result and self.diagnostic_config.auto_active_space_selection:
                # Use diagnostic-informed active space selection
                if diagnostic_result.recommended_active_space_size:
                    n_e, n_o = diagnostic_result.recommended_active_space_size
                    kwargs["target_active_space_size"] = (n_e, n_o)
                    
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

        # Step 3: Intelligent Method Selection
        if mr_method == "auto":
            if diagnostic_result and self.diagnostic_config.auto_method_selection:
                # Use intelligent method selector
                constraint_map = {
                    "minimal": ComputationalConstraint.MINIMAL,
                    "low": ComputationalConstraint.LOW,
                    "moderate": ComputationalConstraint.MODERATE,
                    "high": ComputationalConstraint.HIGH,
                    "unlimited": ComputationalConstraint.UNLIMITED,
                }
                
                accuracy_map = {
                    "qualitative": AccuracyTarget.QUALITATIVE,
                    "standard": AccuracyTarget.STANDARD,
                    "high": AccuracyTarget.HIGH,
                    "benchmark": AccuracyTarget.BENCHMARK,
                }
                
                constraint = constraint_map.get(cost_constraint, ComputationalConstraint.MODERATE)
                accuracy = accuracy_map.get(target_accuracy, AccuracyTarget.STANDARD)
                
                # Get intelligent recommendation
                recommendation = self.intelligent_selector.recommend_method(
                    diagnostic_result, scf_obj, constraint, accuracy,
                    prefer_gpu=kwargs.get("prefer_gpu", True)
                )
                
                results["intelligent_recommendation"] = recommendation
                
                # Use primary method recommendation
                recommended_method_str = recommendation["primary_method"]
                
                # Map to legacy method type for execution
                method_map = {
                    "CASSCF": "casscf",
                    "NEVPT2": "nevpt2",
                    "CASPT2": "caspt2",
                    "DMRG": "dmrg",
                    "Selected-CI": "shci",
                    "MP2": "mp2",
                    "CCSD": "ccsd",
                    "CCSD(T)": "ccsd_t",
                }
                
                method_str = method_map.get(recommended_method_str, "nevpt2")
                
                from ..base import MultireferenceMethodType
                try:
                    recommended_method = MultireferenceMethodType(method_str)
                except ValueError:
                    # Fallback to NEVPT2 if method not available
                    recommended_method = MultireferenceMethodType.NEVPT2
                    
            else:
                # Use legacy method selector
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

        # Step 4: Method execution
        method_class = self.method_selector._method_registry[recommended_method][
            "class"
        ]
        method_instance = method_class(**kwargs)

        mr_result = method_instance.calculate(scf_obj, active_space, **kwargs)
        results["multireference_result"] = mr_result

        # Step 5: Enhanced Analysis and validation
        analysis = self._analyze_results(scf_obj, active_space, mr_result, diagnostic_result)
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

    def run_diagnostics(
        self,
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
        level: str = "hierarchical"
    ) -> Optional[ComprehensiveDiagnosticResult]:
        """
        Run multireference diagnostics on the system.
        
        Args:
            scf_obj: Converged SCF calculation object
            level: Diagnostic level ("fast", "reference", "hierarchical", "full")
            
        Returns:
            Comprehensive diagnostic result or None if disabled
        """
        try:
            if level == "fast":
                # Run only fast screening methods
                fast_results = self.diagnostics.run_fast_screening(scf_obj)
                # Create minimal comprehensive result
                return self.diagnostics._generate_consensus(scf_obj, fast_results)
                
            elif level == "reference":
                # Run only expensive reference methods
                ref_results = self.diagnostics.run_reference_diagnostics(scf_obj)
                return self.diagnostics._generate_consensus(scf_obj, ref_results)
                
            elif level == "hierarchical":
                # Use intelligent hierarchical screening
                return self.diagnostics.run_hierarchical_screening(scf_obj)
                
            elif level == "full":
                # Run comprehensive analysis with all methods
                return self.diagnostics.run_full_analysis(scf_obj)
                
            else:
                # Default to hierarchical
                return self.diagnostics.run_hierarchical_screening(scf_obj)
                
        except Exception as e:
            print(f"Diagnostic analysis failed: {e}")
            return None

    def compare_diagnostic_methods(
        self,
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF]
    ) -> Dict[str, Any]:
        """
        Compare different diagnostic approaches for the system.
        
        Args:
            scf_obj: SCF calculation object
            
        Returns:
            Comparison of diagnostic methods and their recommendations
        """
        try:
            # Run fast screening
            fast_results = self.diagnostics.run_fast_screening(scf_obj)
            fast_consensus = self.diagnostics._generate_consensus(scf_obj, fast_results)
            
            # Run reference methods
            ref_results = self.diagnostics.run_reference_diagnostics(scf_obj)
            ref_consensus = self.diagnostics._generate_consensus(scf_obj, ref_results)
            
            # Full analysis
            full_result = self.diagnostics.run_full_analysis(scf_obj)
            
            # Compare method recommendations
            constraint = ComputationalConstraint.MODERATE
            accuracy = AccuracyTarget.STANDARD
            
            fast_recommendation = self.intelligent_selector.recommend_method(
                fast_consensus, scf_obj, constraint, accuracy
            ) if fast_consensus else None
            
            ref_recommendation = self.intelligent_selector.recommend_method(
                ref_consensus, scf_obj, constraint, accuracy
            ) if ref_consensus else None
            
            full_recommendation = self.intelligent_selector.recommend_method(
                full_result, scf_obj, constraint, accuracy
            ) if full_result else None
            
            return {
                "fast_screening": {
                    "result": fast_consensus,
                    "recommendation": fast_recommendation,
                },
                "reference_methods": {
                    "result": ref_consensus,
                    "recommendation": ref_recommendation,
                },
                "full_analysis": {
                    "result": full_result,
                    "recommendation": full_recommendation,
                },
                "agreement_analysis": self._analyze_method_agreement([
                    fast_recommendation, ref_recommendation, full_recommendation
                ]),
            }
            
        except Exception as e:
            return {"error": str(e)}

    def recommend_computational_strategy(
        self,
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
        available_resources: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Recommend comprehensive computational strategy based on diagnostics.
        
        Args:
            scf_obj: SCF calculation object
            available_resources: Information about computational resources
            
        Returns:
            Strategic recommendations for the calculation
        """
        if available_resources is None:
            available_resources = {"constraint": "moderate", "time_limit_hours": 24}
        
        # Run diagnostics
        diagnostic_result = self.run_diagnostics(scf_obj, "hierarchical")
        
        if not diagnostic_result:
            return {"error": "Diagnostic analysis failed"}
        
        # Determine computational constraint
        constraint_str = available_resources.get("constraint", "moderate")
        constraint_map = {
            "minimal": ComputationalConstraint.MINIMAL,
            "low": ComputationalConstraint.LOW,
            "moderate": ComputationalConstraint.MODERATE,
            "high": ComputationalConstraint.HIGH,
            "unlimited": ComputationalConstraint.UNLIMITED,
        }
        constraint = constraint_map.get(constraint_str, ComputationalConstraint.MODERATE)
        
        # Compare multiple method options
        candidate_methods = ["CASSCF", "NEVPT2", "CASPT2", "DMRG"]
        method_comparison = self.intelligent_selector.compare_method_options(
            diagnostic_result, scf_obj, candidate_methods, constraint
        )
        
        # Estimate feasibility for each method
        feasibility_analysis = {}
        time_limit = available_resources.get("time_limit_hours", 24)
        
        for method, analysis in method_comparison["comparisons"].items():
            estimated_time = analysis["cost_estimate"]["time_hours"]
            feasible = estimated_time <= time_limit
            
            feasibility_analysis[method] = {
                "feasible": feasible,
                "estimated_time_hours": estimated_time,
                "time_limit_hours": time_limit,
                "recommendation": analysis,
            }
        
        # Generate strategy
        strategy = {
            "diagnostic_summary": diagnostic_result.get_summary(),
            "recommended_approach": method_comparison["recommendation"],
            "method_comparison": method_comparison,
            "feasibility_analysis": feasibility_analysis,
            "computational_strategy": self._generate_computational_strategy(
                diagnostic_result, method_comparison, feasibility_analysis
            ),
        }
        
        return strategy

    def _analyze_results(
        self,
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
        active_space: ActiveSpaceResult,
        mr_result: MultireferenceResult,
        diagnostic_result: Optional[ComprehensiveDiagnosticResult] = None,
    ) -> Dict[str, Any]:
        """
        Enhanced analysis of multireference calculation results with diagnostics.

        Args:
            scf_obj: SCF object
            active_space: Active space result
            mr_result: Multireference result
            diagnostic_result: Comprehensive diagnostic result

        Returns:
            Dict with enhanced analysis results
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

        # Enhanced analysis with diagnostics
        if diagnostic_result:
            analysis["diagnostic_validation"] = {
                "consensus_character": diagnostic_result.consensus_character.value,
                "diagnostic_confidence": diagnostic_result.consensus_confidence,
                "method_agreement": diagnostic_result.method_agreement,
                "predicted_vs_observed": self._compare_predicted_vs_observed(
                    diagnostic_result, mr_result
                ),
            }
            
            # Validate method choice
            analysis["method_validation"] = self._validate_method_choice(
                diagnostic_result, mr_result
            )

        return analysis

    def _compare_predicted_vs_observed(
        self,
        diagnostic_result: ComprehensiveDiagnosticResult,
        mr_result: MultireferenceResult
    ) -> Dict[str, Any]:
        """
        Compare diagnostic predictions with actual calculation results.
        
        Args:
            diagnostic_result: Diagnostic predictions
            mr_result: Actual calculation results
            
        Returns:
            Comparison analysis
        """
        comparison = {}
        
        # Compare multireference character assessment
        predicted_char = diagnostic_result.consensus_character
        
        # Infer observed character from calculation results
        if mr_result.occupation_numbers is not None:
            occ_nums = mr_result.occupation_numbers
            fractional_count = int(np.sum((occ_nums > 0.1) & (occ_nums < 1.9)))
            
            if fractional_count == 0:
                observed_char = "none"
            elif fractional_count <= 2:
                observed_char = "weak"
            elif fractional_count <= 4:
                observed_char = "moderate"
            elif fractional_count <= 6:
                observed_char = "strong"
            else:
                observed_char = "very_strong"
        else:
            observed_char = "unknown"
        
        comparison["multireference_character"] = {
            "predicted": predicted_char.value,
            "observed": observed_char,
            "agreement": predicted_char.value == observed_char,
        }
        
        # Compare active space recommendations
        if diagnostic_result.recommended_active_space_size:
            pred_ne, pred_no = diagnostic_result.recommended_active_space_size
            actual_ne = mr_result.n_active_electrons
            actual_no = mr_result.n_active_orbitals
            
            comparison["active_space_size"] = {
                "predicted": (pred_ne, pred_no),
                "used": (actual_ne, actual_no),
                "size_difference": abs(pred_no - actual_no),
            }
        
        return comparison

    def _validate_method_choice(
        self,
        diagnostic_result: ComprehensiveDiagnosticResult,
        mr_result: MultireferenceResult
    ) -> Dict[str, Any]:
        """
        Validate whether the chosen method was appropriate.
        
        Args:
            diagnostic_result: Diagnostic assessment
            mr_result: Calculation results
            
        Returns:
            Method validation analysis
        """
        validation = {}
        
        method_used = mr_result.method
        mr_character = diagnostic_result.consensus_character
        
        # Method appropriateness based on MR character
        method_suitability = {
            "none": ["MP2", "CCSD", "CCSD(T)"],
            "weak": ["CCSD(T)", "CASSCF", "NEVPT2"],
            "moderate": ["CASSCF", "NEVPT2", "CASPT2"],
            "strong": ["NEVPT2", "CASPT2", "DMRG", "Selected-CI"],
            "very_strong": ["DMRG", "Selected-CI", "DMRG-NEVPT2"],
        }
        
        suitable_methods = method_suitability.get(mr_character.value, [])
        is_suitable = any(suitable in method_used.upper() for suitable in 
                         [m.upper() for m in suitable_methods])
        
        validation["method_appropriateness"] = {
            "method_used": method_used,
            "multireference_character": mr_character.value,
            "suitable_methods": suitable_methods,
            "is_appropriate": is_suitable,
        }
        
        # Convergence assessment
        converged = mr_result.convergence_info.get("converged", False)
        validation["convergence_assessment"] = {
            "converged": converged,
            "likely_convergence_issues": not converged and mr_character.value in ["strong", "very_strong"],
        }
        
        return validation

    def _analyze_method_agreement(
        self, 
        recommendations: List[Optional[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Analyze agreement between different diagnostic approaches.
        
        Args:
            recommendations: List of method recommendations
            
        Returns:
            Agreement analysis
        """
        valid_recommendations = [r for r in recommendations if r is not None]
        
        if len(valid_recommendations) < 2:
            return {"insufficient_data": True}
        
        # Extract primary method recommendations
        primary_methods = [r["primary_method"] for r in valid_recommendations]
        
        # Calculate agreement
        unique_methods = set(primary_methods)
        most_common = max(unique_methods, key=primary_methods.count)
        agreement_count = primary_methods.count(most_common)
        agreement_fraction = agreement_count / len(primary_methods)
        
        return {
            "total_recommendations": len(valid_recommendations),
            "unique_methods": list(unique_methods),
            "most_common_method": most_common,
            "agreement_fraction": agreement_fraction,
            "high_agreement": agreement_fraction > 0.7,
        }

    def _generate_computational_strategy(
        self,
        diagnostic_result: ComprehensiveDiagnosticResult,
        method_comparison: Dict[str, Any],
        feasibility_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate strategic computational recommendations.
        
        Args:
            diagnostic_result: Diagnostic assessment
            method_comparison: Method comparison results
            feasibility_analysis: Feasibility analysis
            
        Returns:
            Strategic recommendations
        """
        strategy = {}
        
        # Identify feasible methods
        feasible_methods = [
            method for method, analysis in feasibility_analysis.items()
            if analysis["feasible"]
        ]
        
        # Risk assessment
        mr_character = diagnostic_result.consensus_character
        confidence = diagnostic_result.consensus_confidence
        
        risk_level = "low"
        if mr_character.value in ["strong", "very_strong"]:
            risk_level = "high"
        elif mr_character.value == "moderate" or confidence < 0.7:
            risk_level = "medium"
        
        strategy["risk_assessment"] = {
            "level": risk_level,
            "multireference_character": mr_character.value,
            "diagnostic_confidence": confidence,
        }
        
        # Recommended approach
        if not feasible_methods:
            strategy["approach"] = "infeasible"
            strategy["recommendation"] = "Consider simpler methods or more resources"
        elif len(feasible_methods) == 1:
            strategy["approach"] = "single_method"
            strategy["recommendation"] = f"Proceed with {feasible_methods[0]}"
        else:
            strategy["approach"] = "multi_method_validation"
            strategy["recommendation"] = (
                f"Run {feasible_methods[0]} first, validate with {feasible_methods[1]}"
            )
        
        # Backup strategy
        if risk_level == "high":
            strategy["backup_strategy"] = "Consider DMRG or Selected-CI for validation"
        elif risk_level == "medium":
            strategy["backup_strategy"] = "Run CASSCF for qualitative validation"
        
        return strategy

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
