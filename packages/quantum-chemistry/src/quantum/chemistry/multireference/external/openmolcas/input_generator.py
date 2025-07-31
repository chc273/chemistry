"""
OpenMolcas input file generation utilities.

This module provides templated input file generation for OpenMolcas calculations,
supporting CASSCF, CASPT2, and MS-CASPT2 methods with proper parameter validation.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator
from pyscf import scf

from quantum.chemistry.active_space import ActiveSpaceResult
from quantum.chemistry.fcidump import detect_orbital_symmetries


class OpenMolcasParameters(BaseModel):
    """
    Validated parameters for OpenMolcas calculations.
    
    This class ensures all parameters are within valid ranges and
    provides scientific defaults for different system types.
    """
    
    # Basic calculation parameters
    charge: int = Field(0, description="Molecular charge")
    spin_multiplicity: int = Field(1, description="Spin multiplicity (2S+1)")
    basis_set: str = Field("sto-3g", description="Basis set name")
    
    # Active space parameters
    n_active_electrons: int = Field(..., description="Number of active electrons")
    n_active_orbitals: int = Field(..., description="Number of active orbitals")
    
    # CASSCF parameters
    casscf_max_iter: int = Field(50, description="Maximum CASSCF iterations")
    casscf_convergence: float = Field(1e-8, description="CASSCF convergence threshold")
    casscf_levelshift: float = Field(0.0, description="CASSCF level shift")
    
    # CASPT2 parameters
    ipea_shift: float = Field(0.0, description="IPEA shift parameter", ge=0.0, le=1.0)
    imaginary_shift: float = Field(
        0.0, description="Imaginary shift for intruder states", ge=0.0, le=1.0
    )
    caspt2_max_iter: int = Field(50, description="Maximum CASPT2 iterations")
    caspt2_convergence: float = Field(1e-8, description="CASPT2 convergence threshold")
    
    # Multistate parameters
    multistate: bool = Field(False, description="Use MS-CASPT2")
    n_states: int = Field(1, description="Number of states for MS-CASPT2", ge=1, le=10)
    state_weights: Optional[List[float]] = Field(
        None, description="Weights for state-specific calculations"
    )
    
    # Symmetry and orbital ordering
    point_group: str = Field("C1", description="Point group symmetry")
    orbital_symmetries: Optional[List[str]] = Field(
        None, description="Symmetry labels for active orbitals"
    )
    
    # Computational parameters
    memory_mb: int = Field(2000, description="Memory allocation in MB", ge=100)
    disk_mb: Optional[int] = Field(None, description="Disk space allocation in MB")
    
    @field_validator("state_weights")
    @classmethod
    def validate_state_weights(cls, v, info):
        """Validate state weights match number of states."""
        if v is not None:
            values = info.data
            n_states = values.get("n_states", 1)
            if len(v) != n_states:
                raise ValueError(
                    f"Number of state weights ({len(v)}) must match n_states ({n_states})"
                )
            if abs(sum(v) - 1.0) > 1e-6:
                raise ValueError("State weights must sum to 1.0")
        return v
    
    @field_validator("spin_multiplicity")
    @classmethod
    def validate_spin_multiplicity(cls, v, info):
        """Validate spin multiplicity is consistent with electron count."""
        values = info.data
        if "n_active_electrons" in values:
            n_elec = values["n_active_electrons"]
            if (n_elec + v - 1) % 2 != 0:
                raise ValueError(
                    f"Spin multiplicity {v} inconsistent with {n_elec} electrons"
                )
        return v


class OpenMolcasInputGenerator:
    """
    Template-based OpenMolcas input file generator.
    
    This class provides robust input file generation for OpenMolcas calculations
    with proper error handling, parameter validation, and template management.
    """
    
    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize input generator with template directory.
        
        Args:
            template_dir: Directory containing OpenMolcas input templates
        """
        if template_dir is None:
            # Use default template directory relative to package
            package_dir = Path(__file__).parent.parent.parent.parent.parent.parent
            template_dir = package_dir / "templates"
        
        self.template_dir = Path(template_dir)
        self._load_templates()
    
    def _load_templates(self):
        """Load input file templates from template directory."""
        self.templates = {}
        
        # Try to load templates, create defaults if not found
        template_files = {
            "casscf": "casscf.template",
            "caspt2": "caspt2.template",
            "ms_caspt2": "ms_caspt2.template"
        }
        
        for template_name, filename in template_files.items():
            template_path = self.template_dir / filename
            if template_path.exists():
                with open(template_path, 'r') as f:
                    self.templates[template_name] = f.read()
            else:
                # Use built-in default templates
                self.templates[template_name] = self._get_default_template(template_name)
    
    def _get_default_template(self, template_name: str) -> str:
        """Get default template for given template name."""
        if template_name == "casscf":
            return self._default_casscf_template()
        elif template_name == "caspt2":
            return self._default_caspt2_template()
        elif template_name == "ms_caspt2":
            return self._default_ms_caspt2_template()
        else:
            raise ValueError(f"Unknown template name: {template_name}")
    
    def _default_casscf_template(self) -> str:
        """Default CASSCF input template."""
        return """! OpenMolcas CASSCF calculation
! Generated by quantum-chemistry package
! Active space: ({n_active_electrons},{n_active_orbitals})

&GATEWAY
  COORD = {xyz_filename}
  BASIS = {basis_set}
  GROUP = {point_group}
END OF INPUT

&SEWARD
END OF INPUT

&SCF
  CHARGE = {charge}
  SPIN = {unpaired_electrons}
END OF INPUT

&RASSCF
  NACTEL = {n_active_electrons} 0 0
  INACTIVE = {n_inactive}
  RAS2 = {n_active_orbitals}
  SPIN = {spin_multiplicity}
  SYMMETRY = 1
  CIROOT = {n_states} {n_states}
  MAXITER = {casscf_max_iter}
  THRS = {casscf_convergence}
{orbital_symmetry_section}
END OF INPUT"""
    
    def _default_caspt2_template(self) -> str:
        """Default CASPT2 input template."""
        return """! OpenMolcas CASPT2 calculation
! Generated by quantum-chemistry package
! Active space: ({n_active_electrons},{n_active_orbitals})

&GATEWAY
  COORD = {xyz_filename}
  BASIS = {basis_set}
  GROUP = {point_group}
END OF INPUT

&SEWARD
END OF INPUT

&SCF
  CHARGE = {charge}
  SPIN = {unpaired_electrons}
END OF INPUT

&RASSCF
  NACTEL = {n_active_electrons} 0 0
  INACTIVE = {n_inactive}
  RAS2 = {n_active_orbitals}
  SPIN = {spin_multiplicity}
  SYMMETRY = 1
  CIROOT = 1 1
  MAXITER = {casscf_max_iter}
  THRS = {casscf_convergence}
{orbital_symmetry_section}
END OF INPUT

&CASPT2
  IPEA = {ipea_shift}
  IMAGINARY = {imaginary_shift}
  MAXITER = {caspt2_max_iter}
  CONVERGENCE = {caspt2_convergence}
END OF INPUT"""
    
    def _default_ms_caspt2_template(self) -> str:  
        """Default MS-CASPT2 input template."""
        return """! OpenMolcas MS-CASPT2 calculation
! Generated by quantum-chemistry package
! Active space: ({n_active_electrons},{n_active_orbitals})

&GATEWAY
  COORD = {xyz_filename}
  BASIS = {basis_set}
  GROUP = {point_group}
END OF INPUT

&SEWARD
END OF INPUT

&SCF
  CHARGE = {charge}
  SPIN = {unpaired_electrons}
END OF INPUT

&RASSCF
  NACTEL = {n_active_electrons} 0 0
  INACTIVE = {n_inactive}
  RAS2 = {n_active_orbitals}
  SPIN = {spin_multiplicity}
  SYMMETRY = 1
  CIROOT = {n_states} {n_states}
  MAXITER = {casscf_max_iter}
  THRS = {casscf_convergence}
{orbital_symmetry_section}
END OF INPUT

&CASPT2
  IPEA = {ipea_shift}
  IMAGINARY = {imaginary_shift}
  MAXITER = {caspt2_max_iter}
  CONVERGENCE = {caspt2_convergence}
  MULTISTATE = {n_states}
  XMIXED
END OF INPUT"""
    
    def generate_input(
        self,
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
        active_space: ActiveSpaceResult, 
        calculation_type: str = "caspt2",
        xyz_filename: str = "molecule.xyz",
        **kwargs
    ) -> str:
        """
        Generate OpenMolcas input file content.
        
        Args:
            scf_obj: Converged SCF object
            active_space: Active space selection result
            calculation_type: Type of calculation ("casscf", "caspt2", "ms_caspt2")
            xyz_filename: Name of XYZ coordinate file
            **kwargs: Additional parameters to override defaults
            
        Returns:
            Generated input file content
        """
        # Build parameters with scientific defaults
        params = self._build_parameters(scf_obj, active_space, **kwargs)
        
        # Validate parameters
        params_obj = OpenMolcasParameters(**params)
        
        # Add derived parameters
        template_params = self._build_template_parameters(
            params_obj, scf_obj, active_space, xyz_filename
        )
        
        # Select appropriate template
        if calculation_type not in self.templates:
            raise ValueError(
                f"Unknown calculation type: {calculation_type}. "
                f"Available: {list(self.templates.keys())}"
            )
        
        template = self.templates[calculation_type]
        
        # Format template with parameters
        try:
            return template.format(**template_params)
        except KeyError as e:
            raise ValueError(
                f"Missing parameter for template formatting: {e}. "
                f"Available parameters: {list(template_params.keys())}"
            )
    
    def _build_parameters(
        self,
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
        active_space: ActiveSpaceResult,
        **kwargs
    ) -> Dict[str, Any]:
        """Build calculation parameters with defaults."""
        mol = scf_obj.mol
        
        # Basic molecular properties
        params = {
            "charge": mol.charge,
            "spin_multiplicity": mol.spin + 1,
            "basis_set": mol.basis if isinstance(mol.basis, str) else "sto-3g",
            "n_active_electrons": active_space.n_active_electrons,
            "n_active_orbitals": active_space.n_active_orbitals,
        }
        
        # Detect orbital symmetries if not provided
        if "orbital_symmetries" not in kwargs:
            params["orbital_symmetries"] = detect_orbital_symmetries(
                scf_obj, active_space.orbital_coefficients
            )
        
        # System-specific defaults
        system_type = self._classify_system(mol)
        params.update(self._get_system_defaults(system_type, active_space))
        
        # Override with user-provided parameters
        params.update(kwargs)
        
        return params
    
    def _build_template_parameters(
        self,
        params: OpenMolcasParameters,
        scf_obj: Union[scf.hf.SCF, scf.uhf.UHF],
        active_space: ActiveSpaceResult,
        xyz_filename: str
    ) -> Dict[str, Any]:
        """Build parameters for template formatting."""
        mol = scf_obj.mol
        
        # Calculate derived parameters
        n_inactive = (mol.nelectron - params.n_active_electrons) // 2
        unpaired_electrons = mol.spin
        
        # Build orbital symmetry section
        orbital_symmetry_section = ""
        if params.orbital_symmetries:
            orbsym_indices = [self._symmetry_to_index(sym) 
                            for sym in params.orbital_symmetries]
            orbital_symmetry_section = f"  ORBSYM = {','.join(map(str, orbsym_indices))}"
        
        template_params = {
            # File references
            "xyz_filename": xyz_filename,
            
            # Basic molecular parameters
            "charge": params.charge,
            "spin_multiplicity": params.spin_multiplicity,
            "unpaired_electrons": unpaired_electrons,
            "basis_set": params.basis_set,
            "point_group": params.point_group,
            
            # Active space parameters
            "n_active_electrons": params.n_active_electrons,
            "n_active_orbitals": params.n_active_orbitals,
            "n_inactive": n_inactive,
            "n_states": params.n_states,
            
            # CASSCF parameters
            "casscf_max_iter": params.casscf_max_iter,
            "casscf_convergence": params.casscf_convergence,
            
            # CASPT2 parameters
            "ipea_shift": params.ipea_shift,
            "imaginary_shift": params.imaginary_shift,
            "caspt2_max_iter": params.caspt2_max_iter,
            "caspt2_convergence": params.caspt2_convergence,
            
            # Orbital symmetry section
            "orbital_symmetry_section": orbital_symmetry_section,
        }
        
        return template_params
    
    def _classify_system(self, mol) -> str:
        """Classify molecular system for parameter defaults."""
        elements = {mol.atom_symbol(i) for i in range(mol.natm)}
        
        transition_metals = {'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                            'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd'}
        
        if elements & transition_metals:
            return "transition_metal"
        elif len(elements & {'C', 'H', 'N', 'O'}) >= 2:
            return "organic"
        else:
            return "general"
    
    def _get_system_defaults(self, system_type: str, active_space: ActiveSpaceResult) -> Dict[str, Any]:
        """Get default parameters based on system type."""
        defaults = {
            "memory_mb": 2000,
            "casscf_max_iter": 50,
            "caspt2_max_iter": 50,
        }
        
        if system_type == "transition_metal":
            # Transition metals often need more careful convergence
            defaults.update({
                "ipea_shift": 0.25,  # Standard IPEA shift for TM complexes
                "imaginary_shift": 0.1,  # Help with intruder states
                "casscf_convergence": 1e-7,
                "caspt2_convergence": 1e-7,
                "memory_mb": 4000,  # More memory for larger active spaces
            })
        elif system_type == "organic":
            # Organic molecules typically converge more easily
            defaults.update({
                "ipea_shift": 0.0,  # No IPEA shift needed
                "imaginary_shift": 0.0,
                "casscf_convergence": 1e-8,
                "caspt2_convergence": 1e-8,
            })
        
        # Adjust based on active space size
        if active_space.n_active_orbitals > 12:
            defaults["memory_mb"] *= 2
            defaults["casscf_max_iter"] = 100
            defaults["caspt2_max_iter"] = 100
        
        return defaults
    
    def _symmetry_to_index(self, symmetry_label: str) -> int:
        """Convert symmetry label to OpenMolcas index."""
        # Basic mapping - should be expanded for full point group support
        symmetry_map = {
            'A1': 1, 'A': 1,
            'A2': 2,
            'B1': 3, 'B': 2,
            'B2': 4,
            'E': 5,
            'T1': 6,
            'T2': 7,
        }
        
        return symmetry_map.get(symmetry_label, 1)  # Default to totally symmetric
    
    def save_templates(self, output_dir: Optional[str] = None):
        """
        Save current templates to disk for customization.
        
        Args:
            output_dir: Directory to save templates (default: self.template_dir)
        """
        if output_dir is None:
            output_dir = self.template_dir
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        template_files = {
            "casscf": "casscf.template",
            "caspt2": "caspt2.template", 
            "ms_caspt2": "ms_caspt2.template"
        }
        
        for template_name, filename in template_files.items():
            template_path = output_path / filename
            with open(template_path, 'w') as f:
                f.write(self.templates[template_name])
        
        print(f"Templates saved to {output_path}")