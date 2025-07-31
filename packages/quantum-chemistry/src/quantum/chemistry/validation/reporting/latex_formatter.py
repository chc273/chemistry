"""
LaTeX Formatting Utilities for Quantum Chemistry Reports

This module provides specialized LaTeX formatting for quantum chemistry
data including mathematical expressions, chemical formulas, units, and
scientific notation suitable for publication.
"""

import re
from typing import Dict, List, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class LaTeXFormatter:
    """LaTeX formatting utilities for quantum chemistry reports."""
    
    def __init__(self):
        # Chemical element regex
        self.element_pattern = re.compile(r'\b([A-Z][a-z]?)\b')
        
        # Special LaTeX characters that need escaping
        self.latex_special_chars = {
            '&': r'\&',
            '%': r'\%',
            '$': r'\$',
            '#': r'\#',
            '^': r'\textasciicircum{}',
            '_': r'\_',
            '{': r'\{',
            '}': r'\}',
            '~': r'\textasciitilde{}',
            '\\': r'\textbackslash{}'
        }
        
        # Unit conversions and formatting
        self.unit_mappings = {
            'hartree': r'\si{\hartree}',
            'eV': r'\si{\eV}',
            'kcal/mol': r'\si{\kcal\per\mol}',
            'kJ/mol': r'\si{\kJ\per\mol}',
            'angstrom': r'\si{\angstrom}',
            'bohr': r'\si{\bohr}',
            'MHz': r'\si{\MHz}',
            'cm-1': r'\si{\per\cm}'
        }
    
    def escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters in text."""
        
        if not isinstance(text, str):
            return str(text)
        
        escaped_text = text
        for char, escaped in self.latex_special_chars.items():
            escaped_text = escaped_text.replace(char, escaped)
        
        return escaped_text
    
    def format_chemical_formula(self, formula: str) -> str:
        """Format chemical formula using chemformula package."""
        
        # Simple formatting - in practice might want more sophisticated parsing
        formatted = formula
        
        # Replace numbers with subscripts
        formatted = re.sub(r'(\d+)', r'_{\1}', formatted)
        
        # Wrap in chemformula command
        return f"\\ce{{{formatted}}}"
    
    def format_number_with_uncertainty(
        self, 
        value: float, 
        uncertainty: Optional[float] = None,
        unit: Optional[str] = None,
        precision: int = 4
    ) -> str:
        """Format number with uncertainty using siunitx."""
        
        if uncertainty is not None:
            # Use siunitx for proper uncertainty formatting
            formatted = f"\\num{{{value:.{precision}f} +- {uncertainty:.{precision}f}}}"
        else:
            formatted = f"\\num{{{value:.{precision}f}}}"
        
        if unit and unit in self.unit_mappings:
            formatted += f" {self.unit_mappings[unit]}"
        elif unit:
            formatted += f" \\si{{{unit}}}"
        
        return formatted
    
    def format_scientific_notation(
        self, 
        value: float, 
        precision: int = 2,
        unit: Optional[str] = None
    ) -> str:
        """Format number in scientific notation using siunitx."""
        
        formatted = f"\\num{{{value:.{precision}e}}}"
        
        if unit and unit in self.unit_mappings:
            formatted += f" {self.unit_mappings[unit]}"
        elif unit:
            formatted += f" \\si{{{unit}}}"
        
        return formatted
    
    def format_table(
        self, 
        data: List[Dict[str, Any]], 
        title: str,
        caption: str,
        label: Optional[str] = None
    ) -> str:
        """Format data as LaTeX table using booktabs."""
        
        if not data:
            return ""
        
        # Get column headers
        headers = list(data[0].keys())
        n_cols = len(headers)
        
        # Create column specification
        col_spec = 'l' + 'c' * (n_cols - 1)  # Left align first column, center others
        
        # Escape headers
        escaped_headers = [self.escape_latex(str(header)) for header in headers]
        
        # Start table
        table_lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{caption}}}",
        ]
        
        if label:
            table_lines.append(f"\\label{{{label}}}")
        
        table_lines.extend([
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\toprule",
            " & ".join(escaped_headers) + " \\\\",
            "\\midrule"
        ])
        
        # Add data rows
        for row in data:
            escaped_values = []
            for header in headers:
                value = row.get(header, "")
                
                # Format numbers appropriately
                if isinstance(value, float):
                    if abs(value) < 1e-3 or abs(value) > 1e4:
                        formatted_value = self.format_scientific_notation(value, precision=3)
                    else:
                        formatted_value = f"{value:.4f}"
                else:
                    formatted_value = self.escape_latex(str(value))
                
                escaped_values.append(formatted_value)
            
            table_lines.append(" & ".join(escaped_values) + " \\\\")
        
        # End table
        table_lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        return "\n".join(table_lines)
    
    def format_method_name(self, method: str) -> str:
        """Format method name with proper typography."""
        
        method = method.upper()
        
        # Special formatting for common methods
        method_formats = {
            'CASSCF': r'\textsc{casscf}',
            'NEVPT2': r'\textsc{nevpt2}',
            'CASPT2': r'\textsc{caspt2}',
            'CCSD': r'\textsc{ccsd}',
            'CCSD(T)': r'\textsc{ccsd(t)}',
            'MP2': r'\textsc{mp2}',
            'DFT': r'\textsc{dft}',
            'B3LYP': r'\textsc{b3lyp}',
            'PBE': r'\textsc{pbe}',
            'AVAS': r'\textsc{avas}',
            'APC': r'\textsc{apc}',
            'DMET': r'\textsc{dmet}'
        }
        
        return method_formats.get(method, f"\\textsc{{{method.lower()}}}")
    
    def format_molecule_name(self, name: str) -> str:
        """Format molecule name with chemical formula."""
        
        # Try to detect if it's a chemical formula
        if re.match(r'^[A-Z][a-z]?(\d*[A-Z][a-z]?\d*)*$', name):
            return self.format_chemical_formula(name)
        else:
            return self.escape_latex(name)
    
    def format_error_with_ci(
        self, 
        error: float, 
        ci_lower: float, 
        ci_upper: float,
        unit: str = "hartree"
    ) -> str:
        """Format error with confidence interval."""
        
        ci_width = (ci_upper - ci_lower) / 2
        
        return self.format_number_with_uncertainty(
            error, 
            ci_width, 
            unit, 
            precision=6
        )
    
    def create_figure_environment(
        self, 
        figure_path: str, 
        caption: str,
        label: Optional[str] = None,
        width: str = "0.8\\textwidth",
        placement: str = "htbp"
    ) -> str:
        """Create LaTeX figure environment."""
        
        lines = [
            f"\\begin{{figure}}[{placement}]",
            "\\centering",
            f"\\includegraphics[width={width}]{{{figure_path}}}",
            f"\\caption{{{caption}}}"
        ]
        
        if label:
            lines.append(f"\\label{{{label}}}")
        
        lines.append("\\end{figure}")
        
        return "\n".join(lines)
    
    def format_statistical_summary(self, stats: Dict[str, Any]) -> str:
        """Format statistical summary as LaTeX."""
        
        lines = []
        
        if 'mae' in stats:
            mae_str = self.format_number_with_uncertainty(
                stats['mae'], 
                stats.get('mae_ci_width'), 
                'hartree'
            )
            lines.append(f"\\item Mean Absolute Error: {mae_str}")
        
        if 'rmse' in stats:
            rmse_str = self.format_number_with_uncertainty(
                stats['rmse'], 
                stats.get('rmse_ci_width'), 
                'hartree'
            )
            lines.append(f"\\item Root Mean Square Error: {rmse_str}")
        
        if 'r2' in stats:
            lines.append(f"\\item $R^2$ Score: {stats['r2']:.4f}")
        
        if 'n_samples' in stats:
            lines.append(f"\\item Number of Samples: {stats['n_samples']}")
        
        if lines:
            return "\\begin{itemize}\n" + "\n".join(lines) + "\n\\end{itemize}"
        else:
            return ""
    
    def format_equation(self, equation: str, label: Optional[str] = None) -> str:
        """Format equation with optional label."""
        
        if label:
            return f"\\begin{{equation}}\n{equation}\n\\label{{{label}}}\n\\end{{equation}}"
        else:
            return f"\\begin{{equation*}}\n{equation}\n\\end{{equation*}}"
    
    def format_basis_set(self, basis_set: str) -> str:
        """Format basis set name with proper typography."""
        
        # Common basis set formatting
        basis_formats = {
            'sto-3g': 'STO-3G',
            'cc-pvdz': 'cc-pVDZ',
            'cc-pvtz': 'cc-pVTZ',
            'cc-pvqz': 'cc-pVQZ',
            'aug-cc-pvdz': 'aug-cc-pVDZ',
            'aug-cc-pvtz': 'aug-cc-pVTZ',
            '6-31g': '6-31G',
            '6-31g*': '6-31G*',
            '6-31g**': '6-31G**',
            '6-311g**': '6-311G**'
        }
        
        formatted = basis_formats.get(basis_set.lower(), basis_set)
        return f"\\texttt{{{formatted}}}"
    
    def create_document_preamble(
        self, 
        document_class: str = "article",
        packages: Optional[List[str]] = None
    ) -> str:
        """Create LaTeX document preamble."""
        
        if packages is None:
            packages = [
                "amsmath", "amssymb", "graphicx", "booktabs", 
                "siunitx", "chemformula", "geometry"
            ]
        
        lines = [
            f"\\documentclass{{{document_class}}}",
            "\\usepackage[utf8]{inputenc}",
            "\\usepackage[T1]{fontenc}"
        ]
        
        for package in packages:
            if package == "geometry":
                lines.append("\\usepackage[margin=1in]{geometry}")
            elif package == "siunitx":
                lines.append("\\usepackage{siunitx}")
                lines.append("\\DeclareSIUnit\\hartree{E_h}")
                lines.append("\\DeclareSIUnit\\bohr{a_0}")
            else:
                lines.append(f"\\usepackage{{{package}}}")
        
        return "\n".join(lines)
    
    def format_author_affiliation(
        self, 
        authors: List[str], 
        affiliations: List[str]
    ) -> str:
        """Format author and affiliation information."""
        
        # Simple formatting - could be more sophisticated
        author_str = " \\and ".join(authors)
        
        if affiliations:
            affil_str = "\\\\ ".join(affiliations)
            return f"{author_str} \\\\ \\textit{{{affil_str}}}"
        else:
            return author_str
    
    def format_abstract(self, abstract: str) -> str:
        """Format abstract environment."""
        
        escaped_abstract = self.escape_latex(abstract)
        
        return f"""\\begin{{abstract}}
{escaped_abstract}
\\end{{abstract}}"""
    
    def format_keywords(self, keywords: List[str]) -> str:
        """Format keywords section."""
        
        if not keywords:
            return ""
        
        escaped_keywords = [self.escape_latex(kw) for kw in keywords]
        keywords_str = ", ".join(escaped_keywords)
        
        return f"\\textbf{{Keywords:}} {keywords_str}"