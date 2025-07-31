"""
Automated report generation system for quantum chemistry benchmarking.

This module provides comprehensive report generation capabilities including:
- LaTeX/PDF report generation
- HTML reports with interactive elements  
- Markdown reports for documentation
- Scientific manuscript preparation
- Supplementary information generation
"""

from .report_generator import ReportGenerator, ReportConfiguration
from .latex_formatter import LaTeXFormatter
from .html_formatter import HTMLFormatter

__all__ = [
    "ReportGenerator",
    "ReportConfiguration", 
    "LaTeXFormatter",
    "HTMLFormatter",
]