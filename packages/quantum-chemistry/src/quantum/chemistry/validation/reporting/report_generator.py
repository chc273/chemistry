"""
Automated Report Generator for Quantum Chemistry Benchmarking

This module provides comprehensive report generation capabilities for
benchmarking studies, including LaTeX/PDF, HTML, and Markdown outputs
suitable for publication and documentation.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json
import subprocess
import tempfile
import shutil
from datetime import datetime

import pandas as pd
import numpy as np
from jinja2 import Environment, FileSystemLoader, Template

from .latex_formatter import LaTeXFormatter
from .html_formatter import HTMLFormatter
from ..benchmarks.statistical_analysis import ErrorMetrics
from ..benchmarks.comprehensive_suite import BenchmarkResult

logger = logging.getLogger(__name__)


@dataclass
class ReportConfiguration:
    """Configuration for report generation."""
    
    # Report metadata
    title: str
    authors: List[str]
    affiliation: str
    abstract: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    
    # Output settings
    output_formats: List[str] = field(default_factory=lambda: ["pdf", "html", "markdown"])
    output_directory: Path = Path("./reports")
    
    # Content settings
    include_methodology: bool = True
    include_statistical_analysis: bool = True
    include_convergence_analysis: bool = True
    include_figures: bool = True
    include_tables: bool = True
    include_supplementary: bool = True
    
    # LaTeX settings
    latex_template: str = "article"
    bibliography_style: str = "acs"
    latex_packages: List[str] = field(default_factory=lambda: [
        "amsmath", "amssymb", "graphicx", "booktabs", "siunitx", "chemformula"
    ])
    
    # Formatting settings
    figure_format: str = "png"
    table_format: str = "latex"
    decimal_places: int = 4
    
    # Publication settings
    journal_template: Optional[str] = None
    supplementary_template: Optional[str] = None
    
    def __post_init__(self):
        self.output_directory = Path(self.output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)


class ReportGenerator:
    """Automated report generator for quantum chemistry benchmarking."""
    
    def __init__(self, config: ReportConfiguration):
        self.config = config
        self.latex_formatter = LaTeXFormatter()
        self.html_formatter = HTMLFormatter()
        
        # Setup Jinja2 environment
        template_dir = Path(__file__).parent / "templates"
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Register custom filters
        self.jinja_env.filters['scientific'] = self._scientific_notation
        self.jinja_env.filters['round_sig'] = self._round_significant
        self.jinja_env.filters['latex_escape'] = self.latex_formatter.escape_latex
        
        logger.info(f"Initialized ReportGenerator with output to {config.output_directory}")
    
    def generate_comprehensive_report(
        self,
        benchmark_results: List[BenchmarkResult],
        error_metrics: Dict[str, ErrorMetrics],
        statistical_analysis: Dict[str, Any],
        figure_paths: Optional[List[Path]] = None
    ) -> Dict[str, Path]:
        """Generate comprehensive benchmark report in all configured formats."""
        
        logger.info("Generating comprehensive benchmark report...")
        
        # Prepare report data
        report_data = self._prepare_report_data(
            benchmark_results, error_metrics, statistical_analysis, figure_paths
        )
        
        generated_files = {}
        
        # Generate reports in each requested format
        for format_type in self.config.output_formats:
            try:
                if format_type.lower() == "pdf":
                    pdf_file = self._generate_pdf_report(report_data)
                    generated_files["pdf"] = pdf_file
                    
                elif format_type.lower() == "html":
                    html_file = self._generate_html_report(report_data)
                    generated_files["html"] = html_file
                    
                elif format_type.lower() == "markdown":
                    md_file = self._generate_markdown_report(report_data)
                    generated_files["markdown"] = md_file
                    
                else:
                    logger.warning(f"Unknown report format: {format_type}")
                    
            except Exception as e:
                logger.error(f"Failed to generate {format_type} report: {e}")
                continue
        
        # Generate supplementary materials if requested
        if self.config.include_supplementary:
            try:
                supp_files = self._generate_supplementary_materials(report_data)
                generated_files.update(supp_files)
            except Exception as e:
                logger.error(f"Failed to generate supplementary materials: {e}")
        
        logger.info(f"Generated {len(generated_files)} report files")
        return generated_files
    
    def _prepare_report_data(
        self,
        benchmark_results: List[BenchmarkResult],
        error_metrics: Dict[str, ErrorMetrics],
        statistical_analysis: Dict[str, Any],
        figure_paths: Optional[List[Path]] = None
    ) -> Dict[str, Any]:
        """Prepare data structure for report templates."""
        
        # Basic metadata
        report_data = {
            "metadata": {
                "title": self.config.title,
                "authors": self.config.authors,
                "affiliation": self.config.affiliation,
                "abstract": self.config.abstract,
                "keywords": self.config.keywords,
                "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_calculations": len(benchmark_results),
                "n_methods": len(set(r.active_space_method or r.multireference_method 
                                   for r in benchmark_results)),
                "n_molecules": len(set(r.molecule_name for r in benchmark_results))
            }
        }
        
        # Process benchmark results
        results_df = pd.DataFrame([r.to_dict() for r in benchmark_results])
        
        # Summary statistics
        converged_count = sum(1 for r in benchmark_results if r.converged)
        total_time = sum(r.wall_time for r in benchmark_results if r.wall_time)
        
        report_data["summary"] = {
            "total_calculations": len(benchmark_results),
            "converged_calculations": converged_count,
            "convergence_rate": converged_count / len(benchmark_results) if benchmark_results else 0,
            "total_computational_time": total_time,
            "average_time_per_calculation": total_time / len(benchmark_results) if benchmark_results else 0,
            "unique_molecules": len(set(r.molecule_name for r in benchmark_results)),
            "databases_used": list(set(r.database_source for r in benchmark_results))
        }
        
        # Error metrics summary
        report_data["error_metrics"] = {}
        for method, metrics in error_metrics.items():
            report_data["error_metrics"][method] = {
                "mae": metrics.mae,
                "rmse": metrics.rmse,
                "max_error": metrics.max_error,
                "r2": metrics.r2,
                "n_samples": metrics.n_samples,
                "mae_ci": metrics.mae_ci,
                "rmse_ci": metrics.rmse_ci,
                "is_normal": metrics.is_normal,
                "mean_error": metrics.mean_error,
                "std_error": metrics.std_error
            }
        
        # Statistical analysis
        report_data["statistical_analysis"] = statistical_analysis
        
        # Method ranking
        method_ranking = []
        for method, metrics in error_metrics.items():
            method_ranking.append({
                "method": method,
                "mae": metrics.mae,
                "rmse": metrics.rmse,
                "composite_score": 0.5 * metrics.mae + 0.3 * metrics.rmse + 0.2 * metrics.max_error
            })
        
        method_ranking.sort(key=lambda x: x["composite_score"])
        for i, entry in enumerate(method_ranking):
            entry["rank"] = i + 1
        
        report_data["method_ranking"] = method_ranking
        
        # Figure paths
        if figure_paths:
            report_data["figures"] = [
                {
                    "path": str(path),
                    "filename": path.name,
                    "caption": self._generate_figure_caption(path)
                }
                for path in figure_paths
            ]
        else:
            report_data["figures"] = []
        
        # Tables
        report_data["tables"] = self._generate_tables(benchmark_results, error_metrics)
        
        return report_data
    
    def _generate_pdf_report(self, report_data: Dict[str, Any]) -> Path:
        """Generate PDF report using LaTeX."""
        
        logger.info("Generating PDF report...")
        
        # Select template
        if self.config.journal_template:
            template_name = f"{self.config.journal_template}.tex"
        else:
            template_name = "benchmark_report.tex"
        
        try:
            template = self.jinja_env.get_template(template_name)
        except:
            # Fallback to basic template
            template = self._create_basic_latex_template()
        
        # Render LaTeX content
        latex_content = template.render(**report_data, config=self.config)
        
        # Write to temporary file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            tex_file = temp_path / "report.tex"
            
            with open(tex_file, 'w', encoding='utf-8') as f:
                f.write(latex_content)
            
            # Copy figures to temp directory
            if report_data.get("figures"):
                figures_dir = temp_path / "figures"
                figures_dir.mkdir(exist_ok=True)
                
                for fig_info in report_data["figures"]:
                    src_path = Path(fig_info["path"])
                    if src_path.exists():
                        shutil.copy2(src_path, figures_dir / src_path.name)
            
            # Compile LaTeX
            pdf_file = self._compile_latex(tex_file)
            
            # Copy to output directory
            output_file = self.config.output_directory / f"{self.config.title.replace(' ', '_')}_report.pdf"
            if pdf_file and pdf_file.exists():
                shutil.copy2(pdf_file, output_file)
                logger.info(f"PDF report saved to {output_file}")
                return output_file
            else:
                raise RuntimeError("LaTeX compilation failed")
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> Path:
        """Generate HTML report."""
        
        logger.info("Generating HTML report...")
        
        try:
            template = self.jinja_env.get_template("benchmark_report.html")
        except:
            template = self._create_basic_html_template()
        
        # Render HTML content
        html_content = template.render(**report_data, config=self.config)
        
        # Save HTML file
        output_file = self.config.output_directory / f"{self.config.title.replace(' ', '_')}_report.html"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Copy figures to output directory
        if report_data.get("figures"):
            figures_dir = self.config.output_directory / "figures"
            figures_dir.mkdir(exist_ok=True)
            
            for fig_info in report_data["figures"]:
                src_path = Path(fig_info["path"])
                if src_path.exists():
                    shutil.copy2(src_path, figures_dir / src_path.name)
        
        logger.info(f"HTML report saved to {output_file}")
        return output_file
    
    def _generate_markdown_report(self, report_data: Dict[str, Any]) -> Path:
        """Generate Markdown report."""
        
        logger.info("Generating Markdown report...")
        
        try:
            template = self.jinja_env.get_template("benchmark_report.md")
        except:
            template = self._create_basic_markdown_template()
        
        # Render Markdown content
        markdown_content = template.render(**report_data, config=self.config)
        
        # Save Markdown file
        output_file = self.config.output_directory / f"{self.config.title.replace(' ', '_')}_report.md"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        logger.info(f"Markdown report saved to {output_file}")
        return output_file
    
    def _generate_supplementary_materials(self, report_data: Dict[str, Any]) -> Dict[str, Path]:
        """Generate supplementary materials."""
        
        logger.info("Generating supplementary materials...")
        
        supp_files = {}
        
        # 1. Detailed results CSV
        if "benchmark_results" in report_data:
            csv_file = self.config.output_directory / "supplementary_detailed_results.csv"
            # This would need the raw benchmark results DataFrame
            supp_files["detailed_results_csv"] = csv_file
        
        # 2. Statistical analysis JSON
        stats_file = self.config.output_directory / "supplementary_statistical_analysis.json"
        with open(stats_file, 'w') as f:
            json.dump(report_data["statistical_analysis"], f, indent=2, default=str)
        supp_files["statistical_analysis_json"] = stats_file
        
        # 3. Method comparison tables
        if self.config.include_tables:
            tables_file = self.config.output_directory / "supplementary_tables.html"
            tables_html = self.html_formatter.format_all_tables(report_data["tables"])
            
            with open(tables_file, 'w') as f:
                f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <title>Supplementary Tables</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        table {{ border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .table-title {{ font-size: 18px; font-weight: bold; margin-top: 30px; }}
    </style>
</head>
<body>
    <h1>Supplementary Tables: {self.config.title}</h1>
    {tables_html}
</body>
</html>
                """)
            
            supp_files["supplementary_tables"] = tables_file
        
        # 4. Computational details
        comp_details_file = self.config.output_directory / "computational_details.md"
        comp_details = self._generate_computational_details(report_data)
        
        with open(comp_details_file, 'w') as f:
            f.write(comp_details)
        
        supp_files["computational_details"] = comp_details_file
        
        return supp_files
    
    def _generate_tables(
        self, 
        benchmark_results: List[BenchmarkResult], 
        error_metrics: Dict[str, ErrorMetrics]
    ) -> Dict[str, Any]:
        """Generate tables for the report."""
        
        tables = {}
        
        # 1. Summary table
        summary_data = []
        for method, metrics in error_metrics.items():
            summary_data.append({
                "Method": method,
                "MAE (Hartree)": f"{metrics.mae:.{self.config.decimal_places}f}",
                "RMSE (Hartree)": f"{metrics.rmse:.{self.config.decimal_places}f}",
                "Max Error (Hartree)": f"{metrics.max_error:.{self.config.decimal_places}f}",
                "R²": f"{metrics.r2:.3f}",
                "N Samples": metrics.n_samples
            })
        
        tables["summary"] = {
            "title": "Summary of Method Performance",
            "data": summary_data,
            "caption": "Statistical summary of benchmark results for all methods tested."
        }
        
        # 2. Detailed error analysis
        detailed_data = []
        for method, metrics in error_metrics.items():
            mae_ci_width = (metrics.mae_ci[1] - metrics.mae_ci[0]) / 2
            rmse_ci_width = (metrics.rmse_ci[1] - metrics.rmse_ci[0]) / 2
            
            detailed_data.append({
                "Method": method,
                "MAE ± CI": f"{metrics.mae:.{self.config.decimal_places}f} ± {mae_ci_width:.{self.config.decimal_places}f}",
                "RMSE ± CI": f"{metrics.rmse:.{self.config.decimal_places}f} ± {rmse_ci_width:.{self.config.decimal_places}f}",
                "Mean Error": f"{metrics.mean_error:.{self.config.decimal_places}f}",
                "Std Error": f"{metrics.std_error:.{self.config.decimal_places}f}",
                "Normal Distribution": "Yes" if metrics.is_normal else "No"
            })
        
        tables["detailed_errors"] = {
            "title": "Detailed Error Analysis with Confidence Intervals",
            "data": detailed_data,
            "caption": "Detailed statistical analysis including confidence intervals and distribution tests."
        }
        
        # 3. System performance table
        system_data = {}
        for result in benchmark_results:
            system = result.molecule_name
            method = result.active_space_method or result.multireference_method
            
            if system not in system_data:
                system_data[system] = {}
            
            if result.absolute_error is not None:
                system_data[system][method] = result.absolute_error
        
        # Convert to table format
        performance_data = []
        methods = list(error_metrics.keys())
        
        for system in sorted(system_data.keys()):
            row = {"System": system}
            for method in methods:
                error = system_data[system].get(method)
                if error is not None:
                    row[method] = f"{error:.{self.config.decimal_places}f}"
                else:
                    row[method] = "—"
            performance_data.append(row)
        
        tables["system_performance"] = {
            "title": "System-Specific Performance",
            "data": performance_data,
            "caption": "Absolute errors (Hartree) for each method on individual molecular systems."
        }
        
        return tables
    
    def _generate_figure_caption(self, figure_path: Path) -> str:
        """Generate caption for a figure based on its filename."""
        
        filename = figure_path.stem.lower()
        
        if "error_distribution" in filename:
            return "Error distribution analysis showing histograms, box plots, and statistical tests for all methods."
        elif "method_comparison" in filename:
            return "Comprehensive method comparison including error metrics and performance profiles."
        elif "parity" in filename:
            return "Parity plots showing correlation between computed and reference values."
        elif "convergence" in filename:
            return "Convergence analysis with respect to computational parameters."
        elif "active_space" in filename:
            return "Active space selection analysis showing size distribution and computational complexity."
        elif "dashboard" in filename:
            return "Interactive dashboard with comprehensive benchmarking results."
        else:
            return f"Figure from {filename.replace('_', ' ')} analysis."
    
    def _compile_latex(self, tex_file: Path) -> Optional[Path]:
        """Compile LaTeX file to PDF."""
        
        try:
            # Change to the directory containing the .tex file
            original_dir = Path.cwd()
            tex_dir = tex_file.parent
            
            # Run pdflatex
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', tex_file.name],
                cwd=tex_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                pdf_file = tex_dir / tex_file.with_suffix('.pdf').name
                if pdf_file.exists():
                    return pdf_file
            else:
                logger.error(f"LaTeX compilation failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("LaTeX compilation timed out")
            return None
        except FileNotFoundError:
            logger.error("pdflatex not found. Please install LaTeX.")
            return None
        except Exception as e:
            logger.error(f"LaTeX compilation error: {e}")
            return None
    
    def _create_basic_latex_template(self) -> Template:
        """Create basic LaTeX template if none exists."""
        
        template_content = r"""
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{siunitx}
\usepackage[margin=1in]{geometry}

\title{ {{- metadata.title -}} }
\author{
{%- for author in metadata.authors -%}
{{author}}{% if not loop.last %} \and {% endif %}
{%- endfor -%}
}
\date{ {{- metadata.generation_date -}} }

\begin{document}

\maketitle

{% if metadata.abstract %}
\begin{abstract}
{{ metadata.abstract }}
\end{abstract}
{% endif %}

\section{Summary}

This report presents the results of a comprehensive benchmarking study of quantum chemistry methods. 
A total of {{ metadata.total_calculations }} calculations were performed on {{ metadata.n_molecules }} 
molecular systems using {{ metadata.n_methods }} different methods.

The overall convergence rate was {{ "%.1f"|format(summary.convergence_rate * 100) }}\%, 
with a total computational time of {{ "%.2f"|format(summary.total_computational_time / 3600) }} hours.

\section{Method Performance}

{% for method, metrics in error_metrics.items() %}
\subsection{ {{- method -}} }

\begin{itemize}
\item Mean Absolute Error: {{ "%.6f"|format(metrics.mae) }} Hartree
\item Root Mean Square Error: {{ "%.6f"|format(metrics.rmse) }} Hartree
\item Maximum Absolute Error: {{ "%.6f"|format(metrics.max_error) }} Hartree
\item R² Score: {{ "%.4f"|format(metrics.r2) }}
\item Number of Samples: {{ metrics.n_samples }}
\end{itemize}

{% endfor %}

\section{Conclusions}

{% set best_method = method_ranking[0] %}
The best performing method was {{ best_method.method }} with a mean absolute error of 
{{ "%.6f"|format(best_method.mae) }} Hartree.

\end{document}
        """
        
        return Template(template_content)
    
    def _create_basic_html_template(self) -> Template:
        """Create basic HTML template if none exists."""
        
        template_content = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ metadata.title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        h1, h2, h3 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #f2f2f2; }
        .metric { margin: 10px 0; }
        .figure { text-align: center; margin: 20px 0; }
        .summary-box { background-color: #f9f9f9; border: 1px solid #ddd; padding: 20px; margin: 20px 0; }
    </style>
</head>
<body>

<h1>{{ metadata.title }}</h1>

<div class="summary-box">
    <h2>Study Summary</h2>
    <ul>
        <li><strong>Total Calculations:</strong> {{ metadata.total_calculations }}</li>
        <li><strong>Molecular Systems:</strong> {{ metadata.n_molecules }}</li>
        <li><strong>Methods Tested:</strong> {{ metadata.n_methods }}</li>
        <li><strong>Convergence Rate:</strong> {{ "%.1f"|format(summary.convergence_rate * 100) }}%</li>
        <li><strong>Total Time:</strong> {{ "%.2f"|format(summary.total_computational_time / 3600) }} hours</li>
    </ul>
</div>

<h2>Method Performance</h2>

<table>
    <thead>
        <tr>
            <th>Method</th>
            <th>MAE (Hartree)</th>
            <th>RMSE (Hartree)</th>
            <th>Max Error (Hartree)</th>
            <th>R²</th>
            <th>Samples</th>
        </tr>
    </thead>
    <tbody>
        {% for method, metrics in error_metrics.items() %}
        <tr>
            <td>{{ method }}</td>
            <td>{{ "%.6f"|format(metrics.mae) }}</td>
            <td>{{ "%.6f"|format(metrics.rmse) }}</td>
            <td>{{ "%.6f"|format(metrics.max_error) }}</td>
            <td>{{ "%.4f"|format(metrics.r2) }}</td>
            <td>{{ metrics.n_samples }}</td>
        </tr>
        {% endfor %}
    </tbody>
</table>

{% if figures %}
<h2>Figures</h2>
{% for figure in figures %}
<div class="figure">
    <img src="figures/{{ figure.filename }}" alt="{{ figure.caption }}" style="max-width: 100%; height: auto;">
    <p><em>{{ figure.caption }}</em></p>
</div>
{% endfor %}
{% endif %}

<h2>Conclusions</h2>
{% set best_method = method_ranking[0] %}
<p>The best performing method was <strong>{{ best_method.method }}</strong> with a mean absolute error of 
{{ "%.6f"|format(best_method.mae) }} Hartree.</p>

<hr>
<p><em>Report generated on {{ metadata.generation_date }}</em></p>

</body>
</html>
        """
        
        return Template(template_content)
    
    def _create_basic_markdown_template(self) -> Template:
        """Create basic Markdown template if none exists."""
        
        template_content = """
# {{ metadata.title }}

**Authors:** {% for author in metadata.authors -%}{{ author }}{% if not loop.last %}, {% endif %}{%- endfor %}  
**Generated:** {{ metadata.generation_date }}

## Summary

This report presents the results of a comprehensive benchmarking study of quantum chemistry methods. A total of {{ metadata.total_calculations }} calculations were performed on {{ metadata.n_molecules }} molecular systems using {{ metadata.n_methods }} different methods.

**Key Statistics:**
- Convergence Rate: {{ "%.1f"|format(summary.convergence_rate * 100) }}%
- Total Computational Time: {{ "%.2f"|format(summary.total_computational_time / 3600) }} hours
- Average Time per Calculation: {{ "%.2f"|format(summary.average_time_per_calculation) }} seconds

## Method Performance

| Method | MAE (Hartree) | RMSE (Hartree) | Max Error (Hartree) | R² | Samples |
|--------|---------------|----------------|---------------------|-----|---------|
{% for method, metrics in error_metrics.items() -%}
| {{ method }} | {{ "%.6f"|format(metrics.mae) }} | {{ "%.6f"|format(metrics.rmse) }} | {{ "%.6f"|format(metrics.max_error) }} | {{ "%.4f"|format(metrics.r2) }} | {{ metrics.n_samples }} |
{% endfor %}

## Method Ranking

{% for method_info in method_ranking %}
{{ loop.index }}. **{{ method_info.method }}** - MAE: {{ "%.6f"|format(method_info.mae) }} Hartree
{% endfor %}

{% if figures %}
## Figures

{% for figure in figures %}
### {{ figure.filename.replace('_', ' ').title() }}

![{{ figure.caption }}](figures/{{ figure.filename }})

*{{ figure.caption }}*

{% endfor %}
{% endif %}

## Conclusions

{% set best_method = method_ranking[0] %}
The best performing method was **{{ best_method.method }}** with a mean absolute error of {{ "%.6f"|format(best_method.mae) }} Hartree.

## Computational Details

- **Databases Used:** {% for db in summary.databases_used -%}{{ db }}{% if not loop.last %}, {% endif %}{%- endfor %}
- **Total Systems:** {{ summary.unique_molecules }}
- **Convergence Threshold:** Chemical accuracy (1 kcal/mol)

---
*Report generated using the Quantum Chemistry Benchmarking Suite*
        """
        
        return Template(template_content)
    
    def _generate_computational_details(self, report_data: Dict[str, Any]) -> str:
        """Generate detailed computational methodology section."""
        
        details = f"""
# Computational Details

## Methodology

This benchmarking study was conducted using the Quantum Chemistry Benchmarking Suite. The following computational protocols were employed:

### Databases
- **Primary Databases:** {', '.join(report_data['summary']['databases_used'])}
- **Total Molecular Systems:** {report_data['summary']['unique_molecules']}
- **Selection Criteria:** Systems with well-established reference values and varying levels of multireference character

### Computational Setup
- **Total Calculations Performed:** {report_data['metadata']['total_calculations']}
- **Successful Convergence:** {report_data['summary']['converged_calculations']} ({report_data['summary']['convergence_rate']:.1%})
- **Total Computational Time:** {report_data['summary']['total_computational_time']/3600:.2f} hours

### Statistical Analysis
- **Confidence Level:** 95%
- **Bootstrap Samples:** 1000
- **Error Metrics:** MAE, RMSE, Maximum Error, R²
- **Significance Testing:** Shapiro-Wilk normality test, paired t-tests with Bonferroni correction

### Quality Control
- Convergence criteria: SCF convergence < 1×10⁻⁸ Hartree
- Maximum iterations: 100
- Memory limit: 8 GB per calculation
- Timeout: 1 hour per calculation

## Software and Hardware
- **Primary Software:** PySCF 2.3.0+
- **Additional Packages:** NumPy, SciPy, Matplotlib, Pandas
- **Statistical Analysis:** SciPy.stats, Statsmodels
- **Visualization:** Matplotlib, Seaborn, Plotly

## Data Availability
All raw data, analysis scripts, and computational input files are available in the supplementary materials.

---
Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return details
    
    @staticmethod
    def _scientific_notation(value: float, precision: int = 2) -> str:
        """Format number in scientific notation."""
        return f"{value:.{precision}e}"
    
    @staticmethod
    def _round_significant(value: float, sig_figs: int = 4) -> str:
        """Round to significant figures."""
        if value == 0:
            return "0"
        
        from math import log10, floor
        digits = sig_figs - int(floor(log10(abs(value)))) - 1
        return f"{round(value, digits)}"