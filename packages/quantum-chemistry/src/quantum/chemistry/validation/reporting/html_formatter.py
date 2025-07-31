"""
HTML Formatting Utilities for Quantum Chemistry Reports

This module provides HTML formatting capabilities for creating
interactive web reports with proper chemical notation, tables,
and scientific formatting.
"""

import re
from typing import Dict, List, Any, Optional, Union
import html
import logging

logger = logging.getLogger(__name__)


class HTMLFormatter:
    """HTML formatting utilities for quantum chemistry reports."""
    
    def __init__(self):
        # Chemical element regex
        self.element_pattern = re.compile(r'\b([A-Z][a-z]?)\b')
        
        # Unit mappings for HTML
        self.unit_mappings = {
            'hartree': 'E<sub>h</sub>',
            'eV': 'eV',
            'kcal/mol': 'kcal mol<sup>-1</sup>',
            'kJ/mol': 'kJ mol<sup>-1</sup>',
            'angstrom': 'Å',
            'bohr': 'a<sub>0</sub>',
            'MHz': 'MHz',
            'cm-1': 'cm<sup>-1</sup>'
        }
        
        # CSS styles for scientific formatting
        self.base_styles = """
        <style>
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                line-height: 1.6; 
                color: #333; 
                max-width: 1200px; 
                margin: 0 auto; 
                padding: 20px; 
            }
            h1, h2, h3, h4, h5, h6 { 
                color: #2c3e50; 
                margin-top: 2em; 
            }
            h1 { 
                border-bottom: 3px solid #3498db; 
                padding-bottom: 10px; 
            }
            h2 { 
                border-bottom: 2px solid #ecf0f1; 
                padding-bottom: 5px; 
            }
            .chemical-formula { 
                font-family: 'Courier New', monospace; 
                font-weight: bold; 
            }
            .method-name { 
                font-variant: small-caps; 
                font-weight: bold; 
                color: #8e44ad; 
            }
            .scientific-number { 
                font-family: 'Courier New', monospace; 
            }
            .error-excellent { color: #27ae60; }
            .error-good { color: #f39c12; }
            .error-acceptable { color: #e67e22; }
            .error-poor { color: #e74c3c; }
            
            table { 
                border-collapse: collapse; 
                width: 100%; 
                margin: 20px 0; 
                box-shadow: 0 2px 8px rgba(0,0,0,0.1); 
            }
            th { 
                background-color: #34495e; 
                color: white; 
                padding: 12px; 
                text-align: left; 
                font-weight: bold; 
            }
            td { 
                padding: 10px 12px; 
                border-bottom: 1px solid #ecf0f1; 
            }
            tr:nth-child(even) { 
                background-color: #f8f9fa; 
            }
            tr:hover { 
                background-color: #e8f4fd; 
            }
            
            .summary-box { 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; 
                padding: 20px; 
                border-radius: 10px; 
                margin: 20px 0; 
            }
            .summary-box h2 { 
                color: white; 
                margin-top: 0; 
            }
            
            .metric-card { 
                background: white; 
                border: 1px solid #ddd; 
                border-radius: 8px; 
                padding: 15px; 
                margin: 10px; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
                display: inline-block; 
                min-width: 200px; 
            }
            .metric-title { 
                font-size: 0.9em; 
                color: #7f8c8d; 
                margin-bottom: 5px; 
            }
            .metric-value { 
                font-size: 1.5em; 
                font-weight: bold; 
                color: #2c3e50; 
            }
            
            .figure { 
                text-align: center; 
                margin: 30px 0; 
                page-break-inside: avoid; 
            }
            .figure img { 
                max-width: 100%; 
                height: auto; 
                border: 1px solid #ddd; 
                border-radius: 5px; 
                box-shadow: 0 4px 8px rgba(0,0,0,0.1); 
            }
            .figure-caption { 
                font-size: 0.9em; 
                color: #7f8c8d; 
                margin-top: 10px; 
                font-style: italic; 
            }
            
            .toc { 
                background-color: #f8f9fa; 
                border: 1px solid #e9ecef; 
                border-radius: 5px; 
                padding: 20px; 
                margin: 20px 0; 
            }
            .toc ul { 
                list-style-type: none; 
                padding-left: 0; 
            }
            .toc li { 
                margin: 5px 0; 
            }
            .toc a { 
                text-decoration: none; 
                color: #3498db; 
            }
            .toc a:hover { 
                text-decoration: underline; 
            }
            
            .progress-bar { 
                width: 100%; 
                background-color: #ecf0f1; 
                border-radius: 10px; 
                overflow: hidden; 
                margin: 10px 0; 
            }
            .progress-fill { 
                height: 20px; 
                background: linear-gradient(90deg, #27ae60, #2ecc71); 
                text-align: center; 
                line-height: 20px; 
                color: white; 
                font-size: 0.8em; 
                font-weight: bold; 
            }
            
            @media print { 
                body { margin: 0; } 
                .no-print { display: none; } 
                .figure { page-break-inside: avoid; } 
                table { page-break-inside: avoid; } 
            }
        </style>
        """
    
    def escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return html.escape(str(text))
    
    def format_chemical_formula(self, formula: str) -> str:
        """Format chemical formula with proper subscripts."""
        
        # Replace numbers with subscripts
        formatted = re.sub(r'(\d+)', r'<sub>\1</sub>', formula)
        
        # Handle charge notation
        formatted = re.sub(r'\+(\d*)', r'<sup>+\1</sup>', formatted)
        formatted = re.sub(r'-(\d*)', r'<sup>-\1</sup>', formatted)
        
        return f'<span class="chemical-formula">{formatted}</span>'
    
    def format_method_name(self, method: str) -> str:
        """Format method name with proper typography."""
        
        return f'<span class="method-name">{self.escape_html(method)}</span>'
    
    def format_scientific_number(
        self, 
        value: float, 
        precision: int = 4,
        unit: Optional[str] = None
    ) -> str:
        """Format number with scientific notation and units."""
        
        if abs(value) < 1e-3 or abs(value) > 1e4:
            formatted = f"{value:.{precision}e}"
        else:
            formatted = f"{value:.{precision}f}"
        
        result = f'<span class="scientific-number">{formatted}</span>'
        
        if unit and unit in self.unit_mappings:
            result += f" {self.unit_mappings[unit]}"
        elif unit:
            result += f" {self.escape_html(unit)}"
        
        return result
    
    def format_number_with_uncertainty(
        self, 
        value: float, 
        uncertainty: Optional[float] = None,
        unit: Optional[str] = None,
        precision: int = 4
    ) -> str:
        """Format number with uncertainty."""
        
        if uncertainty is not None:
            if abs(value) < 1e-3 or abs(value) > 1e4:
                value_str = f"{value:.{precision}e}"
                unc_str = f"{uncertainty:.{precision}e}"
            else:
                value_str = f"{value:.{precision}f}"
                unc_str = f"{uncertainty:.{precision}f}"
            
            formatted = f"{value_str} ± {unc_str}"
        else:
            formatted = self.format_scientific_number(value, precision)
            return formatted  # Already includes unit formatting
        
        result = f'<span class="scientific-number">{formatted}</span>'
        
        if unit and unit in self.unit_mappings:
            result += f" {self.unit_mappings[unit]}"
        elif unit:
            result += f" {self.escape_html(unit)}"
        
        return result
    
    def format_error_class(self, error: float, unit: str = "hartree") -> str:
        """Format error with color coding based on magnitude."""
        
        # Error thresholds in Hartree
        if unit.lower() in ["hartree", "eh"]:
            if error < 0.001:  # < 1 mHartree
                css_class = "error-excellent"
            elif error < 0.005:  # < 5 mHartree
                css_class = "error-good"
            elif error < 0.02:  # < 20 mHartree
                css_class = "error-acceptable"
            else:
                css_class = "error-poor"
        else:
            # Default classification for other units
            css_class = "scientific-number"
        
        formatted_value = self.format_scientific_number(error, precision=6, unit=unit)
        return f'<span class="{css_class}">{formatted_value}</span>'
    
    def create_table(
        self, 
        data: List[Dict[str, Any]], 
        title: Optional[str] = None,
        caption: Optional[str] = None,
        sortable: bool = True,
        error_columns: Optional[List[str]] = None
    ) -> str:
        """Create HTML table with optional sorting and error highlighting."""
        
        if not data:
            return "<p>No data available</p>"
        
        headers = list(data[0].keys())
        error_columns = error_columns or []
        
        # Start table
        table_parts = ["<table>"]
        
        if title:
            table_parts.append(f"<caption><strong>{self.escape_html(title)}</strong></caption>")
        
        # Header
        table_parts.append("<thead><tr>")
        for header in headers:
            escaped_header = self.escape_html(str(header))
            if sortable:
                table_parts.append(f'<th onclick="sortTable(this)">{escaped_header}</th>')
            else:
                table_parts.append(f"<th>{escaped_header}</th>")
        table_parts.append("</tr></thead>")
        
        # Body
        table_parts.append("<tbody>")
        for row in data:
            table_parts.append("<tr>")
            for header in headers:
                value = row.get(header, "")
                
                if isinstance(value, float):
                    if header.lower() in error_columns:
                        formatted_value = self.format_error_class(value)
                    else:
                        formatted_value = self.format_scientific_number(value, precision=4)
                elif header.lower().endswith('_method') or 'method' in header.lower():
                    formatted_value = self.format_method_name(str(value))
                elif any(term in header.lower() for term in ['formula', 'molecule', 'chemical']):
                    formatted_value = self.format_chemical_formula(str(value))
                else:
                    formatted_value = self.escape_html(str(value))
                
                table_parts.append(f"<td>{formatted_value}</td>")
            table_parts.append("</tr>")
        table_parts.append("</tbody>")
        
        table_parts.append("</table>")
        
        if caption:
            table_parts.append(f'<p class="table-caption"><em>{self.escape_html(caption)}</em></p>')
        
        # Add sorting JavaScript if needed
        if sortable:
            table_parts.append(self._get_table_sorting_script())
        
        return "\n".join(table_parts)
    
    def create_summary_box(self, data: Dict[str, Any], title: str = "Summary") -> str:
        """Create summary box with key statistics."""
        
        box_parts = [f'<div class="summary-box">']
        box_parts.append(f"<h2>{self.escape_html(title)}</h2>")
        
        for key, value in data.items():
            key_formatted = key.replace('_', ' ').title()
            
            if isinstance(value, float):
                if key.lower().endswith('_rate') or key.lower().endswith('_percentage'):
                    value_formatted = f"{value:.1%}"
                elif key.lower().endswith('_time'):
                    if value > 3600:
                        value_formatted = f"{value/3600:.2f} hours"
                    elif value > 60:
                        value_formatted = f"{value/60:.1f} minutes"
                    else:
                        value_formatted = f"{value:.1f} seconds"
                else:
                    value_formatted = self.format_scientific_number(value)
            elif isinstance(value, list):
                value_formatted = ", ".join(str(v) for v in value)
            else:
                value_formatted = self.escape_html(str(value))
            
            box_parts.append(f"<p><strong>{key_formatted}:</strong> {value_formatted}</p>")
        
        box_parts.append("</div>")
        
        return "\n".join(box_parts)
    
    def create_metric_cards(self, metrics: Dict[str, Dict[str, Any]]) -> str:
        """Create metric cards for method comparison."""
        
        cards_html = []
        
        for method, method_metrics in metrics.items():
            card_parts = ['<div class="metric-card">']
            card_parts.append(f'<div class="metric-title">{self.format_method_name(method)}</div>')
            
            # Key metrics
            if 'mae' in method_metrics:
                mae_formatted = self.format_scientific_number(method_metrics['mae'], unit='hartree')
                card_parts.append(f'<div class="metric-value">{mae_formatted}</div>')
                card_parts.append('<div class="metric-title">Mean Absolute Error</div>')
            
            if 'r2' in method_metrics:
                card_parts.append(f'<div class="metric-value">{method_metrics["r2"]:.4f}</div>')
                card_parts.append('<div class="metric-title">R² Score</div>')
            
            if 'n_samples' in method_metrics:
                card_parts.append(f'<div class="metric-value">{method_metrics["n_samples"]}</div>')
                card_parts.append('<div class="metric-title">Samples</div>')
            
            card_parts.append('</div>')
            cards_html.append("\n".join(card_parts))
        
        return '<div class="metric-cards-container">' + "\n".join(cards_html) + '</div>'
    
    def create_figure(
        self, 
        image_path: str, 
        caption: str,
        alt_text: Optional[str] = None,
        width: Optional[str] = None
    ) -> str:
        """Create figure with caption."""
        
        alt_text = alt_text or caption
        style = f'width: {width};' if width else ''
        
        return f"""
        <div class="figure">
            <img src="{self.escape_html(image_path)}" alt="{self.escape_html(alt_text)}" style="{style}">
            <div class="figure-caption">{self.escape_html(caption)}</div>
        </div>
        """
    
    def create_progress_bar(self, value: float, max_value: float = 100.0, label: str = "") -> str:
        """Create progress bar for visualization."""
        
        percentage = (value / max_value) * 100
        
        return f"""
        <div class="progress-bar">
            <div class="progress-fill" style="width: {percentage:.1f}%">
                {label} {percentage:.1f}%
            </div>
        </div>
        """
    
    def create_table_of_contents(self, sections: List[Dict[str, str]]) -> str:
        """Create table of contents."""
        
        toc_parts = ['<div class="toc">']
        toc_parts.append('<h3>Table of Contents</h3>')
        toc_parts.append('<ul>')
        
        for section in sections:
            title = section.get('title', '')
            anchor = section.get('anchor', title.lower().replace(' ', '-'))
            toc_parts.append(f'<li><a href="#{anchor}">{self.escape_html(title)}</a></li>')
        
        toc_parts.append('</ul>')
        toc_parts.append('</div>')
        
        return "\n".join(toc_parts)
    
    def format_all_tables(self, tables_data: Dict[str, Dict[str, Any]]) -> str:
        """Format all tables in the report data."""
        
        formatted_tables = []
        
        for table_name, table_info in tables_data.items():
            title = table_info.get('title', table_name.replace('_', ' ').title())
            data = table_info.get('data', [])
            caption = table_info.get('caption', '')
            
            # Determine error columns
            error_columns = []
            if data:
                for key in data[0].keys():
                    if any(term in key.lower() for term in ['error', 'mae', 'rmse']):
                        error_columns.append(key.lower())
            
            table_html = self.create_table(
                data, 
                title=title, 
                caption=caption,
                error_columns=error_columns
            )
            
            formatted_tables.append(f'<div id="{table_name}">{table_html}</div>')
        
        return "\n".join(formatted_tables)
    
    def create_document_head(self, title: str, additional_css: str = "") -> str:
        """Create HTML document head with styles."""
        
        return f"""
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{self.escape_html(title)}</title>
            {self.base_styles}
            {additional_css}
        </head>
        """
    
    def _get_table_sorting_script(self) -> str:
        """Get JavaScript for table sorting."""
        
        return """
        <script>
        function sortTable(header) {
            const table = header.closest('table');
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const headerIndex = Array.from(header.parentNode.children).indexOf(header);
            
            // Determine if we're sorting numbers or text
            const isNumeric = rows.every(row => {
                const cell = row.children[headerIndex];
                const text = cell.textContent.trim();
                return !isNaN(parseFloat(text)) || text === '';
            });
            
            // Sort rows
            rows.sort((a, b) => {
                const aText = a.children[headerIndex].textContent.trim();
                const bText = b.children[headerIndex].textContent.trim();
                
                if (isNumeric) {
                    return parseFloat(aText || '0') - parseFloat(bText || '0');
                } else {
                    return aText.localeCompare(bText);
                }
            });
            
            // Re-append sorted rows
            rows.forEach(row => tbody.appendChild(row));
        }
        </script>
        """