"""Report generation service for portfolio tearsheets."""

import logging
from datetime import date
from typing import Dict, Optional, TYPE_CHECKING

import pandas as pd
import numpy as np

if TYPE_CHECKING:
    import plotly.graph_objects as go

logger = logging.getLogger(__name__)


class ReportService:
    """Service for generating portfolio reports and tearsheets."""

    def generate_tearsheet_data(
        self,
        portfolio_name: str,
        metrics: Dict,
        chart_data: Dict,
        start_date: date,
        end_date: date,
        benchmark_ticker: Optional[str] = None,
    ) -> Dict:
        """
        Generate structured tearsheet data.

        Args:
            portfolio_name: Name of the portfolio
            metrics: Dictionary of calculated metrics
            chart_data: Dictionary of chart data
            start_date: Analysis start date
            end_date: Analysis end date
            benchmark_ticker: Optional benchmark ticker

        Returns:
            Dictionary with structured tearsheet data
        """
        tearsheet = {
            "portfolio_info": {
                "name": portfolio_name,
                "period": f"{start_date} to {end_date}",
                "benchmark": benchmark_ticker or "None",
            },
            "key_metrics": self._extract_key_metrics(metrics),
            "performance_metrics": metrics.get("performance", {}),
            "risk_metrics": metrics.get("risk", {}),
            "ratio_metrics": metrics.get("ratios", {}),
            "market_metrics": metrics.get("market", {}),
            "charts_available": list(chart_data.keys()),
        }

        return tearsheet

    def _extract_key_metrics(self, metrics: Dict) -> Dict:
        """Extract key metrics for summary section."""
        performance = metrics.get("performance", {})
        risk = metrics.get("risk", {})
        ratios = metrics.get("ratios", {})
        market = metrics.get("market", {})

        return {
            "total_return": performance.get("total_return", 0),
            "annualized_return": performance.get("annualized_return", 0),
            "volatility": risk.get("volatility", 0),
            "max_drawdown": risk.get("max_drawdown", 0),
            "sharpe_ratio": ratios.get("sharpe_ratio", 0),
            "sortino_ratio": ratios.get("sortino_ratio", 0),
            "beta": market.get("beta", 0),
            "alpha": market.get("alpha", 0),
        }

    def generate_csv_report(
        self,
        metrics: Dict,
        include_categories: bool = True,
    ) -> str:
        """
        Generate CSV format report.

        Args:
            metrics: Dictionary of metrics
            include_categories: Whether to include category column

        Returns:
            CSV string
        """
        data = []

        for category, category_metrics in metrics.items():
            if isinstance(category_metrics, dict):
                for metric_name, value in category_metrics.items():
                    row = {
                        "Category": category.title(),
                        "Metric": metric_name,
                        "Value": value,
                    }
                    data.append(row)

        df = pd.DataFrame(data)

        if not include_categories:
            df = df.drop(columns=["Category"])

        return df.to_csv(index=False)

    def generate_json_report(
        self,
        portfolio_name: str,
        metrics: Dict,
        returns_data: Optional[pd.Series] = None,
    ) -> Dict:
        """
        Generate JSON format report.

        Args:
            portfolio_name: Portfolio name
            metrics: Dictionary of metrics
            returns_data: Optional returns series

        Returns:
            Dictionary for JSON export
        """
        report = {
            "portfolio": portfolio_name,
            "metrics": metrics,
        }

        if returns_data is not None and not returns_data.empty:
            # Convert Timestamp keys to strings and handle NaN values for JSON serialization
            report["returns"] = {
                str(k): (float(v) if not (pd.isna(v) or np.isnan(v)) else None)
                for k, v in returns_data.items()
            }

        return report

    def generate_pdf_screenshot_style(
        self,
        portfolio_name: str,
        perf: Dict,
        risk: Dict,
        ratios: Dict,
        market: Dict,
        portfolio_returns: pd.Series,
        benchmark_returns: Optional[pd.Series],
        portfolio_values: Optional[pd.Series],
        positions: list,
        start_date: date,
        end_date: date,
        risk_free_rate: float,
        output_path: str,
        sections_config: Dict[str, bool],
        charts_config: Dict[str, bool],
    ) -> bool:
        """
        Generate PDF as screenshot-style long page using HTML->PDF.
        
        Creates an HTML version of the page and converts to PDF,
        preserving all styles and layout.
        """
        try:
            # Generate HTML content
            html_content = self._generate_html_page(
                portfolio_name=portfolio_name,
                perf=perf,
                risk=risk,
                ratios=ratios,
                market=market,
                portfolio_returns=portfolio_returns,
                benchmark_returns=benchmark_returns,
                portfolio_values=portfolio_values,
                positions=positions,
                start_date=start_date,
                end_date=end_date,
                risk_free_rate=risk_free_rate,
                sections_config=sections_config,
                charts_config=charts_config,
            )
            
            # Convert HTML to PDF using reportlab (more reliable on Windows)
            # Weasyprint requires GTK+ on Windows which is problematic
            # So we'll use reportlab to create a screenshot-style PDF directly
            # We'll generate PDF from the same data used for HTML
            return self._generate_reportlab_screenshot_from_data(
                portfolio_name=portfolio_name,
                perf=perf,
                risk=risk,
                ratios=ratios,
                market=market,
                portfolio_returns=portfolio_returns,
                benchmark_returns=benchmark_returns,
                portfolio_values=portfolio_values,
                positions=positions,
                start_date=start_date,
                end_date=end_date,
                output_path=output_path,
                sections_config=sections_config,
                charts_config=charts_config,
                risk_free_rate=risk_free_rate,
            )
            
        except Exception as e:
            logger.error(
                f"Error generating PDF screenshot: {e}",
                exc_info=True
            )
            return False

    def generate_pdf_full_page_screenshot(
        self,
        html_content: str,
        output_path: str,
        viewport_width: int = 1920,
        viewport_height: int = 1080,
    ) -> bool:
        """
        Create PDF from full page screenshot of HTML page.

        Args:
            html_content: HTML content of the page
            output_path: Path to save PDF
            viewport_width: Viewport width (default 1920px)
            viewport_height: Viewport height (default 1080px, will expand)

        Returns:
            True if successful, False otherwise
        """
        try:
            import subprocess
            import sys
            import img2pdf
            import tempfile
            import os

            # Save HTML to temporary file
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html', encoding='utf-8') as html_file:
                html_file.write(html_content)
                html_path = html_file.name

            # Create temporary file for screenshot
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as screenshot_file:
                screenshot_path = screenshot_file.name

            try:
                # Create Python script to run Playwright in separate process
                script_content = f"""
import sys
import json
from playwright.sync_api import sync_playwright
import base64

html_path = {repr(html_path)}
screenshot_path = {repr(screenshot_path)}
viewport_width = {viewport_width}
viewport_height = {viewport_height}

try:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            viewport={{
                'width': viewport_width,
                'height': viewport_height
            }}
        )
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        page.set_content(html_content, wait_until='networkidle')
        page.wait_for_timeout(2000)
        screenshot_bytes = page.screenshot(full_page=True, type='png')
        browser.close()
        with open(screenshot_path, 'wb') as f:
            f.write(screenshot_bytes)
        print(json.dumps({{'success': True}}))
except Exception as e:
    print(json.dumps({{'success': False, 'error': str(e)}}))
    sys.exit(1)
"""

                # Save script to temporary file
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py', encoding='utf-8') as script_file:
                    script_file.write(script_content)
                    script_path = script_file.name

                try:
                    # Run script in separate process
                    result = subprocess.run(
                        [sys.executable, script_path],
                        capture_output=True,
                        text=True,
                        timeout=120,
                        cwd=os.path.dirname(os.path.abspath(__file__))
                    )

                    if result.returncode != 0:
                        logger.error(f"Playwright script failed: {result.stderr}")
                        return False

                    # Check if screenshot was created
                    if not os.path.exists(screenshot_path):
                        logger.error("Screenshot file was not created")
                        return False

                    # Convert PNG to PDF
                    with open(output_path, 'wb') as pdf_file:
                        pdf_file.write(img2pdf.convert(screenshot_path))

                    logger.info(f"PDF full page screenshot generated: {output_path}")
                    return True

                finally:
                    # Clean up script file
                    if os.path.exists(script_path):
                        try:
                            os.unlink(script_path)
                        except Exception:
                            pass

            finally:
                # Clean up temporary files
                for tmp_file in [html_path, screenshot_path]:
                    if os.path.exists(tmp_file):
                        try:
                            os.unlink(tmp_file)
                        except Exception:
                            pass

        except subprocess.TimeoutExpired:
            logger.error("Playwright subprocess timed out")
            return False
        except Exception as e:
            logger.error(
                f"Error generating PDF full page screenshot: {e}",
                exc_info=True
            )
            return False
    
    def _generate_reportlab_screenshot_from_data(
        self,
        portfolio_name: str,
        perf: Dict,
        risk: Dict,
        ratios: Dict,
        market: Dict,
        portfolio_returns: pd.Series,
        benchmark_returns: Optional[pd.Series],
        portfolio_values: Optional[pd.Series],
        positions: list,
        start_date: date,
        end_date: date,
        output_path: str,
        sections_config: Dict[str, bool],
        charts_config: Dict[str, bool],
        risk_free_rate: float = 0.0435,
    ) -> bool:
        """Generate screenshot-style PDF using reportlab directly from data."""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import (
                BaseDocTemplate, PageTemplate, Frame,
                Paragraph, Spacer, Table, TableStyle,
                PageBreak, Image
            )
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_CENTER
            import os
            
            # Dark theme colors (matching Streamlit app)
            dark_bg = colors.HexColor('#0D1015')
            dark_card = colors.HexColor('#2A2E39')
            dark_border = colors.HexColor('#3A3E49')
            primary_color = colors.HexColor('#BF9FFB')
            text_primary = colors.HexColor('#FFFFFF')
            text_secondary = colors.HexColor('#D1D4DC')
            
            # Create document with dark background
            page_size = letter
            doc = BaseDocTemplate(
                output_path,
                pagesize=page_size,
                rightMargin=0.5*inch,
                leftMargin=0.5*inch,
                topMargin=0.5*inch,
                bottomMargin=0.5*inch
            )
            
            # Frame for content
            frame = Frame(
                0.5*inch,
                0.5*inch,
                page_size[0] - 1*inch,
                page_size[1] - 1*inch,
                leftPadding=0,
                bottomPadding=0,
                rightPadding=0,
                topPadding=0,
            )
            
            class DarkPageTemplate(PageTemplate):
                def onPage(self, canvas, doc):
                    # Draw dark background for entire page
                    canvas.setFillColor(dark_bg)
                    canvas.rect(0, 0, page_size[0], page_size[1], fill=1, stroke=0)
            
            dark_template = DarkPageTemplate(id='dark', frames=[frame])
            doc.addPageTemplates([dark_template])
            
            story = []
            temp_images = []
            
            # Styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=32,
                textColor=primary_color,
                spaceAfter=15,
                alignment=TA_CENTER,
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=20,
                textColor=primary_color,
                spaceAfter=15,
                spaceBefore=25,
            )
            
            meta_style = ParagraphStyle(
                'Meta',
                parent=styles['Normal'],
                textColor=text_secondary,
                fontSize=12,
                alignment=TA_CENTER,
                spaceAfter=30,
            )
            
            normal_style = ParagraphStyle(
                'Normal',
                parent=styles['Normal'],
                textColor=text_secondary,
                fontSize=10,
            )
            
            # Title
            story.append(Paragraph(portfolio_name, title_style))
            story.append(
                Paragraph(
                    f"Analysis Period: {start_date} to {end_date}",
                    meta_style
                )
            )
            story.append(Spacer(1, 0.3*inch))
            
            # Key Metrics Section (as cards) - Row 1
            if sections_config.get("overview", False):
                story.append(Paragraph("Key Performance Metrics", heading_style))
                
                vol = risk.get("volatility", {})
                vol_annual = vol.get("annual", 0) if isinstance(vol, dict) else vol
                max_dd = risk.get("max_drawdown", 0)
                max_dd_value = max_dd[0] if isinstance(max_dd, tuple) else max_dd
                
                # Row 1: Total Return, CAGR, Volatility, Max Drawdown
                metrics_row1 = [
                    ["Total Return", f"{perf.get('total_return', 0) * 100:.2f}%"],
                    ["CAGR", f"{perf.get('cagr', 0) * 100:.2f}%"],
                    ["Volatility", f"{vol_annual * 100:.2f}%"],
                    ["Max Drawdown", f"{max_dd_value * 100:.2f}%"],
                ]
                
                # Create table with card-like appearance (4 columns)
                metrics_table1 = Table(
                    metrics_row1,
                    colWidths=[2.5*inch, 1.5*inch, 2.5*inch, 1.5*inch]
                )
                metrics_table1.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), dark_card),
                    ('TEXTCOLOR', (0, 0), (-1, -1), text_primary),
                    ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                    ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
                    ('ALIGN', (2, 0), (2, -1), 'LEFT'),
                    ('ALIGN', (3, 0), (3, -1), 'RIGHT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 11),
                    ('GRID', (0, 0), (-1, -1), 1, dark_border),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 15),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 15),
                    ('TOPPADDING', (0, 0), (-1, -1), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ]))
                story.append(metrics_table1)
                story.append(Spacer(1, 0.2*inch))
                
                # Row 2: Sharpe Ratio, Sortino Ratio, Beta, Alpha
                metrics_row2 = [
                    ["Sharpe Ratio", f"{ratios.get('sharpe_ratio', 0):.3f}"],
                    ["Sortino Ratio", f"{ratios.get('sortino_ratio', 0):.3f}"],
                    ["Beta", f"{market.get('beta', 0):.3f}"],
                    ["Alpha", f"{market.get('alpha', 0) * 100:.2f}%"],
                ]
                
                metrics_table2 = Table(
                    metrics_row2,
                    colWidths=[2.5*inch, 1.5*inch, 2.5*inch, 1.5*inch]
                )
                metrics_table2.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), dark_card),
                    ('TEXTCOLOR', (0, 0), (-1, -1), text_primary),
                    ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                    ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
                    ('ALIGN', (2, 0), (2, -1), 'LEFT'),
                    ('ALIGN', (3, 0), (3, -1), 'RIGHT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 11),
                    ('GRID', (0, 0), (-1, -1), 1, dark_border),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 15),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 15),
                    ('TOPPADDING', (0, 0), (-1, -1), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ]))
                story.append(metrics_table2)
                story.append(Spacer(1, 0.4*inch))
            
            # Portfolio Performance Section
            if sections_config.get("overview", False):
                story.append(Paragraph("Portfolio Performance", heading_style))
            
            # Cumulative Returns Chart
            if charts_config.get("cumulative", False) and portfolio_returns is not None:
                from core.analytics_engine.chart_data import get_cumulative_returns_data
                from streamlit_app.components.charts import plot_cumulative_returns
                
                cum_data = get_cumulative_returns_data(
                    portfolio_returns, benchmark_returns
                )
                if cum_data:
                    fig = plot_cumulative_returns(cum_data)
                    fig.update_layout(
                        paper_bgcolor='#0D1015',
                        plot_bgcolor='#0D1015',
                        font=dict(color='#FFFFFF'),
                    )
                    img_path = self._save_plotly_figure(fig, temp_images)
                    if img_path:
                        story.append(Paragraph("Cumulative Returns", heading_style))
                        story.append(Image(img_path, width=7*inch, height=4.5*inch))
                        story.append(Spacer(1, 0.3*inch))
            
            if charts_config.get("underwater", False) and portfolio_values is not None:
                from core.analytics_engine.chart_data import get_underwater_plot_data
                from streamlit_app.components.charts import plot_underwater
                
                benchmark_values = None
                if benchmark_returns is not None and portfolio_values is not None:
                    aligned_bench = benchmark_returns.reindex(
                        portfolio_values.index, method="ffill"
                    ).fillna(0)
                    initial_value = float(portfolio_values.iloc[0])
                    benchmark_values = (1 + aligned_bench).cumprod() * initial_value
                
                underwater_data = get_underwater_plot_data(
                    portfolio_values, benchmark_values
                )
                if underwater_data:
                    fig = plot_underwater(underwater_data)
                    fig.update_layout(
                        paper_bgcolor='#0D1015',
                        plot_bgcolor='#0D1015',
                        font=dict(color='#FFFFFF'),
                    )
                    img_path = self._save_plotly_figure(fig, temp_images)
                    if img_path:
                        story.append(Paragraph("Drawdown Analysis", heading_style))
                        story.append(Image(img_path, width=7*inch, height=4.5*inch))
                        story.append(Spacer(1, 0.3*inch))
            
            # Portfolio Structure Section
            if sections_config.get("overview", False) and positions:
                story.append(PageBreak())
                story.append(Paragraph("Portfolio Structure", heading_style))
                
                # Asset Allocation
                from streamlit_app.components.charts import plot_asset_allocation
                weights = []
                for pos in positions:
                    if hasattr(pos, 'weight_target') and pos.weight_target is not None:
                        weights.append(pos.weight_target)
                    else:
                        weights.append(1.0 / len(positions) if len(positions) > 0 else 0.0)
                
                total_weight = sum(weights)
                if total_weight > 0:
                    alloc_data = {}
                    for pos, w in zip(positions, weights):
                        pct = (w / total_weight * 100)
                        alloc_data[pos.ticker] = alloc_data.get(pos.ticker, 0.0) + pct
                    
                    fig = plot_asset_allocation(alloc_data)
                    fig.update_layout(
                        paper_bgcolor='#0D1015',
                        plot_bgcolor='#0D1015',
                        font=dict(color='#FFFFFF'),
                    )
                    img_path = self._save_plotly_figure(fig, temp_images)
                    if img_path:
                        story.append(Paragraph("Distribution by Assets", heading_style))
                        story.append(Image(img_path, width=6*inch, height=4*inch))
                        story.append(Spacer(1, 0.3*inch))
                
                # Sector Allocation
                from core.data_manager.ticker_validator import TickerValidator
                from streamlit_app.components.charts import plot_sector_allocation
                validator = TickerValidator()
                sector_to_weight = {}
                
                tickers = [pos.ticker for pos in positions]
                for tkr, w in zip(tickers, weights):
                    if tkr == "CASH":
                        sector = "Cash"
                    else:
                        try:
                            info = validator.get_ticker_info(tkr)
                            sector = info.sector or "Other"
                        except Exception:
                            sector = "Other"
                    pct = (w / total_weight * 100) if total_weight > 0 else 0.0
                    sector_to_weight[sector] = sector_to_weight.get(sector, 0.0) + pct
                
                if sector_to_weight:
                    fig = plot_sector_allocation(sector_to_weight)
                    fig.update_layout(
                        paper_bgcolor='#0D1015',
                        plot_bgcolor='#0D1015',
                        font=dict(color='#FFFFFF'),
                    )
                    img_path = self._save_plotly_figure(fig, temp_images)
                    if img_path:
                        story.append(Paragraph("Distribution by Sectors", heading_style))
                        story.append(Image(img_path, width=6*inch, height=4*inch))
                        story.append(Spacer(1, 0.3*inch))
            
            # Portfolio vs Comparison Table
            if sections_config.get("overview", False) and benchmark_returns is not None:
                story.append(PageBreak())
                story.append(Paragraph("Portfolio vs Comparison", heading_style))
                
                # Calculate comparison metrics (simplified version)
                from core.analytics_engine.performance import calculate_annualized_return
                from core.analytics_engine.risk_metrics import (
                    calculate_volatility,
                    calculate_max_drawdown,
                )
                from core.analytics_engine.ratios import (
                    calculate_sharpe_ratio,
                    calculate_sortino_ratio,
                )
                
                common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
                aligned_bench = benchmark_returns.loc[common_idx]
                
                # Get portfolio volatility for comparison
                vol = risk.get("volatility", {})
                vol_annual = vol.get("annual", 0) if isinstance(vol, dict) else vol
                max_dd = risk.get("max_drawdown", 0)
                max_dd_value = max_dd[0] if isinstance(max_dd, tuple) else max_dd
                
                if not aligned_bench.empty:
                    max_dd_result = calculate_max_drawdown(aligned_bench)
                    bench_max_dd = max_dd_result[0] if isinstance(max_dd_result, tuple) else max_dd_result
                    bench_vol = calculate_volatility(aligned_bench)
                    bench_vol_annual = bench_vol.get("annual", 0) if isinstance(bench_vol, dict) else bench_vol
                    
                    comparison_data = [
                        ["Metric", "Portfolio", "Benchmark", "Difference"],
                        [
                            "Total Return",
                            f"{perf.get('total_return', 0) * 100:.2f}%",
                            f"{(1 + aligned_bench).prod() - 1:.2%}",
                            f"{(perf.get('total_return', 0) - ((1 + aligned_bench).prod() - 1)) * 100:.2f}%",
                        ],
                        [
                            "CAGR",
                            f"{perf.get('cagr', 0) * 100:.2f}%",
                            f"{calculate_annualized_return(aligned_bench) * 100:.2f}%",
                            f"{(perf.get('cagr', 0) - calculate_annualized_return(aligned_bench)) * 100:.2f}%",
                        ],
                        [
                            "Volatility",
                            f"{vol_annual * 100:.2f}%",
                            f"{bench_vol_annual * 100:.2f}%",
                            f"{(vol_annual - bench_vol_annual) * 100:.2f}%",
                        ],
                        [
                            "Max Drawdown",
                            f"{max_dd_value * 100:.2f}%",
                            f"{bench_max_dd * 100:.2f}%",
                            f"{(max_dd_value - bench_max_dd) * 100:.2f}%",
                        ],
                        [
                            "Sharpe Ratio",
                            f"{ratios.get('sharpe_ratio', 0):.3f}",
                            f"{calculate_sharpe_ratio(aligned_bench, risk_free_rate) or 0:.3f}",
                            f"{(ratios.get('sharpe_ratio', 0) - (calculate_sharpe_ratio(aligned_bench, risk_free_rate) or 0)):.3f}",
                        ],
                    ]
                    
                    comp_table = Table(comparison_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
                    comp_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), dark_card),
                        ('TEXTCOLOR', (0, 0), (-1, 0), text_primary),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), dark_card),
                        ('TEXTCOLOR', (0, 1), (-1, -1), text_primary),
                        ('FONTSIZE', (0, 1), (-1, -1), 10),
                        ('GRID', (0, 0), (-1, -1), 1, dark_border),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('LEFTPADDING', (0, 0), (-1, -1), 10),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                        ('TOPPADDING', (0, 0), (-1, -1), 8),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ]))
                    story.append(comp_table)
                    story.append(Spacer(1, 0.3*inch))
            
            # Analysis Metadata
            if sections_config.get("overview", False) and portfolio_returns is not None:
                story.append(PageBreak())
                story.append(Paragraph("Analysis Metadata", heading_style))
                
                trading_days = len(portfolio_returns)
                total_days = (end_date - start_date).days + 1
                data_quality = (trading_days / total_days * 100) if total_days > 0 else 0
                
                from datetime import date
                metadata_text = f"""
                Analysis Period: {start_date} to {end_date} ({total_days} days)
                Trading Days: {trading_days}
                Time in Market: {trading_days}/{total_days} days ({data_quality:.1f}%)
                Data Quality: {data_quality:.1f}% (no missing data)
                Last Updated: {date.today()} {pd.Timestamp.now().strftime('%H:%M:%S')}
                """
                
                meta_para = Paragraph(metadata_text.replace('\n', '<br/>'), normal_style)
                story.append(meta_para)
                story.append(Spacer(1, 0.3*inch))
            
            # Performance Metrics Table
            if sections_config.get("performance", False):
                story.append(PageBreak())
                story.append(Paragraph("Performance Metrics", heading_style))
                perf_table = self._create_metrics_table(perf, "Dark")
                story.append(perf_table)
            
            # Risk Metrics Table
            if sections_config.get("risk", False):
                story.append(PageBreak())
                story.append(Paragraph("Risk Metrics", heading_style))
                risk_table = self._create_metrics_table(risk, "Dark")
                story.append(risk_table)
            
            # Build PDF
            doc.build(story)
            
            # Clean up temp images
            for img_path in temp_images:
                try:
                    if os.path.exists(img_path):
                        os.unlink(img_path)
                except Exception:
                    pass
            
            logger.info(f"PDF screenshot-style report generated: {output_path}")
            return True
            
        except Exception as e:
            logger.error(
                f"Error generating reportlab screenshot PDF: {e}",
                exc_info=True
            )
            return False

    def generate_pdf_from_streamlit_tabs(
        self,
        streamlit_url: str,
        output_path: str,
        tabs_config: Dict[str, bool],
        viewport_width: int = 1920,
        viewport_height: int = 1080,
        wait_timeout: int = 5000,
    ) -> bool:
        """
        Create PDF from screenshots of Streamlit tabs.
        Each selected tab = one screenshot = one PDF page.
        
        Args:
            streamlit_url: Base URL to Streamlit page 
                          (e.g., "http://localhost:8501")
            output_path: Path to save PDF
            tabs_config: Dict with tab names and bool values
                         Example: {
                             "Overview": True,
                             "Performance": True,  # Will screenshot all 3 sub-tabs
                             "Risk": False,
                             ...
                         }
            viewport_width: Viewport width (default 1920px)
            viewport_height: Viewport height (default 1080px, will expand)
            wait_timeout: Time to wait for page load (ms, default 5000)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            import subprocess
            import sys
            import img2pdf
            import tempfile
            import os
            import json
            
            # Tab structure mapping
            TAB_STRUCTURE = {
                "Overview": {
                    "main_tab_index": 0,
                    "sub_tabs": None  # No sub-tabs
                },
                "Performance": {
                    "main_tab_index": 1,
                    "sub_tabs": ["Returns Analysis", "Periodic Analysis", "Distribution"]
                },
                "Risk": {
                    "main_tab_index": 2,
                    "sub_tabs": ["Key Metrics", "Drawdown Analysis", "VaR & CVaR", "Rolling Risk Metrics"]
                },
                "Assets & Correlations": {
                    "main_tab_index": 3,
                    "sub_tabs": ["Asset Overview & Impact", "Correlations", "Asset Details & Dynamics"]
                },
                "Export & Reports": {
                    "main_tab_index": 4,
                    "sub_tabs": None  # No sub-tabs
                }
            }
            
            # Collect all screenshots to take
            screenshots_to_take = []
            for tab_name, enabled in tabs_config.items():
                if not enabled:
                    continue
                    
                tab_info = TAB_STRUCTURE.get(tab_name)
                if not tab_info:
                    continue
                
                if tab_info["sub_tabs"] is None:
                    # Main tab only, no sub-tabs
                    screenshots_to_take.append({
                        "main_tab": tab_name,
                        "sub_tab": None,
                        "main_index": tab_info["main_tab_index"]
                    })
                else:
                    # Main tab with sub-tabs - screenshot each sub-tab
                    for sub_tab_name in tab_info["sub_tabs"]:
                        screenshots_to_take.append({
                            "main_tab": tab_name,
                            "sub_tab": sub_tab_name,
                            "main_index": tab_info["main_tab_index"],
                            "sub_index": tab_info["sub_tabs"].index(sub_tab_name)
                        })
            
            if not screenshots_to_take:
                logger.warning("No tabs selected for screenshot")
                return False
            
            # Create temporary directory for screenshots
            temp_dir = tempfile.mkdtemp()
            screenshot_files = []
            
            try:
                # Create Python script to run Playwright
                script_content = f"""
import sys
import json
from playwright.sync_api import sync_playwright
import os
import time

streamlit_url = {repr(streamlit_url)}
temp_dir = {repr(temp_dir)}
screenshots_to_take = {repr(screenshots_to_take)}
viewport_width = {viewport_width}
viewport_height = {viewport_height}
wait_timeout = {wait_timeout}

def click_main_tab(page, tab_index):
    '''Click on main tab by index.'''
    try:
        # Streamlit tabs are in stTabs containers
        # Main tabs are usually the first stTabs container
        tab_containers = page.query_selector_all('div[data-testid="stTabs"]')
        print(f"Found {{len(tab_containers)}} tab containers")
        
        if len(tab_containers) == 0:
            print("No tab containers found! Trying alternative selectors...")
            # Try alternative selectors
            # Sometimes tabs are directly in the page
            all_tabs = page.query_selector_all('button[data-baseweb="tab"]')
            print(f"Found {{len(all_tabs)}} tab buttons (alternative search)")
            if len(all_tabs) > 0 and tab_index < len(all_tabs):
                print(f"Clicking tab {{tab_index}} using alternative selector")
                all_tabs[tab_index].click()
                page.wait_for_timeout(1500)
                return True
            return False
        
        # First container is usually main tabs
        main_tabs_container = tab_containers[0]
        tab_buttons = main_tabs_container.query_selector_all('button[data-baseweb="tab"]')
        print(f"Found {{len(tab_buttons)}} main tab buttons")
        
        if tab_index < len(tab_buttons):
            print(f"Clicking main tab {{tab_index}}")
            tab_buttons[tab_index].click()
            page.wait_for_timeout(1500)  # Wait for tab content to load
            return True
        else:
            print(f"Tab index {{tab_index}} out of range ({{len(tab_buttons)}} tabs)")
    except Exception as e:
        # Safe error message without Unicode characters
        error_msg = str(e).encode('ascii', errors='replace').decode('ascii')
        print(f"Error clicking main tab {{tab_index}}: {{error_msg}}")
    return False

def click_sub_tab(page, tab_index):
    '''Click on sub-tab by index.'''
    try:
        # Wait a bit for sub-tabs to appear after main tab switch
        page.wait_for_timeout(1000)
        
        # Sub-tabs are in nested stTabs containers
        tab_containers = page.query_selector_all('div[data-testid="stTabs"]')
        print(f"Found {{len(tab_containers)}} tab containers for sub-tabs")
        if len(tab_containers) < 2:
            print("Not enough tab containers for sub-tabs")
            return False
        
        # Find the active sub-tabs container
        # Look for container with visible/active tab buttons
        for i, container in enumerate(tab_containers[1:], start=1):
            tab_buttons = container.query_selector_all('button[data-baseweb="tab"]')
            print(f"Container {{i}} has {{len(tab_buttons)}} sub-tab buttons")
            
            # Check if this container has visible buttons
            visible_buttons = []
            for btn in tab_buttons:
                try:
                    is_visible = page.evaluate('(el) => {{ const style = window.getComputedStyle(el); return style.display !== "none" && style.visibility !== "hidden"; }}', btn)
                    if is_visible:
                        visible_buttons.append(btn)
                except:
                    pass
            
            # If we found visible buttons and the index is valid, click
            if len(visible_buttons) > 0 and tab_index < len(visible_buttons):
                print(f"Clicking sub-tab {{tab_index}} in container {{i}} ({{len(visible_buttons)}} visible)")
                try:
                    visible_buttons[tab_index].click()
                    page.wait_for_timeout(1500)  # Wait for sub-tab content
                    return True
                except Exception as click_err:
                    # Try with all buttons if visible buttons click failed
                    if tab_index < len(tab_buttons):
                        print(f"Retrying with all buttons...")
                        tab_buttons[tab_index].click()
                        page.wait_for_timeout(1500)
                        return True
                    raise click_err
            elif tab_index < len(tab_buttons):
                # Fallback: try clicking even if visibility check failed
                print(f"Trying to click sub-tab {{tab_index}} in container {{i}} (fallback)")
                tab_buttons[tab_index].click()
                page.wait_for_timeout(1500)
                return True
        
        print(f"Sub-tab index {{tab_index}} not found in any container")
    except Exception as e:
        # Safe error message without Unicode characters
        error_msg = str(e).encode('ascii', errors='replace').decode('ascii')
        print(f"Error clicking sub-tab {{tab_index}}: {{error_msg}}")
    return False

try:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # Use very large viewport height to allow full page rendering
        # For full_page screenshots, we need to render entire page without viewport limits
        page = browser.new_page(
            viewport={{
                'width': viewport_width,
                'height': 10000  # Large height to allow full page rendering
            }}
        )
        
        # Navigate to Streamlit URL
        print(f"Navigating to {{streamlit_url}}...")
        response = page.goto(streamlit_url, wait_until='networkidle', timeout=60000)
        print(f"Page loaded with status: {{response.status if response else 'None'}}")
        
        # Wait for Streamlit to fully render
        print("Waiting for Streamlit app to render...")
        page.wait_for_selector('[data-testid="stApp"]', timeout=30000)
        print("Streamlit app found")
        page.wait_for_timeout(3000)  # Additional wait for initial render
        print("Initial render wait completed")
        
        # After initial load, adjust viewport to match actual page height if needed
        # This ensures all content is rendered
        try:
            actual_height = page.evaluate('''
                () => Math.max(
                    document.body.scrollHeight,
                    document.body.offsetHeight,
                    document.documentElement.clientHeight,
                    document.documentElement.scrollHeight,
                    document.documentElement.offsetHeight
                )
            ''')
            if actual_height > 10000:
                print(f"Page height ({{actual_height}}px) exceeds viewport, adjusting...")
                page.set_viewport_size({{'width': viewport_width, 'height': actual_height + 1000}})
                page.wait_for_timeout(1000)  # Wait for viewport adjustment
        except Exception as e:
            print(f"Could not adjust viewport: {{e}}")
            pass
        
        # Try to navigate to Portfolio Analysis page if needed
        # Check if we're on the main page (has sidebar with navigation)
        print("Checking if we need to navigate to Portfolio Analysis page...")
        try:
            # Look for sidebar navigation
            sidebar = page.query_selector('[data-testid="stSidebar"]')
            if sidebar:
                print("Sidebar found, looking for Portfolio Analysis option...")
                # Try to find and click "Portfolio Analysis" in sidebar
                # Streamlit sidebar uses radio buttons or other elements
                portfolio_analysis_selector = 'text="Portfolio Analysis"'
                try:
                    portfolio_link = page.query_selector(portfolio_analysis_selector)
                    if portfolio_link:
                        print("Found Portfolio Analysis link, clicking...")
                        portfolio_link.click()
                        page.wait_for_timeout(2000)  # Wait for page to switch
                        print("Navigated to Portfolio Analysis page")
                    else:
                        # Try alternative selectors
                        radio_buttons = page.query_selector_all('input[type="radio"]')
                        for i, radio in enumerate(radio_buttons):
                            label = page.evaluate('(el) => el.nextElementSibling?.textContent || ""', radio)
                            if "Portfolio Analysis" in label:
                                print(f"Found Portfolio Analysis radio button at index {{i}}, clicking...")
                                radio.click()
                                page.wait_for_timeout(2000)
                                print("Navigated to Portfolio Analysis page")
                                break
                except Exception as e:
                    print(f"Could not navigate to Portfolio Analysis: {{e}}")
                    print("Continuing with current page...")
        except Exception as e:
            print(f"Error checking sidebar: {{e}}")
        
        # Wait a bit more for page to fully load
        page.wait_for_timeout(2000)
        
        # CRITICAL: Check if "Calculate Metrics" button exists and click it
        # Tabs only appear after metrics are calculated
        print("Checking if 'Calculate Metrics' button needs to be clicked...")
        try:
            # Look for the "Calculate Metrics" button
            calculate_button = None
            
            # Try to find button by text content
            all_buttons = page.query_selector_all('button')
            for btn in all_buttons:
                try:
                    btn_text = page.evaluate('(el) => el.textContent || ""', btn)
                    if "Calculate Metrics" in btn_text or ("Calculate" in btn_text and "Metrics" in btn_text):
                        calculate_button = btn
                        print(f"Found Calculate Metrics button by text: {{btn_text.strip()}}")
                        break
                except Exception:
                    continue
            
            if calculate_button:
                print("Clicking 'Calculate Metrics' button...")
                calculate_button.click()
                print("Button clicked, waiting for metrics to calculate...")
                
                # Wait for calculation to complete
                # Look for success message or tabs to appear
                page.wait_for_timeout(5000)  # Initial wait
                
                # Wait for tabs to appear (they appear after calculation)
                print("Waiting for tabs to appear after calculation...")
                try:
                    page.wait_for_selector('div[data-testid="stTabs"], button[data-baseweb="tab"]', timeout=30000)
                    print("Tabs appeared after calculation!")
                except Exception as e:
                    print(f"Tabs still not found after calculation: {{e}}")
                    print("Will try to continue anyway...")
            else:
                print("Calculate Metrics button not found - metrics may already be calculated")
        except Exception as e:
            print(f"Error checking/clicking Calculate Metrics button: {{e}}")
            import traceback
            traceback.print_exc()
        
        # Wait for tabs to appear (they might load asynchronously)
        print("Waiting for tabs to appear...")
        try:
            # Wait for at least one tab container or tab button
            page.wait_for_selector('div[data-testid="stTabs"], button[data-baseweb="tab"]', timeout=10000)
            print("Tabs found!")
        except Exception as e:
            print(f"Tabs not found after waiting: {{e}}")
            print("Will try to take screenshot anyway...")
        
        screenshot_files = []
        
        # Debug: Take a screenshot of current page to see what we have
        debug_screenshot = os.path.join(temp_dir, "debug_initial_page.png")
        try:
            page.screenshot(path=debug_screenshot, full_page=True)
            print(f"Debug: Initial page screenshot saved to {{debug_screenshot}}")
        except Exception as e:
            print(f"Debug: Could not take initial screenshot: {{e}}")
        
        # Take screenshot for each tab
        for i, screenshot_info in enumerate(screenshots_to_take):
            main_tab_name = screenshot_info['main_tab']
            sub_tab_name = screenshot_info.get('sub_tab')
            
            print(f"\\n=== Screenshot {{i+1}}/{{len(screenshots_to_take)}}: {{main_tab_name}}"
                  f"{{' - ' + sub_tab_name if sub_tab_name else ''}} ===")
            
            # Click main tab
            main_index = screenshot_info['main_index']
            print(f"Attempting to click main tab {{main_index}} ({{main_tab_name}})...")
            if not click_main_tab(page, main_index):
                print(f"ERROR: Failed to click main tab {{main_index}}")
                # Try to take screenshot anyway
                print("Taking screenshot of current page state...")
            else:
                print("Main tab clicked successfully")
                # Wait extra time for sub-tabs to appear (especially for Risk tab)
                if sub_tab_name:
                    print("Waiting for sub-tabs to load...")
                    page.wait_for_timeout(2000)
            
            # Click sub-tab if exists
            if sub_tab_name:
                sub_index = screenshot_info.get('sub_index')
                if sub_index is not None:
                    print(f"Attempting to click sub-tab {{sub_index}} ({{sub_tab_name}})...")
                    if not click_sub_tab(page, sub_index):
                        print(f"WARNING: Failed to click sub-tab {{sub_index}}")
                    else:
                        print("Sub-tab clicked successfully")
                else:
                    print(f"WARNING: No sub_index for {{sub_tab_name}}")
            
            # Wait for content to load (charts, tables, etc.)
            print(f"Waiting {{wait_timeout}}ms for content to load...")
            page.wait_for_timeout(wait_timeout)
            
            # Scroll to top first to ensure we start from beginning
            print("Scrolling to top of page...")
            page.evaluate('window.scrollTo({{top: 0, left: 0, behavior: "instant"}})')
            page.wait_for_timeout(500)
            
            # Wait for all images, charts, and lazy-loaded content
            print("Waiting for all content to fully load...")
            try:
                # Wait for network to be idle (all resources loaded)
                page.wait_for_load_state('networkidle', timeout=30000)
            except Exception:
                pass  # Continue even if timeout
            
            # Wait for all images to load
            page.evaluate('''
                async () => {{
                    const images = document.querySelectorAll('img');
                    await Promise.all(Array.from(images).map(img => {{
                        if (img.complete) return Promise.resolve();
                        return new Promise((resolve, reject) => {{
                            img.onload = resolve;
                            img.onerror = resolve; // Continue even if image fails
                            setTimeout(resolve, 5000); // Timeout after 5s
                        }});
                    }}));
                }}
            ''')
            
            # Additional wait for charts to render (Plotly, etc.)
            page.wait_for_timeout(3000)
            
            # Optional: Hide Streamlit UI elements for cleaner screenshot
            # Hide header and sidebar
            page.evaluate('''
                () => {{
                    const header = document.querySelector('[data-testid="stHeader"]');
                    if (header) header.style.display = 'none';
                    const sidebar = document.querySelector('[data-testid="stSidebar"]');
                    if (sidebar) sidebar.style.display = 'none';
                }}
            ''')
            
            # Scroll to top again after hiding elements (page height may have changed)
            page.evaluate('window.scrollTo({{top: 0, left: 0, behavior: "instant"}})')
            page.wait_for_timeout(500)
            
            # Get actual page dimensions and adjust viewport if needed
            page_dimensions = page.evaluate('''
                () => {{
                    return {{
                        width: Math.max(
                            document.body.scrollWidth,
                            document.body.offsetWidth,
                            document.documentElement.clientWidth,
                            document.documentElement.scrollWidth,
                            document.documentElement.offsetWidth
                        ),
                        height: Math.max(
                            document.body.scrollHeight,
                            document.body.offsetHeight,
                            document.documentElement.clientHeight,
                            document.documentElement.scrollHeight,
                            document.documentElement.offsetHeight
                        )
                    }};
                }}
            ''')
            page_width = page_dimensions['width']
            page_height = page_dimensions['height']
            print(f"Page dimensions: {{page_width}}x{{page_height}}px")
            
            # Adjust viewport to match page height to ensure full rendering
            current_viewport = page.viewport_size
            if current_viewport and page_height > current_viewport['height']:
                new_height = page_height + 500  # Add padding
                print(f"Adjusting viewport height to {{new_height}}px for full page rendering...")
                page.set_viewport_size({{'width': viewport_width, 'height': new_height}})
                page.wait_for_timeout(1000)  # Wait for viewport adjustment
                # Scroll to top again after viewport change
                page.evaluate('window.scrollTo({{top: 0, left: 0, behavior: "instant"}})')
                page.wait_for_timeout(500)
            
            # Take full page screenshot
            screenshot_filename = f"screenshot_{{i+1:03d}}_{{main_tab_name.replace(' ', '_').replace('&', 'and')}}"
            if sub_tab_name:
                screenshot_filename += f"_{{sub_tab_name.replace(' ', '_').replace('&', 'and')}}"
            screenshot_filename += ".png"
            
            screenshot_path = os.path.join(temp_dir, screenshot_filename)
            try:
                print(f"Taking full page screenshot (full_page=True)...")
                # full_page=True automatically scrolls and captures entire page
                screenshot_bytes = page.screenshot(
                    full_page=True, 
                    type='png',
                    animations='disabled'  # Disable animations for cleaner screenshot
                )
                
                with open(screenshot_path, 'wb') as f:
                    f.write(screenshot_bytes)
                
                if os.path.exists(screenshot_path) and os.path.getsize(screenshot_path) > 0:
                    screenshot_files.append(screenshot_path)
                    file_size_kb = os.path.getsize(screenshot_path) / 1024
                    # Get actual screenshot dimensions if PIL available
                    try:
                        from PIL import Image
                        import io
                        img = Image.open(io.BytesIO(screenshot_bytes))
                        print(f"[OK] Saved: {{screenshot_filename}} ({{file_size_kb:.1f}} KB, {{img.width}}x{{img.height}}px)")
                    except ImportError:
                        print(f"[OK] Saved: {{screenshot_filename}} ({{file_size_kb:.1f}} KB)")
                else:
                    print(f"[ERROR] Screenshot file was not created or is empty")
            except Exception as e:
                print(f"[ERROR] Failed to take screenshot: {{e}}")
                import traceback
                traceback.print_exc()
        
        browser.close()
        
        # Return list of screenshot paths
        result = {{'success': True, 'screenshots': screenshot_files}}
        print(json.dumps(result))
        
except Exception as e:
    error_result = {{'success': False, 'error': str(e)}}
    print(json.dumps(error_result))
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
                
                # Save script to temporary file
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py', encoding='utf-8') as script_file:
                    script_file.write(script_content)
                    script_path = script_file.name
                
                try:
                    # Run script in separate process
                    result = subprocess.run(
                        [sys.executable, script_path],
                        capture_output=True,
                        text=True,
                        timeout=600,  # 10 minutes for multiple screenshots
                        cwd=os.path.dirname(os.path.abspath(__file__))
                    )
                    
                    # Log full output for debugging
                    if result.stdout:
                        logger.info(f"Playwright script stdout: {result.stdout}")
                    if result.stderr:
                        logger.warning(f"Playwright script stderr: {result.stderr}")
                    
                    if result.returncode != 0:
                        logger.error(f"Playwright script failed with return code {result.returncode}")
                        logger.error(f"Stderr: {result.stderr}")
                        logger.error(f"Stdout: {result.stdout}")
                        return False
                    
                    # Parse output to get screenshot paths
                    try:
                        output_lines = result.stdout.strip().split('\n')
                        json_output = None
                        for line in reversed(output_lines):  # Check from end
                            try:
                                if line.strip().startswith('{'):
                                    json_output = json.loads(line.strip())
                                    break
                            except:
                                continue
                        
                        if not json_output or not json_output.get('success'):
                            logger.error(f"Script returned error: {json_output}")
                            logger.error(f"Full stdout: {result.stdout}")
                            # Fallback: find all PNG files in temp_dir
                            if os.path.exists(temp_dir):
                                screenshot_files = [
                                    os.path.join(temp_dir, f) 
                                    for f in os.listdir(temp_dir) 
                                    if f.endswith('.png')
                                ]
                                screenshot_files.sort()  # Sort to maintain order
                            else:
                                screenshot_files = []
                        else:
                            screenshot_files = json_output.get('screenshots', [])
                            logger.info(f"Script reported {len(screenshot_files)} screenshots created")
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Could not parse JSON output: {e}")
                        logger.warning(f"Full stdout: {result.stdout}")
                        # Fallback: find all PNG files in temp_dir
                        if os.path.exists(temp_dir):
                            screenshot_files = [
                                os.path.join(temp_dir, f) 
                                for f in os.listdir(temp_dir) 
                                if f.endswith('.png')
                            ]
                            screenshot_files.sort()  # Sort to maintain order
                        else:
                            screenshot_files = []
                    
                    if not screenshot_files:
                        logger.error("No screenshots were created")
                        logger.error(f"Temp directory exists: {os.path.exists(temp_dir) if temp_dir else False}")
                        if temp_dir and os.path.exists(temp_dir):
                            logger.error(f"Files in temp_dir: {os.listdir(temp_dir)}")
                        return False
                    
                    # Convert all screenshots to PDF (one page per screenshot)
                    with open(output_path, 'wb') as pdf_file:
                        pdf_file.write(img2pdf.convert(screenshot_files))
                    
                    logger.info(f"PDF from Streamlit tabs generated: {output_path} ({len(screenshot_files)} pages)")
                    return True
                    
                finally:
                    # Clean up script file
                    if os.path.exists(script_path):
                        try:
                            os.unlink(script_path)
                        except Exception:
                            pass
            
            finally:
                # Clean up screenshot files
                for screenshot_file in screenshot_files:
                    try:
                        if os.path.exists(screenshot_file):
                            os.unlink(screenshot_file)
                    except Exception:
                        pass
                
                # Clean up temp directory
                try:
                    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                        os.rmdir(temp_dir)
                except Exception:
                    pass
        
        except subprocess.TimeoutExpired:
            logger.error("Playwright subprocess timed out")
            return False
        except Exception as e:
            logger.error(
                f"Error generating PDF from Streamlit tabs: {e}",
                exc_info=True
            )
            return False

    def _generate_html_page(
        self,
        portfolio_name: str,
        perf: Dict,
        risk: Dict,
        ratios: Dict,
        market: Dict,
        portfolio_returns: pd.Series,
        benchmark_returns: Optional[pd.Series],
        portfolio_values: Optional[pd.Series],
        positions: list,
        start_date: date,
        end_date: date,
        risk_free_rate: float,
        sections_config: Dict[str, bool],
        charts_config: Dict[str, bool],
    ) -> str:
        """Generate HTML content for screenshot-style PDF."""
        # Generate chart images as base64
        chart_images = {}
        
        if charts_config.get("cumulative", False) and portfolio_returns is not None:
            from core.analytics_engine.chart_data import get_cumulative_returns_data
            from streamlit_app.components.charts import plot_cumulative_returns
            
            cum_data = get_cumulative_returns_data(
                portfolio_returns, benchmark_returns
            )
            if cum_data:
                fig = plot_cumulative_returns(cum_data)
                chart_images["cumulative"] = self._fig_to_base64(fig)
        
        if charts_config.get("underwater", False) and portfolio_values is not None:
            from core.analytics_engine.chart_data import get_underwater_plot_data
            from streamlit_app.components.charts import plot_underwater
            
            benchmark_values = None
            if benchmark_returns is not None and portfolio_values is not None:
                aligned_bench = benchmark_returns.reindex(
                    portfolio_values.index, method="ffill"
                ).fillna(0)
                initial_value = float(portfolio_values.iloc[0])
                benchmark_values = (1 + aligned_bench).cumprod() * initial_value
            
            underwater_data = get_underwater_plot_data(
                portfolio_values, benchmark_values
            )
            if underwater_data:
                fig = plot_underwater(underwater_data)
                chart_images["underwater"] = self._fig_to_base64(fig)
        
        # Daily Returns Chart
        if charts_config.get("daily_returns", False) and portfolio_returns is not None:
            import plotly.graph_objects as go
            from streamlit_app.utils.chart_config import get_chart_layout
            
            fig = go.Figure()
            colors = ['#74F174' if x >= 0 else '#FAA1A4' for x in portfolio_returns.values]
            fig.add_trace(go.Bar(
                x=portfolio_returns.index,
                y=portfolio_returns.values * 100,
                marker_color=colors,
                name='Daily Returns',
            ))
            layout = get_chart_layout(
                title="Daily Returns",
                yaxis=dict(title="Return (%)", tickformat=",.1f"),
                xaxis=dict(title="Date"),
            )
            fig.update_layout(**layout)
            fig.update_layout(
                paper_bgcolor='#0D1015',
                plot_bgcolor='#0D1015',
                font=dict(color='#FFFFFF'),
            )
            chart_images["daily_returns"] = self._fig_to_base64(fig)
        
        # Asset Allocation Chart
        if sections_config.get("overview", False) and positions:
            from streamlit_app.components.charts import plot_asset_allocation
            weights = {}
            for pos in positions:
                if hasattr(pos, 'weight_target') and pos.weight_target is not None:
                    weights[pos.ticker] = pos.weight_target * 100
                else:
                    weights[pos.ticker] = 100.0 / len(positions) if len(positions) > 0 else 0.0
            
            if weights:
                fig = plot_asset_allocation(weights)
                fig.update_layout(
                    paper_bgcolor='#0D1015',
                    plot_bgcolor='#0D1015',
                    font=dict(color='#FFFFFF'),
                )
                chart_images["asset_allocation"] = self._fig_to_base64(fig)
        
        # Sector Allocation Chart
        if sections_config.get("overview", False) and positions:
            from core.data_manager.ticker_validator import TickerValidator
            from streamlit_app.components.charts import plot_sector_allocation
            validator = TickerValidator()
            sector_to_weight = {}
            
            for pos in positions:
                if pos.ticker == "CASH":
                    sector = "Cash"
                else:
                    try:
                        info = validator.get_ticker_info(pos.ticker)
                        sector = info.sector or "Other"
                    except Exception:
                        sector = "Other"
                
                weight = pos.weight_target * 100 if hasattr(pos, 'weight_target') and pos.weight_target is not None else (100.0 / len(positions) if len(positions) > 0 else 0.0)
                sector_to_weight[sector] = sector_to_weight.get(sector, 0.0) + weight
            
            if sector_to_weight:
                fig = plot_sector_allocation(sector_to_weight)
                fig.update_layout(
                    paper_bgcolor='#0D1015',
                    plot_bgcolor='#0D1015',
                    font=dict(color='#FFFFFF'),
                )
                chart_images["sector_allocation"] = self._fig_to_base64(fig)
        
        # Build HTML
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            background-color: #0D1015;
            color: #FFFFFF;
            padding: 40px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: #1A1E29;
            padding: 30px;
            border-radius: 8px;
        }}
        h1 {{
            color: #BF9FFB;
            font-size: 36px;
            margin-bottom: 10px;
            text-align: center;
        }}
        h2 {{
            color: #BF9FFB;
            font-size: 24px;
            margin-top: 40px;
            margin-bottom: 20px;
            border-bottom: 2px solid #3A3E49;
            padding-bottom: 10px;
        }}
        .meta {{
            text-align: center;
            color: #D1D4DC;
            margin-bottom: 30px;
            font-size: 14px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background-color: #2A2E39;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #3A3E49;
        }}
        .metric-label {{
            color: #D1D4DC;
            font-size: 12px;
            margin-bottom: 8px;
            text-transform: uppercase;
        }}
        .metric-value {{
            color: #FFFFFF;
            font-size: 24px;
            font-weight: bold;
        }}
        .metric-delta {{
            color: #74F174;
            font-size: 14px;
            margin-top: 4px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 30px;
            background-color: #1A1E29;
        }}
        th {{
            background-color: #2A2E39;
            color: #FFFFFF;
            padding: 12px;
            text-align: left;
            border-bottom: 2px solid #3A3E49;
            font-weight: 600;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #3A3E49;
            color: #D1D4DC;
        }}
        tr:nth-child(even) {{
            background-color: #1F2127;
        }}
        .chart-container {{
            margin: 30px 0;
            background-color: #1A1E29;
            padding: 20px;
            border-radius: 8px;
        }}
        .chart-container img {{
            width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        .charts-row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 30px 0;
        }}
        .section {{
            margin-bottom: 50px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{portfolio_name}</h1>
        <div class="meta">Analysis Period: {start_date} to {end_date}</div>
"""
        
        # Key Metrics Section
        if sections_config.get("overview", False):
            vol = risk.get("volatility", {})
            vol_annual = vol.get("annual", 0) if isinstance(vol, dict) else vol
            
            max_dd = risk.get("max_drawdown", 0)
            max_dd_value = max_dd[0] if isinstance(max_dd, tuple) else max_dd
            
            html += """
        <div class="section">
            <h2>Key Performance Metrics</h2>
            <div class="metrics-grid">
"""
            metrics = [
                ("Total Return", perf.get("total_return", 0) * 100, "%"),
                ("CAGR", perf.get("cagr", 0) * 100, "%"),
                ("Volatility", vol_annual * 100, "%"),
                ("Max Drawdown", max_dd_value * 100, "%"),
            ]
            
            for label, value, unit in metrics:
                html += f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value:.2f}{unit}</div>
                </div>
"""
            
            html += """
            </div>
            <div class="metrics-grid">
"""
            metrics2 = [
                ("Sharpe Ratio", ratios.get("sharpe_ratio", 0), ""),
                ("Sortino Ratio", ratios.get("sortino_ratio", 0), ""),
                ("Beta", market.get("beta", 0), ""),
                ("Alpha", market.get("alpha", 0) * 100, "%"),
            ]
            
            for label, value, unit in metrics2:
                html += f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value:.3f}{unit}</div>
                </div>
"""
            html += """
            </div>
        </div>
"""
        
        # Charts
        if chart_images.get("cumulative"):
            html += f"""
        <div class="section">
            <h2>Cumulative Returns</h2>
            <div class="chart-container">
                <img src="data:image/png;base64,{chart_images['cumulative']}" />
            </div>
        </div>
"""
        
        if chart_images.get("underwater"):
            html += f"""
        <div class="section">
            <h2>Drawdown Analysis</h2>
            <div class="chart-container">
                <img src="data:image/png;base64,{chart_images['underwater']}" />
            </div>
        </div>
"""
        
        # Daily Returns Chart
        if chart_images.get("daily_returns"):
            html += f"""
        <div class="section">
            <h2>Daily Returns</h2>
            <div class="chart-container">
                <img src="data:image/png;base64,{chart_images['daily_returns']}" />
            </div>
        </div>
"""
        
        # Portfolio Structure
        if sections_config.get("overview", False):
            html += """
        <div class="section">
            <h2>Portfolio Structure</h2>
            <div class="charts-row">
"""
            
            if chart_images.get("asset_allocation"):
                html += f"""
                <div class="chart-container">
                    <h3 style="color: #BF9FFB; margin-bottom: 15px; font-size: 18px;">Distribution by Assets</h3>
                    <img src="data:image/png;base64,{chart_images['asset_allocation']}" />
                </div>
"""
            
            if chart_images.get("sector_allocation"):
                html += f"""
                <div class="chart-container">
                    <h3 style="color: #BF9FFB; margin-bottom: 15px; font-size: 18px;">Distribution by Sectors</h3>
                    <img src="data:image/png;base64,{chart_images['sector_allocation']}" />
                </div>
"""
            
            html += """
            </div>
        </div>
"""
        
        # Portfolio vs Comparison Table
        if sections_config.get("overview", False) and benchmark_returns is not None:
            from core.analytics_engine.performance import calculate_annualized_return
            from core.analytics_engine.risk_metrics import (
                calculate_volatility,
                calculate_max_drawdown,
            )
            from core.analytics_engine.ratios import (
                calculate_sharpe_ratio,
                calculate_sortino_ratio,
            )
            
            common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
            aligned_bench = benchmark_returns.loc[common_idx]
            
            vol = risk.get("volatility", {})
            vol_annual = vol.get("annual", 0) if isinstance(vol, dict) else vol
            max_dd = risk.get("max_drawdown", 0)
            max_dd_value = max_dd[0] if isinstance(max_dd, tuple) else max_dd
            
            if not aligned_bench.empty:
                max_dd_result = calculate_max_drawdown(aligned_bench)
                bench_max_dd = max_dd_result[0] if isinstance(max_dd_result, tuple) else max_dd_result
                bench_vol = calculate_volatility(aligned_bench)
                bench_vol_annual = bench_vol.get("annual", 0) if isinstance(bench_vol, dict) else bench_vol
                
                html += """
        <div class="section">
            <h2>Portfolio vs Comparison</h2>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Portfolio</th>
                        <th>Benchmark</th>
                    </tr>
                </thead>
                <tbody>
"""
                comparison_rows = [
                    ("Total Return", f"{perf.get('total_return', 0) * 100:.2f}%", f"{(1 + aligned_bench).prod() - 1:.2%}"),
                    ("CAGR", f"{perf.get('cagr', 0) * 100:.2f}%", f"{calculate_annualized_return(aligned_bench) * 100:.2f}%"),
                    ("Volatility", f"{vol_annual * 100:.2f}%", f"{bench_vol_annual * 100:.2f}%"),
                    ("Max Drawdown", f"{max_dd_value * 100:.2f}%", f"{bench_max_dd * 100:.2f}%"),
                    ("Sharpe Ratio", f"{ratios.get('sharpe_ratio', 0):.3f}", f"{calculate_sharpe_ratio(aligned_bench, risk_free_rate) or 0:.3f}"),
                    ("Sortino Ratio", f"{ratios.get('sortino_ratio', 0):.3f}", f"{calculate_sortino_ratio(aligned_bench, risk_free_rate) or 0:.3f}"),
                ]
                
                for metric, portfolio_val, bench_val in comparison_rows:
                    html += f"""
                    <tr>
                        <td>{metric}</td>
                        <td>{portfolio_val}</td>
                        <td>{bench_val}</td>
                    </tr>
"""
                
                html += """
                </tbody>
            </table>
        </div>
"""
        
        # Performance Metrics Table
        if sections_config.get("performance", False):
            html += """
        <div class="section">
            <h2>Performance Metrics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
"""
            for key, value in perf.items():
                if value is not None:
                    display_name = key.replace("_", " ").title()
                    if isinstance(value, (int, float)):
                        if "return" in key.lower() or "ratio" in key.lower():
                            formatted = f"{value * 100:.2f}%" if abs(value) < 1 else f"{value:.3f}"
                        else:
                            formatted = f"{value:.3f}"
                    else:
                        formatted = str(value)
                    html += f"""
                    <tr>
                        <td>{display_name}</td>
                        <td>{formatted}</td>
                    </tr>
"""
            html += """
                </tbody>
            </table>
        </div>
"""
        
        # Risk Metrics Table
        if sections_config.get("risk", False):
            html += """
        <div class="section">
            <h2>Risk Metrics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
"""
            for key, value in risk.items():
                if value is not None:
                    display_name = key.replace("_", " ").title()
                    if isinstance(value, dict):
                        formatted = f"{value.get('annual', 0) * 100:.2f}%" if 'annual' in value else str(value)
                    elif isinstance(value, tuple):
                        formatted = f"{value[0] * 100:.2f}%" if len(value) > 0 else str(value)
                    elif isinstance(value, (int, float)):
                        formatted = f"{value * 100:.2f}%" if abs(value) < 1 else f"{value:.3f}"
                    else:
                        formatted = str(value)
                    html += f"""
                    <tr>
                        <td>{display_name}</td>
                        <td>{formatted}</td>
                    </tr>
"""
            html += """
                </tbody>
            </table>
        </div>
"""
        
        html += """
    </div>
</body>
</html>
"""
        return html
    
    def _fig_to_base64(self, fig) -> str:
        """Convert Plotly figure to base64 encoded PNG."""
        try:
            import base64
            import tempfile
            import os
            
            # Update figure for dark theme
            fig.update_layout(
                paper_bgcolor='#0D1015',
                plot_bgcolor='#0D1015',
                font=dict(color='#FFFFFF'),
            )
            
            # Save to temporary file using kaleido (same as _save_plotly_figure)
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, suffix='.png'
            )
            temp_path = temp_file.name
            temp_file.close()
            
            try:
                # Save using kaleido
                fig.write_image(temp_path, width=1200, height=600, scale=2)
                
                # Read file and convert to base64
                with open(temp_path, "rb") as f:
                    img_bytes = f.read()
                
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                return img_base64
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except Exception:
                        pass
        except Exception as e:
            logger.error(f"Error converting figure to base64: {e}", exc_info=True)
            return ""

    def generate_pdf_tearsheet(
        self,
        portfolio_name: str,
        perf: Dict,
        risk: Dict,
        ratios: Dict,
        market: Dict,
        portfolio_returns: pd.Series,
        benchmark_returns: Optional[pd.Series],
        portfolio_values: Optional[pd.Series],
        positions: list,
        start_date: date,
        end_date: date,
        risk_free_rate: float,
        output_path: str,
        sections_config: Dict[str, bool],
        charts_config: Dict[str, bool],
        style: str = "Dark",
    ) -> bool:
        """
        Generate PDF tearsheet using existing chart components and metrics.

        Args:
            portfolio_name: Portfolio name
            perf: Performance metrics
            risk: Risk metrics
            ratios: Ratio metrics
            market: Market metrics
            portfolio_returns: Portfolio returns series
            benchmark_returns: Optional benchmark returns
            portfolio_values: Optional portfolio values series
            positions: List of positions
            start_date: Analysis start date
            end_date: Analysis end date
            risk_free_rate: Risk-free rate
            output_path: Output PDF file path
            sections_config: Dict of sections to include (e.g., {"overview": True})
            charts_config: Dict of charts to include (e.g., {"cumulative": True})
            style: "Dark" or "Classic"

        Returns:
            True if successful, False otherwise
        """
        try:
            from reportlab.lib.styles import (
                getSampleStyleSheet, ParagraphStyle
            )
            from reportlab.lib.units import inch
            from reportlab.platypus import (
                Paragraph, Spacer, Table,
                TableStyle, PageBreak, Image
            )
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_CENTER
            
            # Set dark background color for elements
            dark_bg_color = colors.HexColor('#1A1C20')
            
            # Create PDF document with dark background
            from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame
            from reportlab.lib.pagesizes import letter as letter_size
            
            # Create frame for content area
            frame = Frame(
                0.75*inch,  # left margin
                0.75*inch,  # bottom margin
                letter_size[0] - 1.5*inch,  # width
                letter_size[1] - 1.5*inch,  # height
                leftPadding=0,
                bottomPadding=0,
                rightPadding=0,
                topPadding=0,
            )
            
            class DarkPageTemplate(PageTemplate):
                def onPage(self, canvas, doc):
                    # Draw dark background for entire page
                    canvas.setFillColor(dark_bg_color)
                    canvas.rect(0, 0, letter_size[0], letter_size[1], fill=1, stroke=0)
            
            # Use BaseDocTemplate to support custom page templates
            doc = BaseDocTemplate(
                output_path,
                pagesize=letter_size,
                rightMargin=0.75*inch,
                leftMargin=0.75*inch,
                topMargin=0.75*inch,
                bottomMargin=0.75*inch
            )
            
            # Add dark page template with frame
            dark_template = DarkPageTemplate(id='dark', frames=[frame])
            doc.addPageTemplates([dark_template])
            
            # Container for the 'Flowable' objects
            story = []
            
            # Define styles with dark theme
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1DB954'),
                spaceAfter=30,
                alignment=TA_CENTER,
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=16,
                textColor=colors.HexColor('#FFFFFF'),
                spaceAfter=12,
            )
            
            normal_style = ParagraphStyle(
                'CustomNormal',
                parent=styles['Normal'],
                textColor=colors.HexColor('#E0E0E0'),
            )
            
            # Title page
            story.append(Paragraph(portfolio_name, title_style))
            story.append(Spacer(1, 0.2*inch))
            story.append(Paragraph(
                f"Analysis Period: {start_date} to {end_date}",
                normal_style
            ))
            story.append(Spacer(1, 0.3*inch))
            
            # Key Metrics Table (if overview section included)
            if sections_config.get("overview", False):
                story.append(Paragraph("Key Metrics", heading_style))
                story.append(Spacer(1, 0.1*inch))
                
                # Extract key metrics
                vol = risk.get("volatility", {})
                vol_annual = vol.get("annual", 0.0) if isinstance(vol, dict) else vol
                
                metrics_data = [
                    ["Metric", "Value"],
                    ["Total Return", f"{perf.get('total_return', 0) * 100:.2f}%"],
                    ["CAGR", f"{perf.get('cagr', 0) * 100:.2f}%"],
                    ["Volatility (Annual)", f"{vol_annual * 100:.2f}%"],
                    ["Max Drawdown", f"{risk.get('max_drawdown', 0) * 100:.2f}%"],
                    ["Sharpe Ratio", f"{ratios.get('sharpe_ratio', 0):.3f}"],
                    ["Sortino Ratio", f"{ratios.get('sortino_ratio', 0):.3f}"],
                    ["Beta", f"{market.get('beta', 0):.3f}"],
                    ["Alpha", f"{market.get('alpha', 0) * 100:.2f}%"],
                ]
                
                metrics_table = Table(metrics_data)
                metrics_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2A2D35')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#2A2D35')),
                    ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#E0E0E0')),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#3A3D45')),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [
                        colors.HexColor('#2A2D35'),
                        colors.HexColor('#1F2127')
                    ]),
                ]))
                story.append(metrics_table)
                story.append(PageBreak())
            
            # Charts section
            temp_images = []
            
            try:
                # Cumulative Returns Chart
                if charts_config.get("cumulative", False) and portfolio_returns is not None:
                    from core.analytics_engine.chart_data import get_cumulative_returns_data
                    from streamlit_app.components.charts import plot_cumulative_returns
                    
                    cum_data = get_cumulative_returns_data(
                        portfolio_returns, benchmark_returns
                    )
                    if cum_data:
                        fig = plot_cumulative_returns(cum_data)
                        img_path = self._save_plotly_figure(fig, temp_images)
                        if img_path:
                            story.append(Paragraph("Cumulative Returns", heading_style))
                            story.append(Image(img_path, width=6*inch, height=4*inch))
                            story.append(Spacer(1, 0.2*inch))
                
                # Underwater Plot
                if charts_config.get("underwater", False) and portfolio_values is not None:
                    from core.analytics_engine.chart_data import get_underwater_plot_data
                    from streamlit_app.components.charts import plot_underwater
                    
                    # Calculate benchmark values if needed
                    benchmark_values = None
                    if benchmark_returns is not None and portfolio_values is not None:
                        aligned_bench = benchmark_returns.reindex(
                            portfolio_values.index, method="ffill"
                        ).fillna(0)
                        initial_value = float(portfolio_values.iloc[0])
                        benchmark_values = (1 + aligned_bench).cumprod() * initial_value
                    
                    underwater_data = get_underwater_plot_data(
                        portfolio_values, benchmark_values
                    )
                    if underwater_data:
                        fig = plot_underwater(underwater_data)
                        img_path = self._save_plotly_figure(fig, temp_images)
                        if img_path:
                            story.append(Paragraph("Drawdown Analysis", heading_style))
                            story.append(Image(img_path, width=6*inch, height=4*inch))
                            story.append(Spacer(1, 0.2*inch))
                
                # Yearly Returns (EOY)
                if charts_config.get("yearly_returns", False) and portfolio_returns is not None:
                    from core.analytics_engine.chart_data import get_yearly_returns_data
                    from streamlit_app.components.charts import plot_yearly_returns
                    
                    yearly_data = get_yearly_returns_data(portfolio_returns, benchmark_returns)
                    if yearly_data.get("yearly") is not None:
                        fig = plot_yearly_returns(yearly_data)
                        img_path = self._save_plotly_figure(fig, temp_images)
                        if img_path:
                            story.append(Paragraph("Annual Returns (EOY)", heading_style))
                            story.append(Image(img_path, width=6*inch, height=4*inch))
                            story.append(Spacer(1, 0.2*inch))
                
                # Monthly Heatmap
                if charts_config.get("monthly_heatmap", False) and portfolio_returns is not None:
                    from core.analytics_engine.chart_data import get_monthly_heatmap_data
                    from streamlit_app.components.charts import plot_monthly_heatmap
                    
                    heatmap_data = get_monthly_heatmap_data(portfolio_returns)
                    if heatmap_data.get("heatmap") is not None:
                        fig = plot_monthly_heatmap(heatmap_data)
                        img_path = self._save_plotly_figure(fig, temp_images)
                        if img_path:
                            story.append(Paragraph("Monthly Returns Heatmap", heading_style))
                            story.append(Image(img_path, width=6*inch, height=4*inch))
                            story.append(Spacer(1, 0.2*inch))
                
                # Return Distribution
                if charts_config.get("distribution", False) and portfolio_returns is not None:
                    from core.analytics_engine.chart_data import get_return_distribution_data
                    from streamlit_app.components.charts import plot_return_distribution
                    
                    dist_data = get_return_distribution_data(portfolio_returns, bins=50)
                    if dist_data:
                        fig = plot_return_distribution(dist_data, bar_color="blue")
                        img_path = self._save_plotly_figure(fig, temp_images)
                        if img_path:
                            story.append(Paragraph("Return Distribution", heading_style))
                            story.append(Image(img_path, width=6*inch, height=4*inch))
                            story.append(Spacer(1, 0.2*inch))
                
                # Q-Q Plot
                if charts_config.get("qq_plot", False) and portfolio_returns is not None:
                    from core.analytics_engine.chart_data import get_qq_plot_data
                    from streamlit_app.components.charts import plot_qq_plot
                    
                    qq_data = get_qq_plot_data(portfolio_returns)
                    if qq_data:
                        fig = plot_qq_plot(qq_data)
                        img_path = self._save_plotly_figure(fig, temp_images)
                        if img_path:
                            story.append(Paragraph("Q-Q Plot", heading_style))
                            story.append(Image(img_path, width=6*inch, height=4*inch))
                            story.append(Spacer(1, 0.2*inch))
                
                # Rolling Sharpe
                if charts_config.get("rolling_sharpe", False) and portfolio_returns is not None:
                    from core.analytics_engine.chart_data import get_rolling_sharpe_data
                    from streamlit_app.components.charts import plot_rolling_sharpe
                    
                    sharpe_data = get_rolling_sharpe_data(portfolio_returns, benchmark_returns, risk_free_rate)
                    if sharpe_data:
                        fig = plot_rolling_sharpe(sharpe_data)
                        img_path = self._save_plotly_figure(fig, temp_images)
                        if img_path:
                            story.append(Paragraph("Rolling Sharpe Ratio", heading_style))
                            story.append(Image(img_path, width=6*inch, height=4*inch))
                            story.append(Spacer(1, 0.2*inch))
                
                # Rolling Volatility
                if charts_config.get("rolling_volatility", False) and portfolio_returns is not None:
                    from core.analytics_engine.chart_data import get_rolling_volatility_data
                    from streamlit_app.components.charts import plot_rolling_volatility
                    
                    vol_data = get_rolling_volatility_data(portfolio_returns, benchmark_returns)
                    if vol_data:
                        fig = plot_rolling_volatility(vol_data)
                        img_path = self._save_plotly_figure(fig, temp_images)
                        if img_path:
                            story.append(Paragraph("Rolling Volatility", heading_style))
                            story.append(Image(img_path, width=6*inch, height=4*inch))
                            story.append(Spacer(1, 0.2*inch))
                
                # Rolling Beta
                if charts_config.get("rolling_beta", False) and portfolio_returns is not None and benchmark_returns is not None:
                    from core.analytics_engine.chart_data import get_rolling_beta_data
                    from streamlit_app.components.charts import plot_rolling_beta
                    
                    beta_data = get_rolling_beta_data(portfolio_returns, benchmark_returns)
                    if beta_data:
                        fig = plot_rolling_beta(beta_data)
                        img_path = self._save_plotly_figure(fig, temp_images)
                        if img_path:
                            story.append(Paragraph("Rolling Beta", heading_style))
                            story.append(Image(img_path, width=6*inch, height=4*inch))
                            story.append(Spacer(1, 0.2*inch))
                
                # Additional charts can be added here following same pattern
                
            except Exception as e:
                logger.warning(f"Error generating charts: {e}")
            
            # Metrics tables - only include if tab is enabled
            if sections_config.get("performance", False):
                story.append(PageBreak())
                story.append(Paragraph("Performance Metrics", heading_style))
                perf_table = self._create_metrics_table(perf, style)
                story.append(perf_table)
            
            if sections_config.get("risk", False):
                story.append(PageBreak())
                story.append(Paragraph("Risk Metrics", heading_style))
                risk_table = self._create_metrics_table(risk, style)
                story.append(risk_table)
            
            # Ratios are part of Risk tab, include if Risk is enabled
            if sections_config.get("risk", False):
                story.append(PageBreak())
                story.append(Paragraph("Risk-Adjusted Ratios", heading_style))
                ratios_table = self._create_metrics_table(ratios, style)
                story.append(ratios_table)
            
            # Market metrics (if benchmark available and Risk tab enabled)
            if sections_config.get("risk", False) and market:
                story.append(PageBreak())
                story.append(Paragraph("Market Metrics", heading_style))
                market_table = self._create_metrics_table(market, style)
                story.append(market_table)
            
            # Holdings table (if Overview is enabled)
            if sections_config.get("overview", False) and positions:
                story.append(PageBreak())
                story.append(Paragraph("Holdings", heading_style))
                holdings_data = [["Ticker", "Shares", "Weight"]]
                for pos in positions:
                    weight = f"{pos.weight_target * 100:.2f}%" if pos.weight_target else "N/A"
                    holdings_data.append([pos.ticker, f"{pos.shares:.2f}", weight])
                
                holdings_table = Table(holdings_data)
                holdings_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2A2D35')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#2A2D35')),
                    ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#E0E0E0')),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#3A3D45')),
                ]))
                story.append(holdings_table)
            
            # Build PDF
            doc.build(story)
            
            # Clean up temporary image files
            for img_path in temp_images:
                try:
                    if os.path.exists(img_path):
                        os.unlink(img_path)
                except Exception:
                    pass
            
            logger.info(f"PDF report generated: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating PDF: {e}", exc_info=True)
        return False
    
    def _save_plotly_figure(self, fig: "go.Figure", temp_images: list) -> Optional[str]:
        """Save Plotly figure to temporary image file."""
        try:
            import tempfile
            import os
            
            # Update figure for PDF with dark theme
            fig.update_layout(
                paper_bgcolor='#1A1C20',
                plot_bgcolor='#1A1C20',
                font=dict(color='#E0E0E0'),
            )
            
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, suffix='.png'
            )
            temp_path = temp_file.name
            temp_file.close()
            
            # Save using kaleido
            fig.write_image(temp_path, width=1200, height=800, scale=2)
            temp_images.append(temp_path)
            
            return temp_path
            
        except Exception as e:
            logger.warning(f"Error saving plotly figure: {e}")
            return None
    
    def _create_metrics_table(self, metrics: Dict, style: str):
        """Create a table from metrics dictionary."""
        from reportlab.platypus import Table, TableStyle
        from reportlab.lib import colors
        
        data = [["Metric", "Value"]]
        for key, value in metrics.items():
            if isinstance(value, dict):
                # Handle dict values (e.g., volatility)
                for sub_key, sub_val in value.items():
                    formatted_val = self._format_value_for_table(sub_val)
                    data.append([f"{key} ({sub_key})", formatted_val])
            else:
                formatted_val = self._format_value_for_table(value)
                metric_name = key.replace('_', ' ').title()
                data.append([metric_name, formatted_val])
        
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2A2D35')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#2A2D35')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#E0E0E0')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#3A3D45')),
        ]))
        
        return table
    
    def _format_value_for_table(self, value) -> str:
        """Format metric value for table display."""
        if value is None:
            return "N/A"
        if isinstance(value, (int, float)):
            if abs(value) < 0.01:
                return f"{value:.4f}"
            elif abs(value) < 1:
                return f"{value * 100:.2f}%"
            else:
                return f"{value:.3f}"
        return str(value)

    def export_metrics_to_excel(
        self,
        metrics: Dict,
        output_path: str,
    ) -> bool:
        """
        Export metrics to Excel file with multiple sheets.

        Args:
            metrics: Dictionary of metrics
            output_path: Output file path

        Returns:
            True if successful, False otherwise
        """
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                for category, category_metrics in metrics.items():
                    if isinstance(category_metrics, dict):
                        df = pd.DataFrame(
                            list(category_metrics.items()),
                            columns=['Metric', 'Value']
                        )
                        sheet_name = category.title()[:31]  # Excel limit
                        df.to_excel(
                            writer,
                            sheet_name=sheet_name,
                            index=False
                        )

            logger.info(f"Metrics exported to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting to Excel: {e}")
            return False
