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
