"""Report generation service for portfolio tearsheets."""

import logging
from datetime import date
from typing import Dict, Optional

import pandas as pd
import numpy as np

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

    def generate_pdf_tearsheet(
        self,
        tearsheet_data: Dict,
        output_path: str,
    ) -> bool:
        """
        Generate PDF tearsheet (placeholder for future implementation).

        Args:
            tearsheet_data: Structured tearsheet data
            output_path: Output file path

        Returns:
            True if successful, False otherwise
        """
        # TODO: Implement PDF generation using reportlab or matplotlib
        logger.warning("PDF generation not yet implemented")
        return False

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
