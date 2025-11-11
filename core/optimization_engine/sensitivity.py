"""Sensitivity analysis for portfolio optimization."""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from core.optimization_engine.base import BaseOptimizer, OptimizationResult

logger = logging.getLogger(__name__)


class SensitivityAnalyzer:
    """
    Analyzer for sensitivity of optimization results to parameter changes.
    
    Performs sensitivity analysis on:
    - Expected returns
    - Covariance matrix
    - Constraints
    """
    
    def __init__(
        self,
        optimizer: BaseOptimizer,
        base_result: OptimizationResult,
    ) -> None:
        """
        Initialize sensitivity analyzer.
        
        Args:
            optimizer: Optimizer instance used for base optimization
            base_result: Base optimization result
        """
        self.optimizer = optimizer
        self.base_result = base_result
        self.tickers = optimizer.tickers
    
    def analyze_return_sensitivity(
        self,
        variation_range: float = 0.1,
        num_points: int = 10,
        constraints: Optional[Dict[str, any]] = None,
    ) -> pd.DataFrame:
        """
        Analyze sensitivity to changes in expected returns.
        
        Args:
            variation_range: Range of variation (±variation_range)
            num_points: Number of points to test
        
        Returns:
            DataFrame with weights for each variation point
        """
        results = []
        
        # Get base mean returns
        base_returns = self.optimizer._mean_returns.values
        
        # Generate variation points
        variations = np.linspace(
            -variation_range,
            variation_range,
            num_points,
        )
        
        for var in variations:
            # Adjust returns
            adjusted_returns = base_returns * (1.0 + var)
            
            # Temporarily modify optimizer's mean returns
            original_returns = self.optimizer._mean_returns.copy()
            self.optimizer._mean_returns = pd.Series(
                adjusted_returns,
                index=self.optimizer.tickers,
            )
            
            try:
                # Re-optimize with constraints
                result = self.optimizer.optimize(constraints=constraints)
                
                # Store results
                row = {"variation": var}
                for i, ticker in enumerate(self.tickers):
                    row[ticker] = float(result.weights[i])
                results.append(row)
            except Exception as e:
                logger.warning(
                    f"Sensitivity analysis failed for variation {var}: {e}"
                )
            finally:
                # Restore original returns
                self.optimizer._mean_returns = original_returns
        
        return pd.DataFrame(results)
    
    def analyze_covariance_sensitivity(
        self,
        variation_range: float = 0.1,
        num_points: int = 10,
        constraints: Optional[Dict[str, any]] = None,
    ) -> pd.DataFrame:
        """
        Analyze sensitivity to changes in covariance matrix.
        
        Args:
            variation_range: Range of variation (±variation_range)
            num_points: Number of points to test
        
        Returns:
            DataFrame with weights for each variation point
        """
        results = []
        
        # Get base covariance matrix
        base_cov = self.optimizer._cov_matrix.values
        
        # Generate variation points
        variations = np.linspace(
            -variation_range,
            variation_range,
            num_points,
        )
        
        for var in variations:
            # Adjust covariance (scale all elements)
            adjusted_cov = base_cov * (1.0 + var)
            
            # Temporarily modify optimizer's covariance
            original_cov = self.optimizer._cov_matrix.copy()
            self.optimizer._cov_matrix = pd.DataFrame(
                adjusted_cov,
                index=self.optimizer.tickers,
                columns=self.optimizer.tickers,
            )
            
            try:
                # Re-optimize with constraints
                result = self.optimizer.optimize(constraints=constraints)
                
                # Store results
                row = {"variation": var}
                for i, ticker in enumerate(self.tickers):
                    row[ticker] = float(result.weights[i])
                results.append(row)
            except Exception as e:
                logger.warning(
                    f"Sensitivity analysis failed for variation {var}: {e}"
                )
            finally:
                # Restore original covariance
                self.optimizer._cov_matrix = original_cov
        
        return pd.DataFrame(results)
    
    def analyze_constraint_sensitivity(
        self,
        constraint_name: str,
        constraint_range: tuple,
        num_points: int = 10,
    ) -> pd.DataFrame:
        """
        Analyze sensitivity to constraint changes.
        
        Args:
            constraint_name: Name of constraint to vary
                           (e.g., "max_weight", "max_volatility")
            constraint_range: (min_value, max_value) range
            num_points: Number of points to test
        
        Returns:
            DataFrame with weights for each constraint value
        """
        results = []
        
        min_val, max_val = constraint_range
        constraint_values = np.linspace(min_val, max_val, num_points)
        
        for val in constraint_values:
            # Build constraints with varied value
            constraints = {constraint_name: val}
            
            try:
                # Re-optimize
                result = self.optimizer.optimize(constraints=constraints)
                
                # Store results
                row = {constraint_name: val}
                for i, ticker in enumerate(self.tickers):
                    row[ticker] = float(result.weights[i])
                results.append(row)
            except Exception as e:
                logger.warning(
                    f"Sensitivity analysis failed for {constraint_name}="
                    f"{val}: {e}"
                )
        
        return pd.DataFrame(results)
    
    def analyze_parameter_sensitivity(
        self,
        parameter_name: str,
        parameter_range: tuple,
        num_points: int = 10,
    ) -> pd.DataFrame:
        """
        Analyze sensitivity to method-specific parameters.
        
        Args:
            parameter_name: Name of parameter to vary
            parameter_range: (min_value, max_value) range
            num_points: Number of points to test
        
        Returns:
            DataFrame with weights for each parameter value
        """
        results = []
        
        min_val, max_val = parameter_range
        parameter_values = np.linspace(min_val, max_val, num_points)
        
        for val in parameter_values:
            try:
                # Call optimize with method-specific parameter
                # This requires optimizer to support parameter passing
                # For now, use a generic approach
                if hasattr(self.optimizer, "optimize"):
                    # Try to pass parameter if optimizer supports it
                    result = self.optimizer.optimize(**{parameter_name: val})
                else:
                    continue
                
                # Store results
                row = {parameter_name: val}
                for i, ticker in enumerate(self.tickers):
                    row[ticker] = float(result.weights[i])
                results.append(row)
            except Exception as e:
                logger.warning(
                    f"Sensitivity analysis failed for {parameter_name}="
                    f"{val}: {e}"
                )
        
        return pd.DataFrame(results)

