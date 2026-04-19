/**
 * Opti notebook series (01–10) mapped to API method keys.
 * Order matches AVAILABLE_OPTIMIZATION_METHODS and the Jupyter folder naming.
 */

import {
  AVAILABLE_OPTIMIZATION_METHODS,
  METHOD_NAMES,
  type OptimizationMethodKey,
} from "../optimization/config";

const NOTEBOOK_FILE: Record<OptimizationMethodKey, string> = {
  mean_variance: "01_Portfolio_Optimization_Markowitz.ipynb",
  black_litterman: "02_Portfolio_Optimization_Black-Litterman.ipynb",
  risk_parity: "03_Portfolio_Optimization_Risk_Parity.ipynb",
  hrp: "04_Portfolio_Optimization_HRP.ipynb",
  cvar_optimization: "05_Portfolio_Optimization_CVaR.ipynb",
  mean_cvar: "06_Portfolio_Optimization_Mean_CVaR.ipynb",
  robust: "07_Portfolio_Optimization_Robust.ipynb",
  max_diversification: "08_Portfolio_Optimization_Maximum_Diversification.ipynb",
  min_correlation: "09_Portfolio_Optimization_Minimum_Correlation.ipynb",
  inverse_correlation: "10_Portfolio_Optimization_Inverse_Correlation.ipynb",
};

export type NotebookMethodEntry = {
  /** "01" … "10" */
  seriesId: string;
  method: OptimizationMethodKey;
  displayName: string;
  notebookFile: string;
};

export const NOTEBOOK_METHOD_SERIES: NotebookMethodEntry[] =
  AVAILABLE_OPTIMIZATION_METHODS.map((method, i) => ({
    seriesId: String(i + 1).padStart(2, "0"),
    method,
    displayName: METHOD_NAMES[method] ?? method,
    notebookFile: NOTEBOOK_FILE[method],
  }));
