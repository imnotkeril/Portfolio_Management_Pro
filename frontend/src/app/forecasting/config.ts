/** Streamlit forecasting.py — method groups and defaults */

export const METHOD_TABS = [
  { id: "classical", label: "Classical" },
  { id: "ml", label: "Machine Learning" },
  { id: "dl", label: "Deep Learning" },
  { id: "simple", label: "Simple" },
] as const;

export const METHODS_BY_TAB: Record<string, string[]> = {
  classical: ["arima", "garch", "arima_garch"],
  ml: ["xgboost", "random_forest", "svm"],
  dl: ["lstm", "tcn", "ssa_maemd_tcn"],
  simple: ["prophet"],
};

export const METHOD_LABELS: Record<string, string> = {
  arima: "ARIMA",
  garch: "GARCH",
  arima_garch: "ARIMA-GARCH",
  xgboost: "XGBoost",
  random_forest: "Random Forest",
  svm: "SVM / SVR",
  lstm: "LSTM",
  tcn: "TCN",
  ssa_maemd_tcn: "SSA-MAEMD-TCN",
  prophet: "Prophet",
  ensemble: "Ensemble (weighted)",
};

export const MODEL_BLURBS: Record<string, string> = {
  arima: "Classical mean forecast — trends and short horizons; needs roughly stationary returns after differencing.",
  garch: "Volatility (variance) clustering — use for risk, not level prices alone.",
  arima_garch: "Mean via ARIMA + volatility via GARCH — joint return and uncertainty view.",
  xgboost: "Gradient boosted trees — strong nonlinear fit; optional technical features.",
  random_forest: "Bagged trees — robust baseline ML on tabular features.",
  svm: "Kernel regression — smaller samples and nonlinear boundaries.",
  lstm: "Recurrent network — sequence patterns; heavier compute.",
  tcn: "Temporal convolutional network — parallelizable deep sequence model.",
  ssa_maemd_tcn: "Hybrid decomposition + TCN — advanced pipeline; may need extra deps.",
  prophet: "Facebook Prophet — seasonality and holidays; quick baselines.",
  ensemble: "Weighted average of successful runs — weights favor lower validation MAPE.",
};

export const HORIZON_PRESETS: Record<string, number | null> = {
  "1 Day": 1,
  "1 Week (5 days)": 5,
  "2 Weeks (10 days)": 10,
  "1 Month (21 days)": 21,
  "3 Months (63 days)": 63,
  "6 Months (126 days)": 126,
  "1 Year (252 days)": 252,
  Custom: null,
};

export const TRAINING_WINDOW_RATIOS: Record<string, { ratio: number; description: string }> = {
  "30% (Recommended)": {
    ratio: 0.3,
    description: "Balance between freshness and stability — default in Streamlit.",
  },
  "50%": {
    ratio: 0.5,
    description: "More training data — calmer markets / longer memory.",
  },
  "60%": {
    ratio: 0.6,
    description: "Maximum training share — conservative, stable assets.",
  },
};

export function defaultParamsForMethod(method: string): Record<string, unknown> {
  switch (method) {
    case "arima":
      return { auto: true };
    case "garch":
      return { p: 1, q: 1 };
    case "arima_garch":
      return { auto_arima: true, garch_p: 1, garch_q: 1 };
    case "xgboost":
      return {
        n_estimators: 100,
        max_depth: 6,
        learning_rate: 0.1,
        use_technical_features: true,
      };
    case "random_forest":
      return { n_estimators: 100, max_depth: 10 };
    case "svm":
      return { C: 1.0, epsilon: 0.01, kernel: "rbf" };
    case "prophet":
      return { growth: "linear", seasonality: true, holidays: false };
    default:
      return {};
  }
}

export const CHART_PALETTE = [
  "#bf9ffb",
  "#7dc4e4",
  "#74f174",
  "#ffd066",
  "#faa1a4",
  "#a78bfa",
  "#22d3ee",
  "#f472b6",
  "#34d399",
  "#fb923c",
];
