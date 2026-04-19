/** Mirrors streamlit_app/pages/portfolio_optimization.py (methods & objectives). */

export const METHOD_NAMES: Record<string, string> = {
  mean_variance: "Mean-Variance (Markowitz)",
  black_litterman: "Black-Litterman",
  risk_parity: "Risk Parity",
  hrp: "Hierarchical Risk Parity (HRP)",
  cvar_optimization: "CVaR Optimization",
  mean_cvar: "Mean-CVaR",
  robust: "Robust Optimization",
  max_diversification: "Maximum Diversification",
  min_correlation: "Minimum Correlation",
  inverse_correlation: "Inverse Correlation Weighting",
};

export const AVAILABLE_OPTIMIZATION_METHODS = [
  "mean_variance",
  "black_litterman",
  "risk_parity",
  "hrp",
  "cvar_optimization",
  "mean_cvar",
  "robust",
  "max_diversification",
  "min_correlation",
  "inverse_correlation",
] as const;

export type OptimizationMethodKey = (typeof AVAILABLE_OPTIMIZATION_METHODS)[number];

export const OBJECTIVE_METHOD_MAPPING: Record<
  string,
  { methods: string[]; display: string; default_for: string[] }
> = {
  maximize_sharpe: {
    methods: ["mean_variance", "black_litterman", "robust"],
    display: "Maximize Sharpe Ratio",
    default_for: ["mean_variance", "black_litterman", "robust"],
  },
  minimize_volatility: {
    methods: ["mean_variance", "black_litterman"],
    display: "Minimize Volatility",
    default_for: [],
  },
  maximize_return: {
    methods: ["mean_variance", "black_litterman"],
    display: "Maximize Expected Return",
    default_for: [],
  },
  minimize_cvar: {
    methods: ["cvar_optimization", "mean_cvar"],
    display: "Minimize CVaR / Expected Shortfall",
    default_for: ["cvar_optimization"],
  },
};

export const FIXED_OBJECTIVE_METHODS = new Set([
  "risk_parity",
  "hrp",
  "max_diversification",
  "min_correlation",
  "inverse_correlation",
  "cvar_optimization",
  "mean_cvar",
  "min_variance",
  "max_return",
  "max_sharpe",
  "equal_weight",
  "market_cap",
  "kelly_criterion",
  "min_tracking_error",
  "max_alpha",
]);

export const METHODS_NEEDING_BENCHMARK = new Set(["min_tracking_error", "max_alpha"]);

export const BENCHMARK_CHART_PRESETS = [
  "None",
  "AOR",
  "SPY",
  "QQQ",
  "VTI",
  "DIA",
  "IWM",
] as const;

export const TRAINING_WINDOW_OPTIONS: Record<string, { ratio: number; description: string }> = {
  "30% (Recommended)": {
    ratio: 0.3,
    description:
      "Balance between data freshness and statistical reliability. Suitable for most cases.",
  },
  "50%": {
    ratio: 0.5,
    description: "More data for training. Suitable for stable markets and long-term strategies.",
  },
  "60%": {
    ratio: 0.6,
    description:
      "Maximum statistical reliability. Use for stable assets and conservative portfolios.",
  },
};

export function getAvailableObjectives(method: string): string[] {
  if (FIXED_OBJECTIVE_METHODS.has(method)) return [];
  const out: string[] = [];
  for (const [key, data] of Object.entries(OBJECTIVE_METHOD_MAPPING)) {
    if (data.methods.includes(method)) out.push(key);
  }
  return out;
}

export function getDefaultObjective(method: string): string | null {
  if (FIXED_OBJECTIVE_METHODS.has(method)) return null;
  for (const [key, data] of Object.entries(OBJECTIVE_METHOD_MAPPING)) {
    if (data.default_for.includes(method)) return key;
  }
  const avail = getAvailableObjectives(method);
  return avail[0] ?? null;
}

export function validateWeightConstraints(
  minWeight: number | undefined,
  maxWeight: number | undefined,
  numAssets: number,
): string[] {
  const warnings: string[] = [];
  if (minWeight != null && minWeight * numAssets > 1.0) {
    const maxAllowed = 1.0 / numAssets;
    warnings.push(
      `Minimum weight (${(minWeight * 100).toFixed(1)}%) is too high for ${numAssets} assets. Recommended: min ≤ ${(maxAllowed * 100).toFixed(2)}%`,
    );
  }
  if (maxWeight != null && minWeight != null && maxWeight < minWeight) {
    warnings.push(
      `Maximum weight (${(maxWeight * 100).toFixed(1)}%) is less than minimum (${(minWeight * 100).toFixed(1)}%)`,
    );
  }
  return warnings;
}

/** Short + long description for expander (subset of Streamlit). */
export const METHOD_DESCRIPTIONS: Record<string, { short: string; long: string }> = {
  mean_variance: {
    short: "Markowitz mean-variance optimization",
    long: `Classic Nobel-prize framework: maximize expected return for a given risk level using historical returns and covariance. Best for strategic allocation when inputs are stable; sensitive to estimation error and may produce extreme weights without constraints.`,
  },
  black_litterman: {
    short: "Market equilibrium blended with investor views",
    long: `Combines CAPM-implied equilibrium returns with optional views (Bayesian blend). Typically more stable weights than pure Markowitz. Good when you want to mix data with judgment.`,
  },
  risk_parity: {
    short: "Equal risk contribution from each asset",
    long: `Weights assets so each contributes similarly to portfolio risk (not equal dollars). Often robust when return forecasts are uncertain.`,
  },
  hrp: {
    short: "Hierarchical clustering-based risk parity",
    long: `Uses clustering on the correlation structure; robust to covariance estimation error. Heavier compute; good for many assets.`,
  },
  cvar_optimization: {
    short: "Minimize tail risk (CVaR)",
    long: `Focuses on expected shortfall beyond VaR rather than variance. Suitable for fat tails and risk-averse mandates.`,
  },
  mean_cvar: {
    short: "Maximize return per unit of CVaR",
    long: `Balances expected return against tail risk (Return/CVaR).`,
  },
  robust: {
    short: "Robust optimization under parameter uncertainty",
    long: `Optimizes over uncertainty sets for returns/covariance; more conservative, less sensitive to point estimates.`,
  },
  max_diversification: {
    short: "Maximize diversification ratio",
    long: `Maximizes portfolio diversification benefit relative to weighted average asset risk; return-agnostic.`,
  },
  min_correlation: {
    short: "Minimize average pairwise correlation",
    long: `Pushes toward assets that move more independently; simple correlation-focused construction.`,
  },
  inverse_correlation: {
    short: "Analytical inverse-correlation weights",
    long: `Fast analytical weights from correlation structure; good for quick experiments.`,
  },
};

export type MethodPreset = {
  longOnly: boolean;
  maxCashPct: number;
  useMinW: boolean;
  minWPct: number;
  useMaxW: boolean;
  maxWPct: number;
  useMinReturn: boolean;
  minReturnPct: number;
  useDivLambda: boolean;
  divLambda: number;
  useOOS: boolean;
  trainingWindowLabel: string;
  includeFrontier: boolean;
  includeSensitivity: boolean;
  sensitivityType: "returns" | "covariance";
  objective?: string;
  cvarConfidence?: number;
  robustUr?: number;
  robustUc?: number;
  benchmarkOpt?: string;
};

const BASE_PRESET: MethodPreset = {
  longOnly: true,
  maxCashPct: 10,
  useMinW: true,
  minWPct: 1,
  useMaxW: true,
  maxWPct: 30,
  useMinReturn: false,
  minReturnPct: 3,
  useDivLambda: false,
  divLambda: 1,
  useOOS: true,
  trainingWindowLabel: "30% (Recommended)",
  includeFrontier: true,
  includeSensitivity: false,
  sensitivityType: "returns",
};

export const METHOD_PRESETS: Record<string, MethodPreset> = {
  mean_variance: {
    ...BASE_PRESET,
    objective: "maximize_sharpe",
  },
  black_litterman: {
    ...BASE_PRESET,
    objective: "maximize_sharpe",
    useMinW: true,
    minWPct: 2,
    useMaxW: true,
    maxWPct: 25,
  },
  risk_parity: {
    ...BASE_PRESET,
    useMinW: true,
    minWPct: 1,
    useMaxW: true,
    maxWPct: 35,
    useDivLambda: true,
    divLambda: 0.5,
  },
  hrp: {
    ...BASE_PRESET,
    useMinW: false,
    minWPct: 0,
    useMaxW: true,
    maxWPct: 40,
  },
  cvar_optimization: {
    ...BASE_PRESET,
    cvarConfidence: 0.95,
    useMinW: true,
    minWPct: 1,
    useMaxW: true,
    maxWPct: 25,
  },
  mean_cvar: {
    ...BASE_PRESET,
    cvarConfidence: 0.95,
    useMinW: true,
    minWPct: 1,
    useMaxW: true,
    maxWPct: 25,
  },
  robust: {
    ...BASE_PRESET,
    objective: "maximize_sharpe",
    robustUr: 0.1,
    robustUc: 0.1,
    useMinW: true,
    minWPct: 1,
    useMaxW: true,
    maxWPct: 30,
  },
  max_diversification: {
    ...BASE_PRESET,
    useMinW: true,
    minWPct: 1,
    useMaxW: true,
    maxWPct: 35,
    useDivLambda: true,
    divLambda: 0.5,
  },
  min_correlation: {
    ...BASE_PRESET,
    useMinW: true,
    minWPct: 1,
    useMaxW: true,
    maxWPct: 35,
    useDivLambda: true,
    divLambda: 0.5,
  },
  inverse_correlation: {
    ...BASE_PRESET,
    useMinW: false,
    minWPct: 0,
    useMaxW: true,
    maxWPct: 35,
  },
};

export function getMethodPreset(method: string): MethodPreset {
  return METHOD_PRESETS[method] ?? BASE_PRESET;
}
