import { getAvailableObjectives, getMethodPreset } from "@/app/optimization/config";

export const COVARIANCE_METHOD_OPTIONS = [
  { value: "shrink", label: "Shrinkage (α)" },
  { value: "ledoit_wolf", label: "Ledoit–Wolf" },
  { value: "sample", label: "Sample covariance" },
] as const;

export type CovarianceMethodValue = (typeof COVARIANCE_METHOD_OPTIONS)[number]["value"];

export const HRP_LINKAGE_OPTIONS = [
  "average",
  "single",
  "complete",
  "weighted",
  "centroid",
  "median",
  "ward",
] as const;

export type HrpLinkageValue = (typeof HRP_LINKAGE_OPTIONS)[number];

/** Editable tuning for POST /optimization/full — aligned with engine optimize() kwargs. */
export type OptiMethodTuning = {
  covarianceMethod: CovarianceMethodValue;
  shrinkageAlpha: number;
  confidenceLevel: number;
  uncertaintyRadiusReturns: number;
  uncertaintyRadiusCov: number;
  blackLittermanTau: number;
  hrpLinkageMethod: HrpLinkageValue;
  meanCvarOptimizationMode: "cvar_cap" | "penalty";
  meanCvarCapRelax: number;
  meanCvarRiskAversion: number;
  robustKappa: string;
  robustLambda: string;
  useRobustAdvanced: boolean;
  /** Mean-variance: use inequality return floor when optimizing with target return (frontier sweeps). */
  targetReturnAsFloor: boolean;
  frontierNPoints: number;
};

const METHODS_WITH_COVARIANCE = new Set([
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
]);

export function methodUsesCovarianceParams(method: string): boolean {
  return METHODS_WITH_COVARIANCE.has(method);
}

export function methodSupportsDiversificationLambda(method: string): boolean {
  return (
    method === "mean_variance" ||
    method === "risk_parity" ||
    method === "max_diversification" ||
    method === "min_correlation"
  );
}

/** Defaults when the user picks a method (or resets the panel). */
export function getInitialOptiMethodTuning(method: string): OptiMethodTuning {
  const preset = getMethodPreset(method);
  return {
    covarianceMethod: "shrink",
    shrinkageAlpha: 0.25,
    confidenceLevel: preset.cvarConfidence ?? 0.95,
    uncertaintyRadiusReturns: preset.robustUr ?? 0.1,
    uncertaintyRadiusCov: preset.robustUc ?? 0.1,
    blackLittermanTau: 0.05,
    hrpLinkageMethod: "average",
    meanCvarOptimizationMode: "cvar_cap",
    meanCvarCapRelax: 1.08,
    meanCvarRiskAversion: 1.0,
    robustKappa: "",
    robustLambda: "",
    useRobustAdvanced: false,
    targetReturnAsFloor: false,
    frontierNPoints: 150,
  };
}

export function buildMethodParamsFromTuning(
  method: string,
  tuning: OptiMethodTuning,
  objective: string | null,
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  const availObj = getAvailableObjectives(method);
  if (availObj.length && objective) {
    out.objective = objective;
  }

  if (methodUsesCovarianceParams(method)) {
    out.covariance_method = tuning.covarianceMethod;
    out.shrinkage_alpha = tuning.shrinkageAlpha;
  }

  if (method === "black_litterman") {
    out.tau = tuning.blackLittermanTau;
  }
  if (method === "hrp") {
    out.linkage_method = tuning.hrpLinkageMethod;
  }
  if (method === "cvar_optimization") {
    out.confidence_level = tuning.confidenceLevel;
  }
  if (method === "mean_cvar") {
    out.confidence_level = tuning.confidenceLevel;
    out.optimization_mode = tuning.meanCvarOptimizationMode;
    out.cvar_cap_relax = tuning.meanCvarCapRelax;
    out.risk_aversion = tuning.meanCvarRiskAversion;
  }
  if (method === "robust") {
    out.uncertainty_radius_returns = tuning.uncertaintyRadiusReturns;
    out.uncertainty_radius_cov = tuning.uncertaintyRadiusCov;
    if (tuning.useRobustAdvanced) {
      const k = Number(tuning.robustKappa);
      const l = Number(tuning.robustLambda);
      if (Number.isFinite(k)) out.robust_kappa = k;
      if (Number.isFinite(l)) out.robust_lambda = l;
    }
  }
  if (method === "mean_variance" && tuning.targetReturnAsFloor) {
    out.target_return_as_floor = true;
  }

  return out;
}
