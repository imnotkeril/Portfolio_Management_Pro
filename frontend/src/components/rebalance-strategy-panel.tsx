"use client";

import { useMemo } from "react";
import {
  REBALANCE_INTERVAL_OPTIONS,
  rebalanceIntervalLabel,
} from "@/lib/rebalance";
import type { Portfolio } from "@/lib/types";

function Tip({ text }: { text: string }) {
  return (
    <span className="tooltip-wrap">
      <span className="tooltip-icon">?</span>
      <span className="tooltip-bubble">{text}</span>
    </span>
  );
}

export type StrategyWeightRow = {
  ticker: string;
  weightPct: number;
};

type Props = {
  portfolio: Portfolio;
  intervalValue: string;
  onIntervalChange: (value: string) => void;
  weightRows: StrategyWeightRow[];
  onWeightChange: (ticker: string, weightPct: number) => void;
  onSave?: () => void;
  onPreview?: () => void;
  saving?: boolean;
  previewing?: boolean;
  showSaveButton?: boolean;
};

export function RebalanceStrategyPanel({
  portfolio,
  intervalValue,
  onIntervalChange,
  weightRows,
  onWeightChange,
  onSave,
  onPreview,
  saving,
  previewing,
  showSaveButton,
}: Props) {
  const totalPct = useMemo(
    () => weightRows.reduce((s, r) => s + (Number.isFinite(r.weightPct) ? r.weightPct : 0), 0),
    [weightRows],
  );
  const weightsOk = Math.abs(totalPct - 100) < 0.05;

  const sortedRows = useMemo(
    () =>
      [...weightRows].sort((a, b) => {
        if (a.ticker === "CASH") return 1;
        if (b.ticker === "CASH") return -1;
        return a.ticker.localeCompare(b.ticker);
      }),
    [weightRows],
  );

  return (
    <div className="panel p-5 space-y-4">
      <div>
        <h3 className="text-base font-semibold text-white">
          Target-weight rebalancing
        </h3>
        <p className="text-sm text-white/50 mt-1">
          Splits, dividends, and rebalancing run automatically from your first
          transaction through today whenever the portfolio is loaded. BUY/SELL
          rows appear on the Transactions tab on each scheduled date.
        </p>
      </div>

      <div className="rounded-lg border border-white/10 bg-white/5 px-4 py-3 text-sm text-white/60">
        {portfolio.rebalance_interval_months ? (
          <>
            Active schedule:{" "}
            <span className="text-white">
              {rebalanceIntervalLabel(portfolio.rebalance_interval_months)}
            </span>
            . First rebalance is{" "}
            {portfolio.rebalance_interval_months} month(s) after inception, then
            every {portfolio.rebalance_interval_months} months.
          </>
        ) : (
          <>Rebalancing is off. Choose a frequency and save to enable it.</>
        )}
      </div>

      <div>
        <label className="label">
          Rebalance frequency
          <Tip text="How often holdings are brought back to target weights (automatic)" />
        </label>
        <select
          className="input"
          value={intervalValue}
          onChange={(e) => onIntervalChange(e.target.value)}
        >
          {REBALANCE_INTERVAL_OPTIONS.map((o) => (
            <option key={o.value || "off"} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      </div>

      {sortedRows.length > 0 ? (
        <div>
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-sm font-medium text-white">Target weights</h4>
            <span
              className={`text-xs ${weightsOk ? "text-emerald-400" : "text-amber-400"}`}
            >
              Total: {totalPct.toFixed(1)}%
              {!weightsOk ? " (must equal 100%)" : ""}
            </span>
          </div>
          <table className="data-table text-sm">
            <thead>
              <tr>
                <th>Ticker</th>
                <th>Target %</th>
              </tr>
            </thead>
            <tbody>
              {sortedRows.map((row) => (
                <tr key={row.ticker}>
                  <td className="font-mono">{row.ticker}</td>
                  <td>
                    <input
                      type="number"
                      className="input w-24"
                      min={0}
                      max={100}
                      step={0.1}
                      value={row.weightPct}
                      onChange={(e) =>
                        onWeightChange(row.ticker, Number(e.target.value) || 0)
                      }
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="text-sm text-white/50">
          Add positions or transactions first, then set target weights here.
        </p>
      )}

      <div className="flex flex-wrap gap-2">
        {showSaveButton && onSave ? (
          <button
            type="button"
            className="btn btn-primary"
            disabled={saving || !weightsOk || sortedRows.length === 0}
            onClick={onSave}
          >
            {saving ? "Saving…" : "Save strategy"}
          </button>
        ) : null}
        {onPreview ? (
          <button
            type="button"
            className="btn btn-secondary"
            disabled={previewing || sortedRows.length === 0}
            onClick={onPreview}
          >
            {previewing ? "Previewing…" : "Preview rebalance"}
          </button>
        ) : null}
      </div>
    </div>
  );
}
