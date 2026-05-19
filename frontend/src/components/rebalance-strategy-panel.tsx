"use client";

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

type Props = {
  portfolio: Portfolio;
  intervalValue: string;
  onIntervalChange: (value: string) => void;
  onSave?: () => void;
  saving?: boolean;
  showSaveButton?: boolean;
};

export function RebalanceStrategyPanel({
  portfolio,
  intervalValue,
  onIntervalChange,
  onSave,
  saving,
  showSaveButton,
}: Props) {
  const targets = portfolio.positions
    .filter((p) => p.weight_target != null && p.weight_target > 0)
    .sort((a, b) => {
      if (a.ticker === "CASH") return 1;
      if (b.ticker === "CASH") return -1;
      return a.ticker.localeCompare(b.ticker);
    });

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

      {targets.length > 0 ? (
        <div>
          <h4 className="text-sm font-medium text-white mb-2">Target weights</h4>
          <table className="data-table text-sm">
            <thead>
              <tr>
                <th>Ticker</th>
                <th>Target</th>
              </tr>
            </thead>
            <tbody>
              {targets.map((p) => (
                <tr key={p.ticker}>
                  <td className="font-mono">{p.ticker}</td>
                  <td>{((p.weight_target ?? 0) * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="text-sm text-white/50">
          Target weights will be inferred from your initial BUY transactions when
          the ledger is synced.
        </p>
      )}

      {showSaveButton && onSave ? (
        <button
          type="button"
          className="btn btn-primary"
          disabled={saving}
          onClick={onSave}
        >
          {saving ? "Saving…" : "Save strategy"}
        </button>
      ) : null}
    </div>
  );
}
