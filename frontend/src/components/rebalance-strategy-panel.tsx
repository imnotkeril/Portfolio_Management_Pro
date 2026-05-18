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
  const targets = portfolio.positions.filter(
    (p) => p.ticker !== "CASH" && p.weight_target != null && p.weight_target > 0,
  );

  return (
    <div className="panel p-5 space-y-4">
      <div>
        <h3 className="text-base font-semibold text-white">
          Target-weight rebalancing
        </h3>
        <p className="text-sm text-white/50 mt-1">
          Targets come from each position&apos;s weight (%). On schedule, the
          portfolio should be traded back to those targets. Trade execution is
          planned for a later phase; the schedule is saved now.
        </p>
      </div>

      <div>
        <label className="label">
          Rebalance frequency
          <Tip text="How often to bring holdings back to target weights" />
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
        {!showSaveButton && (
          <p className="text-xs text-white/40 mt-1">
            Current: {rebalanceIntervalLabel(portfolio.rebalance_interval_months)}
          </p>
        )}
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
          No target weights on positions. Set weights when creating or editing
          positions.
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
