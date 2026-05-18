"use client";

import { FormEvent, useCallback, useEffect, useState } from "react";

import { defaultFeesForTransaction } from "@/lib/ib-fees";
import { fetchTickerPrice } from "@/lib/ticker-price-api";

function Tip({ text }: { text: string }) {
  return (
    <span className="tooltip-wrap">
      <span className="tooltip-icon">?</span>
      <span className="tooltip-bubble">{text}</span>
    </span>
  );
}

export type TransactionFormPayload = {
  transaction_date: string;
  transaction_type: string;
  ticker: string;
  shares: number;
  price: number;
  fees: number;
  notes: string | null;
  reinvest?: boolean;
  split_ratio?: number;
};

type Props = {
  onSubmit: (payload: TransactionFormPayload) => Promise<void>;
  disabled?: boolean;
  defaultDate?: string;
};

export function TransactionForm({
  onSubmit,
  disabled,
  defaultDate,
}: Props) {
  const [txDate, setTxDate] = useState(
    defaultDate ?? new Date().toISOString().slice(0, 10),
  );
  const [txType, setTxType] = useState("BUY");
  const [txTicker, setTxTicker] = useState("");
  const [cashAmount, setCashAmount] = useState(0);
  const [txShares, setTxShares] = useState(1);
  const [txPrice, setTxPrice] = useState(0);
  const [txFees, setTxFees] = useState(0);
  const [txNotes, setTxNotes] = useState("");
  const [reinvest, setReinvest] = useState(false);
  const [splitRatio, setSplitRatio] = useState(2);
  const [priceLoading, setPriceLoading] = useState(false);
  const [priceHint, setPriceHint] = useState<string | null>(null);
  const [feesTouched, setFeesTouched] = useState(false);
  const [saving, setSaving] = useState(false);

  const isCashFlow = txType === "DEPOSIT" || txType === "WITHDRAWAL";

  const applyDefaultFees = useCallback(
    (shares: number, price: number, type: string) => {
      if (!feesTouched) {
        setTxFees(defaultFeesForTransaction(type, shares, price));
      }
    },
    [feesTouched],
  );

  const loadHistoricalPrice = useCallback(async () => {
    if (isCashFlow || !txTicker.trim() || !txDate) return;
    setPriceLoading(true);
    setPriceHint(null);
    try {
      const res = await fetchTickerPrice(txTicker, txDate);
      if (res.valid && res.price != null && res.price > 0) {
        setTxPrice(res.price);
        applyDefaultFees(txShares, res.price, txType);
        const dateNote =
          res.price_date && res.requested_date && res.price_date !== res.requested_date
            ? ` (nearest to ${res.requested_date})`
            : "";
        setPriceHint(
          res.price_date
            ? `Close on ${res.price_date}${dateNote}: $${res.price.toFixed(2)}`
            : `$${res.price.toFixed(2)}`,
        );
      } else {
        setPriceHint("No price for this date — enter manually");
      }
    } catch {
      setPriceHint("Could not load price");
    } finally {
      setPriceLoading(false);
    }
  }, [isCashFlow, txTicker, txDate, txShares, txType, applyDefaultFees]);

  useEffect(() => {
    if (isCashFlow) {
      setTxFees(0);
      return;
    }
    const t = window.setTimeout(() => {
      void loadHistoricalPrice();
    }, 400);
    return () => window.clearTimeout(t);
  }, [txTicker, txDate, txType, isCashFlow, loadHistoricalPrice]);

  useEffect(() => {
    if (isCashFlow) return;
    applyDefaultFees(txShares, txPrice, txType);
  }, [txShares, txPrice, txType, isCashFlow, applyDefaultFees]);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setSaving(true);
    try {
      if (isCashFlow) {
        if (cashAmount <= 0) return;
        await onSubmit({
          transaction_date: txDate,
          transaction_type: txType,
          ticker: "CASH",
          shares: cashAmount,
          price: 1,
          fees: 0,
          notes: txNotes || null,
        });
      } else {
        if (!txTicker.trim() || txShares <= 0 || txPrice <= 0) return;
        const payload: TransactionFormPayload = {
          transaction_date: txDate,
          transaction_type: txType,
          ticker: txTicker.trim().toUpperCase(),
          shares: Math.floor(txShares),
          price: txPrice,
          fees: txFees,
          notes: txNotes || null,
        };
        if (txType === "DIVIDEND") payload.reinvest = reinvest;
        if (txType === "SPLIT") payload.split_ratio = splitRatio;
        await onSubmit(payload);
      }
      setTxTicker("");
      setCashAmount(0);
      setTxShares(1);
      setTxPrice(0);
      setTxFees(0);
      setTxNotes("");
      setFeesTouched(false);
      setPriceHint(null);
    } finally {
      setSaving(false);
    }
  };

  const canSubmit = isCashFlow
    ? cashAmount > 0
    : txTicker.trim().length > 0 && txShares > 0 && txPrice > 0;

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <label className="label">
            Date
            <Tip text="Used to load historical close for BUY/SELL" />
          </label>
          <input
            className="input"
            type="date"
            value={txDate}
            max={new Date().toISOString().slice(0, 10)}
            onChange={(e) => setTxDate(e.target.value)}
          />
        </div>
        <div>
          <label className="label">Type</label>
          <select
            className="input"
            value={txType}
            onChange={(e) => {
              setTxType(e.target.value);
              setFeesTouched(false);
            }}
          >
            <option value="BUY">BUY</option>
            <option value="SELL">SELL</option>
            <option value="DEPOSIT">DEPOSIT</option>
            <option value="WITHDRAWAL">WITHDRAWAL</option>
            <option value="DIVIDEND">DIVIDEND</option>
            <option value="SPLIT">SPLIT</option>
          </select>
        </div>
        {isCashFlow ? (
          <div className="md:col-span-1">
            <label className="label">
              Cash amount (USD)
              <Tip text="Deposit or withdraw cash in dollars" />
            </label>
            <input
              className="input"
              type="number"
              min={0}
              step={0.01}
              value={cashAmount || ""}
              onChange={(e) => setCashAmount(Number(e.target.value))}
            />
          </div>
        ) : (
          <>
            <div>
              <label className="label">Ticker</label>
              <input
                className="input"
                placeholder="AAPL"
                value={txTicker}
                onChange={(e) => setTxTicker(e.target.value.toUpperCase())}
              />
            </div>
            <div>
              <label className="label">Shares (whole)</label>
              <input
                className="input"
                type="number"
                min={1}
                step={1}
                value={txShares}
                onChange={(e) =>
                  setTxShares(Math.max(1, Math.floor(Number(e.target.value))))
                }
              />
            </div>
            <div>
              <label className="label">
                Price per share
                {priceLoading ? (
                  <span className="text-white/40 text-xs ml-2">Loading…</span>
                ) : null}
              </label>
              <input
                className="input"
                type="number"
                min={0}
                step={0.01}
                value={txPrice}
                onChange={(e) => {
                  setTxPrice(Number(e.target.value));
                  setFeesTouched(false);
                }}
              />
              {priceHint ? (
                <p className="text-xs text-white/40 mt-1">{priceHint}</p>
              ) : null}
            </div>
            <div>
              <label className="label">
                Fees (IB estimate)
                <Tip text="$0.005/share, min $1, max 1% — editable" />
              </label>
              <input
                className="input"
                type="number"
                min={0}
                step={0.01}
                value={txFees}
                onChange={(e) => {
                  setTxFees(Number(e.target.value));
                  setFeesTouched(true);
                }}
              />
            </div>
          </>
        )}
      </div>

      {txType === "DIVIDEND" && !isCashFlow ? (
        <label className="flex items-center gap-2 text-sm text-white/70">
          <input
            type="checkbox"
            checked={reinvest}
            onChange={(e) => setReinvest(e.target.checked)}
          />
          Reinvest dividend
        </label>
      ) : null}

      {txType === "SPLIT" && !isCashFlow ? (
        <div>
          <label className="label">Split ratio</label>
          <input
            className="input"
            type="number"
            min={0.01}
            step={0.01}
            value={splitRatio}
            onChange={(e) => setSplitRatio(Number(e.target.value))}
          />
        </div>
      ) : null}

      <div>
        <label className="label">Notes (optional)</label>
        <input
          className="input"
          placeholder="Optional notes..."
          value={txNotes}
          onChange={(e) => setTxNotes(e.target.value)}
        />
      </div>

      <button
        type="submit"
        className="btn btn-primary"
        disabled={disabled || saving || !canSubmit}
      >
        {saving ? "Adding…" : "Add Transaction"}
      </button>
    </form>
  );
}
