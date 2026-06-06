"use client";

import { Suspense, useCallback, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";

import { api } from "@/lib/api";
import { useAuth } from "@/contexts/AuthContext";

type BillingStatus = {
  plan: string;
  status: string;
  is_pro: boolean;
  current_period_end: string | null;
  stripe_configured: boolean;
  free_portfolio_limit: number;
};

function BillingContent() {
  const { user, loading: authLoading } = useAuth();
  const searchParams = useSearchParams();
  const [status, setStatus] = useState<BillingStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const loadStatus = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.get<BillingStatus>("/billing/status");
      setStatus(data);
      setMessage(null);
    } catch (err) {
      setMessage(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (user) void loadStatus();
  }, [user, loadStatus]);

  useEffect(() => {
    if (searchParams.get("success") === "1") {
      setMessage("Payment received — refreshing your plan…");
      void loadStatus();
    }
    if (searchParams.get("canceled") === "1") {
      setMessage("Checkout canceled.");
    }
  }, [searchParams, loadStatus]);

  const startCheckout = async () => {
    setBusy(true);
    setMessage(null);
    try {
      const { url } = await api.post<{ url: string }>("/billing/checkout", {});
      window.location.href = url;
    } catch (err) {
      setMessage(err instanceof Error ? err.message : String(err));
      setBusy(false);
    }
  };

  const openPortal = async () => {
    setBusy(true);
    setMessage(null);
    try {
      const { url } = await api.post<{ url: string }>("/billing/portal", {});
      window.location.href = url;
    } catch (err) {
      setMessage(err instanceof Error ? err.message : String(err));
      setBusy(false);
    }
  };

  if (authLoading || loading) {
    return <div className="text-white/60">Loading…</div>;
  }

  const isPro = status?.is_pro ?? false;
  const planLabel = isPro ? "Pro" : "Free";

  return (
    <div className="mx-auto max-w-lg space-y-6">
      <h1 className="text-3xl font-bold text-white">Billing</h1>

      <div className="panel p-6 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-xs text-white/40">Current plan</p>
            <p className="text-2xl font-semibold text-white mt-1">{planLabel}</p>
            {status?.status ? (
              <p className="text-sm text-white/50 mt-1">Status: {status.status}</p>
            ) : null}
          </div>
          <span
            className={`rounded-full px-3 py-1 text-xs font-medium ${
              isPro
                ? "bg-violet-500/20 text-violet-200 border border-violet-400/30"
                : "bg-white/10 text-white/70 border border-white/10"
            }`}
          >
            {planLabel}
          </span>
        </div>

        {!isPro ? (
          <ul className="text-sm text-white/60 space-y-1 list-disc list-inside">
            <li>{status?.free_portfolio_limit ?? 1} portfolio on Free</li>
            <li>Optimization and Forecasting require Pro</li>
          </ul>
        ) : (
          <p className="text-sm text-white/60">
            Unlimited portfolios, optimization, and forecasting.
            {status?.current_period_end
              ? ` Renews through ${status.current_period_end.slice(0, 10)}.`
              : null}
          </p>
        )}

        {!status?.stripe_configured ? (
          <p className="text-sm text-amber-400/90">
            Stripe is not configured on the server. Set STRIPE_SECRET_KEY and
            STRIPE_PRICE_ID_PRO to enable checkout.
          </p>
        ) : null}

        <div className="flex flex-wrap gap-2 pt-2">
          {!isPro && status?.stripe_configured ? (
            <button
              type="button"
              className="btn btn-primary"
              disabled={busy}
              onClick={() => void startCheckout()}
            >
              {busy ? "Redirecting…" : "Upgrade to Pro"}
            </button>
          ) : null}
          {isPro && status?.stripe_configured ? (
            <button
              type="button"
              className="btn btn-secondary"
              disabled={busy}
              onClick={() => void openPortal()}
            >
              {busy ? "Redirecting…" : "Manage subscription"}
            </button>
          ) : null}
        </div>

        {message ? (
          <p className="text-sm text-white/70 border-t border-white/10 pt-3">
            {message}
          </p>
        ) : null}
      </div>

      <p className="text-xs text-white/40">
        Signed in as {user?.email}. Test locally with{" "}
        <code className="text-white/60">stripe listen --forward-to localhost:8000/stripe/webhook</code>
      </p>
    </div>
  );
}

export default function BillingPage() {
  return (
    <Suspense fallback={<div className="text-white/60">Loading…</div>}>
      <BillingContent />
    </Suspense>
  );
}
