"use client";

import { useState } from "react";

import { useAuth } from "@/contexts/AuthContext";

export default function SettingsPage() {
  const { user, loading, logout, deleteAccount } = useAuth();
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

  async function onDelete(event: React.FormEvent) {
    event.preventDefault();
    if (
      !window.confirm(
        "Delete your account permanently? Your portfolios and subscription will be removed. This cannot be undone.",
      )
    ) {
      return;
    }
    setDeleting(true);
    setError(null);
    try {
      await deleteAccount(password);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not delete the account");
      setDeleting(false);
    }
  }

  if (loading) {
    return <div className="text-white/60">Loading…</div>;
  }

  return (
    <div className="mx-auto max-w-lg space-y-6">
      <h1 className="text-3xl font-bold text-white">Settings</h1>
      <div className="panel p-6 space-y-4">
        <div>
          <p className="text-xs text-white/40">Email</p>
          <p className="text-white mt-1">{user?.email ?? "—"}</p>
        </div>
        <a href="/billing" className="btn btn-secondary inline-block text-center">
          Billing & plan
        </a>
        <button type="button" className="btn btn-secondary" onClick={() => logout()}>
          Sign out
        </button>
      </div>

      <form
        onSubmit={onDelete}
        className="panel p-6 space-y-3"
        aria-labelledby="delete-account-title"
      >
        <h2 id="delete-account-title" className="text-lg font-semibold text-white">
          Delete account
        </h2>
        <p className="text-sm text-white/60">
          Permanently removes your account, portfolios and transactions, and
          cancels your subscription. This cannot be undone.
        </p>
        <label htmlFor="delete-password" className="block text-sm text-white/70">
          Confirm with your password
        </label>
        <input
          id="delete-password"
          type="password"
          autoComplete="current-password"
          className="input w-full"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
        {error ? (
          <p role="alert" className="text-sm text-red-400">
            {error}
          </p>
        ) : null}
        <button
          type="submit"
          className="btn btn-secondary min-h-[44px]"
          disabled={deleting || password.length === 0}
        >
          {deleting ? "Deleting…" : "Delete my account"}
        </button>
      </form>
    </div>
  );
}
