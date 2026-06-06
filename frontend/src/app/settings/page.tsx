"use client";

import { useAuth } from "@/contexts/AuthContext";

export default function SettingsPage() {
  const { user, loading, logout } = useAuth();

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
    </div>
  );
}
