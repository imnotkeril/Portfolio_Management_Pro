"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { useAuth } from "@/contexts/AuthContext";

const publicItems = [
  { href: "/", label: "Home" },
  { href: "/login", label: "Sign in" },
  { href: "/register", label: "Register" },
];

const appItems = [
  { href: "/dashboard", label: "Dashboard" },
  { href: "/portfolios", label: "Portfolio List" },
  { href: "/create", label: "Create" },
  { href: "/analysis", label: "Analysis" },
  { href: "/optimization", label: "Optimization" },
  { href: "/risk", label: "Risk" },
  { href: "/forecasting", label: "Forecasting" },
  { href: "/billing", label: "Billing" },
  { href: "/settings", label: "Settings" },
];

export function MainNav() {
  const pathname = usePathname();
  const { user, loading, logout } = useAuth();

  const items = user ? appItems : publicItems;

  return (
    <header className="border-b border-white/10 bg-black/30 backdrop-blur-md">
      <div className="mx-auto flex w-full max-w-7xl items-center justify-between px-6 py-4">
        <Link href={user ? "/dashboard" : "/"} className="text-lg tracking-[0.3em] text-white">
          WILD MARKET
        </Link>
        <nav className="flex flex-wrap items-center gap-2">
          {items.map((item) => {
            const active = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`rounded-md px-3 py-2 text-sm transition ${
                  active
                    ? "border border-violet-300/70 bg-violet-400/10 text-violet-200"
                    : "border border-transparent text-white/70 hover:border-white/20 hover:text-white"
                }`}
              >
                {item.label}
              </Link>
            );
          })}
          {!loading && user ? (
            <button
              type="button"
              onClick={() => logout()}
              className="rounded-md px-3 py-2 text-sm text-white/70 hover:text-white border border-transparent hover:border-white/20"
            >
              Sign out
            </button>
          ) : null}
        </nav>
      </div>
    </header>
  );
}
