"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

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
  const [open, setOpen] = useState(false);

  return (
    <header className="border-b border-white/10 bg-black/30 backdrop-blur-md">
      <div className="mx-auto flex w-full max-w-7xl flex-wrap items-center justify-between px-6 py-4">
        <Link href={user ? "/dashboard" : "/"} className="whitespace-nowrap text-lg tracking-[0.3em] text-white">
          WILD MARKET
        </Link>
        <button
          type="button"
          className="min-h-[44px] min-w-[44px] rounded-md border border-white/20 px-3 text-sm text-white/80 md:hidden"
          aria-expanded={open}
          aria-controls="main-nav-links"
          onClick={() => setOpen((v) => !v)}
        >
          {open ? "Close" : "Menu"}
        </button>
        <nav
          id="main-nav-links"
          aria-label="Main"
          className={`${open ? "flex" : "hidden"} w-full flex-col gap-1 pt-3 md:flex md:w-auto md:flex-row md:flex-wrap md:items-center md:gap-2 md:pt-0`}
        >
          {items.map((item) => {
            const active = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setOpen(false)}
                className={`flex min-h-[44px] items-center rounded-md px-3 py-2 text-sm transition md:min-h-0 ${
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
              className="flex min-h-[44px] items-center rounded-md px-3 py-2 text-sm text-white/70 hover:text-white border border-transparent hover:border-white/20 md:min-h-0"
            >
              Sign out
            </button>
          ) : null}
        </nav>
      </div>
    </header>
  );
}
