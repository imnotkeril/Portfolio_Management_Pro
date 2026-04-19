"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const items = [
  { href: "/", label: "Dashboard" },
  { href: "/create", label: "Create Portfolio" },
  { href: "/portfolios", label: "Portfolio List" },
  { href: "/analysis", label: "Analysis" },
  { href: "/optimization", label: "Optimization" },
  { href: "/risk", label: "Risk Analysis" },
  { href: "/forecasting", label: "Forecasting" },
];

export function MainNav() {
  const pathname = usePathname();

  return (
    <header className="border-b border-white/10 bg-black/30 backdrop-blur-md">
      <div className="mx-auto flex w-full max-w-7xl items-center justify-between px-6 py-4">
        <div className="text-lg tracking-[0.3em] text-white">WILD MARKET</div>
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
        </nav>
      </div>
    </header>
  );
}

