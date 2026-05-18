"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

type TabDef = {
  label: string;
  href: (id: string) => string;
  isActive: (pathname: string, id: string) => boolean;
};

const TABS: TabDef[] = [
  {
    label: "Overview",
    href: (id) => `/portfolio/${id}`,
    isActive: (pathname, id) => pathname === `/portfolio/${id}`,
  },
  {
    label: "Transactions",
    href: (id) => `/portfolio/${id}/transactions`,
    isActive: (pathname, id) =>
      pathname.startsWith(`/portfolio/${id}/transactions`),
  },
  {
    label: "Analysis",
    href: (id) => `/analysis?id=${id}`,
    isActive: (pathname) => pathname === "/analysis",
  },
  {
    label: "Risk",
    href: (id) => `/risk?id=${id}`,
    isActive: (pathname) => pathname === "/risk",
  },
  {
    label: "Optimization",
    href: (id) => `/optimization?id=${id}`,
    isActive: (pathname) =>
      pathname === "/optimization" ||
      pathname.startsWith("/optimization-studio") ||
      pathname === "/opti",
  },
  {
    label: "Forecasting",
    href: (id) => `/forecasting?id=${id}`,
    isActive: (pathname) => pathname === "/forecasting",
  },
];

export function PortfolioTabs({
  portfolioId,
  portfolioName,
}: {
  portfolioId: string;
  portfolioName?: string;
}) {
  const pathname = usePathname();

  return (
    <div className="space-y-4">
      <div>
        <Link
          href="/dashboard"
          className="text-xs text-white/50 hover:text-white"
        >
          ← Dashboard
        </Link>
        <h1 className="mt-1 text-2xl font-bold text-white">
          {portfolioName ?? "Portfolio"}
        </h1>
      </div>
      <nav className="tab-bar flex-wrap">
        {TABS.map((tab) => {
          const href = tab.href(portfolioId);
          const active = tab.isActive(pathname, portfolioId);
          return (
            <Link
              key={tab.label}
              href={href}
              className={`tab-btn ${active ? "active" : ""}`}
            >
              {tab.label}
            </Link>
          );
        })}
      </nav>
    </div>
  );
}
