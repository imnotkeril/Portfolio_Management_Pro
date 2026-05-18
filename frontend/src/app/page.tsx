"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";

type IndexItem = {
  name: string;
  symbol: string;
  price: number | null;
  change: number | null;
  change_pct: number | null;
  series: { x: string; y: number }[];
};

const quickLinks = [
  { href: "/login", label: "Sign in" },
  { href: "/register", label: "Register" },
  { href: "/dashboard", label: "Dashboard" },
];

export default function DashboardPage() {
  const [indices, setIndices] = useState<IndexItem[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .get<IndexItem[]>("/dashboard/indices")
      .then(setIndices)
      .catch((err) => setError(String(err)));
  }, []);

  return (
    <div className="space-y-8">
      <section className="panel p-6">
        <h1 className="text-4xl text-white">Market Dashboard</h1>
        <p className="mt-2 text-white/70">Quick navigation and market indices.</p>
        <div className="mt-6 grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
          {quickLinks.map((item) => (
            <Link key={item.href} href={item.href} className="metric-card text-center text-sm text-white/85 hover:border-violet-300/50">
              {item.label}
            </Link>
          ))}
        </div>
      </section>

      {error ? <div className="text-red-300">{error}</div> : null}

      <section className="space-y-4">
        <h2 className="text-2xl text-white">Market Indices</h2>
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-4">
          {indices.map((index) => {
            const positive = (index.change ?? 0) >= 0;
            return (
              <div key={index.symbol} className="metric-card">
                <div className="text-sm text-white/60">{index.name}</div>
                <div className="mt-2 text-xl text-white">
                  {index.price == null ? "N/A" : index.price.toFixed(2)}
                </div>
                <div className={`mt-1 text-sm ${positive ? "text-emerald-300" : "text-rose-300"}`}>
                  {index.change == null || index.change_pct == null
                    ? "No data"
                    : `${index.change.toFixed(2)} (${index.change_pct.toFixed(2)}%)`}
                </div>
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
}
