"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";

import { PortfolioTabs } from "@/components/portfolio-tabs";
import { api } from "@/lib/api";
import type { Portfolio } from "@/lib/types";

export default function PortfolioLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const params = useParams();
  const id = String(params.id);
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [notFound, setNotFound] = useState(false);

  useEffect(() => {
    setNotFound(false);
    api
      .get<Portfolio>(`/portfolios/${id}`)
      .then(setPortfolio)
      .catch(() => setNotFound(true));
  }, [id]);

  if (notFound) {
    return (
      <div className="panel p-8 text-center">
        <h1 className="text-xl text-white">Portfolio not found</h1>
        <p className="text-sm text-white/50 mt-2">It may have been deleted or you lack access.</p>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      <PortfolioTabs portfolioId={id} portfolioName={portfolio?.name} />
      {children}
    </div>
  );
}
