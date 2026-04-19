import type { Metadata } from "next";
import type { ReactNode } from "react";

export const metadata: Metadata = {
  title: "Optimization Studio | WMC",
  description:
    "Notebook-aligned portfolio optimization (01–10): full API bundle, metrics, charts, efficient frontier.",
};

export default function OptimizationStudioLayout({ children }: { children: ReactNode }) {
  return children;
}
