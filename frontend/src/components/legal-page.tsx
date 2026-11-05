import type { ReactNode } from "react";

export function LegalPage({
  title,
  updated,
  children,
}: {
  title: string;
  updated: string;
  children: ReactNode;
}) {
  return (
    <article className="mx-auto max-w-3xl space-y-5 text-white/80 leading-relaxed">
      <h1 className="text-3xl font-bold text-white">{title}</h1>
      <p className="text-sm text-white/50">Last updated: {updated}</p>
      <p
        role="note"
        className="rounded border border-amber-400/40 bg-amber-400/10 p-3 text-sm text-amber-200"
      >
        Draft: this text has not been reviewed by a lawyer yet. Bracketed items
        are filled in by the operator before launch.
      </p>
      {children}
    </article>
  );
}

export function LegalSection({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <section className="space-y-2">
      <h2 className="text-xl font-semibold text-white">{title}</h2>
      {children}
    </section>
  );
}
