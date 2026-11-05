import type { Metadata } from "next";

import { LegalPage, LegalSection } from "@/components/legal-page";

export const metadata: Metadata = { title: "Terms of Service" };

export default function TermsPage() {
  return (
    <LegalPage title="Terms of Service" updated="2026-11-05">
      <LegalSection title="The service">
        <p>
          WMC Portfolio Terminal, operated by [operator name], provides tools
          to track portfolios and to analyse, optimise, stress-test and
          forecast them. By creating an account you accept these terms.
        </p>
      </LegalSection>
      <LegalSection title="Not investment advice">
        <p>
          All figures, optimisations, risk measures and forecasts are
          calculated from market data and your inputs. They are estimates for
          information only, are not investment, tax or legal advice, and are
          not a recommendation to buy or sell anything. Past performance does
          not predict future results. You decide and you bear the risk.
        </p>
      </LegalSection>
      <LegalSection title="Your account">
        <p>
          You must give a valid email address and keep your password secret.
          You are responsible for activity under your account. You may delete
          your account at any time in Settings.
        </p>
      </LegalSection>
      <LegalSection title="Plans and payment">
        <p>
          The Free plan includes one portfolio. The Pro plan is a recurring
          subscription billed through Stripe at the price shown when you
          upgrade; it renews until you cancel in the customer portal.
          Cancelling stops renewal, and Pro access follows your subscription
          status. Refunds are described in the Refund Policy.
        </p>
      </LegalSection>
      <LegalSection title="Acceptable use">
        <p>
          Do not attempt to break, overload or probe the service, bypass
          limits, access other people&apos;s data, or use it for unlawful
          purposes. We may suspend accounts that do.
        </p>
      </LegalSection>
      <LegalSection title="Data sources and availability">
        <p>
          Market data comes from third parties and can be delayed, incomplete
          or unavailable. The service is provided as is, without a guarantee of
          uninterrupted availability or accuracy.
        </p>
      </LegalSection>
      <LegalSection title="Liability">
        <p>
          To the extent the law allows, we are not liable for losses arising
          from use of the service or from decisions based on its output.
          Nothing here limits liability that cannot be limited by law.
          [Governing law and jurisdiction to be completed by the operator.]
        </p>
      </LegalSection>
      <LegalSection title="Changes and contact">
        <p>
          We may update these terms; the date above changes when we do.
          Questions: [contact email].
        </p>
      </LegalSection>
    </LegalPage>
  );
}
