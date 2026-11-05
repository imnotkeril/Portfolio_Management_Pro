import type { Metadata } from "next";

import { LegalPage, LegalSection } from "@/components/legal-page";

export const metadata: Metadata = { title: "Privacy Policy" };

export default function PrivacyPage() {
  return (
    <LegalPage title="Privacy Policy" updated="2026-11-05">
      <LegalSection title="Who we are">
        <p>
          WMC Portfolio Terminal is operated by [operator name and postal
          address]. Contact for privacy requests: [contact email].
        </p>
      </LegalSection>
      <LegalSection title="What we collect">
        <ul className="list-disc pl-6 space-y-1">
          <li>
            Account data: your email address and a salted hash of your password
            (we never store the password itself).
          </li>
          <li>
            Portfolio data you enter: portfolio names, tickers, quantities,
            prices, dates and transactions.
          </li>
          <li>
            Subscription data: your plan, its status and period end, and the
            Stripe customer and subscription identifiers. Card details are
            handled by Stripe and never reach our servers.
          </li>
          <li>
            Technical data: a session cookie, and your IP address, which is used
            for rate limiting and security logs.
          </li>
        </ul>
      </LegalSection>
      <LegalSection title="Why we use it">
        <p>
          To provide the service you asked for (contract), to charge for the
          Pro plan, and to keep the service secure and prevent abuse
          (legitimate interest). We do not sell personal data and we do not use
          it for advertising.
        </p>
      </LegalSection>
      <LegalSection title="Cookies">
        <p>
          We set one strictly necessary, HttpOnly session cookie to keep you
          signed in. We do not use analytics or advertising cookies.
        </p>
      </LegalSection>
      <LegalSection title="Who processes data for us">
        <ul className="list-disc pl-6 space-y-1">
          <li>Stripe, for payments and subscriptions.</li>
          <li>
            Our hosting and database providers ([web host], [API host],
            [database provider]).
          </li>
          <li>
            Market-data providers receive the ticker symbols you analyse, not
            your identity.
          </li>
        </ul>
        <p>Some of these providers may process data outside your country.</p>
      </LegalSection>
      <LegalSection title="Retention and deletion">
        <p>
          We keep your data while your account exists. You can delete your
          account at any time in Settings. This permanently removes your
          account, portfolios, transactions and subscription record, and
          cancels your subscription. Stripe keeps payment records as required
          by tax and accounting law.
        </p>
      </LegalSection>
      <LegalSection title="Your rights">
        <p>
          Depending on where you live you may have the right to access,
          correct, export or delete your data, to object to processing, and to
          complain to your data protection authority. Write to [contact email]
          and we will respond within one month.
        </p>
      </LegalSection>
      <LegalSection title="Changes">
        <p>We will update this page and its date when this policy changes.</p>
      </LegalSection>
    </LegalPage>
  );
}
