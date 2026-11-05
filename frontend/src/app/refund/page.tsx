import type { Metadata } from "next";

import { LegalPage, LegalSection } from "@/components/legal-page";

export const metadata: Metadata = { title: "Refund Policy" };

export default function RefundPage() {
  return (
    <LegalPage title="Refund Policy" updated="2026-11-05">
      <LegalSection title="Cancelling">
        <p>
          You can cancel the Pro subscription at any time from the Billing page
          (Manage subscription). Cancelling stops future renewals; you keep Pro
          access as long as Stripe reports the subscription as active.
        </p>
      </LegalSection>
      <LegalSection title="Refunds">
        <p>
          [Operator to confirm the refund window, for example a full refund of
          the latest charge on request within 14 days.] To request a refund,
          write to [contact email] from your account email and name the charge.
          Approved refunds go back to the original payment method through
          Stripe and usually appear within 5 to 10 business days.
        </p>
        <p>
          Where consumer law gives you a right of withdrawal or refund, that
          right is not limited by this page.
        </p>
      </LegalSection>
      <LegalSection title="Deleting your account">
        <p>
          Deleting your account cancels the subscription immediately. It does
          not refund past charges on its own; request a refund first if you are
          eligible.
        </p>
      </LegalSection>
    </LegalPage>
  );
}
