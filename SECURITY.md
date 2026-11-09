# Security Policy

## Reporting a vulnerability

Please do not open a public issue for security problems. Report privately through GitHub:
**Security tab -> "Report a vulnerability"** (private security advisory) on this repository.

Include the affected component, reproduction steps and impact. You can expect an
acknowledgement within a few days. Do not test against other users' accounts or real payment data.

## Scope

The FastAPI service, the Next.js app and BFF proxy, and the deployment configuration in this
repository. Secrets (Stripe keys, JWT secret, database URLs) are never committed; use `.env`
files, which are git-ignored.
