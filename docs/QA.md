# QA report - Portfolio Management Pro (branch chore/audit)

Status: IN PROGRESS (written incrementally; verdict at the bottom is filled in last).
Code under test: HEAD 41d17a4 (includes fixes 47cfaa9..a7bac81).
Stack: throwaway docker network `audit-pmp-qa-net`, containers `audit-pmp-qa-*` (postgres 16 db `pmp_qa_test`, API image built from Dockerfile.api, frontend image from frontend/Dockerfile), fake Stripe API container, dummy `whsec_`/`sk_test_` generated at run time. No real keys, no .env read.
