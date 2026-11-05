#!/usr/bin/env bash
# Post-deploy smoke check. No credentials, no secrets, read-only requests only.
# Usage: WEB_URL=https://portfolio-lenz.vercel.app API_URL=https://<space>.hf.space scripts/verify_prod.sh
set -u
: "${WEB_URL:?set WEB_URL}"
: "${API_URL:?set API_URL}"
fail=0

check() { # name url expected_status_regex [extra curl args...]
  local name=$1 url=$2 want=$3
  shift 3
  local code
  code=$(curl -s -m 30 -o /dev/null -w '%{http_code}' "$@" "$url") || code=000
  if [[ $code =~ ^($want)$ ]]; then
    printf 'ok    %-34s %s %s\n' "$name" "$code" "$url"
  else
    printf 'FAIL  %-34s %s (want %s) %s\n' "$name" "$code" "$want" "$url"
    fail=1
  fi
}

check "api /health"            "$API_URL/health" 200
check "api /llms.txt"          "$API_URL/llms.txt" 200
check "api docs hidden in prod" "$API_URL/docs" 404
check "api protected needs auth" "$API_URL/auth/me" 401
# Unsigned webhook must be rejected (400), or 503 when Stripe is not configured; never 200/500.
check "api stripe webhook route" "$API_URL/stripe/webhook" '400|503' -X POST -H 'content-type: application/json' -d '{}'

for p in / /login /register /privacy /terms /refund; do
  check "web $p" "$WEB_URL$p" 200
done
check "web protected redirects" "$WEB_URL/dashboard" '307|308|302'
check "web bff health" "$WEB_URL/api/proxy/health" 200
check "web bff anonymous me" "$WEB_URL/api/auth/me" 200
check "web bff blocks stripe path" "$WEB_URL/api/proxy/stripe/webhook" 404 -X POST

if curl -s -m 30 "$API_URL/llms.txt" | grep -q 'DELETE /auth/me'; then
  echo "ok    llms.txt lists DELETE /auth/me (new build is live)"
else
  echo "FAIL  llms.txt lacks DELETE /auth/me (old build still serving?)"; fail=1
fi

exit $fail
