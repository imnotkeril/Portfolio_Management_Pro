#!/usr/bin/env bash
# Strip Co-authored-by Cursor lines and fix empty author/committer names (Windows/Git Bash).
# Usage: from repo root, run: bash scripts/run_filter_strip_coauthor.sh
# Requires: Git for Windows (bash), Python on PATH.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MSG_FILTER_PY="$ROOT/scripts/git_msg_filter_strip_coauthor.py"
export FILTER_BRANCH_SQUELCH_WARNING=1

cd "$ROOT"

# Stale filter-branch temp (ignore errors if missing/busy)
rm -rf .git-rewrite 2>/dev/null || true

# Prefer Windows path for Python msg-filter (Git filter-branch cwd is not repo root)
if command -v cygpath >/dev/null 2>&1; then
  MSG_FILTER_PY_WIN="$(cygpath -w "$MSG_FILTER_PY")"
else
  MSG_FILTER_PY_WIN="$MSG_FILTER_PY"
fi

git filter-branch -f \
  --env-filter '
if [ -z "${GIT_AUTHOR_NAME:-}" ]; then export GIT_AUTHOR_NAME=imnotkeril; fi
if [ -z "${GIT_COMMITTER_NAME:-}" ]; then export GIT_COMMITTER_NAME=imnotkeril; fi
' \
  --msg-filter "python $MSG_FILTER_PY_WIN" \
  HEAD

echo "OK. Verify: git log --format=%B | grep -i cursoragent || echo 'No cursoragent in messages'"
echo "Then: git push --force-with-lease origin main"
