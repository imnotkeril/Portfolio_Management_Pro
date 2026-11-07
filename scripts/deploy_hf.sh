#!/usr/bin/env bash
# Publish the API to the Hugging Face Space (Docker SDK). The Space is its own git repo,
# so a GitHub merge does not deploy it. Run by the owner (needs `hf auth login` or a git
# credential for huggingface.co); it never prints tokens.
#
# Usage: scripts/deploy_hf.sh [git-ref]      (default: origin/main)
# What it does: clones the Space, keeps the Space's own README.md (it holds the
# `sdk: docker`, `app_port: 7860` front matter), replaces every other file with the tracked
# files of <git-ref>, commits and pushes. Nothing from .env or untracked files is sent.
set -euo pipefail

REF=${1:-origin/main}
SPACE=${HF_SPACE:-imnotkeril/portfolio-management-pro-api}
ROOT=$(git rev-parse --show-toplevel)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

git -C "$ROOT" fetch -q origin
SHA=$(git -C "$ROOT" rev-parse --short "$REF")

git clone -q "https://huggingface.co/spaces/$SPACE" "$WORK/space"
cp "$WORK/space/README.md" "$WORK/space-README.md"
head -n 1 "$WORK/space-README.md" | grep -q '^---$' || { echo "Space README has no front matter; aborting"; exit 1; }

# wipe tracked files of the Space except .git, then lay down the new tree
git -C "$WORK/space" rm -rq . >/dev/null
git -C "$ROOT" archive "$REF" | tar -x -C "$WORK/space"
cp "$WORK/space-README.md" "$WORK/space/README.md"
# frontend and local-only material do not belong in the API Space
rm -rf "$WORK/space/frontend" "$WORK/space/docs"

git -C "$WORK/space" add -A
if git -C "$WORK/space" diff --cached --quiet; then
  echo "Space already matches $REF ($SHA)"; exit 0
fi
git -C "$WORK/space" commit -q -m "Deploy $SPACE from GitHub $SHA"
git -C "$WORK/space" push -q origin HEAD:main
echo "Pushed $SHA to $SPACE. Watch the Space build logs; then run scripts/verify_prod.sh."
