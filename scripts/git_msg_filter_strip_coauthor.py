"""One-off git filter-branch --msg-filter helper: drop Cursor co-author line from commit messages.

Windows: git filter-branch runs from .git-rewrite; use an ABSOLUTE path to this file, e.g.:

  $env:FILTER_BRANCH_SQUELCH_WARNING = "1"
  git filter-branch -f --msg-filter "python E:/Main/Dev/Python/Done/WMC_Portfolio_Management/scripts/git_msg_filter_strip_coauthor.py" HEAD

Adjust the drive/path to match your clone. Then: verify with
`git log --format=%B | findstr /i cursoragent` (empty = OK), then
`git push --force-with-lease origin main`.
"""

import sys

if __name__ == "__main__":
    lines = sys.stdin.readlines()
    out = [ln for ln in lines if "cursoragent@cursor.com" not in ln]
    sys.stdout.write("".join(out))
