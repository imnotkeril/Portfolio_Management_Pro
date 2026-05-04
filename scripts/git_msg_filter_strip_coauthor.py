"""One-off git filter-branch --msg-filter helper: drop Cursor co-author line from commit messages."""

import sys

if __name__ == "__main__":
    lines = sys.stdin.readlines()
    out = [ln for ln in lines if "cursoragent@cursor.com" not in ln]
    sys.stdout.write("".join(out))
