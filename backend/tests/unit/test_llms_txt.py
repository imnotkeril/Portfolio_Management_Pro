"""api/llms.txt must describe exactly the routes the app serves."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from api.main import app

ROOT = Path(__file__).resolve().parents[2]


def _documented() -> set[str]:
    text = (ROOT / "api" / "llms.txt").read_text(encoding="utf-8")
    return set(re.findall(r"^### ([A-Z]+ \S+)$", text, flags=re.MULTILINE))


def _served() -> set[str]:
    ops = set()
    for path, methods in app.openapi()["paths"].items():
        for method in methods:
            ops.add(f"{method.upper()} {path}")
    return ops


def test_llms_txt_lists_exactly_the_served_operations() -> None:
    assert _documented() == _served()


def test_llms_txt_is_current_with_generator() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "gen_llms_txt.py"), "--check"],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, "run: python scripts/gen_llms_txt.py"


def test_llms_txt_endpoint_is_public_markdown(api_client) -> None:
    response = api_client.get("/llms.txt")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert response.text.startswith("# WMC Portfolio API")
    assert "sk_live" not in response.text and "whsec_" not in response.text
