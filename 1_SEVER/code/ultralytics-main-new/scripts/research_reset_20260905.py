"""Read-only result audit and control-textbook page rendering for the agent research brief."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research_reset_20260905"
BOOK = Path("C:/Users/33836/Desktop/自动控制原理 (胡寿松) (z-library.sk, 1lib.sk, z-lib.sk).pdf")


def render(pages):
    directory = OUT / "book_pages"
    directory.mkdir(parents=True, exist_ok=True)
    with fitz.open(BOOK) as doc:
        for number in pages:
            target = directory / f"pdf_page_{number:03d}.png"
            doc[number - 1].get_pixmap(matrix=fitz.Matrix(1.6, 1.6)).save(target)
            print(target)


def audit():
    raw = json.loads((OUT / "history/history_architecture_inventory.json").read_text(encoding="utf-8"))
    previous = json.loads((ROOT / "reports/sage_v5_20260904/history/history_architecture_inventory.json")
                          .read_text(encoding="utf-8"))
    old = {r["path"]: r["csv_hash"] for r in previous["runs"]}
    seen, rows = set(), []
    for row in raw["runs"]:
        if row["csv_hash"] in seen:
            continue
        seen.add(row["csv_hash"])
        rows.append(row)
    overview = dict(
        csv_files=len(raw["runs"]), unique_contents=len(rows),
        changed_or_new=[r["path"] for r in raw["runs"] if old.get(r["path"]) != r["csv_hash"]],
        groups=dict(Counter(r["series"] for r in rows)),
        caveat="An identical CSV is not an independent repeat. Missing latest V5 is not a failed run.",
    )
    (OUT / "results_delta.json").write_text(json.dumps(overview, indent=2), encoding="utf-8")
    print(json.dumps(overview, indent=2))
    with fitz.open(BOOK) as doc:
        metadata = dict(path=str(BOOK), sha256=hashlib.sha256(BOOK.read_bytes()).hexdigest(), pages=len(doc),
                        text_layer_sample=[len(doc[i].get_text()) for i in (0, 10, 100)],
                        scope="Scan. Read selected rendered chapter pages; no claim of reading all 635 pages.")
    (OUT / "book_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", default="")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.pages:
        render([int(n) for n in args.pages.split(",")])
    else:
        audit()
