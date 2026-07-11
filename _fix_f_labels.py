# -*- coding: utf-8 -*-
"""Fix mislabeled 'f' polygons -> 'orange_immature' in the two batch-2 JSONs.

Text-level replace to preserve LabelMe formatting, then re-parse to verify.
Safe to delete after use.
"""
import json
from pathlib import Path

ROOT = Path(r"E:\mastercode\data\orange_wuxi\annotion_x_2")
FILES = ["IMG20231120161107_BURST005.json", "IMG20231120161340_BURST001.json"]
OLD, NEW = "f", "orange_immature"

for name in FILES:
    fp = ROOT / name
    txt = fp.read_text(encoding="utf-8")
    before = json.loads(txt)
    n_f = sum(1 for s in before.get("shapes", []) if s.get("label") == OLD)

    # targeted replace on the label field only, both spacing variants
    patched = txt.replace(f'"label": "{OLD}"', f'"label": "{NEW}"')
    patched = patched.replace(f'"label":"{OLD}"', f'"label":"{NEW}"')

    after = json.loads(patched)  # must stay valid JSON
    still_f = sum(1 for s in after.get("shapes", []) if s.get("label") == OLD)
    now_oi = sum(1 for s in after.get("shapes", []) if s.get("label") == NEW)
    was_oi = sum(1 for s in before.get("shapes", []) if s.get("label") == NEW)

    assert still_f == 0, f"{name}: still has {still_f} 'f' after fix"
    assert now_oi == was_oi + n_f, f"{name}: count mismatch"

    fp.write_text(patched, encoding="utf-8")
    print(f"{name}: fixed {n_f} 'f' -> orange_immature "
          f"(orange_immature {was_oi} -> {now_oi}, total shapes {len(after.get('shapes', []))})")

print("done.")
