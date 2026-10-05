# -*- coding: utf-8 -*-
"""Extract scene order + docstrings per lesson into _scene_docs.json."""
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
out = {}
for d in sorted(glob.glob(os.path.join(ROOT, "L*"))):
    name = os.path.basename(d)
    if not os.path.isdir(d) or not re.fullmatch(r"L\d+", name):
        continue
    src = open(os.path.join(d, "scenes.py"), encoding="utf-8-sig").read()
    scenes = []
    for m in re.finditer(
        r'class (S\d+_\w+)\(LessonScene\):\s*\n\s*"""(.*?)"""', src, re.S
    ):
        scenes.append(
            {
                "name": m.group(1),
                "doc": re.sub(r"\s+", " ", m.group(2)).strip(),
            }
        )
    out[name] = scenes
    print(name, len(scenes))

json.dump(
    out,
    open(os.path.join(ROOT, "_scene_docs.json"), "w", encoding="utf-8"),
    ensure_ascii=False,
    indent=1,
)
