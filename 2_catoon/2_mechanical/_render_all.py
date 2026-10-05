# -*- coding: utf-8 -*-
"""Parallel 1080p30 render of every Lxx scenes.py scene.

manim exits 1 on success noise (version check) — success is judged by
the output mp4 existing and being non-trivial.
"""
import glob
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
LOGDIR = os.path.join(ROOT, "_render_log")
os.makedirs(LOGDIR, exist_ok=True)
PY = sys.executable
QUALITY_DIR = "1080p30"

tasks = []
for d in sorted(glob.glob(os.path.join(ROOT, "L*"))):
    lesson = os.path.basename(d)
    if not os.path.isdir(d) or not re.fullmatch(r"L\d+", lesson):
        continue
    src = open(os.path.join(d, "scenes.py"), encoding="utf-8-sig").read()
    for s in re.findall(r"class (S\d+_\w+)\(LessonScene\)", src):
        out = os.path.join(d, "media", "videos", "scenes", QUALITY_DIR, s + ".mp4")
        tasks.append((d, lesson, s, out))

print(f"{len(tasks)} scenes to render", flush=True)
t0 = time.time()
done = [0]


def run(t):
    d, lesson, s, out = t
    tag = f"{lesson}/{s}"
    if os.path.exists(out) and os.path.getsize(out) > 10000:
        done[0] += 1
        return f"skip {tag}"
    log = os.path.join(LOGDIR, f"{lesson}_{s}.log")
    with open(log, "w", encoding="utf-8") as lf:
        subprocess.run(
            [PY, "-m", "manim", "-qh", "--fps", "30", "scenes.py", s],
            cwd=d, stdout=lf, stderr=subprocess.STDOUT,
        )
    ok = os.path.exists(out) and os.path.getsize(out) > 10000
    done[0] += 1
    el = time.time() - t0
    print(f"[{done[0]}/{len(tasks)} {el:.0f}s] {'ok' if ok else 'FAIL'} {tag}",
          flush=True)
    return tag


with ThreadPoolExecutor(max_workers=6) as ex:
    list(ex.map(run, tasks))

print(f"ALL DONE in {(time.time()-t0)/60:.1f} min", flush=True)
