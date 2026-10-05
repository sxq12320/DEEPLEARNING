# -*- coding: utf-8 -*-
"""Check audio coverage + duration vs scene video duration."""
import glob
import json
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
narr = json.load(open(os.path.join(ROOT, "narrations.json"), encoding="utf-8"))


def dur(p):
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", p],
        capture_output=True, text=True)
    return float(json.loads(out.stdout)["format"]["duration"])


n_miss = 0
worst = []
for lesson, scenes in narr.items():
    for scene in scenes:
        ap = os.path.join(ROOT, "_build", "audio", lesson, scene + ".mp3")
        vp = os.path.join(ROOT, lesson, "media", "videos", "scenes",
                          "480p15", scene + ".mp4")
        if not os.path.exists(ap):
            print("MISSING AUDIO", lesson, scene)
            n_miss += 1
            continue
        a, v = dur(ap), dur(vp)
        if a > v:
            worst.append((a - v, lesson, scene, round(a, 1), round(v, 1)))

print("missing:", n_miss)
worst.sort(reverse=True)
print("audio-longer-than-video (overrun s):")
for w in worst:
    print(f"  +{w[0]:.1f}s {w[1]}/{w[2]} audio={w[3]}s video={w[4]}s")
