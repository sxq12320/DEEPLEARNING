# -*- coding: utf-8 -*-
"""Mux narration onto 1080p30 scene clips, then concat per lesson.

Per scene: pad audio with silence when shorter; freeze last frame when the
narration overruns the animation (scenes end on a takeaway hold anyway).
Output: 课程成片/Lxx_<title>_1080p30.mp4
"""
import glob
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
VDIR = os.path.join("media", "videos", "scenes", "1080p30")
ADIR = os.path.join(ROOT, "_build", "audio")
SEGDIR = os.path.join(ROOT, "_build", "segments")
OUTDIR = os.path.join(ROOT, "课程成片")
TAIL_PAD = 0.7          # silence tail on each scene narration
FF = ["-y", "-v", "error"]

TITLES = {
    "L01": "什么是机械", "L02": "机构能不能动", "L03": "让速度看得见",
    "L04": "机器中的暗力量", "L05": "让机器不抖", "L06": "机器的心跳",
    "L07": "四根杆的智慧·上", "L08": "四根杆的智慧·下",
    "L09": "按剧本运动的机器", "L10": "最完美的曲线·上",
    "L11": "最完美的曲线·下", "L12": "齿轮的组合艺术",
    "L13": "精巧的小机构", "L14": "让机器成为系统",
}


def dur(p):
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", p],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    return float(json.loads(out.stdout)["format"]["duration"])


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print("FFMPEG FAIL:", " ".join(cmd), "\n", r.stderr[-800:])
        sys.exit(1)


def make_segment(lesson, scene):
    """Return path of segment mp4 (uniform: h264 1080p30 + aac 48k stereo)."""
    v = os.path.join(ROOT, lesson, VDIR, scene + ".mp4")
    a = os.path.join(ADIR, lesson, scene + ".mp3")
    seg = os.path.join(SEGDIR, lesson, scene + ".mp4")
    if os.path.exists(seg) and os.path.getsize(seg) > 10000:
        return seg
    os.makedirs(os.path.dirname(seg), exist_ok=True)
    vd, ad = dur(v), dur(a) + TAIL_PAD
    if ad <= vd:
        run(["ffmpeg", *FF, "-i", v, "-i", a, "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "copy", "-af", f"apad=whole_dur={vd}",
             "-ar", "48000", "-ac", "2", "-c:a", "aac", "-t", f"{vd:.3f}", seg])
    else:
        ext = ad - vd
        run(["ffmpeg", *FF, "-i", v, "-i", a, "-filter_complex",
             f"[0:v]tpad=stop_mode=clone:stop={ext:.3f}[v]",
             "-map", "[v]", "-map", "1:a:0",
             "-c:v", "libx264", "-preset", "medium", "-crf", "18",
             "-pix_fmt", "yuv420p", "-r", "30",
             "-af", f"apad=whole_dur={ad}",
             "-ar", "48000", "-ac", "2", "-c:a", "aac",
             "-t", f"{ad:.3f}", seg])
    return seg


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    narr = json.load(open(os.path.join(ROOT, "narrations.json"),
                          encoding="utf-8"))
    for lesson in sorted(narr):
        d = os.path.join(ROOT, lesson)
        src = open(os.path.join(d, "scenes.py"), encoding="utf-8-sig").read()
        order = re.findall(r"class (S\d+_\w+)\(LessonScene\)", src)
        segs = []
        for s in order:
            v = os.path.join(d, VDIR, s + ".mp4")
            if not os.path.exists(v):
                print(f"MISSING VIDEO {lesson}/{s} — skip lesson")
                break
            segs.append(make_segment(lesson, s))
        else:
            out = os.path.join(
                OUTDIR, f"{lesson}_{TITLES[lesson]}_1080p30.mp4")
            cmd = ["ffmpeg", *FF]
            for sg in segs:
                cmd += ["-i", sg]
            cin = "".join(f"[{i}:v:0][{i}:a:0]" for i in range(len(segs)))
            cmd += [
                "-filter_complex",
                f"{cin}concat=n={len(segs)}:v=1:a=1[v][a]",
                "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p", "-r", "30",
                "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2",
                "-movflags", "+faststart", out]
            run(cmd)
            print(f"{lesson}: {len(segs)} scenes -> {os.path.basename(out)} "
                  f"{dur(out):.0f}s", flush=True)


if __name__ == "__main__":
    main()
