# -*- coding: utf-8 -*-
"""用法: python _frame.py <mp4路径> <时刻秒> <输出png>"""
import subprocess
import sys

v, t, out = sys.argv[1], sys.argv[2], sys.argv[3]
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", t, "-i", v,
                "-frames:v", "1", out], check=True)
print("ok", out)
