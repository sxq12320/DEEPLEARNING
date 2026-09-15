# -*- coding: utf-8 -*-
"""一次性清理：纯中文 ctext 字符串里的 x_y 记号去掉下划线（x_y → xy）。
mtex/MathTex/r""/f"" 行跳过（LaTeX 需要下标；f-string 内有代码）。"""
import glob
import re

PAT = re.compile(r'([A-Za-z\u0370-\u03c9])_([A-Za-z0-9]{1,3})')
STRLIT = re.compile(r'"(?:[^"\\]|\\.)*"')
SKIP = ('mtex', 'MathTex', 'font_size', 'sub_')

tot = 0
for f in glob.glob('L*/scenes.py') + ['_smoke/scenes.py']:
    lines = open(f, encoding='utf-8').read().split('\n')
    changed = 0
    for i, ln in enumerate(lines):
        st = ln.strip()
        if any(k in ln for k in SKIP) or st.startswith(('r"', "r'", 'f"', "f'")):
            continue
        new = STRLIT.sub(lambda m: PAT.sub(lambda mm: mm.group(1) + mm.group(2),
                                           m.group(0)), ln)
        if new != ln:
            lines[i] = new
            changed += 1
    if changed:
        open(f, 'w', encoding='utf-8').write('\n'.join(lines))
        print(f, changed)
        tot += changed
print('total lines touched:', tot)
