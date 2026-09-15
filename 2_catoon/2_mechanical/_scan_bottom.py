# -*- coding: utf-8 -*-
"""扫描：场景内在 takeaway() 之前创建、且未被 FadeOut/Transform 的
to_edge(DOWN) 常驻文本 —— 它们会与底部结论条叠印。"""
import re
import glob

for f in sorted(glob.glob('L*/scenes.py')):
    src = open(f, encoding='utf-8').read()
    for s in re.split(r'(?=class S\d)', src):
        name = s.split('(')[0].strip()
        ti = s.rfind('self.takeaway')
        if ti < 0:
            continue
        before = s[:ti]
        for m in re.finditer(r'(\w+)\s*=\s*(?:ctext|mtex|bullets)\(', before):
            var = m.group(1)
            seg = before[m.end():m.end() + 400]
            if 'to_edge(DOWN' in seg:
                after = before[m.end():]
                if (f'FadeOut({var})' not in after
                        and f'Transform({var}' not in after):
                    print(f'{f} {name}: {var}')
