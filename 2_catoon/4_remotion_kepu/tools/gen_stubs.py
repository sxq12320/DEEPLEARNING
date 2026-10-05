import os

BASE = os.path.join(os.path.dirname(__file__), '..', 'src', 'episodes')

TPL = """import React from 'react';
import {AbsoluteFill} from 'remotion';
import {C, FONT_HEAD} from '../lib/style';

export const E{n} = ({data}: {data?: any}) => (
  <AbsoluteFill style={{background: C.bg0, justifyContent: 'center', alignItems: 'center'}}>
    <div style={{color: C.text, fontFamily: FONT_HEAD, fontSize: 80}}>{n} 占位</div>
  </AbsoluteFill>
);
"""

for n in range(206, 251):
    p = os.path.join(BASE, 'E%d.tsx' % n)
    if not os.path.exists(p):
        open(p, 'w', encoding='utf-8').write(TPL.replace('{n}', str(n)))
print('done')
