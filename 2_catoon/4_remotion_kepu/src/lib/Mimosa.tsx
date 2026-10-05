import {useCurrentFrame} from 'remotion';
import {C, clamp01} from './style';

/* 茎的贝塞尔参数（局部坐标，原点在根部） */
const P0 = {x: 0, y: 0};
const P1 = {x: 34, y: -150};
const P2 = {x: -24, y: -310};

export const stemPoint = (t: number) => ({
  x: (1 - t) * (1 - t) * P0.x + 2 * t * (1 - t) * P1.x + t * t * P2.x,
  y: (1 - t) * (1 - t) * P0.y + 2 * t * (1 - t) * P1.y + t * t * P2.y,
});

export const stemPath = `M ${P0.x} ${P0.y} Q ${P1.x} ${P1.y} ${P2.x} ${P2.y}`;

/* ---------- 一根羽轴 + 两列小叶 ---------- */
const Pinna = ({angle, len, fold, pairs = 5}: {angle: number; len: number; fold: number; pairs?: number}) => (
  <g transform={`rotate(${angle})`}>
    <line x1={0} y1={0} x2={len} y2={0} stroke={C.stemDark} strokeWidth={5} strokeLinecap="round" />
    {Array.from({length: pairs}).map((_, i) => {
      const px = 18 + (i * (len - 30)) / (pairs - 1);
      const fi = clamp01(fold * 1.55 - i * 0.09);
      return [-1, 1].map((side) => (
        <g key={`${i}${side}`}
          transform={`translate(${px},0) rotate(${side * (40 - fi * 34)}) scale(${1 - fi * 0.18},${1 - fi * 0.1})`}>
          <ellipse cx={21} cy={0} rx={22} ry={8.5} fill={i % 2 ? C.leaf : '#63cf80'} />
          <ellipse cx={23} cy={2.6} rx={20} ry={4} fill={C.leafDark} opacity={0.45} />
        </g>
      ));
    })}
  </g>
);

/* ---------- 一片复叶：叶柄 + 对生两支羽轴 ---------- */
export const Leaf = ({x, y, a0, dir, fold, len = 200}: {
  x: number; y: number; a0: number; dir: 1 | -1; fold: number; len?: number;
}) => {
  const angle = a0 + dir * fold * 74;
  return (
    <g transform={`translate(${x},${y}) rotate(${angle})`}>
      <line x1={0} y1={0} x2={64} y2={0} stroke={C.stem} strokeWidth={9} strokeLinecap="round" />
      {/* 叶枕：叶柄根部的小鼓包 */}
      <circle cx={6} cy={0} r={11} fill={C.stemDark} />
      <g transform={`translate(64,0)`}>
        <Pinna angle={-24 - fold * 10} len={len} fold={fold} />
        <Pinna angle={24 + fold * 10} len={len} fold={fold} />
      </g>
    </g>
  );
};

/* ---------- 整株含羞草 ----------
   folds: 每片叶子的折叠度 0..1（底部→顶部）
   droop: 整株蔫下去的程度 0..1                                        */
export const MimosaPlant = ({x, y, scale = 1, folds, droop = 0, sway = true}: {
  x: number; y: number; scale?: number; folds: number[]; droop?: number; sway?: boolean;
}) => {
  const f = useCurrentFrame();
  const idle = sway ? Math.sin(f * 0.045) * 1.6 : 0;
  const leafCfg = [
    {t: 0.32, dir: 1 as const, a0: -34},
    {t: 0.55, dir: -1 as const, a0: -146},
    {t: 0.78, dir: 1 as const, a0: -42},
    {t: 0.95, dir: -1 as const, a0: -100},
  ];
  return (
    <g transform={`translate(${x},${y}) scale(${scale}) rotate(${idle + droop * 5})`}>
      {/* 土堆 */}
      <ellipse cx={0} cy={8} rx={150} ry={26} fill="#2a3160" />
      <ellipse cx={-20} cy={2} rx={90} ry={14} fill="#333b70" />
      {/* 茎 */}
      <path d={stemPath} fill="none" stroke={C.stem} strokeWidth={13} strokeLinecap="round" />
      <path d={stemPath} fill="none" stroke={C.stemDark} strokeWidth={5} strokeLinecap="round"
        transform="translate(-3,0)" opacity={0.5} />
      {/* 顶端小叶芽 */}
      <g transform={`translate(${stemPoint(0.98).x},${stemPoint(0.98).y}) rotate(-95)`}>
        <ellipse cx={16} cy={-8} rx={20} ry={9} fill={C.leaf} transform="rotate(-30)" />
        <ellipse cx={16} cy={8} rx={20} ry={9} fill={C.leafDark} transform="rotate(30)" />
      </g>
      {leafCfg.map((lc, i) => {
        const p = stemPoint(lc.t);
        return <Leaf key={i} x={p.x} y={p.y} a0={lc.a0} dir={lc.dir}
          fold={clamp01(folds[i] ?? 0)} len={185 - i * 12} />;
      })}
    </g>
  );
};
