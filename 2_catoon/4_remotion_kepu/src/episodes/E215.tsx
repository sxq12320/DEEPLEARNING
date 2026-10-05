import {AbsoluteFill, Audio, Sequence, interpolate, interpolateColors, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01, rand} from '../lib/style';
import {Backdrop, Ripple, Stamp, SceneTag, Chip, Subtitles, ProgressBar, FadeIn, Sun, Moon} from '../lib/atoms';
import tl from '../timeline_215.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：朝颜靛蓝 × 蓝紫花冠 × 凋谢粉（纯色平涂） ============ */
const P = {
  bg: '#33477c',      // 清晨靛蓝
  bgDeep: '#26355f',
  bgLite: '#405a94',
  card: '#2b3d6b',    // 卡片底
  plate: '#f4ecd4',   // 米盘
  ink: '#2d3760',     // 盘上深字
  face: '#6f7de8',    // 花冠蓝紫
  faceD: '#4f5ac0',
  faceHi: '#aab2f2',  // 星纹浅紫
  throat: '#f4f1ff',  // 喉部
  pink: '#ef8bad',    // 凋谢粉
  pinkD: '#c7648d',
  red: '#d65f78',     // 红紫
  pale: '#e9d3d9',    // 褪色面纱
  leaf: '#63c878',
  leafD: '#3f9152',
  vine: '#4da866',
  vineD: '#2f7a48',
  calyx: '#4f9d63',
  amber: '#ffc94d',
  coral: '#ff7b6b',
  sky: '#5cc8ff',
  water: '#4fc3f7',
  night: '#1d2850',
  moon: '#ffe9a8',
  cream: '#f6efdd',
  soil: '#483726',
  soilD: '#31251a',
};

/* ---------- 五浅裂花冠轮廓 ---------- */
const lobedPath = (R: number, dip = 0.82) => {
  const notch = (i: number) => {
    const a = ((-54 + i * 72) * Math.PI) / 180;
    return [Math.cos(a) * R * dip, Math.sin(a) * R * dip];
  };
  const tip = (i: number) => {
    const a = ((-18 + i * 72) * Math.PI) / 180;
    return [Math.cos(a) * R * 1.16, Math.sin(a) * R * 1.16];
  };
  const [x0, y0] = notch(0);
  let d = `M ${x0.toFixed(1)} ${y0.toFixed(1)} `;
  for (let i = 0; i < 5; i++) {
    const [cx, cy] = tip(i);
    const [x1, y1] = notch(i + 1);
    d += `Q ${cx.toFixed(1)} ${cy.toFixed(1)} ${x1.toFixed(1)} ${y1.toFixed(1)} `;
  }
  return d + 'Z';
};

/* ---------- 五角星纹 ---------- */
const starPath = (Ro: number, Ri: number, offDeg = -18) => {
  let d = '';
  for (let i = 0; i < 10; i++) {
    const a = ((offDeg + i * 36) * Math.PI) / 180;
    const r = i % 2 === 0 ? Ro : Ri;
    d += `${i === 0 ? 'M' : 'L'} ${(Math.cos(a) * r).toFixed(1)} ${(Math.sin(a) * r).toFixed(1)} `;
  }
  return d + 'Z';
};

/* ---------- 花色插值：蓝紫→粉→红紫 ---------- */
const faceCol = (hue: number) => interpolateColors(hue, [0, 0.55, 1], [P.face, P.pink, P.red]);
const faceColD = (hue: number) => interpolateColors(hue, [0, 0.55, 1], [P.faceD, P.pinkD, '#a8455c']);
const starCol = (hue: number) => interpolateColors(hue, [0, 0.55, 1], [P.throat, '#f6d0dc', '#eebcc8']);

/* ---------- 牵牛花正面（open 展开 / wilt 0→1 分三阶段：卷边→褪色→闭合） ---------- */
const Bloom = ({x, y, r = 110, s = 1, rot = 0, open = 1, wilt = 0, hue = 0}: {
  x: number; y: number; r?: number; s?: number; rot?: number;
  open?: number; wilt?: number; hue?: number;
}) => {
  const o = Math.max(0.001, clamp01(open));
  const curlK = clamp01(wilt * 3);
  const fadeK = clamp01(wilt * 3 - 1);
  const closeK = clamp01(wilt * 3 - 2);
  const col = faceCol(hue);
  const colD = faceColD(hue);
  const rr = r * o;
  return (
    <g transform={`translate(${x},${y + closeK * 30 * s}) rotate(${rot + closeK * 16}) scale(${s * (0.6 + 0.4 * o) * (1 - closeK * 0.34)})`}>
      {/* 花冠筒背 */}
      <path d={`M ${-r * 0.15} ${r * 0.12} L ${-r * 0.1} ${r * 1.02} Q 0 ${r * 1.16} ${r * 0.1} ${r * 1.02} L ${r * 0.15} ${r * 0.12} Z`} fill={colD} />
      {/* 花冠盘 */}
      <path d={lobedPath(rr * 1.05)} fill={col} />
      <path d={starPath(rr * 0.82, rr * 0.34)} fill={starCol(hue)} opacity={0.92 - fadeK * 0.3} />
      <circle r={rr * 0.2} fill={P.throat} />
      <circle r={rr * 0.09} fill={P.moon} opacity={0.95} />
      {/* 卷边：瓣尖内卷弧 */}
      {curlK > 0.02 && Array.from({length: 5}).map((_, i) => {
        const a = ((-18 + i * 72) * Math.PI) / 180;
        const cx = Math.cos(a) * rr * 0.98, cy = Math.sin(a) * rr * 0.98;
        return (
          <path key={i} d={`M ${-rr * 0.2} 0 Q 0 ${-rr * 0.3 * curlK} ${rr * 0.2} 0`}
            transform={`translate(${cx},${cy}) rotate(${(a * 180) / Math.PI + 90})`}
            stroke={colD} strokeWidth={5.5} fill="none" strokeLinecap="round" opacity={0.3 + curlK * 0.65} />
        );
      })}
      {/* 褪色面纱 */}
      {fadeK > 0.02 && <path d={lobedPath(rr * 1.05)} fill={P.pale} opacity={fadeK * 0.42} />}
      {/* 萎缩皱褶 */}
      {closeK > 0.02 && [-0.5, 0, 0.5].map((k, i) => (
        <path key={i} d={`M ${k * rr * 1.1} ${-rr * 0.55} Q ${k * rr * 0.8} 0 ${k * rr * 1.0} ${rr * 0.5}`}
          stroke={colD} strokeWidth={4} fill="none" opacity={closeK * 0.5} strokeLinecap="round" />
      ))}
    </g>
  );
};

/* ---------- 牵牛花侧视漏斗（同上三阶段蔫） ---------- */
const Trumpet = ({x, y, s = 1, rot = 0, open = 1, wilt = 0, hue = 0}: {
  x: number; y: number; s?: number; rot?: number;
  open?: number; wilt?: number; hue?: number;
}) => {
  const o = clamp01(open);
  const curlK = clamp01(wilt * 3);
  const fadeK = clamp01(wilt * 3 - 1);
  const closeK = clamp01(wilt * 3 - 2);
  const col = faceCol(hue);
  const colD = faceColD(hue);
  const w = 62 * (0.24 + 0.76 * o) * (1 - closeK * 0.5);
  const h = 104 * (0.42 + 0.58 * o) * (1 - closeK * 0.32);
  const rimRy = 14 * o * (1 - closeK * 0.5);
  const funnel = `M -7 -10 C -9 ${-h * 0.35} ${-w * 0.55} ${-h * 0.7} ${-w} ${-h} Q 0 ${-h - rimRy * 2.4} ${w} ${-h} C ${w * 0.55} ${-h * 0.7} 9 ${-h * 0.35} 7 -10 Z`;
  return (
    <g transform={`translate(${x},${y + closeK * 24}) rotate(${rot + closeK * 14}) scale(${s})`}>
      {/* 花萼 */}
      <path d="M -14 0 L 14 0 L 8 -16 L -8 -16 Z" fill={P.calyx} />
      <path d="M -14 0 L -20 11 M 14 0 L 20 11 M 0 1 L 0 13" stroke={P.calyx} strokeWidth={5} strokeLinecap="round" />
      {/* 漏斗花冠 */}
      <path d={funnel} fill={col} />
      <ellipse cx={0} cy={-h} rx={w} ry={rimRy} fill={P.faceHi} opacity={0.5 - fadeK * 0.28} />
      {/* 筒内褶线 */}
      {o > 0.4 && [-0.5, 0, 0.5].map((k, i) => (
        <path key={i} d={`M ${k * 9} -14 C ${k * w * 0.72} ${-h * 0.5} ${k * w * 0.92} ${-h - 3}`}
          stroke={colD} strokeWidth={3.5} fill="none" opacity={0.45} />
      ))}
      {/* 卷边：口沿两端下卷 */}
      {curlK > 0.02 && (
        <g stroke={colD} strokeWidth={5} fill="none" strokeLinecap="round">
          <path d={`M ${-w} ${-h} q ${-16 * curlK} ${18 * curlK} ${4} ${32 * curlK}`} />
          <path d={`M ${w} ${-h} q ${16 * curlK} ${18 * curlK} ${-4} ${32 * curlK}`} />
        </g>
      )}
      {fadeK > 0.02 && <path d={funnel} fill={P.pale} opacity={fadeK * 0.45} />}
      {/* 闭合皱褶 */}
      {closeK > 0.02 && [-0.4, 0.4].map((k, i) => (
        <path key={i} d={`M ${k * w} ${-h * 0.88} q ${k * 18} ${h * 0.34} ${k * 5} ${h * 0.68}`}
          stroke={colD} strokeWidth={4} fill="none" opacity={closeK * 0.6} strokeLinecap="round" />
      ))}
    </g>
  );
};

/* ---------- 螺旋花苞 ---------- */
const Bud = ({x, y, s = 1, rot = 0, k = 1, col = P.face}: {
  x: number; y: number; s?: number; rot?: number; k?: number; col?: string;
}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s * Math.max(0.001, k)})`} opacity={k > 0.02 ? 1 : 0}>
    <path d="M -13 2 L 13 2 L 8 -12 L -8 -12 Z" fill={P.calyx} />
    <path d="M -13 2 L -19 12 M 13 2 L 19 12" stroke={P.calyx} strokeWidth={5} strokeLinecap="round" />
    <path d="M -11 -10 Q -16 -50 -5 -82 Q 0 -95 6 -82 Q 15 -48 11 -10 Z" fill={col} />
    <path d="M -9 -20 Q 3 -28 9 -36" stroke={P.faceD} strokeWidth={3.4} fill="none" strokeLinecap="round" opacity={0.65} />
    <path d="M -8 -36 Q 2 -44 8 -52" stroke={P.faceD} strokeWidth={3.2} fill="none" strokeLinecap="round" opacity={0.6} />
    <path d="M -6 -52 Q 1 -60 6 -66" stroke={P.faceD} strokeWidth={3} fill="none" strokeLinecap="round" opacity={0.55} />
    <ellipse cx={1} cy={-86} rx={5.5} ry={7.5} fill={P.faceHi} />
  </g>
);

/* ---------- 心形叶 ---------- */
const LeafHeart = ({x, y, s = 1, rot = 0, k = 1, col = P.leaf}: {
  x: number; y: number; s?: number; rot?: number; k?: number; col?: string;
}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s * Math.max(0.001, k)})`} opacity={k > 0.02 ? 1 : 0}>
    <path d="M 0 -4 C -18 -24 -42 -14 -38 12 C -34 32 -14 42 0 56 C 14 42 34 32 38 12 C 42 -14 18 -24 0 -4 Z" fill={col} />
    <path d="M 0 0 L 0 44 M 0 14 L -17 25 M 0 14 L 17 25 M 0 29 L -11 36 M 0 29 L 11 36"
      stroke={P.leafD} strokeWidth={3} fill="none" opacity={0.55} strokeLinecap="round" />
    <line x1={0} y1={-4} x2={0} y2={-18} stroke={P.vineD} strokeWidth={5} strokeLinecap="round" />
  </g>
);

/* ---------- 藤蔓：蛇形茎 + 卷须 ---------- */
const Vine = ({x, y, h = 320, rot = 0, sway = 1, seed = 0, k = 1, leaves = true}: {
  x: number; y: number; h?: number; rot?: number; sway?: number; seed?: number; k?: number; leaves?: boolean;
}) => {
  const f = useCurrentFrame();
  const sw = Math.sin(f * 0.045 + seed * 1.7) * sway;
  return (
    <g transform={`translate(${x},${y}) rotate(${rot + sw}) scale(1,${Math.max(0.001, k)})`}>
      <path d={`M 0 0 C -34 ${-h * 0.28} 30 ${-h * 0.55} -8 ${-h * 0.78} C -26 ${-h * 0.9} 8 ${-h * 0.97} 16 ${-h}`}
        stroke={P.vine} strokeWidth={7} fill="none" strokeLinecap="round" />
      <path d={`M 16 ${-h} q 20 -4 18 -22 q -2 -14 -16 -12 q -10 2 -8 12`} stroke={P.vine} strokeWidth={4} fill="none" strokeLinecap="round" />
      {leaves && (
        <g>
          <LeafHeart x={-16} y={-h * 0.3} rot={-38} s={0.85} />
          <LeafHeart x={14} y={-h * 0.56} rot={32} s={0.8} />
          <LeafHeart x={-10} y={-h * 0.76} rot={-12} s={0.66} />
        </g>
      )}
    </g>
  );
};

/* ---------- 蜜蜂 ---------- */
const Bee = ({x, y, s = 1, flip = 1, k = 1, sleep = 0}: {
  x: number; y: number; s?: number; flip?: number; k?: number; sleep?: number;
}) => {
  const f = useCurrentFrame();
  const wig = Math.sin(f * 0.55) * 10 * (1 - sleep);
  const kk = Math.max(0.001, k);
  return (
    <g transform={`translate(${x + Math.sin(f * 0.3) * 7 * (1 - sleep)},${y + Math.cos(f * 0.42) * 6 * (1 - sleep)}) scale(${flip * s * kk},${s * kk}) rotate(${sleep * 24})`}
      opacity={kk > 0.02 ? 1 : 0}>
      <ellipse cx={-4} cy={-17} rx={11} ry={7} fill="#dff0ff" opacity={0.85} transform={`rotate(${-24 - wig} -4 -17)`} />
      <ellipse cx={10} cy={-16} rx={10} ry={6.4} fill="#dff0ff" opacity={0.7} transform={`rotate(${22 + wig} 10 -16)`} />
      <ellipse rx={21} ry={13.5} fill={P.amber} />
      <rect x={-6} y={-13.5} width={6.5} height={27} rx={3} fill="#4d3a1e" />
      <rect x={4} y={-12.5} width={5.5} height={25} rx={2.7} fill="#4d3a1e" />
      <circle cx={21} cy={-3} r={7.5} fill="#4d3a1e" />
      <circle cx={23} cy={-5} r={2} fill="#fff" />
      <path d="M -20 2 L -28 6" stroke="#4d3a1e" strokeWidth={3.5} strokeLinecap="round" />
    </g>
  );
};

/* ---------- 钟面 ---------- */
const ClockFace = ({x, y, r = 92, k = 1, hAng = 0, mAng = 0, accent = P.coral}: {
  x: number; y: number; r?: number; k?: number; hAng?: number; mAng?: number; accent?: string;
}) => (
  <g transform={`translate(${x},${y}) scale(${Math.max(0.001, k)})`} opacity={k > 0.02 ? 1 : 0}>
    <circle r={r} fill={P.plate} stroke={P.ink} strokeWidth={6} />
    {Array.from({length: 12}).map((_, i) => {
      const a = (i / 12) * Math.PI * 2;
      return <line key={i} x1={Math.cos(a) * r * 0.78} y1={Math.sin(a) * r * 0.78}
        x2={Math.cos(a) * r * 0.9} y2={Math.sin(a) * r * 0.9} stroke={P.ink} strokeWidth={5} strokeLinecap="round" />;
    })}
    <line x1={0} y1={0} x2={0} y2={-r * 0.5} stroke={P.ink} strokeWidth={9} strokeLinecap="round" transform={`rotate(${hAng})`} />
    <line x1={0} y1={0} x2={r * 0.4} y2={0} stroke={accent} strokeWidth={6} strokeLinecap="round" transform={`rotate(${mAng})`} />
    <circle r={8} fill={accent} />
  </g>
);

/* ---------- 叉号（否定标记） ---------- */
const XMark = ({x, y, r = 70, at, col = P.coral}: {x: number; y: number; r?: number; at: number; col?: string}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const kk = Math.max(0.001, clamp01(spring({frame: f - at, fps, config: {damping: 11, stiffness: 190}})));
  const o = interpolate(f, [at, at + 5], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  if (o <= 0) return null;
  return (
    <g transform={`translate(${x},${y}) scale(${kk})`} opacity={o} stroke={col} strokeWidth={13} strokeLinecap="round">
      <circle r={r} fill="none" opacity={0.9} />
      <line x1={-r * 0.62} y1={-r * 0.62} x2={r * 0.62} y2={r * 0.62} />
      <line x1={r * 0.62} y1={-r * 0.62} x2={-r * 0.62} y2={r * 0.62} />
    </g>
  );
};

/* ---------- 玻璃水杯 ---------- */
const Jar = ({x, y, s = 1, k = 1, fill = 0.6}: {x: number; y: number; s?: number; k?: number; fill?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s * Math.max(0.001, k)})`} opacity={k > 0.02 ? 1 : 0}>
    <path d="M -58 -120 L -58 42 Q -58 64 -36 64 L 36 64 Q 58 64 58 42 L 58 -120"
      fill="none" stroke={P.cream} strokeWidth={6} strokeLinecap="round" />
    <path d={`M -52 ${40 - 138 * fill} L -52 38 Q -52 56 -34 56 L 34 56 Q 52 56 52 38 L 52 ${40 - 138 * fill} Z`}
      fill={P.water} opacity={0.55} />
  </g>
);

/* ---------- 遮阳伞 ---------- */
const Umbrella = ({x, y, s = 1, k = 1, col = P.coral}: {x: number; y: number; s?: number; k?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s * Math.max(0.001, k)})`} opacity={k > 0.02 ? 1 : 0}>
    <line x1={0} y1={-4} x2={0} y2={150} stroke={P.cream} strokeWidth={7} strokeLinecap="round" />
    <path d="M -100 0 Q -64 -58 0 -62 Q 64 -58 100 0 Q 66 -16 33 -2 Q 0 -16 -33 -2 Q -66 -16 -100 0 Z" fill={col} />
    <circle cx={0} cy={-62} r={6} fill={P.cream} />
  </g>
);

/* ---------- 浇水壶（tilt>20 洒水） ---------- */
const WateringCan = ({x, y, s = 1, tilt = 0, k = 1}: {x: number; y: number; s?: number; tilt?: number; k?: number}) => {
  const f = useCurrentFrame();
  return (
    <g transform={`translate(${x},${y}) rotate(${tilt}) scale(${s * Math.max(0.001, k)})`} opacity={k > 0.02 ? 1 : 0}>
      <path d="M 26 -18 L 88 -50 L 96 -37 L 34 -4 Z" fill={P.sky} />
      <rect x={-56} y={-36} width={88} height={68} rx={16} fill={P.sky} />
      <rect x={-24} y={-50} width={34} height={16} rx={8} fill={P.sky} />
      <path d="M -56 -28 q -30 6 -26 40" stroke={P.sky} strokeWidth={11} fill="none" strokeLinecap="round" />
      {tilt > 20 && [0, 1, 2].map((i) => {
        const t = (f * 1.6 + i * 9) % 26;
        return <circle key={i} cx={92 + i * 15 + t * 1.6} cy={-40 + t * t * 0.24} r={6 - i} fill={P.water} opacity={0.9 - t * 0.03} />;
      })}
    </g>
  );
};

/* ---------- 谢幕清单行 ---------- */
const CheckRow = ({x, y, label, on, w = 430}: {x: number; y: number; label: string; on: number; w?: number}) => (
  <g opacity={0.45 + clamp01(on) * 0.55}>
    <rect x={x} y={y - 30} width={w} height={60} rx={16} fill={P.bgLite} opacity={0.85} />
    <circle cx={x + 36} cy={y} r={17} fill={on > 0.5 ? P.amber : 'rgba(0,0,0,0)'} stroke={P.cream} strokeWidth={4} />
    {on > 0.02 && (
      <path d="M -9 0 L -3 8 L 11 -8" stroke={P.ink} strokeWidth={6} fill="none" strokeLinecap="round"
        transform={`translate(${x + 36},${y}) scale(${Math.max(0.001, on)})`} />
    )}
    <text x={x + 74} y={y + 12} fill={P.cream} fontSize={35} fontFamily={FONT_HEAD}>{label}</text>
  </g>
);

/* ---------- pH 表盘 ---------- */
const PhGauge = ({x, y, r = 118, ph = 7, k = 1}: {x: number; y: number; r?: number; ph?: number; k?: number}) => {
  const ang = interpolate(ph, [4.8, 8.2], [64, -64], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  return (
    <g transform={`translate(${x},${y}) scale(${Math.max(0.001, k)})`} opacity={k > 0.02 ? 1 : 0}>
      <circle r={r} fill={P.plate} stroke={P.ink} strokeWidth={6} />
      {[8, 7, 6, 5].map((v) => {
        const a = ((interpolate(v, [4.8, 8.2], [64, -64]) - 90) * Math.PI) / 180;
        return (
          <g key={v}>
            <line x1={Math.cos(a) * r * 0.72} y1={Math.sin(a) * r * 0.72} x2={Math.cos(a) * r * 0.86} y2={Math.sin(a) * r * 0.86}
              stroke={P.ink} strokeWidth={5} strokeLinecap="round" />
            <text x={Math.cos(a) * r * 0.55} y={Math.sin(a) * r * 0.55 + 10} textAnchor="middle" fill={P.ink}
              fontSize={26} fontFamily={FONT_HEAD} fontWeight={700}>{v}</text>
          </g>
        );
      })}
      <line x1={0} y1={r * 0.14} x2={0} y2={-r * 0.62} stroke={P.coral} strokeWidth={8} strokeLinecap="round" transform={`rotate(${ang})`} />
      <circle r={9} fill={P.coral} />
      <text y={r * 0.42} textAnchor="middle" fill={P.ink} fontSize={30} fontFamily={FONT_HEAD} fontWeight={800}>pH</text>
    </g>
  );
};

/* ---------- 云朵 ---------- */
const Cloud = ({x, y, s = 1, k = 1}: {x: number; y: number; s?: number; k?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s * Math.max(0.001, k)})`} opacity={k > 0.02 ? 1 : 0}>
    <ellipse cx={-22} cy={6} rx={26} ry={18} fill={P.cream} />
    <ellipse cx={8} cy={-6} rx={32} ry={24} fill={P.cream} />
    <ellipse cx={34} cy={8} rx={24} ry={16} fill={P.cream} />
    <rect x={-40} y={2} width={94} height={18} rx={9} fill={P.cream} />
  </g>
);

/* ============ 标题卡 ============ */
const TitleCard = ({data}: {data: any}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 13, mass: 0.7}});
  const pop = spring({frame: f - 6, fps, config: {damping: 10, stiffness: 140}});
  const o = interpolate(f, [0, 8], [0, 1]);
  return (
    <AbsoluteFill>
      <Backdrop top={P.bg} bottom={P.bg} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <circle cx={960} cy={348} r={210} fill={P.plate} />
        {/* 藤蔓沿盘底攀上 */}
        <path d="M 780 520 Q 830 440 900 400" stroke={P.vine} strokeWidth={6} fill="none" strokeLinecap="round" />
        <path d="M 1140 520 Q 1100 440 1030 400" stroke={P.vine} strokeWidth={6} fill="none" strokeLinecap="round" />
        <LeafHeart x={818} y={468} rot={-40} s={0.8} k={clamp01(pop)} />
        <LeafHeart x={1112} y={464} rot={40} s={0.75} k={clamp01(pop)} />
        <Bud x={1106} y={430} rot={18} s={0.85} k={clamp01(pop)} />
        <g transform="translate(960,340)">
          <Bloom x={0} y={0} r={128} open={clamp01(pop)} />
        </g>
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 260}}>
          <div style={{color: P.faceHi, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 26}}>
            冷知识 · 第 {data.episode} 期
          </div>
          <div style={{color: C.text, fontSize: 92, fontWeight: 900, fontFamily: FONT_HEAD, letterSpacing: 5}}>
            {data.title}
          </div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ============ S0 对照实验：晒太阳的盆栽 vs 泡水+遮阴，同步到点蔫 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);

  const popL = spring({frame: f - 6, fps, config: {damping: 11, stiffness: 130}});
  const popR = spring({frame: f - 14, fps, config: {damping: 11, stiffness: 130}});
  const clockK = spring({frame: f - 24, fps, config: {damping: 12, stiffness: 140}});

  // 中午到点：两边同步蔫
  const wiltK = clamp01((f - 300) / 95);
  const hueK = wiltK * 0.62;
  // 钟：8点 → 13点
  const hAng = interpolate(f, [40, 330], [240, 390], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const mAng = f * 10;
  // 遮阳伞 cue1 弹出
  const umbK = spring({frame: f - b1 - 30, fps, config: {damping: 10, stiffness: 150}});

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={0} name={sec.name} accent={P.amber} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <rect x={0} y={872} width={1920} height={64} fill={P.soil} />
        <rect x={0} y={872} width={1920} height={11} fill={P.soilD} />
        {/* 中缝虚线：A/B 对照 */}
        <line x1={960} y1={330} x2={960} y2={850} stroke={P.cream} strokeWidth={4} strokeDasharray="6 20" opacity={0.25} strokeLinecap="round" />

        {/* ---- 左：晒太阳的盆栽 ---- */}
        <g transform={`translate(0,${(1 - clamp01(popL)) * 40})`} opacity={clamp01(popL)}>
          <path d="M 330 872 L 350 796 L 510 796 L 530 872 Z" fill="#b96a44" />
          <rect x={322} y={780} width={216} height={22} rx={11} fill={P.soilD} />
          <Vine x={430} y={796} h={250} sway={1.4} seed={1} />
          <Trumpet x={452} y={582} s={1.35} rot={-8} open={1} wilt={wiltK} hue={hueK} />
          <Bud x={360} y={640} rot={-30} s={0.8} k={1 - wiltK * 0.4} />
        </g>
        {/* 太阳 + 叉号 */}
        <Sun x={270} y={165} r={62} />
        <XMark x={270} y={165} r={86} at={232} />
        <text x={270} y={310} textAnchor="middle" fill={P.cream} fontSize={30} fontFamily={FONT_HEAD}
          opacity={clamp01((f - 240) / 10)}>不是晒的</text>

        {/* ---- 右：水杯扦插 + 遮阳 ---- */}
        <g transform={`translate(0,${(1 - clamp01(popR)) * 40})`} opacity={clamp01(popR)}>
          <line x1={1440} y1={858} x2={1440} y2={640} stroke={P.vineD} strokeWidth={7} strokeLinecap="round" />
          <LeafHeart x={1406} y={700} rot={-42} s={0.75} />
          <Trumpet x={1452} y={636} s={1.3} rot={6} open={1} wilt={wiltK} hue={hueK} />
          <Jar x={1440} y={856} s={1.05} />
        </g>
        <Umbrella x={1440} y={400} s={1.05} k={clamp01(umbK)} />
        {clamp01(umbK) > 0.5 && <Ripple x={1440} y={856} startFrame={b1 + 36} color={P.water} />}

        {/* ---- 中：时钟 8:00→13:00 ---- */}
        <ClockFace x={960} y={205} r={92} k={clamp01(clockK)} hAng={hAng} mAng={mAng} />
        <text x={960} y={336} textAnchor="middle" fill={P.cream} fontSize={29} fontFamily={FONT_HEAD}
          opacity={clamp01(clockK) * 0.85}>清晨 → 正午</text>
      </svg>
      <Chip text="一朵花只开半天" x={150} y={400} at={beat(0) + 70} accent={P.faceHi} />
      <Stamp text="太阳不背锅" x={480} y={300} at={240} color={P.coral} size={50} rot={-6} />
      <Chip text="泡水 + 遮阴 · 一样收工" x={1230} y={560} at={b1 + 60} accent={P.water} />
      <Stamp text="程序定好的报废" x={180} y={700} at={b1 + 130} color={P.amber} size={48} rot={5} />
    </FadeIn>
  );
};

/* ============ S1 基因闹钟：夜幕花苞→细胞泵膨压→恒光房照样开 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);

  const panelK = spring({frame: f - 4, fps, config: {damping: 13, stiffness: 120}});
  const clockK = spring({frame: f - 14, fps, config: {damping: 12, stiffness: 140}});
  // 钟：3:30 → 5:30
  const hAng = interpolate(f, [20, 430], [105, 168], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});

  // 膨压泵入 → 花苞展开
  const pumpK = clamp01((f - b1 - 20) / 110);
  const budK = 1 - clamp01((f - b1 - 130) / 42);
  const openK = clamp01(spring({frame: f - b1 - 138, fps, config: {damping: 12, stiffness: 110}}));

  // 恒光房卡片
  const roomK = spring({frame: f - b2 - 8, fps, config: {damping: 12, stiffness: 130}});
  const roomOpenK = clamp01(spring({frame: f - b2 - 60, fps, config: {damping: 12, stiffness: 120}}));

  return (
    <FadeIn>
      <Backdrop top={P.bgDeep} bottom={P.bgDeep} />
      <SceneTag index={1} name={sec.name} accent={P.faceHi} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 夜幕窗 */}
        <g transform={`translate(0,${(1 - clamp01(panelK)) * 40})`} opacity={clamp01(panelK)}>
          <rect x={110} y={150} width={1060} height={690} rx={44} fill={P.night} />
          <Moon x={270} y={290} scale={1} />
          {Array.from({length: 10}).map((_, i) => {
            const tw = 0.5 + Math.sin(f / 9 + i * 2.2) * 0.5;
            return <circle key={i} cx={160 + rand(i, 71) * 960} cy={190 + rand(i, 72) * 300}
              r={rand(i, 73) * 2.4 + 1.4} fill="#fff" opacity={0.35 + tw * 0.5} />;
          })}
          <rect x={150} y={790} width={980} height={50} rx={22} fill="#16204a" />
          <Vine x={560} y={806} h={250} sway={0.9} seed={3} />
          <Bud x={574} y={582} rot={-6} s={1.5} k={budK} />
          <Trumpet x={578} y={596} s={1.6} rot={-5} open={openK} />
          {openK > 0.4 && <Ripple x={578} y={500} startFrame={b1 + 150} color={P.faceHi} />}
          {/* 细胞膨压放大镜 */}
          <line x1={470} y1={640} x2={395} y2={662} stroke={P.cream} strokeWidth={4} opacity={0.6} strokeDasharray="6 8" />
          <circle cx={318} cy={668} r={104} fill={P.plate} stroke={P.cream} strokeWidth={5} />
          {[0, 1, 2].map((i) => {
            const g = 0.66 + pumpK * 0.42;
            return <circle key={i} cx={286 + i * 32} cy={668 + (i % 2) * 8} r={24 * g}
              fill="#63e2c6" stroke={P.vineD} strokeWidth={3.5} opacity={0.9} />;
          })}
          {[0, 1, 2, 3, 4].map((i) => {
            const p = clamp01((f - b1 - 10 - i * 14) / 46);
            return <circle key={i} cx={238 + p * 60} cy={622 + i * 22} r={6} fill={P.amber}
              opacity={p > 0 && p < 1 ? 1 : 0} />;
          })}
          <text x={318} y={794} textAnchor="middle" fill={P.cream} fontSize={27} fontFamily={FONT_HEAD}
            opacity={clamp01((f - b1) / 10)}>基部细胞泵入离子+水</text>
        </g>

        {/* 基因钟 */}
        <ClockFace x={940} y={310} r={98} k={clamp01(clockK)} hAng={hAng} mAng={f * 9} accent={P.faceHi} />
        <text x={940} y={450} textAnchor="middle" fill={P.cream} fontSize={30} fontFamily={FONT_HEAD}
          opacity={clamp01(clockK) * 0.9}>闹钟写在基因里</text>

        {/* 恒光房 */}
        <g transform={`translate(${(1 - clamp01(roomK)) * 60},0)`} opacity={clamp01(roomK)}>
          <rect x={1230} y={240} width={580} height={540} rx={40} fill={P.card} stroke={P.faceHi} strokeWidth={4} />
          <path d="M 1480 320 L 1520 280 L 1560 280 L 1600 320 Z" fill={P.amber} />
          <rect x={1514} y={320} width={12} height={26} fill={P.amber} />
          {[0, 1, 2].map((i) => (
            <line key={i} x1={1440 + i * 80} y1={352} x2={1420 + i * 100} y2={400} stroke={P.amber}
              strokeWidth={5} strokeLinecap="round" opacity={0.5 + 0.4 * Math.sin(f / 6 + i)} />
          ))}
          <text x={1520} y={425} textAnchor="middle" fill={P.cream} fontSize={34} fontFamily={FONT_HEAD} fontWeight={800}>恒定光照房间</text>
          <text x={1520} y={466} textAnchor="middle" fill={P.cream} fontSize={26} fontFamily={FONT_HEAD} opacity={0.7}>没有日出信号</text>
          <path d="M 1440 726 L 1454 668 L 1586 668 L 1600 726 Z" fill="#b96a44" />
          <rect x={1434} y={652} width={172} height={18} rx={9} fill={P.soilD} />
          <Vine x={1520} y={668} h={140} sway={0.5} seed={8} leaves={false} />
          <line x1={1534} y1={532} x2={1533} y2={602} stroke={P.vineD} strokeWidth={6} strokeLinecap="round" />
          <Bud x={1530} y={606} s={0.85} k={1 - roomOpenK} />
          <Trumpet x={1532} y={608} s={0.85} rot={4} open={roomOpenK} />
          <ClockFace x={1700} y={330} r={44} hAng={hAng} mAng={f * 9} accent={P.faceHi} />
        </g>
      </svg>
      <Stamp text="基因定的" x={1250} y={150} at={beat(0) + 40} color={P.faceHi} size={50} rot={-5} />
      <Chip text="膨压 = 细胞充气" x={140} y={530} at={b1 + 30} accent={P.amber} />
      <Stamp text="几十分钟全开！" x={780} y={620} at={b1 + 165} color={P.face} size={48} rot={5} />
      <Stamp text="照样踩点！" x={1290} y={820} at={b2 + 95} color={P.amber} size={50} rot={-4} />
    </FadeIn>
  );
};

/* ============ S2 计划下线：卷边→褪色→闭合清单打勾；浇水救不回；养分回收 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);

  const popL = spring({frame: f - 8, fps, config: {damping: 11, stiffness: 130}});
  const panelK = spring({frame: f - 20, fps, config: {damping: 12, stiffness: 130}});

  // 三阶段蔫：卷边 ~f226，褪色 ~f332，闭合 ~f438
  const wiltK = clamp01((f - 120) / 320);
  const hueK = clamp01(wiltK * 1.4);
  const chk = (th: number) => clamp01(spring({frame: f - th, fps, config: {damping: 10, stiffness: 180}}));

  // 浇水抢救失败
  const canK = clamp01((f - 296) / 16);
  const tilt = interpolate(f, [300, 326, 408, 432], [0, 40, 40, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const canOut = interpolate(f, [438, 468], [0, -430], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});

  // 养分回收：花瓣 → 花萼
  const recK = clamp01((f - b1 - 18) / 40);
  const calyxPulse = recK * (1 + 0.12 * Math.sin(f / 5));

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={2} name={sec.name} accent={P.coral} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <rect x={0} y={872} width={1920} height={64} fill={P.soil} />
        <rect x={0} y={872} width={1920} height={11} fill={P.soilD} />

        {/* 左：凋谢中的花 */}
        <g transform={`translate(0,${(1 - clamp01(popL)) * 40})`} opacity={clamp01(popL)}>
          <path d="M 330 872 L 350 796 L 510 796 L 530 872 Z" fill="#b96a44" />
          <rect x={322} y={780} width={216} height={22} rx={11} fill={P.soilD} />
          <Vine x={430} y={796} h={260} sway={0.7} seed={5} />
          <Trumpet x={454} y={578} s={1.6} rot={-8} open={1} wilt={wiltK} hue={hueK} />
          {/* 养分回收箭头：花瓣 → 花萼 */}
          {recK > 0 && (
            <g opacity={recK}>
              <path d="M 398 540 Q 368 596 408 630" stroke={P.amber} strokeWidth={6} fill="none"
                strokeDasharray="10 10" strokeLinecap="round" />
              <path d="M 398 640 L 412 626 L 418 646 Z" fill={P.amber} transform={`translate(0,0)`} />
              <path d="M 506 548 Q 530 600 496 632" stroke={P.amber} strokeWidth={6} fill="none"
                strokeDasharray="10 10" strokeLinecap="round" />
              <path d="M 504 646 L 494 626 L 514 632 Z" fill={P.amber} />
            </g>
          )}
          <ellipse cx={454} cy={592} rx={30 * calyxPulse} ry={14 * calyxPulse} fill={P.calyx} opacity={0.55 * recK} />
        </g>

        {/* 浇水壶 */}
        {canK > 0 && <WateringCan x={210 + canOut} y={330} s={1} tilt={tilt} k={canK} />}
        <g opacity={1 - clamp01((f - 492) / 20)}>
          <XMark x={450} y={470} r={62} at={418} />
        </g>

        {/* 右：谢幕流程清单 */}
        <g transform={`translate(${(1 - clamp01(panelK)) * 60},0)`} opacity={clamp01(panelK)}>
          <rect x={1180} y={170} width={640} height={430} rx={38} fill={P.card} stroke={P.faceHi} strokeWidth={4} />
          <text x={1500} y={238} textAnchor="middle" fill={P.cream} fontSize={38} fontFamily={FONT_HEAD} fontWeight={800}>
            花瓣细胞 · 程序性死亡
          </text>
          <line x1={1216} y1={330} x2={1216} y2={520} stroke={P.cream} strokeWidth={4} opacity={0.25} />
          <CheckRow x={1230} y={322} label="卷边" on={chk(226)} />
          <CheckRow x={1230} y={416} label="褪色" on={chk(332)} />
          <CheckRow x={1230} y={510} label="闭合萎缩" on={chk(438)} />
        </g>

        <text x={1500} y={680} textAnchor="middle" fill={P.cream} fontSize={30} fontFamily={FONT_HEAD}
          opacity={clamp01((f - 380) / 12)}>凋谢按顺序来 · 不是乱败</text>
      </svg>
      <Chip text="不是渴死，是程序" x={150} y={660} at={beat(0) + 60} accent={P.coral} />
      <Stamp text="补水救不回" x={180} y={560} at={424} color={P.sky} size={50} rot={-6} />
      <Chip text="养分回收进花萼" x={620} y={700} at={b1 + 40} accent={P.amber} />
      <Stamp text="谢幕也算账" x={1060} y={95} at={b1 + 80} color={P.calyx} size={48} rot={4} />
    </FadeIn>
  );
};

/* ============ S3 变色秀：花色蓝紫→粉→红紫，同步 pH 表盘 + 液泡剖面 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);

  const popL = spring({frame: f - 8, fps, config: {damping: 11, stiffness: 130}});
  const hueK = clamp01((f - 55) / 330);
  const wiltK = clamp01((f - 390) / 170) * 0.55;
  const ph = interpolate(hueK, [0, 1], [7.8, 5.0]);
  const gaugeK = spring({frame: f - b1 - 6, fps, config: {damping: 12, stiffness: 130}});
  const cellK = spring({frame: f - b1 - 40, fps, config: {damping: 12, stiffness: 130}});

  const swatches = [
    {x: 420, col: P.face, label: '清晨'},
    {x: 545, col: P.pink, label: '上午'},
    {x: 670, col: P.red, label: '中午'},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={3} name={sec.name} accent={P.pink} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：大花变色 */}
        <g transform={`translate(0,${(1 - clamp01(popL)) * 40})`} opacity={clamp01(popL)}>
          <path d="M 470 800 L 486 700 Q 500 640 522 596" stroke={P.vine} strokeWidth={8} fill="none" strokeLinecap="round" />
          <LeafHeart x={468} y={726} rot={-50} s={0.8} />
          <Bloom x={540} y={450} r={168} open={1} hue={hueK} wilt={wiltK} />
          {/* 时间弧箭头 */}
          <path d="M 330 260 A 240 240 0 0 1 770 262" stroke={P.cream} strokeWidth={5} fill="none"
            strokeDasharray="14 14" opacity={0.55} />
          <path d="M 756 246 L 786 262 L 754 280 Z" fill={P.cream} opacity={0.7} />
          <Sun x={330} y={225} r={34} opacity={0.95} />
          {/* 变色里程碑 */}
          {swatches.map((sw, i) => {
            const on = clamp01((f - (60 + i * 150)) / 20);
            return (
              <g key={i} opacity={0.35 + on * 0.65}>
                <circle cx={sw.x} cy={768} r={24} fill={sw.col} stroke={P.cream} strokeWidth={on > 0.5 ? 4 : 0} />
                <text x={sw.x} y={826} textAnchor="middle" fill={P.cream} fontSize={27} fontFamily={FONT_HEAD}>{sw.label}</text>
              </g>
            );
          })}
          <path d="M 450 768 L 514 768 M 576 768 L 640 768" stroke={P.cream} strokeWidth={4} opacity={0.5} />
        </g>

        {/* 右上：pH 表盘 */}
        <PhGauge x={1500} y={300} r={116} ph={ph} k={clamp01(gaugeK)} />
        <text x={1500} y={468} textAnchor="middle" fill={P.cream} fontSize={30} fontFamily={FONT_HEAD}
          opacity={clamp01(gaugeK) * 0.9}>花瓣液泡酸碱度</text>

        {/* 右下：细胞剖面卡 */}
        <g transform={`translate(${(1 - clamp01(cellK)) * 60},0)`} opacity={clamp01(cellK)}>
          <rect x={1150} y={510} width={650} height={330} rx={36} fill={P.card} stroke={P.faceHi} strokeWidth={4} />
          <text x={1475} y={566} textAnchor="middle" fill={P.cream} fontSize={32} fontFamily={FONT_HEAD} fontWeight={800}>液泡里的花青素</text>
          {/* 液泡 */}
          <circle cx={1420} cy={690} r={92} fill={faceCol(hueK)} stroke={P.cream} strokeWidth={4} />
          {Array.from({length: 9}).map((_, i) => (
            <circle key={i} cx={1420 + (rand(i, 81) - 0.5) * 120} cy={690 + (rand(i, 82) - 0.5) * 110}
              r={7 + rand(i, 83) * 5} fill={starCol(hueK)} opacity={0.85} />
          ))}
          {/* H+ 注入 */}
          {[0, 1, 2, 3].map((i) => {
            const p = clamp01((f - b1 - 90 - i * 26) / 50);
            if (p <= 0 || p >= 1) return null;
            return (
              <g key={i} opacity={p < 0.85 ? 1 : (1 - p) * 6}>
                <circle cx={1230 + p * 130} cy={600 + i * 52} r={17} fill={P.amber} />
                <text x={1230 + p * 130} y={608 + i * 52} textAnchor="middle" fill={P.ink}
                  fontSize={26} fontFamily={FONT_HEAD} fontWeight={900}>+</text>
              </g>
            );
          })}
          <text x={1652} y={620} fill={P.cream} fontSize={28} fontFamily={FONT_HEAD}
            opacity={clamp01((f - b1 - 80) / 14)}>酸度↑</text>
          <text x={1652} y={668} fill={P.red} fontSize={28} fontFamily={FONT_HEAD} fontWeight={700}
            opacity={clamp01((f - b1 - 100) / 14)}>颜色→红</text>
        </g>
      </svg>
      <Stamp text="顺便变个色" x={800} y={170} at={beat(0) + 70} color={P.pink} size={50} rot={-5} />
      <Chip text="花青素 · 遇酸变红" x={130} y={600} at={b1 + 20} accent={P.faceHi} />
      <Stamp text="天然 pH 试纸" x={1240} y={135} at={b2 + 10} color={P.amber} size={48} rot={4} />
      <Chip text="一场化学实验直播" x={760} y={620} at={b2 + 60} accent={P.pink} />
    </FadeIn>
  );
};

/* ============ S4 半天生意经：蜂蝶活跃峰 vs 开花窗口对齐；每日上新；天气微调 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);

  const chartK = spring({frame: f - 8, fps, config: {damping: 12, stiffness: 120}});
  const bellK = clamp01((f - 24) / 36);
  const winK = clamp01((f - beat(0) - 90) / 26);
  const beesOut = clamp01((f - 268) / 30);      // 中午歇工
  const cardK1 = spring({frame: f - b1 - 8, fps, config: {damping: 12, stiffness: 130}});
  const cardK2 = spring({frame: f - b2 - 8, fps, config: {damping: 12, stiffness: 130}});
  const bud2K = spring({frame: f - b1 - 60, fps, config: {damping: 9, stiffness: 160}});

  const TX = (h: number) => 280 + (h - 5) * (1380 / 14);   // 时刻→x
  const ticks = [5, 9, 12, 15, 19];

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={4} name={sec.name} accent={P.leaf} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* ---- 上：营业窗口对照图 ---- */}
        <g transform={`translate(0,${(1 - clamp01(chartK)) * 40})`} opacity={clamp01(chartK)}>
          {/* 蜂蝶活跃度钟形 */}
          <path d={`M ${TX(5)} 430 C 480 430 545 235 690 235 C 850 235 1040 430 ${TX(19)} 430 Z`}
            fill={P.leaf} opacity={0.32 * bellK} />
          <path d={`M ${TX(5)} 430 C 480 430 545 235 690 235 C 850 235 1040 430 ${TX(19)} 430`}
            stroke={P.leaf} strokeWidth={6} fill="none" opacity={bellK * 0.9} strokeLinecap="round" />
          <text x={690} y={206} textAnchor="middle" fill={P.leaf} fontSize={30} fontFamily={FONT_HEAD}
            fontWeight={700} opacity={bellK}>蜂蝶活跃度</text>
          {/* 时间轴 */}
          <line x1={TX(5)} y1={430} x2={TX(19)} y2={430} stroke={P.cream} strokeWidth={5} strokeLinecap="round" />
          <path d="M 1650 418 L 1682 430 L 1650 442 Z" fill={P.cream} />
          {ticks.map((t) => (
            <g key={t}>
              <line x1={TX(t)} y1={418} x2={TX(t)} y2={444} stroke={P.cream} strokeWidth={4} />
              <text x={TX(t)} y={486} textAnchor="middle" fill={P.cream} fontSize={27} fontFamily={FONT_HEAD} opacity={0.8}>
                {`${t}:00`}
              </text>
            </g>
          ))}
          {/* 开花窗口 */}
          <rect x={TX(5)} y={508} width={Math.max(0, (TX(12) - TX(5)) * winK)} height={20} rx={10} fill={P.faceHi} />
          {winK > 0.95 && <rect x={TX(12)} y={508} width={TX(19) - TX(12)} height={20} rx={10} fill={P.cream} opacity={0.18} />}
          <text x={TX(8.5)} y={560} textAnchor="middle" fill={P.faceHi} fontSize={28} fontFamily={FONT_HEAD}
            fontWeight={700} opacity={winK}>开花窗口 · 客户在线时段</text>
          {/* 蜜蜂：上午忙活，中午跑路 */}
          <Bee x={560} y={330} s={0.95} k={clamp01((f - 40) / 14) * (1 - beesOut)} />
          <Bee x={730} y={290} s={0.85} flip={-1} k={clamp01((f - 60) / 14) * (1 - beesOut)} />
          <Bee x={880} y={345} s={0.9} k={clamp01((f - 80) / 14) * (1 - beesOut)} />
          {/* 中午歇工的蜂 */}
          <Bee x={1090} y={360} s={0.9} k={beesOut} sleep={1} />
          <text x={1120} y={330} fill={P.cream} fontSize={34} fontFamily={FONT_HEAD}
            opacity={beesOut * 0.9}>Z z z</text>
          <Sun x={TX(12)} y={140} r={44} />
        </g>

        {/* ---- 左下：藤蔓每日上新 ---- */}
        <g transform={`translate(0,${(1 - clamp01(cardK1)) * 50})`} opacity={clamp01(cardK1)}>
          <rect x={140} y={586} width={760} height={280} rx={36} fill={P.card} stroke={P.leaf} strokeWidth={4} />
          <text x={520} y={640} textAnchor="middle" fill={P.cream} fontSize={32} fontFamily={FONT_HEAD} fontWeight={800}>藤蔓每天上新一批</text>
          <path d="M 220 730 C 360 706 560 716 830 700" stroke={P.vine} strokeWidth={7} fill="none" strokeLinecap="round" />
          <LeafHeart x={330} y={742} rot={-140} s={0.62} />
          <LeafHeart x={640} y={726} rot={150} s={0.6} />
          {/* 今天：已蔫 */}
          <line x1={300} y1={726} x2={300} y2={690} stroke={P.vineD} strokeWidth={5} />
          <Trumpet x={300} y={690} s={0.8} rot={14} open={1} wilt={0.92} hue={0.72} />
          <text x={300} y={800} textAnchor="middle" fill={P.cream} fontSize={27} fontFamily={FONT_HEAD} opacity={0.7}>今天·已收工</text>
          {/* 明天：高亮花苞 */}
          <line x1={520} y1={716} x2={520} y2={672} stroke={P.vineD} strokeWidth={5} />
          <circle cx={520} cy={640} r={44 * clamp01(bud2K)} fill="none" stroke={P.amber} strokeWidth={4}
            strokeDasharray="8 10" opacity={0.4 + 0.4 * Math.sin(f / 7)} />
          <Bud x={520} y={672} s={0.95} k={clamp01(bud2K)} />
          <text x={520} y={800} textAnchor="middle" fill={P.amber} fontSize={27} fontFamily={FONT_HEAD} fontWeight={700}
            opacity={clamp01(bud2K)}>明天·预备</text>
          {/* 后天：排队花苞 */}
          <line x1={740} y1={708} x2={740} y2={676} stroke={P.vineD} strokeWidth={5} />
          <Bud x={740} y={676} s={0.8} k={clamp01(bud2K) * 0.9} />
          <text x={740} y={800} textAnchor="middle" fill={P.cream} fontSize={27} fontFamily={FONT_HEAD}
            opacity={0.55 * clamp01(bud2K)}>后天·排队</text>
        </g>

        {/* ---- 右下：天气微调 ---- */}
        <g transform={`translate(${(1 - clamp01(cardK2)) * 60},0)`} opacity={clamp01(cardK2)}>
          <rect x={1010} y={586} width={810} height={280} rx={36} fill={P.card} stroke={P.sky} strokeWidth={4} />
          <text x={1415} y={640} textAnchor="middle" fill={P.cream} fontSize={32} fontFamily={FONT_HEAD} fontWeight={800}>天气只管微调</text>
          {/* 晴热天 */}
          <Sun x={1090} y={702} r={30} opacity={0.95} />
          <rect x={1140} y={692} width={290} height={20} rx={10} fill={P.coral} />
          <text x={1450} y={710} fill={P.cream} fontSize={28} fontFamily={FONT_HEAD}>≈7h · 正午下线</text>
          {/* 阴凉天 */}
          <Cloud x={1090} y={782} s={0.62} />
          <rect x={1140} y={772} width={460} height={20} rx={10} fill={P.leaf} />
          <text x={1620} y={790} fill={P.cream} fontSize={28} fontFamily={FONT_HEAD}>阴凉 · 撑到下午</text>
          <text x={1415} y={846} textAnchor="middle" fill={P.cream} fontSize={26} fontFamily={FONT_HEAD} opacity={0.6}>
            时钟定方向 · 温度定快慢
          </text>
        </g>
      </svg>
      <Stamp text="窗口卡得准" x={1330} y={180} at={beat(0) + 130} color={P.leaf} size={48} rot={-4} />
      <Chip text="便利店逻辑 · 高频上新" x={150} y={240} at={b1 + 30} accent={P.amber} />
      <Chip text="单朵短命 · 整株赚翻" x={1060} y={520} at={b1 + 90} accent={P.faceHi} />
    </FadeIn>
  );
};

/* ============ 结尾卡 ============ */
const EndCard = () => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 14}});
  const pop = spring({frame: f - 5, fps, config: {damping: 10, stiffness: 140}});
  return (
    <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
      <Backdrop top={P.bgDeep} bottom={P.bgDeep} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <ellipse cx={960} cy={1006} rx={1050} ry={100} fill="#1b2750" />
        <Moon x={1620} y={170} scale={0.9} opacity={0.9} />
        <g transform="translate(700,960)">
          <Vine x={0} y={0} h={250} rot={-16} sway={0.6} seed={9} />
          <Trumpet x={-18} y={-238} s={1.3} rot={-30} open={1} wilt={0.9} hue={0.75} />
        </g>
        <g transform="translate(1120,960)">
          <Vine x={0} y={0} h={220} rot={12} sway={0.6} seed={11} />
          <Bud x={14} y={-216} s={1.15} rot={8} k={clamp01(pop)} />
        </g>
        <circle cx={1134} cy={700} r={50 * clamp01(pop)} fill="none" stroke={P.amber} strokeWidth={4}
          strokeDasharray="8 12" opacity={0.5} />
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -180}}>
        <div style={{fontSize: 68, fontWeight: 800, color: C.text}}>只活半天，也要开得尽兴</div>
        <div style={{fontSize: 38, color: C.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E215 = ({data = tl}: {data?: any}) => {
  const {fps} = useVideoConfig();
  return (
    <AbsoluteFill style={{background: P.bg}}>
      <Sequence durationInFrames={Math.round(data.titleCardSec * fps)}>
        <TitleCard data={data} />
      </Sequence>

      {data.sections.map((sec: Sec, i: number) => {
        const Scene = SCENES[i];
        return (
          <Sequence key={i} from={Math.round(sec.start * fps)} durationInFrames={Math.round(sec.dur * fps)}>
            <Scene sec={sec} />
            <Audio src={staticFile(sec.audio)} />
          </Sequence>
        );
      })}

      <Sequence from={Math.round((data.totalSec - data.endCardSec) * fps)} durationInFrames={Math.round(data.endCardSec * fps)}>
        <EndCard />
      </Sequence>

      <Subtitles data={data} />
      <ProgressBar />
    </AbsoluteFill>
  );
};
