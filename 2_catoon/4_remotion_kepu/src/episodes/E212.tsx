import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01, rand} from '../lib/style';
import {Backdrop, Ripple, Stamp, SceneTag, Chip, Subtitles, ProgressBar, FadeIn} from '../lib/atoms';
import tl from '../timeline_212.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：椒园青绿 × 辣椒红 × 奶油盘（纯色平涂） ============ */
const P = {
  bg: '#23584a',       // 主底：深青绿
  bgLite: '#2b675a',   // 亮一号
  bgDeep: '#173e35',   // 深一号
  card: '#1b4438',     // 卡片底
  plate: '#f6ead0',    // 奶油圆盘
  plateD: '#e2cfa5',
  chili: '#e53428',    // 辣椒红
  chiliD: '#b21f1f',
  chiliDD: '#7e1414',
  chiliH: '#ff8a5c',
  calyx: '#4fae62',    // 萼片果柄绿
  calyxD: '#2f7d43',
  plac: '#fbf3dc',     // 胎座白
  placD: '#e0cf9f',
  seed: '#f5d98b',     // 籽
  seedD: '#cfa64f',
  fire: '#ffb03a',     // 辣椒素分子
  fireD: '#ff7b3d',
  gate: '#46b58f',     // TRPV1 受体
  gateD: '#2c8a6b',
  bird: '#5aa7e8',     // 鸟
  birdD: '#2f6fb0',
  birdBelly: '#e3f1ff',
  beak: '#ffb03a',
  mouse: '#a98a68',    // 哺乳动物
  mouseD: '#7d6248',
  mouseEar: '#e0ac93',
  skin: '#ffcf9e',
  skinD: '#e8a76f',
  amber: '#ffc94d',
  coral: '#ff7b6b',
  mint: '#8fe3c0',
  cream: '#f6ead0',
  ink: '#21352b',
  water: '#8fd0ff',
  milk: '#fdf6e3',
  spore: '#a9b47a',
  sporeD: '#7a8850',
};

/* ---------- 三次贝塞尔取点 ---------- */
const cubic = (p0: number[], p1: number[], p2: number[], p3: number[], t: number) => {
  const u = 1 - t;
  return [
    u * u * u * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t * t * t * p3[0],
    u * u * u * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t * t * t * p3[1],
  ];
};

/* ---------- 辣椒身路径：origin 在萼片端，尖朝右下 ---------- */
const podD =
  'M 0 0 C 80 -34 185 -32 240 8 C 258 22 272 44 278 66 ' +
  'C 240 58 200 52 160 48 C 105 41 45 30 8 16 C 0 12 -2 6 0 0 Z';
const podInnerD =
  'M 16 6 C 88 -22 176 -20 226 12 C 241 22 251 37 255 51 ' +
  'C 224 45 190 41 156 38 C 108 32 58 25 20 15 C 14 12 13 9 16 6 Z';

/* ---------- 整椒 ---------- */
const ChiliPod = ({x, y, s = 1, rot = 0, pop = 1, tint = P.chili}: {
  x: number; y: number; s?: number; rot?: number; pop?: number; tint?: string;
}) => {
  const k = Math.max(0.001, pop);
  return (
    <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s * k})`} opacity={pop > 0 ? 1 : 0}>
      {/* 果柄 */}
      <path d="M 0 -6 Q -22 -28 -32 -50" stroke={P.calyxD} strokeWidth={13} fill="none" strokeLinecap="round" />
      <path d={podD} fill={tint} />
      {/* 棱线 */}
      <path d="M 30 2 C 110 -18 190 -14 238 14" stroke={P.chiliD} strokeWidth={4} fill="none" opacity={0.4} />
      <path d="M 28 16 C 105 4 185 10 244 40" stroke={P.chiliD} strokeWidth={4} fill="none" opacity={0.26} />
      {/* 高光 */}
      <path d="M 42 -14 C 110 -28 170 -26 212 -6" stroke="#fff" strokeWidth={8} fill="none" strokeLinecap="round" opacity={0.35} />
      {/* 萼片 */}
      <path d="M -22 -14 C -8 -34 30 -36 46 -14 C 30 0 -6 0 -22 -14 Z" fill={P.calyx} />
      <path d="M -12 -8 C 0 -20 24 -22 34 -10 C 22 -1 -2 1 -12 -8 Z" fill={P.calyxD} opacity={0.65} />
    </g>
  );
};

/* ---------- 纵剖面：外壁 + 内腔 + 胎座条 + 一排籽 ---------- */
const ChiliCut = ({x, y, s = 1, rot = 0, pop = 1, placHi = 0, seeds = true}: {
  x: number; y: number; s?: number; rot?: number; pop?: number; placHi?: number; seeds?: boolean;
}) => {
  const f = useCurrentFrame();
  const k = Math.max(0.001, pop);
  const pulse = placHi * (0.55 + 0.45 * Math.sin(f / 7));
  return (
    <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s * k})`} opacity={pop > 0 ? 1 : 0}>
      <path d={podD} fill={P.chili} />
      <path d={podInnerD} fill={P.plac} />
      {/* 胎座条（白色隔膜） */}
      <rect x={40} y={-12} width={152} height={21} rx={10.5} fill={P.placD} />
      <rect x={46} y={-9} width={140} height={7} rx={3.5} fill="#fff" opacity={0.7} />
      {/* 籽挂在胎座下 */}
      {seeds && [0, 1, 2, 3, 4].map((i) => (
        <g key={i}>
          <line x1={62 + i * 27} y1={9} x2={62 + i * 27} y2={17} stroke={P.placD} strokeWidth={3} />
          <ellipse cx={62 + i * 27} cy={26 + (i % 2) * 3} rx={9} ry={10.5} fill={P.seed} stroke={P.seedD} strokeWidth={2.5} />
        </g>
      ))}
      <path d={podD} fill="none" stroke={P.chiliDD} strokeWidth={5} />
      {/* 胎座高亮圈 */}
      {pulse > 0 && (
        <ellipse cx={116} cy={0} rx={110} ry={34} fill="none" stroke={P.amber}
          strokeWidth={6} strokeDasharray="18 12" opacity={pulse} />
      )}
    </g>
  );
};

/* ---------- 辣椒素分子：六元环头 + 锯齿长链，像一把钥匙 ---------- */
const Caps = ({x, y, s = 1, rot = 0, o = 1}: {x: number; y: number; s?: number; rot?: number; o?: number}) => {
  const pts = Array.from({length: 6}, (_, i) => {
    const a = (Math.PI / 6) + i * (Math.PI / 3);
    return `${22 * Math.cos(a)},${22 * Math.sin(a)}`;
  }).join(' ');
  return (
    <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`} opacity={o}>
      <path d="M 20 8 L 44 20 L 60 10 L 80 24 L 96 13" stroke={P.fireD} strokeWidth={7}
        fill="none" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx={103} cy={11} r={7} fill={P.fire} />
      <polygon points={pts} fill={P.fire} stroke={P.fireD} strokeWidth={4} />
      <circle r={7} fill="#fff" opacity={0.55} />
    </g>
  );
};

/* ---------- 鸟 ---------- */
const Bird = ({x, y, s = 1, rot = 0, flap = 1, perch = false, o = 1}: {
  x: number; y: number; s?: number; rot?: number; flap?: number; perch?: boolean; o?: number;
}) => {
  const f = useCurrentFrame();
  const w = Math.sin(f * 0.55) * 20 * flap;
  return (
    <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`} opacity={o}>
      {/* 尾羽 */}
      <path d="M -38 -8 L -70 -20 L -62 -2 L -72 12 L -40 12 Z" fill={P.birdD} />
      {perch && (
        <g stroke={P.beak} strokeWidth={5} strokeLinecap="round">
          <line x1={-4} y1={34} x2={-4} y2={52} />
          <line x1={12} y1={34} x2={12} y2={52} />
        </g>
      )}
      <ellipse rx={44} ry={37} fill={P.bird} />
      <ellipse cx={9} cy={13} rx={28} ry={22} fill={P.birdBelly} />
      <ellipse cx={-6} cy={-4} rx={27} ry={15} fill={P.birdD} transform={`rotate(${-14 + w * 0.9})`} />
      <circle cx={24} cy={-11} r={7.5} fill="#fff" />
      <circle cx={26.5} cy={-11} r={4} fill={P.ink} />
      <path d="M 42 -13 L 64 -5 L 42 4 Z" fill={P.beak} />
    </g>
  );
};

/* ---------- 老鼠（哺乳动物代表），默认朝右 ---------- */
const Mouse = ({x, y, s = 1, flip = false, walk = 0, o = 1}: {
  x: number; y: number; s?: number; flip?: boolean; walk?: number; o?: number;
}) => {
  const f = useCurrentFrame();
  const legW = Math.sin(f * 0.9) * 7 * walk;
  return (
    <g transform={`translate(${x},${y}) scale(${flip ? -s : s},${s})`} opacity={o}>
      <path d="M -46 10 C -78 22 -96 8 -104 -14" stroke={P.mouseD} strokeWidth={7} fill="none" strokeLinecap="round" />
      <ellipse cx={-20 + legW * 0.4} cy={30} rx={16} ry={8} fill={P.mouseD} />
      <ellipse cx={26 - legW * 0.4} cy={30} rx={14} ry={7} fill={P.mouseD} />
      <ellipse rx={50} ry={32} fill={P.mouse} />
      <circle cx={42} cy={-8} r={24} fill={P.mouse} />
      <circle cx={30} cy={-30} r={13} fill={P.mouseD} />
      <circle cx={30} cy={-30} r={6.5} fill={P.mouseEar} />
      <circle cx={64} cy={-10} r={5} fill={P.coral} />
      <circle cx={48} cy={-16} r={4.5} fill={P.ink} />
      {[0, 1, 2].map((i) => (
        <line key={i} x1={58} y1={-6 + i * 4} x2={86} y2={-14 + i * 8} stroke={P.ink} strokeWidth={2} opacity={0.55} />
      ))}
    </g>
  );
};

/* ---------- TRPV1 受体门（∩ 形，中间是通道） ---------- */
const Gate = ({x, y, s = 1, dull = 1}: {x: number; y: number; s?: number; dull?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={dull}>
    <path
      d="M -62 70 L -62 -28 Q -62 -82 -6 -82 L 6 -82 Q 62 -82 62 -28 L 62 70 L 26 70
         L 26 -24 Q 26 -40 0 -40 Q -26 -40 -26 -24 L -26 70 Z"
      fill={P.gate} stroke={P.gateD} strokeWidth={5} />
    <line x1={-12} y1={-6} x2={12} y2={-6} stroke={P.gateD} strokeWidth={4} strokeLinecap="round" opacity={0.7} />
    <line x1={-12} y1={16} x2={12} y2={16} stroke={P.gateD} strokeWidth={4} strokeLinecap="round" opacity={0.7} />
  </g>
);

/* ---------- 报警弧（wifi 形向上） ---------- */
const AlarmArcs = ({x, y, k, col = P.coral}: {x: number; y: number; k: number; col?: string}) => {
  const f = useCurrentFrame();
  if (k <= 0) return null;
  return (
    <g transform={`translate(${x},${y})`} opacity={k}>
      {[52, 88, 124].map((r, i) => (
        <path key={r} d={`M ${-r} 0 A ${r} ${r} 0 0 1 ${r} 0`}
          stroke={col} strokeWidth={9} fill="none" strokeLinecap="round"
          opacity={0.55 + 0.45 * Math.sin(f / 5 + i * 1.4)} />
      ))}
    </g>
  );
};

/* ---------- 温度计（汞柱 level 0..1，mark 处有刻度） ---------- */
const Thermo = ({x, y, h = 300, level, mark = 0.62}: {x: number; y: number; h?: number; level: number; mark?: number}) => {
  const markY = 14 + (h - 30) * (1 - mark);
  return (
    <g transform={`translate(${x},${y})`}>
      <rect x={-17} y={0} width={34} height={h} rx={17} fill="#fdf8ea" stroke={P.plateD} strokeWidth={3} />
      <circle cy={h + 24} r={32} fill="#fdf8ea" stroke={P.plateD} strokeWidth={3} />
      <rect x={-7} y={14 + (h - 30) * (1 - level)} width={14} height={(h - 30) * level + 10} fill={P.coral} />
      <circle cy={h + 24} r={21} fill={P.coral} />
      <line x1={17} y1={markY} x2={46} y2={markY} stroke={P.cream} strokeWidth={5} />
      <text x={54} y={markY + 10} fill={P.cream} fontSize={30} fontFamily={FONT_HEAD} fontWeight={700}>43°C</text>
      {[0.25, 0.45, 0.82].map((m, i) => (
        <line key={i} x1={17} y1={14 + (h - 30) * (1 - m)} x2={34} y2={14 + (h - 30) * (1 - m)}
          stroke={P.cream} strokeWidth={3} opacity={0.5} />
      ))}
    </g>
  );
};

/* ---------- 火苗 ---------- */
const Flame = ({x, y, s = 1, o = 1}: {x: number; y: number; s?: number; o?: number}) => {
  const f = useCurrentFrame();
  const w = Math.sin(f * 0.5) * 4;
  return (
    <g transform={`translate(${x},${y}) scale(${s}) rotate(${w})`} opacity={o}>
      <path d="M 0 -34 C 16 -16 20 -4 20 8 C 20 26 10 36 0 36 C -10 36 -20 26 -20 8 C -20 -4 -16 -16 0 -34 Z" fill={P.fireD} />
      <path d="M 0 -14 C 9 -2 11 6 11 12 C 11 24 5 30 0 30 C -5 30 -11 24 -11 12 C -11 6 -9 -2 0 -14 Z" fill={P.fire} />
    </g>
  );
};

/* ---------- 四角星（内啡肽/火花） ---------- */
const Star4 = ({x, y, s = 1, o = 1, col = P.amber, rot = 0}: {
  x: number; y: number; s?: number; o?: number; col?: string; rot?: number;
}) => (
  <path transform={`translate(${x},${y}) scale(${s}) rotate(${rot})`} opacity={o}
    d="M 0 -18 L 5 -5 L 18 0 L 5 5 L 0 18 L -5 5 L -18 0 L -5 -5 Z" fill={col} />
);

/* ---------- 汗滴 ---------- */
const Sweat = ({x, y0, drop = 130, at, s = 1}: {x: number; y0: number; drop?: number; at: number; s?: number}) => {
  const f = useCurrentFrame();
  const t = f - at;
  if (t < 0 || t > 46) return null;
  const p = clamp01(t / 40);
  return (
    <path transform={`translate(${x},${y0 + drop * p}) scale(${s})`} opacity={0.95 * (1 - p * 0.6)}
      d="M 0 -16 C 9 -2 12 6 0 14 C -12 6 -9 -2 0 -16 Z" fill={P.water} />
  );
};

/* ---------- 粒子迸散 ---------- */
const Burst = ({x, y, at, n = 10, col = P.amber, spd = 1}: {
  x: number; y: number; at: number; n?: number; col?: string; spd?: number;
}) => {
  const f = useCurrentFrame();
  const t = f - at;
  if (t < 0) return null;
  const p = clamp01(t / 50);
  if (p >= 1) return null;
  return (
    <g>
      {Array.from({length: n}).map((_, i) => {
        const a = rand(i, 31) * Math.PI * 2;
        const d = p * (36 + rand(i, 32) * 78) * spd;
        return <circle key={i} cx={x + Math.cos(a) * d} cy={y + Math.sin(a) * d * 0.85 - p * 26}
          r={5 * (1 - p) + 2} fill={col} opacity={(1 - p) * 0.9} />;
      })}
    </g>
  );
};

/* ---------- 种子与嫩芽 ---------- */
const Seed = ({x, y, s = 1, o = 1}: {x: number; y: number; s?: number; o?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={o}>
    <ellipse rx={10} ry={12} fill={P.seed} stroke={P.seedD} strokeWidth={2.5} />
    <ellipse rx={3.5} ry={5} fill={P.seedD} opacity={0.7} />
  </g>
);

const Sprout = ({x, y, pop = 1}: {x: number; y: number; pop?: number}) => {
  const k = Math.max(0.001, pop);
  return (
    <g transform={`translate(${x},${y}) scale(${k})`} opacity={pop > 0 ? 1 : 0}>
      <line x1={0} y1={0} x2={0} y2={-30} stroke={P.calyxD} strokeWidth={6} strokeLinecap="round" />
      <ellipse cx={-14} cy={-34} rx={15} ry={8} fill={P.calyx} transform="rotate(-30 -14 -34)" />
      <ellipse cx={14} cy={-38} rx={15} ry={8} fill={P.calyx} transform="rotate(30 14 -38)" />
    </g>
  );
};

/* ---------- 杯子（白水 / 牛奶） ---------- */
const Cup = ({x, y, liquid, dotsMode, dotsK}: {
  x: number; y: number; liquid: 'water' | 'milk'; dotsMode: 'float' | 'sink'; dotsK: number;
}) => {
  const f = useCurrentFrame();
  const isWater = liquid === 'water';
  return (
    <g transform={`translate(${x},${y})`}>
      {/* 液体 */}
      <path d="M -72 -28 L 72 -28 L 60 92 L -60 92 Z"
        fill={isWater ? P.water : P.milk} opacity={isWater ? 0.55 : 0.95} />
      {/* 辣椒素点 */}
      {Array.from({length: 6}).map((_, i) => {
        const bx = -52 + i * 21 + rand(i, 44) * 8;
        if (dotsMode === 'float') {
          const bobY = -26 + Math.sin(f * 0.15 + i * 1.7) * 3;
          return <circle key={i} cx={bx} cy={bobY} r={8} fill={P.chili} opacity={dotsK} />;
        }
        const sy = -26 + dotsK * 55;
        return <circle key={i} cx={bx} cy={sy} r={8} fill={P.chili} opacity={0.95 * (1 - dotsK)} />;
      })}
      {/* 杯壁 */}
      <path d="M -86 -104 L 86 -104 L 68 96 L -68 96 Z" fill="none"
        stroke="#eaf5ee" strokeWidth={6} opacity={0.85} strokeLinejoin="round" />
    </g>
  );
};

/* ---------- 锁（受体比喻）；hole='slot' 时对不上圆头钥匙 ---------- */
const Lock = ({x, y, s = 1, hole = 'slot', flash = 0}: {
  x: number; y: number; s?: number; hole?: 'slot' | 'round'; flash?: number;
}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <path d="M -36 -34 L -36 -60 A 36 36 0 0 1 36 -60 L 36 -34" fill="none"
      stroke={P.plateD} strokeWidth={15} />
    <rect x={-60} y={-34} width={120} height={100} rx={18}
      fill={flash > 0 ? P.coral : P.amber} />
    {hole === 'slot'
      ? <rect x={-7} y={-4} width={14} height={44} rx={7} fill={P.ink} />
      : <g><circle cy={8} r={13} fill={P.ink} /><rect x={-6} y={8} width={12} height={26} fill={P.ink} /></g>}
  </g>
);

/* ---------- 霉菌孢子 ---------- */
const Spore = ({x, y, s = 1, o = 1}: {x: number; y: number; s?: number; o?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={o}>
    {Array.from({length: 8}).map((_, i) => {
      const a = (i / 8) * Math.PI * 2;
      return <line key={i} x1={Math.cos(a) * 13} y1={Math.sin(a) * 13}
        x2={Math.cos(a) * 22} y2={Math.sin(a) * 22} stroke={P.sporeD} strokeWidth={4} strokeLinecap="round" />;
    })}
    <circle r={15} fill={P.spore} />
    <circle cx={-4} cy={-4} r={4} fill={P.sporeD} opacity={0.7} />
    <circle cx={5} cy={5} r={3} fill={P.sporeD} opacity={0.7} />
  </g>
);

/* ---------- 盾牌 ---------- */
const Shield = ({x, y, s = 1, o = 1}: {x: number; y: number; s?: number; o?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={o}>
    <path d="M 0 -64 C 34 -52 56 -47 64 -44 C 64 8 50 44 0 68 C -50 44 -64 8 -64 -44 C -56 -47 -34 -52 0 -64 Z"
      fill={P.mint} opacity={0.28} />
    <path d="M 0 -64 C 34 -52 56 -47 64 -44 C 64 8 50 44 0 68 C -50 44 -64 8 -64 -44 C -56 -47 -34 -52 0 -64 Z"
      fill="none" stroke={P.mint} strokeWidth={7} />
    <path d="M -24 -2 L -6 16 L 28 -22" stroke={P.mint} strokeWidth={9} fill="none" strokeLinecap="round" strokeLinejoin="round" />
  </g>
);

/* ============ 标题卡 ============ */
const TitleCard = ({data}: {data: any}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 13, mass: 0.7}});
  const pop = spring({frame: f - 6, fps, config: {damping: 10, stiffness: 150}});
  const birdK = spring({frame: f - 14, fps, config: {damping: 9, stiffness: 160}});
  const o = interpolate(f, [0, 8], [0, 1]);
  return (
    <AbsoluteFill>
      <Backdrop top={P.bg} bottom={P.bg} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <circle cx={960} cy={360} r={240} fill={P.plate} />
        <g transform="translate(960,352) rotate(-14) scale(1.28) translate(-139,-10)">
          <ChiliPod x={0} y={0} pop={pop} />
        </g>
        <g transform={`translate(1150,${182 - 8 * (1 - clamp01(birdK))}) scale(${Math.max(0.001, birdK)})`}>
          <Bird x={0} y={0} s={0.62} perch flap={0.4} />
        </g>
        <Mouse x={742} y={462} s={0.5} o={clamp01(pop)} />
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 265}}>
          <div style={{color: P.amber, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 26}}>
            冷知识 · 第 {data.episode} 期
          </div>
          <div style={{color: C.text, fontSize: 92, fontWeight: 900, fontFamily: FONT_HEAD, letterSpacing: 4}}>
            {data.title}
          </div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ============ S0 假警报：温度计过 43°C + 辣椒素钥匙插入 TRPV1 → 假警报 → 出汗 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b0 = beat(0), b1 = beat(1), b2 = beat(2);

  const plateK = spring({frame: f - 6, fps, config: {damping: 12, stiffness: 130}});
  const gateK = spring({frame: f - 16, fps, config: {damping: 10, stiffness: 150}});

  // 温度计升温（cue1）
  const heat = clamp01((f - b1 - 15) / 50);
  const heatAlarm = clamp01((f - b1 - 60) / 14) * (1 - clamp01((f - b2 - 60) / 30));

  // 辣椒素钥匙：cue2 飞入 → 插入受体通道 → 报警
  const keyIn = clamp01((f - b2 - 8) / 42);
  const kx = interpolate(keyIn, [0, 1], [1400, 796]);
  const ky = interpolate(keyIn, [0, 1], [320, 350]);
  const dock = clamp01((f - b2 - 52) / 22);
  const capO = f < b2 + 8 ? 0 : 1 - clamp01((f - b2 - 150) / 20);
  const alarm2 = clamp01((f - b2 - 74) / 10);

  // 痛觉信号沿神经线跑
  const nerveK = alarm2 > 0 ? (f - b2 - 80) * 9 : 0;

  // 脸：出现 → 出汗脸红
  const faceK = spring({frame: f - b1 - 30, fps, config: {damping: 12, stiffness: 140}});
  const react = clamp01((f - b2 - 90) / 14);

  const GX = 790, GY = 505; // 受体中心
  return (
    <FadeIn>
      <Backdrop top={P.bgDeep} bottom={P.bgDeep} />
      <SceneTag index={0} name={sec.name} accent={P.amber} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <circle cx={GX} cy={GY - 10} r={400} fill={P.plate} opacity={clamp01(plateK)} />

        {/* 细胞膜：两排磷脂 */}
        <g opacity={clamp01(plateK)}>
          {Array.from({length: 19}).map((_, i) => {
            const x = 452 + i * 38;
            if (x > 730 && x < 850) return null;
            return (
              <g key={i}>
                <circle cx={x} cy={GY - 40} r={10} fill={P.gate} opacity={0.85} />
                <line x1={x} y1={GY - 30} x2={x} y2={GY - 2} stroke={P.gateD} strokeWidth={4} />
                <circle cx={x} cy={GY + 52} r={10} fill={P.gate} opacity={0.85} />
                <line x1={x} y1={GY + 42} x2={x} y2={GY + 14} stroke={P.gateD} strokeWidth={4} />
              </g>
            );
          })}
        </g>
        <text x={452} y={GY + 105} fill={P.ink} fontSize={28} fontFamily={FONT_HEAD} opacity={0.75}>细胞膜 · 痛觉神经末梢</text>

        {/* TRPV1 受体门 */}
        <g transform={`translate(0,${(1 - clamp01(gateK)) * 40})`} opacity={clamp01(gateK)}>
          <Gate x={GX} y={GY} s={1.15} />
        </g>
        <text x={608} y={GY - 78} textAnchor="end" fill={P.ink} fontSize={30} fontFamily={FONT_HEAD} fontWeight={700}>
          TRPV1 受体
        </text>

        {/* 报警弧 */}
        <AlarmArcs x={GX} y={GY - 118} k={Math.max(heatAlarm, alarm2)} />
        <AlarmArcs x={GX} y={GY - 118} k={alarm2 * 0.6} col={P.fire} />

        {/* 痛觉神经线：信号往下送 */}
        <path id="nerve" d={`M ${GX} ${GY + 70} C ${GX + 60} ${GY + 190} ${GX + 180} ${GY + 170} ${GX + 250} ${GY + 220}`}
          fill="none" stroke={P.gateD} strokeWidth={6} strokeDasharray="10 12" opacity={0.6} />
        {alarm2 > 0 && (
          <circle r={11} fill={P.coral}
            opacity={alarm2}>
            <animateMotion dur="1.6s" repeatCount="indefinite"
              path={`M ${GX} ${GY + 70} C ${GX + 60} ${GY + 190} ${GX + 180} ${GY + 170} ${GX + 250} ${GY + 220}`} />
          </circle>
        )}
        <g transform={`translate(${GX + 275},${GY + 232})`} opacity={alarm2}>
          <circle r={36} fill={P.coral} />
          <text y={12} textAnchor="middle" fill="#fff" fontSize={34} fontFamily={FONT_HEAD} fontWeight={900}>痛</text>
        </g>

        {/* 温度计：本职是测灼伤温度 */}
        <Thermo x={195} y={330} h={300} level={heat * 0.85} mark={0.62} />
        <Flame x={195} y={705} s={0.8} o={heat > 0 ? Math.min(1, heat * 3) * (1 - alarm2 * 0.3) : 0} />
        <text x={150} y={300} fill={P.cream} fontSize={28} fontFamily={FONT_HEAD} opacity={0.85}>本职：测灼伤</text>

        {/* 辣椒素钥匙飞入 */}
        {capO > 0 && (
          <Caps x={kx} y={ky + dock * 78} s={1.05} rot={-35 + keyIn * 15} o={capO} />
        )}
        {alarm2 > 0 && <Burst x={GX} y={GY - 60} at={b2 + 74} col={P.coral} n={12} />}
        <Ripple x={GX} y={GY} startFrame={b2 + 74} color={P.coral} />

        {/* 脸：灼痛反射 */}
        <g transform={`translate(1630,350) scale(${Math.max(0.001, clamp01(faceK))})`}>
          <circle r={102} fill={P.skin} />
          <circle cx={-34} cy={-12} r={9} fill={P.ink} />
          <circle cx={34} cy={-12} r={9} fill={P.ink} />
          {/* 闭眼忍痛（react 时闭眼） */}
          {react > 0.4 && (
            <g stroke={P.ink} strokeWidth={6} strokeLinecap="round">
              <path d="M -46 -14 L -22 -10 M -46 -8 L -22 -14" />
              <path d="M 22 -10 L 46 -14 M 22 -14 L 46 -8" />
            </g>
          )}
          <path d={react > 0.4 ? 'M -24 38 Q 0 22 24 38' : 'M -20 34 Q 0 44 20 34'}
            stroke={P.ink} strokeWidth={7} fill="none" strokeLinecap="round" />
          <ellipse cx={-56} cy={20} rx={18} ry={11} fill={P.coral} opacity={react * 0.75} />
          <ellipse cx={56} cy={20} rx={18} ry={11} fill={P.coral} opacity={react * 0.75} />
        </g>
        <Sweat x={1548} y0={300} at={b2 + 100} s={1.4} />
        <Sweat x={1705} y0={290} at={b2 + 112} s={1.15} />
        <Sweat x={1590} y0={268} at={b2 + 126} s={1.2} />
      </svg>
      <Stamp text="辣归痛觉管！" x={620} y={105} at={b0 + 32} color={P.coral} size={52} rot={5} />
      <Chip text="TRPV1 = 烫伤报警器" x={1310} y={560} at={b1 + 75} accent={P.mint} />
      <Stamp text="伪造的烧伤" x={1260} y={140} at={b2 + 96} color={P.fireD} size={50} rot={-6} />
      <Chip text="出汗脸红 = 烫伤反射" x={120} y={800} at={b2 + 130} accent={P.amber} />
    </FadeIn>
  );
};

/* ============ S1 弹药库：纵剖面胎座高亮 + 籽只沾光；右：水 vs 牛奶溶解对照 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b0 = beat(0), b1 = beat(1);

  const podK = spring({frame: f - 10, fps, config: {damping: 11, stiffness: 130}});
  const placHi = clamp01((f - 45) / 20);
  const panelK = spring({frame: f - b1 - 12, fps, config: {damping: 13, stiffness: 120}});
  const milkK = clamp01((f - b1 - 110) / 50);

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={1} name={sec.name} accent={P.amber} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <circle cx={560} cy={490} r={378} fill={P.plate} />
        {/* 纵剖辣椒 */}
        <ChiliCut x={330} y={468} s={2.0} rot={-4} pop={podK} placHi={placHi} />
        {/* 从胎座向外渗的辣椒素点 */}
        {Array.from({length: 8}).map((_, i) => {
          const st = 70 + i * 26;
          const t = f - st;
          if (t < 0 || t > 70) return null;
          const bx = 450 + rand(i, 51) * 330;
          const by = 462 + rand(i, 52) * 60;
          return <circle key={i} cx={bx + t * 0.9} cy={by - t * 1.1} r={7}
            fill={P.fireD} opacity={0.85 * (1 - t / 70)} />;
        })}
        {/* 胎座引线标注 */}
        {placHi > 0.4 && (
          <g opacity={placHi}>
            <line x1={500} y1={372} x2={538} y2={440} stroke={P.ink} strokeWidth={3.5} strokeDasharray="7 6" />
            <text x={500} y={352} textAnchor="middle" fill={P.ink} fontSize={34} fontFamily={FONT_HEAD} fontWeight={700}>
              胎座 · 火力大本营
            </text>
          </g>
        )}
        {/* 籽引线 */}
        {placHi > 0.4 && (
          <g opacity={placHi}>
            <line x1={510} y1={585} x2={480} y2={668} stroke={P.ink} strokeWidth={3.5} strokeDasharray="7 6" />
            <text x={470} y={705} textAnchor="middle" fill={P.ink} fontSize={30} fontFamily={FONT_HEAD} fontWeight={700}>
              籽：只沾光不产辣
            </text>
          </g>
        )}

        {/* 右：溶解对照面板 */}
        <g transform={`translate(0,${(1 - clamp01(panelK)) * 60})`} opacity={clamp01(panelK)}>
          <rect x={1150} y={270} width={660} height={570} rx={40} fill={P.card} stroke={P.mint} strokeWidth={4} />
          <text x={1480} y={340} textAnchor="middle" fill={P.cream} fontSize={36} fontFamily={FONT_HEAD} fontWeight={700}>
            辣椒素溶于谁？
          </text>
          {/* 水杯 */}
          <Cup x={1360} y={600} liquid="water" dotsMode="float" dotsK={clamp01((f - b1 - 55) / 20)} />
          <text x={1360} y={760} textAnchor="middle" fill={P.cream} fontSize={30} fontFamily={FONT_HEAD}>灌白水</text>
          <g transform={`translate(1452,480) scale(${Math.max(0.001, spring({frame: f - b1 - 80, fps, config: {damping: 9, stiffness: 170}}))})`}>
            <circle r={38} fill={P.coral} />
            <path d="M -16 -16 L 16 16 M 16 -16 L -16 16" stroke="#fff" strokeWidth={9} strokeLinecap="round" />
          </g>
          {/* 牛奶杯 */}
          <Cup x={1620} y={600} liquid="milk" dotsMode="sink" dotsK={milkK} />
          <text x={1620} y={760} textAnchor="middle" fill={P.cream} fontSize={30} fontFamily={FONT_HEAD}>冰牛奶</text>
          <g transform={`translate(1712,480) scale(${Math.max(0.001, spring({frame: f - b1 - 150, fps, config: {damping: 9, stiffness: 170}}))})`}>
            <circle r={38} fill={P.calyx} />
            <path d="M -18 0 L -5 15 L 20 -14" stroke="#fff" strokeWidth={9} fill="none" strokeLinecap="round" strokeLinejoin="round" />
          </g>
        </g>
      </svg>
      <Chip text="籽不产辣 · 只沾光" x={140} y={150} at={b0 + 50} accent={P.seed} />
      <Stamp text="胎座才是弹药库" x={140} y={248} at={b0 + 120} color={P.chili} size={48} rot={-5} />
      <Chip text="脂溶性 · 水冲不走" x={140} y={150 + 0} at={b1 + 20} accent={P.mint} />
      <Stamp text="脂肪带走辣椒素" x={1180} y={160} at={b1 + 165} color={P.amber} size={46} rot={4} />
    </FadeIn>
  );
};

/* ============ S2 门禁：椒株下——老鼠嚼碎种子被赶走 vs 鸟整吞快递播种 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);

  const plantK = spring({frame: f - 8, fps, config: {damping: 12, stiffness: 120}});

  // 老鼠：进 → 咬碎籽 → 逃跑
  const mIn = clamp01((f - b1 - 20) / 90);
  const mx = interpolate(mIn, [0, 1], [-140, 520]);
  const bite = f > b1 + 118;
  const flee = clamp01((f - b1 - 140) / 70);
  const mx2 = bite ? 520 + flee * flee * -700 : mx;
  const mouseWalk = (f > b1 + 20 && f < b1 + 110) || flee > 0 && flee < 1 ? 1 : 0;

  // 鸟：飞入悬停 → 飞走撒种
  const leg1 = clamp01((f - b1 - 160) / 70);
  const leg2 = clamp01((f - b1 - 245) / 75);
  const p1 = cubic([1980, 230], [1500, 120], [1150, 300], [820, 430], leg1);
  const p2 = cubic([820, 430], [1080, 220], [1420, 240], [1900, 430], leg2);
  const bx = leg2 > 0 ? p2[0] : p1[0];
  const by = leg2 > 0 ? p2[1] : p1[1];
  const birdO = leg1 > 0 && leg2 < 1 ? 1 : 0;

  // 撒种：leg2 中段丢三颗
  const seedDrops = [0.28, 0.52, 0.76].map((tt, i) => {
    const pos = cubic([820, 430], [1080, 220], [1420, 240], [1900, 430], tt);
    const dropAt = b1 + 245 + Math.round(tt * 75);
    const t = f - dropAt;
    const sy = t < 0 ? pos[1] : Math.min(830, pos[1] + (t / 55) * (830 - pos[1]) * (t / 55));
    return {x: pos[0], y: sy, landed: t > 55, at: dropAt + 55, i};
  });

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={2} name={sec.name} accent={P.amber} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 地面 */}
        <ellipse cx={960} cy={905} rx={1150} ry={58} fill={P.bgDeep} />

        {/* 椒株：茎 + 叶 + 两只垂椒 */}
        <g transform={`translate(0,${(1 - clamp01(plantK)) * 60})`} opacity={clamp01(plantK)}>
          <path d="M 700 862 C 695 720 665 570 640 470" stroke={P.calyxD} strokeWidth={15} fill="none" strokeLinecap="round" />
          <path d="M 700 700 C 740 640 790 600 840 575" stroke={P.calyxD} strokeWidth={11} fill="none" strokeLinecap="round" />
          <ellipse cx={600} cy={600} rx={66} ry={26} fill={P.calyx} transform="rotate(-28 600 600)" />
          <ellipse cx={815} cy={648} rx={58} ry={22} fill={P.calyx} transform="rotate(18 815 648)" />
          <ellipse cx={655} cy={430} rx={52} ry={20} fill={P.calyx} transform="rotate(-12 655 430)" />
          <ChiliPod x={642} y={478} rot={88} s={1.22} pop={clamp01(plantK)} />
          <ChiliPod x={838} y={578} rot={62} s={0.95} pop={clamp01(plantK)} />
        </g>

        {/* 老鼠 */}
        {(mIn > 0 && flee < 1) && (
          <g>
            <Mouse x={mx2} y={795} s={1.15} flip={flee > 0.02} walk={mouseWalk} />
            {/* 咬碎的籽 */}
            {bite && flee < 0.35 && (
              <g transform="translate(582,782)" opacity={1 - flee * 2.4}>
                <path d="M -14 -8 L -2 6 L -14 18 Z" fill={P.seed} stroke={P.seedD} strokeWidth={2.5} />
                <path d="M 14 -8 L 2 6 L 14 18 Z" fill={P.seed} stroke={P.seedD} strokeWidth={2.5} />
              </g>
            )}
          </g>
        )}
        <Burst x={582} y={782} at={b1 + 118} col={P.seed} n={9} spd={0.7} />

        {/* 鸟的拖尾 + 本体 */}
        {birdO > 0 && leg2 < 0.98 && (
          <g opacity={birdO}>
            {Array.from({length: 5}).map((_, i) => {
              const tt = Math.max(0, (leg2 > 0 ? leg2 : leg1) - 0.04 - i * 0.05);
              const pp = leg2 > 0
                ? cubic([820, 430], [1080, 220], [1420, 240], [1900, 430], tt)
                : cubic([1980, 230], [1500, 120], [1150, 300], [820, 430], tt);
              return <circle key={i} cx={pp[0]} cy={pp[1]} r={5.5 - i * 0.8} fill={P.cream} opacity={0.4 - i * 0.06} />;
            })}
            <Bird x={bx} y={by} s={1.0} rot={Math.sin(f * 0.3) * 6} flap={1} />
          </g>
        )}

        {/* 落下的种子 → 发芽 */}
        {seedDrops.map((sd, i) => (
          <g key={i}>
            {!sd.landed && f > sd.at - 55 && <Seed x={sd.x} y={sd.y} s={1.1} />}
            {sd.landed && (
              <g>
                <Seed x={sd.x} y={836} s={1} />
                <Sprout x={sd.x} y={828} pop={clamp01((f - sd.at - 15) / 20)} />
              </g>
            )}
          </g>
        ))}

        {/* 客户名单牌 */}
        <g opacity={clamp01((f - b2 - 10) / 16)}>
          <rect x={1330} y={120} width={480} height={150} rx={24} fill={P.card} stroke={P.mint} strokeWidth={4} />
          <text x={1570} y={172} textAnchor="middle" fill={P.coral} fontSize={32} fontFamily={FONT_HEAD} fontWeight={700}>
            哺乳动物 ✗ 嚼碎种子
          </text>
          <text x={1570} y={228} textAnchor="middle" fill={P.mint} fontSize={32} fontFamily={FONT_HEAD} fontWeight={700}>
            鸟类 ✓ 无损快递员
          </text>
        </g>
      </svg>
      <Chip text="同一颗果 · 两种待遇" x={140} y={150} at={40} accent={P.amber} />
      <Stamp text="嚼碎种子·差评" x={250} y={580} at={b1 + 135} color={P.coral} size={48} rot={-7} />
      <Stamp text="整吞·无损快递" x={620} y={100} at={b1 + 280} color={P.calyx} size={46} rot={5} />
      <Chip text="野外实验：真的管用" x={140} y={700} at={b2 + 25} accent={P.mint} />
    </FadeIn>
  );
};

/* ============ S3 空枪：鸟锁孔对不上 + 辣椒压根没受体 + 兼职防腐盾 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);

  const cardK = spring({frame: f - 8, fps, config: {damping: 13, stiffness: 120}});

  // 钥匙①：撞鸟锁 → 弹开
  const k1in = clamp01((f - 30) / 50);
  const k1x = interpolate(k1in, [0, 1], [420, 420]);
  const k1y = interpolate(k1in, [0, 1], [-60, 275]);
  const bounce = clamp01((f - 92) / 14);
  const bounceBack = clamp01((f - 96) / 40);
  const k1yFinal = k1y - bounceBack * 90;
  const k1rot = -90 + Math.sin(Math.max(0, f - 90) * 0.9) * 18 * (1 - bounceBack);
  const k1o = k1in > 0 ? 1 - clamp01((f - 140) / 25) : 0;

  // 钥匙②：滑向辣椒 → 滑走
  const k2in = clamp01((f - b1 - 20) / 55);
  const k2x = interpolate(k2in, [0, 1], [1850, 1470]);
  const k2y = interpolate(k2in, [0, 1], [640, 330]);
  const slideOff = clamp01((f - b1 - 82) / 40);
  const k2o = k2in > 0 ? 1 - slideOff : 0;

  // 辣椒无受体：虚线幽灵 + ✗
  const ghostK = clamp01((f - b1 - 95) / 18);
  const noK = spring({frame: f - b1 - 110, fps, config: {damping: 9, stiffness: 170}});

  // 防腐带
  const bandK = spring({frame: f - b2 - 8, fps, config: {damping: 13, stiffness: 110}});

  return (
    <FadeIn>
      <Backdrop top={P.bgDeep} bottom={P.bgDeep} />
      <SceneTag index={3} name={sec.name} accent={P.mint} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左卡：鸟的受体 */}
        <g transform={`translate(0,${(1 - clamp01(cardK)) * 50})`} opacity={clamp01(cardK)}>
          <rect x={140} y={170} width={690} height={360} rx={36} fill={P.card} stroke={P.mint} strokeWidth={4} />
          <text x={485} y={232} textAnchor="middle" fill={P.cream} fontSize={34} fontFamily={FONT_HEAD} fontWeight={700}>
            鸟 · 同款 TRPV1
          </text>
          <Lock x={360} y={400} s={1.15} hole="slot" flash={bounce * Math.max(0, 1 - (f - 125) / 30)} />
          <text x={485} y={505} textAnchor="middle" fill={P.mint} fontSize={28} fontFamily={FONT_HEAD}>
            锁孔齿形对不上
          </text>
          {/* 淡定吃椒的鸟 */}
          <Bird x={660} y={420} s={0.78} flap={0.15} perch />
          {[0, 1, 2].map((i) => (
            <circle key={i} cx={700 + i * 16} cy={468 - (i % 2) * 8} r={7} fill={P.chili} />
          ))}
        </g>
        {/* 弹开的钥匙① + 火花 */}
        {k1o > 0 && (
          <Caps x={k1x} y={k1yFinal} s={0.95} rot={k1rot} o={k1o} />
        )}
        {bounce > 0 && (
          <g opacity={bounce}>
            <Star4 x={330} y={310} s={1.2} col={P.cream} />
            <Star4 x={430} y={290} s={0.9} col={P.cream} rot={30} />
            <Star4 x={390} y={352} s={0.7} col={P.fire} rot={-20} />
          </g>
        )}

        {/* 右卡：辣椒自己 */}
        <g transform={`translate(0,${(1 - clamp01(cardK)) * 50})`} opacity={clamp01(cardK)}>
          <rect x={1090} y={170} width={690} height={360} rx={36} fill={P.card} stroke={P.mint} strokeWidth={4} />
          <text x={1435} y={232} textAnchor="middle" fill={P.cream} fontSize={34} fontFamily={FONT_HEAD} fontWeight={700}>
            辣椒 · 植物没有受体
          </text>
          <ChiliPod x={1290} y={400} s={1.25} rot={-10} pop={clamp01(cardK)} />
          {/* 幽灵受体虚线 + ✗ */}
          <g opacity={ghostK}>
            <g transform="translate(1200,430) scale(0.75)">
              <path
                d="M -62 70 L -62 -28 Q -62 -82 -6 -82 L 6 -82 Q 62 -82 62 -28 L 62 70 L 26 70
                   L 26 -24 Q 26 -40 0 -40 Q -26 -40 -26 -24 L -26 70 Z"
                fill="none" stroke={P.mint} strokeWidth={6} strokeDasharray="12 10" opacity={0.7} />
            </g>
          </g>
          <g transform={`translate(1200,430) scale(${Math.max(0.001, noK)})`}>
            <circle r={42} fill={P.coral} />
            <path d="M -18 -18 L 18 18 M 18 -18 L -18 18" stroke="#fff" strokeWidth={10} strokeLinecap="round" />
          </g>
          <text x={1435} y={505} textAnchor="middle" fill={P.mint} fontSize={28} fontFamily={FONT_HEAD}>
            想辣自己都没器官
          </text>
        </g>
        {/* 滑走的钥匙② */}
        {k2o > 0 && (
          <Caps x={k2x + slideOff * 130} y={k2y + slideOff * 160} s={0.95} rot={-160 + slideOff * 60} o={k2o} />
        )}

        {/* 底带：兼职防腐剂 */}
        <g transform={`translate(0,${(1 - clamp01(bandK)) * 60})`} opacity={clamp01(bandK)}>
          <rect x={140} y={580} width={1640} height={270} rx={36} fill={P.card} />
          <circle cx={500} cy={715} r={168} fill="none" stroke={P.mint} strokeWidth={6}
            strokeDasharray="20 14" opacity={0.8} />
          <circle cx={500} cy={715} r={168} fill={P.mint} opacity={0.08} />
          <ChiliPod x={330} y={712} s={1.35} rot={-8} pop={clamp01(bandK)} />
          <Shield x={500} y={700} s={0.85} o={clamp01((f - b2 - 40) / 16)} />
          <text x={1130} y={648} fill={P.cream} fontSize={36} fontFamily={FONT_HEAD} fontWeight={700}>
            兼职防腐剂
          </text>
          <text x={1130} y={698} fill={P.mint} fontSize={28} fontFamily={FONT_HEAD}>
            抑制真菌 · 保护果实
          </text>
          {/* 弹开的孢子：沿下方弹道飘向护盾再弹回 */}
          {Array.from({length: 6}).map((_, i) => {
            const st = b2 + 30 + i * 14;
            const t = f - st;
            if (t < 0) return null;
            const inP = clamp01(t / 60);
            const bounceP = clamp01((t - 60) / 45);
            const sx = 1750 - inP * 1000 + bounceP * 520;
            const sy = 745 + i * 15 - inP * 10 + bounceP * 55;
            return <Spore key={i} x={sx} y={sy} s={0.9 + rand(i, 3) * 0.4} o={1 - bounceP} />;
          })}
        </g>
      </svg>
      <Chip text="钥匙开不了鸟的锁" x={140} y={100} at={55} accent={P.amber} />
      <Stamp text="弹开！" x={560} y={120} at={96} color={P.coral} size={46} rot={-6} />
      <Stamp text="压根没受体" x={1460} y={110} at={b1 + 120} color={P.amber} size={46} rot={5} />
      <Stamp text="一物三用" x={1630} y={790} at={b2 + 110} color={P.mint} size={44} rot={-4} />
    </FadeIn>
  );
};

/* ============ S4 计划外：人类良性受虐 + 受体脱敏训练 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);

  const faceK = spring({frame: f - 10, fps, config: {damping: 12, stiffness: 120}});
  const biteAt = b1 + 40;
  const bite = f >= biteAt;
  const pain = clamp01((f - biteAt) / 8) * (1 - clamp01((f - biteAt - 55) / 25));
  const joy = clamp01((f - biteAt - 70) / 20);

  // 辣椒送入口
  const feed = clamp01((f - b1 - 5) / 35);
  const podX = interpolate(feed, [0, 1], [-60, 320]);
  const podY = interpolate(feed, [0, 1], [880, 822]);

  // 右侧耐受训练
  const panelK = spring({frame: f - b2 - 8, fps, config: {damping: 13, stiffness: 120}});
  const hits = [b2 + 40, b2 + 72, b2 + 104];
  const hitCount = hits.filter((h) => f > h).length;
  const dull = 1 - hitCount * 0.2;
  const gauge = clamp01((f - b2 - 120) / 90);

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={4} name={sec.name} accent={P.coral} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 大脸 */}
        <g transform={`translate(540,520) scale(${Math.max(0.001, clamp01(faceK))})`}>
          <circle r={210} fill={P.skin} />
          {/* 头发 */}
          <path d="M -195 -60 A 208 208 0 0 1 195 -60 L 195 -95 A 208 208 0 0 0 -195 -95 Z" fill={P.ink} opacity={0.85} />
          {/* 眼：痛时紧闭 >_< */}
          {pain > 0.35 ? (
            <g stroke={P.ink} strokeWidth={9} strokeLinecap="round">
              <path d="M -95 -30 L -50 -14 M -95 -6 L -50 -26" />
              <path d="M 50 -14 L 95 -30 M 50 -26 L 95 -6" />
            </g>
          ) : (
            <g>
              <circle cx={-72} cy={-22} r={13} fill={P.ink} />
              <circle cx={72} cy={-22} r={13} fill={P.ink} />
              {joy > 0.5 && (
                <g stroke={P.ink} strokeWidth={8} strokeLinecap="round">
                  <path d="M -90 -24 Q -72 -40 -54 -24" fill="none" />
                  <path d="M 54 -24 Q 72 -40 90 -24" fill="none" />
                </g>
              )}
            </g>
          )}
          {/* 眉 */}
          <path d={joy > 0.5 ? 'M -100 -62 Q -72 -74 -44 -62' : 'M -100 -58 Q -72 -70 -44 -58'}
            stroke={P.ink} strokeWidth={8} fill="none" strokeLinecap="round" />
          <path d={joy > 0.5 ? 'M 44 -62 Q 72 -74 100 -62' : 'M 44 -58 Q 72 -70 100 -58'}
            stroke={P.ink} strokeWidth={8} fill="none" strokeLinecap="round" />
          {/* 嘴：咬合 */}
          {bite ? (
            <ellipse cx={10} cy={95} rx={joy > 0.5 ? 60 : 44} ry={joy > 0.5 ? 34 : 26} fill={P.chiliDD} />
          ) : (
            <ellipse cx={10} cy={95} rx={26} ry={16} fill={P.chiliDD} />
          )}
          {/* 腮红 */}
          <ellipse cx={-118} cy={60} rx={30} ry={18} fill={P.coral} opacity={Math.max(pain, joy) * 0.7} />
          <ellipse cx={138} cy={60} rx={30} ry={18} fill={P.coral} opacity={Math.max(pain, joy) * 0.7} />
        </g>
        {/* 咬下的辣椒 */}
        <g transform={`translate(${podX},${podY}) rotate(-32) scale(1.15)`} opacity={feed > 0 ? 1 : 0}>
          <ChiliPod x={0} y={0} pop={1} />
        </g>
        {/* 痛火花 + 汗 */}
        {bite && <Burst x={560} y={620} at={biteAt} col={P.coral} n={12} spd={0.9} />}
        <Sweat x={365} y0={430} at={biteAt + 12} s={1.5} />
        <Sweat x={715} y0={415} at={biteAt + 24} s={1.3} />
        {/* 内啡肽金星 */}
        {[0, 1, 2, 3, 4].map((i) => {
          const at = biteAt + 85 + i * 12;
          const o = clamp01((f - at) / 12) * (0.6 + 0.4 * Math.sin(f / 6 + i));
          const a = -Math.PI * 0.85 + i * 0.42;
          return <Star4 key={i} x={540 + Math.cos(a) * 265} y={520 + Math.sin(a) * 255}
            s={1.1 + rand(i, 5) * 0.5} o={o} />;
        })}

        {/* 右：受体耐受训练面板 */}
        <g transform={`translate(0,${(1 - clamp01(panelK)) * 60})`} opacity={clamp01(panelK)}>
          <rect x={1100} y={290} width={700} height={540} rx={36} fill={P.card} stroke={P.mint} strokeWidth={4} />
          <text x={1450} y={352} textAnchor="middle" fill={P.cream} fontSize={34} fontFamily={FONT_HEAD} fontWeight={700}>
            受体的耐受训练
          </text>
          {/* 迷你膜 + 受体 */}
          {Array.from({length: 9}).map((_, i) => (
            <g key={i}>
              <circle cx={1180 + i * 68} cy={455} r={8} fill={P.gate} opacity={0.6} />
              <circle cx={1180 + i * 68} cy={505} r={8} fill={P.gate} opacity={0.6} />
            </g>
          ))}
          <Gate x={1450} y={482} s={0.82} dull={Math.max(0.42, dull)} />
          {/* 三次撞击 */}
          {hits.map((h, i) => {
            const t = clamp01((f - h + 24) / 24);
            const gone = f > h + 18;
            if (t <= 0 || gone) return null;
            const kx2 = interpolate(t, [0, 1], [1860, 1490]);
            return <Caps key={i} x={kx2} y={478} s={0.72} rot={-160} o={1} />;
          })}
          {hits.map((h, i) => (
            <Burst key={i} x={1490} y={478} at={h} col={P.fire} n={7} spd={0.55} />
          ))}
          <text x={1450} y={590} textAnchor="middle" fill={P.mint} fontSize={27} fontFamily={FONT_HEAD} opacity={0.9}>
            受体被反复脱敏 → 变迟钝
          </text>
          {/* 阈值条 */}
          <text x={1180} y={672} fill={P.cream} fontSize={28} fontFamily={FONT_HEAD}>痛觉阈值</text>
          <rect x={1180} y={692} width={540} height={34} rx={17} fill="#0f2f27" stroke={P.mint} strokeWidth={3} />
          <rect x={1180} y={692} width={540 * Math.max(0.06, gauge * 0.9)} height={34} rx={17} fill={P.amber} />
          <text x={1190} y={762} fill={P.cream} fontSize={24} fontFamily={FONT_HEAD} opacity={0.7}>新手</text>
          <text x={1698} y={762} textAnchor="end" fill={P.amber} fontSize={24} fontFamily={FONT_HEAD} fontWeight={700}>老吃辣的人</text>
        </g>
      </svg>
      <Stamp text="良性受虐" x={170} y={160} at={biteAt + 20} color={P.coral} size={54} rot={-6} />
      <Chip text="痛完有内啡肽兜底" x={620} y={150} at={biteAt + 110} accent={P.amber} />
      <Stamp text="受体被练钝了" x={1180} y={150} at={b2 + 140} color={P.amber} size={48} rot={5} />
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
        <ellipse cx={960} cy={1018} rx={1050} ry={110} fill="#0f2b24" />
        <circle cx={1040} cy={900} r={185} fill={P.plate} />
        <g transform="translate(905,912) rotate(-12) scale(1.15)">
          <ChiliPod x={0} y={0} pop={pop} />
        </g>
        <g transform={`translate(1155,835) scale(${Math.max(0.001, pop)})`}>
          <Bird x={0} y={0} s={0.62} perch flap={0.3} />
        </g>
        <Mouse x={750} y={955} s={0.55} o={clamp01(pop)} />
        <Sprout x={1330} y={950} pop={clamp01(pop)} />
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -230}}>
        <div style={{fontSize: 70, fontWeight: 800, color: C.text}}>每一口辣，都是植物的精准外交</div>
        <div style={{fontSize: 38, color: C.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E212 = ({data = tl}: {data?: any}) => {
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
