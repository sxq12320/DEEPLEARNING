import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01, rand} from '../lib/style';
import {Backdrop, Stamp, SceneTag, Chip, Subtitles, ProgressBar, FadeIn, Sun, Moon, Stars, Bug} from '../lib/atoms';
import tl from '../timeline_217.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：深夜蓝 × 荧光青绿 × 蘑菇暖橙 ============ */
const P = {
  bg: '#0e1734',      // 主底：深夜蓝
  bgDeep: '#090f22',
  bgLite: '#16224a',
  card: '#1b2954',
  panel: '#0a1128',   // 放大窗/卡片内底
  glow: '#7dffb0',    // 荧光青绿（发光本体）
  glowMid: '#3fe08c',
  glowDeep: '#1d9e62',
  cap: '#e8805a',     // 菌盖暖橙
  capD: '#b85a3c',
  stem: '#f2ddba',    // 菌柄米白
  stemD: '#caa87e',
  wood: '#8a5a33',    // 腐木
  woodD: '#5f3c20',
  woodL: '#b08155',
  cream: '#f6efdc',
  ink: '#171008',
  muted: '#9aa7d0',
  amber: '#ffc94d',
  violet: '#9b7ede',  // 荧光素酶
  violetD: '#5a3d8f',
  red: '#ff6b6b',
  plastic: '#8f9db8', // LED 假蘑菇
  plasticD: '#5d6b8a',
  night: '#070b18',
};

/* ---------- 荧光放射短线（平涂描边，非渐变辉光） ---------- */
const GlowRays = ({x, y, r0 = 40, r1 = 95, n = 10, k = 1, col = P.glow, seed = 0, lw = 6, arc = Math.PI * 2}: {
  x: number; y: number; r0?: number; r1?: number; n?: number; k?: number; col?: string; seed?: number; lw?: number; arc?: number;
}) => {
  const f = useCurrentFrame();
  return (
    <g transform={`translate(${x},${y})`} opacity={k}>
      {Array.from({length: n}).map((_, i) => {
        const a = -Math.PI / 2 - arc / 2 + (i / Math.max(1, n - 1)) * arc + seed;
        const L = r0 + (r1 - r0) * (0.6 + 0.4 * Math.sin(f / 7 + i * 1.9 + seed));
        return <line key={i} x1={Math.cos(a) * r0} y1={Math.sin(a) * r0}
          x2={Math.cos(a) * L} y2={Math.sin(a) * L}
          stroke={col} strokeWidth={lw} strokeLinecap="round" />;
      })}
    </g>
  );
};

/* ---------- 荧光微粒：半径呼吸的小圆点 ---------- */
const GlowDots = ({cx, cy, rx, ry, n = 10, k = 1, r = 5, col = P.glow, seed = 3}: {
  cx: number; cy: number; rx: number; ry: number; n?: number; k?: number; r?: number; col?: string; seed?: number;
}) => {
  const f = useCurrentFrame();
  return (
    <g opacity={k}>
      {Array.from({length: n}).map((_, i) => {
        const tw = 0.35 + 0.65 * Math.abs(Math.sin(f / 11 + i * 2.4 + seed));
        return <circle key={i} cx={cx + (rand(i, seed) - 0.5) * 2 * rx} cy={cy + (rand(i, seed + 9) - 0.5) * 2 * ry}
          r={r * tw} fill={col} opacity={0.3 + tw * 0.65} />;
      })}
    </g>
  );
};

/* ---------- 发光蘑菇：柄+菌褶+伞盖；gill=菌褶亮度 aura=整朵光晕 ---------- */
const Mushroom = ({x, y, s = 1, rot = 0, gill = 1, aura = 0, cap = P.cap, capD = P.capD, dim = 0}: {
  x: number; y: number; s?: number; rot?: number; gill?: number; aura?: number; cap?: string; capD?: string; dim?: number;
}) => {
  const f = useCurrentFrame();
  const pulse = 0.8 + 0.2 * Math.sin(f / 9);
  const g = gill * pulse;
  const stemC = dim > 0.5 ? '#6b5b44' : P.stem;
  const capC = dim > 0.5 ? '#7a5a48' : cap;
  const capDC = dim > 0.5 ? '#54402f' : capD;
  return (
    <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`}>
      {aura > 0.03 && <GlowRays x={0} y={-96} r0={72} r1={118} n={9} k={aura * pulse} lw={5} arc={Math.PI * 1.5} />}
      <path d="M -17 0 C -21 -44 -16 -72 -10 -94 L 10 -94 C 16 -72 21 -44 17 0 Z" fill={stemC} />
      {[-58, -40, -22, -5, 12, 30, 48, 62].map((gx, i) => (
        <line key={i} x1={gx} y1={-97} x2={gx * 0.32} y2={-78}
          stroke={g > 0.3 ? P.glow : stemC} strokeWidth={5} strokeLinecap="round"
          opacity={0.3 + g * 0.7} />
      ))}
      {g > 0.05 && <ellipse cx={0} cy={-86} rx={66} ry={10} fill={P.glow} opacity={0.38 * g} />}
      <path d="M -88 -98 Q 0 -196 88 -98 Q 0 -80 -88 -98 Z" fill={capC} stroke={capDC} strokeWidth={4} />
      {dim < 0.5 && (
        <g>
          <ellipse cx={-32} cy={-136} rx={13} ry={7} fill={P.cream} opacity={0.5} transform="rotate(-16 -32 -136)" />
          <ellipse cx={24} cy={-150} rx={9} ry={5} fill={P.cream} opacity={0.45} transform="rotate(10 24 -150)" />
          <ellipse cx={48} cy={-118} rx={7} ry={4} fill={P.cream} opacity={0.4} />
        </g>
      )}
    </g>
  );
};

/* ---------- 腐木 ---------- */
const Log = ({x, y, w = 560, h = 150, rot = 0}: {x: number; y: number; w?: number; h?: number; rot?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot})`}>
    <rect x={0} y={0} width={w} height={h} rx={h / 2} fill={P.wood} />
    <rect x={8} y={h * 0.55} width={w - 16} height={h * 0.38} rx={h * 0.19} fill={P.woodD} opacity={0.35} />
    <ellipse cx={10} cy={h / 2} rx={26} ry={h / 2 - 8} fill={P.woodL} />
    <ellipse cx={10} cy={h / 2} rx={15} ry={h / 2 - 20} fill={P.wood} />
    <path d={`M ${w * 0.3} ${h * 0.28} l 42 14 l 36 -11`} stroke={P.woodD} strokeWidth={5} fill="none" strokeLinecap="round" />
    <path d={`M ${w * 0.6} ${h * 0.62} l 44 12 l 30 -9`} stroke={P.woodD} strokeWidth={5} fill="none" strokeLinecap="round" />
    <ellipse cx={w * 0.48} cy={h * 0.4} rx={16} ry={9} fill={P.woodD} opacity={0.6} />
  </g>
);

/* ---------- 萤火虫：腹部方块波闪烁 ---------- */
const Firefly = ({x, y, s = 1, rot = 0, lit = 1}: {x: number; y: number; s?: number; rot?: number; lit?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`}>
    <ellipse cx={-4} cy={-13} rx={17} ry={8} fill="#cfd8ff" opacity={0.65} transform="rotate(-34 -4 -13)" />
    <ellipse cx={8} cy={-13} rx={17} ry={8} fill="#cfd8ff" opacity={0.65} transform="rotate(30 8 -13)" />
    <ellipse cx={0} cy={0} rx={27} ry={13} fill="#3d3670" />
    <circle cx={25} cy={-2} r={8} fill="#2c2754" />
    <circle cx={-25} cy={1} r={10} fill={lit > 0.4 ? '#eaff9e' : '#4b5570'} />
    {lit > 0.4 && <GlowRays x={-25} y={1} r0={15} r1={32} n={7} k={lit} lw={4} col="#eaff9e" />}
    <circle cx={-25} cy={1} r={5} fill={lit > 0.4 ? '#ffffff' : '#6b7690'} />
  </g>
);

/* ---------- LED 塑料假蘑菇 ---------- */
const FakeShroom = ({x, y, s = 1, lit = 1}: {x: number; y: number; s?: number; lit?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    {lit > 0.03 && <GlowRays x={0} y={-96} r0={72} r1={118} n={9} k={lit} lw={5} arc={Math.PI * 1.5} />}
    <path d="M -17 0 C -21 -44 -16 -72 -10 -94 L 10 -94 C 16 -72 21 -44 17 0 Z" fill={P.plastic} />
    {lit > 0.05 && <ellipse cx={0} cy={-86} rx={64} ry={10} fill={P.glow} opacity={0.35 * lit} />}
    <path d="M -88 -98 Q 0 -196 88 -98 Q 0 -80 -88 -98 Z" fill={P.plastic} stroke={P.plasticD} strokeWidth={4} />
    <rect x={-14} y={-48} width={28} height={16} rx={7} fill={lit > 0.4 ? P.glow : '#3c4666'} stroke={P.plasticD} strokeWidth={2.5} />
    <line x1={-6} y1={-32} x2={-6} y2={-22} stroke={P.plasticD} strokeWidth={3} />
    <line x1={6} y1={-32} x2={6} y2={-22} stroke={P.plasticD} strokeWidth={3} />
  </g>
);

/* ---------- 分子零件 ---------- */
const HexMol = ({x, y, s = 1, col = P.amber, op = 1}: {x: number; y: number; s?: number; col?: string; op?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
    <polygon points="30,0 15,26 -15,26 -30,0 -15,-26 15,-26" fill={col} stroke={P.ink} strokeWidth={3} />
    <circle r={8} fill={P.ink} opacity={0.2} />
  </g>
);
const O2 = ({x, y, s = 1, op = 1}: {x: number; y: number; s?: number; op?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
    <circle cx={-11} cy={0} r={15} fill="#6fc3ff" stroke="#2e6a9e" strokeWidth={3} />
    <circle cx={11} cy={0} r={15} fill="#6fc3ff" stroke="#2e6a9e" strokeWidth={3} />
  </g>
);

/* ---------- 荧光素酶：左侧开嘴的紫色圆 ---------- */
const Enzyme = ({x, y, mouth = 0.5}: {x: number; y: number; mouth?: number}) => (
  <g transform={`translate(${x},${y})`}>
    <circle r={66} fill={P.violet} stroke={P.violetD} strokeWidth={4} />
    <polygon points={`0,4 ${-88},${-24 - mouth * 30} ${-88},${34 + mouth * 30}`} fill={P.panel} />
    <circle cx={20} cy={-20} r={9} fill={P.violetD} opacity={0.6} />
    <circle cx={30} cy={16} r={6} fill={P.violetD} opacity={0.5} />
  </g>
);

/* ---------- 反应闪光星爆 ---------- */
const Burst = ({x, y, k, col = P.glow}: {x: number; y: number; k: number; col?: string}) => {
  if (k <= 0.01) return null;
  const pts: string[] = [];
  for (let i = 0; i < 16; i++) {
    const r = (i % 2 ? 22 : 48) * k;
    const a = (i / 16) * Math.PI * 2 - Math.PI / 2;
    pts.push(`${Math.cos(a) * r},${Math.sin(a) * r}`);
  }
  return (
    <g transform={`translate(${x},${y})`} opacity={clamp01(k * 1.5)}>
      <polygon points={pts.join(' ')} fill={col} stroke={P.cream} strokeWidth={3} />
      <GlowRays x={0} y={0} r0={54 * k} r1={92 * k} n={10} k={k} lw={5} />
    </g>
  );
};

/* ---------- 鬼火火苗 ---------- */
const Wisp = ({x, y, s = 1, seed = 0, k = 1}: {x: number; y: number; s?: number; seed?: number; k?: number}) => {
  const f = useCurrentFrame();
  const w = Math.sin(f / 5 + seed * 2.2) * 9;
  const dy = Math.sin(f / 14 + seed) * 12;
  return (
    <g transform={`translate(${x + w},${y + dy}) scale(${s})`} opacity={k}>
      <path d={`M 0 0 Q ${-22 + w * 0.5} -34 ${-8 + w * 0.4} -66 Q ${-2} -88 0 -96 Q ${12 + w * 0.2} -74 ${10 + w * 0.3} -50 Q ${22 + w * 0.5} -26 0 0 Z`} fill={P.glowMid} />
      <path d="M 0 -8 Q -8 -26 0 -52 Q 8 -30 6 -16 Q 4 -9 0 -8 Z" fill="#d8ffd6" opacity={0.9} />
    </g>
  );
};

/* ---------- 发光盆栽（基因移植成果） ---------- */
const GlowPlant = ({x, y, s = 1, k = 1}: {x: number; y: number; s?: number; k?: number}) => {
  const f = useCurrentFrame();
  const pulse = 0.7 + 0.3 * Math.sin(f / 8);
  const leaves: [number, number, number][] = [[-50, -118, -38], [52, -140, 36], [-42, -198, -26], [44, -214, 28], [0, -258, 0]];
  return (
    <g transform={`translate(${x},${y}) scale(${s})`} opacity={k}>
      <path d="M -58 0 L 58 0 L 44 72 L -44 72 Z" fill="#c9724a" stroke="#96502f" strokeWidth={4} />
      <rect x={-66} y={-18} width={132} height={24} rx={11} fill="#d9815a" stroke="#96502f" strokeWidth={3} />
      <path d="M 0 -12 C -8 -90 8 -170 0 -252" stroke={P.glowMid} strokeWidth={9} fill="none" strokeLinecap="round" />
      {leaves.map(([lx, ly, lr], i) => (
        <g key={i} transform={`translate(${lx * 0.55},${ly}) rotate(${lr})`}>
          <ellipse rx={46} ry={20} fill={P.glowDeep} stroke={P.glowMid} strokeWidth={3.5} />
          <line x1={-32} y1={0} x2={32} y2={0} stroke={P.glow} strokeWidth={4 * pulse} strokeLinecap="round" />
          <line x1={-8} y1={-9} x2={20} y2={-9} stroke={P.glow} strokeWidth={2.5 * pulse} opacity={0.8} />
          <line x1={-18} y1={9} x2={10} y2={9} stroke={P.glow} strokeWidth={2.5 * pulse} opacity={0.8} />
        </g>
      ))}
      <GlowDots cx={0} cy={-160} rx={115} ry={120} n={9} k={k * pulse} r={5} />
    </g>
  );
};

/* ---------- 亮度函数：白天低、夜里高（20.5→次日3.5 平台） ---------- */
const bright = (h: number) => {
  if (h >= 20.5 || h <= 3.5) return 1;
  if (h > 3.5 && h < 5.8) return 1 - 0.9 * (h - 3.5) / 2.3;
  if (h >= 5.8 && h < 18) return 0.1;
  return 0.1 + 0.9 * (h - 18) / 2.5;
};

/* ============ 标题卡 ============ */
const TitleCard = ({data}: {data: any}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 13, mass: 0.7}});
  const o = interpolate(f, [0, 8], [0, 1]);
  const p0 = Math.max(0.001, spring({frame: f - 4, fps, config: {damping: 10, stiffness: 150}}));
  const p1 = Math.max(0.001, spring({frame: f - 12, fps, config: {damping: 10, stiffness: 150}}));
  return (
    <AbsoluteFill>
      <Backdrop top={P.bg} bottom={P.bg} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <circle cx={960} cy={330} r={212} fill={P.panel} stroke={P.card} strokeWidth={6} />
        <g transform={`translate(960,452) scale(${p0})`}>
          <rect x={-160} y={-18} width={320} height={48} rx={24} fill={P.wood} />
          <ellipse cx={-150} cy={6} rx={16} ry={21} fill={P.woodL} />
          <path d="M -80 4 l 46 10 l 34 -8" stroke={P.woodD} strokeWidth={4} fill="none" strokeLinecap="round" />
        </g>
        <g transform={`translate(884,440) scale(${p0})`}><Mushroom x={0} y={0} s={0.95} gill={1} aura={0.9} /></g>
        <g transform={`translate(1046,446) scale(${p1})`}><Mushroom x={0} y={0} s={0.64} gill={1} aura={0.8} cap="#d1703f" /></g>
        <g transform={`translate(986,450) scale(${p1})`}><Mushroom x={0} y={0} s={0.48} gill={0.9} aura={0.7} cap={P.amber} capD="#c98a2e" /></g>
        <GlowDots cx={960} cy={290} rx={170} ry={100} n={10} r={5} />
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 300}}>
          <div style={{color: P.glow, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 24}}>
            冷知识 · 第 {data.episode} 期
          </div>
          <div style={{color: C.text, fontSize: 86, fontWeight: 900, fontFamily: FONT_HEAD, letterSpacing: 5}}>
            {data.title}
          </div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ============ S0 冷光流水线：菌褶剖面 + 反应放大窗 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const pop = Math.max(0.001, spring({frame: f - 6, fps, config: {damping: 11, stiffness: 120}}));
  const panelK = clamp01((f - (b1 + 15)) / 22);

  const CX = 1350, CY = 450, R = 280;
  const T = 108;
  const t = f < 20 ? 0 : (f - 20) % T;
  const tIn = clamp01(t / 34);
  const tO2 = clamp01((t - 12) / 34);
  const flash = t > 44 && t < 80 ? Math.sin(((t - 44) / 36) * Math.PI) : 0;
  const MX = CX - 58, MY = CY + 98;                 // 酶嘴汇合点
  const molX = (CX - 235) + tIn * (MX - (CX - 235));
  const molY = (CY + 26) + tIn * (MY - (CY + 26));
  const oX = (CX + 105) + tO2 * (MX - (CX + 105));
  const oY = (CY - 195) + tO2 * (MY - (CY - 195));
  const mouth = 0.25 + tIn * 0.75 - flash * 0.9;

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={0} name={sec.name} accent={P.glow} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <defs>
          <clipPath id="s0zoom"><circle cx={CX} cy={CY} r={R - 8} /></clipPath>
        </defs>
        {/* 左：腐木上的发光蘑菇剖面 */}
        <Log x={170} y={600} w={460} h={110} />
        <g transform={`translate(400,610) scale(${1.35 * pop})`}>
          <Mushroom x={0} y={0} gill={1} aura={0.75} />
        </g>
        <Mushroom x={560} y={614} s={0.62} gill={0.85} aura={0.5} rot={8} />
        <GlowDots cx={400} cy={450} rx={200} ry={120} n={8} k={0.8} />
        {/* 菌褶定位圈 → 放大窗 */}
        <circle cx={405} cy={476} r={72} fill="none" stroke={P.glow} strokeWidth={5} strokeDasharray="11 9" />
        <path d="M 472 460 C 680 420 860 420 1075 440" stroke={P.glow} strokeWidth={4}
          strokeDasharray="10 11" fill="none" opacity={0.8} />
        {/* 右：反应放大窗 */}
        <circle cx={CX} cy={CY} r={R} fill={P.panel} stroke={P.glowMid} strokeWidth={6} />
        <text x={CX} y={CY - R - 18} textAnchor="middle" fill={P.glow} fontSize={30}
          fontFamily={FONT_HEAD} letterSpacing={4}>菌褶细胞里 · 化学流水线</text>
        <g clipPath="url(#s0zoom)">
          {/* 轨道虚线 */}
          <path d={`M ${CX - 235} ${CY + 26} L ${MX} ${MY}`} stroke={P.cream} strokeWidth={3.5}
            strokeDasharray="4 12" opacity={0.4} />
          <path d={`M ${CX + 105} ${CY - 195} L ${MX} ${MY}`} stroke={P.cream} strokeWidth={3.5}
            strokeDasharray="4 12" opacity={0.4} />
          <Enzyme x={CX} y={CY + 102} mouth={Math.max(0.05, mouth)} />
          <HexMol x={molX} y={molY} s={0.9} op={flash > 0.15 ? 1 - flash : 1} />
          <O2 x={oX} y={oY} s={0.9} op={flash > 0.15 ? 1 - flash : 1} />
          <Burst x={MX} y={MY - 10} k={flash} />
          {/* 光子上飘 */}
          {flash > 0.3 && [0, 1, 2, 3, 4].map((i) => {
            const dt = (t - 50) - i * 4;
            if (dt < 0) return null;
            return <circle key={i} cx={MX - 26 + i * 14} cy={MY - 34 - dt * 3.4 * (1 + i * 0.13)}
              r={7 - i * 0.6} fill={P.glow} opacity={clamp01(1 - dt / 46) * 0.95} />;
          })}
          {flash > 0.25 && (
            <text x={CX + 20} y={CY - 58} fill={P.glow} fontSize={30} fontFamily={FONT_HEAD}
              fontWeight={700} letterSpacing={3}>绿光 ≈520nm</text>
          )}
          {/* 固定标签 */}
          <text x={CX - 240} y={CY - 8} fill={P.amber} fontSize={27} fontFamily={FONT_HEAD}>真菌荧光素</text>
          <text x={CX + 132} y={CY - 200} fill="#6fc3ff" fontSize={27} fontFamily={FONT_HEAD}>O₂</text>
          <text x={CX} y={CY + 216} textAnchor="middle" fill={P.violet} fontSize={27} fontFamily={FONT_HEAD}>荧光素酶</text>
        </g>
        {/* 能量账本（cue1） */}
        <g opacity={panelK}>
          <rect x={140} y={742} width={560} height={152} rx={18} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={168} y={782} fill={P.cream} fontSize={28} fontFamily={FONT_HEAD} fontWeight={700}>能量账本：几乎全变光</text>
          <text x={168} y={828} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>灯泡</text>
          <rect x={252} y={806} width={340} height={24} rx={12} fill={P.red} />
          <rect x={252} y={806} width={76} height={24} rx={12} fill={P.amber} />
          <text x={606} y={827} fill={P.red} fontSize={24} fontFamily={FONT_HEAD}>大半变热</text>
          <text x={168} y={872} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>蘑菇</text>
          <rect x={252} y={850} width={340 * panelK} height={24} rx={12} fill={P.glow} />
          <text x={606} y={871} fill={P.glow} fontSize={24} fontFamily={FONT_HEAD}>≈100%</text>
        </g>
      </svg>
      <Chip text="荧光素 + 荧光素酶 + 氧" x={140} y={140} at={beat(0) + 15} accent={P.amber} />
      <Chip text="冷光 · 不烫手" x={140} y={250} at={beat(0) + 45} accent={P.glow} />
      <Stamp text="2018 年才破解的通路" x={1050} y={800} at={b1 + 60} color={P.glowDeep} size={44} rot={-4} />
      <Chip text="效率≈100% · 人工光源都是弟弟" x={700} y={140} at={b1 + 100} accent={P.glow} />
    </FadeIn>
  );
};

/* ============ S1 萤火虫一闪一闪 vs 蘑菇常亮 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const cardK = clamp01((f - (b1 + 8)) / 20);
  const pop = Math.max(0.001, spring({frame: f - 6, fps, config: {damping: 11, stiffness: 120}}));

  // 萤火虫沿贝塞尔巡航，尾部方块波闪烁
  const qq = (t: number, a: number, c: number, b: number) => (1 - t) * (1 - t) * a + 2 * (1 - t) * t * c + t * t * b;
  const P0 = {x: 150, y: 640}, PC = {x: 470, y: 290}, P1 = {x: 890, y: 560};
  const cyc = 210;
  const p = (f % cyc) / cyc;
  const fx = qq(p, P0.x, PC.x, P1.x), fy = qq(p, P0.y, PC.y, P1.y);
  const vx = qq(p + 0.01, P0.x, PC.x, P1.x) - fx, vy = qq(p + 0.01, P0.y, PC.y, P1.y) - fy;
  const rot = Math.atan2(vy, vx) * 180 / Math.PI;
  const lit = (f % 48) < 27 ? 1 : 0;

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={1} name={sec.name} accent={P.amber} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={16} from={0} dur={30} />
        {/* 中缝虚线 */}
        <line x1={960} y1={140} x2={960} y2={720} stroke={P.cream} strokeWidth={3} strokeDasharray="4 18" opacity={0.3} />
        {/* 左：萤火虫 */}
        <text x={480} y={188} textAnchor="middle" fill={P.cream} fontSize={42} fontFamily={FONT_HEAD} fontWeight={700}>萤火虫</text>
        <text x={480} y={232} textAnchor="middle" fill={P.muted} fontSize={28} fontFamily={FONT_HEAD}>信号灯 · 找配偶</text>
        {/* 方波：一闪一闪 */}
        <path d="M 300 300 h44 v-30 h44 v30 h50 v-30 h50 v30 h46 v-30 h46 v30 h44 v-30 h44 v30"
          stroke="#eaff9e" strokeWidth={5} fill="none" strokeLinecap="round" transform="translate(0,14)" />
        {/* 飞行虚线轨迹 */}
        <path d={`M ${P0.x} ${P0.y} Q ${PC.x} ${PC.y} ${P1.x} ${P1.y}`} stroke={P.muted} strokeWidth={3.5}
          strokeDasharray="3 15" fill="none" opacity={0.5} />
        {/* 余辉残点：按经过时的亮灭留信号 */}
        {[1, 2, 3, 4, 5, 6, 7].map((k) => {
          const pp = p - k * 0.045;
          if (pp <= 0) return null;
          const wasLit = ((f - k * 9.5) % 48) < 27;
          return <circle key={k} cx={qq(pp, P0.x, PC.x, P1.x)} cy={qq(pp, P0.y, PC.y, P1.y)}
            r={8 - k * 0.7} fill="#eaff9e" opacity={wasLit ? 0.85 - k * 0.1 : 0.08} />;
        })}
        <Firefly x={fx} y={fy} s={1.15} rot={rot * 0.4} lit={lit} />
        {/* 右：蘑菇常亮 */}
        <text x={1440} y={188} textAnchor="middle" fill={P.glow} fontSize={42} fontFamily={FONT_HEAD} fontWeight={700}>发光蘑菇</text>
        <text x={1440} y={232} textAnchor="middle" fill={P.muted} fontSize={28} fontFamily={FONT_HEAD}>夜灯 · 等快递员</text>
        <line x1={1300} y1={314} x2={1600} y2={314} stroke={P.glow} strokeWidth={6} strokeLinecap="round" />
        <GlowDots cx={1450} cy={300} rx={140} ry={40} n={6} k={0.8} r={4} />
        <Log x={1290} y={700} w={330} h={100} />
        <g transform={`translate(1455,708) scale(${1.25 * pop})`}>
          <Mushroom x={0} y={0} gill={1} aura={0.9} />
        </g>
        <GlowDots cx={1455} cy={560} rx={150} ry={100} n={8} k={0.85} />
        {/* 底部对照卡：两种荧光素 */}
        <g opacity={cardK}>
          <rect x={540} y={742} width={840} height={150} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <g transform="translate(760,806)">
            <polygon points="0,-34 9,-11 33,-10 14,5 21,29 0,15 -21,29 -14,5 -33,-10 -9,-11"
              fill="#eaff9e" stroke={P.ink} strokeWidth={3} />
          </g>
          <text x={760} y={872} textAnchor="middle" fill={P.cream} fontSize={27} fontFamily={FONT_HEAD}>虫 · 荧光素</text>
          <text x={960} y={830} textAnchor="middle" fill={P.red} fontSize={62} fontFamily={FONT_HEAD} fontWeight={900}>≠</text>
          <HexMol x={1160} y={806} s={1.15} col={P.glow} />
          <text x={1160} y={872} textAnchor="middle" fill={P.cream} fontSize={27} fontFamily={FONT_HEAD}>菇 · 荧光素</text>
        </g>
      </svg>
      <Chip text="一闪一闪发信号" x={150} y={380} at={beat(0) + 25} accent="#eaff9e" />
      <Chip text="整夜常亮不闪烁" x={1330} y={380} at={beat(0) + 55} accent={P.glow} />
      <Stamp text="两套配方 · 各自发明" x={700} y={140} at={b1 + 15} color={P.glowDeep} size={46} rot={4} />
      <Chip text="弱到白天看不见" x={1410} y={140} at={b1 + 70} accent={P.amber} />
    </FadeIn>
  );
};

/* ============ S2 LED 假蘑菇实验：光=广告位 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2), b3 = beat(3);
  const zoomK = Math.max(0.001, spring({frame: f - b2, fps, config: {damping: 12, stiffness: 120}}));
  const gillOnly = clamp01((f - b3) / 20);          // cue3：只亮菌褶

  // 四只甲虫扑向亮组
  const lands = [[408, 542], [512, 530], [446, 578], [536, 566]];
  const bugs = lands.map(([lx, ly], i) => {
    const at = b1 + 8 + i * 32;
    const t = f - at;
    if (t < 0) return null;
    const p = clamp01(t / 75);
    const x0 = -80, y0 = 280 + i * 90, cx = 170, cy = 160 + i * 60;
    const bx = (1 - p) * (1 - p) * x0 + 2 * (1 - p) * p * cx + p * p * lx;
    const by = (1 - p) * (1 - p) * y0 + 2 * (1 - p) * p * cy + p * p * ly;
    const arrived = p >= 1;
    return (
      <g key={i}>
        {!arrived && (
          <path d={`M ${x0} ${y0} Q ${cx} ${cy} ${bx} ${by}`} stroke={P.muted} strokeWidth={3}
            strokeDasharray="3 13" fill="none" opacity={0.5} />
        )}
        <Bug x={bx} y={by} scale={0.85} rot={arrived ? Math.sin(f / 9 + i) * 10 : -20} walk={arrived ? 0.4 : 1} />
      </g>
    );
  });
  const landed = lands.filter((_, i) => f > b1 + 8 + i * 32 + 75).length;
  // 灭灯组：一只虫路过又飞走
  const snubT = f - (b1 + 130);
  const snubX = snubT > 0 && snubT < 110 ? 730 + snubT * 1.2 : null;

  // 放大窗内：甲虫爬菌褶（cue2）→ 飞出带孢子
  const walkT = Math.max(0, f - (b2 + 25));
  const bugZX = 1555 - 120 + Math.min(240, walkT * 1.05);
  const exitT = f - (b2 + 155);
  const spores = Math.min(6, Math.floor(Math.max(0, walkT - 18) / 15));

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={2} name={sec.name} accent={P.glow} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <defs>
          <clipPath id="s2zoom"><circle cx={1555} cy={415} r={222} /></clipPath>
        </defs>
        <rect x={0} y={800} width={1920} height={280} fill={P.night} />
        <rect x={0} y={800} width={1920} height={6} fill="#1a2447" />
        {/* 实验组：亮 / 灭 */}
        <g transform="translate(470,795) scale(1.45)"><FakeShroom x={0} y={0} lit={1} /></g>
        <g transform="translate(800,795) scale(1.45)" opacity={0.9}><FakeShroom x={0} y={0} lit={0} /></g>
        {/* 灭组角落蛛网 */}
        <g transform="translate(872,470)" opacity={0.55} stroke="#5a6b8a" strokeWidth={2.5} fill="none">
          <path d="M 0 0 Q 30 26 60 30 M 0 0 Q 24 40 22 72 M 0 0 Q 44 8 66 -6" />
          <path d="M 14 12 Q 32 30 52 32" />
        </g>
        {bugs}
        {snubX !== null && <Bug x={snubX} y={430 + Math.sin(snubT / 6) * 24} scale={0.7} rot={15} walk={1} />}
        {/* 计数气泡 */}
        <g transform="translate(470,392)" opacity={b1 > 0 ? 1 : 0}>
          <circle r={42} fill={P.glowDeep} stroke={P.glow} strokeWidth={4} />
          <text x={0} y={13} textAnchor="middle" fill="#fff" fontSize={40} fontFamily={FONT_HEAD} fontWeight={800}>
            {landed}
          </text>
          <text x={0} y={-56} textAnchor="middle" fill={P.glow} fontSize={26} fontFamily={FONT_HEAD}>招来的虫</text>
        </g>
        <g transform="translate(800,392)">
          <circle r={42} fill={P.card} stroke={P.plasticD} strokeWidth={4} />
          <text x={0} y={13} textAnchor="middle" fill={P.muted} fontSize={40} fontFamily={FONT_HEAD} fontWeight={800}>0</text>
          <text x={0} y={-56} textAnchor="middle" fill={P.muted} fontSize={26} fontFamily={FONT_HEAD}>对照组</text>
        </g>
        {/* 右：放大窗（cue2 入场） */}
        <g transform={`translate(1555,415) scale(${zoomK}) translate(-1555,-415)`}>
          <circle cx={1555} cy={415} r={230} fill={P.panel} stroke={P.glowMid} strokeWidth={6} />
          <text x={1555} y={700} textAnchor="middle" fontSize={28} fontFamily={FONT_HEAD}
            fill={gillOnly > 0.5 ? P.glow : P.cream}>
            {gillOnly > 0.5 ? '只亮菌褶 · 灯打在货架上' : '虫子爬过 → 孢子粘一身带走'}
          </text>
          <g clipPath="url(#s2zoom)">
            {/* 菌褶板：一排下垂发光薄片 */}
            <g opacity={1 - gillOnly}>
              {[-150, -100, -50, 0, 50, 100, 150].map((dx, i) => {
                const top = 218 - (1 - (dx / 175) * (dx / 175)) * 34;
                return <rect key={i} x={1555 + dx - 8} y={top} width={16} height={218}
                  rx={8} fill={i % 2 ? P.glowMid : P.glow} opacity={0.95} />;
              })}
              <path d="M 1380 210 Q 1555 160 1730 210 L 1730 190 L 1380 190 Z" fill={P.capD} opacity={0.85} />
              {/* 爬行的甲虫 + 粘上的孢子 */}
              {exitT < 0 && walkT > 0 && (
                <g>
                  <Bug x={bugZX} y={478} scale={1.35} rot={Math.sin(f / 8) * 6} walk={1} />
                  {Array.from({length: spores}).map((_, i) => (
                    <circle key={i} cx={bugZX - 20 + rand(i, 5) * 60} cy={478 + 26 + rand(i, 6) * 18}
                      r={5} fill={P.cream} />
                  ))}
                </g>
              )}
              {/* 带孢子飞走 */}
              {exitT >= 0 && exitT < 70 && (
                <g>
                  <Bug x={bugZX + exitT * 7} y={478 - exitT * 5.4} scale={1.35} rot={-38} walk={1} />
                  {[0, 1, 2, 3, 4].map((i) => (
                    <circle key={i} cx={bugZX + exitT * 7 - 30 - i * 26} cy={478 - exitT * 5.4 + 30 + i * 12}
                      r={5} fill={P.cream} opacity={clamp01(1 - (exitT - i * 6) / 50)} />
                  ))}
                </g>
              )}
            </g>
            {/* cue3：整朵蘑菇只亮菌褶 */}
            <g opacity={gillOnly}>
              <Mushroom x={1555} y={570} s={1.02} gill={1} aura={0.55} dim={1} />
              <ellipse cx={1555} cy={484} rx={118} ry={34} fill="none" stroke={P.glow}
                strokeWidth={4} strokeDasharray="10 8" />
            </g>
          </g>
        </g>
      </svg>
      <Stamp text="2015 · LED 假蘑菇实验" x={140} y={140} at={beat(0) + 12} color={P.glowDeep} size={42} rot={-4} />
      <Chip text="排除气味 · 光是独立吸引" x={140} y={250} at={b1 + 60} accent={P.glow} />
      <Chip text="虫 = 免费快递员" x={1060} y={720} at={b2 + 40} accent={P.amber} />
      <Stamp text="广告位精准投放" x={1140} y={800} at={b3 + 25} color={P.glowMid} size={44} rot={4} />
    </FadeIn>
  );
};

/* ============ S3 只在夜里营业：24h 亮度排班表 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);
  const barsK = clamp01((f - (b2 + 10)) / 25);
  const labK = clamp01((f - (b2 + 55)) / 22);

  const X0 = 140, X1 = 1240, W = X1 - X0, BASE = 590, TOP = 300;
  const hx = (h: number) => X0 + (h / 24) * W;
  const hc = (f * 0.24) % 24;                        // 扫描时刻
  const pts: string[] = [];
  for (let h = 0; h <= 24; h += 0.5) pts.push(`${hx(h)},${BASE - bright(h) * (BASE - TOP)}`);
  // 昆虫活动量：夜里高
  const act = (h: number) => (h >= 19 || h <= 4 ? 46 : h < 7 ? 30 : 8);

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={3} name={sec.name} accent={P.glow} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 昼夜分区底带 */}
        <rect x={X0} y={TOP - 40} width={hx(5.8) - X0} height={BASE - TOP + 40} fill={P.night} opacity={0.85} />
        <rect x={hx(5.8)} y={TOP - 40} width={hx(18) - hx(5.8)} height={BASE - TOP + 40} fill="#23336b" opacity={0.55} />
        <rect x={hx(18)} y={TOP - 40} width={X1 - hx(18)} height={BASE - TOP + 40} fill={P.night} opacity={0.85} />
        <Sun x={hx(12)} y={195} r={42} />
        <Moon x={hx(21)} y={215} scale={0.6} />
        <text x={hx(12)} y={400} textAnchor="middle" fill={P.amber} fontSize={26} fontFamily={FONT_HEAD}>白天：亮了也白亮</text>
        <text x={hx(21)} y={400} textAnchor="middle" fill={P.glow} fontSize={26} fontFamily={FONT_HEAD}>夜班</text>
        {/* 亮度曲线 */}
        <polygon points={`${X0},${BASE} ${pts.join(' ')} ${X1},${BASE}`} fill={P.glow} opacity={0.13} />
        <polyline points={pts.join(' ')} fill="none" stroke={P.glow} strokeWidth={6} strokeLinejoin="round" />
        {/* 轴 */}
        <line x1={X0} y1={BASE} x2={X1} y2={BASE} stroke={P.cream} strokeWidth={4} />
        {[0, 6, 12, 18, 24].map((h) => (
          <g key={h}>
            <line x1={hx(h)} y1={BASE} x2={hx(h)} y2={BASE + 12} stroke={P.cream} strokeWidth={4} />
            <text x={hx(h)} y={BASE + 46} textAnchor="middle" fill={P.cream} fontSize={26}
              fontFamily={FONT_HEAD}>{h}时</text>
          </g>
        ))}
        <text x={X0} y={TOP - 54} fill={P.glow} fontSize={28} fontFamily={FONT_HEAD} fontWeight={700}>发光亮度 · 生物钟排班</text>
        {/* 扫描光标 */}
        <line x1={hx(hc)} y1={TOP - 30} x2={hx(hc)} y2={BASE} stroke={P.amber} strokeWidth={4} strokeDasharray="8 8" />
        <circle cx={hx(hc)} cy={BASE - bright(hc) * (BASE - TOP)} r={11} fill={P.amber} stroke={P.cream} strokeWidth={3} />
        {/* 昆虫活动量柱（cue2） */}
        <g opacity={barsK}>
          {Array.from({length: 24}).map((_, h) => (
            <rect key={h} x={hx(h) + 6} y={790 - act(h) * barsK} width={W / 24 - 12} height={act(h) * barsK}
              rx={5} fill={P.violet} opacity={0.9} />
          ))}
          <line x1={X0} y1={790} x2={X1} y2={790} stroke={P.muted} strokeWidth={3} />
          <text x={X0} y={838} fill={P.violet} fontSize={27} fontFamily={FONT_HEAD} fontWeight={700}>
            昆虫活动量 · 峰值正好对上</text>
        </g>
        {/* 右上：实况预览窗 */}
        <rect x={1330} y={230} width={460} height={360} rx={20} fill={P.panel} stroke={P.card} strokeWidth={5} />
        <rect x={1354} y={252} width={130} height={40} rx={12} fill={P.card} />
        <text x={1419} y={280} textAnchor="middle" fill={P.muted} fontSize={24} fontFamily={FONT_HEAD}>此刻亮度</text>
        {bright(hc) > 0.4
          ? <Moon x={1730} y={268} scale={0.4} />
          : <Sun x={1730} y={268} r={22} />}
        <Mushroom x={1560} y={556} s={1.06} gill={bright(hc)} aura={bright(hc) * 0.8} />
        <rect x={1354} y={602} width={412 * bright(hc)} height={14} rx={7} fill={P.glow} />
        <rect x={1354} y={602} width={412} height={14} rx={7} fill="none" stroke={P.card} strokeWidth={3} />
        {/* 右下：恒黑实验室（cue2） */}
        <g opacity={labK}>
          <rect x={1330} y={668} width={460} height={200} rx={18} fill={P.night} stroke={P.glowMid}
            strokeWidth={4} strokeDasharray="12 9" />
          <text x={1560} y={712} textAnchor="middle" fill={P.glow} fontSize={27} fontFamily={FONT_HEAD}>
            恒黑实验室 · 照样按点亮</text>
          <Mushroom x={1470} y={836} s={0.62} gill={1} aura={0.7} />
          {/* 小钟表 */}
          <g transform="translate(1660,782)">
            <circle r={40} fill={P.panel} stroke={P.cream} strokeWidth={4} />
            <line x1={0} y1={0} x2={0} y2={-24} stroke={P.cream} strokeWidth={4} strokeLinecap="round"
              transform={`rotate(${f * 3})`} />
            <line x1={0} y1={0} x2={16} y2={0} stroke={P.glow} strokeWidth={4} strokeLinecap="round"
              transform={`rotate(${f * 0.6})`} />
            <circle r={5} fill={P.cream} />
          </g>
        </g>
      </svg>
      <Stamp text="白天开灯 = 白浪费" x={140} y={138} at={b1 + 8} color={P.red} size={44} rot={-4} />
      <Chip text="能量用在刀刃上" x={900} y={655} at={b1 + 50} accent={P.amber} />
      <Chip text="没日没夜也照点开关" x={380} y={655} at={b2 + 60} accent={P.glow} />
      <Stamp text="节能+效率一起算" x={1150} y={140} at={b2 + 100} color={P.glowDeep} size={44} rot={4} />
    </FadeIn>
  );
};

/* ============ S4 foxfire 鬼火 + 偷配方 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);
  const veinPulse = 0.55 + 0.45 * Math.sin(f / 8);
  const plantK = clamp01((f - (b1 + 5)) / 25);
  const warnK = clamp01((f - (b2 + 5)) / 16);

  return (
    <FadeIn>
      <Backdrop top={P.bgDeep} bottom={P.bgDeep} />
      <SceneTag index={4} name={sec.name} accent={P.glow} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={14} from={0} dur={30} />
        <rect x={0} y={800} width={1920} height={280} fill={P.night} />
        {/* 左：蜜环菌感染的朽木 = foxfire */}
        <g transform="translate(120,590)">
          <rect x={0} y={0} width={640} height={180} rx={90} fill={P.wood} />
          <rect x={10} y={100} width={620} height={68} rx={34} fill={P.woodD} opacity={0.4} />
          <ellipse cx={14} cy={90} rx={46} ry={80} fill={P.woodL} />
          <ellipse cx={14} cy={90} rx={30} ry={56} fill={P.wood} />
          <ellipse cx={14} cy={90} rx={14} ry={28} fill={P.woodD} />
          {/* 发光菌丝脉络 */}
          <g stroke={P.glow} fill="none" strokeLinecap="round" opacity={veinPulse}>
            <path d="M 60 60 Q 150 40 240 66 T 430 60 T 610 74" strokeWidth={6} />
            <path d="M 90 120 Q 190 142 300 118 T 560 128" strokeWidth={5} />
            <path d="M 240 66 L 258 96 L 236 118" strokeWidth={4} />
            <path d="M 430 60 L 448 92 L 424 112" strokeWidth={4} />
            <path d="M 14 60 Q 30 90 14 122" strokeWidth={4} />
          </g>
          <GlowDots cx={320} cy={90} rx={290} ry={70} n={10} k={veinPulse} r={4.5} />
          {/* 木上小蘑菇 */}
          <Mushroom x={200} y={-4} s={0.5} gill={0.95} aura={0.6} rot={-6} />
          <Mushroom x={330} y={-2} s={0.42} gill={0.9} aura={0.5} rot={7} cap={P.amber} capD="#c98a2e" />
        </g>
        {/* 鬼火火苗 */}
        <Wisp x={310} y={548} s={0.9} seed={1} />
        <Wisp x={500} y={516} s={1.05} seed={3} />
        <Wisp x={640} y={556} s={0.75} seed={6} />
        {/* 右：配方被偷去 → 发光盆栽 */}
        <g opacity={plantK}>
          {/* 基因快递箭头 */}
          <path d="M 880 430 C 1020 400 1120 430 1230 500" stroke={P.glow} strokeWidth={5}
            strokeDasharray="12 10" fill="none" markerEnd="none" />
          <polygon points="1230,500 1204,486 1208,512" fill={P.glow} transform="rotate(28 1230 500)" />
          <HexMol x={1055} y={404} s={0.85} col={P.glow} />
          <text x={1055} y={352} textAnchor="middle" fill={P.glow} fontSize={28}
            fontFamily={FONT_HEAD}>发光基因搬进植物</text>
          <GlowPlant x={1520} y={726} s={1.12} k={plantK} />
          {/* 小蘑菇比手势：旁边躲不开的小本尊 */}
          <Mushroom x={1290} y={798} s={0.6} gill={0.9} aura={0.6} rot={-8} />
        </g>
        {/* 有毒警告（cue2） */}
        <g opacity={warnK}>
          <g transform="translate(1060,760)">
            <circle r={58} fill="none" stroke={P.red} strokeWidth={9} />
            <line x1={-40} y1={40} x2={40} y2={-40} stroke={P.red} strokeWidth={9} strokeLinecap="round" />
            <Mushroom x={0} y={22} s={0.4} gill={0.6} aura={0} />
          </g>
          <text x={1060} y={862} textAnchor="middle" fill={P.red} fontSize={30} fontFamily={FONT_HEAD} fontWeight={700}>
            有毒！只许看不许碰</text>
        </g>
        <text x={400} y={840} textAnchor="middle" fill={P.muted} fontSize={27} fontFamily={FONT_HEAD}>
          foxfire · 朽木夜明</text>
      </svg>
      <Chip text="已知 80+ 种会发光" x={140} y={140} at={beat(0) + 15} accent={P.glow} />
      <Chip text="古人叫它鬼火 · 一战士兵也见过" x={140} y={250} at={beat(0) + 60} accent={P.amber} />
      <Stamp text="不用电的自发光盆栽" x={1150} y={140} at={b1 + 30} color={P.glowDeep} size={44} rot={4} />
      <Chip text="蜜环菌：林子荧光霸主" x={140} y={700} at={beat(0) + 100} accent={P.glowMid} />
    </FadeIn>
  );
};

/* ============ 结尾卡 ============ */
const EndCard = () => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 14}});
  const pop = Math.max(0.001, spring({frame: f - 5, fps, config: {damping: 10, stiffness: 140}}));
  return (
    <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
      <Backdrop top={P.bgDeep} bottom={P.bgDeep} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <ellipse cx={960} cy={1010} rx={1080} ry={120} fill={P.night} />
        <g transform={`translate(960,940) scale(${pop})`}>
          <rect x={-220} y={-24} width={440} height={62} rx={31} fill={P.wood} />
          <ellipse cx={-208} cy={7} rx={20} ry={27} fill={P.woodL} />
        </g>
        <g transform={`translate(838,922) scale(${pop})`}><Mushroom x={0} y={0} s={0.85} gill={1} aura={0.85} /></g>
        <g transform={`translate(962,912) scale(${pop})`}><Mushroom x={0} y={0} s={1.05} gill={1} aura={1} cap={P.amber} capD="#c98a2e" /></g>
        <g transform={`translate(1090,924) scale(${pop})`}><Mushroom x={0} y={0} s={0.7} gill={1} aura={0.8} cap="#d1703f" /></g>
        <GlowDots cx={960} cy={760} rx={400} ry={130} n={12} r={5} />
        <Wisp x={600} y={880} s={0.6} seed={2} k={0.9} />
        <Wisp x={1330} y={870} s={0.7} seed={5} k={0.9} />
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -230}}>
        <div style={{fontSize: 62, fontWeight: 800, color: C.text}}>那不是鬼火，是蘑菇在打广告</div>
        <div style={{fontSize: 38, color: C.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E217 = ({data = tl}: {data?: any}) => {
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
