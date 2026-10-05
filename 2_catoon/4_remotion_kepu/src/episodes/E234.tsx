import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01} from '../lib/style';
import {Backdrop, Stamp, SceneTag, Chip, Subtitles, ProgressBar, FadeIn, Stars, Moon} from '../lib/atoms';
import tl from '../timeline_234.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：风暴夜 · 墨靛 × 闪电黄 × 雨青（纯色平涂） ============ */
const P = {
  bg: '#1b2a45',        // 风暴夜墨靛
  bgDeep: '#121d33',
  bgLite: '#233454',
  card: '#22314f',
  cardLite: '#2e3f63',
  ink: '#101a2e',       // 压亮色块的深字
  cream: '#f2ecdc',
  muted: '#93a7cc',
  bolt: '#ffd94d',      // 闪电黄
  boltD: '#e8a92e',
  rain: '#74c9e8',      // 雨青
  rainD: '#3f97c4',
  wave: '#8fe3ff',      // 脑电
  vio: '#b48cf2',       // 纺锤波紫
  vioD: '#8f6ad8',
  coral: '#ff7a6e',     // 警报/拦截
  coralD: '#d8564c',
  mint: '#5fe3b3',      // 放行/通过
  sky: '#7ab8ff',
  skin: '#ffcf9e',
  skinD: '#e8a76f',
  sheet: '#dfe6f2',
  pillow: '#f6efdd',
  blanket: '#4d7fc4',
  blanketD: '#3a62a0',
  wood: '#7a5a3e',
  woodD: '#543c27',
  cloud: '#54648c',
  cloudD: '#3d4a6e',
  brain: '#e8a0b4',
  brainD: '#c07a92',
  n1: '#ffd97b',
  n2: '#f7a54f',
  n3: '#8a6bb5',
  rem: '#5fe3b3',
};

/* ---------- 极坐标 / 圆弧 ---------- */
const pol = (cx: number, cy: number, r: number, deg: number): [number, number] =>
  [cx + r * Math.cos(deg * Math.PI / 180), cy + r * Math.sin(deg * Math.PI / 180)];

const arcPath = (cx: number, cy: number, r: number, a0: number, a1: number) => {
  const [x0, y0] = pol(cx, cy, r, a0);
  const [x1, y1] = pol(cx, cy, r, a1);
  const large = a1 - a0 > 180 ? 1 : 0;
  return `M ${x0.toFixed(1)} ${y0.toFixed(1)} A ${r} ${r} 0 ${large} 1 ${x1.toFixed(1)} ${y1.toFixed(1)}`;
};

/* ---------- 正弦折线 ---------- */
const wavePath = (x0: number, x1: number, y: number, amp: number, freq: number, ph = 0) => {
  const n = 36;
  let d = '';
  for (let i = 0; i <= n; i++) {
    const px = x0 + (x1 - x0) * (i / n);
    const py = y + Math.sin(i * freq + ph) * amp;
    d += `${i ? 'L' : 'M'} ${px.toFixed(1)} ${py.toFixed(1)} `;
  }
  return d;
};

/* ---------- 纺锤波：钟形包络的快振荡 ---------- */
const spindlePath = (x0: number, x1: number, y: number, amp: number, cycles: number, ph = 0) => {
  const n = 72;
  let d = '';
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    const px = x0 + (x1 - x0) * t;
    const py = y + Math.sin(Math.PI * t) * amp * Math.sin(t * cycles * Math.PI * 2 + ph);
    d += `${i ? 'L' : 'M'} ${px.toFixed(1)} ${py.toFixed(1)} `;
  }
  return d;
};

/* ---------- 闪电 ---------- */
const Bolt = ({x, y, s = 1, col = P.bolt, rot = 0}: {x: number; y: number; s?: number; col?: string; rot?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`}>
    <path d="M -10 -58 L 24 -58 L 6 -10 L 30 -10 L -16 62 L -2 4 L -28 4 Z"
      fill={col} stroke={P.boltD} strokeWidth={4} strokeLinejoin="round" />
  </g>
);

/* ---------- 雷雨云：云团 + 可选闪电 + 可选雨丝 ---------- */
const StormCloud = ({x, y, s = 1, bolt = 1, rain = 1}: {x: number; y: number; s?: number; bolt?: number; rain?: number}) => {
  const f = useCurrentFrame();
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      {rain > 0 && (
        <g opacity={rain}>
          {[-64, -22, 20, 62].map((dx, i) => {
            const dy = ((f * 5 + i * 31) % 56);
            return <line key={i} x1={dx - dy * 0.2} y1={58 + dy} x2={dx - 8 - dy * 0.2} y2={84 + dy}
              stroke={P.rain} strokeWidth={7} strokeLinecap="round" opacity={0.8} />;
          })}
        </g>
      )}
      {bolt > 0 && <Bolt x={10} y={96} s={0.72 * bolt} />}
      <ellipse cx={-52} cy={6} rx={58} ry={42} fill={P.cloudD} />
      <ellipse cx={48} cy={8} rx={56} ry={40} fill={P.cloudD} />
      <ellipse cx={0} cy={-12} rx={86} ry={54} fill={P.cloud} />
      <ellipse cx={-30} cy={-30} rx={48} ry={30} fill={P.cloud} opacity={0.85} />
    </g>
  );
};

/* ---------- 声波弧（向右传播） ---------- */
const SoundArcs = ({x, y, col = P.wave, n = 3, gap = 26, r0 = 26, a0 = -55, a1 = 55, k = 1}: {
  x: number; y: number; col?: string; n?: number; gap?: number; r0?: number; a0?: number; a1?: number; k?: number;
}) => {
  const f = useCurrentFrame();
  return (
    <g transform={`translate(${x},${y})`} opacity={k}>
      {Array.from({length: n}).map((_, i) => {
        const rr = r0 + i * gap + Math.sin(f * 0.25 + i) * 3;
        return <path key={i} d={arcPath(0, 0, rr, a0, a1)} stroke={col} strokeWidth={7}
          fill="none" strokeLinecap="round" opacity={0.95 - i * 0.28} />;
      })}
    </g>
  );
};

/* ---------- 侧躺睡觉的床 ---------- */
const BedSleep = ({x, y, s = 1, blanket = P.blanket, hair = P.ink}: {
  x: number; y: number; s?: number; blanket?: string; hair?: string;
}) => {
  const f = useCurrentFrame();
  const breathe = Math.sin(f * 0.12) * 4;
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      {/* 床架 */}
      <rect x={-252} y={-96} width={26} height={196} rx={11} fill={P.woodD} />
      <rect x={-252} y={-108} width={26} height={26} rx={11} fill={P.wood} />
      <rect x={-250} y={52} width={570} height={34} rx={15} fill={P.wood} />
      <rect x={-216} y={86} width={24} height={50} rx={8} fill={P.woodD} />
      <rect x={272} y={86} width={24} height={50} rx={8} fill={P.woodD} />
      {/* 床垫 + 枕头 */}
      <rect x={-238} y={-6} width={548} height={58} rx={24} fill={P.sheet} />
      <ellipse cx={-188} cy={-26} rx={70} ry={30} fill={P.pillow} />
      {/* 被子身体（呼吸起伏） */}
      <path d={`M -120 -8 Q -100 ${-74 - breathe} -20 ${-70 - breathe}
        Q 120 ${-60 - breathe} 220 ${-44 - breathe} Q 286 -34 290 -8 Z`} fill={blanket} />
      <path d="M -120 -8 Q -104 -58 -34 -64" stroke={P.blanketD} strokeWidth={8}
        fill="none" strokeLinecap="round" opacity={0.6} />
      {/* 头 */}
      <g transform={`translate(-166,${-52 + breathe * 0.3})`}>
        <circle r={38} fill={P.skin} />
        <path d="M -38 -8 A 38 38 0 0 1 30 -26 L 30 -36 Q -6 -54 -38 -30 Z" fill={hair} />
        <path d="M -18 6 Q -11 12 -4 6" stroke={P.ink} strokeWidth={4} fill="none" strokeLinecap="round" />
        <path d="M 2 22 Q 10 26 18 22" stroke={P.skinD} strokeWidth={4} fill="none" strokeLinecap="round" />
      </g>
    </g>
  );
};

/* ---------- Zzz ---------- */
const Zzz = ({x, y, k = 1, col = P.sky, s = 1}: {x: number; y: number; k?: number; col?: string; s?: number}) => {
  const f = useCurrentFrame();
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      {[0, 1, 2].map((i) => {
        const up = (f * 1.4 + i * 24) % 72;
        return (
          <text key={i} x={i * 44} y={-up * 0.75} fill={col} fontSize={48 - i * 8}
            fontFamily={FONT_HEAD} fontWeight={900}
            opacity={k * clamp01(1 - up / 72)}
            transform={`rotate(${-10 + i * 9} ${i * 44} ${-up * 0.75})`}>
            Z
          </text>
        );
      })}
    </g>
  );
};

/* ---------- 耳朵 ---------- */
const EarIcon = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <path d="M 22 -58 C 58 -54 76 -22 72 10 C 68 42 46 56 32 72 C 24 82 22 90 12 90"
      stroke={P.skin} strokeWidth={22} fill="none" strokeLinecap="round" />
    <path d="M 24 -20 C 42 -16 50 0 46 18 C 42 34 30 40 24 48"
      stroke={P.skinD} strokeWidth={13} fill="none" strokeLinecap="round" />
  </g>
);

/* ---------- 大脑 ---------- */
const BrainIcon = ({x, y, s = 1, col = P.brain}: {x: number; y: number; s?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <path d="M -66 8 C -74 -28 -52 -56 -18 -60 C -6 -74 22 -74 32 -60 C 60 -56 74 -32 68 -4
      C 78 18 62 44 34 46 C 24 60 -8 62 -20 48 C -48 50 -72 32 -66 8 Z" fill={col} />
    <path d="M -46 -18 Q -30 -30 -14 -20 M -40 8 Q -24 -2 -8 8 M 6 -34 Q 22 -40 34 -28
      M 12 20 Q 28 12 42 22 M -8 34 Q 2 28 12 36"
      stroke={P.brainD} strokeWidth={6} fill="none" strokeLinecap="round" />
  </g>
);

/* ---------- 丘脑闸门：两根门柱 + 下落的栅栏 ---------- */
const ThalamusGate = ({x, y, s = 1, k = 0}: {x: number; y: number; s?: number; k?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-74} y={-84} width={148} height={18} rx={9} fill={P.cardLite} />
    <rect x={-74} y={-84} width={18} height={168} rx={9} fill={P.cardLite} />
    <rect x={56} y={-84} width={18} height={168} rx={9} fill={P.cardLite} />
    {/* 栅栏下落 */}
    <g transform={`translate(0,${-84 + clamp01(k) * 96})`}>
      {[-38, -13, 12, 37].map((dx, i) => (
        <rect key={i} x={dx} y={0} width={13} height={72} rx={6} fill={P.coral} />
      ))}
    </g>
    <text x={0} y={114} textAnchor="middle" fill={P.cream} fontSize={26}
      fontFamily={FONT_HEAD} fontWeight={800}>丘脑门卫</text>
  </g>
);

/* ---------- 对勾 / 叉 圆点 ---------- */
const CheckDot = ({x, y, r = 20, k = 1, ok = true, col}: {
  x: number; y: number; r?: number; k?: number; ok?: boolean; col?: string;
}) => {
  const kk = Math.max(0.001, clamp01(k));
  const c = col || (ok ? P.mint : P.coral);
  return (
    <g transform={`translate(${x},${y}) scale(${kk})`} opacity={kk > 0.02 ? 1 : 0}>
      <circle r={r} fill={c} />
      {ok ? (
        <path d={`M ${-r * 0.45} 0 L ${-r * 0.1} ${r * 0.4} L ${r * 0.5} ${-r * 0.38}`}
          stroke={P.ink} strokeWidth={5.5} fill="none" strokeLinecap="round" strokeLinejoin="round" />
      ) : (
        <path d={`M ${-r * 0.38} ${-r * 0.38} L ${r * 0.38} ${r * 0.38} M ${r * 0.38} ${-r * 0.38} L ${-r * 0.38} ${r * 0.38}`}
          stroke="#fff" strokeWidth={5.5} strokeLinecap="round" />
      )}
    </g>
  );
};

/* ---------- 睁眼 / 睡眼 小脸 ---------- */
const FaceIcon = ({x, y, s = 1, mode = 'sleep', hair = P.ink}: {
  x: number; y: number; s?: number; mode?: 'sleep' | 'wake'; hair?: string;
}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <circle r={42} fill={P.skin} />
    <path d="M -42 -12 A 42 42 0 0 1 38 -26 L 38 -36 Q -2 -56 -42 -38 Z" fill={hair} />
    {mode === 'sleep' ? (
      <path d="M -20 0 Q -14 6 -8 0 M 8 0 Q 14 6 20 0" stroke={P.ink} strokeWidth={4.5}
        fill="none" strokeLinecap="round" />
    ) : (
      <g>
        <ellipse cx={-14} cy={0} rx={8} ry={10} fill="#fff" />
        <ellipse cx={14} cy={0} rx={8} ry={10} fill="#fff" />
        <circle cx={-13} cy={1} r={3.6} fill={P.ink} />
        <circle cx={15} cy={1} r={3.6} fill={P.ink} />
        <path d="M -8 20 L 8 20" stroke={P.ink} strokeWidth={4} strokeLinecap="round" />
      </g>
    )}
  </g>
);

/* ---------- 婴儿床 ---------- */
const BabyCrib = ({x, y, s = 1, crying = 1}: {x: number; y: number; s?: number; crying?: number}) => {
  const f = useCurrentFrame();
  const squirm = Math.sin(f * 0.5) * 3 * crying;
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      {/* 哭声弧 */}
      <g opacity={crying}>
        {[0, 1, 2].map((i) => {
          const rr = 34 + i * 24 + Math.sin(f * 0.4 + i) * 4;
          return <path key={i} d={arcPath(0, -56, rr, -125, -55)} stroke={P.coral}
            strokeWidth={7} fill="none" strokeLinecap="round" opacity={0.95 - i * 0.3} />;
        })}
      </g>
      {/* 床体 */}
      <rect x={-92} y={-6} width={184} height={74} rx={14} fill={P.wood} />
      {[-62, -21, 20, 61].map((dx, i) => (
        <rect key={i} x={dx - 6} y={-6} width={12} height={74} rx={6} fill={P.woodD} />
      ))}
      <rect x={-92} y={-6} width={184} height={16} rx={8} fill={P.woodD} />
      <rect x={-84} y={68} width={16} height={30} rx={6} fill={P.woodD} />
      <rect x={68} y={68} width={16} height={30} rx={6} fill={P.woodD} />
      {/* 婴儿 */}
      <g transform={`translate(${squirm},0)`}>
        <ellipse cx={18} cy={2} rx={52} ry={24} fill={P.mint} opacity={0.85} />
        <circle cx={-34} cy={-16} r={26} fill={P.skin} />
        <path d="M -54 -24 Q -34 -40 -14 -24" stroke={P.ink} strokeWidth={4} fill="none" strokeLinecap="round" />
        <path d="M -42 -12 Q -37 -7 -32 -12" stroke={P.ink} strokeWidth={3.4} fill="none" strokeLinecap="round" />
        {crying > 0.3 && (
          <path d="M -30 -2 Q -26 4 -22 -2" stroke={P.coralD} strokeWidth={4} fill="none" strokeLinecap="round" />
        )}
      </g>
    </g>
  );
};

/* ---------- 电池 ---------- */
const Battery = ({x, y, w = 110, h = 48, fill = 0.3, col = P.coral}: {
  x: number; y: number; w?: number; h?: number; fill?: number; col?: string;
}) => (
  <g transform={`translate(${x},${y})`}>
    <rect x={-w / 2} y={-h / 2} width={w} height={h} rx={11} fill="none" stroke={col} strokeWidth={6} />
    <rect x={w / 2 + 5} y={-h * 0.18} width={11} height={h * 0.36} rx={5} fill={col} />
    <rect x={-w / 2 + 8} y={-h / 2 + 8} width={(w - 16) * clamp01(fill)} height={h - 16} rx={6} fill={col} />
  </g>
);

/* ---------- 酒杯 + 药丸 ---------- */
const CupPill = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <path d="M -30 -44 L 30 -44 L 24 6 Q 20 22 0 22 Q -20 22 -24 6 Z" fill={P.sky} opacity={0.9} />
    <path d="M -26 -30 L 26 -30 L 22 4 Q 18 16 0 16 Q -18 16 -22 4 Z" fill={P.coralD} />
    <line x1={0} y1={22} x2={0} y2={46} stroke={P.sky} strokeWidth={8} strokeLinecap="round" />
    <line x1={-22} y1={48} x2={22} y2={48} stroke={P.sky} strokeWidth={8} strokeLinecap="round" />
    <g transform="translate(52,20) rotate(40)">
      <rect x={-20} y={-11} width={40} height={22} rx={11} fill={P.cream} />
      <rect x={-20} y={-11} width={20} height={22} rx={11} fill={P.coral} />
    </g>
  </g>
);

/* ============ 标题卡 ============ */
const TitleCard = ({data}: {data: any}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 13, mass: 0.7}});
  const pop = spring({frame: f - 6, fps, config: {damping: 10, stiffness: 140}});
  const o = interpolate(f, [0, 8], [0, 1]);
  const flash = (f % 46 < 4) ? 1 : 0;
  const zK = clamp01((f - 20) / 18);
  return (
    <AbsoluteFill>
      <Backdrop top={P.bg} bottom={P.bg} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={20} from={0} dur={24} />
        <circle cx={960} cy={352} r={216} fill={P.cardLite} />
        <circle cx={960} cy={352} r={216} fill={P.bolt} opacity={flash * 0.12} />
        <g transform={`translate(960,368) scale(${clamp01(pop)})`}>
          <g transform="translate(0,-118) scale(0.92)">
            <StormCloud x={0} y={0} s={1} bolt={0} rain={0} />
          </g>
          <Bolt x={14} y={-6} s={0.9} rot={flash ? -6 : 0} />
          <g transform="translate(-14,132) scale(0.72)">
            <BedSleep x={0} y={0} s={1} />
          </g>
          <g transform="translate(96,-196)">
            <Zzz x={0} y={0} k={zK} col={P.rain} />
          </g>
        </g>
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 292}}>
          <div style={{color: P.bolt, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 26}}>
            冷知识 · 第 {data.episode} 期
          </div>
          <div style={{color: C.text, fontSize: 80, fontWeight: 900, fontFamily: FONT_HEAD, letterSpacing: 5}}>
            {data.title}
          </div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ============ S0 耳朵没下班：听觉通路剖面 + 丘脑闸门拦截 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1); // ≈150f “差别不在耳朵灵不灵…”

  const cardK = spring({frame: f - 8, fps, config: {damping: 13, stiffness: 120}});
  const nodeK = (i: number) => clamp01(spring({frame: f - 22 - i * 12, fps, config: {damping: 12, stiffness: 150}}));
  const gateK = clamp01((f - b1) / 16);            // 闸门下落
  const blockK = clamp01(spring({frame: f - b1 - 10, fps, config: {damping: 10, stiffness: 160}}));
  const botK = spring({frame: f - 60, fps, config: {damping: 13, stiffness: 120}});
  const blip = Math.max(0, Math.sin(f * 0.09)) * (0.4 + gateK * 0.6);

  // 信号小球：耳 → 闸门口排队（被拦）
  const dotX = (i: number) => {
    const raw = (f * 5 + i * 150) % 400;
    const x = 620 + raw;
    const stop = 980;
    return Math.min(x, stop + Math.sin((f * 0.5 + i) * 1.2) * 6);
  };

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={0} name={sec.name} accent={P.bolt} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* ---- 通路剖面卡 ---- */}
        <g opacity={clamp01(cardK)}>
          <rect x={110} y={190} width={1700} height={355} rx={36} fill={P.card} stroke={P.bolt} strokeWidth={4} />
          <text x={960} y={246} textAnchor="middle" fill={P.cream} fontSize={33}
            fontFamily={FONT_HEAD} fontWeight={800}>
            睡着后 · 听觉通路还在上班
          </text>
          {/* 连线 */}
          <line x1={350} y1={390} x2={1430} y2={390} stroke={P.muted} strokeWidth={5}
            strokeDasharray="12 12" opacity={0.5} />
          {/* 雷声源 */}
          <g transform={`translate(255,352) scale(${nodeK(0)})`} opacity={nodeK(0)}>
            <StormCloud x={0} y={-26} s={0.5} bolt={1} rain={0} />
            <text x={0} y={112} textAnchor="middle" fill={P.cream} fontSize={24}
              fontFamily={FONT_HEAD} fontWeight={800}>深夜炸雷</text>
          </g>
          {/* 耳朵 */}
          <g transform={`translate(575,378) scale(${nodeK(1)})`} opacity={nodeK(1)}>
            <SoundArcs x={-86} y={0} col={P.wave} n={2} r0={24} gap={24} a0={-50} a1={50} />
            <EarIcon x={0} y={-8} s={0.9} />
            <text x={0} y={112} textAnchor="middle" fill={P.cream} fontSize={24}
              fontFamily={FONT_HEAD} fontWeight={800}>耳朵照常收</text>
          </g>
          {/* 脑干 */}
          <g transform={`translate(770,382) scale(${nodeK(2)})`} opacity={nodeK(2)}>
            <circle r={34} fill={P.cardLite} stroke={P.rain} strokeWidth={4} />
            <text x={0} y={9} textAnchor="middle" fill={P.rain} fontSize={24}
              fontFamily={FONT_HEAD} fontWeight={800}>脑干</text>
            <text x={0} y={112} textAnchor="middle" fill={P.muted} fontSize={23}
              fontFamily={FONT_HEAD}>信号一路送达</text>
          </g>
          {/* 丘脑闸门 */}
          <g transform={`translate(1040,382) scale(${nodeK(3)})`} opacity={nodeK(3)}>
            <ThalamusGate x={0} y={-6} s={0.92} k={gateK} />
            <g transform="translate(0,140)">
              <text x={0} y={0} textAnchor="middle" fill={gateK > 0.5 ? P.coral : P.muted}
                fontSize={24} fontFamily={FONT_HEAD} fontWeight={800}>
                {gateK > 0.5 ? '深睡期·关门' : '浅睡·门还开着'}
              </text>
            </g>
          </g>
          {/* 皮层 */}
          <g transform={`translate(1330,382) scale(${nodeK(4)})`} opacity={nodeK(4)}>
            <BrainIcon x={0} y={-14} s={0.86} col={P.brain} />
            <text x={0} y={112} textAnchor="middle" fill={P.muted} fontSize={24}
              fontFamily={FONT_HEAD}>皮层·没人拆信</text>
          </g>
          {/* 行进信号点：在闸门前堆住 */}
          {[0, 1, 2].map((i) => {
            const x = dotX(i);
            const nearGate = x > 940;
            return (
              <circle key={i} cx={x} cy={390} r={10} fill={nearGate ? P.coral : P.wave}
                opacity={(0.55 + i * 0.2) * clamp01(nodeK(2))} />
            );
          })}
          {/* 拦截标记 */}
          <g transform={`translate(1010,316) scale(${blockK})`} opacity={blockK}>
            <circle r={26} fill={P.coral} />
            <path d="M -10 -10 L 10 10 M 10 -10 L -10 10" stroke="#fff" strokeWidth={7}
              strokeLinecap="round" />
          </g>
          <text x={1130} y={322} fill={P.coral} fontSize={25} fontFamily={FONT_HEAD}
            fontWeight={800} opacity={blockK}>
            拦在门口 · 不上报
          </text>
        </g>

        {/* ---- 左下：人照睡 ---- */}
        <g opacity={clamp01(botK)}>
          <rect x={110} y={570} width={830} height={290} rx={36} fill={P.card} stroke={P.mint} strokeWidth={4} />
          <text x={525} y={622} textAnchor="middle" fill={P.cream} fontSize={30}
            fontFamily={FONT_HEAD} fontWeight={800}>外面炸雷 · 里面照睡</text>
          <g transform="translate(500,740)">
            <BedSleep x={0} y={0} s={0.82} />
            <g transform="translate(232,-58)">
              <Zzz x={0} y={0} k={1} col={P.sky} s={0.9} />
            </g>
            <g transform="translate(-296,-118)">
              <Bolt x={0} y={0} s={0.62} />
            </g>
          </g>
          <text x={525} y={832} textAnchor="middle" fill={P.mint} fontSize={25}
            fontFamily={FONT_HEAD} fontWeight={800}>不是听不见 · 是听见了也不上报</text>
        </g>

        {/* ---- 右下：皮层仍录得反应 ---- */}
        <g opacity={clamp01(botK)}>
          <rect x={980} y={570} width={830} height={290} rx={36} fill={P.card} stroke={P.wave} strokeWidth={4} />
          <text x={1395} y={622} textAnchor="middle" fill={P.cream} fontSize={30}
            fontFamily={FONT_HEAD} fontWeight={800}>皮层其实录得声音反应</text>
          {/* EEG 带 */}
          <rect x={1060} y={652} width={660} height={96} rx={18} fill={P.bgDeep} />
          <path d={wavePath(1080, 1700, 700, 10, 0.9, f * 0.12)} stroke={P.wave}
            strokeWidth={4.5} fill="none" opacity={0.85} />
          {/* 声音到达的尖峰 */}
          <path d={`M 1270 700 L 1286 ${700 - 56 * blip - 8} L 1302 700`} stroke={P.bolt}
            strokeWidth={6} fill="none" strokeLinecap="round" strokeLinejoin="round" />
          <text x={1395} y={786} textAnchor="middle" fill={P.muted} fontSize={24}
            fontFamily={FONT_HEAD}>电反应还在（尖峰） · 只是没触发"该醒了"</text>
          <g transform="translate(1760,700)">
            <FaceIcon x={0} y={0} s={0.66} mode="sleep" />
          </g>
          <text x={1395} y={832} textAnchor="middle" fill={P.wave} fontSize={25}
            fontFamily={FONT_HEAD} fontWeight={800}>信号到了门口 · 没被送进去</text>
        </g>
      </svg>
      <Stamp text="听见了 · 没上报" x={800} y={92} at={b1 + 6} color={P.coral} size={46} rot={-4} />
      <Chip text="不是隔音 · 是门卫在分拣" x={1290} y={100} at={b1 + 40} accent={P.bolt} />
    </FadeIn>
  );
};

/* ============ S1 唤醒阈值有刻度：跨栏图 + 天生档位对照 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1); // ≈189f N1/N2/N3 门槛
  const b2 = beat(2); // ≈390f 同样的雷撞深睡/浅睡

  const cardK = spring({frame: f - 8, fps, config: {damping: 13, stiffness: 120}});
  const hurdleK = (i: number) => clamp01(spring({frame: f - 30 - i * 16, fps, config: {damping: 12, stiffness: 140}}));
  // 雷球跨栏：b1 后从左侧出发，飞过 N1、N2，撞上 N3
  const fly = clamp01((f - b1 - 10) / 70);
  const hitK = clamp01(spring({frame: f - b1 - 84, fps, config: {damping: 9, stiffness: 170}}));
  const shake = hitK * Math.sin(f * 1.1) * 5;
  const pairK = (i: number) => clamp01(spring({frame: f - b2 - i * 18, fps, config: {damping: 12, stiffness: 150}}));

  const ground = 742;
  const hurdles = [
    {x: 350, h: 96, col: P.n1, label: 'N1 · 一点动静就醒', wake: true},
    {x: 610, h: 190, col: P.n2, label: 'N2 · 得再大点声', wake: true},
    {x: 870, h: 296, col: P.n3, label: 'N3 · 抱起都不醒', wake: false},
  ];
  // 雷球轨迹：匀速右移，y 520 直线；到 N3 前停
  const boltX = 210 + fly * 590;
  const boltY = 516;

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={1} name={sec.name} accent={P.coral} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* ---- 左卡：跨栏阈值 ---- */}
        <g opacity={clamp01(cardK)}>
          <rect x={110} y={190} width={1050} height={660} rx={40} fill={P.card} stroke={P.coral} strokeWidth={4} />
          <text x={635} y={248} textAnchor="middle" fill={P.cream} fontSize={33}
            fontFamily={FONT_HEAD} fontWeight={800}>
            唤醒阈值 = 门槛高度 · 随睡眠阶段换挡
          </text>
          {/* 地面 */}
          <line x1={180} y1={ground} x2={1090} y2={ground} stroke={P.muted} strokeWidth={5} />
          {/* 三道栏 */}
          {hurdles.map((h, i) => (
            <g key={i} transform={`translate(${h.x},${ground}) scale(1,${hurdleK(i)})`} opacity={hurdleK(i)}>
              <rect x={-56} y={-h.h} width={112} height={20} rx={10} fill={h.col} />
              <rect x={-10} y={-h.h} width={20} height={h.h} fill={h.col} opacity={0.85} />
              <rect x={-44} y={-14} width={88} height={14} rx={7} fill={P.cardLite} />
            </g>
          ))}
          {/* 栏下标签 */}
          {hurdles.map((h, i) => (
            <g key={`t${i}`} opacity={hurdleK(i)}>
              <text x={h.x} y={790} textAnchor="middle" fill={h.col} fontSize={26}
                fontFamily={FONT_HEAD} fontWeight={900}>
                {h.label.split('·')[0]}
              </text>
              <text x={h.x} y={826} textAnchor="middle" fill={P.muted} fontSize={22}
                fontFamily={FONT_HEAD}>
                {h.label.split('·')[1]}
              </text>
            </g>
          ))}
          {/* 音量参考虚线：同一个雷 */}
          <line x1={190} y1={boltY} x2={820} y2={boltY} stroke={P.bolt} strokeWidth={4}
            strokeDasharray="10 12" opacity={0.55} />
          <text x={200} y={boltY - 18} fill={P.bolt} fontSize={23} fontFamily={FONT_HEAD}
            fontWeight={700}>同一个雷的音量</text>
          {/* 飞行雷球 */}
          <g transform={`translate(${Math.min(boltX, 830) + shake},${boltY})`}>
            <Bolt x={0} y={0} s={0.66} rot={90} />
          </g>
          {/* 越过 N1/N2 的醒眼 */}
          <g opacity={fly > 0.28 ? 1 : 0}>
            <FaceIcon x={350} y={330} s={0.7} mode="wake" />
            <text x={350} y={408} textAnchor="middle" fill={P.n1} fontSize={22}
              fontFamily={FONT_HEAD} fontWeight={800}>翻过去→醒</text>
          </g>
          <g opacity={fly > 0.62 ? 1 : 0}>
            <FaceIcon x={610} y={330} s={0.7} mode="wake" />
            <text x={610} y={408} textAnchor="middle" fill={P.n2} fontSize={22}
              fontFamily={FONT_HEAD} fontWeight={800}>勉强翻过去</text>
          </g>
          {/* 撞 N3 */}
          <g transform={`translate(872,${boltY - 92}) scale(${hitK})`} opacity={hitK}>
            <circle r={30} fill={P.coral} />
            <path d="M -12 -12 L 12 12 M 12 -12 L -12 12" stroke="#fff" strokeWidth={8}
              strokeLinecap="round" />
          </g>
          <text x={872} y={boltY - 140} textAnchor="middle" fill={P.coral} fontSize={26}
            fontFamily={FONT_HEAD} fontWeight={900} opacity={hitK}>
            雷也翻不过去
          </text>
          <g opacity={hitK}>
            <FaceIcon x={1030} y={330} s={0.7} mode="sleep" />
            <text x={1030} y={408} textAnchor="middle" fill={P.n3} fontSize={22}
              fontFamily={FONT_HEAD} fontWeight={800}>继续睡</text>
          </g>
        </g>

        {/* ---- 右卡：天生档位 ---- */}
        <g opacity={clamp01(cardK)}>
          <rect x={1200} y={190} width={610} height={660} rx={40} fill={P.card} stroke={P.n3} strokeWidth={4} />
          <text x={1505} y={248} textAnchor="middle" fill={P.cream} fontSize={33}
            fontFamily={FONT_HEAD} fontWeight={800}>
            档位还是天生的
          </text>
          {/* 低阈值 */}
          <g transform={`translate(0,${(1 - pairK(0)) * 40})`} opacity={pairK(0)}>
            <rect x={1240} y={310} width={530} height={190} rx={28} fill={P.bgLite}
              stroke={P.n1} strokeWidth={4} />
            <FaceIcon x={1330} y={400} s={0.95} mode="wake" />
            <SoundArcs x={1396} y={400} col={P.n1} n={2} r0={20} gap={20} a0={-55} a1={55} />
            <text x={1440} y={380} fill={P.n1} fontSize={27} fontFamily={FONT_HEAD} fontWeight={800}>
              低阈值体质
            </text>
            <text x={1440} y={424} fill={P.muted} fontSize={24} fontFamily={FONT_HEAD}>
              一点声音就睁眼
            </text>
            <text x={1440} y={462} fill={P.muted} fontSize={22} fontFamily={FONT_HEAD}>
              不是神经衰弱 · 是配置
            </text>
          </g>
          {/* 高阈值 */}
          <g transform={`translate(0,${(1 - pairK(1)) * 40})`} opacity={pairK(1)}>
            <rect x={1240} y={530} width={530} height={190} rx={28} fill={P.bgLite}
              stroke={P.n3} strokeWidth={4} />
            <FaceIcon x={1330} y={620} s={0.95} mode="sleep" />
            <g transform="translate(1330,566)">
              <Bolt x={0} y={-8} s={0.4} />
            </g>
            <text x={1440} y={600} fill={P.n3} fontSize={27} fontFamily={FONT_HEAD} fontWeight={800}>
              高阈值体质
            </text>
            <text x={1440} y={644} fill={P.muted} fontSize={24} fontFamily={FONT_HEAD}>
              雷都打不动
            </text>
            <text x={1440} y={682} fill={P.muted} fontSize={22} fontFamily={FONT_HEAD}>
              跟年龄·缺觉·药物都有关
            </text>
          </g>
          <text x={1505} y={786} textAnchor="middle" fill={P.cream} fontSize={25}
            fontFamily={FONT_HEAD} fontWeight={800} opacity={pairK(1)}>
            同一个雷 · 撞上哪一档算哪一档
          </text>
        </g>
      </svg>
      <Stamp text="阈值有刻度" x={690} y={92} at={b1 + 8} color={P.coral} size={48} rot={-4} />
      <Chip text="不是睡死 · 是门槛高" x={1260} y={100} at={b2 + 10} accent={P.n3} />
    </FadeIn>
  );
};

/* ============ S2 纺锤波是隔音棉：滤网剖面 + 哈佛散点实验 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1); // ≈289f “哈佛团队…”

  const cardK = spring({frame: f - 8, fps, config: {damping: 13, stiffness: 120}});
  const wallK = clamp01(spring({frame: f - 70, fps, config: {damping: 13, stiffness: 120}}));
  const calmK = clamp01((f - 120) / 20);
  const plotK = spring({frame: f - b1 - 6, fps, config: {damping: 12, stiffness: 130}});
  const lineK = clamp01((f - b1 - 60) / 50);
  const dotK = (i: number) => clamp01(spring({frame: f - b1 - 30 - i * 8, fps, config: {damping: 11, stiffness: 170}}));
  const badgeK = clamp01(spring({frame: f - b1 - 150, fps, config: {damping: 10, stiffness: 160}}));

  // 信号点：穿不过滤网，在墙前消失
  const sigX = (i: number) => 240 + ((f * 5.5 + i * 130) % 330);

  // 散点数据（纺锤波密度 ↑ → 睡稳 ↑）
  const pts = [
    {x: 1300, y: 592}, {x: 1360, y: 560}, {x: 1420, y: 540}, {x: 1480, y: 500},
    {x: 1540, y: 516}, {x: 1600, y: 452}, {x: 1660, y: 420}, {x: 1700, y: 388},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={2} name={sec.name} accent={P.vio} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* ---- 左卡：纺锤波滤网 ---- */}
        <g opacity={clamp01(cardK)}>
          <rect x={110} y={190} width={980} height={660} rx={40} fill={P.card} stroke={P.vio} strokeWidth={4} />
          <text x={600} y={248} textAnchor="middle" fill={P.cream} fontSize={33}
            fontFamily={FONT_HEAD} fontWeight={800}>
            丘脑在 N2 造纺锤波 · 当滤网用
          </text>
          {/* 声源 */}
          <g transform="translate(200,470)">
            <StormCloud x={0} y={-40} s={0.42} bolt={0} rain={0} />
            <SoundArcs x={56} y={0} col={P.wave} n={3} r0={24} gap={26} a0={-55} a1={55} />
            <text x={0} y={110} textAnchor="middle" fill={P.muted} fontSize={23}
              fontFamily={FONT_HEAD}>噪音信号</text>
          </g>
          {/* 行进点：到墙前停下变淡 */}
          {[0, 1, 2].map((i) => {
            const x = sigX(i);
            const near = x > 500;
            return <circle key={i} cx={x} cy={470} r={9} fill={P.wave}
              opacity={near ? 0.15 : 0.9} />;
          })}
          {/* 滤网墙：三条纺锤波叠成帘 */}
          <g transform={`translate(620,486) scale(${wallK},1)`} opacity={wallK}>
            <rect x={-88} y={-170} width={176} height={340} rx={26} fill={P.vio} opacity={0.12} />
            <rect x={-88} y={-170} width={176} height={340} rx={26} fill="none"
              stroke={P.vio} strokeWidth={4} strokeDasharray="10 10" />
            {[-96, 0, 96].map((dy, i) => (
              <path key={i} d={spindlePath(-72, 72, dy, 34, 7, f * 0.3 + i * 2)}
                stroke={P.vio} strokeWidth={6} fill="none" strokeLinecap="round" />
            ))}
            <text x={0} y={-196} textAnchor="middle" fill={P.vio} fontSize={26}
              fontFamily={FONT_HEAD} fontWeight={900}>纺锤波 11–16Hz</text>
            <text x={0} y={196} textAnchor="middle" fill={P.vio} fontSize={24}
              fontFamily={FONT_HEAD} fontWeight={800}>把信号挡在皮层外</text>
          </g>
          {/* 墙后：皮层安静 */}
          <g opacity={calmK}>
            <BrainIcon x={880} y={452} s={1.05} col={P.brain} />
            <text x={880} y={552} textAnchor="middle" fill={P.mint} fontSize={25}
              fontFamily={FONT_HEAD} fontWeight={800}>皮层没被吵到</text>
            <g transform="translate(952,404)">
              <Zzz x={0} y={0} k={0.9} col={P.sky} s={0.7} />
            </g>
          </g>
          <text x={600} y={800} textAnchor="middle" fill={P.muted} fontSize={25}
            fontFamily={FONT_HEAD}>
            像隔音棉 · 也像门卫 · 专挡感官信号
          </text>
        </g>

        {/* ---- 右卡：哈佛 2010 散点 ---- */}
        <g transform={`translate(${(1 - clamp01(plotK)) * 60},0)`} opacity={clamp01(plotK)}>
          <rect x={1130} y={190} width={680} height={660} rx={40} fill={P.card} stroke={P.wave} strokeWidth={4} />
          <text x={1470} y={248} textAnchor="middle" fill={P.cream} fontSize={33}
            fontFamily={FONT_HEAD} fontWeight={800}>
            哈佛 2010 · 可以直接预测
          </text>
          {/* 坐标轴 */}
          <line x1={1250} y1={660} x2={1750} y2={660} stroke={P.muted} strokeWidth={4} />
          <line x1={1250} y1={660} x2={1250} y2={320} stroke={P.muted} strokeWidth={4} />
          <path d="M 1250 320 l -10 22 m 10 -22 l 10 22" stroke={P.muted} strokeWidth={4}
            fill="none" strokeLinecap="round" />
          <text x={1230} y={316} textAnchor="end" fill={P.muted} fontSize={22}
            fontFamily={FONT_HEAD}>睡稳</text>
          <text x={1500} y={706} textAnchor="middle" fill={P.muted} fontSize={23}
            fontFamily={FONT_HEAD}>纺锤波密度 →</text>
          {/* 趋势虚线 */}
          <line x1={1280} y1={610} x2={1720} y2={376} stroke={P.vio} strokeWidth={7}
            strokeDasharray="14 12" strokeLinecap="round"
            pathLength={1} strokeDashoffset={1 - lineK} opacity={0.9} />
          {/* 散点 */}
          {pts.map((p, i) => (
            <g key={i} transform={`translate(${p.x},${p.y}) scale(${dotK(i)})`} opacity={dotK(i)}>
              <circle r={13} fill={P.wave} stroke={P.cardLite} strokeWidth={4} />
            </g>
          ))}
          {/* 结论胶囊 */}
          <g transform={`translate(1470,780) scale(${badgeK})`} opacity={badgeK}>
            <rect x={-240} y={-34} width={480} height={62} rx={31} fill={P.bgLite}
              stroke={P.vio} strokeWidth={4} />
            <text x={0} y={10} textAnchor="middle" fill={P.vio} fontSize={27}
              fontFamily={FONT_HEAD} fontWeight={900}>
              密度越高 · 越吵不醒
            </text>
          </g>
        </g>
      </svg>
      <Stamp text="脑电里的隔音棉" x={560} y={92} at={60} color={P.vio} size={46} rot={-4} />
      <Chip text="体质差异有脑电标记" x={1280} y={100} at={b1 + 30} accent={P.wave} />
    </FadeIn>
  );
};

/* ============ S3 大脑是挑着听：妈妈与婴儿 + 分类器分拣 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1); // ≈102f 母亲睡死/婴儿哼一声就醒
  const b2 = beat(2); // ≈347f 分类器 + 习惯调教

  const sceneK = spring({frame: f - 10, fps, config: {damping: 13, stiffness: 120}});
  const cryK = clamp01((f - b1 - 6) / 20);
  const cardK = spring({frame: f - b1 - 30, fps, config: {damping: 12, stiffness: 130}});
  const inK = (i: number) => clamp01(spring({frame: f - b1 - 50 - i * 14, fps, config: {damping: 11, stiffness: 160}}));
  const sortK = clamp01((f - b2 - 4) / 20);
  const stripK = spring({frame: f - b2 - 60, fps, config: {damping: 13, stiffness: 120}});

  const inputs = [
    {label: '雷声', col: P.muted, ok: false},
    {label: '婴儿哭', col: P.coral, ok: true},
    {label: '自己名字', col: P.mint, ok: true},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={3} name={sec.name} accent={P.mint} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* ---- 左：妈妈雷暴里睡死 + 婴儿一哼就醒 ---- */}
        <g opacity={clamp01(sceneK)}>
          <g transform="translate(300,280)">
            <StormCloud x={0} y={0} s={0.55} bolt={1} rain={1} />
          </g>
          <g transform="translate(430,520)">
            <BedSleep x={0} y={0} s={0.92} blanket={P.blanket} />
            <g transform="translate(-70,-116)">
              <Zzz x={0} y={0} k={1} col={P.sky} s={0.85} />
            </g>
          </g>
          <text x={430} y={668} textAnchor="middle" fill={P.muted} fontSize={25}
            fontFamily={FONT_HEAD} fontWeight={800}>妈妈：雷暴里睡死</text>
          {/* 婴儿床 */}
          <g transform="translate(880,470)">
            <BabyCrib x={0} y={0} s={0.95} crying={Math.max(0.25, cryK)} />
          </g>
          <text x={880} y={600} textAnchor="middle" fill={P.coral} fontSize={25}
            fontFamily={FONT_HEAD} fontWeight={800} opacity={cryK}>
            哼一声 · 她立刻醒
          </text>
          {/* 哭声飘向妈妈 */}
          <g opacity={cryK}>
            <path d="M 790 400 Q 640 330 560 380" stroke={P.coral} strokeWidth={6}
              fill="none" strokeDasharray="12 10" strokeLinecap="round" />
            <path d="M 560 380 l 26 -16 m -26 16 l 4 30" stroke={P.coral} strokeWidth={6}
              fill="none" strokeLinecap="round" />
          </g>
        </g>

        {/* ---- 右卡：分类器 ---- */}
        <g transform={`translate(${(1 - clamp01(cardK)) * 60},0)`} opacity={clamp01(cardK)}>
          <rect x={1120} y={190} width={690} height={470} rx={40} fill={P.card} stroke={P.mint} strokeWidth={4} />
          <text x={1465} y={246} textAnchor="middle" fill={P.cream} fontSize={32}
            fontFamily={FONT_HEAD} fontWeight={800}>
            大脑是台分类器 · 按意义分流
          </text>
          {/* 三条输入 */}
          {inputs.map((inp, i) => (
            <g key={i} transform={`translate(0,${(1 - inK(i)) * 30})`} opacity={inK(i)}>
              <rect x={1160} y={300 + i * 92} width={200} height={66} rx={33}
                fill={P.bgLite} stroke={inp.col} strokeWidth={4} />
              <text x={1260} y={343 + i * 92} textAnchor="middle" fill={inp.col}
                fontSize={27} fontFamily={FONT_HEAD} fontWeight={800}>{inp.label}</text>
              <line x1={1362} y1={333 + i * 92} x2={1450} y2={430} stroke={inp.col}
                strokeWidth={5} strokeDasharray="8 8" opacity={0.8} />
            </g>
          ))}
          {/* 分拣箱 */}
          <g transform="translate(1500,430)">
            <rect x={-46} y={-60} width={112} height={120} rx={20} fill={P.cardLite}
              stroke={P.cream} strokeWidth={4} />
            <circle r={20} fill={P.cream} opacity={0.9} transform="translate(10,-18)" />
            <circle r={12} fill={P.mint} transform="translate(26,22)" />
            <circle r={12} fill={P.coral} transform="translate(-4,26)" />
            <text x={10} y={94} textAnchor="middle" fill={P.cream} fontSize={23}
              fontFamily={FONT_HEAD} fontWeight={800}>分拣</text>
          </g>
          {/* 两条输出 */}
          <g opacity={sortK}>
            <line x1={1562} y1={392} x2={1640} y2={340} stroke={P.mint} strokeWidth={5}
              strokeDasharray="8 8" />
            <line x1={1562} y1={468} x2={1640} y2={520} stroke={P.muted} strokeWidth={5}
              strokeDasharray="8 8" />
            <rect x={1646} y={300} width={150} height={66} rx={16} fill={P.mint} opacity={0.16} />
            <rect x={1646} y={300} width={150} height={66} rx={16} fill="none"
              stroke={P.mint} strokeWidth={4} />
            <text x={1721} y={342} textAnchor="middle" fill={P.mint} fontSize={26}
              fontFamily={FONT_HEAD} fontWeight={800}>上报→醒</text>
            <rect x={1646} y={486} width={150} height={66} rx={16} fill={P.muted} opacity={0.12} />
            <rect x={1646} y={486} width={150} height={66} rx={16} fill="none"
              stroke={P.muted} strokeWidth={4} />
            <text x={1721} y={528} textAnchor="middle" fill={P.muted} fontSize={26}
              fontFamily={FONT_HEAD} fontWeight={800}>拦截→睡</text>
          </g>
          <text x={1465} y={622} textAnchor="middle" fill={P.muted} fontSize={23}
            fontFamily={FONT_HEAD} opacity={sortK}>
            有意义的信号 · 优先穿透门卫
          </text>
        </g>

        {/* ---- 底条：能被习惯调教 ---- */}
        <g transform={`translate(0,${(1 - clamp01(stripK)) * 50})`} opacity={clamp01(stripK)}>
          <rect x={110} y={706} width={1700} height={160} rx={32} fill={P.card} stroke={P.rain} strokeWidth={4} />
          {/* 马路车流 */}
          <g transform="translate(260,786)">
            <rect x={-110} y={14} width={220} height={14} rx={7} fill={P.cardLite} />
            {[-70, 10, 80].map((dx, i) => {
              const mx = ((f * 2.4 + i * 90) % 240) - 120;
              return <rect key={i} x={mx} y={-20} width={46} height={26} rx={10}
                fill={[P.rain, P.bolt, P.coral][i]} opacity={0.9} />;
            })}
            <text x={0} y={64} textAnchor="middle" fill={P.muted} fontSize={22}
              fontFamily={FONT_HEAD}>马路边车流</text>
          </g>
          <text x={640} y={770} fill={P.cream} fontSize={29} fontFamily={FONT_HEAD} fontWeight={800}>
            还能被习惯调教
          </text>
          <text x={640} y={818} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            室友打鼾：第一周抓狂 → 一个月后归档"不用醒"
          </text>
          {/* 归档小票 */}
          <g transform="translate(1660,786)">
            <rect x={-66} y={-34} width={132} height={68} rx={14} fill={P.bgLite}
              stroke={P.rain} strokeWidth={4} />
            <text x={0} y={-4} textAnchor="middle" fill={P.rain} fontSize={22}
              fontFamily={FONT_HEAD} fontWeight={800}>重复噪音</text>
            <text x={0} y={24} textAnchor="middle" fill={P.muted} fontSize={20}
              fontFamily={FONT_HEAD}>→不送上去</text>
          </g>
        </g>
      </svg>
      <Stamp text="挑着听 · 不是全关" x={650} y={92} at={b1 + 6} color={P.mint} size={46} rot={-4} />
      <Chip text="哭声能穿透 · 雷声被归档" x={1250} y={96} at={b2 + 10} accent={P.coral} />
    </FadeIn>
  );
};

/* ============ S4 差异从哪来：三层来源 + 90 分钟周期撞窗 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1); // ≈66f 年龄/缺觉/药物
  const b2 = beat(2); // ≈311f 时机·九十分钟一循环

  const cardK = spring({frame: f - 8, fps, config: {damping: 13, stiffness: 120}});
  const rowK = (i: number) => clamp01(spring({frame: f - b1 - i * 18, fps, config: {damping: 12, stiffness: 150}}));
  const graphK = spring({frame: f - b2 - 10, fps, config: {damping: 12, stiffness: 130}});
  const curveK = clamp01((f - b2 - 40) / 80);
  const boltAK = clamp01(spring({frame: f - b2 - 120, fps, config: {damping: 10, stiffness: 160}}));
  const boltBK = clamp01(spring({frame: f - b2 - 165, fps, config: {damping: 10, stiffness: 160}}));
  const noteK = clamp01((f - b2 - 210) / 20);

  const rows = [
    {icon: 'age', col: P.n3, title: '年龄占大头', desc: '小孩深睡厚 · 老人深睡薄'},
    {icon: 'tired', col: P.coral, title: '睡眠压力', desc: '严重缺觉 · 阈值飙升睡更死'},
    {icon: 'drug', col: P.bolt, title: '药物酒精', desc: '安眠药加深 · 咖啡因反向'},
  ];

  // hypnogram 折线：x 790→1720，y 越下越深
  const HY = {wake: 330, light: 420, n2: 500, deep: 640};
  const hypo = `M 800 ${HY.light} L 880 ${HY.n2} L 960 ${HY.deep} L 1040 ${HY.deep}
    L 1120 ${HY.n2} L 1180 ${HY.wake - 10} L 1240 ${HY.n2} L 1320 ${HY.deep}
    L 1400 ${HY.deep} L 1470 ${HY.n2} L 1540 ${HY.wake + 10} L 1620 ${HY.n2}
    L 1700 ${HY.light} L 1720 ${HY.light}`;

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={4} name={sec.name} accent={P.rain} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* ---- 左列：三层来源 ---- */}
        <g opacity={clamp01(cardK)}>
          <rect x={110} y={190} width={570} height={660} rx={40} fill={P.card} stroke={P.rain} strokeWidth={4} />
          <text x={395} y={248} textAnchor="middle" fill={P.cream} fontSize={31}
            fontFamily={FONT_HEAD} fontWeight={800}>
            差异从哪来 · 三层叠加
          </text>
          {rows.map((r, i) => (
            <g key={i} transform={`translate(0,${(1 - rowK(i)) * 34})`} opacity={rowK(i)}>
              <rect x={150} y={300 + i * 175} width={490} height={140} rx={24}
                fill={P.bgLite} stroke={r.col} strokeWidth={4} />
              {/* 图标 */}
              {r.icon === 'age' && (
                <g transform="translate(228,370)">
                  <FaceIcon x={0} y={-16} s={0.62} mode="sleep" hair={P.ink} />
                  <FaceIcon x={0} y={30} s={0.45} mode="wake" hair="#d8d8e0" />
                </g>
              )}
              {r.icon === 'tired' && <Battery x={228} y={370} w={100} h={46} fill={0.18} col={P.coral} />}
              {r.icon === 'drug' && <CupPill x={228} y={366} s={0.9} />}
              <text x={306} y={362 + i * 175} fill={r.col} fontSize={27}
                fontFamily={FONT_HEAD} fontWeight={800}>{r.title}</text>
              <text x={306} y={404 + i * 175} fill={P.muted} fontSize={23}
                fontFamily={FONT_HEAD}>{r.desc}</text>
            </g>
          ))}
          <text x={395} y={816} textAnchor="middle" fill={P.muted} fontSize={23}
            fontFamily={FONT_HEAD} opacity={rowK(2)}>
            天生体质 × 近期状态 × 物质干扰
          </text>
        </g>

        {/* ---- 右卡：90 分钟周期 hypnogram ---- */}
        <g transform={`translate(${(1 - clamp01(graphK)) * 60},0)`} opacity={clamp01(graphK)}>
          <rect x={720} y={190} width={1090} height={660} rx={40} fill={P.card} stroke={P.n2} strokeWidth={4} />
          <text x={1265} y={248} textAnchor="middle" fill={P.cream} fontSize={31}
            fontFamily={FONT_HEAD} fontWeight={800}>
            时机：睡眠约 90 分钟一循环
          </text>
          {/* 深度参考线 */}
          {[
            {y: HY.light, t: '浅睡', col: P.n1},
            {y: HY.n2, t: 'N2', col: P.n2},
            {y: HY.deep, t: '深睡', col: P.n3},
          ].map((lv, i) => (
            <g key={i}>
              <line x1={790} y1={lv.y} x2={1730} y2={lv.y} stroke={lv.col} strokeWidth={3}
                strokeDasharray="6 12" opacity={0.4} />
              <text x={770} y={lv.y + 8} textAnchor="end" fill={lv.col} fontSize={22}
                fontFamily={FONT_HEAD}>{lv.t}</text>
            </g>
          ))}
          {/* 睡眠深度曲线 */}
          <path d={hypo} stroke={P.wave} strokeWidth={8} fill="none" strokeLinecap="round"
            strokeLinejoin="round" pathLength={1} strokeDasharray="1"
            strokeDashoffset={1 - curveK} />
          <text x={1260} y={790} textAnchor="middle" fill={P.muted} fontSize={22}
            fontFamily={FONT_HEAD}>入睡后时间 →</text>
          {/* 雷 A：撞深睡谷 */}
          <g transform={`translate(1000,${HY.deep - 66}) scale(${boltAK})`} opacity={boltAK}>
            <Bolt x={0} y={-40} s={0.6} />
            <line x1={0} y1={0} x2={0} y2={44} stroke={P.coral} strokeWidth={5}
              strokeDasharray="6 8" />
            <circle cx={0} cy={58} r={26} fill={P.coral} />
            <path d="M -10 48 L 10 68 M 10 48 L -10 68" stroke="#fff" strokeWidth={7}
              strokeLinecap="round" />
            <text x={-8} y={112} textAnchor="middle" fill={P.coral} fontSize={24}
              fontFamily={FONT_HEAD} fontWeight={800}>撞深睡 · 没感觉</text>
          </g>
          {/* 雷 B：撞浅睡峰 */}
          <g transform={`translate(1560,${HY.wake - 66}) scale(${boltBK})`} opacity={boltBK}>
            <Bolt x={0} y={-40} s={0.6} />
            <line x1={0} y1={0} x2={0} y2={44} stroke={P.mint} strokeWidth={5}
              strokeDasharray="6 8" />
            <circle cx={0} cy={58} r={26} fill={P.mint} />
            <path d="M -11 58 L -3 66 L 13 50" stroke={P.ink} strokeWidth={7}
              fill="none" strokeLinecap="round" strokeLinejoin="round" />
            <text x={0} y={112} textAnchor="middle" fill={P.mint} fontSize={24}
              fontFamily={FONT_HEAD} fontWeight={800}>撞浅睡 · 立刻醒</text>
          </g>
          <text x={1265} y={830} textAnchor="middle" fill={P.cream} fontSize={26}
            fontFamily={FONT_HEAD} fontWeight={800} opacity={noteK}>
            雷没变 · 撞上哪个窗口 · 运气和生理各一半
          </text>
        </g>
      </svg>
      <Stamp text="运气占一半" x={740} y={92} at={b2 + 8} color={P.rain} size={48} rot={-4} />
      <Chip text="三层叠加 · 别怪神经" x={1250} y={100} at={b1 + 60} accent={P.n3} />
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
        <Stars n={18} from={0} dur={22} />
        <Moon x={1630} y={180} scale={0.8} />
        <g transform={`translate(960,850) scale(${clamp01(pop) * 0.72})`}>
          <g transform="translate(-280,-60)">
            <BedSleep x={0} y={0} s={1} />
          </g>
          <g transform="translate(290,-240)">
            <StormCloud x={0} y={0} s={0.8} bolt={1} rain={0} />
          </g>
          <g transform="translate(-440,-290)">
            <Zzz x={0} y={0} k={1} col={P.sky} />
          </g>
        </g>
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -330}}>
        <div style={{fontSize: 60, fontWeight: 800, color: C.text}}>雷还是那个雷 · 看门卫关到第几格</div>
        <div style={{fontSize: 38, color: C.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E234 = ({data = tl}: {data?: any}) => {
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
