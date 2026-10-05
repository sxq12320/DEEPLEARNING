import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01, rand} from '../lib/style';
import {Backdrop, Stamp, SceneTag, Chip, Subtitles, ProgressBar, FadeIn, Moon, Stars, Sun} from '../lib/atoms';
import tl from '../timeline_221.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：深靛夜蓝 × 月亮暖黄 × 脑电青 ============ */
const P = {
  bg: '#111a3e',
  bgDeep: '#0a0f27',
  bgLite: '#1a2452',
  card: '#26336b',
  panel: '#0c1230',
  rail: '#7f8fc9',
  cyan: '#56e6d2',     // 脑电青
  cyanD: '#2ea795',
  moon: '#ffd97a',     // 月亮暖黄
  moonD: '#dfa94c',
  n1: '#93a5ee',       // N1 浅睡
  n2: '#6f7fd8',       // N2
  n3: '#4a59c4',       // N3 深睡
  n3D: '#333f92',
  rem: '#b48cf2',      // REM 做梦
  remD: '#8a63d8',
  coral: '#ff6b6b',    // 闹钟 / 警告
  coralD: '#c94a4a',
  cream: '#f6efdc',
  ink: '#141331',
  muted: '#9aa7d0',
  cement: '#7d8398',
  cementD: '#5b6072',
};

/* ---------- Zzz ---------- */
const Zzz = ({x, y, k = 1, col = P.moon}: {x: number; y: number; k?: number; col?: string}) => {
  const f = useCurrentFrame();
  return (
    <g transform={`translate(${x},${y})`} opacity={k}>
      {[0, 1, 2].map((i) => {
        const t = (f * 0.9 + i * 20) % 60;
        const p = t / 60;
        return (
          <text key={i} x={i * 24 + Math.sin(p * 3) * 10} y={-p * 72}
            fontSize={24 + i * 13} fill={col} fontFamily={FONT_HEAD}
            fontWeight={700} opacity={0.95 * (1 - p)}>z</text>
        );
      })}
    </g>
  );
};

/* ---------- 闹钟 ---------- */
const Alarm = ({x, y, s = 1, on = 1}: {x: number; y: number; s?: number; on?: number}) => {
  const f = useCurrentFrame();
  const wob = on > 0 ? Math.sin(f * 0.9) * 6 * on : 0;
  const arcO = on > 0 ? 0.35 + 0.55 * Math.abs(Math.sin(f * 0.55)) : 0;
  return (
    <g transform={`translate(${x},${y}) rotate(${wob}) scale(${s})`}>
      {on > 0 && [-1, 1].map((sd) => (
        <g key={sd} opacity={arcO}>
          <path d={`M ${sd * 56} -34 Q ${sd * 84} -50 ${sd * 90} -80`} stroke={P.coral} strokeWidth={5} fill="none" strokeLinecap="round" />
          <path d={`M ${sd * 66} -14 Q ${sd * 98} -26 ${sd * 106} -52`} stroke={P.coral} strokeWidth={4} fill="none" strokeLinecap="round" opacity={0.6} />
        </g>
      ))}
      <circle cx={-33} cy={-56} r={15} fill={P.coral} />
      <circle cx={33} cy={-56} r={15} fill={P.coral} />
      <rect x={-5} y={-72} width={10} height={12} rx={4} fill={P.coralD} />
      <circle r={50} fill={P.cream} stroke={P.coral} strokeWidth={6} />
      {[0, 1, 2, 3].map((i) => (
        <line key={i} x1={0} y1={-40} x2={0} y2={-34} stroke={P.ink} strokeWidth={4}
          transform={`rotate(${i * 90})`} />
      ))}
      <line x1={0} y1={0} x2={0} y2={-28} stroke={P.ink} strokeWidth={5} strokeLinecap="round" />
      <line x1={0} y1={0} x2={20} y2={10} stroke={P.ink} strokeWidth={5} strokeLinecap="round" />
      <circle r={5} fill={P.coral} />
      <line x1={-30} y1={50} x2={-42} y2={68} stroke={P.coral} strokeWidth={7} strokeLinecap="round" />
      <line x1={30} y1={50} x2={42} y2={68} stroke={P.coral} strokeWidth={7} strokeLinecap="round" />
    </g>
  );
};

/* ---------- 床上的睡觉小人 ---------- */
const BedSleeper = ({x, y, s = 1, face = 'sleep', bob = 0}: {
  x: number; y: number; s?: number; face?: 'sleep' | 'groggy' | 'fresh'; bob?: number;
}) => {
  const f = useCurrentFrame();
  const dy = bob > 0 ? Math.sin(f / 7) * 3 * bob : 0;
  return (
    <g transform={`translate(${x},${y + dy}) scale(${s})`}>
      <rect x={-170} y={-8} width={360} height={30} rx={13} fill={P.card} />
      <rect x={-170} y={-118} width={24} height={120} rx={10} fill={P.card} />
      <rect x={-116} y={-52} width={104} height={42} rx={19} fill={P.cream} />
      <path d="M -70 -12 Q 30 -78 172 -20 L 178 -8 L -70 -8 Z" fill={P.n2} stroke={P.n3D} strokeWidth={3} />
      <circle cx={-82} cy={-58} r={29} fill={C.skin} />
      <path d="M -110 -66 Q -82 -96 -54 -66 L -54 -58 Q -82 -78 -110 -58 Z" fill="#4a3550" />
      {face === 'sleep' && (
        <g stroke={P.ink} strokeWidth={3.5} strokeLinecap="round" fill="none">
          <path d="M -96 -56 q 6 7 13 0" />
          <path d="M -76 -56 q 6 7 13 0" />
          <path d="M -88 -40 q 6 5 12 0" />
        </g>
      )}
      {face === 'groggy' && (
        <g>
          <path d="M -97 -60 a 7 7 0 1 1 7 7 a 4 4 0 1 0 -4 -4" stroke={P.ink} strokeWidth={3} fill="none" />
          <path d="M -77 -60 a 7 7 0 1 1 7 7 a 4 4 0 1 0 -4 -4" stroke={P.ink} strokeWidth={3} fill="none" />
          <ellipse cx={-82} cy={-38} rx={8} ry={10} fill={P.ink} />
        </g>
      )}
      {face === 'fresh' && (
        <g>
          <circle cx={-92} cy={-58} r={4.5} fill={P.ink} />
          <circle cx={-72} cy={-58} r={4.5} fill={P.ink} />
          <path d="M -92 -42 q 10 10 20 0" stroke={P.ink} strokeWidth={3.5} fill="none" strokeLinecap="round" />
        </g>
      )}
    </g>
  );
};

/* ---------- 环线小列车 ---------- */
const MiniTrain = ({x, y, rot = 0, s = 1}: {x: number; y: number; rot?: number; s?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`}>
    <rect x={-48} y={-21} width={96} height={42} rx={15} fill={P.moon} stroke={P.moonD} strokeWidth={3.5} />
    <rect x={-36} y={-13} width={19} height={17} rx={4} fill={P.panel} />
    <rect x={-9} y={-13} width={19} height={17} rx={4} fill={P.panel} />
    <rect x={18} y={-13} width={19} height={17} rx={4} fill={P.panel} />
    <circle cx={-26} cy={23} r={7.5} fill={P.ink} />
    <circle cx={26} cy={23} r={7.5} fill={P.ink} />
    <circle cx={49} cy={0} r={5.5} fill={P.cream} />
    <text x={-2} y={-30} textAnchor="middle" fontSize={20} fill={P.moon} fontFamily={FONT_HEAD}>z</text>
  </g>
);

/* ---------- 环线轨道：N1顶 → N2右 → N3底 → REM左 ---------- */
const STN = [
  {k: 'N1', ang: -90, col: P.n1, dx: 0, dy: -32, anchor: 'middle' as const},
  {k: 'N2', ang: 0, col: P.n2, dx: 44, dy: 9, anchor: 'start' as const},
  {k: 'N3', ang: 90, col: P.n3, dx: 0, dy: 48, anchor: 'middle' as const},
  {k: 'REM', ang: 180, col: P.rem, dx: -44, dy: 9, anchor: 'end' as const},
];

const Ring = ({cx, cy, rx, ry, parkedAt, speed = 1.35, labels = true, fontS = 1, dotR = 12}: {
  cx: number; cy: number; rx: number; ry: number; parkedAt?: number; speed?: number;
  labels?: boolean; fontS?: number; dotR?: number;
}) => {
  const f = useCurrentFrame();
  const angDeg = parkedAt !== undefined ? parkedAt : -90 + f * speed;
  const a = angDeg * Math.PI / 180;
  const tx = cx + rx * Math.cos(a);
  const ty = cy + ry * Math.sin(a);
  const rot = Math.atan2(ry * Math.cos(a), -rx * Math.sin(a)) * 180 / Math.PI;
  return (
    <g>
      <ellipse cx={cx} cy={cy} rx={rx} ry={ry} fill="none" stroke={P.rail}
        strokeWidth={7} strokeDasharray="16 13" strokeLinecap="round" />
      {STN.map((s) => {
        const sa = s.ang * Math.PI / 180;
        const sx = cx + rx * Math.cos(sa);
        const sy = cy + ry * Math.sin(sa);
        return (
          <g key={s.k}>
            <circle cx={sx} cy={sy} r={dotR} fill={s.col} stroke={P.cream} strokeWidth={3.5} />
            {labels && (
              <text x={sx + s.dx * fontS} y={sy + s.dy * fontS}
                textAnchor={s.anchor} fill={s.col} fontSize={30 * fontS}
                fontFamily={FONT_HEAD} fontWeight={700}>
                {s.k}
              </text>
            )}
          </g>
        );
      })}
      <MiniTrain x={tx} y={ty} rot={rot} s={0.9} />
    </g>
  );
};

/* ---------- 脑电波形 ---------- */
const Eeg = ({x, y, w, type, col, lw = 5}: {
  x: number; y: number; w: number; type: 'n1' | 'n2' | 'n3' | 'rem'; col: string; lw?: number;
}) => {
  const pts: string[] = [];
  const N = 64;
  for (let i = 0; i <= N; i++) {
    const u = i / N;
    let yy = 0;
    if (type === 'n1') yy = Math.sin(u * Math.PI * 4) * 9 + Math.sin(u * 22) * 2.5;
    else if (type === 'n3') yy = Math.sin(u * Math.PI * 3.2) * 24;
    else if (type === 'n2') {
      const g1 = Math.exp(-((u - 0.3) ** 2) / 0.005);
      const g2 = Math.exp(-((u - 0.72) ** 2) / 0.005);
      yy = Math.sin(u * 250) * 20 * (g1 + g2) + Math.sin(u * 36) * 2.5;
    } else yy = (rand(i, 4) - 0.5) * 26 + Math.sin(u * 30) * 5;
    pts.push(`${(x - w / 2 + u * w).toFixed(1)},${(y + yy).toFixed(1)}`);
  }
  return <polyline points={pts.join(' ')} fill="none" stroke={col} strokeWidth={lw}
    strokeLinecap="round" strokeLinejoin="round" />;
};

/* ---------- 大脑 ---------- */
const Brain = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <path d="M -88 8 C -108 -40 -80 -92 -30 -98 C -12 -116 30 -116 48 -96 C 94 -90 112 -44 94 -4
             C 110 30 94 68 60 76 C 44 98 -8 102 -28 82 C -78 88 -104 48 -88 8 Z"
      fill={P.n2} stroke={P.n3D} strokeWidth={5} />
    <path d="M -58 -52 Q -30 -62 -14 -36" stroke={P.n3D} strokeWidth={4.5} fill="none" strokeLinecap="round" />
    <path d="M 12 -72 Q 42 -60 36 -30" stroke={P.n3D} strokeWidth={4.5} fill="none" strokeLinecap="round" />
    <path d="M -68 18 Q -38 8 -28 38" stroke={P.n3D} strokeWidth={4.5} fill="none" strokeLinecap="round" />
    <path d="M 18 32 Q 52 26 58 54" stroke={P.n3D} strokeWidth={4.5} fill="none" strokeLinecap="round" />
    <path d="M -20 62 Q 4 70 26 60" stroke={P.n3D} strokeWidth={4} fill="none" strokeLinecap="round" />
  </g>
);

/* ---------- 剪刀 ---------- */
const Scissor = ({x, y, rot = 0, s = 1}: {x: number; y: number; rot?: number; s?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`}>
    <line x1={-24} y1={-18} x2={32} y2={12} stroke={P.muted} strokeWidth={6} strokeLinecap="round" />
    <line x1={-24} y1={18} x2={32} y2={-12} stroke={P.muted} strokeWidth={6} strokeLinecap="round" />
    <circle cx={-32} cy={-23} r={11} fill="none" stroke={P.coral} strokeWidth={5} />
    <circle cx={-32} cy={23} r={11} fill="none" stroke={P.coral} strokeWidth={5} />
  </g>
);

/* ---------- 整晚睡眠深度图数据 ---------- */
const LVL = {WAKE: 0.07, REM: 0.28, N1: 0.47, N2: 0.67, N3: 0.9} as const;
const HPTS: [number, keyof typeof LVL][] = [
  [0, 'N1'], [0.02, 'N2'], [0.05, 'N3'], [0.1, 'N3'], [0.13, 'N2'], [0.155, 'REM'], [0.185, 'REM'], [0.205, 'N1'],
  [0.225, 'N2'], [0.25, 'N3'], [0.285, 'N3'], [0.32, 'N2'], [0.35, 'REM'], [0.4, 'REM'], [0.415, 'N1'],
  [0.435, 'N2'], [0.46, 'N3'], [0.485, 'N3'], [0.52, 'N2'], [0.55, 'REM'], [0.61, 'REM'], [0.63, 'N1'],
  [0.65, 'N2'], [0.72, 'N2'], [0.74, 'REM'], [0.8, 'REM'], [0.835, 'N1'],
  [0.85, 'N2'], [0.875, 'REM'], [0.97, 'REM'], [1, 'WAKE'],
];
const REM_SPANS: [number, number][] = [[0.155, 0.185], [0.35, 0.4], [0.55, 0.61], [0.74, 0.8], [0.875, 0.97]];
const N3_SPANS: [number, number][] = [[0.05, 0.1], [0.25, 0.285], [0.46, 0.485]];
const CYC = [0, 0.205, 0.415, 0.63, 0.835, 1];

const hypnoPts = (x: number, y: number, w: number, h: number, tMax = 1) =>
  HPTS.filter((p) => p[0] <= tMax).map(([t, l]) => [x + t * w, y + LVL[l] * h] as const);

const Hypno = ({x, y, w, h, col = P.cyan, lw = 5, tMax = 1, dash}: {
  x: number; y: number; w: number; h: number; col?: string; lw?: number; tMax?: number; dash?: string;
}) => (
  <polyline
    points={hypnoPts(x, y, w, h, tMax).map((p) => p.join(',')).join(' ')}
    fill="none" stroke={col} strokeWidth={lw} strokeLinejoin="round" strokeLinecap="round"
    strokeDasharray={dash} />
);

/* ---------- 小钟面（20分钟扇形） ---------- */
const Clock20 = ({x, y, r = 80}: {x: number; y: number; r?: number}) => (
  <g transform={`translate(${x},${y})`}>
    <circle r={r} fill={P.panel} stroke={P.cream} strokeWidth={5} />
    <path d={`M 0 0 L 0 ${-r + 10} A ${r - 10} ${r - 10} 0 0 1 ${(r - 10) * Math.sin(Math.PI * 2 / 3)} ${-(r - 10) * Math.cos(Math.PI * 2 / 3)} Z`}
      fill={P.cyan} opacity={0.85} />
    {Array.from({length: 12}).map((_, i) => (
      <line key={i} x1={0} y1={-r + 8} x2={0} y2={-r + 16} stroke={P.cream} strokeWidth={3}
        transform={`rotate(${i * 30})`} />
    ))}
    <line x1={0} y1={0} x2={(r - 22) * Math.sin(Math.PI * 2 / 3)} y2={-(r - 22) * Math.cos(Math.PI * 2 / 3)}
      stroke={P.cyan} strokeWidth={6} strokeLinecap="round" />
    <line x1={0} y1={0} x2={0} y2={-r + 26} stroke={P.cream} strokeWidth={4} strokeLinecap="round" />
    <circle r={6} fill={P.cream} />
  </g>
);

/* ---------- 眼睛图标 ---------- */
const EyeIcon = ({x, y, s = 1, open = 1}: {x: number; y: number; s?: number; open?: number}) => {
  const f = useCurrentFrame();
  const px = Math.sin(f / 6) * 14 * open;
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      <ellipse rx={52} ry={30 * open + 4} fill={P.cream} />
      <ellipse rx={52} ry={30 * open + 4} fill="none" stroke={P.ink} strokeWidth={4} />
      <circle cx={px} r={17} fill={P.rem} stroke={P.remD} strokeWidth={4} />
      <circle cx={px + 5} cy={-5} r={5} fill="#fff" opacity={0.9} />
    </g>
  );
};

/* ---------- 星星小闪光 ---------- */
const Spark = ({x, y, s = 1, col = P.moon, k = 1}: {x: number; y: number; s?: number; col?: string; k?: number}) => {
  const f = useCurrentFrame();
  const sc = s * (0.75 + 0.25 * Math.sin(f / 7 + x));
  return (
    <polygon points={`0,-16 4,-4 16,0 4,4 0,16 -4,4 -16,0 -4,-4`}
      transform={`translate(${x},${y}) scale(${sc})`} fill={col} opacity={k} />
  );
};

/* ============ 标题卡 ============ */
const TitleCard = ({data}: {data: any}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 13, mass: 0.7}});
  const o = interpolate(f, [0, 8], [0, 1]);
  const p0 = Math.max(0.001, spring({frame: f - 4, fps, config: {damping: 10, stiffness: 150}}));
  return (
    <AbsoluteFill>
      <Backdrop top={P.bg} bottom={P.bg} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={22} from={0} dur={24} />
        <Moon x={960} y={180} scale={1.05} />
        <g transform={`translate(0,40) scale(${p0} 1)`} opacity={p0}>
          <ellipse cx={960} cy={430} rx={230} ry={82} fill="none" stroke={P.rail}
            strokeWidth={6} strokeDasharray="14 12" strokeLinecap="round" />
          {STN.map((st) => {
            const sa = st.ang * Math.PI / 180;
            return <circle key={st.k} cx={960 + 230 * Math.cos(sa)} cy={430 + 82 * Math.sin(sa)}
              r={7} fill={st.col} />;
          })}
        </g>
        {(() => {
          const a = (-90 + f * 1.2) * Math.PI / 180;
          const rot = Math.atan2(82 * Math.cos(a), -230 * Math.sin(a)) * 180 / Math.PI;
          return <MiniTrain x={960 + 230 * Math.cos(a)} y={470 + 82 * Math.sin(a)} rot={rot} s={0.62} />;
        })()}
        <Spark x={700} y={200} s={0.9} />
        <Spark x={1240} y={150} s={0.7} />
        <Spark x={1150} y={290} s={0.55} col={P.cyan} />
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 330}}>
          <div style={{color: P.moon, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 24}}>
            冷知识 · 第 {data.episode} 期
          </div>
          <div style={{color: C.text, fontSize: 82, fontWeight: 900, fontFamily: FONT_HEAD, letterSpacing: 5}}>
            {data.title}
          </div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ============ S0 哪一站下车：水泥拔人 vs 环线夜班车 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);
  const popL = Math.max(0.001, spring({frame: f - 6, fps, config: {damping: 11, stiffness: 120}}));
  const popR = Math.max(0.001, spring({frame: f - 130, fps, config: {damping: 11, stiffness: 120}}));
  const ringK = clamp01((f - b1) / 24);
  const wakeK = clamp01((f - b2) / 20);
  // 环线上的拽下车报警
  const n3a = (90 * Math.PI) / 180;
  const n3x = 960 + 460 * Math.cos(n3a), n3y = 688 + 132 * Math.sin(n3a);

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={0} name={sec.name} accent={P.moon} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={14} from={0} dur={26} />
        {/* 左：半夜被拽醒 */}
        <g transform={`translate(140,150) scale(${popL}) translate(-140,-150)`}>
          <rect x={140} y={150} width={700} height={272} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          {/* 水泥块 */}
          <rect x={230} y={338} width={250} height={72} rx={10} fill={P.cement} />
          <rect x={230} y={338} width={250} height={18} rx={9} fill={P.cementD} opacity={0.7} />
          <path d="M 268 356 l 26 16 l 22 -12 M 400 372 l 30 -14 l 24 18" stroke={P.cementD}
            strokeWidth={4} fill="none" strokeLinecap="round" />
          {/* 被拔出来的人 */}
          <g transform={`translate(${355 + Math.sin(f / 5) * 2},${308 - 6 - Math.abs(Math.sin(f / 9)) * 5})`}>
            <circle r={27} fill={C.skin} />
            <path d="M -26 -8 Q 0 -34 26 -8 L 26 0 Q 0 -22 -26 0 Z" fill="#4a3550" />
            <path d="M -15 -2 a 6 6 0 1 1 6 6 a 3.5 3.5 0 1 0 -3.5 -3.5" stroke={P.ink} strokeWidth={2.6} fill="none" />
            <path d="M 4 -2 a 6 6 0 1 1 6 6 a 3.5 3.5 0 1 0 -3.5 -3.5" stroke={P.ink} strokeWidth={2.6} fill="none" />
            <ellipse cx={0} cy={14} rx={7} ry={9} fill={P.ink} />
            <line x1={-18} y1={24} x2={-34} y2={-14} stroke={C.skin} strokeWidth={9} strokeLinecap="round" />
            <line x1={18} y1={24} x2={34} y2={-14} stroke={C.skin} strokeWidth={9} strokeLinecap="round" />
          </g>
          {/* 上升箭头 */}
          {[0, 1, 2].map((i) => {
            const t = (f * 1.7 + i * 26) % 78;
            return (
              <path key={i} d="M -16 0 L 0 -14 L 16 0" stroke={P.coral} strokeWidth={6}
                fill="none" strokeLinecap="round"
                transform={`translate(355,${252 - t})`} opacity={clamp01(1 - t / 78)} />
            );
          })}
          <Alarm x={610} y={290} s={0.95} on={1} />
          <text x={490} y={400} textAnchor="middle" fill={P.coral} fontSize={28}
            fontFamily={FONT_HEAD} fontWeight={700}>半夜被拽醒 · 像拔出水泥</text>
        </g>
        {/* 右：自然睁眼 */}
        <g transform={`translate(1010,150) scale(${popR}) translate(-1010,-150)`} opacity={popR > 0.01 ? 1 : 0}>
          <rect x={1010} y={150} width={700} height={272} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <rect x={1050} y={330} width={300} height={26} rx={12} fill={P.card} />
          <rect x={1050} y={238} width={22} height={104} rx={9} fill={P.card} />
          <rect x={1096} y={288} width={100} height={38} rx={17} fill={P.cream} />
          {/* 坐起来伸懒腰的人 */}
          <g transform={`translate(1170,290)`}>
            <rect x={-30} y={0} width={86} height={46} rx={20} fill={P.n2} />
            <circle cx={0} cy={-18} r={27} fill={C.skin} />
            <path d="M -25 -28 Q 0 -52 25 -28 L 25 -20 Q 0 -40 -25 -20 Z" fill="#4a3550" />
            <circle cx={-10} cy={-18} r={4.4} fill={P.ink} />
            <circle cx={10} cy={-18} r={4.4} fill={P.ink} />
            <path d="M -9 -4 q 9 9 18 0" stroke={P.ink} strokeWidth={3.2} fill="none" strokeLinecap="round" />
            <line x1={-22} y1={8} x2={-52} y2={-38} stroke={C.skin} strokeWidth={9} strokeLinecap="round" />
            <line x1={50} y1={8} x2={78} y2={-34} stroke={C.skin} strokeWidth={9} strokeLinecap="round" />
          </g>
          <Sun x={1620} y={228} r={42} />
          <Spark x={1290} y={210} s={0.8} />
          <Spark x={1420} y={250} s={0.6} col={P.cyan} />
          <Spark x={1540} y={330} s={0.7} />
          <text x={1360} y={400} textAnchor="middle" fill={P.cyan} fontSize={28}
            fontFamily={FONT_HEAD} fontWeight={700}>睡够自然醒 · 神清气爽</text>
        </g>
        {/* 环线夜班车道线图（cue1） */}
        <g opacity={ringK}>
          <Ring cx={960} cy={688} rx={460} ry={132} fontS={1} />
          {/* 中心标注 */}
          <text x={960} y={662} textAnchor="middle" fill={P.muted} fontSize={26}
            fontFamily={FONT_HEAD}>环线夜班车</text>
          <text x={960} y={712} textAnchor="middle" fill={P.cream} fontSize={46}
            fontFamily={FONT_HEAD} fontWeight={800}>≈90 分钟 / 圈</text>
          {/* 平线被否掉的小图标 */}
          <g transform="translate(688,704)">
            <line x1={-62} y1={0} x2={62} y2={0} stroke={P.muted} strokeWidth={6} strokeLinecap="round" />
            <text x={0} y={-18} textAnchor="middle" fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>平线</text>
            <line x1={-52} y1={-34} x2={52} y2={26} stroke={P.coral} strokeWidth={7} strokeLinecap="round" />
          </g>
          {/* N3 站旁的闹钟：被拽下车点 */}
          <g opacity={wakeK}>
            <Alarm x={n3x + 108} y={n3y - 32} s={0.62} on={1} />
            <text x={n3x + 178} y={n3y - 22} fill={P.coral} fontSize={27}
              fontFamily={FONT_HEAD} fontWeight={700}>闹钟在这响</text>
          </g>
        </g>
      </svg>
      <Chip text="睡多久 ≠ 醒得爽" x={150} y={136} at={beat(0) + 40} accent={P.moon} />
      <Chip text="一晚 4–5 圈" x={1440} y={76} at={b1 + 30} accent={P.cyan} />
      <Stamp text="差别在下车的那一站" x={660} y={452} at={b1 + 55} color={P.n3} size={42} rot={-4} />
    </FadeIn>
  );
};

/* ============ S1 环线四站：N1→N2→N3→REM 剖面 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const at = (s: number) => Math.round(s * fps);   // 本节只有 1 个 cue，按内容节奏手动对位
  const atN2 = at(4.4), atN3 = at(9.4), atREM = at(13.6);

  const cards = [
    {k: 'N1', col: P.n1, cx: 320, top: 212, type: 'n1' as const,
      l1: '半睡半醒', l2: '碰一下还嘴硬说没睡', icon: 'eye'},
    {k: 'N2', col: P.n2, cx: 755, top: 298, type: 'n2' as const,
      l1: '纺锤波登场', l2: '整晚约一半耗在这站', icon: 'donut'},
    {k: 'N3', col: P.n3, cx: 1190, top: 424, type: 'n3' as const,
      l1: 'delta 大慢波', l2: '血压心率最低 · 最难叫醒', icon: 'bell'},
    {k: 'REM', col: P.rem, cx: 1625, top: 326, type: 'rem' as const,
      l1: '眼球乱转 · 全身被锁', l2: '梦就在这儿开演', icon: 'lock'},
  ];
  const ats = [at(1.0), atN2, atN3, atREM];
  const rs = cards.map((_, i) =>
    Math.max(0.001, spring({frame: f - ats[i], fps, config: {damping: 11, stiffness: 130}})));

  // 列车沿坡道滑到最新一站
  const railPts = cards.map((c) => [c.cx, c.top - 4] as const);
  const tx = railPts[0][0] + (railPts[1][0] - railPts[0][0]) * Math.min(1, rs[1])
    + (railPts[2][0] - railPts[1][0]) * Math.min(1, rs[2]) + (railPts[3][0] - railPts[2][0]) * Math.min(1, rs[3]);
  const ty = railPts[0][1] + (railPts[1][1] - railPts[0][1]) * Math.min(1, rs[1])
    + (railPts[2][1] - railPts[1][1]) * Math.min(1, rs[2]) + (railPts[3][1] - railPts[2][1]) * Math.min(1, rs[3]);

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={1} name={sec.name} accent={P.cyan} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={12} from={0} dur={26} />
        {/* 下行坡道 */}
        <path d={`M 150 ${railPts[0][1] - 26} L ${railPts.map((p) => `${p[0]} ${p[1]}`).join(' L ')} L 1800 ${railPts[3][1] - 40}`}
          fill="none" stroke={P.rail} strokeWidth={8} strokeDasharray="2 20" strokeLinecap="round" />
        <MiniTrain x={tx} y={ty - 30} s={0.8}
          rot={Math.atan2((railPts[Math.min(3, rs.reduce((a, r) => a + (r > 0.5 ? 1 : 0), 0))][1] - ty),
            railPts[Math.min(3, rs.reduce((a, r) => a + (r > 0.5 ? 1 : 0), 0))][0] - tx) * 180 / Math.PI} />
        {cards.map((c, i) => (
          <g key={c.k} transform={`translate(${c.cx},${c.top}) scale(${rs[i]}) translate(${-c.cx},${-c.top})`}
            opacity={clamp01(rs[i] * 2)}>
            {/* 站牌 */}
            <circle cx={c.cx} cy={c.top} r={46} fill={c.col} stroke={P.cream} strokeWidth={5} />
            <text x={c.cx} y={c.top + 13} textAnchor="middle" fill={P.ink} fontSize={c.k === 'REM' ? 27 : 34}
              fontFamily={FONT_HEAD} fontWeight={900}>{c.k}</text>
            {/* 卡片 */}
            <rect x={c.cx - 190} y={c.top + 58} width={380} height={252} rx={18}
              fill={P.panel} stroke={c.col} strokeWidth={4} />
            <rect x={c.cx - 160} y={c.top + 84} width={320} height={88} rx={12} fill={P.bgDeep} />
            <Eeg x={c.cx} y={c.top + 128} w={290} type={c.type} col={c.col} lw={4.5} />
            {/* 小图标 */}
            {c.icon === 'eye' && (
              <g transform={`translate(${c.cx},${c.top + 196})`}>
                <ellipse rx={30} ry={17} fill="none" stroke={P.cream} strokeWidth={4} />
                <path d="M -30 0 Q 0 -20 30 -2" stroke={P.cream} strokeWidth={4} fill="none" />
                <circle cx={4} cy={-2} r={7} fill={c.col} />
              </g>
            )}
            {c.icon === 'donut' && (
              <g transform={`translate(${c.cx},${c.top + 196})`}>
                <circle r={22} fill="none" stroke={P.card} strokeWidth={10} />
                <path d="M 0 -22 A 22 22 0 0 1 0 22" fill="none" stroke={c.col} strokeWidth={10} />
                <text x={0} y={7} textAnchor="middle" fontSize={17} fill={P.cream}
                  fontFamily={FONT_HEAD} fontWeight={700}>50%</text>
              </g>
            )}
            {c.icon === 'bell' && (
              <g transform={`translate(${c.cx},${c.top + 196})`}>
                <path d="M -20 6 Q -20 -16 0 -18 Q 20 -16 20 6 L 26 12 L -26 12 Z"
                  fill="none" stroke={P.cream} strokeWidth={4} strokeLinejoin="round" />
                <circle cx={0} cy={18} r={4} fill={P.cream} />
                <line x1={-24} y1={-22} x2={26} y2={22} stroke={P.coral} strokeWidth={5} strokeLinecap="round" />
              </g>
            )}
            {c.icon === 'lock' && (
              <g transform={`translate(${c.cx},${c.top + 196})`}>
                <ellipse cx={-16} cy={-4} rx={22} ry={13} fill="none" stroke={P.cream} strokeWidth={3.5} />
                <circle cx={-16 + Math.sin(f / 5) * 7} cy={-4} r={6.5} fill={c.col} />
                <rect x={10} y={-4} width={24} height={22} rx={5} fill={P.coral} />
                <path d="M 15 -4 v -8 a 7 7 0 0 1 14 0 v 8" fill="none" stroke={P.coral} strokeWidth={4} />
              </g>
            )}
            <text x={c.cx} y={c.top + 250} textAnchor="middle" fill={P.cream} fontSize={26}
              fontFamily={FONT_HEAD} fontWeight={700}>{c.l1}</text>
            <text x={c.cx} y={c.top + 286} textAnchor="middle" fill={P.muted} fontSize={23}
              fontFamily={FONT_HEAD}>{c.l2}</text>
          </g>
        ))}
        {/* 方向箭头：浅→深→折回 */}
        <text x={140} y={770} fill={P.muted} fontSize={26} fontFamily={FONT_HEAD}>浅</text>
        <path d="M 200 762 L 1700 762" stroke={P.muted} strokeWidth={3} strokeDasharray="6 14" opacity={0.55} />
        <polygon points="1710,762 1688,752 1688,772" fill={P.muted} opacity={0.8} />
        <text x={1740} y={770} fill={P.muted} fontSize={26} fontFamily={FONT_HEAD}>做梦</text>
        <text x={960} y={838} textAnchor="middle" fill={P.muted} fontSize={27} fontFamily={FONT_HEAD}>
          从浅到深再折回来 · 一趟约 90 分钟</text>
      </svg>
      <Chip text="一圈四站 · 路线固定" x={580} y={76} at={at(0.6)} accent={P.cyan} />
      <Chip text="REM：脑醒着 · 身锁住" x={1330} y={76} at={atREM + 25} accent={P.rem} />
      <Stamp text="最难叫醒是 N3" x={180} y={620} at={atN3 + 45} color={P.n3} size={42} rot={-5} />
    </FadeIn>
  );
};

/* ============ S2 整晚排班图：前半夜修身体 / 后半夜做梦 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const panelK = clamp01((f - b1) / 22);
  const slide = (1 - Math.max(0.001, spring({frame: f - b1, fps, config: {damping: 13}}))) * 320;
  const drawK = clamp01(f / 80);

  const X0 = 250, X1 = 1240, Y0 = 250, HH = 430, YB = Y0 + HH;
  const pts = hypnoPts(X0, Y0, X1 - X0, HH);
  const lineStr = pts.map((p) => p.join(',')).join(' ');
  const drawn = Math.max(2, Math.floor(pts.length * drawK));
  const fillStr = `${X0},${YB} ${pts.slice(0, drawn).map((p) => p.join(',')).join(' ')} ${pts[Math.min(drawn, pts.length - 1)][0]},${YB}`;

  const labels: [keyof typeof LVL, string, string][] = [
    ['WAKE', '清醒', P.muted], ['REM', 'REM', P.rem], ['N1', 'N1', P.n1], ['N2', 'N2', P.n2], ['N3', 'N3', '#9aa8f0'],
  ];
  // 右侧小面板坐标
  const MX = 1375, MY = 308, MW = 380, MH = 160;
  const cut = 0.45;

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={2} name={sec.name} accent={P.n1} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 主图面板 */}
        <rect x={140} y={210} width={1170} height={520} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <Moon x={188} y={240} scale={0.36} />
        <Sun x={1248} y={256} r={26} />
        {/* 水平网格 + 左轴标签 */}
        {labels.map(([k, name, col]) => (
          <g key={k}>
            <line x1={X0} y1={Y0 + LVL[k] * HH} x2={X1} y2={Y0 + LVL[k] * HH}
              stroke={P.muted} strokeWidth={2} strokeDasharray="4 14" opacity={0.22} />
            <text x={X0 - 14} y={Y0 + LVL[k] * HH + 9} textAnchor="end" fill={col}
              fontSize={25} fontFamily={FONT_HEAD} fontWeight={700}>{name}</text>
          </g>
        ))}
        {/* 圈数分隔 */}
        {CYC.slice(1, -1).map((t, i) => (
          <line key={i} x1={X0 + t * (X1 - X0)} y1={Y0} x2={X0 + t * (X1 - X0)} y2={YB}
            stroke={P.muted} strokeWidth={2.5} strokeDasharray="8 10" opacity={0.4} />
        ))}
        {CYC.slice(0, -1).map((t, i) => (
          <text key={i} x={X0 + (t + (CYC[i + 1] - t) / 2) * (X1 - X0)} y={Y0 - 14}
            textAnchor="middle" fill={P.muted} fontSize={22} fontFamily={FONT_HEAD}>第{i + 1}圈</text>
        ))}
        {/* REM / N3 色块 */}
        {REM_SPANS.map(([a, b], i) => (
          <rect key={'r' + i} x={X0 + a * (X1 - X0)} y={Y0 + LVL.REM * HH - 16}
            width={(b - a) * (X1 - X0)} height={YB - (Y0 + LVL.REM * HH) + 16}
            fill={P.rem} opacity={0.15 * drawK} />
        ))}
        {N3_SPANS.map(([a, b], i) => (
          <rect key={'n' + i} x={X0 + a * (X1 - X0)} y={Y0 + LVL.N2 * HH}
            width={(b - a) * (X1 - X0)} height={YB - (Y0 + LVL.N2 * HH)}
            fill={P.n3} opacity={0.32 * drawK} />
        ))}
        {/* 曲线本体 */}
        <polygon points={fillStr} fill={P.cyan} opacity={0.1} />
        <polyline points={lineStr} fill="none" stroke={P.cyan} strokeWidth={5}
          strokeLinejoin="round" strokeLinecap="round"
          strokeDasharray="2400" strokeDashoffset={2400 * (1 - drawK)} />
        {/* 底轴 + 时间标签 */}
        <line x1={X0} y1={YB} x2={X1} y2={YB} stroke={P.cream} strokeWidth={3.5} />
        <text x={X0} y={YB + 36} fill={P.muted} fontSize={24} fontFamily={FONT_HEAD}>入睡</text>
        <text x={X1} y={YB + 36} textAnchor="end" fill={P.muted} fontSize={24} fontFamily={FONT_HEAD}>清晨</text>
        {/* 底部注解 */}
        <g transform="translate(288,772)">
          <circle cx={-20} cy={-8} r={13} fill="none" stroke="#9aa8f0" strokeWidth={5} />
          <line x1={-11} y1={1} x2={22} y2={30} stroke="#9aa8f0" strokeWidth={9} strokeLinecap="round" />
          <text x={44} y={0} fill="#9aa8f0" fontSize={27} fontFamily={FONT_HEAD} fontWeight={700}>
            前半夜 · 深睡主场：生长激素修身体</text>
        </g>
        <g transform="translate(880,772)">
          <circle cx={-24} cy={-4} r={10} fill={P.rem} />
          <circle cx={-8} cy={-14} r={7} fill={P.rem} opacity={0.8} />
          <circle cx={4} cy={-22} r={4.5} fill={P.rem} opacity={0.6} />
          <text x={26} y={0} fill={P.rem} fontSize={27} fontFamily={FONT_HEAD} fontWeight={700}>
            后半夜 · REM 拉长：梦挤在天亮前</text>
        </g>
        {/* 右：只睡三四小时面板（cue1） */}
        <g transform={`translate(${slide},0)`} opacity={panelK}>
          <rect x={1340} y={210} width={440} height={520} rx={20} fill={P.panel} stroke={P.coral} strokeWidth={4} />
          <text x={1372} y={262} fill={P.coral} fontSize={31} fontFamily={FONT_HEAD} fontWeight={800}>
            只睡 4 小时就起</text>
          <Scissor x={1716} y={254} rot={-20} s={0.8} />
          {/* 迷你深度图 */}
          <rect x={MX - 12} y={MY - 26} width={MW + 24} height={MH + 58} rx={12} fill={P.bgDeep} />
          <Hypno x={MX} y={MY} w={MW} h={MH} col={P.muted} lw={3.5} dash="5 8" />
          <g clipPath={`url(#s2cut)`}>
            <Hypno x={MX} y={MY} w={MW} h={MH} col={P.cyan} lw={3.5} />
            {REM_SPANS.map(([a, b], i) => (
              <rect key={i} x={MX + a * MW} y={MY + LVL.REM * MH - 8} width={(b - a) * MW}
                height={MY + MH - (MY + LVL.REM * MH) + 8} fill={P.rem} opacity={0.2} />
            ))}
          </g>
          <clipPath id="s2cut"><rect x={MX - 12} y={MY - 28} width={MW * cut + 12} height={MH + 60} /></clipPath>
          {/* 截断线 + 剪刀 */}
          <line x1={MX + cut * MW} y1={MY - 18} x2={MX + cut * MW} y2={MY + MH + 34}
            stroke={P.coral} strokeWidth={4} strokeDasharray="9 8" />
          {/* 被砍区域打叉 */}
          <line x1={MX + cut * MW + 40} y1={MY + LVL.REM * MH - 10} x2={MX + MW - 30} y2={MY + MH + 10}
            stroke={P.coral} strokeWidth={6} strokeLinecap="round" opacity={0.85} />
          <line x1={MX + MW - 30} y1={MY + LVL.REM * MH - 10} x2={MX + cut * MW + 40} y2={MY + MH + 10}
            stroke={P.coral} strokeWidth={6} strokeLinecap="round" opacity={0.85} />
          <text x={1372} y={530} fill={P.cream} fontSize={27} fontFamily={FONT_HEAD}>砍掉的不只是时长</text>
          <text x={1372} y={572} fill={P.coral} fontSize={29} fontFamily={FONT_HEAD} fontWeight={800}>
            是几乎全部的 REM</text>
          {/* 对照条 */}
          <text x={1372} y={622} fill={P.muted} fontSize={23} fontFamily={FONT_HEAD}>睡够：REM ≈ 1/4 晚</text>
          <rect x={1372} y={634} width={360} height={24} rx={12} fill={P.card} />
          <rect x={1372} y={634} width={360 * 0.25 * panelK} height={24} rx={12} fill={P.rem} />
          <text x={1372} y={690} fill={P.muted} fontSize={23} fontFamily={FONT_HEAD}>短睡：REM 只剩零头</text>
          <rect x={1372} y={702} width={360} height={24} rx={12} fill={P.card} />
          <rect x={1372} y={702} width={Math.max(14, 360 * 0.05 * panelK)} height={24} rx={12} fill={P.coral} />
        </g>
      </svg>
      <Chip text="一晚上跑 4–5 圈" x={1350} y={76} at={beat(0) + 30} accent={P.cyan} />
      <Chip text="每圈内容比例都在变" x={140} y={136} at={beat(0) + 75} accent={P.n1} />
      <Stamp text="总时长相同 ≠ 等价" x={620} y={92} at={b1 + 40} color={P.coralD} size={42} rot={3} />
    </FadeIn>
  );
};

/* ============ S3 深睡里被拽醒：睡眠惰性 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const popL = Math.max(0.001, spring({frame: f - 6, fps, config: {damping: 11, stiffness: 120}}));
  const cardBK = clamp01((f - b1) / 20);
  const slideB = (1 - Math.max(0.001, spring({frame: f - b1, fps, config: {damping: 13}}))) * 160;
  const boot = Math.min(0.32, clamp01(f / 200) * 0.32);
  const bootFast = clamp01((f - b1 - 12) / 26);
  const spin = (f * 6) % 360;

  return (
    <FadeIn>
      <Backdrop top={P.bgDeep} bottom={P.bgDeep} />
      <SceneTag index={3} name={sec.name} accent={P.coral} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：停在 N3 被闹钟硬拽 */}
        <g transform={`translate(140,140) scale(${popL}) translate(-140,-140)`}>
          <rect x={140} y={150} width={580} height={710} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={178} y={466} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>此刻停在这一站</text>
          {/* 深度标尺 */}
          <line x1={230} y1={238} x2={230} y2={430} stroke={P.card} strokeWidth={6} strokeLinecap="round" />
          {[{k: 'N1', y: 244, c: P.n1}, {k: 'N2', y: 306, c: P.n2}, {k: 'N3', y: 396, c: P.n3}].map((d) => (
            <g key={d.k}>
              <circle cx={230} cy={d.y} r={11} fill={d.c} />
              <text x={252} y={d.y + 9} fill={d.c} fontSize={26} fontFamily={FONT_HEAD} fontWeight={700}>{d.k}</text>
            </g>
          ))}
          <circle cx={230} cy={396} r={20 + Math.sin(f / 6) * 3} fill="none" stroke={P.coral} strokeWidth={4} />
          <Alarm x={480} y={270} s={0.85} on={1} />
          <path d="M 430 310 C 380 340 320 366 262 388" stroke={P.coral} strokeWidth={4}
            strokeDasharray="7 9" fill="none" opacity={0.75} />
          {/* 床和被拽醒的人 */}
          <rect x={200} y={640} width={420} height={30} rx={13} fill={P.card} />
          <rect x={200} y={540} width={26} height={112} rx={10} fill={P.card} />
          <rect x={262} y={606} width={110} height={40} rx={18} fill={P.cream} />
          <path d="M 308 640 Q 410 574 590 626 L 596 640 Z" fill={P.n2} stroke={P.n3D} strokeWidth={3} />
          <g transform={`translate(${330 - Math.abs(Math.sin(f / 8)) * 8},${560 - Math.abs(Math.sin(f / 8)) * 10}) rotate(${-14 - Math.sin(f / 8) * 5})`}>
            <circle r={30} fill={C.skin} />
            <path d="M -29 -10 Q 0 -36 29 -10 L 29 -2 Q 0 -24 -29 -2 Z" fill="#4a3550" />
            <path d="M -16 -4 a 6.5 6.5 0 1 1 6.5 6.5 a 3.5 3.5 0 1 0 -3.5 -3.5" stroke={P.ink} strokeWidth={2.6} fill="none" />
            <path d="M 4 -4 a 6.5 6.5 0 1 1 6.5 6.5 a 3.5 3.5 0 1 0 -3.5 -3.5" stroke={P.ink} strokeWidth={2.6} fill="none" />
            <ellipse cx={0} cy={16} rx={7} ry={9} fill={P.ink} />
          </g>
          {[0, 1, 2].map((i) => {
            const t = (f * 1.9 + i * 24) % 72;
            return <path key={i} d="M -14 0 L 0 -12 L 14 0" stroke={P.coral} strokeWidth={5}
              fill="none" strokeLinecap="round" transform={`translate(330,${508 - t})`}
              opacity={clamp01(1 - t / 72)} />;
          })}
          {/* 床头柜闹钟 */}
          <rect x={560} y={560} width={110} height={18} rx={7} fill={P.card} />
          <Alarm x={616} y={520} s={0.62} on={1} />
          {/* 开机进度条：卡住 */}
          <text x={180} y={726} fill={P.cream} fontSize={26} fontFamily={FONT_HEAD} fontWeight={700}>
            皮层开机中…</text>
          <rect x={180} y={744} width={420} height={26} rx={13} fill={P.card} />
          <rect x={180} y={744} width={420 * boot * (0.94 + 0.06 * Math.sin(f / 5))} height={26}
            rx={13} fill={P.coral} />
          <text x={612} y={766} fill={P.coral} fontSize={24} fontFamily={FONT_HEAD} fontWeight={700}>
            ≈30% 卡住</text>
          <g transform={`translate(655,757) rotate(${spin})`} opacity={0.9}>
            <path d="M 0 -16 A 16 16 0 0 1 16 0" fill="none" stroke={P.coral} strokeWidth={5} strokeLinecap="round" />
          </g>
          <text x={180} y={822} fill={P.muted} fontSize={24} fontFamily={FONT_HEAD}>
            学名：睡眠惰性 · 懵几分钟到半小时</text>
        </g>
        {/* 右上：大脑低功耗卡 */}
        <rect x={760} y={150} width={1020} height={390} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <text x={800} y={206} fill={P.coral} fontSize={33} fontFamily={FONT_HEAD} fontWeight={800}>
          从 N3 里被硬拽醒</text>
        <text x={800} y={246} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
          皮层还泡在慢波里 · 前额叶没跟上</text>
        <Brain x={1030} y={390} s={1.15} />
        <circle cx={972} cy={352} r={40} fill="none" stroke={P.coral} strokeWidth={4} strokeDasharray="8 7" />
        <text x={972} y={296} textAnchor="middle" fill={P.coral} fontSize={24}
          fontFamily={FONT_HEAD} fontWeight={700}>前额叶</text>
        {/* 仪表条 */}
        <text x={1280} y={330} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>前额叶血流</text>
        <rect x={1280} y={346} width={430} height={24} rx={12} fill={P.card} />
        <rect x={1280} y={346} width={430 * 0.22} height={24} rx={12} fill={P.coral} />
        <text x={1720} y={367} fill={P.coral} fontSize={24} fontFamily={FONT_HEAD} fontWeight={700}>低</text>
        <text x={1280} y={420} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>判断力</text>
        <rect x={1280} y={436} width={430} height={24} rx={12} fill={P.card} />
        <rect x={1280} y={436} width={430 * 0.15} height={24} rx={12} fill={P.coral} />
        <text x={1720} y={457} fill={P.coral} fontSize={24} fontFamily={FONT_HEAD} fontWeight={700}>不在线</text>
        {/* 状态小胶囊 */}
        {['昏沉', '迟钝', '烦躁'].map((t, i) => (
          <g key={t} transform={`translate(${1280 + i * 150},496)`}>
            <rect width={120} height={48} rx={24} fill={P.card} stroke={P.coral} strokeWidth={3} />
            <text x={60} y={33} textAnchor="middle" fill={P.cream} fontSize={25}
              fontFamily={FONT_HEAD} fontWeight={700}>{t}</text>
          </g>
        ))}
        {/* 右下：浅睡/REM尾醒对比（cue1） */}
        <g transform={`translate(0,${slideB})`} opacity={cardBK}>
          <rect x={760} y={570} width={1020} height={290} rx={20} fill={P.panel} stroke={P.cyanD} strokeWidth={4} />
          <text x={800} y={628} fill={P.cyan} fontSize={33} fontFamily={FONT_HEAD} fontWeight={800}>
            从浅睡 / REM 尾巴醒来</text>
          <text x={800} y={668} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            本来就停在站台门口 · 启动快得多</text>
          <EyeIcon x={1030} y={752} s={0.95} />
          <text x={1180} y={726} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>开机</text>
          <rect x={1180} y={742} width={420} height={24} rx={12} fill={P.card} />
          <rect x={1180} y={742} width={Math.max(20, 420 * bootFast)} height={24} rx={12} fill={P.cyan} />
          <text x={1180} y={812} fill={P.cyan} fontSize={27} fontFamily={FONT_HEAD} fontWeight={700}>
            甚至能赶在闹钟响之前睁眼</text>
          <g transform={`translate(1680,752)`}>
            <circle r={36} fill="none" stroke={P.cyan} strokeWidth={6} />
            <path d="M -16 0 L -4 13 L 18 -12" stroke={P.cyan} strokeWidth={7} fill="none"
              strokeLinecap="round" strokeLinejoin="round" />
          </g>
        </g>
      </svg>
      <Chip text="同一闹钟 · 站不同命不同" x={140} y={136} at={beat(0) + 30} accent={P.coral} />
      <Stamp text="夜班老手：先坐两分钟" x={1040} y={92} at={b1 + 60} color={P.n3} size={40} rot={3} />
    </FadeIn>
  );
};

/* ============ S4 别掐表，掐站 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const cardsK = clamp01((f - b1) / 20);
  const cardPop = Math.max(0.001, spring({frame: f - b1, fps, config: {damping: 12}}));

  const sx = (m: number) => 300 + ((m - 50) / 90) * 1260;   // 50→140 分钟刻度
  const dots = [78, 84, 88, 92, 96, 102, 108, 112, 74];

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={4} name={sec.name} accent={P.moon} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 上：周期时长分布尺 */}
        <rect x={140} y={150} width={1640} height={296} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <text x={176} y={424} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
          一圈的长度（分钟）· 同一个人每晚都在漂</text>
        {/* 刻度带 */}
        <rect x={sx(60)} y={300} width={sx(140) - sx(60)} height={10} rx={5} fill={P.card} />
        <rect x={sx(70)} y={292} width={sx(120) - sx(70)} height={26} rx={13} fill={P.n2} opacity={0.85} />
        {[70, 90, 120].map((m) => (
          <g key={m}>
            <line x1={sx(m)} y1={318} x2={sx(m)} y2={336} stroke={P.cream} strokeWidth={4} />
            <text x={sx(m)} y={368} textAnchor="middle" fill={P.cream} fontSize={24}
              fontFamily={FONT_HEAD}>{m}</text>
          </g>
        ))}
        {/* 漂移的小点：不同夜晚 */}
        {dots.map((m, i) => (
          <circle key={i} cx={sx(m)} cy={284 + ((rand(i, 31) - 0.5) * 14)} r={8}
            fill={P.cyan} opacity={0.5 + rand(i, 32) * 0.5} />
        ))}
        {/* 平均90旗标 */}
        <line x1={sx(90)} y1={292} x2={sx(90)} y2={242} stroke={P.moon} strokeWidth={4} />
        <path d={`M ${sx(90)} 242 L ${sx(90) + 52} 254 L ${sx(90)} 266 Z`} fill={P.moon} />
        <text x={sx(90) + 66} y={262} fill={P.moon} fontSize={26} fontFamily={FONT_HEAD} fontWeight={700}>
          平均 90</text>
        {/* 掐表无用：打叉秒表 */}
        <g transform="translate(1700,312)">
          <circle r={46} fill={P.panel} stroke={P.muted} strokeWidth={5} />
          <rect x={-8} y={-62} width={16} height={14} rx={5} fill={P.muted} />
          <line x1={0} y1={0} x2={0} y2={-26} stroke={P.muted} strokeWidth={5} strokeLinecap="round" />
          <circle r={58} fill="none" stroke={P.coral} strokeWidth={7} opacity={0.9} />
          <line x1={-41} y1={41} x2={41} y2={-41} stroke={P.coral} strokeWidth={7} strokeLinecap="round" />
        </g>
        <text x={1700} y={410} textAnchor="middle" fill={P.coral} fontSize={25}
          fontFamily={FONT_HEAD} fontWeight={700}>掐表倒推没用</text>
        {/* 下：两张操作卡（cue1） */}
        <g opacity={cardsK} transform={`translate(0,${(1 - cardPop) * 120})`}>
          {/* 卡1：午睡20分钟 */}
          <rect x={140} y={490} width={750} height={380} rx={20} fill={P.panel} stroke={P.cyanD} strokeWidth={4} />
          <text x={180} y={552} fill={P.cyan} fontSize={33} fontFamily={FONT_HEAD} fontWeight={800}>
            ① 午睡掐 20 分钟</text>
          <text x={180} y={594} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            在深睡门口就掉头 · 醒了不难受</text>
          <Clock20 x={400} y={720} r={78} />
          {/* 掉头示意：小坡道 + U 形箭头 */}
          <g transform="translate(560,690)">
            <path d="M 0 -20 L 90 6 L 160 34" stroke={P.rail} strokeWidth={7}
              strokeDasharray="2 16" strokeLinecap="round" fill="none" />
            <rect x={168} y={4} width={120} height={86} rx={12} fill="none"
              stroke={P.coral} strokeWidth={4} strokeDasharray="9 8" />
            <text x={228} y={56} textAnchor="middle" fill={P.coral} fontSize={24}
              fontFamily={FONT_HEAD} fontWeight={700}>N3 深睡</text>
            <path d="M 158 30 q -34 10 -40 -30" stroke={P.cyan} strokeWidth={7} fill="none" strokeLinecap="round" />
            <polygon points="118,-6 104,10 130,12" fill={P.cyan} />
            <text x={114} y={-34} fill={P.cyan} fontSize={24} fontFamily={FONT_HEAD}>掉头</text>
          </g>
          {/* 卡2：起床时间固定 */}
          <rect x={930} y={490} width={850} height={380} rx={20} fill={P.panel} stroke={P.moonD} strokeWidth={4} />
          <text x={970} y={552} fill={P.moon} fontSize={33} fontFamily={FONT_HEAD} fontWeight={800}>
            ② 起床时间固定</text>
          <text x={970} y={594} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            生物钟才能把每一圈的位置摆稳</text>
          <g transform="translate(1130,700)">
            <Alarm x={0} y={0} s={0.95} on={0} />
            {/* 锚：钉住 */}
            <g transform="translate(0,88)">
              <line x1={0} y1={-6} x2={0} y2={34} stroke={P.moon} strokeWidth={6} strokeLinecap="round" />
              <path d="M -20 18 Q 0 44 20 18" fill="none" stroke={P.moon} strokeWidth={6} strokeLinecap="round" />
              <circle cy={-10} r={6} fill="none" stroke={P.moon} strokeWidth={4} />
            </g>
          </g>
          {/* 三圈对齐 */}
          {[0, 1, 2].map((i) => (
            <g key={i} transform={`translate(1440,${640 + i * 62})`} opacity={0.95}>
              <ellipse rx={150} ry={24} fill="none" stroke={i === 1 ? P.moon : P.rail}
                strokeWidth={i === 1 ? 5 : 4} strokeDasharray="10 9" />
              <circle cx={-150} cy={0} r={8} fill={P.moon} />
            </g>
          ))}
          <line x1={1290} y1={610} x2={1290} y2={782} stroke={P.moon} strokeWidth={3}
            strokeDasharray="5 8" opacity={0.7} />
          <text x={1440} y={836} textAnchor="middle" fill={P.muted} fontSize={24}
            fontFamily={FONT_HEAD}>每天在同一站上车 · 每圈位置对齐</text>
        </g>
      </svg>
      <Chip text="70–120 分钟都正常" x={140} y={136} at={beat(0) + 35} accent={P.cyan} />
      <Chip text="掐表抓住的只有一半" x={540} y={136} at={beat(0) + 95} accent={P.coral} />
      <Stamp text="闹钟替你决定哪站下车" x={1010} y={84} at={b1 + 70} color={P.n3} size={40} rot={-2} />
    </FadeIn>
  );
};

/* ============ 结尾卡 ============ */
const EndCard = () => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 14}});
  const pop = Math.max(0.001, spring({frame: f - 5, fps, config: {damping: 10, stiffness: 140}}));
  const a = -90 * Math.PI / 180;
  return (
    <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
      <Backdrop top={P.bgDeep} bottom={P.bgDeep} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={20} from={0} dur={24} />
        <Moon x={960} y={250} scale={1.2} />
        <g transform={`translate(960,760) scale(${pop})`} opacity={pop}>
          <ellipse cx={0} cy={0} rx={400} ry={112} fill="none" stroke={P.rail}
            strokeWidth={7} strokeDasharray="16 13" strokeLinecap="round" />
          {STN.map((st) => {
            const sa = st.ang * Math.PI / 180;
            return <circle key={st.k} cx={400 * Math.cos(sa)} cy={112 * Math.sin(sa)}
              r={11} fill={st.col} stroke={P.cream} strokeWidth={3} />;
          })}
          {/* 到站停稳的列车 */}
          <MiniTrain x={400 * Math.cos(a)} y={112 * Math.sin(a) - 4} rot={0} s={0.95} />
          <circle cx={400 * Math.cos(a)} cy={112 * Math.sin(a)}
            r={22 + Math.sin(f / 7) * 4} fill="none" stroke={P.moon} strokeWidth={4} />
        </g>
        <Zzz x={1060} y={620} k={0.9} />
        <Spark x={500} y={640} s={0.8} />
        <Spark x={1420} y={600} s={0.65} col={P.cyan} />
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -160}}>
        <div style={{fontSize: 60, fontWeight: 800, color: C.text}}>闹钟替你决定在哪一站下车</div>
        <div style={{fontSize: 36, color: P.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E221 = ({data = tl}: {data?: any}) => {
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
