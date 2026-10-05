import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01} from '../lib/style';
import {Backdrop, Ripple, Stamp, SceneTag, Chip, Subtitles, ProgressBar, FadeIn} from '../lib/atoms';
import tl from '../timeline_229.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：极夜监控蓝 × 脑电荧光绿 × 梦粉 × 琥珀（纯色平涂） ============ */
const P = {
  bg: '#101f38',
  bgDeep: '#0a1526',
  bgLite: '#18294a',
  panel: '#20345c',
  panelHi: '#2c467a',
  screen: '#0b1730',
  cream: '#f3eedd',
  ink: '#0d1830',
  card: '#f6efdc',
  cardInk: '#333a5f',
  eeg: '#57e6a9',
  eegD: '#2fae7c',
  pink: '#ffa1c8',
  pinkD: '#d9739f',
  amber: '#ffc45e',
  amberD: '#d9993a',
  coral: '#ff7a66',
  coralD: '#d95a48',
  sky: '#74c9ff',
  skyD: '#4a93c7',
  violet: '#a78bfa',
  violetD: '#7d63cc',
  brain: '#f6aec6',
  brainD: '#d47e9c',
  skin: '#f2c79b',
  skinD: '#d99f6b',
  pillow: '#eef1fb',
  blanket: '#3f5fa8',
  blanketD: '#2e4479',
  rat: '#c9b6a4',
  ratD: '#a08a76',
  grey: '#93a2c2',
  trash: '#4a5578',
};

/* ---------- 月牙 ---------- */
const Moon = ({x, y, s = 1, op = 1, hole = P.bg}: {x: number; y: number; s?: number; op?: number; hole?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
    <circle r={54} fill={P.amber} />
    <circle cx={24} cy={-14} r={44} fill={hole} />
    <circle cx={-20} cy={20} r={7} fill={P.amberD} />
    <circle cx={-33} cy={-4} r={4.5} fill={P.amberD} />
  </g>
);

/* ---------- 脑电波形（prog=画线进度） ---------- */
const EEGWave = ({x, y, w = 640, amp = 1, color = P.eeg, prog = 1, sw = 6, phase = 0}: {
  x: number; y: number; w?: number; amp?: number; color?: string; prog?: number; sw?: number; phase?: number;
}) => {
  const f = useCurrentFrame();
  const n = 44;
  const d = Array.from({length: n}).map((_, i) => {
    const px = (i / (n - 1)) * w;
    const py = Math.sin(i * 0.85 + f * 0.16 + phase) * 0.5 + Math.sin(i * 0.3 + f * 0.34 + phase) * 0.5;
    return `${i === 0 ? 'M' : 'L'} ${px.toFixed(1)} ${(-py * 52 * amp).toFixed(1)}`;
  }).join(' ');
  return (
    <g transform={`translate(${x},${y})`}>
      <path d={d} fill="none" stroke={color} strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round"
        pathLength={1400} strokeDasharray={1400} strokeDashoffset={1400 * (1 - prog)} />
    </g>
  );
};

/* ---------- 露头盖脑的夜班小人 ---------- */
const HeadBrain = ({x, y, s = 1, op = 1, on = 0, z = 0, rot = 0}: {
  x: number; y: number; s?: number; op?: number; on?: number; z?: number; rot?: number;
}) => {
  const f = useCurrentFrame();
  const spark = 0.5 + 0.5 * Math.sin(f / 6);
  return (
    <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`} opacity={op}>
      {/* 露出的脑 */}
      <g transform="translate(0,-52) scale(0.76)">
        <path d="M -84 14 C -104 -18 -92 -62 -58 -74 C -46 -104 4 -110 32 -92 C 70 -100 100 -72 96 -34
          C 118 -14 108 24 82 36 C 80 64 44 80 16 66 C -10 80 -56 62 -62 34 C -82 30 -92 24 -84 14 Z"
          fill={P.brain} stroke={P.brainD} strokeWidth={6} strokeLinejoin="round" />
        <g stroke={P.brainD} strokeWidth={5} fill="none" strokeLinecap="round">
          <path d="M -66 -34 Q -42 -56 -16 -38" />
          <path d="M -52 12 Q -24 -6 4 14" />
          <path d="M 12 -62 Q 42 -78 66 -56" />
          <path d="M 26 34 Q 52 18 76 36" />
        </g>
      </g>
      {/* 夜班安全帽 + 头灯 */}
      <g transform="translate(0,-146)" opacity={on}>
        <path d="M -36 4 A 40 36 0 0 1 36 4" fill={P.amber} stroke={P.amberD} strokeWidth={5} />
        <rect x={-46} y={0} width={92} height={14} rx={7} fill={P.amberD} />
        <circle cx={0} cy={-18} r={8} fill={P.cream} stroke={P.amberD} strokeWidth={4} />
      </g>
      {/* 夜班火花 */}
      <g opacity={on * spark} stroke={P.amber} strokeWidth={5} strokeLinecap="round">
        <line x1={-88} y1={-118} x2={-98} y2={-132} />
        <line x1={88} y1={-114} x2={100} y2={-126} />
      </g>
      {/* 下半张脸（头盖打开） */}
      <path d="M -100 -2 A 100 100 0 0 0 100 -2 Z" fill={P.skin} stroke={P.skinD}
        strokeWidth={6} strokeLinejoin="round" />
      <g stroke={P.ink} strokeWidth={4.5} fill="none" strokeLinecap="round">
        <path d="M -46 34 Q -35 44 -24 34" />
        <path d="M 24 34 Q 35 44 46 34" />
        <path d="M -9 60 Q 0 66 9 60" />
      </g>
      {z > 0 && [0, 1, 2].map((i) => {
        const t = (f / 30 + i * 0.9) % 3;
        return (
          <text key={i} x={72 + i * 38 + t * 20} y={-150 - t * 48} fill={P.sky} fontSize={40 - i * 7}
            fontFamily={FONT_HEAD} fontWeight={800} opacity={z * (1 - t / 3)}>Z</text>
        );
      })}
    </g>
  );
};

/* ---------- 眼球（REM 狂转） ---------- */
const Eyeball = ({x, y, s = 1, op = 1, dart = 1}: {x: number; y: number; s?: number; op?: number; dart?: number}) => {
  const f = useCurrentFrame();
  const dx = (Math.sin(f * 0.55) * 14 + Math.sin(f * 0.21) * 6) * dart;
  return (
    <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
      <ellipse rx={54} ry={36} fill={P.pillow} stroke={P.panelHi} strokeWidth={6} />
      <g transform={`translate(${dx},0)`}>
        <circle r={17} fill={P.sky} stroke={P.skyD} strokeWidth={4} />
        <circle r={8} fill={P.ink} />
        <circle cx={-5} cy={-5} r={3.5} fill="#fff" />
      </g>
      <path d="M -54 -12 A 60 44 0 0 1 54 -12" fill="none" stroke={P.skinD} strokeWidth={5} />
    </g>
  );
};

/* ---------- 挂锁 ---------- */
const LockIcon = ({x, y, s = 1, op = 1}: {x: number; y: number; s?: number; op?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
    <path d="M -20 -6 A 20 20 0 0 1 20 -6" fill="none" stroke={P.amberD} strokeWidth={9} />
    <rect x={-32} y={-6} width={64} height={56} rx={12} fill={P.amber} stroke={P.amberD} strokeWidth={5} />
    <circle cy={14} r={8} fill={P.ink} />
    <rect x={-3} y={18} width={6} height={14} rx={3} fill={P.ink} />
  </g>
);

/* ---------- 海马（小海马造型） ---------- */
const Hippo = ({x, y, s = 1, op = 1}: {x: number; y: number; s?: number; op?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
    <path d="M -50 40 Q -40 -40 30 -50" fill="none" stroke={P.eeg} strokeWidth={30} strokeLinecap="round" />
    <circle cx={44} cy={-52} r={22} fill={P.eeg} />
    <line x1={60} y1={-56} x2={88} y2={-64} stroke={P.eeg} strokeWidth={12} strokeLinecap="round" />
    <circle cx={50} cy={-58} r={5} fill={P.ink} />
    <path d="M -50 40 Q -62 52 -72 40" fill="none" stroke={P.eegD} strokeWidth={12} strokeLinecap="round" />
  </g>
);

/* ---------- 老鼠 ---------- */
const Rat = ({x, y, s = 1, flip = false, op = 1, sleep = false}: {
  x: number; y: number; s?: number; flip?: boolean; op?: number; sleep?: boolean;
}) => (
  <g transform={`translate(${x},${y}) scale(${flip ? -s : s},${s})`} opacity={op}>
    <path d="M -34 10 Q -70 16 -78 -6" fill="none" stroke={P.ratD} strokeWidth={7} strokeLinecap="round" />
    <ellipse rx={42} ry={26} fill={P.rat} stroke={P.ratD} strokeWidth={5} />
    <circle cx={38} cy={-10} r={20} fill={P.rat} stroke={P.ratD} strokeWidth={5} />
    <circle cx={34} cy={-26} r={9} fill={P.rat} stroke={P.ratD} strokeWidth={4} />
    <circle cx={34} cy={-26} r={4} fill={P.pink} />
    {sleep
      ? <path d="M 40 -14 Q 46 -10 52 -14" fill="none" stroke={P.ink} strokeWidth={3.5} strokeLinecap="round" />
      : <circle cx={46} cy={-13} r={4} fill={P.ink} />}
    <circle cx={57} cy={-6} r={4} fill={P.pinkD} />
    <line x1={-16} y1={24} x2={-16} y2={32} stroke={P.ratD} strokeWidth={6} strokeLinecap="round" />
    <line x1={16} y1={24} x2={16} y2={32} stroke={P.ratD} strokeWidth={6} strokeLinecap="round" />
  </g>
);

/* ---------- 神经元序列条：lit[i] 0..1 ---------- */
const PipStrip = ({x, y, s = 1, gap = 130, colors, lit}: {
  x: number; y: number; s?: number; gap?: number; colors: string[]; lit: number[];
}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <line x1={0} y1={0} x2={(colors.length - 1) * gap} y2={0} stroke={P.panelHi}
      strokeWidth={5} strokeDasharray="8 8" />
    {colors.map((c, i) => {
      const l = clamp01(lit[i] ?? 0);
      return (
        <g key={i}>
          <circle cx={i * gap} cy={0} r={14 + 9 * l} fill={c} opacity={0.32 + 0.68 * l}
            stroke={c} strokeWidth={4} />
          <text x={i * gap} y={8} textAnchor="middle" fontSize={21} fill={l > 0.55 ? P.ink : c}
            fontFamily={FONT_HEAD} fontWeight={800}>{i + 1}</text>
        </g>
      );
    })}
  </g>
);

/* ---------- 记忆卡片 ---------- */
const MemCard = ({x, y, rot = 0, op = 1, s = 1, label, tag, color}: {
  x: number; y: number; rot?: number; op?: number; s?: number; label: string; tag?: string; color: string;
}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`} opacity={op}>
    <rect x={-75} y={-46} width={150} height={92} rx={14} fill={P.card} stroke={color} strokeWidth={5} />
    <rect x={-75} y={-46} width={150} height={30} rx={14} fill={color} />
    <rect x={-75} y={-30} width={150} height={14} fill={color} />
    {tag
      ? <text x={0} y={-25} textAnchor="middle" fontSize={19} fill={P.ink} fontFamily={FONT_HEAD} fontWeight={800}>{tag}</text>
      : <circle cx={0} cy={-31} r={5} fill={P.ink} opacity={0.5} />}
    <text x={0} y={20} textAnchor="middle" fontSize={34} fill={P.cardInk} fontFamily={FONT_HEAD} fontWeight={800}>{label}</text>
  </g>
);

/* ---------- 传送带 ---------- */
const Belt = ({x0, x1, y, op = 1}: {x0: number; x1: number; y: number; op?: number}) => {
  const f = useCurrentFrame();
  return (
    <g opacity={op}>
      <line x1={x0} y1={y - 16} x2={x1} y2={y - 16} stroke={P.panelHi} strokeWidth={6} />
      <line x1={x0} y1={y + 16} x2={x1} y2={y + 16} stroke={P.panelHi} strokeWidth={6} />
      {Array.from({length: Math.floor((x1 - x0) / 90)}).map((_, i) => (
        <g key={i} transform={`translate(${x0 + 45 + i * 90},${y}) rotate(${f * 6})`}>
          <circle r={13} fill={P.panel} stroke={P.grey} strokeWidth={4} />
          <line x1={-8} y1={0} x2={8} y2={0} stroke={P.grey} strokeWidth={3} />
        </g>
      ))}
    </g>
  );
};

/* ---------- 长期存储库 ---------- */
const Vault = ({x, y, s = 1, op = 1}: {x: number; y: number; s?: number; op?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
    <rect x={-110} y={-70} width={220} height={140} rx={18} fill={P.panel} stroke={P.eeg} strokeWidth={6} />
    <rect x={-110} y={-70} width={220} height={36} rx={18} fill={P.eeg} />
    <rect x={-110} y={-52} width={220} height={18} fill={P.eeg} />
    <text x={0} y={-44} textAnchor="middle" fontSize={25} fill={P.ink} fontFamily={FONT_HEAD} fontWeight={800}>长期存储</text>
    <rect x={-70} y={30} width={140} height={26} rx={9} fill={P.screen} />
    <rect x={-80} y={-6} width={160} height={10} rx={5} fill={P.panelHi} />
  </g>
);

/* ---------- 垃圾桶 ---------- */
const Trash = ({x, y, s = 1, op = 1}: {x: number; y: number; s?: number; op?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
    <path d="M -70 -48 L 70 -48 L 54 62 L -54 62 Z" fill={P.trash} stroke={P.grey}
      strokeWidth={5} strokeLinejoin="round" />
    <rect x={-84} y={-64} width={168} height={20} rx={10} fill={P.grey} />
    <text x={0} y={98} textAnchor="middle" fontSize={26} fill={P.grey} fontFamily={FONT_HEAD}>扔掉</text>
  </g>
);

/* ---------- 仪表盘 ---------- */
const Gauge = ({x, y, s = 1, v = 0.5, label, op = 1}: {
  x: number; y: number; s?: number; v?: number; label: string; op?: number;
}) => {
  const pt = (k: number, r: number) => {
    const a = (180 - k * 180) * Math.PI / 180;
    return [Math.cos(a) * r, -Math.sin(a) * r];
  };
  const arc = (k0: number, k1: number, r: number) =>
    `M ${pt(k0, r)[0]} ${pt(k0, r)[1]} A ${r} ${r} 0 0 1 ${pt(k1, r)[0]} ${pt(k1, r)[1]}`;
  const [nx, ny] = pt(v, 82);
  return (
    <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
      <path d={arc(0, 0.35, 110)} stroke={P.eeg} strokeWidth={18} fill="none" strokeLinecap="round" />
      <path d={arc(0.35, 1, 110)} stroke={P.coralD} strokeWidth={18} fill="none" strokeLinecap="round" opacity={0.55} />
      <text x={pt(0, 134)[0]} y={pt(0, 134)[1] + 8} textAnchor="middle" fontSize={23} fill={P.grey} fontFamily={FONT_HEAD}>0</text>
      <text x={pt(1, 134)[0]} y={pt(1, 134)[1] + 8} textAnchor="middle" fontSize={23} fill={P.grey} fontFamily={FONT_HEAD}>高</text>
      <line x1={0} y1={0} x2={nx} y2={ny} stroke={P.cream} strokeWidth={8} strokeLinecap="round" />
      <circle r={12} fill={P.cream} stroke={P.panelHi} strokeWidth={4} />
      <text x={0} y={54} textAnchor="middle" fontSize={26} fill={P.cream} fontFamily={FONT_HEAD} fontWeight={700}>{label}</text>
    </g>
  );
};

/* ---------- 开关（onK 1=ON → 0=OFF） ---------- */
const Toggle = ({x, y, s = 1, onK = 1, op = 1}: {x: number; y: number; s?: number; onK?: number; op?: number}) => {
  const kx = interpolate(onK, [0, 1], [-40, 40]);
  return (
    <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
      <rect x={-88} y={-32} width={176} height={64} rx={32} fill={P.panel} stroke={P.panelHi} strokeWidth={5} />
      <rect x={-88} y={-32} width={176} height={64} rx={32} fill={P.coral} opacity={0.16 * onK} />
      <circle cx={kx} cy={0} r={26} fill={onK > 0.5 ? P.coral : P.grey} stroke={P.cream} strokeWidth={4} />
      <text x={kx > 0 ? -42 : 44} y={9} textAnchor="middle" fontSize={23} fill={P.cream}
        fontFamily={FONT_HEAD} fontWeight={800}>{onK > 0.5 ? 'ON' : 'OFF'}</text>
    </g>
  );
};

/* ---------- 烦心事云（calm 0=风暴 → 1=平静） ---------- */
const MoodCloud = ({x, y, s = 1, calm = 0}: {x: number; y: number; s?: number; calm?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <g opacity={1 - calm}>
      <path d="M -78 16 A 38 38 0 0 1 -36 -32 A 46 46 0 0 1 44 -26 A 36 36 0 0 1 80 16 Z"
        fill={P.coral} stroke={P.coralD} strokeWidth={5} />
      <path d="M -10 30 L -26 66 L -6 60 L -20 96" fill="none" stroke={P.amber}
        strokeWidth={8} strokeLinecap="round" strokeLinejoin="round" />
    </g>
    <g opacity={calm}>
      <path d="M -78 16 A 38 38 0 0 1 -36 -32 A 46 46 0 0 1 44 -26 A 36 36 0 0 1 80 16 Z"
        fill={P.sky} stroke={P.skyD} strokeWidth={5} />
      <circle cx={58} cy={-46} r={22} fill={P.amber} stroke={P.amberD} strokeWidth={4} />
    </g>
  </g>
);

/* ---------- 杏仁核 ---------- */
const Amygdala = ({x, y, s = 1, op = 1, hot = 0}: {x: number; y: number; s?: number; op?: number; hot?: number}) => {
  const f = useCurrentFrame();
  const pulse = 0.5 + 0.5 * Math.sin(f / 8);
  return (
    <g transform={`translate(${x},${y}) rotate(-24) scale(${s})`} opacity={op}>
      <ellipse rx={64} ry={40} fill={P.coral} stroke={P.coralD} strokeWidth={6} />
      <ellipse cx={-12} cy={-8} rx={34} ry={20} fill={P.coralD} opacity={0.4} />
      <ellipse rx={92} ry={60} fill="none" stroke={P.coral} strokeWidth={4}
        strokeDasharray="10 9" opacity={hot * pulse} />
    </g>
  );
};

/* ---------- 神经元 + 突触芽（f/t0 驱动逐个长出，shrink 统一缩） ---------- */
const SynTree = ({x, y, s = 1, f, t0, shrink = 1, hot = -1}: {
  x: number; y: number; s?: number; f: number; t0: number; shrink?: number; hot?: number;
}) => {
  const buds = [
    {a: -75, r: 340, R: 30}, {a: -45, r: 300, R: 40}, {a: -15, r: 330, R: 26},
    {a: 15, r: 310, R: 44}, {a: 50, r: 320, R: 28}, {a: 78, r: 300, R: 34},
  ];
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      {buds.map((b, i) => {
        const rad = b.a * Math.PI / 180;
        const bx = Math.cos(rad) * b.r;
        const by = Math.sin(rad) * b.r;
        const g = Math.max(0.001, spring({frame: f - t0 - i * 9, fps: 30, config: {damping: 12, stiffness: 130}}));
        const rr = Math.max(0.001, b.R * g * shrink * (i === hot ? 1.2 : 1));
        return (
          <g key={i}>
            <line x1={0} y1={0} x2={bx} y2={by} stroke={P.grey} strokeWidth={7}
              strokeLinecap="round" opacity={0.65 * g} />
            <circle cx={bx} cy={by} r={rr} fill={i === hot ? P.amber : P.violet}
              stroke={i === hot ? P.amberD : P.violetD} strokeWidth={5} />
            {i === hot && shrink < 0.9 && (
              <circle cx={bx} cy={by} r={rr + 12} fill="none" stroke={P.amber}
                strokeWidth={4} strokeDasharray="8 7" />
            )}
          </g>
        );
      })}
      <circle r={62} fill={P.violet} stroke={P.violetD} strokeWidth={6} />
      <circle r={30} fill={P.violetD} opacity={0.55} />
      <circle cx={-14} cy={-10} r={9} fill={P.cream} opacity={0.85} />
    </g>
  );
};

/* ---------- 任务工单 ---------- */
const TaskTicket = ({x, y, at, n, label, accent}: {
  x: number; y: number; at: number; n: string; label: string; accent: string;
}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const g = Math.max(0.001, spring({frame: f - at, fps, config: {damping: 12, stiffness: 140}}));
  if (g <= 0.01 && f < at) return null;
  return (
    <g transform={`translate(${x},${y}) scale(${g})`} opacity={clamp01(g)}>
      <rect x={-128} y={-52} width={256} height={104} rx={18} fill={P.panel} />
      <rect x={-128} y={-52} width={14} height={104} rx={7} fill={accent} />
      <circle cx={-68} cy={0} r={30} fill="none" stroke={accent} strokeWidth={4} />
      <text x={-68} y={11} textAnchor="middle" fontSize={30} fill={accent} fontFamily={FONT_HEAD} fontWeight={800}>{n}</text>
      <text x={34} y={12} textAnchor="middle" fontSize={34} fill={P.cream} fontFamily={FONT_HEAD} fontWeight={800}>{label}</text>
    </g>
  );
};

/* ============ 标题卡 ============ */
const TitleCard = ({data}: {data: any}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 13, mass: 0.7}});
  const o = interpolate(f, [0, 8], [0, 1]);
  const p0 = Math.max(0.001, spring({frame: f - 4, fps, config: {damping: 10, stiffness: 150}}));
  const bob = Math.sin(f / 15) * 6;
  return (
    <AbsoluteFill>
      <Backdrop top={P.bg} bottom={P.bg} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <circle cx={960} cy={330} r={214} fill={P.cream} />
        <circle cx={960} cy={330} r={192} fill={P.panel} />
        <g transform={`translate(960,${330 + bob * 0.4}) scale(${p0})`}>
          <g transform="translate(-960,-330)">
            <HeadBrain x={960} y={352} s={1.0} on={1} z={1} />
            <Moon x={852} y={232} s={0.58} hole={P.panel} />
            {/* 三颗记忆碎片 */}
            {[0, 1, 2].map((i) => (
              <circle key={i} cx={905 + i * 55} cy={498} r={11}
                fill={[P.eeg, P.amber, P.pink][i]} />
            ))}
            <line x1={894} y1={498} x2={1026} y2={498} stroke={P.panelHi} strokeWidth={4}
              strokeDasharray="2 14" />
          </g>
        </g>
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 300}}>
          <div style={{color: P.eeg, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 26}}>
            冷知识 · 第 {data.episode} 期
          </div>
          <div style={{color: C.text, fontSize: 92, fontWeight: 900, fontFamily: FONT_HEAD, letterSpacing: 6}}>
            {data.title}
          </div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ============ S0 大脑换了个夜班：露头小人在床上 + 脑电监护 + 夜班三件事 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b0 = beat(0), b1 = beat(1), b2 = beat(2);

  const on = clamp01((f - b0 - 15) / 25);                    // 大脑上岗
  const eegAmp = 0.22 + 0.78 * clamp01((f - b1) / 45);       // REM 脑电活泼
  const eyeK = clamp01((f - b1 - 55) / 20);
  const lockK = Math.max(0.001, spring({frame: f - b1 - 90, fps, config: {damping: 11, stiffness: 150}}));

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={0} name={sec.name} accent={P.eeg} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：床 + 露头夜班小人 */}
        <ellipse cx={430} cy={872} rx={330} ry={26} fill={P.bgDeep} />
        <ellipse cx={300} cy={756} rx={96} ry={36} fill={P.pillow} stroke={P.grey} strokeWidth={5} />
        <rect x={420} y={742} width={360} height={86} rx={20} fill={P.blanket} stroke={P.blanketD} strokeWidth={5} />
        <rect x={420} y={742} width={360} height={24} rx={12} fill={P.blanketD} opacity={0.55} />
        <rect x={440} y={828} width={26} height={50} rx={8} fill={P.blanketD} />
        <rect x={720} y={828} width={26} height={50} rx={8} fill={P.blanketD} />
        <HeadBrain x={330} y={664} s={1.12} rot={-10} on={on} z={1} />
        <g opacity={on}>
          <path d="M 470 480 L 420 560" stroke={P.cream} strokeWidth={4} strokeDasharray="8 8" />
          <text x={480} y={472} fill={P.cream} fontSize={29} fontFamily={FONT_HEAD} fontWeight={700}>大脑 · 夜班中</text>
        </g>
        {/* 右上：脑电监护屏 */}
        <rect x={990} y={185} width={850} height={262} rx={22} fill={P.panel} stroke={P.panelHi} strokeWidth={5} />
        <rect x={1020} y={216} width={790} height={200} rx={12} fill={P.screen} />
        <text x={1042} y={254} fill={P.eeg} fontSize={26} fontFamily={FONT_HEAD} fontWeight={700}>脑电 EEG</text>
        <text x={1790} y={254} textAnchor="end" fill={P.grey} fontSize={24} fontFamily={FONT_HEAD}
          opacity={eegAmp > 0.6 ? 1 : 0.35}>活泼得接近清醒</text>
        <EEGWave x={1045} y={330} w={740} amp={eegAmp} prog={clamp01((f - 10) / 50)} />
        {/* 右中：眼球狂转 + 肌肉锁死 */}
        <g opacity={eyeK}>
          <Eyeball x={1150} y={600} s={1.15} dart={1} />
          <text x={1150} y={692} textAnchor="middle" fill={P.cream} fontSize={28}
            fontFamily={FONT_HEAD} fontWeight={700}>眼球狂转</text>
        </g>
        <g opacity={clamp01(lockK)}>
          <rect x={1380} y={588} width={170} height={52} rx={26} fill={P.skin} stroke={P.skinD} strokeWidth={5} />
          <circle cx={1560} cy={614} r={36} fill={P.skin} stroke={P.skinD} strokeWidth={5} />
          <g transform={`translate(1470,540) scale(${lockK})`}>
            <LockIcon x={0} y={0} s={0.85} />
          </g>
          <text x={1490} y={692} textAnchor="middle" fill={P.cream} fontSize={28}
            fontFamily={FONT_HEAD} fontWeight={700}>肌肉锁死</text>
        </g>
        {/* 右下：夜班三件事工单 */}
        <TaskTicket x={1135} y={800} at={b2 + 8} n="①" label="重播记忆" accent={P.eeg} />
        <TaskTicket x={1425} y={800} at={b2 + 22} n="②" label="挑拣入库" accent={P.amber} />
        <TaskTicket x={1715} y={800} at={b2 + 36} n="③" label="处理情绪" accent={P.pink} />
      </svg>
      <Chip text="睡着 ≠ 放假 · 只是换班" x={100} y={150} at={b0 + 15} accent={P.eeg} />
      <Chip text="内部狂欢 · 外部失联" x={100} y={240} at={b1 + 110} accent={P.pink} />
      <Stamp text="REM · 快速眼动期" x={1440} y={120} at={b1 + 15} color={P.skyD} size={44} rot={4} />
    </FadeIn>
  );
};

/* ============ S1 老鼠梦里重跑迷宫：白天记录 vs 睡着重放 对照 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b0 = beat(0), b1 = beat(1);

  const zoneX = [200, 350, 500, 650, 800];
  const zoneC = [P.eeg, P.amber, P.pink, P.sky, P.violet];
  // 老鼠往返
  const p = (((f - b0 - 20) / 200) % 2 + 2) % 2;
  const tri = Math.abs(p - 1);
  const ratX = 200 + tri * 600;
  const flip = p < 1;
  const litDay = zoneX.map((zx) => clamp01(1 - Math.abs(ratX - zx) / 95));
  // 夜里快速重放
  const nightK = clamp01((f - 170) / 25);
  const act = Math.floor(Math.max(0, f - 210) / 5) % 5;
  const litNight = zoneX.map((_, i) => (f > 210 ? (i === act ? 1 : (i === (act + 4) % 5 ? 0.4 : 0.12)) : 0));
  const fastK = Math.max(0.001, spring({frame: f - 215, fps, config: {damping: 10, stiffness: 140}}));

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={1} name={sec.name} accent={P.amber} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：白天跑迷宫 */}
        <rect x={80} y={180} width={840} height={600} rx={24} fill={P.panel} />
        <circle cx={136} cy={238} r={20} fill={P.amber} />
        {[0, 45, 90, 135, 180, 225, 270, 315].map((a) => {
          const r = a * Math.PI / 180;
          return <line key={a} x1={136 + Math.cos(r) * 30} y1={238 + Math.sin(r) * 30}
            x2={136 + Math.cos(r) * 42} y2={238 + Math.sin(r) * 42} stroke={P.amber}
            strokeWidth={5} strokeLinecap="round" />;
        })}
        <text x={176} y={250} fill={P.cream} fontSize={32} fontFamily={FONT_HEAD} fontWeight={800}>白天 · 跑直线迷宫</text>
        {/* 跑道 */}
        <line x1={170} y1={470} x2={830} y2={470} stroke={P.panelHi} strokeWidth={64} strokeLinecap="round" />
        <line x1={170} y1={470} x2={830} y2={470} stroke={P.grey} strokeWidth={4} strokeDasharray="14 14" opacity={0.6} />
        {zoneX.map((zx, i) => (
          <circle key={i} cx={zx} cy={470} r={16} fill={zoneC[i]} stroke={P.cream} strokeWidth={3} opacity={0.9} />
        ))}
        <Rat x={ratX} y={424} s={1.1} flip={flip} />
        {/* 电极记录点 */}
        <line x1={850} y1={330} x2={850} y2={420} stroke={P.grey} strokeWidth={5} />
        <circle cx={850} cy={326} r={9} fill={P.coral} />
        <text x={842} y={300} textAnchor="end" fill={P.grey} fontSize={24} fontFamily={FONT_HEAD}>海马电极</text>
        <PipStrip x={230} y={628} gap={135} colors={zoneC} lit={litDay} />
        <text x={230} y={700} fill={P.grey} fontSize={25} fontFamily={FONT_HEAD}>海马神经元 · 按顺序放电</text>
        {/* 中缝小箭头 */}
        <path d="M 928 620 L 950 634 L 928 648" fill="none" stroke={P.cream} strokeWidth={5}
          strokeLinecap="round" strokeLinejoin="round" opacity={nightK} />
        {/* 右：睡着后重放 */}
        <g opacity={nightK}>
          <rect x={960} y={180} width={880} height={600} rx={24} fill={P.panel} />
          <Moon x={1024} y={228} s={0.44} hole={P.panel} />
          <text x={1074} y={240} fill={P.cream} fontSize={32} fontFamily={FONT_HEAD} fontWeight={800}>夜里 · 睡着以后</text>
          <ellipse cx={1140} cy={520} rx={180} ry={18} fill={P.bgDeep} />
          <Rat x={1140} y={468} s={1.5} sleep op={1} />
          {[0, 1, 2].map((i) => {
            const t = (f / 32 + i * 0.9) % 3;
            return (
              <text key={i} x={1210 + i * 34 + t * 18} y={410 - t * 40} fill={P.sky}
                fontSize={34 - i * 6} fontFamily={FONT_HEAD} fontWeight={800}
                opacity={nightK * (1 - t / 3)}>Z</text>
            );
          })}
          <PipStrip x={1090} y={628} gap={135} colors={zoneC} lit={litNight} />
          <text x={1090} y={700} fill={P.grey} fontSize={25} fontFamily={FONT_HEAD}>同一批神经元 · 原顺序 · 时间压缩</text>
          <g transform={`translate(1640,560) scale(${fastK})`}>
            <rect x={-88} y={-34} width={176} height={66} rx={16} fill={P.amber} />
            <text x={0} y={11} textAnchor="middle" fontSize={30} fill={P.ink}
              fontFamily={FONT_HEAD} fontWeight={800}>×N 快进</text>
          </g>
        </g>
      </svg>
      <Chip text="威尔逊 & 麦克诺顿 · 1994" x={100} y={140} at={b0 + 20} accent={P.amber} />
      <Chip text="清醒时记录 → 睡着后重放" x={100} y={836} at={200} accent={P.eeg} />
      <Stamp text="复的还是原片" x={760} y={120} at={b1 + 12} color={P.eegD} size={46} rot={-4} />
      {f > b1 + 12 && <Ripple x={860} y={160} startFrame={b1 + 12} color={P.eeg} />}
    </FadeIn>
  );
};

/* ============ S2 回放其实是筛片：慢波节拍 + 传送带分拣 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b0 = beat(0), b1 = beat(1), b2 = beat(2);

  const waveK = clamp01((f - b1) / 25);
  const beltOn = f > b1 + 25;
  const CARDS = [
    {label: '段子', tag: '情绪', color: P.pink, keep: true},
    {label: '单词', tag: '', color: P.grey, keep: false},
    {label: '路线', tag: '挂钩', color: P.sky, keep: true},
    {label: '公式', tag: '常用', color: P.amber, keep: true},
  ];
  const spawn0 = b1 + 30;

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={2} name={sec.name} accent={P.sky} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 顶部慢波 */}
        <g opacity={waveK}>
          <path d="M 240 240 C 330 155 420 155 510 240 C 600 325 690 325 780 240 C 870 155 960 155 1050 240
            C 1140 325 1230 325 1320 240 C 1410 155 1500 155 1590 240 C 1625 268 1650 285 1680 292"
            fill="none" stroke={P.sky} strokeWidth={26} strokeLinecap="round" opacity={0.22} />
          <path d="M 240 240 C 330 155 420 155 510 240 C 600 325 690 325 780 240 C 870 155 960 155 1050 240
            C 1140 325 1230 325 1320 240 C 1410 155 1500 155 1590 240 C 1625 268 1650 285 1680 292"
            fill="none" stroke={P.sky} strokeWidth={11} strokeLinecap="round"
            strokeDasharray="44 26" strokeDashoffset={-f * 4} />
          <text x={240} y={352} fill={P.sky} fontSize={27} fontFamily={FONT_HEAD} fontWeight={700}>
            慢波 · 大规模同步放电 = 搬运的节拍</text>
        </g>
        {/* 海马吐卡片 */}
        <ellipse cx={300} cy={640} rx={150} ry={18} fill={P.bgDeep} />
        <Hippo x={292} y={560} s={1.35} />
        <text x={300} y={676} textAnchor="middle" fill={P.eeg} fontSize={27} fontFamily={FONT_HEAD} fontWeight={700}>
          海马 · 短期记忆</text>
        {/* 传送带 */}
        <Belt x0={420} x1={1620} y={560} op={beltOn ? 1 : 0.45} />
        {/* 存储库 + 垃圾桶 */}
        <Vault x={1660} y={420} s={1} op={waveK} />
        <Trash x={1660} y={712} s={1} op={waveK} />
        {/* 飞行中的记忆卡 */}
        {beltOn && Array.from({length: 5}).map((_, i) => {
          const s0 = spawn0 + i * 75;
          const card = CARDS[i % 4];
          const x = 420 + (f - s0) * 5.2;
          if (f < s0 || x > 1780) return null;
          const p2 = clamp01((x - 1560) / 95);
          let cx = x, cy = 505, rot = 0, sc = 1, op = clamp01((f - s0) / 6);
          if (card.keep) {
            cx = x + (1660 - x) * p2;
            cy = 505 - 43 * p2;
            sc = 1 - 0.45 * p2;
            op *= 1 - clamp01((p2 - 0.78) / 0.2);
          } else {
            cx = x + (1660 - x) * p2;
            cy = 505 + 148 * p2;
            rot = 14 * p2;
            sc = 1 - 0.3 * p2;
            op *= 1 - clamp01((p2 - 0.78) / 0.2);
          }
          return <MemCard key={i} x={cx} y={cy} rot={rot} s={sc} op={op} label={card.label} tag={card.tag} color={card.color} />;
        })}
        {/* 分拣提示 */}
        <g opacity={clamp01((f - b2 - 10) / 20)}>
          <path d="M 1520 500 L 1600 462" stroke={P.eeg} strokeWidth={5} strokeDasharray="9 8" />
          <path d="M 1520 560 L 1602 636" stroke={P.grey} strokeWidth={5} strokeDasharray="9 8" />
        </g>
      </svg>
      <Chip text="重播 ≠ 原样存档" x={100} y={140} at={b0 + 12} accent={P.pink} />
      <Chip text="带情绪 · 反复用 · 能挂钩 → 先搬" x={100} y={830} at={b2 + 10} accent={P.amber} />
      <Stamp text="有标记的先上车" x={1380} y={120} at={b2 + 30} color={P.violetD} size={44} rot={4} />
    </FadeIn>
  );
};

/* ============ S3 情绪的无压重播室：杏仁核在岗 + 应激归零 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b0 = beat(0), b1 = beat(1);

  const gv = interpolate(f, [b0 + 40, b0 + 150], [0.7, 0.02], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const offK = 1 - clamp01((f - b0 - 165) / 18);
  const calmK = clamp01((f - b1) / 35);
  const spin = f * 1.5;

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={3} name={sec.name} accent={P.coral} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 重播室 */}
        <rect x={180} y={250} width={700} height={540} rx={32} fill={P.panel} stroke={P.panelHi} strokeWidth={6} />
        <text x={530} y={308} textAnchor="middle" fill={P.eeg} fontSize={31} fontFamily={FONT_HEAD} fontWeight={800}>
          REM · 情绪重播室</text>
        <rect x={260} y={340} width={540} height={330} rx={16} fill={P.screen} />
        <MoodCloud x={530} y={500} s={1.5} calm={calmK} />
        <text x={530} y={646} textAnchor="middle" fill={P.grey} fontSize={25} fontFamily={FONT_HEAD}
          opacity={1 - calmK}>白天的烦心事 · 重播中</text>
        <text x={530} y={646} textAnchor="middle" fill={P.eeg} fontSize={25} fontFamily={FONT_HEAD} fontWeight={700}
          opacity={calmK}>第二天再看 · 钝了一层</text>
        {/* 旋转回放箭头 */}
        <g transform={`translate(322,628) rotate(${spin})`} opacity={0.9}>
          <path d="M -16 0 A 16 16 0 1 1 0 -16" fill="none" stroke={P.amber} strokeWidth={5} strokeLinecap="round" />
          <path d="M 0 -16 L -8 -26 L 8 -24 Z" fill={P.amber} />
        </g>
        <text x={530} y={756} textAnchor="middle" fill={P.grey} fontSize={25} fontFamily={FONT_HEAD}>
          无压环境 · 安心重播</text>
        {/* 杏仁核 */}
        <Amygdala x={1050} y={392} s={1.15} hot={clamp01((f - b0 - 50) / 25)} />
        <text x={1050} y={506} textAnchor="middle" fill={P.coral} fontSize={27} fontFamily={FONT_HEAD} fontWeight={700}>
          杏仁核 · 情绪在岗</text>
        <path d="M 962 430 Q 920 450 884 468" fill="none" stroke={P.coral} strokeWidth={5}
          strokeDasharray="9 8" opacity={clamp01((f - b0 - 50) / 25)} />
        {/* 去甲肾上腺素表 */}
        <Gauge x={1460} y={390} s={1.1} v={gv} label="去甲肾上腺素" />
        <text x={1460} y={540} textAnchor="middle" fill={P.grey} fontSize={25} fontFamily={FONT_HEAD}
          opacity={gv < 0.12 ? 1 : 0.3}>≈ 归零 · 压力激素的近亲</text>
        {/* 应激开关 */}
        <text x={1460} y={612} textAnchor="middle" fill={P.cream} fontSize={28} fontFamily={FONT_HEAD} fontWeight={700}>
          应激信号</text>
        <Toggle x={1460} y={676} s={1.05} onK={offK} />
      </svg>
      <Chip text="REM · 重心换成情绪脱敏" x={100} y={140} at={b0 + 15} accent={P.coral} />
      <Stamp text="过夜疗法" x={1330} y={795} at={b1 + 22} color={P.coralD} size={48} rot={-4} />
      <Chip text="冲击被磨掉一层" x={100} y={830} at={b1 + 55} accent={P.amber} />
    </FadeIn>
  );
};

/* ============ S4 一边存一边删：突触缩圈 + 梦的功能三派 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b0 = beat(0), b1 = beat(1);

  const shrinkK = interpolate(f, [b0 + 160, b0 + 230], [1, 0.55], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const shrinkOn = shrinkK < 0.95;
  const views = [
    {n: '①', t: '“运行时噪音”派', s: '搬运时的主观伴随 · 没独立功能'},
    {n: '②', t: '“有功能”派', s: '威胁模拟 · 情绪练习场'},
    {n: '③', t: '“分层级”派', s: '底层在加工 · 上层是体验'},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={4} name={sec.name} accent={P.violet} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：突触缩圈 */}
        <text x={110} y={296} fill={P.cream} fontSize={30} fontFamily={FONT_HEAD} fontWeight={800}>
          白天 · 学习让突触长满</text>
        <text x={110} y={348} fill={P.amber} fontSize={30} fontFamily={FONT_HEAD} fontWeight={800}
          opacity={shrinkOn ? 1 : 0}>夜里 · 统一缩一圈</text>
        <text x={110} y={392} fill={C.muted} fontSize={25} fontFamily={FONT_HEAD}
          opacity={shrinkOn ? 1 : 0}>信号更突出 · 噪音压下去</text>
        <SynTree x={430} y={530} s={1} f={f} t0={b0 + 12} shrink={shrinkK} hot={3} />
        <g opacity={shrinkOn ? 1 : 0}>
          <path d="M 762 606 L 830 630" stroke={P.amber} strokeWidth={4} strokeDasharray="8 7" />
          <text x={842} y={640} fill={P.amber} fontSize={26} fontFamily={FONT_HEAD} fontWeight={700}>重要连接留下</text>
        </g>
        <text x={110} y={858} fill={P.grey} fontSize={25} fontFamily={FONT_HEAD}
          opacity={shrinkOn ? 1 : 0}>突触稳态假说 · 证据还在补</text>
        {/* 右：三派争论 */}
        <text x={1060} y={208} fill={P.cream} fontSize={33} fontFamily={FONT_HEAD} fontWeight={800}
          opacity={clamp01((f - b1) / 20)}>梦本身有没有功能？三派还在吵</text>
        {views.map((v, i) => {
          const g = Math.max(0.001, spring({frame: f - b1 - 12 - i * 20, fps, config: {damping: 12, stiffness: 140}}));
          const y = 260 + i * 132;
          return (
            <g key={i} transform={`translate(${(1 - g) * 70},0)`} opacity={g}>
              <rect x={1060} y={y} width={770} height={108} rx={18} fill={P.panel} />
              <rect x={1060} y={y} width={14} height={108} rx={7} fill={P.violet} />
              <circle cx={1130} cy={y + 54} r={30} fill="none" stroke={P.violet} strokeWidth={4} />
              <text x={1130} y={y + 65} textAnchor="middle" fontSize={30} fill={P.violet}
                fontFamily={FONT_HEAD} fontWeight={800}>{v.n}</text>
              <text x={1186} y={y + 48} fill={P.cream} fontSize={30} fontFamily={FONT_HEAD} fontWeight={800}>{v.t}</text>
              <text x={1186} y={y + 86} fill={C.muted} fontSize={25} fontFamily={FONT_HEAD}>{v.s}</text>
            </g>
          );
        })}
        {/* 漏出的灰 */}
        <g opacity={clamp01((f - b1 - 150) / 25)}>
          <HeadBrain x={1150} y={800} s={0.55} on={0.9} z={0} />
          <circle cx={1250} cy={736} r={10} fill={P.panel} stroke={P.panelHi} strokeWidth={4} />
          <circle cx={1286} cy={706} r={15} fill={P.panel} stroke={P.panelHi} strokeWidth={4} />
          <ellipse cx={1350} cy={668} rx={60} ry={38} fill={P.panel} stroke={P.panelHi} strokeWidth={4} />
          {[0, 1, 2].map((i) => {
            const t = (f / 26 + i * 0.7) % 2;
            return <circle key={i} cx={1330 + i * 22} cy={668 + t * 12} r={3.5}
              fill={P.grey} opacity={0.9 - t * 0.4} />;
          })}
          <text x={1460} y={668} fill={P.cream} fontSize={27} fontFamily={FONT_HEAD} fontWeight={700}>能记住的梦</text>
          <text x={1460} y={708} fill={C.muted} fontSize={25} fontFamily={FONT_HEAD}>夜班施工漏出的一点灰</text>
        </g>
      </svg>
      <Chip text="突触稳态假说 · 托诺尼" x={100} y={140} at={b0 + 25} accent={P.violet} />
      <Chip text="不只存 · 还在删" x={660} y={140} at={b0 + 140} accent={P.eeg} />
      <Stamp text="学界还在吵" x={1520} y={120} at={b1 + 80} color={P.coralD} size={46} rot={5} />
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
        <ellipse cx={960} cy={1020} rx={1100} ry={120} fill={P.bg} />
        <ellipse cx={960} cy={958} rx={240} ry={44} fill={P.pillow} stroke={P.grey} strokeWidth={5} />
        <g transform={`translate(960,798) scale(${pop * 1.45})`}>
          <HeadBrain x={0} y={0} s={1} on={1} z={1} />
        </g>
        <Moon x={640} y={640} s={0.85} hole={P.bgDeep} />
        <EEGWave x={1150} y={690} w={380} amp={0.7} sw={5} />
        {[0, 1, 2].map((i) => (
          <circle key={i} cx={1240 + i * 60} cy={830} r={10}
            fill={[P.eeg, P.amber, P.pink][i]} opacity={0.9} />
        ))}
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -300}}>
        <div style={{fontSize: 56, fontWeight: 800, color: C.text}}>睡着不是停工，是换了个班上</div>
        <div style={{fontSize: 38, color: C.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E229 = ({data = tl}: {data?: any}) => {
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
