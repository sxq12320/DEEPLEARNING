import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01, rand} from '../lib/style';
import {Backdrop, Ripple, Stamp, SceneTag, Chip, Subtitles, ProgressBar, FadeIn} from '../lib/atoms';
import tl from '../timeline_222.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：夜航靛蓝 × 月光黄 × 晨光橙 × 薄荷绿（纯色平涂） ============ */
const P = {
  bg: '#262c55',
  bgDeep: '#1a1e40',
  bgLite: '#323a6c',
  card: '#3d447f',
  cream: '#f4eedd',
  ink: '#171a38',
  moon: '#ffd873',
  moonD: '#e0ab40',
  sun: '#ff9e45',
  sunD: '#d1762a',
  sky: '#7ecbf2',
  skyD: '#4d93c4',
  mint: '#86e2b0',
  mintD: '#4dab7d',
  coral: '#ff7d6e',
  coralD: '#cf5244',
  violet: '#b79cf5',
  violetD: '#8467cf',
  blanket: '#5d6fbb',
  blanketD: '#46529b',
  pillow: '#eef1fb',
  mattress: '#d9dff2',
  skin: '#f2c79b',
  skinD: '#d99f6b',
  wood: '#8f6240',
  woodD: '#6a4729',
  grey: '#9aa3c4',
  brain: '#f2a9bd',
  brainD: '#d17a94',
};

/* 二次贝塞尔取点 / 朝向 */
const qpos = (p: number, x0: number, y0: number, x1: number, y1: number, x2: number, y2: number) => {
  const u = 1 - p;
  return [u * u * x0 + 2 * u * p * x1 + p * p * x2, u * u * y0 + 2 * u * p * y1 + p * p * y2];
};
const qdir = (p: number, x0: number, y0: number, x1: number, y1: number, x2: number, y2: number) =>
  Math.atan2(2 * (1 - p) * (y1 - y0) + 2 * p * (y2 - y1), 2 * (1 - p) * (x1 - x0) + 2 * p * (x2 - x1)) * 180 / Math.PI;

/* ---------- 小太阳 ---------- */
const SunIcon = ({x, y, r = 54, s = 1, op = 1, spin = 0.5}: {x: number; y: number; r?: number; s?: number; op?: number; spin?: number}) => {
  const f = useCurrentFrame();
  return (
    <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
      <g transform={`rotate(${f * spin})`}>
        {Array.from({length: 8}).map((_, i) => (
          <rect key={i} x={-7} y={-r - 34} width={14} height={26} rx={7} fill={P.sun} transform={`rotate(${i * 45})`} />
        ))}
      </g>
      <circle r={r} fill={P.sun} />
      <circle cy={r * 0.26} r={r * 0.7} fill={P.sunD} opacity={0.55} />
    </g>
  );
};

/* ---------- 月亮（月牙，hole=底色挖空） ---------- */
const MoonIcon = ({x, y, s = 1, op = 1, hole = P.bg}: {x: number; y: number; s?: number; op?: number; hole?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
    <circle r={56} fill={P.moon} />
    <circle cx={26} cy={-16} r={46} fill={hole} />
    <circle cx={-18} cy={22} r={8} fill={P.moonD} />
    <circle cx={-34} cy={-6} r={5} fill={P.moonD} />
  </g>
);

/* ---------- 闹钟 ---------- */
const AlarmClock = ({x, y, s = 1, op = 1, hour = 7, min = 0, shake = 0}: {
  x: number; y: number; s?: number; op?: number; hour?: number; min?: number; shake?: number;
}) => {
  const f = useCurrentFrame();
  const hr = ((hour % 12) + min / 60) * 30 - 90;
  const mr = min * 6 - 90;
  const j = shake > 0 ? Math.sin(f * 1.7) * 5 * shake : 0;
  return (
    <g transform={`translate(${x},${y}) rotate(${j}) scale(${s})`} opacity={op}>
      <line x1={-32} y1={42} x2={-48} y2={68} stroke={P.ink} strokeWidth={8} strokeLinecap="round" />
      <line x1={32} y1={42} x2={48} y2={68} stroke={P.ink} strokeWidth={8} strokeLinecap="round" />
      <circle cx={-36} cy={-50} r={17} fill={P.coral} stroke={P.coralD} strokeWidth={5} />
      <circle cx={36} cy={-50} r={17} fill={P.coral} stroke={P.coralD} strokeWidth={5} />
      <line x1={0} y1={-62} x2={0} y2={-76} stroke={P.ink} strokeWidth={6} strokeLinecap="round" />
      <circle r={52} fill={P.cream} stroke={P.ink} strokeWidth={7} />
      {Array.from({length: 12}).map((_, i) => {
        const a = i * 30 * Math.PI / 180;
        return <line key={i} x1={Math.sin(a) * 40} y1={-Math.cos(a) * 40} x2={Math.sin(a) * 46} y2={-Math.cos(a) * 46}
          stroke={P.ink} strokeWidth={4} />;
      })}
      <line x1={0} y1={0} x2={Math.cos(hr * Math.PI / 180) * 26} y2={Math.sin(hr * Math.PI / 180) * 26}
        stroke={P.ink} strokeWidth={8} strokeLinecap="round" />
      <line x1={0} y1={0} x2={Math.cos(mr * Math.PI / 180) * 38} y2={Math.sin(mr * Math.PI / 180) * 38}
        stroke={P.coralD} strokeWidth={5} strokeLinecap="round" />
      <circle r={6} fill={P.ink} />
      {shake > 0 && (
        <g stroke={P.coral} strokeWidth={5} strokeLinecap="round" opacity={shake}>
          <line x1={-72} y1={-56} x2={-88} y2={-66} />
          <line x1={72} y1={-56} x2={88} y2={-66} />
        </g>
      )}
    </g>
  );
};

/* ---------- 床 ---------- */
const Bed = ({x, y, s = 1, op = 1}: {x: number; y: number; s?: number; op?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
    <rect x={-196} y={-196} width={34} height={222} rx={12} fill={P.wood} stroke={P.woodD} strokeWidth={5} />
    <circle cx={-179} cy={-200} r={13} fill={P.woodD} />
    <rect x={-150} y={26} width={26} height={54} rx={8} fill={P.woodD} />
    <rect x={150} y={26} width={26} height={54} rx={8} fill={P.woodD} />
    <rect x={-168} y={-40} width={356} height={74} rx={18} fill={P.mattress} stroke={P.card} strokeWidth={5} />
    <rect x={-146} y={-68} width={104} height={42} rx={19} fill={P.pillow} stroke={P.card} strokeWidth={5} />
    <rect x={-30} y={-72} width={218} height={80} rx={18} fill={P.blanket} stroke={P.blanketD} strokeWidth={5} />
    <rect x={-30} y={-72} width={218} height={24} rx={12} fill={P.blanketD} opacity={0.55} />
  </g>
);

/* ---------- 睡觉小人头（awake=睁眼瞪天花板） ---------- */
const Sleeper = ({x, y, s = 1, op = 1, awake = false, z = 0}: {
  x: number; y: number; s?: number; op?: number; awake?: boolean; z?: number;
}) => {
  const f = useCurrentFrame();
  return (
    <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
      <circle r={46} fill={P.skin} stroke={P.skinD} strokeWidth={5} />
      <path d="M -46 -4 A 46 46 0 0 1 46 -4 Z" fill="#54407a" />
      {awake ? (
        <g>
          <circle cx={-17} cy={10} r={5.5} fill={P.ink} />
          <circle cx={19} cy={10} r={5.5} fill={P.ink} />
          <line x1={-8} y1={30} x2={10} y2={30} stroke={P.ink} strokeWidth={4} strokeLinecap="round" />
        </g>
      ) : (
        <g stroke={P.ink} strokeWidth={4} fill="none" strokeLinecap="round">
          <path d="M -26 8 Q -17 16 -8 8" />
          <path d="M 10 8 Q 19 16 28 8" />
          <path d="M -2 30 Q 6 35 14 30" />
        </g>
      )}
      {z > 0 && [0, 1, 2].map((i) => {
        const t = (f / 30 + i * 0.9) % 3;
        return (
          <text key={i} x={56 + i * 40 + t * 22} y={-36 - t * 52} fill={P.sky} fontSize={42 - i * 7}
            fontFamily={FONT_HEAD} fontWeight={800} opacity={z * (1 - t / 3)}>Z</text>
        );
      })}
    </g>
  );
};

/* ---------- 小飞机 ---------- */
const Plane = ({x, y, s = 1, rot = 0, op = 1}: {x: number; y: number; s?: number; rot?: number; op?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`} opacity={op}>
    <path d="M -62 -14 L -50 -42 L -36 -42 L -42 -15 Z" fill={P.coral} stroke={P.coralD} strokeWidth={4} strokeLinejoin="round" />
    <path d="M -4 -16 L 20 -54 L 38 -54 L 22 -14 Z" fill={P.coral} stroke={P.coralD} strokeWidth={4} strokeLinejoin="round" />
    <path d="M -64 4 Q -66 -16 -30 -19 L 32 -19 Q 66 -17 71 -1 Q 66 13 32 14 L -30 14 Q -64 14 -64 4 Z"
      fill={P.pillow} stroke={P.skyD} strokeWidth={4} />
    <path d="M -6 12 L 16 44 L 32 44 L 20 12 Z" fill={P.coral} stroke={P.coralD} strokeWidth={4} strokeLinejoin="round" />
    <circle cx={50} cy={-9} r={7} fill={P.sky} />
    <circle cx={28} cy={-9} r={4.5} fill={P.sky} />
    <circle cx={12} cy={-9} r={4.5} fill={P.sky} />
  </g>
);

/* ---------- 大脑 + 视交叉上核 ---------- */
const Brain = ({x, y, s = 1, op = 1, scn = 1}: {x: number; y: number; s?: number; op?: number; scn?: number}) => {
  const f = useCurrentFrame();
  const pulse = 0.55 + 0.45 * Math.sin(f / 7);
  return (
    <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
      <path d="M -96 20 C -122 -22 -104 -72 -66 -86 C -52 -118 -4 -126 26 -104 C 66 -116 104 -90 106 -48
        C 134 -28 130 16 104 34 C 110 70 74 94 42 80 C 16 100 -36 94 -52 66 C -86 66 -106 46 -96 20 Z"
        fill={P.brain} stroke={P.brainD} strokeWidth={6} strokeLinejoin="round" />
      <g stroke={P.brainD} strokeWidth={5} fill="none" strokeLinecap="round">
        <path d="M -78 -42 Q -50 -68 -22 -46" />
        <path d="M -64 18 Q -32 -4 -4 20" />
        <path d="M 8 -74 Q 44 -92 70 -64" />
        <path d="M 30 32 Q 60 12 88 36" />
        <path d="M -30 -8 Q 0 -28 30 -10" />
      </g>
      {scn > 0 && (
        <g opacity={scn}>
          <circle cx={8} cy={54} r={36} fill="none" stroke={P.violet} strokeWidth={4}
            strokeDasharray="10 8" opacity={pulse * scn} />
          <circle cx={8} cy={54} r={15} fill={P.violet} stroke={P.violetD} strokeWidth={4} />
        </g>
      )}
    </g>
  );
};

/* ---------- 褪黑素胶囊 ---------- */
const MelatoPill = ({x, y, s = 1, rot = 0, op = 1}: {x: number; y: number; s?: number; rot?: number; op?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`} opacity={op}>
    <rect x={-44} y={-19} width={88} height={38} rx={19} fill={P.violet} stroke={P.violetD} strokeWidth={5} />
    <path d="M 0 -19 L 0 19" stroke={P.violetD} strokeWidth={5} />
    <rect x={-44} y={-19} width={44} height={38} rx={19} fill={P.violetD} opacity={0.45} />
    <text x={20} y={8} textAnchor="middle" fontSize={20} fill="#fff" fontFamily={FONT_HEAD} fontWeight={800}>困</text>
  </g>
);

/* ---------- 皮质醇小太阳 ---------- */
const CortisolSun = ({x, y, s = 1, op = 1}: {x: number; y: number; s?: number; op?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
    {[0, 45, 90, 135, 180, 225, 270, 315].map((a) => {
      const r = a * Math.PI / 180;
      return <line key={a} x1={Math.cos(r) * 20} y1={Math.sin(r) * 20} x2={Math.cos(r) * 34} y2={Math.sin(r) * 34}
        stroke={P.sunD} strokeWidth={7} strokeLinecap="round" />;
    })}
    <circle r={19} fill={P.sun} />
    <text x={0} y={7} textAnchor="middle" fontSize={17} fill={P.ink} fontFamily={FONT_HEAD} fontWeight={800}>醒</text>
  </g>
);

/* ---------- 体温计 ---------- */
const Thermo = ({x, y, s = 1, op = 1, k = 0.4}: {x: number; y: number; s?: number; op?: number; k?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
    <rect x={-11} y={-76} width={22} height={116} rx={11} fill={P.pillow} stroke={P.card} strokeWidth={5} />
    <circle cy={48} r={24} fill={P.sky} stroke={P.skyD} strokeWidth={5} />
    <rect x={-5} y={30 - 90 * k} width={10} height={90 * k + 18} fill={P.sky} />
    {[-40, -10, 20].map((ty) => <line key={ty} x1={12} y1={ty} x2={24} y2={ty} stroke={P.card} strokeWidth={4} />)}
  </g>
);

/* ---------- 床头柜 ---------- */
const NightTable = ({x, y, s = 1, op = 1}: {x: number; y: number; s?: number; op?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
    <rect x={-70} y={-12} width={140} height={26} rx={10} fill={P.wood} stroke={P.woodD} strokeWidth={5} />
    <rect x={-58} y={14} width={116} height={76} rx={10} fill={P.woodD} />
    <circle cx={0} cy={52} r={7} fill={P.cream} />
  </g>
);

/* ---------- 窗（夜↔晨） ---------- */
const WindowView = ({x, y, s = 1, op = 1, day = 0}: {x: number; y: number; s?: number; op?: number; day?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
    <rect x={-112} y={-142} width={224} height={284} rx={16} fill="#10142e" stroke={P.card} strokeWidth={7} />
    <g opacity={1 - day}>
      <MoonIcon x={-34} y={-56} s={0.62} hole="#10142e" />
      {[0, 1, 2, 3, 4].map((i) => (
        <circle key={i} cx={-80 + rand(i, 31) * 160} cy={-110 + rand(i, 32) * 120} r={2.6 + rand(i, 33) * 2}
          fill="#fff" opacity={0.8} />
      ))}
    </g>
    <g opacity={day}>
      <SunIcon x={10} y={-30} r={44} s={0.9} spin={0.25} />
      <path d="M -112 104 Q -56 66 0 94 Q 56 122 112 90 L 112 142 L -112 142 Z" fill="#2b4a7a" opacity={0.9} />
    </g>
    <line x1={-112} y1={0} x2={112} y2={0} stroke={P.card} strokeWidth={6} />
    <line x1={0} y1={-142} x2={0} y2={142} stroke={P.card} strokeWidth={6} />
    <rect x={-140} y={-154} width={42} height={308} rx={14} fill={P.blanketD} />
    <rect x={98} y={-154} width={42} height={308} rx={14} fill={P.blanketD} />
  </g>
);

/* ---------- 城市点（航线端点） ---------- */
const CityDot = ({x, y, name, sub, accent}: {x: number; y: number; name: string; sub: string; accent: string}) => (
  <g>
    <circle cx={x} cy={y} r={13} fill={accent} stroke={P.cream} strokeWidth={4} />
    <circle cx={x} cy={y} r={22} fill="none" stroke={accent} strokeWidth={3} strokeDasharray="6 7" />
    <text x={x} y={y + 52} textAnchor="middle" fill={P.cream} fontSize={30} fontFamily={FONT_HEAD} fontWeight={800}>{name}</text>
    <text x={x} y={y + 86} textAnchor="middle" fill={C.muted} fontSize={25} fontFamily={FONT_HEAD}>{sub}</text>
  </g>
);

/* ---------- 腺苷罐子 ---------- */
const AdTank = ({x, y, s = 1, op = 1, k = 0.8}: {x: number; y: number; s?: number; op?: number; k?: number}) => {
  const f = useCurrentFrame();
  return (
    <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
      <rect x={-85} y={-118} width={170} height={236} rx={20} fill="rgba(238,241,251,0.08)" stroke={P.card} strokeWidth={6} />
      <path d={`M -77 ${110 - 212 * k} L 77 ${110 - 212 * k} L 77 102 Q 77 110 69 110 L -69 110 Q -77 110 -77 102 Z`}
        fill={P.violet} opacity={0.9} />
      <ellipse cy={110 - 212 * k} rx={77} ry={9} fill={P.violetD} opacity={0.9} />
      {[0, 1, 2].map((i) => {
        const t = ((f / 34 + i * 0.37) % 1);
        return <circle key={i} cx={-40 + i * 38} cy={100 - t * 190 * k} r={6}
          fill={P.cream} opacity={k > 0.05 ? (1 - t) * 0.7 : 0} />;
      })}
      <rect x={-30} y={-132} width={60} height={16} rx={8} fill={P.card} />
    </g>
  );
};

/* ---------- 午睡计时表（0–40 分钟） ---------- */
const NapGauge = ({x, y, s = 1, op = 1, k = 0}: {x: number; y: number; s?: number; op?: number; k?: number}) => {
  const pt = (deg: number, r: number) => [Math.cos(deg * Math.PI / 180) * r, -Math.sin(deg * Math.PI / 180) * r];
  const [nx, ny] = pt(180 - k * 180, 96);
  const arc = (d0: number, d1: number, r: number) =>
    `M ${pt(d0, r)[0]} ${pt(d0, r)[1]} A ${r} ${r} 0 0 1 ${pt(d1, r)[0]} ${pt(d1, r)[1]}`;
  return (
    <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
      <path d={arc(180, 90, 118)} fill="none" stroke={P.mint} strokeWidth={20} strokeLinecap="round" />
      <path d={arc(90, 45, 118)} fill="none" stroke={P.card} strokeWidth={20} strokeLinecap="round" />
      <path d={arc(45, 0, 118)} fill="none" stroke={P.coral} strokeWidth={20} strokeLinecap="round" />
      {[180, 90, 45, 0].map((d, i) => {
        const [tx, ty] = pt(d, 140);
        return <text key={d} x={tx} y={ty + 10} textAnchor="middle" fontSize={26} fill={P.cream}
          fontFamily={FONT_HEAD} fontWeight={700}>{['0', '20', '30', '40'][i] + "'"}</text>;
      })}
      <line x1={0} y1={0} x2={nx} y2={ny} stroke={P.cream} strokeWidth={8} strokeLinecap="round" />
      <circle r={12} fill={P.cream} stroke={P.card} strokeWidth={5} />
    </g>
  );
};

/* ---------- 连续七天稳定作息条 ---------- */
const WeekBars = ({x, y, s = 1, op = 1, t0 = 0}: {x: number; y: number; s?: number; op?: number; t0?: number}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const days = ['一', '二', '三', '四', '五', '六', '日'];
  return (
    <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
      {days.map((d, i) => {
        const g = Math.max(0.001, spring({frame: f - t0 - i * 7, fps, config: {damping: 11, stiffness: 150}}));
        const h = 74 * g;
        return (
          <g key={d}>
            <rect x={i * 86 - 26} y={-h} width={52} height={h} rx={10} fill={i < 5 ? P.mint : P.sun}
              stroke={i < 5 ? P.mintD : P.sunD} strokeWidth={4} />
            <line x1={i * 86} y1={-h - 18} x2={i * 86} y2={-h - 8} stroke={P.cream} strokeWidth={5}
              strokeLinecap="round" opacity={g} />
            <text x={i * 86} y={34} textAnchor="middle" fill={P.cream} fontSize={26} fontFamily={FONT_HEAD}>{d}</text>
          </g>
        );
      })}
      <line x1={-40} y1={0} x2={536} y2={0} stroke={P.card} strokeWidth={6} strokeLinecap="round" />
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
  const p1 = Math.max(0.001, spring({frame: f - 12, fps, config: {damping: 10, stiffness: 150}}));
  const bob = Math.sin(f / 15) * 7;
  return (
    <AbsoluteFill>
      <Backdrop top={P.bg} bottom={P.bg} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <circle cx={960} cy={352} r={218} fill={P.cream} />
        <circle cx={960} cy={352} r={196} fill={P.card} />
        <g transform={`translate(960,352) scale(${p0})`}>
          <g transform="translate(-960,-352)">
            <path d="M 830 330 Q 960 214 1090 330" fill="none" stroke={P.sky} strokeWidth={5} strokeDasharray="12 10" />
            <Plane x={960} y={268 + bob * 0.7} s={0.72} rot={-5} />
            <MoonIcon x={848} y={262} s={0.58} hole={P.card} />
            <SunIcon x={1082} y={266} r={40} s={0.62} spin={0.3} />
            <Bed x={950} y={486} s={0.74} />
            <Sleeper x={882} y={452} s={0.58} />
            <AlarmClock x={1078} y={452} s={0.62} hour={12} min={0} />
          </g>
        </g>
        <g transform={`translate(748,${452 - bob * 0.5}) scale(${p1})`}>
          <NightTable x={0} y={0} s={0.8} />
        </g>
        <g transform={`translate(1180,${500 + bob * 0.6}) scale(${p1})`}>
          <MelatoPill x={0} y={0} s={0.7} rot={-14} />
        </g>
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 300}}>
          <div style={{color: P.moon, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 26}}>
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

/* ============ S0 周五夜→周六午：卧室 + 时差航线 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * 30);
  const b0 = beat(0), b1 = beat(1), b2 = beat(2);

  const dayK = clamp01((f - b1 + 20) / 40);                       // 窗里月亮→太阳
  const flyK = clamp01((f - b1 - 10) / 25);                        // 航线展开
  const trip = Math.abs(((f - b1 - 30) / 130) % 2 - 1);            // 往返 0→1→0
  const flying = f > b1 + 10;
  const [px, py] = qpos(trip, 1620, 700, 1430, 420, 1240, 700);
  const prot = qdir(Math.min(0.999, trip + 0.004), 1620, 700, 1430, 420, 1240, 700);
  const billK = Math.max(0.001, spring({frame: f - b2 - 12, fps: 30, config: {damping: 10, stiffness: 140}}));

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={0} name={sec.name} accent={P.moon} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：卧室 */}
        <ellipse cx={400} cy={872} rx={330} ry={30} fill={P.bgDeep} />
        <WindowView x={680} y={300} s={0.85} day={dayK} />
        <Bed x={400} y={780} s={1.42} />
        <Sleeper x={268} y={714} s={1.02} z={dayK < 0.5 ? 1 : 0} />
        <NightTable x={150} y={800} s={1} />
        <AlarmClock x={150} y={712} s={0.9} hour={12} min={0} shake={f < b0 + 55 ? 1 : 0} />
        <text x={700} y={512} textAnchor="middle" fill={C.muted} fontSize={27} fontFamily={FONT_HEAD}
          opacity={dayK}>窗外已是周六中午</text>
        {/* 右：时差航线 */}
        <path d="M 1040 760 Q 1430 690 1830 760" fill="none" stroke={P.card} strokeWidth={6} strokeLinecap="round" />
        <CityDot x={1620} y={700} name="北京" sub="平时 7:00 起" accent={P.mint} />
        <CityDot x={1240} y={700} name="迪拜" sub="周末 11:00 起" accent={P.coral} />
        <path d="M 1620 700 Q 1430 420 1240 700" fill="none" stroke={P.sky} strokeWidth={5}
          strokeDasharray="14 12" opacity={flyK} />
        {flying && (
          <g>
            {[0.06, 0.12, 0.18].map((d, i) => {
              const pp = trip - d * Math.sign(Math.cos(((f - b1 - 30) / 130) * Math.PI));
              if (pp < 0 || pp > 1) return null;
              const [tx, ty] = qpos(Math.max(0, Math.min(1, pp)), 1620, 700, 1430, 420, 1240, 700);
              return <circle key={i} cx={tx} cy={ty} r={7 - i * 1.6} fill={P.sky} opacity={0.5 - i * 0.13} />;
            })}
            <Plane x={px} y={py} s={0.85} rot={prot} />
          </g>
        )}
        {/* 周一账单 */}
        <g transform={`translate(1430,300) rotate(-5) scale(${billK})`}>
          <rect x={-150} y={-92} width={300} height={184} rx={14} fill={P.cream} stroke={P.card} strokeWidth={5} />
          <rect x={-150} y={-92} width={300} height={44} rx={14} fill={P.coral} />
          <text x={0} y={-60} textAnchor="middle" fontSize={27} fill="#fff" fontFamily={FONT_HEAD} fontWeight={800}>周一账单 · 到付</text>
          <text x={-118} y={-8} fontSize={27} fill={P.ink} fontFamily={FONT_HEAD}>昏沉 ×1</text>
          <text x={-118} y={36} fontSize={27} fill={P.ink} fontFamily={FONT_HEAD}>入睡困难 ×1</text>
          <text x={-118} y={76} fontSize={25} fill={P.coralD} fontFamily={FONT_HEAD} fontWeight={800}>利息：痛苦加倍</text>
        </g>
        {f > b2 + 12 && <Ripple x={1430} y={300} startFrame={b2 + 12} color={P.cream} />}
      </svg>
      <Chip text="周五 2:00 睡 → 周六 12:00 起" x={100} y={150} at={b0 + 18} accent={P.moon} />
      <Chip text="生物钟被拽走两个时区" x={100} y={240} at={b1 + 30} accent={P.sky} />
      <Stamp text="社会时差" x={1560} y={140} at={b1 + 42} color={P.coralD} size={52} rot={5} />
      <Chip text="赖床的快感是借来的" x={100} y={820} at={b2 + 40} accent={P.violet} />
    </FadeIn>
  );
};

/* ============ S1 视交叉上核：只对晨光对表的总闸 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b0 = beat(0), b1 = beat(1), b2 = beat(2);

  const cards = [
    {y: 250, accent: P.violet, title: '褪黑素', sub: '入夜开工 → 催你犯困'},
    {y: 432, accent: P.sun, title: '皮质醇', sub: '清晨爬坡 → 把你叫醒'},
    {y: 614, accent: P.sky, title: '体温', sub: '凌晨谷底 · 白天爬升'},
  ];
  const beamK = clamp01((f - b1 + 10) / 30);
  const driftDeg = 90 + Math.max(0, f - b2 - 20) * 0.35;

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={1} name={sec.name} accent={P.violet} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <ellipse cx={430} cy={838} rx={300} ry={30} fill={P.bgDeep} />
        {/* 晨光 → 眼睛 */}
        <SunIcon x={150} y={150} r={46} s={0.9} />
        <g opacity={beamK}>
          <line x1={210} y1={200} x2={330} y2={420} stroke={P.sun} strokeWidth={7} strokeDasharray="16 12" />
          <text x={130} y={330} fill={P.sun} fontSize={27} fontFamily={FONT_HEAD} fontWeight={700}>晨光对表</text>
          <circle cx={338} cy={428} r={12} fill={P.pillow} stroke={P.card} strokeWidth={4} />
          <circle cx={342} cy={428} r={5} fill={P.ink} />
        </g>
        {/* 大脑 */}
        <Brain x={440} y={470} s={2.15} scn={clamp01((f - b0 - 15) / 20)} />
        <g opacity={clamp01((f - b0 - 40) / 18)}>
          <path d="M 452 590 L 340 780" stroke={P.cream} strokeWidth={4} strokeDasharray="9 8" />
          <text x={120} y={826} fill={P.cream} fontSize={30} fontFamily={FONT_HEAD} fontWeight={700}>视交叉上核</text>
          <text x={120} y={866} fill={C.muted} fontSize={26} fontFamily={FONT_HEAD}>米粒大 · 约两万个细胞 · 总闸</text>
        </g>
        {/* 激素排班卡 */}
        {cards.map((c, i) => {
          const g = Math.max(0.001, spring({frame: f - b1 - 10 - i * 18, fps, config: {damping: 12, stiffness: 140}}));
          return (
            <g key={i} opacity={g}>
              <path d={`M 700 470 L ${990 - (1 - g) * 60} ${c.y + 62}`} stroke={c.accent} strokeWidth={4}
                strokeDasharray="8 9" opacity={0.7} />
              <rect x={990} y={c.y} width={780} height={132} rx={20} fill={P.card}
                transform={`translate(${-(1 - g) * 60},0)`} />
              <rect x={990} y={c.y} width={16} height={132} rx={8} fill={c.accent}
                transform={`translate(${-(1 - g) * 60},0)`} />
              <g transform={`translate(1090,${c.y + 66}) scale(${g})`}>
                {i === 0 && <MelatoPill x={0} y={0} s={0.85} />}
                {i === 1 && <CortisolSun x={0} y={0} s={1} />}
                {i === 2 && <Thermo x={0} y={0} s={0.8} k={0.32} />}
              </g>
              <text x={1180} y={c.y + 58} fill={P.cream} fontSize={36} fontFamily={FONT_HEAD} fontWeight={800}
                transform={`translate(${-(1 - g) * 60},0)`}>{c.title}</text>
              <text x={1180} y={c.y + 102} fill={C.muted} fontSize={29} fontFamily={FONT_HEAD}
                transform={`translate(${-(1 - g) * 60},0)`}>{c.sub}</text>
            </g>
          );
        })}
        {/* 出厂周期小表盘 */}
        <g opacity={clamp01((f - b2 - 10) / 20)}>
          <circle cx={1500} cy={806} r={72} fill={P.cream} stroke={P.ink} strokeWidth={6} />
          {Array.from({length: 12}).map((_, i) => {
            const a = i * 30 * Math.PI / 180;
            return <line key={i} x1={1500 + Math.sin(a) * 56} y1={806 - Math.cos(a) * 56}
              x2={1500 + Math.sin(a) * 62} y2={806 - Math.cos(a) * 62} stroke={P.ink} strokeWidth={3} />;
          })}
          <line x1={1500} y1={806} x2={1500 + Math.cos((driftDeg - 90) * Math.PI / 180) * 48}
            y2={806 + Math.sin((driftDeg - 90) * Math.PI / 180) * 48} stroke={P.coralD} strokeWidth={6} strokeLinecap="round" />
          <circle cx={1500} cy={806} r={7} fill={P.ink} />
          <text x={1500} y={900} textAnchor="middle" fill={P.cream} fontSize={27} fontFamily={FONT_HEAD} fontWeight={700}>出厂 ≈24.2 小时</text>
          <text x={1660} y={840} fill={C.muted} fontSize={25} fontFamily={FONT_HEAD}>不对表就慢慢漂</text>
        </g>
      </svg>
      <Chip text="只认光和固定作息" x={640} y={130} at={b0 + 30} accent={P.sun} />
      <Stamp text="不管今天周几" x={1480} y={150} at={b2 + 45} color={P.violetD} size={46} rot={-4} />
      <Chip text="晚睡晚起 = 它眼里的改时区" x={830} y={820} at={b2 + 90} accent={P.violet} />
    </FadeIn>
  );
};

/* ============ S2 睡眠中点实测 + 北京⇄迪拜每周往返 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b0 = beat(0), b1 = beat(1), b2 = beat(2);

  const t2x = (t: number) => 200 + (t - 20) * 43.75;
  const g1 = clamp01((f - b0 - 15) / 25);
  const g2 = clamp01((f - b0 - 40) / 25);
  const midK = clamp01((f - 175) / 22);
  const trip = Math.abs(((f - b1 - 10) / 120) % 2 - 1);
  const flying = f > b1;
  const [px, py] = qpos(trip, 1600, 430, 1430, 295, 1260, 430);
  const prot = qdir(Math.min(0.999, trip + 0.004), 1600, 430, 1430, 295, 1260, 430);
  const risk = [70, 118, 168];

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={2} name={sec.name} accent={P.sky} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：24h 睡眠条 */}
        <text x={200} y={272} fill={P.cream} fontSize={31} fontFamily={FONT_HEAD} fontWeight={700}>睡眠中点 · 一条轴量时差</text>
        <line x1={200} y1={640} x2={900} y2={640} stroke={P.cream} strokeWidth={5} strokeLinecap="round" />
        {[20, 24, 28, 32, 36].map((t, i) => (
          <g key={t} opacity={0.85}>
            <line x1={t2x(t)} y1={640} x2={t2x(t)} y2={654} stroke={P.cream} strokeWidth={4} />
            <text x={t2x(t)} y={688} textAnchor="middle" fill={C.muted} fontSize={25} fontFamily={FONT_HEAD}>
              {['20:00', '0:00', '4:00', '8:00', '12:00'][i]}
            </text>
          </g>
        ))}
        <MoonIcon x={140} y={620} s={0.5} />
        <SunIcon x={955} y={620} r={30} s={0.6} spin={0.3} />
        {/* 工作日条 23:00–7:00 */}
        <rect x={t2x(23)} y={400} width={350 * g1} height={64} rx={18} fill={P.mint} stroke={P.mintD} strokeWidth={5} />
        <text x={t2x(23) - 14} y={437} textAnchor="end" fill={P.mint} fontSize={28} fontFamily={FONT_HEAD} fontWeight={700}>工作日</text>
        <text x={t2x(23) + 12} y={382} fill={C.muted} fontSize={25} fontFamily={FONT_HEAD}>23:00 – 7:00</text>
        {/* 休息日条 1:00–10:00 */}
        <rect x={t2x(25)} y={530} width={350 * g2} height={64} rx={18} fill={P.violet} stroke={P.violetD} strokeWidth={5} />
        <text x={t2x(25) - 14} y={567} textAnchor="end" fill={P.violet} fontSize={28} fontFamily={FONT_HEAD} fontWeight={700}>休息日</text>
        <text x={t2x(25) + 12} y={512} fill={C.muted} fontSize={25} fontFamily={FONT_HEAD}>1:00 – 10:00</text>
        {/* 中点差 */}
        <g opacity={midK}>
          <rect x={498} y={424} width={16} height={16} fill={P.cream} transform="rotate(45 506 432)" />
          <rect x={607} y={554} width={16} height={16} fill={P.cream} transform="rotate(45 615 562)" />
          <line x1={506} y1={432} x2={615} y2={562} stroke={P.cream} strokeWidth={4} strokeDasharray="8 7" />
          <line x1={615} y1={562} x2={615} y2={640} stroke={P.coral} strokeWidth={4} strokeDasharray="8 7" />
          <line x1={506} y1={432} x2={506} y2={470} stroke={P.coral} strokeWidth={4} strokeDasharray="8 7" />
          <text x={795} y={572} fill={P.coral} fontSize={30} fontFamily={FONT_HEAD} fontWeight={800}>中点差 ≈2.5h</text>
        </g>
        <text x={200} y={780} fill={C.muted} fontSize={27} fontFamily={FONT_HEAD}>夜猫子型能差到 3 小时以上</text>
        {/* 右：每周往返航线 */}
        <rect x={1040} y={200} width={820} height={330} rx={24} fill={P.card} />
        <text x={1080} y={258} fill={P.cream} fontSize={30} fontFamily={FONT_HEAD} fontWeight={700}>上万人作息实测 · 罗恩内伯格团队</text>
        <CityDot x={1600} y={430} name="北京" sub="工作日时区" accent={P.mint} />
        <CityDot x={1260} y={430} name="迪拜" sub="周末时区" accent={P.coral} />
        <path d="M 1600 430 Q 1430 295 1260 430" fill="none" stroke={P.sky} strokeWidth={5}
          strokeDasharray="12 10" opacity={clamp01((f - b1) / 20)} />
        {flying && <Plane x={px} y={py} s={0.72} rot={prot} />}
        <text x={1430} y={578} textAnchor="middle" fill={P.sky} fontSize={26} fontFamily={FONT_HEAD}
          opacity={clamp01((f - b1) / 20)}>每周末往返一趟</text>
        {/* 右下：差值→风险 */}
        <g opacity={clamp01((f - b2) / 25)}>
          {risk.map((h, i) => {
            const gg = clamp01((f - b2 - i * 12) / 20);
            return <rect key={i} x={1180 + i * 170} y={820 - h * gg} width={96} height={h * gg} rx={12}
              fill={P.coral} opacity={0.5 + i * 0.22} />;
          })}
          <path d="M 1150 780 L 1700 640" stroke={P.cream} strokeWidth={5} strokeDasharray="12 9" />
          <path d="M 1700 640 L 1668 646 L 1682 668 Z" fill={P.cream} />
          <text x={1430} y={878} textAnchor="middle" fill={P.cream} fontSize={28} fontFamily={FONT_HEAD}>
            差值越大 → 肥胖·抑郁占比越高</text>
          <text x={1700} y={600} textAnchor="end" fill={P.grey} fontSize={24} fontFamily={FONT_HEAD}>相关 ≠ 因果</text>
        </g>
      </svg>
      <Chip text="睡眠中点差 = 时差量" x={100} y={150} at={b0 + 25} accent={P.sky} />
      <Stamp text="一年飞一百多趟" x={1330} y={140} at={b1 + 65} color={P.skyD} size={44} rot={4} />
      <Chip text="作息不规律 → 可测量的数" x={100} y={820} at={b2 + 30} accent={P.coral} />
    </FadeIn>
  );
};

/* ============ S3 补得回腺苷，补不回节律 + 宾大实验双曲线 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * 30);
  const b0 = beat(0), b1 = beat(1), b2 = beat(2);

  const tankK = interpolate(f, [b0 + 10, b0 + 150], [0.82, 0.3], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const rIn = clamp01((f - b1) / 22);
  const draw = clamp01((f - b2 - 10) / 170);
  const steps = 14;
  const objPts = Array.from({length: steps + 1}).map((_, i) => {
    const x = 400 + (900 * i) / steps;
    const y = 706 + (126 * Math.pow(i / steps, 0.85));
    return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
  }).join(' ');
  const subjPts = `M 400 758 L 1300 786`;

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={3} name={sec.name} accent={P.mint} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <line x1={965} y1={160} x2={965} y2={620} stroke={P.cream} strokeWidth={3} strokeDasharray="4 18" opacity={0.3} />
        {/* 左：腺苷罐 还清一点 */}
        <ellipse cx={330} cy={700} rx={150} ry={20} fill={P.bgDeep} />
        <AdTank x={330} y={440} s={1.05} k={tankK} />
        <text x={330} y={646} textAnchor="middle" fill={P.violet} fontSize={27} fontFamily={FONT_HEAD} fontWeight={700}>腺苷 · 促睡物质</text>
        <Sleeper x={620} y={420} s={1.25} z={1} />
        <SunIcon x={620} y={240} r={34} s={0.6} op={clamp01((f - b0 - 90) / 25)} spin={0.3} />
        <text x={620} y={560} textAnchor="middle" fill={P.mint} fontSize={27} fontFamily={FONT_HEAD}
          opacity={clamp01((f - b0 - 90) / 25)}>第二天确实松快些</text>
        {/* 右：周日晚上干瞪眼 */}
        <g opacity={rIn}>
          <MoonIcon x={1130} y={280} s={0.85} hole={P.bgLite} />
          <Bed x={1420} y={520} s={1.2} />
          <Sleeper x={1308} y={464} s={1.05} awake />
          <AlarmClock x={1660} y={430} s={0.78} hour={2} min={42} />
          <text x={1745} y={540} textAnchor="middle" fill={P.coral} fontSize={26} fontFamily={FONT_HEAD} fontWeight={700}>周日 02:40</text>
          <MelatoPill x={1130} y={620} s={0.9} rot={-8} />
          <path d="M 1210 620 L 1360 620" stroke={P.violet} strokeWidth={5} strokeDasharray="10 9" />
          <path d="M 1360 620 L 1340 610 L 1340 630 Z" fill={P.violet} />
          <text x={1420} y={668} fill={C.muted} fontSize={26} fontFamily={FONT_HEAD}>褪黑素按新表还没开工</text>
        </g>
        {/* 底：宾大实验双曲线 */}
        <g opacity={clamp01((f - b2) / 20)}>
          <text x={860} y={672} textAnchor="middle" fill={P.cream} fontSize={28} fontFamily={FONT_HEAD} fontWeight={700}>
            宾大实验 · 每晚只睡 6 小时 × 14 天</text>
          <line x1={380} y1={852} x2={1320} y2={852} stroke={P.cream} strokeWidth={4} />
          <line x1={380} y1={690} x2={380} y2={852} stroke={P.cream} strokeWidth={4} />
          <text x={356} y={706} textAnchor="end" fill={C.muted} fontSize={24} fontFamily={FONT_HEAD}>注意力</text>
          <text x={400} y={882} textAnchor="middle" fill={C.muted} fontSize={24} fontFamily={FONT_HEAD}>第1天</text>
          <text x={1300} y={882} textAnchor="middle" fill={C.muted} fontSize={24} fontFamily={FONT_HEAD}>第14天</text>
          <line x1={380} y1={830} x2={1320} y2={830} stroke={P.grey} strokeWidth={4} strokeDasharray="10 9" opacity={0.7} />
          <text x={368} y={824} textAnchor="end" fill={P.grey} fontSize={24} fontFamily={FONT_HEAD}>≈ 通宵水平</text>
          <path d={subjPts} fill="none" stroke={P.mint} strokeWidth={7} strokeLinecap="round"
            strokeDasharray="1600" strokeDashoffset={1600 * (1 - draw)} />
          <path d={objPts} fill="none" stroke={P.coral} strokeWidth={7} strokeLinecap="round"
            strokeDasharray="1600" strokeDashoffset={1600 * (1 - draw)} />
          <g opacity={clamp01((f - b2 - 60) / 20)}>
            <rect x={1400} y={740} width={26} height={10} rx={5} fill={P.coral} />
            <text x={1440} y={752} fill={P.cream} fontSize={26} fontFamily={FONT_HEAD}>客观表现 · 一路掉</text>
            <rect x={1400} y={792} width={26} height={10} rx={5} fill={P.mint} />
            <text x={1440} y={804} fill={P.cream} fontSize={26} fontFamily={FONT_HEAD}>主观感觉 · “我还行”</text>
          </g>
        </g>
      </svg>
      <Chip text="睡眠债的大头能还一部分" x={100} y={150} at={b0 + 20} accent={P.mint} />
      <Chip text="节律和结构补不回来" x={960} y={150} at={b1 + 25} accent={P.coral} />
      <Stamp text="自我感觉最不可信" x={1380} y={135} at={b2 + 130} color={P.coralD} size={44} rot={-4} />
    </FadeIn>
  );
};

/* ============ S4 正确补法三件套 + 连稳一两周 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b0 = beat(0), b1 = beat(1), b2 = beat(2);

  const tips = [
    {at: b0 + 15, icon: 0, title: '周末起床 ≤ 平时 +1 小时', sub: '醒了就去晒晨光 · 照常对表'},
    {at: b1 + 15, icon: 1, title: '提前睡 > 赖床补', sub: '吃足前半夜的深睡份额'},
    {at: b1 + 75, icon: 2, title: '午睡 20 分钟掉头', sub: '超 30 分钟撞深睡 · 反而更昏'},
  ];
  const sweep = clamp01((f - b1 - 90) / 60) * 0.5;

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={4} name={sec.name} accent={P.mint} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：三张做法卡 */}
        {tips.map((t, i) => {
          const g = Math.max(0.001, spring({frame: f - t.at, fps, config: {damping: 12, stiffness: 140}}));
          const y = 200 + i * 205;
          return (
            <g key={i} transform={`translate(${-(1 - g) * 80},0)`} opacity={g}>
              <rect x={140} y={y} width={880} height={168} rx={22} fill={P.card} />
              <rect x={140} y={y} width={16} height={168} rx={8} fill={P.mint} />
              <circle cx={260} cy={y + 84} r={62} fill={P.bgLite} />
              {t.icon === 0 && (
                <g transform={`translate(260,${y + 84})`}>
                  <AlarmClock x={-16} y={0} s={0.62} hour={7} min={0} />
                  <AlarmClock x={34} y={8} s={0.5} hour={8} min={0} />
                </g>
              )}
              {t.icon === 1 && (
                <g transform={`translate(260,${y + 84})`}>
                  <MoonIcon x={14} y={0} s={0.62} hole={P.bgLite} />
                  <path d="M -52 0 L -18 0" stroke={P.mint} strokeWidth={6} strokeLinecap="round" />
                  <path d="M -52 0 L -34 -12 L -34 12 Z" fill={P.mint} />
                </g>
              )}
              {t.icon === 2 && (
                <g transform={`translate(260,${y + 84})`}>
                  <path d="M -40 18 A 46 46 0 0 1 40 18" fill="none" stroke={P.mint} strokeWidth={9} strokeLinecap="round" />
                  <line x1={0} y1={18} x2={-22} y2={-16} stroke={P.cream} strokeWidth={6} strokeLinecap="round" />
                  <circle cy={18} r={7} fill={P.cream} />
                </g>
              )}
              <text x={352} y={y + 72} fill={P.cream} fontSize={37} fontFamily={FONT_HEAD} fontWeight={800}>{t.title}</text>
              <text x={352} y={y + 122} fill={C.muted} fontSize={29} fontFamily={FONT_HEAD}>{t.sub}</text>
              <text x={960} y={y + 100} textAnchor="middle" fill={P.mint} fontSize={50} fontFamily={FONT_HEAD}
                fontWeight={800} opacity={g}>{i + 1}</text>
            </g>
          );
        })}
        {/* 右上：午睡计时表 */}
        <text x={1490} y={196} textAnchor="middle" fill={P.cream} fontSize={30} fontFamily={FONT_HEAD} fontWeight={700}>午睡计时</text>
        <NapGauge x={1490} y={470} s={1.15} k={sweep} />
        <text x={1490} y={575} textAnchor="middle" fill={P.mint} fontSize={27} fontFamily={FONT_HEAD} fontWeight={700}
          opacity={clamp01((f - b1 - 150) / 20)}>20 分钟 ✓ 在深睡门口掉头</text>
        {/* 右下：连稳七天 */}
        <text x={1490} y={640} textAnchor="middle" fill={P.cream} fontSize={30} fontFamily={FONT_HEAD} fontWeight={700}
          opacity={clamp01((f - b2) / 20)}>欠得狠就连稳一两周 · 分期还</text>
        <WeekBars x={1240} y={812} s={1} t0={b2 + 15} op={clamp01((f - b2) / 20)} />
      </svg>
      <Chip text="睡前纠结补不补 · 也在抬皮质醇" x={140} y={820} at={b2 + 60} accent={P.violet} />
      <Stamp text="规律比时长更硬" x={1430} y={140} at={b2 + 90} color={P.mintD} size={48} rot={4} />
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
        <ellipse cx={960} cy={1015} rx={1100} ry={120} fill={P.bg} />
        <Bed x={880} y={930} s={1.35} />
        <Sleeper x={754} y={870} s={0.95} z={1} />
        <MoonIcon x={620} y={640} s={0.8} hole={P.bgDeep} />
        <SunIcon x={1300} y={700} r={38} s={0.7} op={0.9} spin={0.25} />
        <g transform={`translate(1530,960) rotate(6) scale(${pop})`}>
          <Plane x={0} y={0} s={0.78} rot={-4} />
          <rect x={-70} y={-96} width={150} height={42} rx={12} fill={P.card} />
          <text x={5} y={-66} textAnchor="middle" fontSize={24} fill={P.cream} fontFamily={FONT_HEAD}>本周末停飞</text>
        </g>
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -280}}>
        <div style={{fontSize: 60, fontWeight: 800, color: C.text}}>让生物钟以为今天没什么两样</div>
        <div style={{fontSize: 38, color: C.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E222 = ({data = tl}: {data?: any}) => {
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
