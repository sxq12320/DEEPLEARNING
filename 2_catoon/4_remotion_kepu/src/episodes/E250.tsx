import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01, rand} from '../lib/style';
import {Backdrop, Stamp, SceneTag, Chip, Subtitles, ProgressBar, FadeIn, Moon, Stars} from '../lib/atoms';
import tl from '../timeline_250.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：深夜卧室 · 靛紫夜 × 月光黄 × 肌肉红 ============ */
const P = {
  bg: '#251f49',
  bgDeep: '#1a1636',
  panel: '#302a5c',
  card: '#3f3874',
  ink: '#151029',
  cream: '#f6eeda',
  muted: '#ada8d8',
  moon: '#ffd98c',
  moonD: '#e5b364',
  muscle: '#ff7a5c',
  muscleD: '#d8513c',
  nerve: '#ffe066',
  ion: '#63e2c6',
  ionD: '#37b59c',
  sky: '#6fc6f2',
  skyD: '#3f93c4',
  coral: '#ff6f61',
  violet: '#b48be0',
  sheet: '#5d54a0',
  sheetD: '#484084',
  pillow: '#f0e7d2',
  bed: '#453e7e',
};

const wrap = (v: number, m: number) => ((v % m) + m) % m;

/* ---------- 面板标题 ---------- */
const PanelTitle = ({x, y, text, col = P.muted}: {x: number; y: number; text: string; col?: string}) => (
  <text x={x} y={y} fill={col} fontSize={25} fontFamily={FONT_HEAD}>{text}</text>
);

/* ---------- 对勾 / 叉 ---------- */
const Check = ({x, y, s = 1, col = P.ion}: {x: number; y: number; s?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} stroke={col} strokeWidth={8}
    strokeLinecap="round" strokeLinejoin="round" fill="none">
    <path d="M -15 1 L -5 12 L 17 -12" />
  </g>
);
const MarkX = ({x, y, s = 1, col = P.coral}: {x: number; y: number; s?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} stroke={col} strokeWidth={7} strokeLinecap="round">
    <line x1={-12} y1={-12} x2={12} y2={12} /><line x1={12} y1={-12} x2={-12} y2={12} />
  </g>
);

/* ---------- 圆角标签牌 ---------- */
const TagPill = ({x, y, t, col, s = 1, w = 210, fs = 29}: {
  x: number; y: number; t: string; col: string; s?: number; w?: number; fs?: number;
}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-w / 2} y={-30} width={w} height={60} rx={16} fill={P.panel} stroke={col} strokeWidth={4} />
    <text x={0} y={11} textAnchor="middle" fontSize={fs} fill={col}
      fontFamily={FONT_HEAD} fontWeight={800}>{t}</text>
  </g>
);

/* ---------- 痉挛闪电 ---------- */
const Zap = ({x, y, s = 1, col = P.nerve, rot = 0}: {
  x: number; y: number; s?: number; col?: string; rot?: number;
}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`}>
    <path d="M -7 -34 L 15 -34 L 3 -8 L 19 -8 L -13 34 L -4 2 L -19 2 Z"
      fill={col} stroke={P.ink} strokeWidth={3} strokeLinejoin="round" />
  </g>
);

/* ---------- 水滴 / 汗 ---------- */
const Drop = ({x, y, s = 1, col = P.sky}: {x: number; y: number; s?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <path d="M 0 -16 Q 13 2 13 10 A 13 13 0 0 1 -13 10 Q -13 2 0 -16 Z" fill={col} />
  </g>
);

/* ---------- 锁 / 开锁 ---------- */
const LockIcon = ({x, y, s = 1, col = P.nerve}: {x: number; y: number; s?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-22} y={-6} width={44} height={38} rx={10} fill={col} />
    <path d="M -13 -6 v-12 a13 13 0 0 1 26 0 v12" fill="none" stroke={col} strokeWidth={7} />
    <circle cy={12} r={5} fill={P.ink} />
  </g>
);
const LockOpen = ({x, y, s = 1, col = P.ion}: {x: number; y: number; s?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-22} y={-6} width={44} height={38} rx={10} fill="none" stroke={col} strokeWidth={6} />
    <path d="M -13 -6 v-12 a13 13 0 0 1 26 0" fill="none" stroke={col} strokeWidth={7} />
    <circle cx={0} cy={12} r={5} fill={col} />
  </g>
);

/* ---------- 肌肉条（肌腱 + 肌腹 + 纤维线） ---------- */
const MuscleStrip = ({x, y, len = 420, th = 60, col = P.muscle, shake = 0, fiber = P.muscleD}: {
  x: number; y: number; len?: number; th?: number; col?: string; shake?: number; fiber?: string;
}) => {
  const f = useCurrentFrame();
  const sh = shake ? Math.sin(f * 2.6) * shake : 0;
  return (
    <g transform={`translate(${x},${y + sh})`}>
      <rect x={-len / 2 - 52} y={-9} width={52} height={18} rx={9} fill={P.pillow} />
      <rect x={len / 2} y={-9} width={52} height={18} rx={9} fill={P.pillow} />
      <rect x={-len / 2} y={-th / 2} width={len} height={th} rx={th / 2} fill={col} />
      {Array.from({length: 4}).map((_, i) => (
        <line key={i} x1={-len / 2 + 18} y1={-th / 2 + ((i + 1) * th) / 5}
          x2={len / 2 - 18} y2={-th / 2 + ((i + 1) * th) / 5}
          stroke={fiber} strokeWidth={3.2} opacity={0.5} />
      ))}
    </g>
  );
};

/* ---------- 小腿侧视（膝左踝右，ank>0 勾起脚尖朝上，ank<0 绷直下垂） ---------- */
const LowerLeg = ({x, y, s = 1, ank = -14, cramp = 0, flip = 1, skin = C.skin, musc = P.muscle}: {
  x: number; y: number; s?: number; ank?: number; cramp?: number; flip?: number;
  skin?: string; musc?: string;
}) => {
  const f = useCurrentFrame();
  const sh = cramp > 0 ? Math.sin(f * 2.4) * 1.7 * cramp : 0;
  const t = clamp01((ank + 30) / 95);          // 0 = 绷直缩短位, 1 = 回勾拉长位
  const ry = 24 - 9 * t;
  const rx = 50 + 16 * t;
  return (
    <g transform={`translate(${x},${y}) scale(${flip * s},${s})`}>
      <g transform={`translate(0,${sh})`}>
        {/* 跟腱 */}
        <line x1={-58} y1={-46} x2={-8} y2={4} stroke={C.skinDark} strokeWidth={9}
          strokeLinecap="round" opacity={0.55} />
        {/* 小腿外形 */}
        <path d="M -176 -18 C -178 -52 -150 -70 -104 -66 L -36 -54 C -10 -48 2 -32 4 -16 L 6 16
          C -20 32 -110 40 -148 32 C -170 27 -180 6 -176 -18 Z" fill={skin} />
        <circle cx={-168} cy={-2} r={27} fill={skin} />
        {/* 腓肠肌肌腹 */}
        <ellipse cx={-88} cy={-38} rx={rx} ry={ry} fill={musc} opacity={0.92} />
        {/* 脚：绕踝（0,6) 旋转 */}
        <g transform={`rotate(${-ank} 0 6)`}>
          <rect x={-10} y={-10} width={104} height={30} rx={15} fill={skin} />
          <circle cx={92} cy={5} r={16} fill={skin} />
          <line x1={66} y1={-3} x2={88} y2={-4} stroke={C.skinDark} strokeWidth={3.4}
            strokeLinecap="round" opacity={0.6} />
        </g>
        <circle cx={0} cy={6} r={7} fill={C.skinDark} opacity={0.55} />
      </g>
    </g>
  );
};

/* ---------- 脚小图标（小腿竖直 + 脚绕踝转，ank>0 勾脚，负值绷直） ---------- */
const FootGlyph = ({x, y, s = 1, ank = 0, col = P.cream}: {
  x: number; y: number; s?: number; ank?: number; col?: string;
}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <line x1={-14} y1={-36} x2={-14} y2={-2} stroke={col} strokeWidth={12} strokeLinecap="round" />
    <g transform={`rotate(${-ank} -14 -2)`}>
      <line x1={-14} y1={-2} x2={24} y2={-2} stroke={col} strokeWidth={12} strokeLinecap="round" />
    </g>
  </g>
);

/* ---------- 床 + 躺着的人 ---------- */
const SleepBed = ({x, y, s = 1, cramp = 1, zzz = 0, prop = 0}: {
  x: number; y: number; s?: number; cramp?: number; zzz?: number; prop?: number;
}) => {
  const f = useCurrentFrame();
  const kick = cramp > 0 ? Math.sin(f * 2.6) * 1.8 * cramp : 0;
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      <ellipse cx={-6} cy={150} rx={385} ry={17} fill={P.ink} opacity={0.4} />
      {/* 床头板 */}
      <rect x={-356} y={-176} width={34} height={300} rx={15} fill={P.bed} />
      <circle cx={-339} cy={-184} r={19} fill={P.bed} />
      {/* 床架 + 床腿 */}
      <rect x={-368} y={66} width={716} height={40} rx={17} fill={P.bed} />
      <rect x={-336} y={104} width={30} height={30} rx={11} fill={P.bed} />
      <rect x={300} y={104} width={30} height={30} rx={11} fill={P.bed} />
      {/* 床垫 */}
      <rect x={-360} y={0} width={712} height={70} rx={22} fill={P.pillow} />
      {/* 枕头 */}
      <ellipse cx={-262} cy={-26} rx={84} ry={30} fill={P.cream} />
      {/* 头 */}
      <g transform="translate(-252,-64)">
        <circle r={36} fill={C.skin} />
        <path d="M -35 -4 Q -32 -40 2 -38 Q 32 -36 33 -6 Q 0 -26 -35 -4 Z" fill={P.ink} />
        {cramp > 0
          ? <circle cx={12} cy={2} r={4.4} fill={P.ink} />
          : <path d="M 6 4 q 9 7 18 0" fill="none" stroke={P.ink} strokeWidth={3.4} strokeLinecap="round" />}
        {cramp > 0 && <Drop x={-30} y={-24} s={0.8} />}
      </g>
      {/* 被子（盖到小腿根） */}
      <path d="M -208 -24 C -160 -62 -60 -74 40 -62 C 116 -52 160 -34 172 -12 L 172 44 L -208 44 Z"
        fill={P.sheet} />
      <line x1={-120} y1={-52} x2={-120} y2={40} stroke={P.sheetD} strokeWidth={4} opacity={0.5} />
      <line x1={-8} y1={-60} x2={-8} y2={40} stroke={P.sheetD} strokeWidth={4} opacity={0.5} />
      {/* 露出的小腿 + 脚 */}
      <g transform={`translate(170,-26) translate(0,${kick})`}>
        <LowerLeg x={0} y={0} s={0.62} ank={prop ? -4 : -16} cramp={0}
          musc={cramp ? P.muscle : P.ion} />
      </g>
      {prop > 0 && <ellipse cx={268} cy={34} rx={62} ry={20} fill={P.cream} />}
      {/* 抽筋闪电 */}
      {cramp > 0 && (
        <g>
          <Zap x={212} y={-118} s={0.95} />
          <Zap x={296} y={-84} s={0.6} rot={28} />
          {/* 疼痛放射线 */}
          <line x1={252} y1={-72} x2={276} y2={-94} stroke={P.coral} strokeWidth={6} strokeLinecap="round" />
          <line x1={318} y1={-52} x2={342} y2={-64} stroke={P.coral} strokeWidth={6} strokeLinecap="round" />
        </g>
      )}
      {/* zzz */}
      {zzz > 0 && [0, 1, 2].map((i) => (
        <text key={i} x={-190 + i * 36} y={-140 - i * 34} fontSize={30 + i * 9} fill={P.muted}
          fontFamily={FONT_HEAD} opacity={0.35 + 0.3 * Math.sin(f * 0.12 + i * 1.5)}>z</text>
      ))}
    </g>
  );
};

/* ---------- 流动信号线 ---------- */
const FlowLine = ({x1, y1, x2, y2, col = P.nerve, n = 4, spd = 0.02, dash, on = 1, lw = 6}: {
  x1: number; y1: number; x2: number; y2: number; col?: string; n?: number; spd?: number;
  dash?: string; on?: number; lw?: number;
}) => {
  const f = useCurrentFrame();
  return (
    <g>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={col} strokeWidth={lw}
        strokeLinecap="round" strokeDasharray={dash} />
      {on > 0 && Array.from({length: n}).map((_, i) => {
        const p = wrap(f * spd + i / n, 1);
        return <circle key={i} cx={x1 + (x2 - x1) * p} cy={y1 + (y2 - y1) * p} r={9} fill={col} />;
      })}
    </g>
  );
};

/* ---------- 神经元 ---------- */
const Neuron = ({x, y, s = 1, fire = 0, col = P.nerve}: {
  x: number; y: number; s?: number; fire?: number; col?: string;
}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    {[0, 60, 120, 180, 240, 300].map((a) => (
      <line key={a} x1={Math.cos((a * Math.PI) / 180) * 18} y1={Math.sin((a * Math.PI) / 180) * 18}
        x2={Math.cos((a * Math.PI) / 180) * 36} y2={Math.sin((a * Math.PI) / 180) * 36}
        stroke={col} strokeWidth={6} strokeLinecap="round" opacity={0.8} />
    ))}
    <circle r={20} fill={fire > 0 ? col : P.card} stroke={col} strokeWidth={5} />
    {fire > 0 && <circle r={6} fill={P.ink} />}
  </g>
);

/* ---------- 大脑 ---------- */
const Brain = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <path d="M -64 6 C -70 -30 -40 -56 -4 -54 C 34 -56 62 -32 60 2 C 62 30 36 48 0 46
      C -38 48 -60 34 -64 6 Z" fill={P.card} stroke={P.muted} strokeWidth={5} />
    <path d="M -40 -20 q 14 -14 26 -2 q 14 -12 26 0" fill="none" stroke={P.muted}
      strokeWidth={5} strokeLinecap="round" />
    <path d="M -46 10 q 16 -12 30 0 q 16 -10 30 2" fill="none" stroke={P.muted}
      strokeWidth={5} strokeLinecap="round" />
  </g>
);

/* ---------- 迷你小人（跑 / 站 / 拉伸） ---------- */
const MiniRunner = ({x, y, s = 1, col = P.cream, phase = 0}: {
  x: number; y: number; s?: number; col?: string; phase?: number;
}) => {
  const f = useCurrentFrame();
  const q = Math.sin(f * 0.35 + phase);
  return (
    <g transform={`translate(${x},${y}) scale(${s})`} stroke={col} strokeWidth={9}
      strokeLinecap="round" fill="none">
      <circle cx={10} cy={-66} r={13} fill={col} stroke="none" />
      <line x1={6} y1={-50} x2={-4} y2={-22} />
      <line x1={-4} y1={-22} x2={10 + 14 * q} y2={0} />
      <line x1={-4} y1={-22} x2={-12 - 12 * q} y2={2} />
      <line x1={4} y1={-44} x2={20 - 10 * q} y2={-30} />
      <line x1={2} y1={-42} x2={-14 + 10 * q} y2={-30} />
    </g>
  );
};
const StandFig = ({x, y, s = 1, col = P.cream}: {x: number; y: number; s?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} stroke={col} strokeWidth={9}
    strokeLinecap="round" fill="none">
    <circle cy={-58} r={12} fill={col} stroke="none" />
    <line x1={0} y1={-44} x2={0} y2={-14} />
    <line x1={0} y1={-14} x2={10} y2={8} />
    <line x1={0} y1={-14} x2={-10} y2={8} />
    <line x1={0} y1={-36} x2={12} y2={-24} />
    <line x1={0} y1={-36} x2={-12} y2={-24} />
  </g>
);
const StretchFig = ({x, y, s = 1, col = P.ion}: {x: number; y: number; s?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={64} y={-124} width={14} height={124} rx={6} fill={P.card} />
    <g stroke={col} strokeWidth={10} strokeLinecap="round" fill="none">
      <circle cx={14} cy={-98} r={13} fill={col} stroke="none" />
      <path d="M 12 -82 L -12 -34" />
      <path d="M 8 -72 L 60 -80" />
      <path d="M 6 -60 L 58 -56" />
      <path d="M -12 -34 L 12 -16 L 26 -2" />
      <path d="M -12 -34 L -52 -12 L -60 -2" />
    </g>
    <line x1={-72} y1={4} x2={72} y2={4} stroke={P.muted} strokeWidth={5} strokeLinecap="round" />
  </g>
);

/* ---------- 水杯 / 胶囊 / 肠道 ---------- */
const Glass = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <path d="M -26 -44 L -20 40 L 20 40 L 26 -44 Z" fill={P.card} />
    <path d="M -23 -16 L -19 36 L 19 36 L 23 -16 Z" fill={P.sky} opacity={0.85} />
    <ellipse cy={-16} rx={23} ry={6} fill={P.skyD} />
  </g>
);
const Caps = ({x, y, s = 1, rot = -25}: {x: number; y: number; s?: number; rot?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`}>
    <rect x={-34} y={-13} width={68} height={26} rx={13} fill={P.violet} />
    <rect x={-34} y={-13} width={34} height={26} rx={13} fill={P.pillow} />
    <line x1={0} y1={-13} x2={0} y2={13} stroke={P.ink} strokeWidth={3} opacity={0.4} />
  </g>
);
const Gut = ({x, y, s = 1, col = P.muted}: {x: number; y: number; s?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} fill="none" stroke={col}
    strokeWidth={9} strokeLinecap="round">
    <path d="M -34 -22 L 24 -22 Q 40 -22 40 -8 Q 40 6 24 6 L -24 6 Q -40 6 -40 20 Q -40 34 -24 34 L 34 34" />
  </g>
);

/* ---------- 凌晨两点的小钟 ---------- */
const Clock2am = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <circle r={46} fill={P.bgDeep} stroke={P.moon} strokeWidth={6} />
    {[0, 1, 2, 3].map((i) => (
      <line key={i} x1={0} y1={-40} x2={0} y2={-32} stroke={P.muted} strokeWidth={4}
        transform={`rotate(${i * 90})`} />
    ))}
    <line x1={0} y1={0} x2={0} y2={-32} stroke={P.cream} strokeWidth={6} strokeLinecap="round" />
    <line x1={0} y1={0} x2={17} y2={-10} stroke={P.moon} strokeWidth={7} strokeLinecap="round" />
    <circle r={5} fill={P.moon} />
  </g>
);

/* ---------- 角度弧（踝处，x 轴正向 = 小腿延长线） ---------- */
const AngleArc = ({x, y, ank, r = 110, col = P.moon, label, lx, ly}: {
  x: number; y: number; ank: number; r?: number; col?: string; label: string;
  lx?: number; ly?: number;
}) => {
  const a = (-ank * Math.PI) / 180;
  const ex = x + r * Math.cos(a);
  const ey = y + r * Math.sin(a);
  const sweep = ank >= 0 ? 0 : 1;
  const tx = lx !== undefined ? lx : x + (r + 36) * Math.cos(a / 2);
  const ty = ly !== undefined ? ly : y + (r + 36) * Math.sin(a / 2);
  return (
    <g>
      <path d={`M ${x + r} ${y} A ${r} ${r} 0 0 ${sweep} ${ex} ${ey}`} fill="none"
        stroke={col} strokeWidth={5} strokeLinecap="round" strokeDasharray="8 8" />
      <text x={tx} y={ty + 8} textAnchor="middle" fontSize={23} fill={col}
        fontFamily={FONT_HEAD} fontWeight={800}>{label}</text>
    </g>
  );
};

/* ============ 标题卡：半夜床上抽筋 ============ */
const TitleCard = ({data}: {data: any}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 13, mass: 0.7}});
  const o = interpolate(f, [0, 8], [0, 1]);
  const pA = Math.max(0.001, spring({frame: f - 4, fps, config: {damping: 10, stiffness: 150}}));
  const pB = Math.max(0.001, spring({frame: f - 16, fps, config: {damping: 10, stiffness: 150}}));
  return (
    <AbsoluteFill>
      <Backdrop top={P.bg} bottom={P.bg} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={22} from={0} dur={30} />
        <Moon x={1640} y={185} scale={1.05} />
        <g transform={`translate(960,600) scale(${pA}) translate(-960,-600)`} opacity={clamp01(pA)}>
          <SleepBed x={960} y={600} s={1.02} cramp={1} />
        </g>
        <g transform={`translate(500,318) scale(${pB})`} opacity={clamp01(pB)}>
          <TagPill x={0} y={0} t="小腿拧成麻花" col={P.muscle} w={252} />
        </g>
        <g transform={`translate(1420,318) scale(${pB})`} opacity={clamp01(pB)}>
          <TagPill x={0} y={0} t="偏偏挑半夜" col={P.moon} w={240} />
        </g>
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 500}}>
          <div style={{color: P.moon, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 24}}>
            冷知识 · 第 {data.episode} 期
          </div>
          <div style={{color: C.text, fontSize: 72, fontWeight: 900, fontFamily: FONT_HEAD, letterSpacing: 4}}>
            {data.title}
          </div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ============ S0 半夜疼醒：现场 + 钙背不动的锅 + 四个条件 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);
  const popBed = Math.max(0.001, spring({frame: f - 8, fps, config: {damping: 11, stiffness: 120}}));
  const popCa = Math.max(0.001, spring({frame: f - 24, fps, config: {damping: 11, stiffness: 120}}));
  const caX = clamp01((f - b1 - 8) / 16);
  const cardK = [0, 1, 2, 3].map((i) =>
    Math.max(0.001, spring({frame: f - b2 - 4 - i * 12, fps, config: {damping: 12, stiffness: 140}})));
  const conds = [
    {col: P.sky, t: '水分没补上', sub: '几小时没喝 · 轻度脱水'},
    {col: P.ion, t: '离子浓度乱', sub: '钙钠钾镁 · 失衡'},
    {col: P.nerve, t: '神经误触发', sub: '睡着后 · 抑制下调'},
    {col: P.muscle, t: '脚尖整夜绷', sub: '小腿被摆在缩短位'},
  ];
  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={0} name={sec.name} accent={P.moon} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：半夜现场 */}
        <g transform={`translate(570,330) scale(${popBed}) translate(-570,-330)`} opacity={clamp01(popBed * 2)}>
          <rect x={120} y={140} width={900} height={470} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <PanelTitle x={160} y={196} text="半夜现场 · 疼得弹起来" />
          <Moon x={920} y={215} scale={0.5} />
          <SleepBed x={575} y={470} s={0.8} cramp={1} />
          <text x={160} y={576} fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>
            小腿拧成麻花 · 脚趾绷成钩子
          </text>
        </g>
        {/* 右：第一嫌疑人 */}
        <g transform={`translate(1430,330) scale(${popCa}) translate(-1430,-330)`} opacity={clamp01(popCa * 2)}>
          <rect x={1060} y={140} width={740} height={470} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <PanelTitle x={1100} y={196} text="脱口而出 · 第一嫌疑人" />
          {/* 钙片瓶 */}
          <g transform="translate(1210,400)">
            <rect x={-46} y={-84} width={92} height={140} rx={18} fill={P.card}
              stroke={P.muted} strokeWidth={4} />
            <rect x={-32} y={-104} width={64} height={24} rx={8} fill={P.moon} />
            <rect x={-30} y={-56} width={60} height={62} rx={12} fill={P.bgDeep} />
            <text x={0} y={-12} textAnchor="middle" fontSize={34} fill={P.moon}
              fontFamily={FONT_HEAD} fontWeight={900}>Ca</text>
          </g>
          <text x={1420} y={370} textAnchor="middle" fontSize={74} fill={P.cream}
            fontFamily={FONT_HEAD} fontWeight={900}>缺钙！</text>
          <text x={1420} y={440} textAnchor="middle" fontSize={24} fill={P.muted}
            fontFamily={FONT_HEAD}>确实参与 · 但背不动整口锅</text>
          {/* 打叉 */}
          {caX > 0 && (
            <g opacity={caX} transform={`translate(1420,346) scale(${0.6 + 0.4 * caX})`}>
              <line x1={-130} y1={42} x2={130} y2={-56} stroke={P.coral} strokeWidth={16}
                strokeLinecap="round" />
            </g>
          )}
          {caX > 0.7 && (
            <text x={1420} y={530} textAnchor="middle" fontSize={28} fill={P.coral}
              fontFamily={FONT_HEAD} fontWeight={900}>钙：这锅我不全背</text>
          )}
        </g>
        {/* 下：条件凑齐 */}
        <rect x={120} y={640} width={1680} height={240} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <PanelTitle x={160} y={696} text="真正的配方 · 几个条件恰好凑齐" col={P.cream} />
        {conds.map((cd, i) => {
          const cx = 340 + i * 420;
          const k = cardK[i];
          return (
            <g key={i} transform={`translate(${cx},778) scale(${k})`} opacity={clamp01(k * 2)}>
              <rect x={-175} y={-62} width={350} height={124} rx={18} fill={P.bgDeep}
                stroke={cd.col} strokeWidth={4} />
              {i === 0 && <Drop x={-118} y={-6} s={1.15} col={cd.col} />}
              {i === 1 && (
                <g transform="translate(-118,-4)">
                  <circle cx={-14} cy={-12} r={11} fill={P.moon} />
                  <circle cx={12} cy={0} r={11} fill={P.ion} />
                  <circle cx={-8} cy={16} r={11} fill={P.violet} />
                </g>
              )}
              {i === 2 && <Zap x={-118} y={-2} s={0.95} col={cd.col} />}
              {i === 3 && <FootGlyph x={-104} y={16} s={0.8} ank={-62} col={cd.col} />}
              <text x={30} y={-10} textAnchor="middle" fontSize={27} fill={cd.col}
                fontFamily={FONT_HEAD} fontWeight={900}>{cd.t}</text>
              <text x={30} y={32} textAnchor="middle" fontSize={19} fill={P.muted}
                fontFamily={FONT_HEAD}>{cd.sub}</text>
            </g>
          );
        })}
      </svg>
      <Chip text="锅不只一口" x={1490} y={66} at={30} accent={P.moon} />
      <Stamp text="条件凑齐了" x={545} y={86} at={b2 + 36} color={P.moonD} size={42} rot={-3} />
    </FadeIn>
  );
};

/* ============ S1 抽的到底是什么：收缩信号卡死 + 小腿三头肌 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const popL = Math.max(0.001, spring({frame: f - 8, fps, config: {damping: 11, stiffness: 120}}));
  const popR = Math.max(0.001, spring({frame: f - 16, fps, config: {damping: 11, stiffness: 120}}));
  const legK = Math.max(0.001, spring({frame: f - b1 - 6, fps, config: {damping: 11, stiffness: 120}}));
  const normLen = 380 + 44 * Math.sin(f * 0.055);
  return (
    <FadeIn>
      <Backdrop top={P.bgDeep} bottom={P.bgDeep} />
      <SceneTag index={1} name={sec.name} accent={P.muscle} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：收缩信号卡死 */}
        <g transform={`translate(560,510) scale(${popL}) translate(-560,-510)`} opacity={clamp01(popL * 2)}>
          <rect x={120} y={140} width={880} height={740} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <PanelTitle x={160} y={196} text="一块肌肉 · 卡在收缩挡位" />
          {/* 信号通路 */}
          <text x={292} y={252} textAnchor="end" fontSize={21} fill={P.muted} fontFamily={FONT_HEAD}>
            运动神经元
          </text>
          <Neuron x={330} y={296} s={1.05} fire={1} />
          <FlowLine x1={368} y1={296} x2={664} y2={296} col={P.nerve} n={4} spd={0.028} />
          <text x={516} y={272} textAnchor="middle" fontSize={22} fill={P.nerve}
            fontFamily={FONT_HEAD} fontWeight={800}>收缩指令 · 一直放电</text>
          {/* 放松指令断线 */}
          <line x1={352} y1={318} x2={664} y2={366} stroke={P.muted} strokeWidth={5}
            strokeDasharray="10 12" opacity={0.85} />
          <MarkX x={512} y={344} s={0.9} />
          <text x={516} y={410} textAnchor="middle" fontSize={22} fill={P.coral}
            fontFamily={FONT_HEAD} fontWeight={800}>放松指令 · 半路断线</text>
          <MuscleStrip x={790} y={300} len={118} th={52} shake={1.1} />
          <Zap x={790} y={222} s={0.72} />
          {/* 两种状态对照 */}
          <text x={160} y={468} fontSize={22} fill={P.cream} fontFamily={FONT_HEAD}>
            同一块肌肉 · 两种状态
          </text>
          <MuscleStrip x={530} y={548} len={normLen} th={58} col={P.sheet} fiber={P.sheetD} />
          <text x={530} y={618} textAnchor="middle" fontSize={21} fill={P.muted}
            fontFamily={FONT_HEAD}>正常 · 缩完还能松开</text>
          <MuscleStrip x={530} y={716} len={238} th={84} shake={1.5} />
          <LockIcon x={530} y={644} s={0.8} />
          <Zap x={716} y={716} s={0.66} rot={16} />
          <text x={530} y={796} textAnchor="middle" fontSize={22} fill={P.muscle}
            fontFamily={FONT_HEAD} fontWeight={800}>抽筋 · 锁死在短位</text>
          <text x={530} y={838} textAnchor="middle" fontSize={20} fill={P.muted}
            fontFamily={FONT_HEAD}>缺血 + 过紧 = 疼</text>
        </g>
        {/* 右：小腿三头肌（板先出，内容随 cue1 点亮） */}
        <g transform={`translate(1420,510) scale(${popR}) translate(-1420,-510)`} opacity={clamp01(popR * 2)}>
          <rect x={1040} y={140} width={760} height={740} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <PanelTitle x={1080} y={196} text="头号目标 · 居然总是它" />
          {/* 虚位以待的轮廓 */}
          <g opacity={clamp01((1 - legK) * 0.9)}>
            <rect x={1120} y={300} width={600} height={330} rx={18} fill="none"
              stroke={P.card} strokeWidth={4} strokeDasharray="14 14" />
            <text x={1420} y={478} textAnchor="middle" fontSize={24} fill={P.muted}
              fontFamily={FONT_HEAD}>它是谁？</text>
          </g>
          <g transform={`translate(1420,500) scale(${legK}) translate(-1420,-500)`}
            opacity={clamp01(legK * 2)}>
            <LowerLeg x={1430} y={500} s={1.55} ank={-14} cramp={0.35} />
            {/* 标注引线 */}
            <line x1={1352} y1={444} x2={1560} y2={366} stroke={P.muscle} strokeWidth={3}
              strokeDasharray="5 7" />
            <text x={1570} y={360} fontSize={22} fill={P.muscle} fontFamily={FONT_HEAD}
              fontWeight={800}>腓肠肌</text>
            <line x1={1416} y1={508} x2={1620} y2={576} stroke={P.muted} strokeWidth={3}
              strokeDasharray="5 7" />
            <text x={1630} y={582} fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>跟腱</text>
          </g>
          <g opacity={clamp01(legK * 2)}>
            <text x={1080} y={668} fontSize={22} fill={P.coral} fontFamily={FONT_HEAD}>
              睡觉时脚尖下垂 → 它被整晚摆在缩短位
            </text>
            <line x1={1080} y1={706} x2={1760} y2={706} stroke={P.card} strokeWidth={3}
              strokeDasharray="6 10" />
            <text x={1080} y={752} fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>
              白天 · 走路站立爬楼全包圆
            </text>
            <text x={1080} y={800} fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>
              夜里 · 神经兴奋性本来就偏高
            </text>
          </g>
        </g>
      </svg>
      <Chip text="短路卡死" x={1490} y={66} at={26} accent={P.muscle} />
      <Stamp text="松不回去了" x={545} y={86} at={b1 + 30} color={P.muscleD} size={42} rot={-3} />
    </FadeIn>
  );
};

/* ============ S2 电解质这笔账：离子搬运 + 四条丢失路 + 失衡点各异 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);
  const popL = Math.max(0.001, spring({frame: f - 8, fps, config: {damping: 11, stiffness: 120}}));
  const popR = Math.max(0.001, spring({frame: f - b1 - 4, fps, config: {damping: 11, stiffness: 120}}));
  const personK = [0, 1, 2].map((i) =>
    Math.max(0.001, spring({frame: f - b2 + 44 - i * 14, fps, config: {damping: 12, stiffness: 140}})));
  const ions = [
    {ch: 0, col: P.moon, ph: 0, t: 'Ca'},
    {ch: 1, col: P.sky, ph: 1.9, t: 'Na'},
    {ch: 2, col: P.ion, ph: 3.6, t: 'K'},
    {ch: 1, col: P.violet, ph: 5.1, t: 'Mg'},
  ];
  const leaks = [
    {t: '大量出汗', sub: '汗里全是盐'},
    {t: '腹泻脱水', sub: '连水带盐丢'},
    {t: '利尿剂', sub: '药性副作用'},
    {t: '比赛后半程', sub: '抽筋率飙升'},
  ];
  const people = [
    {tag: '甲', h: [88, 40, 70, 30]},
    {tag: '乙', h: [36, 78, 34, 82]},
    {tag: '丙', h: [62, 30, 84, 44]},
  ];
  const ionCols = [P.moon, P.sky, P.ion, P.violet];
  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={2} name={sec.name} accent={P.ion} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左上：细胞膜搬运工 */}
        <g transform={`translate(620,355) scale(${popL}) translate(-620,-355)`} opacity={clamp01(popL * 2)}>
          <rect x={120} y={140} width={1000} height={430} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <PanelTitle x={160} y={196} text="细胞膜两侧 · 离子来回搬运" />
          <text x={176} y={318} fontSize={20} fill={P.muted} fontFamily={FONT_HEAD}>细胞外</text>
          <text x={176} y={500} fontSize={20} fill={P.muted} fontFamily={FONT_HEAD}>细胞内</text>
          {/* 膜双层 */}
          <line x1={180} y1={368} x2={1060} y2={368} stroke={P.card} strokeWidth={12} strokeLinecap="round" />
          <line x1={180} y1={436} x2={1060} y2={436} stroke={P.card} strokeWidth={12} strokeLinecap="round" />
          {/* 三个通道 */}
          {[380, 640, 880].map((cx) => (
            <rect key={cx} x={cx - 22} y={346} width={44} height={112} rx={14}
              fill={P.bgDeep} stroke={P.ionD} strokeWidth={4} />
          ))}
          {/* 穿梭的离子 */}
          {ions.map((io, i) => {
            const cx = [380, 640, 880][io.ch];
            const yy = 402 + Math.sin(f * 0.075 + io.ph) * 62;
            return (
              <g key={i}>
                <circle cx={cx + Math.sin(f * 0.3 + i) * 6} cy={yy} r={17} fill={io.col} />
                <text x={cx + Math.sin(f * 0.3 + i) * 6} y={yy + 7} textAnchor="middle"
                  fontSize={19} fill={P.ink} fontFamily={FONT_HEAD} fontWeight={900}>{io.t}</text>
              </g>
            );
          })}
          {/* 游离离子 */}
          {[0, 1, 2, 3].map((i) => (
            <circle key={i} cx={250 + i * 220 + rand(i, 5) * 60} cy={290 - rand(i, 6) * 34}
              r={10} fill={ionCols[i]} opacity={0.55} />
          ))}
          <text x={160} y={536} fontSize={21} fill={P.muted} fontFamily={FONT_HEAD}>
            哪个浓度乱了套 · 收缩舒张都可能卡壳
          </text>
        </g>
        {/* 右上：四条丢失路 */}
        <g transform={`translate(1480,355) scale(${popR}) translate(-1480,-355)`} opacity={clamp01(popR * 2)}>
          <rect x={1160} y={140} width={640} height={430} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <PanelTitle x={1200} y={196} text="每天都在漏 · 四条路" />
          {leaks.map((lk, i) => {
            const cx = 1330 + (i % 2) * 320;
            const cy = 300 + Math.floor(i / 2) * 150;
            return (
              <g key={i} transform={`translate(${cx},${cy})`}>
                {i === 0 && <Drop x={-104} y={-6} s={1.1} col={P.sky} />}
                {i === 1 && <Gut x={-104} y={-14} s={0.8} col={P.moon} />}
                {i === 2 && <Caps x={-104} y={-8} s={0.85} />}
                {i === 3 && <MiniRunner x={-104} y={14} s={0.62} col={P.coral} />}
                <text x={-52} y={-8} fontSize={24} fill={P.cream} fontFamily={FONT_HEAD}
                  fontWeight={800}>{lk.t}</text>
                <text x={-52} y={30} fontSize={19} fill={P.muted} fontFamily={FONT_HEAD}>{lk.sub}</text>
              </g>
            );
          })}
          <text x={1200} y={536} fontSize={21} fill={P.coral} fontFamily={FONT_HEAD}>
            汗一直在出 · 电解质一直在丢
          </text>
        </g>
        {/* 下：失衡点人人不同 */}
        <rect x={120} y={610} width={1680} height={270} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <PanelTitle x={160} y={666} text="所以光补一种 · 不一定对症" col={P.cream} />
        {people.map((pp, i) => {
          const cx = 420 + i * 460;
          const k = personK[i];
          return (
            <g key={i} transform={`translate(${cx},792) scale(${k})`} opacity={clamp01(k * 2)}>
              <rect x={-170} y={-92} width={340} height={168} rx={18} fill={P.bgDeep}
                stroke={P.card} strokeWidth={3.6} />
              <text x={-140} y={-58} fontSize={24} fill={P.muted} fontFamily={FONT_HEAD}
                fontWeight={800}>{pp.tag}</text>
              {pp.h.map((h, j) => (
                <g key={j}>
                  <rect x={-108 + j * 62} y={42 - h * 0.8} width={34} height={h * 0.8}
                    rx={8} fill={ionCols[j]} />
                </g>
              ))}
              <line x1={-124} y1={42} x2={124} y2={42} stroke={P.muted} strokeWidth={3} />
              <text x={0} y={68} textAnchor="middle" fontSize={19} fill={P.muted}
                fontFamily={FONT_HEAD}>缺的各不相同</text>
            </g>
          );
        })}
        {/* 图例 */}
        {['Ca', 'Na', 'K', 'Mg'].map((t, j) => (
          <g key={t} transform={`translate(1620,${664 + j * 48})`}>
            <circle cx={0} cy={0} r={12} fill={ionCols[j]} />
            <text x={22} y={8} fontSize={20} fill={P.muted} fontFamily={FONT_HEAD}>{t}</text>
          </g>
        ))}
      </svg>
      <Chip text="钙钠钾镁" x={1490} y={66} at={26} accent={P.ion} />
      <Stamp text="失衡点各异" x={545} y={86} at={b2 + 34} color={P.ionD} size={42} rot={-3} />
    </FadeIn>
  );
};

/* ============ S3 睡姿是半个凶手：踝跖屈缩短位 vs 回勾解锁 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const popL = Math.max(0.001, spring({frame: f - 8, fps, config: {damping: 11, stiffness: 120}}));
  const popR = Math.max(0.001, spring({frame: f - b1 - 4, fps, config: {damping: 11, stiffness: 120}}));
  const ankL = -14 - Math.sin(f * 0.08) * 4;
  const ankR = 62 - Math.abs(Math.sin(f * 0.06)) * 10;
  return (
    <FadeIn>
      <Backdrop top={P.bgDeep} bottom={P.bgDeep} />
      <SceneTag index={3} name={sec.name} accent={P.moon} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：踝跖屈 = 缩短位 */}
        <g transform={`translate(535,510) scale(${popL}) translate(-535,-510)`} opacity={clamp01(popL * 2)}>
          <rect x={120} y={140} width={830} height={740} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <PanelTitle x={160} y={196} text="睡着 · 脚自己摆成绷直位" />
          <Moon x={880} y={196} scale={0.42} />
          {/* 床边沿 */}
          <line x1={170} y1={610} x2={900} y2={610} stroke={P.card} strokeWidth={10}
            strokeLinecap="round" opacity={0.7} />
          <LowerLeg x={520} y={548} s={1.7} ank={ankL} cramp={0.4} />
          <AngleArc x={520} y={554} ank={ankL} r={105} col={P.moon} label="踝跖屈" lx={742} ly={668} />
          {/* 肌腹状态条 */}
          <MuscleStrip x={330} y={756} len={160} th={52} col={P.muscle} />
          <text x={530} y={748} fontSize={22} fill={P.muscle} fontFamily={FONT_HEAD}
            fontWeight={800}>小腿三头肌 · 缩短位</text>
          <text x={530} y={788} fontSize={20} fill={P.muted} fontFamily={FONT_HEAD}>
            张力感知跟着跑偏
          </text>
          <text x={160} y={852} fontSize={21} fill={P.muted} fontFamily={FONT_HEAD}>
            一整晚绷着 · 触发信号一来就卡死
          </text>
        </g>
        {/* 右：回勾 = 解锁 */}
        <g transform={`translate(1385,510) scale(${popR}) translate(-1385,-510)`} opacity={clamp01(popR * 2)}>
          <rect x={970} y={140} width={830} height={740} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <PanelTitle x={1010} y={196} text="脚尖往回勾 · 强行解锁" />
          <line x1={1020} y1={610} x2={1750} y2={610} stroke={P.card} strokeWidth={10}
            strokeLinecap="round" opacity={0.7} />
          <LowerLeg x={1390} y={548} s={1.7} ank={ankR} cramp={0} musc={P.ion} />
          <AngleArc x={1390} y={554} ank={ankR} r={105} col={P.ion} label="回勾" lx={1570} ly={452} />
          {/* 勾脚方向箭头 */}
          <path d="M 1560 420 Q 1520 360 1450 352" fill="none" stroke={P.ion}
            strokeWidth={7} strokeLinecap="round" strokeDasharray="10 9" />
          <polygon points="1436,348 1460,340 1454,364" fill={P.ion} />
          <MuscleStrip x={1160} y={756} len={230} th={46} col={P.ion} fiber={P.ionD} />
          <text x={1420} y={748} fontSize={22} fill={P.ion} fontFamily={FONT_HEAD}
            fontWeight={800}>肌肉被拉回长位</text>
          <LockOpen x={1690} y={736} s={0.8} />
          <text x={1010} y={852} fontSize={21} fill={P.cream} fontFamily={FONT_HEAD}>
            疼痛往往立刻缓解 · 这是力学不是玄学
          </text>
        </g>
      </svg>
      <Chip text="半个凶手是睡姿" x={1450} y={66} at={26} accent={P.moon} />
      <Stamp text="一勾就松" x={545} y={86} at={b1 + 34} color={P.ionD} size={44} rot={-3} />
    </FadeIn>
  );
};

/* ============ S4 半夜的完美条件：神经误触发 + 凑齐 + 救急预防 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);
  const popL = Math.max(0.001, spring({frame: f - 8, fps, config: {damping: 11, stiffness: 120}}));
  const popB = Math.max(0.001, spring({frame: f - b1 + 30, fps, config: {damping: 11, stiffness: 120}}));
  const rowAt = [b2 - 50, b2 - 32, b2 - 14, b2 + 10, b2 + 30, b2 + 50];
  const rowK = rowAt.map((a) =>
    Math.max(0.001, spring({frame: f - a, fps, config: {damping: 12, stiffness: 150}})));
  const zapK = clamp01(1 - Math.abs(wrap(f, 52) - 8) / 8);
  const conds4 = [
    {x: 250, y: 726, col: P.sky, t: '脱水'},
    {x: 250, y: 842, col: P.violet, t: '疲劳'},
    {x: 856, y: 726, col: P.muscle, t: '缩短位'},
    {x: 856, y: 842, col: P.nerve, t: '神经兴奋'},
  ];
  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={4} name={sec.name} accent={P.nerve} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左上：睡着后神经没全关 */}
        <g transform={`translate(580,365) scale(${popL}) translate(-580,-365)`} opacity={clamp01(popL * 2)}>
          <rect x={120} y={140} width={920} height={450} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <PanelTitle x={160} y={196} text="睡着后 · 值班的并没全关" />
          <Brain x={300} y={316} s={1.05} />
          <text x={300} y={262} textAnchor="middle" fontSize={22} fill={P.muted}
            fontFamily={FONT_HEAD}>大脑抑制 · 整体下调</text>
          <line x1={300} y1={372} x2={300} y2={492} stroke={P.nerve} strokeWidth={7}
            strokeLinecap="round" opacity={0.8} />
          <Neuron x={300} y={516} s={1} fire={zapK > 0.55 ? 1 : 0} />
          <text x={246} y={522} textAnchor="end" fontSize={20} fill={P.muted}
            fontFamily={FONT_HEAD}>运动神经元</text>
          <FlowLine x1={336} y1={516} x2={640} y2={516} col={P.nerve} n={2} spd={0.018} lw={5} />
          <MuscleStrip x={750} y={516} len={120} th={48} shake={zapK * 2.2} />
          {/* 小刺激漂来 + 误触发闪电 */}
          {[0, 1, 2].map((i) => {
            const p = wrap(f * 0.016 + i * 0.33, 1);
            const cx = 150 + p * 128;
            const cy = 440 + Math.sin(i * 2.1) * 18 + p * 60;
            return <circle key={i} cx={cx} cy={cy} r={6}
              fill={[P.violet, P.sky, P.coral][i]} opacity={clamp01(p * 3) * 0.8} />;
          })}
          {zapK > 0.05 && <Zap x={352} y={462} s={zapK} rot={-14} />}
          <text x={160} y={560} fontSize={21} fill={P.muted} fontFamily={FONT_HEAD}>
            位置一变 · 局部缺血 · 代谢废物 → 都可能误触发
          </text>
        </g>
        {/* 左下：半夜全凑齐 */}
        <g transform={`translate(580,750) scale(${popB}) translate(-580,-750)`} opacity={clamp01(popB * 2)}>
          <rect x={120} y={620} width={920} height={260} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <PanelTitle x={160} y={676} text="到半夜 · 条件全凑齐" />
          <Clock2am x={560} y={782} s={0.92} />
          <text x={560} y={858} textAnchor="middle" fontSize={20} fill={P.moon}
            fontFamily={FONT_HEAD} fontWeight={800}>凌晨 · 体温最低</text>
          {conds4.map((cd, i) => {
            const dx = 560 - cd.x;
            const dy = 782 - cd.y;
            const dd = Math.hypot(dx, dy);
            const ax = cd.x + (dx / dd) * 66;
            const ay = cd.y + (dy / dd) * 66;
            return (
              <g key={i}>
                <line x1={ax} y1={ay} x2={cd.x + (dx / dd) * (dd - 62)}
                  y2={cd.y + (dy / dd) * (dd - 62)} stroke={cd.col} strokeWidth={4}
                  strokeDasharray="6 8" opacity={0.8} />
                <rect x={cd.x - 78} y={cd.y - 28} width={156} height={56} rx={16}
                  fill={P.bgDeep} stroke={cd.col} strokeWidth={3.6} />
                <text x={cd.x} y={cd.y + 9} textAnchor="middle" fontSize={23} fill={cd.col}
                  fontFamily={FONT_HEAD} fontWeight={800}>{cd.t}</text>
              </g>
            );
          })}
        </g>
        {/* 右：救急与预防 */}
        <rect x={1080} y={140} width={720} height={740} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <PanelTitle x={1120} y={196} text="救急与预防 · 今晚就能用" col={P.cream} />
        <text x={1120} y={252} fontSize={24} fill={P.ion} fontFamily={FONT_HEAD} fontWeight={900}>
          正在抽 · 反向解锁
        </text>
        <g opacity={clamp01(rowK[0] * 2)} transform={`translate(0,${(1 - rowK[0]) * 14})`}>
          <FootGlyph x={1188} y={316} s={0.85} ank={30} col={P.ion} />
          <text x={1260} y={308} fontSize={24} fill={P.cream} fontFamily={FONT_HEAD}>
            脚尖往回勾 · 按住揉一揉
          </text>
          <text x={1260} y={344} fontSize={20} fill={P.ion} fontFamily={FONT_HEAD}>
            几十秒就能缓解
          </text>
        </g>
        <g opacity={clamp01(rowK[1] * 2)} transform={`translate(0,${(1 - rowK[1]) * 14})`}>
          <StandFig x={1190} y={428} s={0.8} col={P.ion} />
          <text x={1260} y={408} fontSize={24} fill={P.cream} fontFamily={FONT_HEAD}>
            下床站一站 · 走两步
          </text>
          <text x={1260} y={444} fontSize={20} fill={P.ion} fontFamily={FONT_HEAD}>
            自重把小腿拉长
          </text>
        </g>
        <g opacity={clamp01(rowK[2] * 2)} transform={`translate(0,${(1 - rowK[2]) * 14})`}>
          <MarkX x={1190} y={500} s={0.85} />
          <text x={1260} y={508} fontSize={24} fill={P.coral} fontFamily={FONT_HEAD}>
            别猛蹬腿 · 等于追加收缩指令
          </text>
        </g>
        <line x1={1120} y1={548} x2={1760} y2={548} stroke={P.card} strokeWidth={3}
          strokeDasharray="6 10" />
        <text x={1120} y={592} fontSize={24} fill={P.moon} fontFamily={FONT_HEAD} fontWeight={900}>
          平时防 · 一项项拆
        </text>
        <g opacity={clamp01(rowK[3] * 2)} transform={`translate(0,${(1 - rowK[3]) * 14})`}>
          <Glass x={1190} y={648} s={0.78} />
          <text x={1260} y={640} fontSize={24} fill={P.cream} fontFamily={FONT_HEAD}>
            白天喝够水 · 别等渴
          </text>
        </g>
        <g opacity={clamp01(rowK[4] * 2)} transform={`translate(0,${(1 - rowK[4]) * 14})`}>
          <StretchFig x={1176} y={756} s={0.52} />
          <text x={1260} y={724} fontSize={24} fill={P.cream} fontFamily={FONT_HEAD}>
            睡前拉伸小腿 · 三十秒
          </text>
        </g>
        <g opacity={clamp01(rowK[5] * 2)} transform={`translate(0,${(1 - rowK[5]) * 14})`}>
          <ellipse cx={1176} cy={806} rx={40} ry={15} fill={P.cream} />
          <FootGlyph x={1196} y={796} s={0.7} ank={-30} col={P.moon} />
          <text x={1260} y={800} fontSize={24} fill={P.cream} fontFamily={FONT_HEAD}>
            脚尖垫高点 · 别整夜绷着
          </text>
          <text x={1260} y={836} fontSize={20} fill={P.muted} fontFamily={FONT_HEAD}>
            频繁抽 · 去查电解质和神经
          </text>
        </g>
      </svg>
      <Chip text="今晚就能试" x={1490} y={66} at={26} accent={P.nerve} />
      <Stamp text="把条件拆掉" x={545} y={86} at={b2 + 34} color={P.moonD} size={42} rot={-3} />
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
        <Stars n={20} from={0} dur={26} />
        <Moon x={1520} y={200} scale={0.95} />
        <g transform={`translate(960,790) scale(${pop})`} opacity={clamp01(pop)}>
          <SleepBed x={0} y={0} s={0.95} cramp={0} zzz={1} prop={1} />
          <TagPill x={-440} y={-230} t="睡前拉拉小腿" col={P.ion} w={250} />
          <TagPill x={430} y={-230} t="水杯放床头" col={P.sky} w={230} />
        </g>
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -330}}>
        <div style={{fontSize: 52, fontWeight: 800, color: C.text}}>把条件拆掉 · 睡个安稳觉</div>
        <div style={{fontSize: 36, color: P.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E250 = ({data = tl}: {data?: any}) => {
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
