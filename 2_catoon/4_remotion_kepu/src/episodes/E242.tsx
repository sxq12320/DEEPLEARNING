import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01} from '../lib/style';
import {Backdrop, Stamp, SceneTag, Chip, Subtitles, ProgressBar, FadeIn} from '../lib/atoms';
import tl from '../timeline_242.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：训练馆墨绿 × 肌纤维珊瑚 × 乳酸黄 × 信号红 × 恢复青 ============ */
const P = {
  bg: '#13302a',
  bgDeep: '#0b211d',
  bgLite: '#1c4038',
  panel: '#17342d',
  card: '#2b5346',
  ink: '#0d1a16',
  cream: '#f4eeda',
  muted: '#a4c9bb',
  muscle: '#ff7a5c',
  muscleD: '#cf5138',
  muscleL: '#ffb59d',
  fascia: '#ffc9b3',
  tendon: '#f0e2c6',
  lact: '#ffd166',
  lactD: '#e0a53d',
  alarm: '#ff5d5d',
  alarmD: '#c93a3a',
  aqua: '#55d9c0',
  aquaD: '#2fa08c',
  sky: '#5cc8ff',
  vio: '#c9a8ff',
  nerve: '#ffe08a',
  track: '#ff9e57',
};

/* ---------- 小叉 / 小勾 ---------- */
const MarkX = ({x, y, s = 1, col = P.alarm}: {x: number; y: number; s?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} stroke={col} strokeWidth={7} strokeLinecap="round">
    <line x1={-13} y1={-13} x2={13} y2={13} /><line x1={13} y1={-13} x2={-13} y2={13} />
  </g>
);
/* ---------- 疼痛火花 ---------- */
const Spark = ({x, y, s = 1, col = P.alarm, spin = 1}: {x: number; y: number; s?: number; col?: string; spin?: number}) => {
  const f = useCurrentFrame();
  const pulse = 0.75 + Math.abs(Math.sin(f * 0.35)) * 0.45;
  return (
    <g transform={`translate(${x},${y}) scale(${s * pulse}) rotate(${f * spin})`} stroke={col}
      strokeWidth={5} strokeLinecap="round">
      {[0, 1, 2, 3, 4, 5].map((i) => {
        const a = i * 60 * Math.PI / 180;
        return <line key={i} x1={Math.cos(a) * 12} y1={Math.sin(a) * 12}
          x2={Math.cos(a) * 30} y2={Math.sin(a) * 30} />;
      })}
      <circle r={6} fill={col} stroke="none" />
    </g>
  );
};

/* ---------- 圆角标签牌 ---------- */
const TagPill = ({x, y, t, col, s = 1, w = 200, fs = 30, tcol}: {
  x: number; y: number; t: string; col: string; s?: number; w?: number; fs?: number; tcol?: string;
}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-w / 2} y={-30} width={w} height={60} rx={16} fill={P.panel} stroke={col} strokeWidth={4} />
    <text x={0} y={11} textAnchor="middle" fontSize={fs} fill={tcol ?? col}
      fontFamily={FONT_HEAD} fontWeight={800}>{t}</text>
  </g>
);

/* ---------- 楼梯上横着走的人 ---------- */
const StairPerson = ({x, y, s = 1}: {x: number; y: number; s?: number}) => {
  const f = useCurrentFrame();
  const wob = Math.sin(f * 0.12) * 2;
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      {/* 楼梯（向右下行） */}
      {[0, 1, 2, 3].map((i) => (
        <rect key={i} x={-215 + i * 72} y={-140 + i * 60} width={72} height={236 - i * 60}
          fill={P.card} stroke={P.bgLite} strokeWidth={3} />
      ))}
      <rect x={73} y={96} width={170} height={20} rx={8} fill={P.card} />
      {/* 扶手 */}
      <line x1={-250} y1={-215} x2={80} y2={-45} stroke={P.tendon} strokeWidth={10} strokeLinecap="round" />
      <line x1={-215} y1={-192} x2={-215} y2={-140} stroke={P.tendon} strokeWidth={8} strokeLinecap="round" />
      <line x1={-71} y1={-132} x2={-71} y2={-80} stroke={P.tendon} strokeWidth={8} strokeLinecap="round" />
      {/* 小人：僵硬下楼 */}
      <g transform={`translate(-45,${wob})`}>
        {/* 腿：一高一低踩台阶 */}
        <line x1={-4} y1={-66} x2={-60} y2={-14} stroke={P.sky} strokeWidth={15} strokeLinecap="round" />
        <line x1={-4} y1={-66} x2={40} y2={-8} stroke={P.sky} strokeWidth={15} strokeLinecap="round" />
        <rect x={-80} y={-20} width={34} height={14} rx={7} fill={P.cream} />
        <rect x={26} y={-14} width={34} height={14} rx={7} fill={P.cream} />
        {/* 躯干微前倾 */}
        <line x1={-8} y1={-140} x2={-2} y2={-64} stroke={P.muscle} strokeWidth={22} strokeLinecap="round" />
        {/* 手臂抓扶手 */}
        <line x1={-8} y1={-122} x2={58} y2={-92} stroke={P.muscle} strokeWidth={12} strokeLinecap="round" />
        <circle cx={60} cy={-91} r={8} fill={C.skin} />
        {/* 头 */}
        <circle cx={-12} cy={-164} r={24} fill={C.skin} />
        <path d="M -36 -170 Q -12 -194 12 -170 L 12 -160 Q -12 -180 -36 -160 Z" fill={P.ink} />
        {/* 龇牙表情 */}
        <path d="M -22 -164 q 4 5 8 0 M -4 -164 q 4 5 8 0" stroke={P.ink} strokeWidth={3}
          fill="none" strokeLinecap="round" />
        <path d="M -16 -150 l 14 0 l -7 7 Z" fill={P.ink} />
        {/* 大腿酸痛火花 */}
        <Spark x={-44} y={-40} s={0.55} />
        <Spark x={24} y={-38} s={0.45} />
      </g>
    </g>
  );
};

/* ---------- 哑铃 ---------- */
const Dumbbell = ({x, y, s = 1, rot = 0, col = P.cream}: {x: number; y: number; s?: number; rot?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`}>
    <rect x={-72} y={-9} width={144} height={18} rx={9} fill={col} />
    <rect x={-88} y={-32} width={20} height={64} rx={8} fill={col} />
    <rect x={-64} y={-42} width={18} height={84} rx={8} fill={col} />
    <rect x={68} y={-32} width={20} height={64} rx={8} fill={col} />
    <rect x={46} y={-42} width={18} height={84} rx={8} fill={col} />
  </g>
);

/* ---------- 乳酸分子（无辜脸） ---------- */
const Lactate = ({x, y, s = 1, op = 1}: {x: number; y: number; s?: number; op?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
    <line x1={-58} y1={10} x2={-26} y2={-8} stroke={P.cream} strokeWidth={7} strokeLinecap="round" />
    <line x1={26} y1={-8} x2={58} y2={10} stroke={P.cream} strokeWidth={7} strokeLinecap="round" />
    <line x1={0} y1={-38} x2={0} y2={-62} stroke={P.cream} strokeWidth={7} strokeLinecap="round" />
    <line x1={80} y1={2} x2={104} y2={-14} stroke={P.cream} strokeWidth={6} />
    <line x1={86} y1={16} x2={110} y2={0} stroke={P.cream} strokeWidth={6} />
    <circle cx={-70} cy={16} r={25} fill={P.muted} />
    <circle cx={0} cy={-12} r={27} fill={P.muted} />
    <circle cx={70} cy={16} r={25} fill={P.muted} />
    <circle cx={0} cy={-74} r={15} fill={P.alarm} />
    <circle cx={120} cy={-10} r={15} fill={P.alarm} />
    <circle cx={-70} cy={58} r={11} fill={P.cream} />
    {/* 无辜表情 */}
    <circle cx={-9} cy={-16} r={3.6} fill={P.ink} />
    <circle cx={9} cy={-16} r={3.6} fill={P.ink} />
    <path d="M -6 -2 q 6 5 12 0" stroke={P.ink} strokeWidth={3} fill="none" strokeLinecap="round" />
  </g>
);

/* ---------- 小钟 ---------- */
const MiniClock = ({x, y, s = 1, label = '', col = P.aqua}: {x: number; y: number; s?: number; label?: string; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <circle r={52} fill={P.cream} stroke={col} strokeWidth={6} />
    {Array.from({length: 12}).map((_, i) => (
      <line key={i} x1={0} y1={-44} x2={0} y2={-36} stroke={P.ink} strokeWidth={3.4}
        strokeLinecap="round" transform={`rotate(${i * 30})`} />
    ))}
    <line x1={0} y1={0} x2={0} y2={-26} stroke={P.ink} strokeWidth={6} strokeLinecap="round" />
    <line x1={0} y1={0} x2={18} y2={8} stroke={P.ink} strokeWidth={5} strokeLinecap="round" />
    <circle r={6} fill={P.ink} />
    {label && (
      <text x={0} y={86} textAnchor="middle" fontSize={26} fill={col}
        fontFamily={FONT_HEAD} fontWeight={900}>{label}</text>
    )}
  </g>
);

/* ---------- Z线盘 ---------- */
const ZDisc = ({x, h = 128, tilt = 0, col = P.tendon}: {x: number; h?: number; tilt?: number; col?: string}) => (
  <path d="M 0 -64 l 9 16 l -9 16 l 9 16 l -9 16 l 9 16 l -9 16 l 9 16 l -9 16"
    transform={`translate(${x},0) rotate(${tilt}) scale(1,${h / 128})`} stroke={col} strokeWidth={9}
    fill="none" strokeLinecap="round" strokeLinejoin="round" />
);

/* ---------- 肌节细肌丝/粗肌丝 ---------- */
const Filaments = ({x0, x1, fray = 0}: {x0: number; x1: number; fray?: number}) => (
  <g>
    {[-36, 0, 36].map((y, i) => (
      <g key={i}>
        <line x1={x0 + 8} y1={y} x2={x0 + 8 + (x1 - x0 - 16) * (fray ? 0.55 : 0.85)} y2={y}
          stroke={P.fascia} strokeWidth={5} strokeLinecap="round" />
        {fray > 0 && (
          <line x1={x0 + 8 + (x1 - x0 - 16) * 0.55} y1={y} x2={x0 + 8 + (x1 - x0 - 16) * 0.8} y2={y + (i - 1) * 10}
            stroke={P.fascia} strokeWidth={4} strokeDasharray="5 7" strokeLinecap="round" opacity={0.7} />
        )}
      </g>
    ))}
    {/* 粗肌丝排（中段） */}
    <g stroke={P.cream} strokeWidth={9} strokeLinecap="round">
      <line x1={(x0 + x1) / 2 - 34} y1={-18} x2={(x0 + x1) / 2 + 34} y2={-18} />
      <line x1={(x0 + x1) / 2 - 34} y1={18} x2={(x0 + x1) / 2 + 34} y2={18} />
    </g>
    {[-18, 18].map((y) =>
      [-22, 0, 22].map((dx, i) => (
        <circle key={`${y}${i}`} cx={(x0 + x1) / 2 + dx} cy={y} r={5.5} fill={P.track} stroke="none" />
      ))
    )}
  </g>
);

/* ---------- 巨噬细胞（戴安全帽的施工员） ---------- */
const Macro = ({x, y, s = 1, op = 1, eat = false}: {x: number; y: number; s?: number; op?: number; eat?: boolean}) => {
  const f = useCurrentFrame();
  const wig = Math.sin(f * 0.18 + x * 0.1) * 3;
  return (
    <g transform={`translate(${x},${y + wig}) scale(${s})`} opacity={op}>
      {[0, 1, 2, 3, 4, 5].map((i) => {
        const a = i * 60 * Math.PI / 180 + wig * 0.04;
        return <circle key={i} cx={Math.cos(a) * 40} cy={Math.sin(a) * 37} r={12} fill={P.vio} />;
      })}
      <circle r={38} fill={P.vio} />
      <circle cx={-12} cy={0} r={7} fill={P.bgDeep} />
      <circle cx={12} cy={0} r={7} fill={P.bgDeep} />
      <circle cx={-10} cy={-2} r={2.4} fill={P.cream} />
      <circle cx={14} cy={-2} r={2.4} fill={P.cream} />
      {/* 安全帽 */}
      <path d="M -30 -24 A 30 30 0 0 1 30 -24 Z" fill={P.lact} />
      <rect x={-38} y={-28} width={76} height={11} rx={5.5} fill={P.lactD} />
      {eat && <circle cx={30} cy={28} r={9} fill={P.muscleL} />}
    </g>
  );
};

/* ---------- 盾牌 ---------- */
const Shield = ({x, y, s = 1, col = P.aqua}: {x: number; y: number; s?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <path d="M 0 -34 L 30 -24 L 30 6 Q 30 30 0 40 Q -30 30 -30 6 L -30 -24 Z" fill={col} />
    <path d="M -12 2 L -3 12 L 14 -8" stroke={P.ink} strokeWidth={6} fill="none"
      strokeLinecap="round" strokeLinejoin="round" />
  </g>
);

/* ---------- 迷你肌节图标 ---------- */
const MiniSarco = ({x, y, s = 1, torn = false}: {x: number; y: number; s?: number; torn?: boolean}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-92} y={-30} width={184} height={60} rx={30} fill={P.muscleD} opacity={0.35} />
    <rect x={-92} y={-30} width={184} height={60} rx={30} fill="none" stroke={P.muscle} strokeWidth={4} />
    <line x1={-46} y1={-22} x2={-46} y2={22} stroke={P.tendon} strokeWidth={6} />
    {torn ? (
      <g>
        <line x1={56} y1={-22} x2={52} y2={22} stroke={P.tendon} strokeWidth={6} transform="rotate(14 54 0)" />
        <line x1={-40} y1={-10} x2={4} y2={-10} stroke={P.fascia} strokeWidth={4} strokeDasharray="6 6" />
        <line x1={-40} y1={10} x2={0} y2={10} stroke={P.fascia} strokeWidth={4} strokeDasharray="6 6" />
        <circle cx={16} cy={-14} r={4} fill={P.muscleL} />
        <circle cx={26} cy={6} r={3.4} fill={P.muscleL} />
        <circle cx={12} cy={16} r={3} fill={P.fascia} />
      </g>
    ) : (
      <g>
        <line x1={46} y1={-22} x2={46} y2={22} stroke={P.tendon} strokeWidth={6} />
        <line x1={-40} y1={-10} x2={40} y2={-10} stroke={P.fascia} strokeWidth={4} />
        <line x1={-40} y1={10} x2={40} y2={10} stroke={P.fascia} strokeWidth={4} />
        <line x1={-16} y1={0} x2={16} y2={0} stroke={P.cream} strokeWidth={7} strokeLinecap="round" />
      </g>
    )}
  </g>
);

/* ---------- 下蹲小人 ---------- */
const SquatGlyph = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <line x1={-42} y1={-74} x2={-42} y2={-38} stroke={P.lact} strokeWidth={6} strokeLinecap="round" />
    <polygon points="-42,-30 -49,-42 -35,-42" fill={P.lact} />
    <circle cx={-4} cy={-58} r={13} fill={C.skin} />
    <line x1={-2} y1={-45} x2={8} y2={-20} stroke={P.muscle} strokeWidth={12} strokeLinecap="round" />
    <path d="M 8 -20 L -14 -4 L 6 14" stroke={P.sky} strokeWidth={11} fill="none"
      strokeLinecap="round" strokeLinejoin="round" />
    <path d="M 8 -20 L 24 -6 L 32 14" stroke={P.sky} strokeWidth={11} fill="none"
      strokeLinecap="round" strokeLinejoin="round" />
    <line x1={-36} y1={16} x2={44} y2={16} stroke={P.card} strokeWidth={7} strokeLinecap="round" />
  </g>
);

/* ---------- 下坡跑小人 ---------- */
const RunGlyph = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <line x1={-50} y1={26} x2={52} y2={-4} stroke={P.card} strokeWidth={8} strokeLinecap="round" />
    <polygon points="52,-4 34,-12 38,4" fill={P.lact} />
    <circle cx={10} cy={-60} r={12} fill={C.skin} />
    <line x1={8} y1={-48} x2={0} y2={-22} stroke={P.muscle} strokeWidth={11} strokeLinecap="round" />
    <line x1={6} y1={-42} x2={-16} y2={-28} stroke={P.muscle} strokeWidth={8} strokeLinecap="round" />
    <line x1={0} y1={-22} x2={-22} y2={8} stroke={P.sky} strokeWidth={10} strokeLinecap="round" />
    <line x1={0} y1={-22} x2={26} y2={-2} stroke={P.sky} strokeWidth={10} strokeLinecap="round" />
  </g>
);

/* ---------- 哑铃慢放（手+哑铃+下降箭头） ---------- */
const LowerGlyph = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <line x1={-44} y1={-76} x2={-44} y2={-40} stroke={P.lact} strokeWidth={6} strokeLinecap="round" />
    <polygon points="-44,-32 -51,-44 -37,-44" fill={P.lact} />
    <line x1={0} y1={-70} x2={0} y2={-40} stroke={C.skin} strokeWidth={10} strokeLinecap="round" />
    <circle cx={0} cy={-34} r={10} fill={C.skin} />
    <Dumbbell x={0} y={2} s={0.42} />
    <line x1={-36} y1={20} x2={40} y2={20} stroke={P.card} strokeWidth={7} strokeLinecap="round" />
  </g>
);

/* ---------- 卧推凳 ---------- */
const BenchPress = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-95} y={30} width={190} height={16} rx={8} fill={P.card} />
    <rect x={-74} y={46} width={14} height={32} rx={6} fill={P.card} />
    <rect x={60} y={46} width={14} height={32} rx={6} fill={P.card} />
    <rect x={-115} y={-60} width={230} height={10} rx={5} fill={P.cream} />
    <rect x={-128} y={-78} width={16} height={46} rx={6} fill={P.cream} />
    <rect x={112} y={-78} width={16} height={46} rx={6} fill={P.cream} />
    <circle cx={-72} cy={14} r={16} fill={C.skin} />
    <rect x={-56} y={2} width={128} height={26} rx={13} fill={P.muscle} />
    <line x1={22} y1={0} x2={34} y2={-54} stroke={P.muscle} strokeWidth={10} strokeLinecap="round" />
  </g>
);

/* ---------- 攀岩墙 ---------- */
const ClimbWall = ({x, y, s = 1}: {x: number; y: number; s?: number}) => {
  const holds = [
    {hx: -62, hy: -52, c: P.lact}, {hx: 12, hy: -64, c: P.alarm}, {hx: 66, hy: -30, c: P.aqua},
    {hx: -24, hy: -12, c: P.sky}, {hx: 52, hy: 26, c: P.vio}, {hx: -58, hy: 40, c: P.track},
    {hx: 6, hy: 62, c: P.cream},
  ];
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      <rect x={-112} y={-88} width={224} height={176} rx={16} fill={P.card} />
      {holds.map((h, i) => (
        <ellipse key={i} cx={h.hx} cy={h.hy} rx={13} ry={9} fill={h.c}
          transform={`rotate(${i * 37} ${h.hx} ${h.hy})`} />
      ))}
      {/* 挂着的小人 */}
      <g>
        <line x1={-24} y1={-12} x2={-10} y2={-42} stroke={P.muscle} strokeWidth={10} strokeLinecap="round" />
        <circle cx={-12} cy={-44} r={6} fill={C.skin} />
        <circle cx={-30} cy={-2} r={13} fill={C.skin} />
        <line x1={-28} y1={8} x2={-24} y2={38} stroke={P.muscle} strokeWidth={13} strokeLinecap="round" />
        <line x1={-24} y1={38} x2={-46} y2={60} stroke={P.sky} strokeWidth={10} strokeLinecap="round" />
        <line x1={-24} y1={38} x2={-2} y2={58} stroke={P.sky} strokeWidth={10} strokeLinecap="round" />
      </g>
    </g>
  );
};

/* ---------- 散步小人 ---------- */
const WalkGlyph = ({x, y, s = 1, col = P.aqua}: {x: number; y: number; s?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <circle cx={2} cy={-46} r={11} fill={C.skin} />
    <line x1={0} y1={-34} x2={-2} y2={-10} stroke={col} strokeWidth={10} strokeLinecap="round" />
    <line x1={-2} y1={-28} x2={-16} y2={-14} stroke={col} strokeWidth={7} strokeLinecap="round" />
    <line x1={-2} y1={-28} x2={12} y2={-16} stroke={col} strokeWidth={7} strokeLinecap="round" />
    <line x1={-2} y1={-10} x2={-18} y2={14} stroke={P.sky} strokeWidth={9} strokeLinecap="round" />
    <line x1={-2} y1={-10} x2={14} y2={12} stroke={P.sky} strokeWidth={9} strokeLinecap="round" />
  </g>
);

/* ---------- 睡觉小人 ---------- */
const SleepGlyph = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-52} y={-6} width={104} height={18} rx={9} fill={P.card} />
    <rect x={-48} y={12} width={8} height={16} rx={4} fill={P.card} />
    <rect x={40} y={12} width={8} height={16} rx={4} fill={P.card} />
    <rect x={-46} y={-24} width={30} height={16} rx={7} fill={P.cream} />
    <circle cx={-26} cy={-26} r={11} fill={C.skin} />
    <path d="M -10 -10 Q 16 -30 44 -8 L 44 2 L -10 2 Z" fill={P.sky} />
    <text x={18} y={-34} fontSize={22} fill={P.aqua} fontFamily={FONT_HEAD} fontWeight={800}>z</text>
    <text x={34} y={-46} fontSize={16} fill={P.aqua} fontFamily={FONT_HEAD} fontWeight={800}>z</text>
  </g>
);

/* ---------- 蛋白质鸡腿 ---------- */
const ProteinGlyph = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <line x1={14} y1={8} x2={34} y2={26} stroke={P.cream} strokeWidth={8} strokeLinecap="round" />
    <circle cx={40} cy={30} r={7} fill={P.cream} />
    <circle cx={32} cy={38} r={7} fill={P.cream} />
    <path d="M -34 -20 Q -34 -44 -8 -44 Q 20 -44 24 -18 Q 27 6 6 14 Q -18 20 -30 4 Q -38 -6 -34 -20 Z"
      fill={P.muscle} />
    <circle cx={-10} cy={-16} r={6} fill={P.muscleL} opacity={0.8} />
  </g>
);

/* ---------- 冰块 ---------- */
const IceCube = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-26} y={-26} width={52} height={52} rx={12} fill={P.sky} opacity={0.85} />
    <path d="M -14 -14 L -2 -14 M -14 -2 L -6 -2" stroke={P.cream} strokeWidth={5} strokeLinecap="round" />
    {[0, 90, 180, 270].map((a) => (
      <line key={a} x1={0} y1={-38} x2={0} y2={-30} stroke={P.sky} strokeWidth={4}
        strokeLinecap="round" transform={`rotate(${a})`} />
    ))}
  </g>
);

/* ---------- 止痛胶囊 ---------- */
const PillCap = ({x, y, s = 1, rot = -25}: {x: number; y: number; s?: number; rot?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`}>
    <rect x={-34} y={-15} width={68} height={30} rx={15} fill={P.cream} />
    <rect x={0} y={-15} width={19} height={30} fill={P.alarm} />
    <circle cx={19} cy={0} r={15} fill={P.alarm} />
    <rect x={-2} y={-15} width={4} height={30} fill={P.ink} opacity={0.4} />
  </g>
);

/* ---------- 炎症介质六边形 ---------- */
const HexMed = ({x, y, s = 1, col = P.alarm}: {x: number; y: number; s?: number; col?: string}) => (
  <polygon points="16,0 8,14 -8,14 -16,0 -8,-14 8,-14"
    transform={`translate(${x},${y}) scale(${s})`} fill={col} />
);

/* ---------- 肱二头肌（结尾） ---------- */
const Bicep = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-96} y={8} width={124} height={62} rx={31} fill={C.skin} />
    <rect x={-32} y={-74} width={58} height={112} rx={29} fill={C.skin} />
    <circle cx={-4} cy={-84} r={27} fill={C.skin} />
    <circle cx={-52} cy={0} r={31} fill={C.skin} />
    <rect x={-96} y={42} width={124} height={28} rx={14} fill={C.skinDark} opacity={0.5} />
    <path d="M 52 -70 l 18 -18 M 62 -38 l 26 -10 M 60 -6 l 26 4" stroke={P.lact}
      strokeWidth={8} strokeLinecap="round" />
  </g>
);

/* ============ 标题卡 ============ */
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
        <g transform={`translate(960,330) scale(${pA}) translate(-960,-330)`} opacity={clamp01(pA)}>
          {/* 左：横着下楼的腿 */}
          <StairPerson x={330} y={330} s={0.92} />
          {/* 中：哑铃 + 肌节 */}
          <Dumbbell x={960} y={300} s={1.15} />
          <MiniSarco x={960} y={408} s={1} torn />
          <text x={960} y={478} textAnchor="middle" fontSize={23} fill={P.muted}
            fontFamily={FONT_HEAD}>练的是肌肉 · 伤的是肌节</text>
          {/* 右：+24h 日历钟 */}
          <g transform="translate(1510,300)">
            <rect x={-110} y={-96} width={220} height={200} rx={20} fill={P.panel}
              stroke={P.alarm} strokeWidth={5} />
            <rect x={-110} y={-96} width={220} height={52} rx={20} fill={P.alarm} />
            <text x={0} y={-60} textAnchor="middle" fontSize={30} fill={P.cream}
              fontFamily={FONT_HEAD} fontWeight={900}>第二天</text>
            <text x={0} y={16} textAnchor="middle" fontSize={58} fill={P.alarm}
              fontFamily={FONT_HEAD} fontWeight={900}>+24h</text>
            <text x={0} y={62} textAnchor="middle" fontSize={24} fill={P.muted}
              fontFamily={FONT_HEAD}>疼痛才上岗</text>
          </g>
          {/* 哑铃→日历 延迟箭头 */}
          <path d="M 1070 300 L 1360 300" stroke={P.muted} strokeWidth={5}
            strokeDasharray="12 10" />
          <polygon points="1372,300 1354,291 1354,309" fill={P.muted} />
        </g>
        {/* 两个小标签 */}
        <g transform={`translate(330,570) scale(${pB})`} opacity={clamp01(pB)}>
          <TagPill x={0} y={0} t="当天 · 只是酸" col={P.aqua} w={280} />
        </g>
        <g transform={`translate(1510,570) scale(${pB})`} opacity={clamp01(pB)}>
          <TagPill x={0} y={0} t="隔天 · 疼到扶墙" col={P.alarm} w={300} />
        </g>
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 440}}>
          <div style={{color: P.track, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 24}}>
            冷知识 · 第 {data.episode} 期
          </div>
          <div style={{color: C.text, fontSize: 74, fontWeight: 900, fontFamily: FONT_HEAD, letterSpacing: 4}}>
            {data.title}
          </div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ============ S0 乳酸冤案：分子离场 + 双时间线 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const popL = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const popR = Math.max(0.001, spring({frame: f - 32, fps, config: {damping: 11, stiffness: 120}}));
  const lactK = clamp01((f - 40) / 60);            // 乳酸曲线
  const clearK = clamp01((f - 120) / 30);          // 乳酸分子离场
  const painK = clamp01((f - b1 - 10) / 80);       // 疼痛曲线
  const verdictK = clamp01((f - b1 - 280) / 26);   // DOMS 揭示

  const lactPath = 'M 1140 305 C 1160 460 1210 505 1270 505 L 1720 505';
  const painPath = 'M 1140 505 L 1280 505 C 1340 502 1365 335 1455 315 C 1530 298 1590 315 1650 400 C 1680 440 1700 480 1712 498';

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={0} name={sec.name} accent={P.lact} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：嫌疑人乳酸 · 已离场 */}
        <g transform={`translate(120,150) scale(${popL}) translate(-120,-150)`} opacity={clamp01(popL * 2)}>
          <rect x={120} y={150} width={740} height={470} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={156} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>嫌疑人 · 乳酸</text>
          {/* 分子 + 离场 */}
          <g opacity={1 - clearK * 0.85} transform={`translate(${470 + clearK * 150},${365 - clearK * 30})`}>
            <Lactate x={0} y={0} s={1.15} />
          </g>
          {/* 离场箭头 */}
          {clearK > 0.1 && (
            <g opacity={clearK}>
              <path d="M 560 300 C 620 272 660 252 710 246" stroke={P.aqua} strokeWidth={5}
                strokeDasharray="10 9" fill="none" />
              <polygon points="722,244 702,238 706,254" fill={P.aqua} />
              <text x={628} y={226} textAnchor="middle" fontSize={24} fill={P.aqua}
                fontFamily={FONT_HEAD} fontWeight={900}>≈1h 清空离场</text>
            </g>
          )}
          {/* 去向：被当燃料烧掉 */}
          <g transform="translate(280,462)">
            <MiniClock x={0} y={0} s={0.72} col={P.aqua} label="运动后 +1h" />
            <text x={86} y={-8} fontSize={24} fill={P.cream} fontFamily={FONT_HEAD} fontWeight={800}>
              血液乳酸回基线</text>
            <text x={86} y={28} fontSize={21} fill={P.muted} fontFamily={FONT_HEAD}>
              被心脏和慢肌当燃料烧掉</text>
          </g>
          <text x={520} y={598} textAnchor="middle" fill={P.lact} fontSize={27}
            fontFamily={FONT_HEAD} fontWeight={900}>等不到第二天 · 背锅几十年</text>
        </g>
        {/* 右：两条时间线对不上 */}
        <g transform={`translate(1040,150) scale(${popR}) translate(-1040,-150)`} opacity={clamp01(popR * 2)}>
          <rect x={1040} y={150} width={740} height={470} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={1076} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            乳酸 vs 疼痛 · 时间轴对不上</text>
          {/* 坐标 */}
          <line x1={1120} y1={250} x2={1120} y2={505} stroke={P.muted} strokeWidth={4} />
          <polygon points="1120,238 1112,256 1128,256" fill={P.muted} />
          <line x1={1120} y1={505} x2={1730} y2={505} stroke={P.muted} strokeWidth={4} />
          <polygon points="1744,505 1728,497 1728,513" fill={P.muted} />
          {/* 时间刻度 */}
          {[['练完', 1140], ['+1h', 1270], ['24h', 1455], ['48h', 1580], ['72h', 1700]].map(([t, x]) => (
            <text key={t as string} x={x as number} y={540} textAnchor="middle" fontSize={21}
              fill={P.muted} fontFamily={FONT_HEAD}>{t}</text>
          ))}
          {/* 乳酸曲线 */}
          <path d={lactPath} fill="none" stroke={P.lact} strokeWidth={7} strokeLinecap="round"
            strokeDasharray="900" strokeDashoffset={900 * (1 - lactK)} />
          {lactK > 0.3 && (
            <text x={1190} y={288} fontSize={24} fill={P.lact} fontFamily={FONT_HEAD}
              fontWeight={900}>乳酸浓度</text>
          )}
          {/* 疼痛曲线 */}
          {painK > 0 && (
            <path d={painPath} fill="none" stroke={P.alarm} strokeWidth={7} strokeLinecap="round"
              strokeDasharray="1400" strokeDashoffset={1400 * (1 - painK)} />
          )}
          {painK > 0.6 && (
            <g>
              <circle cx={1520} cy={308} r={10} fill={P.alarm} />
              <rect x={1390} y={232} width={270} height={52} rx={14} fill={P.alarmD} />
              <text x={1525} y={267} textAnchor="middle" fontSize={25} fill={P.cream}
                fontFamily={FONT_HEAD} fontWeight={900}>疼痛峰 24–72h</text>
            </g>
          )}
          {painK > 0.75 && (
            <g>
              <MarkX x={1300} y={420} s={1.1} />
              <text x={1300} y={462} textAnchor="middle" fontSize={22} fill={P.alarm}
                fontFamily={FONT_HEAD} fontWeight={800}>对不上</text>
            </g>
          )}
          <text x={1425} y={590} textAnchor="middle" fill={P.cream} fontSize={24}
            fontFamily={FONT_HEAD} fontWeight={700}>乳酸早退了 · 疼却第二天才到</text>
        </g>
        {/* 底部：真凶揭示（cue1 尾） */}
        {verdictK > 0 && (
          <g opacity={verdictK} transform={`translate(0,${(1 - verdictK) * 60})`}>
            <rect x={120} y={648} width={1680} height={186} rx={20} fill={P.panel}
              stroke={P.alarm} strokeWidth={4} />
            <MiniSarco x={330} y={740} s={1.35} torn />
            <text x={520} y={712} fill={P.alarm} fontSize={34} fontFamily={FONT_HEAD} fontWeight={900}>
              真凶 = 延迟性肌肉酸痛 DOMS</text>
            <text x={520} y={768} fill={P.cream} fontSize={27} fontFamily={FONT_HEAD} fontWeight={700}>
              肌纤维微损伤 × 炎症反应 · 联手作案</text>
            <TagPill x={1560} y={740} t="24–72h 达峰" col={P.alarm} w={290} fs={28} />
          </g>
        )}
      </svg>
      <Chip text="乳酸 ≈1h 清空" x={1380} y={72} at={beat(0) + 25} accent={P.lact} />
      <Stamp text="冤案昭雪" x={620} y={88} at={b1 + 25} color={P.aquaD} size={38} rot={-3} />
    </FadeIn>
  );
};

/* ============ S1 肌节剖面：向心 vs 离心 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const popL = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const popR = Math.max(0.001, spring({frame: f - 28, fps, config: {damping: 11, stiffness: 120}}));
  const pullK = clamp01((f - 50) / 40);
  const tearK = clamp01((f - b1 - 5) / 30);         // 拉花
  const actPop = [0, 1, 2].map((i) => Math.max(0.001, spring({frame: f - 14 - i * 12, fps, config: {damping: 12, stiffness: 140}})));
  const bellK = clamp01((f - b1 - 120) / 24);       // 开工铃
  const pullPulse = f > 50 ? Math.abs(Math.sin((f - 50) * 0.09)) : 0;

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={1} name={sec.name} accent={P.muscle} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：向心收缩 · 安全 */}
        <g transform={`translate(120,150) scale(${popL}) translate(-120,-150)`} opacity={clamp01(popL * 2)}>
          <rect x={120} y={150} width={780} height={430} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={156} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>向心收缩 · 缩短发力</text>
          <TagPill x={760} y={198} t="安全" col={P.aqua} w={130} fs={24} />
          {/* 肌纤维 + 肌节 */}
          <g transform="translate(510,395)">
            <rect x={-310} y={-85} width={620} height={170} rx={85} fill={P.muscleD} opacity={0.3} />
            <rect x={-310} y={-85} width={620} height={170} rx={85} fill="none" stroke={P.muscle} strokeWidth={5} />
            <ZDisc x={-270} /><ZDisc x={-90} /><ZDisc x={90} /><ZDisc x={270} />
            <Filaments x0={-270} x1={-90} />
            <Filaments x0={-90} x1={90} />
            <Filaments x0={90} x1={270} />
            {/* 向内收敛箭头 */}
            {pullK > 0 && (
              <g opacity={pullK} stroke={P.aqua} strokeWidth={9} strokeLinecap="round">
                <line x1={-352} y1={0} x2={-300} y2={0} />
                <polygon points="-292,0 -308,-10 -308,10" fill={P.aqua} stroke="none" />
                <line x1={352} y1={0} x2={300} y2={0} />
                <polygon points="292,0 308,-10 308,10" fill={P.aqua} stroke="none" />
              </g>
            )}
          </g>
          <text x={510} y={540} textAnchor="middle" fill={P.aqua} fontSize={26}
            fontFamily={FONT_HEAD} fontWeight={900}>受力收敛 · 肌节排排站</text>
        </g>
        {/* 右：离心收缩 · 拉花 */}
        <g transform={`translate(1020,150) scale(${popR}) translate(-1020,-150)`} opacity={clamp01(popR * 2)}>
          <rect x={1020} y={150} width={780} height={430} rx={20} fill={P.panel}
            stroke={tearK > 0.4 ? P.alarm : P.card} strokeWidth={4} />
          <text x={1056} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>离心收缩 · 拉长发力</text>
          <TagPill x={1660} y={198} t="高危" col={P.alarm} w={130} fs={24} />
          <g transform="translate(1410,395)">
            <rect x={-310} y={-85} width={620} height={170} rx={85} fill={P.muscleD} opacity={0.3} />
            <rect x={-310} y={-85} width={620} height={170} rx={85} fill="none" stroke={P.muscle} strokeWidth={5} />
            <ZDisc x={-280} /><ZDisc x={-100} /><ZDisc x={60} tilt={tearK * 16} /><ZDisc x={280} />
            <Filaments x0={-280} x1={-100} />
            <Filaments x0={-100} x1={60} />
            <Filaments x0={60} x1={280} fray={tearK} />
            {/* 向外拉扯箭头（脉冲） */}
            {pullK > 0 && (
              <g opacity={pullK} stroke={P.alarm} strokeWidth={9} strokeLinecap="round">
                <line x1={-300 - pullPulse * 16} y1={0} x2={-356 - pullPulse * 16} y2={0} />
                <polygon points={`-364,${0} -348,-10 -348,10`} fill={P.alarm} stroke="none"
                  transform={`translate(${-pullPulse * 16},0)`} />
                <line x1={300 + pullPulse * 16} y1={0} x2={356 + pullPulse * 16} y2={0} />
                <polygon points="364,0 348,-10 348,10" fill={P.alarm} stroke="none"
                  transform={`translate(${pullPulse * 16},0)`} />
              </g>
            )}
            {/* 拉花现场 */}
            {tearK > 0 && (
              <g opacity={tearK}>
                <circle cx={150} cy={-30} r={6} fill={P.muscleL} />
                <circle cx={176} cy={8} r={5} fill={P.fascia} />
                <circle cx={140} cy={30} r={4.4} fill={P.muscleL} />
                <circle cx={196} cy={-14} r={4} fill={P.fascia} />
                <Spark x={168} y={0} s={0.8} />
              </g>
            )}
          </g>
          {tearK > 0.3 && (
            <g opacity={clamp01((tearK - 0.3) * 2)}>
              <rect x={1290} y={462} width={240} height={56} rx={14} fill={P.alarmD} />
              <text x={1410} y={500} textAnchor="middle" fontSize={27} fill={P.cream}
                fontFamily={FONT_HEAD} fontWeight={900}>肌节被拉花!</text>
            </g>
          )}
          <text x={1410} y={568} textAnchor="middle" fill={P.alarm} fontSize={25}
            fontFamily={FONT_HEAD} fontWeight={900}>Z线拉歪 · 膜破漏内容物</text>
        </g>
        {/* 底部：三个高危动作 + 开工铃 */}
        {[0, 1, 2].map((i) => (
          <g key={i} transform={`translate(${300 + i * 300},736) scale(${actPop[i]}) translate(${-(300 + i * 300)},-736)`}
            opacity={clamp01(actPop[i] * 2)}>
            <rect x={300 + i * 300 - 140} y={672} width={280} height={128} rx={18} fill={P.panel}
              stroke={P.muscle} strokeWidth={3.5} />
            {i === 0 && <SquatGlyph x={300 + i * 300} y={726} s={0.85} />}
            {i === 1 && <RunGlyph x={300 + i * 300} y={730} s={0.85} />}
            {i === 2 && <LowerGlyph x={300 + i * 300} y={742} s={0.85} />}
            <text x={300 + i * 300} y={786} textAnchor="middle" fontSize={23} fill={P.cream}
              fontFamily={FONT_HEAD} fontWeight={800}>
              {i === 0 ? '下蹲·下放段' : i === 1 ? '下坡跑' : '哑铃慢放'}</text>
          </g>
        ))}
        {bellK > 0 && (
          <g opacity={bellK} transform={`translate(0,${(1 - bellK) * 50})`}>
            <rect x={1120} y={672} width={680} height={128} rx={18} fill={P.panel} stroke={P.alarm} strokeWidth={4} />
            {/* 铃 */}
            <g transform="translate(1210,736)">
              <path d="M -26 8 A 26 26 0 0 1 26 8 L 30 22 L -30 22 Z" fill={P.lact} />
              <circle cy={30} r={7} fill={P.lactD} />
              <line x1={0} y1={-26} x2={0} y2={-18} stroke={P.lact} strokeWidth={6} strokeLinecap="round" />
            </g>
            <text x={1270} y={722} fill={P.cream} fontSize={26} fontFamily={FONT_HEAD} fontWeight={800}>
              当时不疼 ≠ 没事</text>
            <text x={1270} y={766} fill={P.alarm} fontSize={26} fontFamily={FONT_HEAD} fontWeight={900}>
              炎症开工铃 · 已按下</text>
          </g>
        )}
      </svg>
      <Chip text="拉长发力最伤" x={1420} y={72} at={beat(0) + 25} accent={P.muscle} />
      <Stamp text="肌节 popping" x={620} y={90} at={b1 + 30} color={P.alarmD} size={36} rot={-3} />
    </FadeIn>
  );
};

/* ============ S2 炎症施工队：为什么迟到 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);
  const popL = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const popR = Math.max(0.001, spring({frame: f - 30, fps, config: {damping: 11, stiffness: 120}}));
  const crewPop = [0, 1, 2].map((i) => Math.max(0.001, spring({frame: f - 40 - i * 18, fps, config: {damping: 12, stiffness: 140}})));
  const medK = clamp01((f - 80) / 30);
  const swellK = clamp01((f - 105) / 30);
  const painK = clamp01((f - b1 - 5) / 90);
  const trackK = clamp01((f - b2) / 30);

  const painPath = 'M 1150 505 L 1240 505 C 1300 500 1330 330 1420 312 C 1490 298 1560 315 1620 410 C 1650 458 1680 495 1700 502';

  return (
    <FadeIn>
      <Backdrop top={P.bgDeep} bottom={P.bgDeep} />
      <SceneTag index={2} name={sec.name} accent={P.alarm} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：施工现场 */}
        <g transform={`translate(120,150) scale(${popL}) translate(-120,-150)`} opacity={clamp01(popL * 2)}>
          <rect x={120} y={150} width={760} height={470} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={156} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>损伤现场 · 施工队进场</text>
          {/* 断裂肌纤维 */}
          <path d="M 165 340 L 400 340 L 430 366 L 404 392 L 434 416 L 410 440 L 165 440 Q 140 390 165 340 Z"
            fill={P.muscleD} />
          <path d="M 505 340 L 735 340 Q 760 390 735 440 L 508 440 L 480 416 L 506 392 L 478 366 Z"
            fill={P.muscleD} />
          {[200, 250, 300, 350].map((x) => (
            <line key={x} x1={x} y1={352} x2={x} y2={428} stroke={P.muscle} strokeWidth={4} opacity={0.6} />
          ))}
          {[560, 610, 660, 710].map((x) => (
            <line key={x} x1={x} y1={352} x2={x} y2={428} stroke={P.muscle} strokeWidth={4} opacity={0.6} />
          ))}
          {/* 漏出的碎片 */}
          {[0, 1, 2, 3, 4].map((i) => (
            <circle key={i} cx={438 + i * 14} cy={462 + (i % 2) * 14} r={5}
              fill={i % 2 ? P.fascia : P.muscleL} />
          ))}
          {/* 破口火花 */}
          <Spark x={452} y={388} s={0.6} />
          <text x={468} y={318} textAnchor="middle" fontSize={21} fill={P.muscleL}
            fontFamily={FONT_HEAD} fontWeight={800}>肌纤维破口</text>
          {/* 巨噬细胞施工员：两名行军 + 一名已上纤维开吃 */}
          {[0, 1].map((i) => (
            <g key={i} transform={`translate(${215 + i * 125},${295 + (i % 2) * 6}) scale(${crewPop[i]})`}
              opacity={clamp01(crewPop[i] * 2)}>
              <Macro x={0} y={0} s={0.72} />
            </g>
          ))}
          <g transform={`translate(600,330) scale(${crewPop[2]})`} opacity={clamp01(crewPop[2] * 2)}>
            <Macro x={0} y={0} s={0.66} eat />
          </g>
          {crewPop[1] > 0.6 && (
            <text x={275} y={244} textAnchor="middle" fontSize={22} fill={P.vio}
              fontFamily={FONT_HEAD} fontWeight={800}>巨噬细胞行军进场</text>
          )}
          {/* 炎症介质：破口下方渗出 */}
          {medK > 0 && (
            <g opacity={medK}>
              <HexMed x={450} y={460} s={1} />
              <HexMed x={478} y={472} s={0.85} />
              <HexMed x={426} y={474} s={0.7} />
              <text x={615} y={474} fontSize={21} fill={P.alarm} fontFamily={FONT_HEAD}
                fontWeight={800}>炎症介质渗出</text>
            </g>
          )}
          {/* 水肿压神经 */}
          <path d="M 165 545 q 30 -18 60 0 t 60 0 t 60 0 t 60 0 t 60 0 t 60 0 t 60 0 t 60 0"
            stroke={P.nerve} strokeWidth={7} fill="none" strokeLinecap="round" />
          <circle cx={165} cy={545} r={8} fill={P.nerve} />
          <circle cx={765} cy={545} r={8} fill={P.nerve} />
          {swellK > 0 && (
            <g opacity={swellK}>
              <ellipse cx={350} cy={514} rx={58} ry={30} fill={P.sky} opacity={0.5} />
              <text x={350} y={522} textAnchor="middle" fontSize={20} fill={P.cream}
                fontFamily={FONT_HEAD} fontWeight={800}>水肿</text>
              <Spark x={350} y={542} s={0.55} />
              <text x={640} y={518} fontSize={21} fill={P.alarm} fontFamily={FONT_HEAD}
                fontWeight={800}>肿胀挤神经 → 疼</text>
            </g>
          )}
          <text x={165} y={596} fill={P.muted} fontSize={22}
            fontFamily={FONT_HEAD}>痛觉末梢在纤维外围</text>
          <text x={855} y={596} textAnchor="end" fill={P.muted} fontSize={22}
            fontFamily={FONT_HEAD}>介质：组胺 · 缓激肽 · 前列腺素</text>
        </g>
        {/* 右：疼痛迟到曲线 */}
        <g transform={`translate(1020,150) scale(${popR}) translate(-1020,-150)`} opacity={clamp01(popR * 2)}>
          <rect x={1020} y={150} width={780} height={470} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={1056} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            疼痛曲线 · 跟着工期走</text>
          <line x1={1130} y1={250} x2={1130} y2={505} stroke={P.muted} strokeWidth={4} />
          <polygon points="1130,238 1122,256 1138,256" fill={P.muted} />
          <line x1={1130} y1={505} x2={1730} y2={505} stroke={P.muted} strokeWidth={4} />
          <polygon points="1744,505 1728,497 1728,513" fill={P.muted} />
          {[['当天', 1150], ['第2天', 1360], ['第3天', 1540], ['第5–7天', 1680]].map(([t, x]) => (
            <text key={t as string} x={x as number} y={540} textAnchor="middle" fontSize={21}
              fill={P.muted} fontFamily={FONT_HEAD}>{t}</text>
          ))}
          {/* 伤已发生标记 */}
          <line x1={1150} y1={290} x2={1150} y2={505} stroke={P.muscleL} strokeWidth={4}
            strokeDasharray="8 8" />
          <text x={1150} y={278} textAnchor="middle" fontSize={21} fill={P.muscleL}
            fontFamily={FONT_HEAD} fontWeight={800}>伤·已发生</text>
          {/* 施工队到岗：小安全帽沿路走 */}
          {painK > 0 && [0, 1, 2].map((i) => {
            const hx = 1250 + i * 110, hy = 470 - i * 62;
            const hk = clamp01(painK * 1.4 - i * 0.25);
            if (hk <= 0) return null;
            return (
              <g key={i} transform={`translate(${hx},${hy})`} opacity={hk}>
                <path d="M -16 0 A 16 16 0 0 1 16 0 Z" fill={P.lact} />
                <rect x={-21} y={-3} width={42} height={7} rx={3.5} fill={P.lactD} />
              </g>
            );
          })}
          {painK > 0 && (
            <path d={painPath} fill="none" stroke={P.alarm} strokeWidth={7} strokeLinecap="round"
              strokeDasharray="1400" strokeDashoffset={1400 * (1 - painK)} />
          )}
          {painK > 0.7 && (
            <g opacity={clamp01((painK - 0.7) * 3)}>
              <circle cx={1490} cy={306} r={10} fill={P.alarm} />
              <rect x={1360} y={230} width={260} height={52} rx={14} fill={P.alarmD} />
              <text x={1490} y={265} textAnchor="middle" fontSize={25} fill={P.cream}
                fontFamily={FONT_HEAD} fontWeight={900}>第二天最疼</text>
            </g>
          )}
          {painK > 0.95 && (
            <text x={1660} y={480} textAnchor="middle" fontSize={21} fill={P.aqua}
              fontFamily={FONT_HEAD} fontWeight={800}>消退</text>
          )}
          <text x={1430} y={590} textAnchor="middle" fill={P.cream} fontSize={24}
            fontFamily={FONT_HEAD} fontWeight={700}>行军要 12–48h · 疼自然迟到一天</text>
        </g>
        {/* 底部：伤≠疼 双轨 */}
        {trackK > 0 && (
          <g opacity={trackK} transform={`translate(0,${(1 - trackK) * 60})`}>
            <rect x={120} y={655} width={1680} height={180} rx={20} fill={P.panel} stroke={P.alarm} strokeWidth={4} />
            <TagPill x={230} y={715} t="伤" col={P.muscleL} w={110} fs={30} />
            <TagPill x={230} y={795} t="疼" col={P.alarm} w={110} fs={30} />
            <line x1={320} y1={715} x2={1700} y2={715} stroke={P.card} strokeWidth={5} />
            <line x1={320} y1={795} x2={1700} y2={795} stroke={P.card} strokeWidth={5} />
            <circle cx={400} cy={715} r={13} fill={P.muscleL} />
            <text x={430} y={700} fontSize={22} fill={P.muscleL} fontFamily={FONT_HEAD}
              fontWeight={800}>破口当天就有</text>
            {[700, 960, 1220].map((x, i) => (
              <circle key={i} cx={x} cy={795} r={13 - i * 2} fill={P.alarm} opacity={1 - i * 0.22} />
            ))}
            <text x={700} y={770} fontSize={22} fill={P.alarm} fontFamily={FONT_HEAD}
              fontWeight={800}>疼峰迟到 1–2 天</text>
            {/* 错位括号 */}
            <path d="M 400 735 L 400 755 M 400 755 L 700 755 M 700 755 L 700 775"
              stroke={P.lact} strokeWidth={5} fill="none" />
            <text x={530} y={782} textAnchor="middle" fontSize={24} fill={P.lact}
              fontFamily={FONT_HEAD} fontWeight={900}>≈24h 错位</text>
            <text x={1420} y={755} fontSize={26} fill={P.cream} fontFamily={FONT_HEAD}
              fontWeight={800}>疼不疼和伤没伤 · 是两条时间线</text>
          </g>
        )}
      </svg>
      <Chip text="疼峰 24–72h" x={1420} y={72} at={beat(0) + 25} accent={P.alarm} />
      <Stamp text="伤 ≠ 疼" x={620} y={90} at={b2 + 15} color={P.alarmD} size={38} rot={-3} />
    </FadeIn>
  );
};

/* ============ S3 重复回合效应：专挑没练过的 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);
  const popL = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const popR = Math.max(0.001, spring({frame: f - 34, fps, config: {damping: 11, stiffness: 120}}));
  const bar1 = clamp01((f - 45) / 40);
  const bar2 = clamp01((f - 75) / 40);
  const shieldK = clamp01((f - 85) / 25);
  const bannerK = clamp01((f - b1 - 10) / 26);
  const climbK = clamp01((f - b2) / 30);

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={3} name={sec.name} accent={P.aqua} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：同一动作两次对照 */}
        <g transform={`translate(120,150) scale(${popL}) translate(-120,-150)`} opacity={clamp01(popL * 2)}>
          <rect x={120} y={150} width={740} height={470} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={156} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            同一个动作 · 两次对照</text>
          <TagPill x={700} y={198} t="怪脾气" col={P.aqua} w={140} fs={24} />
          {/* 第1次 */}
          <g transform="translate(340,0)">
            <MiniSarco x={0} y={290} s={0.9} torn />
            <rect x={-46} y={540 - 200 * bar1} width={92} height={200 * bar1} rx={14} fill={P.alarm} />
            {bar1 > 0.9 && <Spark x={0} y={330} s={0.5} />}
            <text x={0} y={580} textAnchor="middle" fontSize={24} fill={P.cream}
              fontFamily={FONT_HEAD} fontWeight={900}>第 1 次</text>
            <text x={0} y={612} textAnchor="middle" fontSize={20} fill={P.alarm}
              fontFamily={FONT_HEAD} fontWeight={800}>疼 3–4 天</text>
          </g>
          {/* 第2次 */}
          <g transform="translate(640,0)">
            <MiniSarco x={0} y={290} s={0.9} />
            {shieldK > 0 && (
              <g opacity={shieldK}>
                <Shield x={104} y={322} s={0.72} />
                <text x={0} y={240} textAnchor="middle" fontSize={19} fill={P.aqua}
                  fontFamily={FONT_HEAD} fontWeight={800}>肌节已加固</text>
              </g>
            )}
            <rect x={-46} y={540 - 44 * bar2} width={92} height={Math.max(1, 44 * bar2)} rx={14} fill={P.aqua} />
            <text x={0} y={580} textAnchor="middle" fontSize={24} fill={P.cream}
              fontFamily={FONT_HEAD} fontWeight={900}>第 2 次</text>
            <text x={0} y={612} textAnchor="middle" fontSize={20} fill={P.aqua}
              fontFamily={FONT_HEAD} fontWeight={800}>几乎不疼</text>
          </g>
          {/* 疼痛轴 */}
          <line x1={200} y1={540} x2={760} y2={540} stroke={P.muted} strokeWidth={3} />
          <text x={206} y={330} fontSize={19} fill={P.muted} fontFamily={FONT_HEAD}
            transform="rotate(-90 206 330)">疼痛程度</text>
          {/* 保护期 */}
          {shieldK > 0.4 && (
            <g opacity={shieldK}>
              <rect x={200} y={212} width={300} height={48} rx={14} fill={P.bgDeep}
                stroke={P.aqua} strokeWidth={3.5} />
              <text x={350} y={244} textAnchor="middle" fontSize={23} fill={P.aqua}
                fontFamily={FONT_HEAD} fontWeight={900}>保护期 数周~数月</text>
            </g>
          )}
        </g>
        {/* 右：保护只认这个动作 */}
        <g transform={`translate(1000,150) scale(${popR}) translate(-1000,-150)`} opacity={clamp01(popR * 2)}>
          <rect x={1000} y={150} width={800} height={470} rx={20} fill={P.panel}
            stroke={climbK > 0.4 ? P.alarm : P.card} strokeWidth={4} />
          <text x={1036} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            但保护是动作特异的</text>
          {/* 卧推：免检 */}
          <BenchPress x={1210} y={420} s={1.15} />
          {popR > 0.5 && <Shield x={1330} y={330} s={0.9} />}
          <text x={1210} y={540} textAnchor="middle" fontSize={23} fill={P.aqua}
            fontFamily={FONT_HEAD} fontWeight={900}>练熟的动作 · 免检</text>
          <text x={1210} y={575} textAnchor="middle" fontSize={20} fill={P.muted}
            fontFamily={FONT_HEAD}>卧推十年也一样</text>
          {/* 箭头 */}
          <path d="M 1340 400 L 1440 400" stroke={P.muted} strokeWidth={5} strokeDasharray="10 9" />
          <polygon points="1452,400 1436,392 1436,408" fill={P.muted} />
          <text x={1396} y={378} textAnchor="middle" fontSize={20} fill={P.muted}
            fontFamily={FONT_HEAD}>换动作</text>
          {/* 攀岩：照疼 */}
          <g transform={`translate(1620,400)`}>
            <ClimbWall x={0} y={0} s={0.92} />
            {climbK > 0 && (
              <g opacity={climbK}>
                <Spark x={-30} y={-40} s={0.7} />
                <MarkX x={76} y={-70} s={0.9} />
              </g>
            )}
          </g>
          <text x={1620} y={540} textAnchor="middle" fontSize={23} fill={P.alarm}
            fontFamily={FONT_HEAD} fontWeight={900}>攀岩新手 · 前臂照疼</text>
          <text x={1620} y={575} textAnchor="middle" fontSize={20} fill={P.muted}
            fontFamily={FONT_HEAD}>保护不跨动作转账</text>
          {/* cue2 高亮 */}
          {climbK > 0.3 && (
            <g opacity={clamp01((climbK - 0.3) * 2)}>
              <rect x={1050} y={246} width={290} height={52} rx={14} fill={P.alarmD} />
              <text x={1195} y={281} textAnchor="middle" fontSize={25} fill={P.cream}
                fontFamily={FONT_HEAD} fontWeight={900}>换个新动作照样中招</text>
            </g>
          )}
        </g>
        {/* 底部：核心句 */}
        {bannerK > 0 && (
          <g opacity={bannerK} transform={`translate(0,${(1 - bannerK) * 60})`}>
            <rect x={120} y={655} width={1680} height={180} rx={20} fill={P.panel} stroke={P.aqua} strokeWidth={4} />
            <Shield x={260} y={745} s={1.3} />
            <text x={360} y={722} fill={P.aqua} fontSize={33} fontFamily={FONT_HEAD} fontWeight={900}>
              疼的不是「练了」· 是「练了没练过的」</text>
            <text x={360} y={782} fill={P.cream} fontSize={27} fontFamily={FONT_HEAD} fontWeight={700}>
              修复把脆弱肌节局部加固 · 第二次就撕不动了 —— 这叫重复回合效应</text>
            <TagPill x={1620} y={745} t="重复回合效应" col={P.aqua} w={280} fs={27} />
          </g>
        )}
      </svg>
      <Chip text="第二次就不疼" x={1410} y={72} at={beat(0) + 25} accent={P.aqua} />
      <Stamp text="适应 · 不是变强" x={620} y={90} at={b2 + 20} color={P.aquaD} size={36} rot={-3} />
    </FadeIn>
  );
};

/* ============ S4 治不了但能防 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);
  const popL = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const popR = Math.max(0.001, spring({frame: f - 26, fps, config: {damping: 11, stiffness: 120}}));
  const rowPop = [0, 1, 2].map((i) => Math.max(0.001, spring({frame: f - 30 - i * 15, fps, config: {damping: 12, stiffness: 140}})));
  const warnK = clamp01((f - b1) / 26);
  const rampK = clamp01((f - b2) / 30);
  const stitch = Math.sin(f * 0.3) * 4;

  const remedies = [
    {icon: 'walk', col: P.aqua, t: '散步慢游', sub: '血流带走炎症介质'},
    {icon: 'sleep', col: P.sky, t: '睡足觉', sub: '生长激素是修复主力'},
    {icon: 'protein', col: P.lact, t: '蛋白质跟上', sub: '修复要原料'},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={4} name={sec.name} accent={P.track} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：缓解三件套 + 警示 */}
        <g transform={`translate(120,150) scale(${popL}) translate(-120,-150)`} opacity={clamp01(popL * 2)}>
          <rect x={120} y={150} width={800} height={470} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={156} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            只能缓解 · 创造条件</text>
          <rect x={690} y={172} width={200} height={48} rx={14} fill={P.bgDeep} stroke={P.alarm} strokeWidth={3.5} />
          <text x={790} y={204} textAnchor="middle" fontSize={24} fill={P.alarm}
            fontFamily={FONT_HEAD} fontWeight={900}>无特效药</text>
          {remedies.map((r, i) => (
            <g key={i} transform={`translate(150,${246 + i * 82}) scale(${rowPop[i]}) translate(-150,${-(246 + i * 82)})`}
              opacity={clamp01(rowPop[i] * 2)}>
              <rect x={150} y={232 + i * 82} width={740} height={70} rx={16} fill={P.bgDeep}
                stroke={r.col} strokeWidth={3} opacity={0.9} />
              <circle cx={200} cy={267 + i * 82} r={26} fill={P.bg} stroke={r.col} strokeWidth={3.5} />
              {r.icon === 'walk' && <WalkGlyph x={200} y={280 + i * 82} s={0.62} />}
              {r.icon === 'sleep' && <SleepGlyph x={200} y={262 + i * 82} s={0.62} />}
              {r.icon === 'protein' && <ProteinGlyph x={200} y={268 + i * 82} s={0.6} />}
              <text x={250} y={277 + i * 82} fontSize={27} fill={r.col}
                fontFamily={FONT_HEAD} fontWeight={900}>{r.t}</text>
              <text x={520} y={277 + i * 82} fontSize={23} fill={P.cream}
                fontFamily={FONT_HEAD}>{r.sub}</text>
            </g>
          ))}
          {/* 冰敷止痛片：压疼不治本 */}
          {warnK > 0 && (
            <g opacity={warnK}>
              <rect x={150} y={500} width={740} height={96} rx={16} fill={P.bgDeep}
                stroke={P.alarm} strokeWidth={3.5} strokeDasharray="10 8" />
              <IceCube x={205} y={548} s={0.8} />
              <PillCap x={285} y={548} s={0.85} />
              <text x={350} y={540} fontSize={25} fill={P.cream} fontFamily={FONT_HEAD} fontWeight={800}>
                冰敷 · 止痛药</text>
              <text x={350} y={576} fontSize={23} fill={P.alarm} fontFamily={FONT_HEAD} fontWeight={900}>
                压得住疼 · 却可能拖慢修复</text>
            </g>
          )}
        </g>
        {/* 右：修复车间 */}
        <g transform={`translate(960,150) scale(${popR}) translate(-960,-150)`} opacity={clamp01(popR * 2)}>
          <rect x={960} y={150} width={840} height={470} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={996} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            疼 = 正在修复的组织</text>
          {/* 肌纤维 + 补丁段 */}
          <g transform="translate(1380,400)">
            <rect x={-330} y={-70} width={660} height={140} rx={70} fill={P.muscleD} opacity={0.35} />
            <rect x={-330} y={-70} width={660} height={140} rx={70} fill="none" stroke={P.muscle} strokeWidth={5} />
            {[-250, -140, 140, 250].map((x) => (
              <line key={x} x1={x} y1={-52} x2={x} y2={52} stroke={P.muscle} strokeWidth={4} opacity={0.6} />
            ))}
            {/* 补丁段 */}
            <rect x={-80} y={-70} width={170} height={140} fill={P.muscleD} />
            {[-80, 90].map((x) => (
              <line key={x} x1={x} y1={-70} x2={x} y2={70} stroke={P.lact} strokeWidth={5}
                strokeDasharray="10 8" />
            ))}
            <g transform={`translate(${-36 + stitch},-30)`} stroke={P.tendon} strokeWidth={6} strokeLinecap="round">
              <line x1={0} y1={0} x2={20} y2={20} /><line x1={20} y1={0} x2={0} y2={20} />
            </g>
            <g transform={`translate(${30 - stitch},10)`} stroke={P.tendon} strokeWidth={6} strokeLinecap="round">
              <line x1={0} y1={0} x2={20} y2={20} /><line x1={20} y1={0} x2={0} y2={20} />
            </g>
            <Macro x={-150} y={-110} s={0.6} />
            <Macro x={150} y={-105} s={0.55} />
            <text x={5} y={-92} textAnchor="middle" fontSize={21} fill={P.lact}
              fontFamily={FONT_HEAD} fontWeight={800}>修补加固中</text>
          </g>
          <text x={1380} y={540} textAnchor="middle" fill={P.cream} fontSize={27}
            fontFamily={FONT_HEAD} fontWeight={900}>练只是下单 · 修才是交付</text>
          <text x={1380} y={582} textAnchor="middle" fill={P.muted} fontSize={22}
            fontFamily={FONT_HEAD}>修好之后的肌节 · 比原来更能扛</text>
        </g>
        {/* 底部：循序渐进台阶 */}
        {rampK > 0 && (
          <g opacity={rampK} transform={`translate(0,${(1 - rampK) * 60})`}>
            <rect x={120} y={655} width={1680} height={180} rx={20} fill={P.panel} stroke={P.track} strokeWidth={4} />
            <text x={165} y={722} fill={P.track} fontSize={31} fontFamily={FONT_HEAD} fontWeight={900}>
              最好的策略 = 预防</text>
            <text x={165} y={776} fill={P.cream} fontSize={25} fontFamily={FONT_HEAD} fontWeight={700}>
              适应赤字越小 · 疼越轻</text>
            {/* 三级台阶 */}
            <g transform="translate(700,0)">
              <rect x={0} y={775} width={280} height={60} rx={12} fill={P.card} />
              <rect x={300} y={715} width={280} height={120} rx={12} fill={P.card} />
              <rect x={600} y={655} width={280} height={180} rx={12} fill={P.track} opacity={0.9} />
              <text x={140} y={812} textAnchor="middle" fontSize={24} fill={P.cream}
                fontFamily={FONT_HEAD} fontWeight={900}>新动作·小量</text>
              <text x={440} y={752} textAnchor="middle" fontSize={24} fill={P.cream}
                fontFamily={FONT_HEAD} fontWeight={900}>复训·先减半</text>
              <text x={740} y={700} textAnchor="middle" fontSize={24} fill={P.ink}
                fontFamily={FONT_HEAD} fontWeight={900}>再往上加</text>
              {/* 爬台阶的人 */}
              <WalkGlyph x={470} y={700} s={0.9} />
              <path d="M 520 668 l 62 -30" stroke={P.cream} strokeWidth={6} strokeLinecap="round"
                strokeDasharray="10 8" />
              <polygon points="588,634 566,642 582,654" fill={P.cream} />
            </g>
          </g>
        )}
      </svg>
      <Chip text="防 > 治" x={1460} y={72} at={beat(0) + 25} accent={P.track} />
      <Stamp text="循序渐进不是鸡汤" x={600} y={90} at={b2 + 20} color={P.track} size={36} rot={-3} />
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
        <g transform={`translate(960,800) scale(${pop})`} opacity={clamp01(pop)}>
          {/* 修好的肌纤维 */}
          <rect x={-330} y={-60} width={660} height={120} rx={60} fill={P.muscleD} opacity={0.35} />
          <rect x={-330} y={-60} width={660} height={120} rx={60} fill="none" stroke={P.muscle} strokeWidth={5} />
          <ZDisc x={-220} h={90} /><ZDisc x={-70} h={90} /><ZDisc x={80} h={90} /><ZDisc x={230} h={90} />
          <Shield x={330} y={-80} s={1.1} />
        </g>
        <g transform={`translate(1490,790) scale(${pop})`} opacity={clamp01(pop)}>
          <Bicep x={0} y={0} s={1.15} />
        </g>
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -300}}>
        <div style={{fontSize: 54, fontWeight: 800, color: C.text}}>疼是施工的噪音 · 修完更结实</div>
        <div style={{fontSize: 36, color: P.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E242 = ({data = tl}: {data?: any}) => {
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
