import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01} from '../lib/style';
import {Backdrop, Stamp, SceneTag, Chip, Subtitles, ProgressBar, FadeIn} from '../lib/atoms';
import tl from '../timeline_249.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：深池蓝 × 泳帽黄 × 氧气青 × 泳道红 ============ */
const P = {
  bg: '#0e3d5c',
  bgLite: '#124a6e',
  bgDeep: '#082a42',
  panel: '#12466a',
  card: '#1e6a96',
  ink: '#062335',
  cream: '#fff3de',
  muted: '#a9c9dc',
  water: '#4fc3f7',
  waterD: '#2b8fc4',
  air: '#7ee0d2',
  airD: '#45b3a4',
  cap: '#ffd166',
  capD: '#e0a83c',
  lane: '#ff6b6b',
  red: '#ff5d73',
  lime: '#8fe06a',
  co2: '#ff8fa3',
  steel: '#cfd8e2',
};

/* ---------- 小叉 / 小勾 ---------- */
const MarkX = ({x, y, s = 1, col = P.red}: {x: number; y: number; s?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} stroke={col} strokeWidth={7} strokeLinecap="round">
    <line x1={-13} y1={-13} x2={13} y2={13} /><line x1={13} y1={-13} x2={-13} y2={13} />
  </g>
);
const Check = ({x, y, s = 1, col = P.lime}: {x: number; y: number; s?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} stroke={col} strokeWidth={9}
    strokeLinecap="round" strokeLinejoin="round" fill="none">
    <path d="M -16 1 L -5 13 L 18 -13" />
  </g>
);

/* ---------- 圆角标签牌 ---------- */
const TagPill = ({x, y, t, col, s = 1, w = 200, fs = 30}: {
  x: number; y: number; t: string; col: string; s?: number; w?: number; fs?: number;
}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-w / 2} y={-30} width={w} height={60} rx={16} fill={P.panel} stroke={col} strokeWidth={4} />
    <text x={0} y={11} textAnchor="middle" fontSize={fs} fill={col}
      fontFamily={FONT_HEAD} fontWeight={800}>{t}</text>
  </g>
);

/* ---------- 指向圆心的压力箭头 ---------- */
const InArrow = ({cx, cy, a, rx1, ry1, rx2, ry2, col = P.cap, sw = 8}: {
  cx: number; cy: number; a: number; rx1: number; ry1: number; rx2: number; ry2: number;
  col?: string; sw?: number;
}) => {
  const rad = (a * Math.PI) / 180;
  const x1 = cx + Math.cos(rad) * rx1, y1 = cy + Math.sin(rad) * ry1;
  const x2 = cx + Math.cos(rad) * rx2, y2 = cy + Math.sin(rad) * ry2;
  return (
    <g>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={col} strokeWidth={sw} strokeLinecap="round" />
      <g transform={`translate(${x2},${y2}) rotate(${a + 180})`}>
        <polygon points="10,0 -6,-9 -6,9" fill={col} />
      </g>
    </g>
  );
};

/* ---------- 右向箭头 ---------- */
const Chev = ({x, y, col = P.muted}: {x: number; y: number; col?: string}) => (
  <g transform={`translate(${x},${y})`} stroke={col} strokeWidth={7} fill="none"
    strokeLinecap="round" strokeLinejoin="round">
    <path d="M -8 -14 L 8 0 L -8 14" />
  </g>
);

/* ---------- 水面波浪线 ---------- */
const Wave = ({x, y, w, col = P.water, sw = 6, amp = 9, seg = 70}: {
  x: number; y: number; w: number; col?: string; sw?: number; amp?: number; seg?: number;
}) => {
  const f = useCurrentFrame();
  const bob = Math.sin(f * 0.1) * 3;
  const n = Math.round(w / seg);
  const pts: string[] = [`M ${x} ${y + bob}`];
  for (let i = 0; i < n; i++) {
    const x0 = x + i * seg;
    pts.push(`Q ${x0 + seg / 4} ${y + bob - amp} ${x0 + seg / 2} ${y + bob} Q ${x0 + (seg * 3) / 4} ${y + bob + amp} ${x0 + seg} ${y + bob}`);
  }
  return <path d={pts.join(' ')} fill="none" stroke={col} strokeWidth={sw} strokeLinecap="round" />;
};

/* ---------- 上浮气泡群 ---------- */
const Bubbles = ({x, y, n = 5, h = 160, col = P.air, speed = 1.8}: {
  x: number; y: number; n?: number; h?: number; col?: string; speed?: number;
}) => {
  const f = useCurrentFrame();
  return (
    <g>
      {Array.from({length: n}).map((_, i) => {
        const yy = y - ((f * speed + i * (h / n)) % h);
        const xx = x + Math.sin(f * 0.09 + i * 1.7) * 12 + (i % 2) * 18;
        const r = 4.5 + (i % 3) * 2.5;
        return <circle key={i} cx={xx} cy={yy} r={r} fill="none" stroke={col}
          strokeWidth={4} opacity={0.75} />;
      })}
    </g>
  );
};

/* ---------- 肺（气管 + 双肺叶） ---------- */
const Lungs = ({x, y, s = 1, col = P.air}: {x: number; y: number; s?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-9} y={-98} width={18} height={58} rx={9} fill={col} opacity={0.9} />
    <path d="M -4 -46 L -34 -26 M 4 -46 L 34 -26" stroke={col} strokeWidth={10}
      strokeLinecap="round" fill="none" />
    <ellipse cx={-46} cy={10} rx={42} ry={62} fill={col} opacity={0.85} transform="rotate(7 -46 10)" />
    <ellipse cx={46} cy={10} rx={42} ry={62} fill={col} opacity={0.85} transform="rotate(-7 46 10)" />
  </g>
);

/* ---------- 戴泳帽的头（侧视朝右） ---------- */
const CapHead = ({x, y, s = 1, rot = 0}: {x: number; y: number; s?: number; rot?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`}>
    <circle r={26} fill={C.skin} />
    <path d="M -26 -4 A 26 26 0 0 1 26 -4 L 23 5 Q 0 -3 -23 5 Z" fill={P.cap} />
    {/* 泳镜 */}
    <rect x={4} y={-8} width={24} height={13} rx={6.5} fill={P.waterD} stroke={P.ink} strokeWidth={3} />
    <rect x={-26} y={-6} width={30} height={5} rx={2.5} fill={P.ink} opacity={0.55} />
    <circle cx={16} cy={-2} r={2.6} fill={P.cream} />
    <path d="M 12 14 Q 18 17 24 13" fill="none" stroke={P.ink} strokeWidth={3.5} strokeLinecap="round" />
  </g>
);

/* ---------- 光头（普通人） ---------- */
const PlainHead = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <circle r={26} fill={C.skin} />
    <path d="M -26 -4 A 26 26 0 0 1 26 -4 Q 12 -14 0 -14 Q -12 -14 -26 -4 Z" fill={P.ink} />
    <circle cx={10} cy={-2} r={3} fill={P.ink} />
    <path d="M 8 14 Q 16 17 23 12" fill="none" stroke={P.ink} strokeWidth={3.5} strokeLinecap="round" />
  </g>
);

/* ---------- 游泳小人（侧视自由泳，脸朝右，头出水换气） ---------- */
const Swimmer = ({x, y, s = 1, suit = P.lane}: {x: number; y: number; s?: number; suit?: string}) => {
  const f = useCurrentFrame();
  const kick = Math.sin(f * 0.55) * 9;
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      {/* 水下打腿 */}
      <path d={`M -108 12 L -158 ${26 + kick}`} stroke={C.skin} strokeWidth={15}
        strokeLinecap="round" fill="none" />
      <path d={`M -106 20 L -150 ${48 - kick}`} stroke={C.skin} strokeWidth={15}
        strokeLinecap="round" fill="none" />
      {/* 躯干 */}
      <rect x={-116} y={-10} width={150} height={44} rx={22} fill={suit} />
      {/* 前伸臂 */}
      <path d="M 6 0 L 62 -18 L 108 -20" stroke={C.skin} strokeWidth={14}
        strokeLinecap="round" strokeLinejoin="round" fill="none" />
      {/* 水下划臂 */}
      <path d="M -2 26 L 34 54 L 80 58" stroke={C.skin} strokeWidth={14}
        strokeLinecap="round" strokeLinejoin="round" fill="none" />
      {/* 头（转向侧上换气） */}
      <CapHead x={56} y={-36} s={1} rot={16} />
    </g>
  );
};

/* ---------- 蝶泳小人（双臂上扬，抬头换气） ---------- */
const FlySwimmer = ({x, y, s = 1}: {x: number; y: number; s?: number}) => {
  const f = useCurrentFrame();
  const bob = Math.sin(f * 0.3) * 5;
  return (
    <g transform={`translate(${x},${y + bob}) scale(${s})`}>
      <path d="M -104 10 L -146 34 M -100 20 L -138 50" stroke={C.skin} strokeWidth={14}
        strokeLinecap="round" fill="none" />
      <rect x={-110} y={-10} width={140} height={42} rx={21} fill={P.cap} />
      {/* 双臂呈翅膀上扬 */}
      <path d="M -4 -4 L 44 -44 L 84 -52" stroke={C.skin} strokeWidth={14}
        strokeLinecap="round" fill="none" />
      <path d="M -16 -2 L -52 -38 L -86 -40" stroke={C.skin} strokeWidth={14}
        strokeLinecap="round" fill="none" />
      <CapHead x={46} y={-30} s={0.95} rot={-6} />
    </g>
  );
};

/* ---------- 跑步小人 ---------- */
const Runner = ({x, y, s = 1}: {x: number; y: number; s?: number}) => {
  const f = useCurrentFrame();
  const run = Math.sin(f * 0.5) * 10;
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      {/* 腿 */}
      <path d={`M 0 -52 L -30 -10 L ${-44 + run} 30`} stroke={P.lime} strokeWidth={14}
        strokeLinecap="round" strokeLinejoin="round" fill="none" />
      <path d={`M 0 -52 L 26 -16 L ${54 - run} 6`} stroke={P.lime} strokeWidth={14}
        strokeLinecap="round" strokeLinejoin="round" fill="none" />
      {/* 躯干前倾 */}
      <line x1={0} y1={-52} x2={16} y2={-104} stroke={P.lime} strokeWidth={16} strokeLinecap="round" />
      {/* 手臂摆动 */}
      <path d={`M 12 -96 L ${44 + run * 0.6} -76`} stroke={P.lime} strokeWidth={11} strokeLinecap="round" />
      <path d={`M 10 -92 L ${-16 - run * 0.6} -66`} stroke={P.lime} strokeWidth={11} strokeLinecap="round" />
      <circle cx={22} cy={-124} r={19} fill={C.skin} />
    </g>
  );
};

/* ---------- 竖直肺活量量筒（底部进气，液柱上升） ---------- */
const GaugeTube = ({x, y, s = 1, h = 400, w = 104, frac, col = P.air}: {
  x: number; y: number; s?: number; h?: number; w?: number; frac: number; col?: string;
}) => {
  const fillH = Math.max(0, (h - 20) * clamp01(frac));
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      <rect x={-w / 2} y={-h} width={w} height={h} rx={18} fill={P.bgDeep}
        stroke={P.steel} strokeWidth={5} />
      {fillH > 2 && (
        <rect x={-w / 2 + 9} y={-14 - fillH} width={w - 18} height={fillH} rx={12} fill={col} />
      )}
      {[1, 2, 3, 4, 5, 6].map((i) => (
        <line key={i} x1={w / 2 - 2} y1={-14 - ((h - 20) * i) / 7} x2={w / 2 + 18}
          y2={-14 - ((h - 20) * i) / 7} stroke={P.muted} strokeWidth={4} opacity={0.7} />
      ))}
      <text x={w / 2 + 22} y={-h + 30} fontSize={21} fill={P.muted} fontFamily={FONT_HEAD}>升</text>
    </g>
  );
};

/* ---------- 吹气的人（肩以上 + 气管接到量筒） ---------- */
const Blower = ({x, y, s = 1, cap = false, suit = P.card}: {
  x: number; y: number; s?: number; cap?: boolean; suit?: string;
}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <path d="M -78 96 Q 0 36 78 96 L 78 110 L -78 110 Z" fill={suit} />
    {cap ? <CapHead x={0} y={-12} s={1.3} /> : <PlainHead x={0} y={-12} s={1.3} />}
    {/* 嘟嘴 */}
    <circle cx={36} cy={-8} r={8} fill={P.ink} />
  </g>
);

/* ---------- 星芒爆发 ---------- */
const Burst = ({x, y, s = 1, col = P.red}: {x: number; y: number; s?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} stroke={col} strokeWidth={8} strokeLinecap="round">
    {Array.from({length: 8}).map((_, i) => (
      <line key={i} x1={0} y1={-30} x2={0} y2={-56} transform={`rotate(${i * 45})`} />
    ))}
    <circle r={16} fill={col} />
  </g>
);

/* ---------- 迷你小人（选拔用） ---------- */
const MiniPerson = ({x, y, s = 1, big = false}: {x: number; y: number; s?: number; big?: boolean}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <circle cy={-34} r={13} fill={C.skin} />
    <rect x={big ? -22 : -14} y={-18} width={big ? 44 : 28} height={46} rx={big ? 16 : 11}
      fill={big ? P.air : P.steel} />
    <line x1={-8} y1={28} x2={-10} y2={52} stroke={big ? P.air : P.steel} strokeWidth={9} strokeLinecap="round" />
    <line x1={8} y1={28} x2={10} y2={52} stroke={big ? P.air : P.steel} strokeWidth={9} strokeLinecap="round" />
  </g>
);

/* ---------- 划水节奏格 ---------- */
const BeatCell = ({x, y, kind}: {x: number; y: number; kind: 'stroke' | 'breath' | 'free'}) => (
  <g transform={`translate(${x},${y})`}>
    <rect x={0} y={0} width={60} height={56} rx={12}
      fill={kind === 'breath' ? P.air : P.bgDeep}
      stroke={kind === 'breath' ? P.air : P.card} strokeWidth={4} />
    {kind === 'stroke' && (
      <g stroke={P.muted} strokeWidth={6} strokeLinecap="round">
        <path d="M 16 38 L 34 20" fill="none" />
        <circle cx={40} cy={16} r={6} fill={P.muted} stroke="none" />
      </g>
    )}
    {kind === 'breath' && (
      <g transform="translate(30,28) scale(0.26)">
        <rect x={-9} y={-98} width={18} height={58} rx={9} fill={P.ink} opacity={0.9} />
        <ellipse cx={-40} cy={8} rx={38} ry={56} fill={P.ink} />
        <ellipse cx={40} cy={8} rx={38} ry={56} fill={P.ink} />
      </g>
    )}
    {kind === 'free' && (
      <g stroke={P.muted} strokeWidth={5} strokeLinecap="round" fill="none">
        <path d="M 16 36 Q 30 26 44 34" />
        <path d="M 20 24 Q 32 16 42 22" opacity={0.6} />
      </g>
    )}
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
        {Array.from({length: 9}).map((_, i) => (
          <circle key={i} cx={140 + i * 205} cy={86 + (i % 2) * 26} r={4} fill={P.cream} opacity={0.3} />
        ))}
        <g transform={`translate(960,540) scale(${pA}) translate(-960,-540)`} opacity={clamp01(pA)}>
          {/* 水面 */}
          <rect x={560} y={556} width={800} height={150} fill={P.water} opacity={0.2} />
          <Wave x={560} y={556} w={800} />
          {/* 泳道线浮标 */}
          {[0, 1, 2, 3].map((i) => (
            <ellipse key={`l${i}`} cx={612 + i * 46} cy={600} rx={15} ry={10}
              fill={i % 2 ? '#f5f7ff' : P.lane} />
          ))}
          {[0, 1, 2, 3].map((i) => (
            <ellipse key={`r${i}`} cx={1190 + i * 46} cy={600} rx={15} ry={10}
              fill={i % 2 ? '#f5f7ff' : P.lane} />
          ))}
          {/* 泳者换气 */}
          <Swimmer x={950} y={560} s={1.15} />
          <Bubbles x={1060} y={548} n={5} h={130} />
          <TagPill x={700} y={330} t="肺活量 6升+" col={P.cap} w={220} fs={32} s={1.05} />
        </g>
        <g transform={`translate(396,310) scale(${pB})`} opacity={clamp01(pB)}>
          <TagPill x={0} y={0} t="水压负重呼吸" col={P.air} w={240} />
        </g>
        <g transform={`translate(1450,300) scale(${pB})`} opacity={clamp01(pB)}>
          <TagPill x={0} y={0} t="换气限频" col={P.co2} w={200} />
        </g>
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 480}}>
          <div style={{color: P.air, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 24}}>
            冷知识 · 第 {data.episode} 期
          </div>
          <div style={{color: C.text, fontSize: 68, fontWeight: 900, fontFamily: FONT_HEAD, letterSpacing: 4}}>
            {data.title}
          </div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ============ S0 数字先吓你一跳：双量筒吹气对比 + 三要素 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const popL = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const popR = Math.max(0.001, spring({frame: f - 55, fps, config: {damping: 11, stiffness: 120}}));
  const fillL = clamp01((f - 24) / 60);
  const fillR = clamp01((f - 70) / 70);
  const badgeK = Math.max(0.001, spring({frame: f - 150, fps, config: {damping: 10, stiffness: 140}}));
  const stripK = clamp01((f - b1 - 6) / 24);

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={0} name={sec.name} accent={P.cap} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：普通人 */}
        <g transform={`translate(90,150) scale(${popL}) translate(-90,-150)`} opacity={clamp01(popL * 2)}>
          <rect x={90} y={150} width={820} height={560} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={126} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            普通人 · 体测吹个三千上下</text>
          <Blower x={280} y={520} s={1.15} suit={P.card} />
          {/* 气管 */}
          <path d="M 322 502 Q 470 540 590 610" fill="none" stroke={P.steel}
            strokeWidth={13} strokeLinecap="round" />
          <GaugeTube x={640} y={660} frac={0.5 * fillL} col={P.steel} />
          <TagPill x={640} y={222} t="≈3.5升" col={P.steel} w={180} fs={32} s={0.95} />
          <text x={500} y={692} textAnchor="middle" fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>
            一次尽力吸气后能呼出的最大气量</text>
        </g>
        {/* 右：游泳选手（紧随 cue0 后半） */}
        <g transform={`translate(960,150) scale(${popR}) translate(-960,-150)`} opacity={clamp01(popR * 2)}>
          <rect x={960} y={150} width={870} height={560} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={996} y={206} fill={P.cream} fontSize={26} fontFamily={FONT_HEAD} fontWeight={900}>
            游泳选手 · 五六千起步，大个子冲七千</text>
          <Blower x={1200} y={520} s={1.15} cap suit={P.lane} />
          <path d="M 1242 502 Q 1390 540 1510 610" fill="none" stroke={P.steel}
            strokeWidth={13} strokeLinecap="round" />
          <GaugeTube x={1560} y={660} frac={0.93 * fillR} col={P.air} />
          {fillR > 0.9 && <Burst x={1560} y={218} s={1} />}
          <TagPill x={1560} y={300} t="6升+" col={P.air} w={160} fs={32} s={0.95} />
          <TagPill x={1720} y={420} t="像仪器坏了" col={P.cap} w={190} fs={24} s={0.9} />
          <text x={1395} y={692} textAnchor="middle" fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>
            同款仪器 · 不是坏了</text>
        </g>
        {/* 中：翻倍徽章（压在两板接缝上，最后画） */}
        <g transform={`translate(935,430) scale(${badgeK})`} opacity={clamp01(badgeK)}>
          <circle r={64} fill={P.bgDeep} stroke={P.cap} strokeWidth={6} />
          <text x={0} y={6} textAnchor="middle" fontSize={42} fill={P.cap}
            fontFamily={FONT_HEAD} fontWeight={900}>≈2×</text>
          <text x={0} y={36} textAnchor="middle" fontSize={19} fill={P.muted}
            fontFamily={FONT_HEAD}>肺活量</text>
        </g>
        {/* 底部：三要素（cue1） */}
        {stripK > 0 && (
          <g opacity={stripK} transform={`translate(0,${(1 - stripK) * 40})`}>
            <rect x={90} y={745} width={1740} height={95} rx={18} fill={P.panel} stroke={P.card} strokeWidth={4} />
            <TagPill x={470} y={792} t="肺总容量" col={P.air} w={200} fs={27} />
            <text x={640} y={804} textAnchor="middle" fontSize={36} fill={P.muted} fontFamily={FONT_HEAD}>+</text>
            <TagPill x={820} y={792} t="胸廓活动度" col={P.air} w={220} fs={27} />
            <text x={990} y={804} textAnchor="middle" fontSize={36} fill={P.muted} fontFamily={FONT_HEAD}>+</text>
            <TagPill x={1170} y={792} t="呼吸肌力量" col={P.air} w={220} fs={27} />
            <text x={1350} y={804} textAnchor="middle" fontSize={36} fill={P.muted} fontFamily={FONT_HEAD}>=</text>
            <text x={1580} y={804} textAnchor="middle" fontSize={30} fill={P.cap}
              fontFamily={FONT_HEAD} fontWeight={900}>综合成绩</text>
          </g>
        )}
      </svg>
      <Chip text="数字看着像仪器坏了" x={1150} y={72} at={beat(0) + 25} accent={P.cap} />
    </FadeIn>
  );
};

/* ============ S1 水压是隐形阻力器：水中剖面 + 工作量对比 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const b2 = beat(2);
  const pop = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const rightK = clamp01((f - b1 - 8) / 26);
  const bar1 = clamp01((f - b1 - 20) / 40);
  const bar2 = clamp01((f - b1 - 34) / 46);
  const muscK = clamp01((f - b2 - 6) / 24);
  const wob = Math.sin(f * 0.2) * 10;

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={1} name={sec.name} accent={P.water} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：水中站立剖面 */}
        <g transform={`translate(90,150) scale(${pop}) translate(-90,-150)`} opacity={clamp01(pop * 2)}>
          <rect x={90} y={150} width={860} height={560} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={126} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            齐胸深的水 · 吸气得先顶开水压</text>
          {/* 站立的人（头出水面） */}
          <PlainHead x={470} y={298} s={1.3} />
          <rect x={452} y={328} width={36} height={52} rx={14} fill={C.skin} />
          {/* 胸廓剖面 */}
          <rect x={390} y={368} width={160} height={294} rx={46} fill={P.card}
            stroke={P.cream} strokeWidth={4} />
          <rect x={462} y={384} width={16} height={260} rx={8} fill={P.bgDeep} />
          <ellipse cx={428} cy={440} rx={27} ry={36} fill={P.air} opacity={0.85} />
          <ellipse cx={512} cy={440} rx={27} ry={36} fill={P.air} opacity={0.85} />
          <path d="M 402 545 Q 470 505 538 545" stroke={P.airD} strokeWidth={11}
            fill="none" strokeLinecap="round" />
          {/* 水体覆盖 */}
          <rect x={140} y={360} width={660} height={320} fill={P.water} opacity={0.2} />
          <line x1={140} y1={360} x2={140} y2={680} stroke={P.waterD} strokeWidth={5} opacity={0.6} />
          <line x1={800} y1={360} x2={800} y2={680} stroke={P.waterD} strokeWidth={5} opacity={0.6} />
          <Wave x={140} y={360} w={660} />
          {/* 四面八方压向胸廓的箭头 */}
          {[0, 30, 60, 90, 120, 150, 180, 200, 225, 315, 340].map((a) => (
            <InArrow key={a} cx={470} cy={500} a={a}
              rx1={252 + wob} ry1={158 + wob * 0.6} rx2={94} ry2={144} col={P.cap} sw={8} />
          ))}
          <TagPill x={800} y={290} t="水从四面八方挤胸腹" col={P.cap} w={280} fs={24} s={0.9} />
        </g>
        {/* 右：工作量对比（cue1） */}
        {rightK > 0 && (
          <g opacity={rightK} transform={`translate(0,${(1 - rightK) * 40})`}>
            <rect x={990} y={150} width={840} height={560} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
            <text x={1026} y={206} fill={P.cream} fontSize={26} fontFamily={FONT_HEAD} fontWeight={900}>
              呼吸肌工作量 · 实测对比</text>
            {/* 双柱 */}
            <line x1={1060} y1={600} x2={1790} y2={600} stroke={P.card} strokeWidth={5} />
            <rect x={1130} y={600 - 200 * bar1} width={170} height={200 * bar1} rx={14} fill={P.steel} />
            <rect x={1420} y={600 - 330 * bar2} width={170} height={330 * bar2} rx={14} fill={P.cap} />
            {bar2 > 0.9 && (
              <g>
                <text x={1660} y={330} textAnchor="middle" fontSize={68} fill={P.cap}
                  fontFamily={FONT_HEAD} fontWeight={900}>+60%</text>
                <line x1={1660} y1={356} x2={1660} y2={392} stroke={P.cap} strokeWidth={8} strokeLinecap="round" />
                <polygon points="1660,352 1646,378 1674,378" fill={P.cap} transform="rotate(180 1660 365)" />
              </g>
            )}
            <text x={1215} y={642} textAnchor="middle" fontSize={26} fill={P.steel}
              fontFamily={FONT_HEAD} fontWeight={800}>陆地 1×</text>
            <text x={1505} y={642} textAnchor="middle" fontSize={26} fill={P.cap}
              fontFamily={FONT_HEAD} fontWeight={800}>齐胸水 1.6×</text>
            {/* cue2：呼吸肌名单 */}
            {muscK > 0 && (
              <g opacity={muscK}>
                <rect x={1026} y={656} width={768} height={42} rx={12} fill={P.bgDeep} opacity={0.6} />
                <text x={1410} y={686} textAnchor="middle" fontSize={23} fill={P.air}
                  fontFamily={FONT_HEAD} fontWeight={800}>
                  膈肌 + 肋间肌 + 辅助吸气肌 = 天天抗阻</text>
              </g>
            )}
          </g>
        )}
      </svg>
      <Chip text="负重呼吸 · 每天几小时" x={1120} y={72} at={beat(0) + 25} accent={P.water} />
      <Stamp text="和举铁练肌肉同款逻辑" x={560} y={86} at={b2 + 30} color={P.waterD} size={36} rot={-3} />
    </FadeIn>
  );
};

/* ============ S2 换气被强行限频：换气窗口 + 节奏对照 + 低氧训练 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const b2 = beat(2);
  const popT = Math.max(0.001, spring({frame: f - 8, fps, config: {damping: 11, stiffness: 120}}));
  const popL = Math.max(0.001, spring({frame: f - 14, fps, config: {damping: 11, stiffness: 120}}));
  const rightK = clamp01((f - b1 - 8) / 26);
  const boxB = Math.max(0.001, spring({frame: f - b1 - 40, fps, config: {damping: 11, stiffness: 130}}));
  const co2K = clamp01((f - b2 - 8) / 40);

  const runCells = Array.from({length: 8});
  const swimCells: ('stroke' | 'breath')[] = ['stroke', 'stroke', 'breath', 'stroke', 'stroke', 'breath', 'stroke', 'stroke'];

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={2} name={sec.name} accent={P.air} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 上：泳姿把呼吸窗口压窄 */}
        <g transform={`translate(90,140) scale(${popT}) translate(-90,-140)`} opacity={clamp01(popT * 2)}>
          <rect x={90} y={140} width={1740} height={300} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={126} y={196} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            脸埋在水里 · 换气窗口被泳姿压窄</text>
          {/* 水体 */}
          <rect x={110} y={300} width={1700} height={124} fill={P.water} opacity={0.2} />
          <Wave x={110} y={300} w={1700} />
          {/* 自由泳换气 */}
          <Swimmer x={560} y={316} s={1.0} />
          <Bubbles x={600} y={330} n={4} h={70} speed={1.4} />
          {/* 换气窗口括号 */}
          <line x1={592} y1={252} x2={592} y2={330} stroke={P.air} strokeWidth={6} strokeLinecap="round" />
          <line x1={664} y1={252} x2={664} y2={330} stroke={P.air} strokeWidth={6} strokeLinecap="round" />
          <line x1={592} y1={255} x2={610} y2={255} stroke={P.air} strokeWidth={6} strokeLinecap="round" />
          <line x1={664} y1={255} x2={646} y2={255} stroke={P.air} strokeWidth={6} strokeLinecap="round" />
          <TagPill x={628} y={216} t="换气窗口 ≈0.3秒" col={P.air} w={260} fs={25} s={0.9} />
          <TagPill x={560} y={392} t="自由泳转体换气" col={P.muted} w={240} fs={23} s={0.85} />
          {/* 蝶泳 */}
          <FlySwimmer x={1330} y={322} s={0.95} />
          <line x1={1370} y1={252} x2={1370} y2={330} stroke={P.cap} strokeWidth={6} strokeLinecap="round" />
          <line x1={1428} y1={252} x2={1428} y2={330} stroke={P.cap} strokeWidth={6} strokeLinecap="round" />
          <TagPill x={1400} y={216} t="蝶泳按节奏抬头" col={P.cap} w={250} fs={25} s={0.9} />
          <TagPill x={1660} y={392} t="蛙泳窗口同样窄" col={P.muted} w={240} fs={23} s={0.85} />
        </g>
        {/* 左下：节奏对照 */}
        <g transform={`translate(90,480) scale(${popL}) translate(-90,-480)`} opacity={clamp01(popL * 2)}>
          <rect x={90} y={480} width={860} height={350} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={126} y={536} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            节奏对照 · 想喘就喘 vs 按点换气</text>
          <text x={150} y={646} fontSize={26} fill={P.cream} fontFamily={FONT_HEAD} fontWeight={800}>跑步</text>
          {runCells.map((_, i) => (
            <BeatCell key={i} x={230 + i * 70} y={600} kind="free" />
          ))}
          <text x={150} y={756} fontSize={26} fill={P.cream} fontFamily={FONT_HEAD} fontWeight={800}>游泳</text>
          {swimCells.map((k, i) => (
            <BeatCell key={i} x={230 + i * 70} y={710} kind={k} />
          ))}
          <text x={300} y={806} fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>
            游泳：三拍才换到一口气</text>
        </g>
        {/* 右下：低氧训练（cue1） */}
        {rightK > 0 && (
          <g opacity={rightK} transform={`translate(0,${(1 - rightK) * 40})`}>
            <rect x={990} y={480} width={840} height={350} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
            <text x={1026} y={536} fill={P.cream} fontSize={26} fontFamily={FONT_HEAD} fontWeight={900}>
              低氧训练 · 人为间歇缺氧</text>
            {/* 三划一吸 */}
            <rect x={1030} y={570} width={320} height={110} rx={16} fill={P.bgDeep} stroke={P.air} strokeWidth={4} />
            <text x={1190} y={610} textAnchor="middle" fontSize={27} fill={P.air}
              fontFamily={FONT_HEAD} fontWeight={900}>三划一吸</text>
            {(['stroke', 'stroke', 'breath'] as const).map((k, i) => (
              <g key={i} transform={`translate(${1068 + i * 88},622) scale(0.72)`}>
                <BeatCell x={0} y={0} kind={k} />
              </g>
            ))}
            <Chev x={1385} y={625} col={P.cap} />
            {/* 五划一吸 */}
            {boxB > 0.02 && (
              <g transform={`translate(1420,570) scale(${boxB}) translate(-1420,-570)`}
                opacity={clamp01(boxB * 2)}>
                <rect x={1420} y={570} width={390} height={110} rx={16} fill={P.bgDeep} stroke={P.co2} strokeWidth={4} />
                <text x={1615} y={610} textAnchor="middle" fontSize={27} fill={P.co2}
                  fontFamily={FONT_HEAD} fontWeight={900}>五划一吸 · 更狠</text>
                {(['stroke', 'stroke', 'stroke', 'stroke', 'breath'] as const).map((k, i) => (
                  <g key={i} transform={`translate(${1452 + i * 70},622) scale(0.72)`}>
                    <BeatCell x={0} y={0} kind={k} />
                  </g>
                ))}
              </g>
            )}
            {/* CO2 耐受条（cue2） */}
            {co2K > 0 && (
              <g opacity={co2K}>
                <rect x={1030} y={722} width={600} height={40} rx={14} fill={P.bgDeep}
                  stroke={P.card} strokeWidth={3} />
                <rect x={1034} y={726} width={592 * co2K * 0.85} height={32} rx={11} fill={P.co2} />
                <text x={1660} y={752} fontSize={24} fill={P.co2} fontFamily={FONT_HEAD} fontWeight={800}>
                  CO₂耐受↑</text>
                <text x={1330} y={806} textAnchor="middle" fontSize={23} fill={P.muted}
                  fontFamily={FONT_HEAD}>每次换气都往极限效率练 · 摄氧效率跟着拉满</text>
              </g>
            )}
          </g>
        )}
      </svg>
      <Chip text="没法像跑步那样想喘就喘" x={1080} y={72} at={beat(0) + 25} accent={P.air} />
      <Stamp text="间歇缺氧是练出来的" x={560} y={86} at={b1 + 70} color={P.co2} size={36} rot={-3} />
    </FadeIn>
  );
};

/* ============ S3 连肺的结构都改了：肺对比 + 俯卧呼吸 + 憋气 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const b2 = beat(2);
  const popL = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const midK = clamp01((f - b1 - 6) / 26);
  const rightK = clamp01((f - b1 - 40) / 26);
  const warnK = clamp01((f - b2 - 4) / 24);
  const barK = clamp01((f - b1 - 60) / 50);

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={3} name={sec.name} accent={P.co2} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：两副肺 */}
        <g transform={`translate(90,150) scale(${popL}) translate(-90,-150)`} opacity={clamp01(popL * 2)}>
          <rect x={90} y={150} width={560} height={560} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={126} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>两副肺不一样</text>
          <Lungs x={245} y={420} s={0.72} col={P.steel} />
          <Lungs x={480} y={430} s={1.0} col={P.air} />
          <circle cx={480} cy={430} r={108} fill="none" stroke={P.cap} strokeWidth={5}
            strokeDasharray="14 10" opacity={0.85} />
          <text x={360} y={392} textAnchor="middle" fontSize={26} fill={P.cap}
            fontFamily={FONT_HEAD} fontWeight={900}>VS</text>
          <text x={245} y={622} textAnchor="middle" fontSize={24} fill={P.muted} fontFamily={FONT_HEAD}>普通人</text>
          <text x={480} y={622} textAnchor="middle" fontSize={24} fill={P.air} fontFamily={FONT_HEAD}
            fontWeight={800}>运动员</text>
          <TagPill x={370} y={668} t="肺总量↑ 残气量↑" col={P.cap} w={280} fs={24} s={0.92} />
        </g>
        {/* 中：俯卧位呼吸（cue1） */}
        {midK > 0 && (
          <g opacity={midK} transform={`translate(0,${(1 - midK) * 40})`}>
            <rect x={690} y={150} width={560} height={560} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
            <text x={726} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
              俯卧位 · 膈肌被水压挤</text>
            {/* 俯卧躯干剖面 */}
            <rect x={770} y={360} width={410} height={200} rx={60} fill={P.card}
              stroke={P.cream} strokeWidth={4} />
            {/* 脊柱（俯卧朝上） */}
            <line x1={800} y1={392} x2={1150} y2={392} stroke={P.bgDeep} strokeWidth={15}
              strokeLinecap="round" strokeDasharray="26 16" />
            {/* 被压平的膈肌 */}
            <path d="M 830 510 Q 975 480 1120 510" stroke={P.airD} strokeWidth={11}
              fill="none" strokeLinecap="round" />
            <text x={975} y={470} textAnchor="middle" fontSize={20} fill={P.airD}
              fontFamily={FONT_HEAD}>膈肌</text>
            {/* 两侧肋间肌高亮 */}
            <ellipse cx={802} cy={462} rx={22} ry={46} fill={P.lime} opacity={0.9} />
            <ellipse cx={1148} cy={462} rx={22} ry={46} fill={P.lime} opacity={0.9} />
            {/* 水从下方顶上来 */}
            {[870, 975, 1080].map((x) => (
              <g key={x}>
                <line x1={x} y1={648} x2={x} y2={576} stroke={P.cap} strokeWidth={9} strokeLinecap="round" />
                <polygon points={`${x},562 ${x - 14},586 ${x + 14},586`} fill={P.cap} />
              </g>
            ))}
            <TagPill x={975} y={300} t="膈肌空间被压缩" col={P.red} w={240} fs={24} s={0.9} />
            <TagPill x={975} y={668} t="肋间肌·辅助肌全面上阵" col={P.lime} w={330} fs={23} s={0.9} />
          </g>
        )}
        {/* 右：静态憋气对比（cue1 后半） */}
        {rightK > 0 && (
          <g opacity={rightK} transform={`translate(0,${(1 - rightK) * 40})`}>
            <rect x={1290} y={150} width={540} height={560} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
            <text x={1326} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
              静态憋气 · 附赠能力</text>
            {/* 三条横bar */}
            <text x={1360} y={286} fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>普通人</text>
            <rect x={1360} y={300} width={130 * Math.min(1, barK * 2)} height={46} rx={12} fill={P.steel} />
            <text x={1510} y={332} fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>≈1分钟</text>
            <text x={1360} y={416} fontSize={22} fill={P.air} fontFamily={FONT_HEAD}>游泳选手</text>
            <rect x={1360} y={430} width={280 * barK} height={46} rx={12} fill={P.air} />
            <text x={1660} y={462} fontSize={22} fill={P.air} fontFamily={FONT_HEAD}>2分钟+</text>
            <text x={1360} y={546} fontSize={22} fill={P.cap} fontFamily={FONT_HEAD}>自由潜</text>
            <rect x={1360} y={560} width={400 * barK} height={46} rx={12} fill="none"
              stroke={P.cap} strokeWidth={5} strokeDasharray="16 12" />
            <text x={1360} y={640} fontSize={22} fill={P.cap} fontFamily={FONT_HEAD}>7分钟 · 另一个极端领域</text>
            {/* 警告（cue2） */}
            {warnK > 0 && (
              <g opacity={warnK}>
                <g transform="translate(1360,682)">
                  <path d="M 0 -26 L 28 18 L -28 18 Z" fill={P.bgDeep} stroke={P.red}
                    strokeWidth={5} strokeLinejoin="round" />
                  <line x1={0} y1={-12} x2={0} y2={4} stroke={P.red} strokeWidth={6} strokeLinecap="round" />
                  <circle cy={11} r={3.5} fill={P.red} />
                </g>
                <text x={1406} y={690} fontSize={23} fill={P.red} fontFamily={FONT_HEAD} fontWeight={800}>
                  浅水昏厥 · 毫无征兆</text>
              </g>
            )}
          </g>
        )}
      </svg>
      <Chip text="适应不止肌肉层面" x={1150} y={72} at={beat(0) + 25} accent={P.co2} />
      <Stamp text="水里憋气别模仿！" x={620} y={86} at={b2 + 10} color={P.red} size={38} rot={3} />
    </FadeIn>
  );
};

/* ============ S4 天赋加训练缺一不可：选材漏斗 + 别的项目 + 阶梯 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const b2 = beat(2);
  const popL = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const rightK = clamp01((f - b1 - 8) / 26);
  const stairK = clamp01((f - b2 - 8) / 26);
  const dropK = clamp01((f - 40) / 40);

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={4} name={sec.name} accent={P.lime} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左上：选材漏斗 */}
        <g transform={`translate(90,150) scale(${popL}) translate(-90,-150)`} opacity={clamp01(popL * 2)}>
          <rect x={90} y={150} width={800} height={400} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={126} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            选材也在筛人 · 天生肺大胸廓灵活先进队</text>
          {/* 五个候选小人 */}
          {[0, 1, 2, 3, 4].map((i) => (
            <g key={i}>
              <MiniPerson x={210 + i * 145} y={300} s={0.9} big={i === 1 || i === 3} />
              {(i === 0 || i === 2 || i === 4) && <MarkX x={210 + i * 145} y={212} s={0.8} />}
            </g>
          ))}
          {/* 漏斗 */}
          <path d="M 220 386 L 780 386 L 540 486 L 460 486 Z" fill={P.bgDeep}
            stroke={P.cream} strokeWidth={5} strokeLinejoin="round" />
          {/* 入选滑落箭头 */}
          {[355, 645].map((x, i) => (
            <g key={x} opacity={dropK}>
              <line x1={x} y1={336} x2={x + (i === 0 ? 60 : -60)} y2={452}
                stroke={P.air} strokeWidth={8} strokeLinecap="round"
                strokeDasharray="10 10" />
              <polygon points={`${x + (i === 0 ? 60 : -60)},452 ${x + (i === 0 ? 44 : -44)},434 ${x + (i === 0 ? 74 : -74)},438`}
                fill={P.air} transform={`translate(0,14)`} />
            </g>
          ))}
          <rect x={380} y={496} width={240} height={44} rx={12} fill={P.bgDeep}
            stroke={P.air} strokeWidth={4} />
          <text x={500} y={526} textAnchor="middle" fontSize={25} fill={P.air}
            fontFamily={FONT_HEAD} fontWeight={900}>专业队</text>
          <text x={760} y={470} fontSize={21} fill={P.muted} fontFamily={FONT_HEAD}>
            结果里混着起点</text>
        </g>
        {/* 右上：别的项目缺了关键压力（cue1） */}
        {rightK > 0 && (
          <g opacity={rightK} transform={`translate(0,${(1 - rightK) * 40})`}>
            <rect x={970} y={150} width={860} height={400} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
            <text x={1006} y={206} fill={P.cream} fontSize={26} fontFamily={FONT_HEAD} fontWeight={900}>
              别的项目 · 缺了水压和限频</text>
            {/* 跑步卡 */}
            <rect x={1010} y={240} width={380} height={230} rx={16} fill={P.bgDeep} stroke={P.lime} strokeWidth={4} />
            <Runner x={1120} y={420} s={0.85} />
            {[0, 1, 2, 3].map((i) => (
              <path key={i} d={`M ${1240 + i * 30} ${300 + (i % 2) * 34} q 14 -10 26 0`}
                fill="none" stroke={P.muted} strokeWidth={5} strokeLinecap="round" opacity={0.8} />
            ))}
            <text x={1200} y={452} textAnchor="middle" fontSize={23} fill={P.lime}
              fontFamily={FONT_HEAD} fontWeight={800}>跑步·骑车 · 想喘就喘</text>
            {/* 游泳卡 */}
            <rect x={1430} y={240} width={380} height={230} rx={16} fill={P.bgDeep} stroke={P.air} strokeWidth={4} />
            <rect x={1450} y={330} width={340} height={80} fill={P.water} opacity={0.2} />
            <Wave x={1450} y={330} w={340} sw={5} amp={7} seg={50} />
            <Swimmer x={1560} y={344} s={0.62} />
            <text x={1620} y={452} textAnchor="middle" fontSize={23} fill={P.air}
              fontFamily={FONT_HEAD} fontWeight={800}>游泳 · 顶着水压按点换气</text>
            <TagPill x={1410} y={516} t="没有同款压力 → 练不出同款呼吸肌" col={P.red} w={480} fs={24} s={0.9} />
          </g>
        )}
        {/* 底部：阶梯（cue2） */}
        {stairK > 0 && (
          <g opacity={stairK} transform={`translate(0,${(1 - stairK) * 40})`}>
            <rect x={90} y={590} width={1740} height={240} rx={20} fill={P.panel} stroke={P.lime} strokeWidth={4} />
            {/* 台阶 */}
            <rect x={140} y={740} width={260} height={70} rx={10} fill={P.card} stroke={P.steel} strokeWidth={4} />
            <rect x={400} y={680} width={260} height={130} rx={10} fill={P.card} stroke={P.steel} strokeWidth={4} />
            <rect x={660} y={620} width={260} height={190} rx={10} fill={P.card} stroke={P.lime} strokeWidth={5} />
            <MiniPerson x={905} y={578} s={0.75} big />
            {/* 上升虚线箭头 */}
            <path d="M 270 720 L 780 590" fill="none" stroke={P.cap} strokeWidth={7}
              strokeDasharray="14 12" strokeLinecap="round" />
            <polygon points="800,582 768,586 788,612" fill={P.cap} />
            <text x={270} y={798} textAnchor="middle" fontSize={22} fill={P.muted}
              fontFamily={FONT_HEAD}>天赋 = 下限</text>
            <text x={790} y={674} textAnchor="middle" fontSize={22} fill={P.lime}
              fontFamily={FONT_HEAD} fontWeight={800}>训练 = 把上限往上挪</text>
            {/* 右侧文案 */}
            <text x={1390} y={676} textAnchor="middle" fontSize={34} fill={P.cream}
              fontFamily={FONT_HEAD} fontWeight={900}>普通人每周规律游泳 · 呼吸肌照样变强</text>
            <text x={1390} y={730} textAnchor="middle" fontSize={24} fill={P.muted}
              fontFamily={FONT_HEAD}>肺活量跟肱二头肌一样 · 吃训练</text>
            <TagPill x={1390} y={788} t="呼吸系统的保养运动" col={P.air} w={320} fs={24} s={0.9} />
          </g>
        )}
      </svg>
      <Chip text="不全是练出来的 · 选材先筛一道" x={1080} y={72} at={beat(0) + 25} accent={P.lime} />
      <Stamp text="天赋定下限 · 训练定上限" x={560} y={86} at={b1 + 60} color={P.airD} size={36} rot={-3} />
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
        <g opacity={clamp01(pop)}>
          {/* 水面 */}
          <rect x={0} y={600} width={1920} height={220} fill={P.water} opacity={0.14} />
          <Wave x={0} y={600} w={1920} seg={90} />
          {/* 两条泳道线 */}
          {Array.from({length: 30}).map((_, i) => (
            <ellipse key={`a${i}`} cx={60 + i * 62} cy={664} rx={15} ry={10}
              fill={i % 2 ? '#f5f7ff' : P.lane} />
          ))}
          {Array.from({length: 30}).map((_, i) => (
            <ellipse key={`b${i}`} cx={60 + i * 62} cy={764} rx={15} ry={10}
              fill={i % 2 ? '#f5f7ff' : P.lane} />
          ))}
          {/* 漂浮的泳帽头 + 气泡 */}
          <g transform={`translate(960,${560 + Math.sin(f * 0.12) * 6}) scale(${pop})`}>
            <CapHead x={0} y={0} s={1.5} />
          </g>
          <Bubbles x={1030} y={580} n={5} h={140} />
        </g>
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -330}}>
        <div style={{fontSize: 52, fontWeight: 800, color: C.text}}>肺活量不是天生 · 是顶着水压练出来的</div>
        <div style={{fontSize: 36, color: P.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E249 = ({data = tl}: {data?: any}) => {
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
