import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01} from '../lib/style';
import {Backdrop, Stamp, SceneTag, Chip, Subtitles, ProgressBar, FadeIn, Moon, Stars} from '../lib/atoms';
import tl from '../timeline_238.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：钢夜蓝 × 压力珊瑚 × REM薄荷 × 皮质醇琥珀 ============ */
const P = {
  bg: '#17263c',
  bgDeep: '#0e1a2c',
  bgLite: '#1f3150',
  panel: '#1b2c47',
  card: '#2d4468',
  ink: '#0c1626',
  cream: '#f4eeda',
  muted: '#98aacf',
  coral: '#ff7d6b',
  coralD: '#d95b4c',
  amber: '#ffc25c',
  amberD: '#e09a38',
  teal: '#5fe3b3',
  tealD: '#36ac87',
  sky: '#79baff',
  skyD: '#4e92da',
  vio: '#c4a8f4',
  vioD: '#9578dd',
  rose: '#ff9fb2',
  water: '#35739e',
  hair: '#453a56',
};

/* ---------- 飘 z ---------- */
const Zzz = ({x, y, k = 1, col = P.teal}: {x: number; y: number; k?: number; col?: string}) => {
  const f = useCurrentFrame();
  return (
    <g transform={`translate(${x},${y})`} opacity={k}>
      {[0, 1, 2].map((i) => {
        const t = (f * 0.8 + i * 18) % 54;
        const p = t / 54;
        return (
          <text key={i} x={i * 26 + Math.sin(p * 3) * 8} y={-p * 64}
            fontSize={26 + i * 12} fill={col} fontFamily={FONT_HEAD} fontWeight={800}
            opacity={0.9 * (1 - p)}>z</text>
        );
      })}
    </g>
  );
};

/* ---------- 床上侧躺睡觉的小人 ---------- */
const Sleeper = ({x, y, s = 1, bob = 1}: {x: number; y: number; s?: number; bob?: number}) => {
  const f = useCurrentFrame();
  const br = Math.sin(f * 0.11) * 4 * bob;
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      <rect x={-306} y={88} width={26} height={52} rx={10} fill={P.card} />
      <rect x={280} y={88} width={26} height={52} rx={10} fill={P.card} />
      <rect x={-345} y={30} width={690} height={66} rx={24} fill={P.card} />
      <rect x={-345} y={30} width={690} height={24} rx={12} fill={P.bgLite} opacity={0.7} />
      <rect x={-320} y={-38} width={178} height={66} rx={28} fill={P.cream} />
      <path d={`M -232 ${-44 - br} L 296 ${-44 - br} Q 336 ${-44 - br} 336 30 L -232 30 Z`} fill={P.teal} />
      <path d={`M -232 ${-44 - br} L 336 ${-44 - br} L 336 ${-20 - br * 0.6} L -232 ${-20 - br * 0.6} Z`}
        fill={P.tealD} opacity={0.55} />
      <g transform={`translate(-206,${-76 - br * 0.5})`}>
        <circle r={46} fill={C.skin} />
        <path d="M -46 -8 A 46 46 0 0 1 46 -8 L 32 -30 Q 0 -44 -32 -30 Z" fill={P.hair} />
        <path d="M -22 2 q 7 7 14 0 M 10 2 q 7 7 14 0" stroke={P.ink} strokeWidth={4.4}
          fill="none" strokeLinecap="round" />
        <ellipse cx={1} cy={20} rx={7} ry={4.4} fill={C.skinDark} />
        <circle cx={-31} cy={14} r={6} fill={P.rose} opacity={0.55} />
      </g>
    </g>
  );
};

/* ---------- 梦泡（圆角矩形 + 拖尾小圆） ---------- */
const Bub = ({x, y, w = 320, h = 160, s = 1, op = 1, tail = 'left', children}: {
  x: number; y: number; w?: number; h?: number; s?: number; op?: number;
  tail?: 'left' | 'right'; children?: React.ReactNode;
}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
    <rect x={-w / 2} y={-h / 2} width={w} height={h} rx={h / 2.4} fill={P.bgLite}
      stroke={P.vioD} strokeWidth={5} />
    {tail === 'left' ? (
      <g>
        <circle cx={-w / 2 + 24} cy={h / 2 + 30} r={13} fill={P.bgLite} stroke={P.vioD} strokeWidth={4} />
        <circle cx={-w / 2 - 6} cy={h / 2 + 62} r={8.6} fill={P.bgLite} stroke={P.vioD} strokeWidth={4} />
      </g>
    ) : (
      <g>
        <circle cx={w / 2 - 24} cy={h / 2 + 30} r={13} fill={P.bgLite} stroke={P.vioD} strokeWidth={4} />
        <circle cx={w / 2 + 6} cy={h / 2 + 62} r={8.6} fill={P.bgLite} stroke={P.vioD} strokeWidth={4} />
      </g>
    )}
    {children}
  </g>
);

/* ---------- 小图标 ---------- */
const IcoClock = ({x, y, s = 1, col = P.amber}: {x: number; y: number; s?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-14} y={-60} width={28} height={16} rx={7} fill={col} />
    <circle r={42} fill={P.cream} stroke={col} strokeWidth={8} />
    <line x1={0} y1={0} x2={0} y2={-25} stroke={P.ink} strokeWidth={6} strokeLinecap="round" />
    <line x1={0} y1={0} x2={17} y2={8} stroke={P.ink} strokeWidth={6} strokeLinecap="round" />
    <rect x={-34} y={40} width={13} height={20} rx={6} fill={col} transform="rotate(20 -27 50)" />
    <rect x={21} y={40} width={13} height={20} rx={6} fill={col} transform="rotate(-20 27 50)" />
  </g>
);

const IcoDoc = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-32} y={-42} width={64} height={84} rx={8} fill={P.cream} />
    <path d="M 18 -42 L 32 -28 L 18 -28 Z" fill={P.muted} />
    {[-20, -4, 12].map((yy, i) => (
      <line key={i} x1={-19} y1={yy} x2={18 - i * 7} y2={yy} stroke={P.card} strokeWidth={7} strokeLinecap="round" />
    ))}
    <circle cx={15} cy={24} r={14} fill={P.coral} />
    <text x={15} y={31} textAnchor="middle" fontSize={19} fill="#fff" fontFamily={FONT_HEAD} fontWeight={900}>!</text>
  </g>
);

const IcoMic = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-13} y={-50} width={26} height={54} rx={13} fill={P.rose} />
    <path d="M -28 -18 A 28 28 0 0 0 28 -18" fill="none" stroke={P.cream} strokeWidth={7} strokeLinecap="round" />
    <line x1={0} y1={10} x2={0} y2={38} stroke={P.cream} strokeWidth={7} strokeLinecap="round" />
    <line x1={-20} y1={42} x2={20} y2={42} stroke={P.cream} strokeWidth={7} strokeLinecap="round" />
  </g>
);

const IcoTrain = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-50} y={-42} width={100} height={74} rx={16} fill={P.sky} />
    <rect x={-38} y={-30} width={34} height={24} rx={6} fill={P.cream} />
    <rect x={4} y={-30} width={34} height={24} rx={6} fill={P.cream} />
    <rect x={-50} y={6} width={100} height={26} rx={10} fill={P.skyD} />
    <circle cx={-25} cy={38} r={10} fill={P.ink} />
    <circle cx={25} cy={38} r={10} fill={P.ink} />
  </g>
);

const Runner = ({x, y, s = 1, col = P.cream, flip = 1}: {
  x: number; y: number; s?: number; col?: string; flip?: number;
}) => {
  const f = useCurrentFrame();
  const a = Math.sin(f * 0.65) * 16;
  return (
    <g transform={`translate(${x},${y}) scale(${flip * s},${s})`}>
      <circle cx={0} cy={-52} r={15} fill={col} />
      <line x1={0} y1={-37} x2={0} y2={-4} stroke={col} strokeWidth={11} strokeLinecap="round" />
      <line x1={0} y1={-30} x2={24 + a * 0.4} y2={-14} stroke={col} strokeWidth={8} strokeLinecap="round" />
      <line x1={0} y1={-30} x2={-22 - a * 0.4} y2={-16} stroke={col} strokeWidth={8} strokeLinecap="round" />
      <line x1={0} y1={-4} x2={20 + a} y2={32} stroke={col} strokeWidth={9} strokeLinecap="round" />
      <line x1={0} y1={-4} x2={-18 - a} y2={32} stroke={col} strokeWidth={9} strokeLinecap="round" />
    </g>
  );
};

const FishSmall = ({x, y, s = 1, col = P.vio, wig = 0}: {
  x: number; y: number; s?: number; col?: string; wig?: number;
}) => {
  const f = useCurrentFrame();
  const w = Math.sin(f * 0.3 + wig) * 4;
  return (
    <g transform={`translate(${x},${y + w}) scale(${s})`}>
      <ellipse rx={24} ry={14} fill={col} />
      <polygon points="20,0 38,-11 38,11" fill={col} />
      <circle cx={-12} cy={-3} r={3} fill={P.ink} />
    </g>
  );
};

const Bottle = ({x, y, s = 1, rot = 0}: {x: number; y: number; s?: number; rot?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`}>
    <rect x={-8} y={-76} width={16} height={26} rx={5} fill={P.tealD} />
    <path d="M -8 -52 L -22 -30 L -22 42 Q -22 56 -10 56 L 10 56 Q 22 56 22 42 L 22 -30 L 8 -52 Z"
      fill={'#3f7d6a'} />
    <rect x={-22} y={-6} width={44} height={34} rx={6} fill={P.cream} />
    <text x={0} y={18} textAnchor="middle" fontSize={19} fill={P.ink} fontFamily={FONT_HEAD} fontWeight={800}>酒</text>
  </g>
);

const Scissors = ({x, y, s = 1, rot = 0}: {x: number; y: number; s?: number; rot?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`}>
    <line x1={-38} y1={-28} x2={30} y2={16} stroke={P.muted} strokeWidth={9} strokeLinecap="round" />
    <line x1={-38} y1={28} x2={30} y2={-16} stroke={P.muted} strokeWidth={9} strokeLinecap="round" />
    <circle cx={-46} cy={-34} r={13} fill="none" stroke={P.coral} strokeWidth={7} />
    <circle cx={-46} cy={34} r={13} fill="none" stroke={P.coral} strokeWidth={7} />
  </g>
);

/* ---------- 曲线工具：Chaikin 平滑 + 进度截断 ---------- */
const chaikin = (pts: [number, number][], n = 2): [number, number][] => {
  let p = pts;
  for (let k = 0; k < n; k++) {
    const q: [number, number][] = [p[0]];
    for (let i = 0; i < p.length - 1; i++) {
      const [ax, ay] = p[i];
      const [bx, by] = p[i + 1];
      q.push([ax * 0.72 + bx * 0.28, ay * 0.72 + by * 0.28]);
      q.push([ax * 0.28 + bx * 0.72, ay * 0.28 + by * 0.72]);
    }
    q.push(p[p.length - 1]);
    p = q;
  }
  return p;
};

const Curve = ({pts, prog, col, lw = 6, dash}: {
  pts: [number, number][]; prog: number; col: string; lw?: number; dash?: string;
}) => {
  const sm = chaikin(pts);
  const uu = clamp01(prog) * (sm.length - 1);
  const seg = Math.floor(uu);
  const out = sm.slice(0, seg + 1);
  if (seg < sm.length - 1) {
    const [ax, ay] = sm[seg];
    const [bx, by] = sm[seg + 1];
    const t = uu - seg;
    out.push([ax + (bx - ax) * t, ay + (by - ay) * t]);
  }
  if (out.length < 2) return null;
  return (
    <polyline points={out.map((p) => p.join(',')).join(' ')} fill="none" stroke={col}
      strokeWidth={lw} strokeLinecap="round" strokeLinejoin="round" strokeDasharray={dash} />
  );
};

/* ============ 标题卡：床上小人 + 装满焦虑的梦泡 ============ */
const TitleCard = ({data}: {data: any}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 13, mass: 0.7}});
  const o = interpolate(f, [0, 8], [0, 1]);
  const pA = Math.max(0.001, spring({frame: f - 4, fps, config: {damping: 11, stiffness: 140}}));
  const pB = Math.max(0.001, spring({frame: f - 16, fps, config: {damping: 11, stiffness: 140}}));
  return (
    <AbsoluteFill>
      <Backdrop top={P.bg} bottom={P.bg} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={22} from={0} dur={24} />
        <Moon x={1690} y={140} scale={0.62} />
        {/* 床上的小人 */}
        <g transform={`translate(960,580) scale(${pA}) translate(-960,-580)`} opacity={clamp01(pA)}>
          <ellipse cx={960} cy={690} rx={560} ry={60} fill={P.bgDeep} opacity={0.8} />
          <Sleeper x={960} y={580} s={1.06} />
          <Zzz x={630} y={430} k={0.75} col={P.teal} />
        </g>
        {/* 焦虑满满的梦泡 */}
        <g transform={`translate(1090,270) scale(${pB}) translate(-1090,-270)`} opacity={clamp01(pB)}>
          <Bub x={1090} y={270} w={620} h={230} s={1} tail="left">
            <IcoClock x={-170} y={6} s={0.95} col={P.coral} />
            <IcoDoc x={-10} y={8} s={0.95} />
            <IcoMic x={140} y={8} s={0.95} />
            <text x={232} y={-26} fontSize={58} fill={P.coral} fontFamily={FONT_HEAD}
              fontWeight={900} transform="rotate(10 232 -26)">!</text>
            <text x={-236} y={-58} fontSize={30} fill={P.muted} fontFamily={FONT_HEAD}>全是焦虑素材</text>
          </Bub>
        </g>
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 480}}>
          <div style={{color: P.teal, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 22}}>
            冷知识 · 第 {data.episode} 期
          </div>
          <div style={{color: C.text, fontSize: 80, fontWeight: 900, fontFamily: FONT_HEAD, letterSpacing: 6}}>
            {data.title}
          </div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ============ S0 睡眠结构被切碎 · 捞鱼比喻 ============ */
const HYPN: [number, number][] = [
  [280, 210], [330, 320], [360, 355], [400, 425], [440, 485], [560, 485], [620, 425], [660, 355],
  [700, 285], [800, 285], [840, 355], [880, 425], [940, 485], [990, 425], [1030, 355],
  [1050, 285], [1160, 285], [1200, 355], [1240, 405], [1290, 285], [1430, 285], [1470, 320], [1520, 210],
];
const REM_W: [number, number][] = [[700, 800], [1050, 1160], [1290, 1430]];
const STAGE_Y = {醒: 210, REM: 285, N1: 355, N2: 425, 深睡: 485};

const Spike = ({x, at, label}: {x: number; at: number; label?: string}) => {
  const f = useCurrentFrame();
  const k = clamp01((f - at) / 14);
  if (k <= 0) return null;
  const yTop = 212, yBot = 300;
  return (
    <g>
      <line x1={x} y1={yBot} x2={x} y2={yBot - (yBot - yTop) * k} stroke={P.coral}
        strokeWidth={6} strokeLinecap="round" />
      {k > 0.7 && (
        <g>
          <polygon points={`${x - 9},${yTop + 16} ${x + 9},${yTop + 16} ${x},${yTop - 2}`} fill={P.coral} />
          <circle cx={x} cy={yTop - 30} r={24} fill={P.vio} opacity={0.95} />
          <text x={x} y={yTop - 22} textAnchor="middle" fontSize={25} fill={P.ink}
            fontFamily={FONT_HEAD} fontWeight={900}>梦</text>
          {label && (
            <text x={x + 40} y={yTop + 8} fontSize={22} fill={P.coral}
              fontFamily={FONT_HEAD} fontWeight={800}>{label}</text>
          )}
        </g>
      )}
    </g>
  );
};

const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);
  const drawK = clamp01((f - 10) / 80);
  const remK = clamp01((f - b1 + 14) / 24);
  const fishPop = Math.max(0.001, spring({frame: f - b2 - 40, fps, config: {damping: 12, stiffness: 110}}));
  const castStart = b2 + 60;
  const casts = f <= castStart ? 0 : Math.min(5, Math.floor((f - castStart) / 26) + 1);
  const netDip = casts > 0 ? Math.abs(Math.sin((f - castStart) * 0.22)) * 44 : 0;
  const caught = Math.min(casts, 4);

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={0} name={sec.name} accent={P.teal} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 上：一夜睡眠结构图 */}
        <rect x={120} y={140} width={1680} height={410} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <text x={156} y={188} fill={P.muted} fontSize={24} fontFamily={FONT_HEAD}>
          压力夜 · 睡眠被切成碎块
        </text>
        {/* 分期横线 + 标签 */}
        {([['醒', STAGE_Y.醒, P.coral], ['REM', STAGE_Y.REM, P.teal], ['N1', STAGE_Y.N1, P.muted],
           ['N2', STAGE_Y.N2, P.muted], ['深睡', STAGE_Y.深睡, P.muted]] as [string, number, string][]).map(([nm, yy, col]) => (
          <g key={nm}>
            <line x1={272} y1={yy} x2={1560} y2={yy} stroke={P.card} strokeWidth={3} strokeDasharray="6 12" />
            <text x={250} y={yy + 8} textAnchor="end" fontSize={23} fill={col}
              fontFamily={FONT_HEAD} fontWeight={nm === 'REM' ? 900 : 400}>{nm}</text>
          </g>
        ))}
        {/* REM 带高亮 */}
        {REM_W.map(([a, b], i) => (
          <g key={i} opacity={remK}>
            <rect x={a} y={269} width={b - a} height={32} rx={16} fill={P.teal} opacity={0.3} />
            <rect x={a + 8} y={278} width={b - a - 16} height={14} rx={7} fill={P.teal} />
          </g>
        ))}
        {/* 结构折线 */}
        <Curve pts={HYPN} prog={drawK} col={P.cream} lw={5.5} />
        {/* 夜里醒来的尖峰：在 REM 里醒 → 捞到一个梦 */}
        <Spike x={750} at={b1 + 20} label="醒在做梦窗口" />
        <Spike x={1105} at={b2 - 30} />
        <Spike x={1360} at={b2 - 4} />
        {/* 时间刻度 */}
        {[['23点', 300], ['1点', 620], ['3点', 940], ['5点', 1250], ['7点', 1500]].map(([t, x]) => (
          <text key={t as string} x={x as number} y={536} textAnchor="middle" fontSize={21}
            fill={P.muted} fontFamily={FONT_HEAD}>{t}</text>
        ))}

        {/* 下：捞鱼比喻 */}
        <rect x={120} y={580} width={1680} height={300} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <g opacity={clamp01(drawK)}>
          <text x={410} y={640} textAnchor="middle" fill={P.cream} fontSize={27}
            fontFamily={FONT_HEAD} fontWeight={800}>夜里真实做的梦 · 产量没变</text>
          {/* 鱼塘 */}
          <ellipse cx={410} cy={740} rx={255} ry={78} fill={P.water} opacity={0.85} />
          <ellipse cx={410} cy={722} rx={255} ry={40} fill={P.skyD} opacity={0.5} />
          <FishSmall x={300} y={726} s={0.8} col={P.vio} wig={0} />
          <FishSmall x={380} y={762} s={0.72} col={P.sky} wig={2} />
          <FishSmall x={462} y={720} s={0.85} col={P.vio} wig={4} />
          <FishSmall x={520} y={756} s={0.7} col={P.rose} wig={1} />
          <FishSmall x={345} y={745} s={0.66} col={P.sky} wig={3} />
          <FishSmall x={480} y={742} s={0.75} col={P.rose} wig={5} />
          <text x={410} y={858} textAnchor="middle" fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>
            鱼 ≈ 6 条 · 固定
          </text>
        </g>
        {fishPop > 0.02 && (
          <g opacity={clamp01(fishPop)} transform={`translate(0,${(1 - fishPop) * 40})`}>
            {/* 网兜 */}
            <g transform={`translate(880,${700 + netDip})`}>
              <line x1={-96} y1={-74} x2={-18} y2={-12} stroke={P.cream} strokeWidth={9} strokeLinecap="round" />
              <circle cx={14} cy={0} r={46} fill="none" stroke={P.amber} strokeWidth={8} />
              {[-18, 0, 18].map((dx) => (
                <line key={dx} x1={dx + 10} y1={-36} x2={dx + 10} y2={36} stroke={P.amber}
                  strokeWidth={3.4} opacity={0.75} />
              ))}
              {[-18, 0, 18].map((dy) => (
                <line key={dy} x1={-24} y1={dy} x2={52} y2={dy} stroke={P.amber}
                  strokeWidth={3.4} opacity={0.75} />
              ))}
            </g>
            <text x={880} y={640} textAnchor="middle" fontSize={27} fill={P.amber}
              fontFamily={FONT_HEAD} fontWeight={800}>醒来 = 撒网 ×{Math.max(casts, 1)}</text>
            {/* 箭头 */}
            <line x1={700} y1={740} x2={790} y2={740} stroke={P.muted} strokeWidth={6} strokeLinecap="round" />
            <polygon points="790,740 772,730 772,750" fill={P.muted} />
            <line x1={960} y1={740} x2={1060} y2={740} stroke={P.muted} strokeWidth={6} strokeLinecap="round" />
            <polygon points="1060,740 1042,730 1042,750" fill={P.muted} />
            {/* 桶 */}
            <g transform="translate(1200,748)">
              <path d="M -78 -62 L 78 -62 L 56 62 L -56 62 Z" fill={P.card} stroke={P.muted} strokeWidth={5} />
              {[0, 1, 2, 3].map((i) => i < caught && (
                <text key={i} x={-42 + (i % 2) * 56} y={-8 + Math.floor(i / 2) * 40}
                  fontSize={26} fill={P.vio} fontFamily={FONT_HEAD} fontWeight={900}>梦</text>
              ))}
            </g>
            <text x={1200} y={858} textAnchor="middle" fontSize={22} fill={P.vio}
              fontFamily={FONT_HEAD} fontWeight={800}>记住的梦 ↑</text>
            {/* 结论 */}
            <g transform="translate(1560,720)">
              <text x={0} y={-20} textAnchor="middle" fontSize={30} fill={P.cream}
                fontFamily={FONT_HEAD} fontWeight={800}>不是梦变多</text>
              <text x={0} y={30} textAnchor="middle" fontSize={30} fill={P.coral}
                fontFamily={FONT_HEAD} fontWeight={900}>是捞到的次数变多</text>
            </g>
          </g>
        )}
      </svg>
      <Chip text="微觉醒 ×N" x={1500} y={66} at={b1 + 30} accent={P.coral} />
      <Stamp text="回忆概率问题" x={560} y={86} at={b2 + 90} color={P.tealD} size={40} rot={-3} />
    </FadeIn>
  );
};

/* ============ S1 皮质醇节律被压平 ============ */
const NORMAL_PT: [number, number][] = [
  [220, 310], [380, 378], [560, 455], [760, 530], [960, 575], [1180, 600], [1360, 545], [1520, 430], [1700, 330],
];
const STRESS_PT: [number, number][] = [
  [220, 355], [380, 415], [560, 455], [760, 472], [960, 466], [1180, 458], [1330, 470], [1440, 440], [1560, 455], [1700, 400],
];

const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);
  const axisK = clamp01((f - 8) / 30);
  const normK = clamp01((f - 16) / 80);
  const stressK = clamp01((f - b1 - 10) / 90);
  const markK = clamp01((f - b1 - 60) / 22);
  const loopK = clamp01((f - b2 - 6) / 26);
  const pulse = 0.5 + Math.abs(Math.sin(f * 0.5)) * 0.5;

  return (
    <FadeIn>
      <Backdrop top={P.bgDeep} bottom={P.bgDeep} />
      <SceneTag index={1} name={sec.name} accent={P.amber} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <rect x={120} y={140} width={1680} height={560} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <text x={960} y={192} textAnchor="middle" fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
          一天里的皮质醇曲线 · 「清醒激素」本该夜里退场
        </text>
        {/* 夜间底纹 */}
        <rect x={940} y={220} width={720} height={420} fill={P.sky} opacity={0.05} />
        <Moon x={968} y={252} scale={0.28} />
        {/* 坐标轴 */}
        <g opacity={axisK}>
          <line x1={200} y1={220} x2={200} y2={640} stroke={P.muted} strokeWidth={4} />
          <polygon points="200,214 192,234 208,234" fill={P.muted} />
          <line x1={200} y1={640} x2={1720} y2={640} stroke={P.muted} strokeWidth={4} />
          <polygon points="1726,640 1706,632 1706,648" fill={P.muted} />
          <text x={168} y={280} textAnchor="end" fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>高</text>
          <text x={168} y={620} textAnchor="end" fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>低</text>
          <text x={160} y={450} textAnchor="middle" fontSize={23} fill={P.amber} fontFamily={FONT_HEAD}
            fontWeight={800} transform="rotate(-90 160 450)">皮质醇</text>
        </g>
        {/* X 轴刻度 */}
        {[['清晨', 300], ['中午', 540], ['傍晚', 780], ['入睡', 940], ['午夜', 1180], ['凌晨', 1420], ['清晨', 1660]].map(([t, x]) => (
          <g key={String(t) + x}>
            <line x1={x as number} y1={640} x2={x as number} y2={652} stroke={P.muted} strokeWidth={3} />
            <text x={x as number} y={680} textAnchor="middle" fontSize={21} fill={P.muted}
              fontFamily={FONT_HEAD}>{t}</text>
          </g>
        ))}
        {/* 正常曲线 */}
        <Curve pts={NORMAL_PT} prog={normK} col={P.sky} lw={7} />
        {normK > 0.8 && (
          <g>
            <circle cx={1180} cy={600} r={10} fill={P.sky} />
            <text x={1740} y={310} textAnchor="end" fontSize={23} fill={P.sky}
              fontFamily={FONT_HEAD} fontWeight={800}>
              — 正常节律 · 午夜降到最低
            </text>
          </g>
        )}
        {/* 压力曲线 */}
        {stressK > 0 && <Curve pts={STRESS_PT} prog={stressK} col={P.amber} lw={7} />}
        {stressK > 0.8 && (
          <g>
            <circle cx={1180} cy={458} r={10} fill={P.amber} />
            <text x={1740} y={255} textAnchor="end" fontSize={23} fill={P.amber}
              fontFamily={FONT_HEAD} fontWeight={800}>
              — 压力节律 · 被压平
            </text>
          </g>
        )}
        {/* 午夜低谷对比标注 */}
        {markK > 0 && (
          <g opacity={markK}>
            <line x1={1180} y1={470} x2={1180} y2={592} stroke={P.coral} strokeWidth={5} strokeDasharray="8 8" />
            <polygon points="1180,470 1170,490 1190,490" fill={P.coral} transform="rotate(180 1180 480)" />
            <g transform={`translate(1180,350)`}>
              <rect x={-190} y={-34} width={380} height={56} rx={16} fill={P.bgDeep}
                stroke={P.coral} strokeWidth={4} />
              <text x={0} y={4} textAnchor="middle" fontSize={25} fill={P.coral}
                fontFamily={FONT_HEAD} fontWeight={800}>该睡觉时不肯退 → 睡眠浅</text>
            </g>
            {/* 凌晨醒小人睁眼 */}
            <g transform="translate(1420,566)">
              <rect x={-70} y={16} width={140} height={24} rx={11} fill={P.card} />
              <circle cx={-30} cy={0} r={22} fill={C.skin} />
              <circle cx={-36} cy={-4} r={3.6 * pulse + 1.5} fill={P.ink} />
              <circle cx={-22} cy={-4} r={3.6 * pulse + 1.5} fill={P.ink} />
              <text x={6} y={8} fontSize={21} fill={P.amber} fontFamily={FONT_HEAD} fontWeight={800}>凌晨易醒</text>
            </g>
          </g>
        )}
        {/* 下：互相喂养的循环 */}
        <rect x={120} y={730} width={1680} height={150} rx={20} fill={P.panel}
          stroke={loopK > 0 ? P.coral : P.card} strokeWidth={4} />
        <text x={196} y={818} fontSize={25} fill={P.muted} fontFamily={FONT_HEAD}>夜里它俩互相喂对方：</text>
        {loopK > 0 && (
          <g opacity={loopK} transform={`translate(0,${(1 - loopK) * 30})`}>
            <rect x={560} y={770} width={330} height={68} rx={18} fill={P.bgDeep} stroke={P.amber} strokeWidth={4} />
            <text x={725} y={815} textAnchor="middle" fontSize={27} fill={P.amber}
              fontFamily={FONT_HEAD} fontWeight={800}>皮质醇夜里偏高</text>
            <rect x={1180} y={770} width={360} height={68} rx={18} fill={P.bgDeep} stroke={P.teal} strokeWidth={4} />
            <text x={1360} y={815} textAnchor="middle" fontSize={27} fill={P.teal}
              fontFamily={FONT_HEAD} fontWeight={800}>睡眠变浅·凌晨易醒</text>
            {/* 双向弧箭头 */}
            <path d="M 900 786 Q 1035 744 1170 786" fill="none" stroke={P.cream} strokeWidth={6}
              strokeLinecap="round" />
            <polygon points="1170,786 1148,776 1152,796" fill={P.cream} />
            <path d="M 1170 822 Q 1035 864 900 822" fill="none" stroke={P.cream} strokeWidth={6}
              strokeLinecap="round" />
            <polygon points="900,822 922,812 918,832" fill={P.cream} />
            <text x={1660} y={820} textAnchor="end" fontSize={25} fill={P.coral}
              fontFamily={FONT_HEAD} fontWeight={800}>→ 谁都睡不好</text>
          </g>
        )}
      </svg>
      <Chip text="皮质醇 = 清醒激素" x={1330} y={66} at={30} accent={P.amber} />
      <Stamp text="节律被压平了" x={580} y={86} at={b1 + 80} color={P.coralD} size={40} rot={-3} />
    </FadeIn>
  );
};

/* ============ S2 REM 欠债 → 反弹 ============ */
const BLOCK_T = ['N', 'R', 'N', 'R', 'N', 'R', 'N'];
const BLOCK_X = [36, 96, 156, 216, 276, 336, 396];
const NIGHTS = [
  {x: 140, name: '正常的一夜', rem: [118, 118, 118], col: P.teal, note: 'REM 占比正常'},
  {x: 700, name: '压力夜', rem: [46, 52, 42], col: P.tealD, note: 'REM 被切碎 · 欠了债'},
  {x: 1260, name: '反弹的夜', rem: [118, 118, 118], col: P.teal, note: '加倍回补 · 梦又密'},
];

const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);
  const pops = [
    Math.max(0.001, spring({frame: f - 12, fps, config: {damping: 12, stiffness: 120}})),
    Math.max(0.001, spring({frame: f - b1 - 4, fps, config: {damping: 12, stiffness: 120}})),
    Math.max(0.001, spring({frame: f - b1 - 64, fps, config: {damping: 12, stiffness: 120}})),
  ];
  const cutK = clamp01((f - b1 - 30) / 24);
  const bounce = spring({frame: f - b1 - 84, fps, config: {damping: 9, mass: 0.7, stiffness: 110}});
  const chainK = clamp01((f - b2 - 8) / 26);

  const night = (nx: number, name: string, remH: number[], col: string, note: string, mode: 'norm' | 'cut' | 'bounce', pk: number) => (
    <g transform={`translate(${nx + 250},380) scale(${pk}) translate(${-(nx + 250)},-380)`}
      opacity={clamp01(pk * 2)}>
      <rect x={nx} y={160} width={500} height={470} rx={20} fill={P.panel}
        stroke={mode === 'bounce' ? P.teal : P.card} strokeWidth={4} />
      <text x={nx + 250} y={212} textAnchor="middle" fontSize={26} fill={mode === 'cut' ? P.coral : P.cream}
        fontFamily={FONT_HEAD} fontWeight={800}>{name}</text>
      {/* 睡眠积木天际线 */}
      <line x1={nx + 30} y1={520} x2={nx + 470} y2={520} stroke={P.muted} strokeWidth={4} />
      {BLOCK_T.map((t, i) => {
        const isR = t === 'R';
        let h = isR ? remH[Math.floor(i / 2)] : 68;
        if (mode === 'bounce' && isR) h = 46 + (178 - 46) * Math.max(0, bounce);
        const bx = nx + BLOCK_X[i];
        const top = 520 - h;
        return (
          <g key={i}>
            <rect x={bx} y={top} width={46} height={h} rx={10}
              fill={isR ? col : P.card} stroke={isR ? P.cream : 'none'} strokeWidth={isR ? 3 : 0}
              opacity={isR ? 1 : 0.85} />
            {isR && (
              <text x={bx + 23} y={Math.max(top - 12, 240)} textAnchor="middle" fontSize={19}
                fill={P.teal} fontFamily={FONT_HEAD} fontWeight={900}>REM</text>
            )}
            {/* 被切碎的裂缝 */}
            {mode === 'cut' && isR && cutK > 0 && (
              <g opacity={cutK} stroke={P.coral} strokeWidth={5} strokeLinecap="round">
                <path d={`M ${bx + 4} ${top + h * 0.32} l 12 10 l -14 8 l 14 9 l -10 8`} fill="none" />
                <path d={`M ${bx + 40} ${top + h * 0.6} l -12 9 l 12 8`} fill="none" />
              </g>
            )}
          </g>
        );
      })}
      {mode === 'cut' && cutK > 0 && (
        <g transform={`translate(${nx + 250},300)`} opacity={cutK}>
          <Scissors x={0} y={0} s={1.15} rot={Math.sin(f * 0.6) * 14} />
        </g>
      )}
      {mode === 'bounce' && clamp01(bounce) > 0.4 && (
        <g transform={`translate(${nx + 250},262)`}>
          <rect x={-96} y={-28} width={192} height={52} rx={15} fill={P.teal} />
          <text x={0} y={8} textAnchor="middle" fontSize={29} fill={P.ink}
            fontFamily={FONT_HEAD} fontWeight={900}>REM 反弹</text>
        </g>
      )}
      <text x={nx + 250} y={586} textAnchor="middle" fontSize={23}
        fill={mode === 'norm' ? P.muted : mode === 'cut' ? P.coral : P.teal}
        fontFamily={FONT_HEAD} fontWeight={800}>{note}</text>
    </g>
  );

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={2} name={sec.name} accent={P.teal} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {night(NIGHTS[0].x, NIGHTS[0].name, NIGHTS[0].rem, NIGHTS[0].col, NIGHTS[0].note, 'norm', pops[0])}
        {night(NIGHTS[1].x, NIGHTS[1].name, NIGHTS[1].rem, NIGHTS[1].col, NIGHTS[1].note, 'cut', pops[1])}
        {night(NIGHTS[2].x, NIGHTS[2].name, NIGHTS[2].rem, NIGHTS[2].col, NIGHTS[2].note, 'bounce', pops[2])}
        {/* 夜与夜之间箭头 */}
        {pops[1] > 0.3 && (
          <g>
            <line x1={648} y1={390} x2={692} y2={390} stroke={P.coral} strokeWidth={7} strokeLinecap="round" />
            <polygon points="692,390 672,379 672,401" fill={P.coral} />
          </g>
        )}
        {pops[2] > 0.3 && (
          <g>
            <line x1={1208} y1={390} x2={1252} y2={390} stroke={P.teal} strokeWidth={7} strokeLinecap="round" />
            <polygon points="1252,390 1232,379 1232,401" fill={P.teal} />
          </g>
        )}
        {/* 底条：惊醒→再入睡→REM偏多 */}
        <rect x={140} y={668} width={1640} height={212} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <text x={176} y={716} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
          另一个助推器：每次惊醒后重新入睡，新周期的 REM 偏多
        </text>
        {chainK > 0 && (
          <g opacity={chainK} transform={`translate(0,${(1 - chainK) * 30})`}>
            {[
              {x: 400, txt: '半夜惊醒', col: P.coral, ico: '⚡'},
              {x: 880, txt: '重新入睡', col: P.sky, ico: 'z'},
              {x: 1360, txt: '新周期 REM 偏多 · 梦更密', col: P.teal, ico: '↻'},
            ].map((n, i) => (
              <g key={i}>
                <rect x={n.x - (i === 2 ? 260 : 150)} y={756} width={i === 2 ? 520 : 300} height={76}
                  rx={19} fill={P.bgDeep} stroke={n.col} strokeWidth={4} />
                <text x={n.x - (i === 2 ? 226 : 116)} y={806} fontSize={30} fill={n.col}
                  fontFamily={FONT_HEAD} fontWeight={900}>{n.ico}</text>
                <text x={n.x + (i === 2 ? 34 : 22)} y={806} textAnchor="middle" fontSize={27}
                  fill={P.cream} fontFamily={FONT_HEAD} fontWeight={800}>{n.txt}</text>
                {i < 2 && (
                  <g>
                    <line x1={n.x + (i === 2 ? 270 : 162)} y1={794} x2={n.x + (i === 2 ? 342 : 222)} y2={794}
                      stroke={P.muted} strokeWidth={6} strokeLinecap="round" />
                    <polygon points={`${n.x + (i === 2 ? 342 : 222)},794 ${n.x + (i === 2 ? 322 : 202)},782 ${n.x + (i === 2 ? 322 : 202)},806`}
                      fill={P.muted} />
                  </g>
                )}
              </g>
            ))}
          </g>
        )}
      </svg>
      <Chip text="稳态补偿" x={1480} y={66} at={b1 + 100} accent={P.teal} />
      <Stamp text="一半是记多 · 一半真多" x={480} y={86} at={b2 + 50} color={P.tealD} size={38} rot={-3} />
    </FadeIn>
  );
};

/* ============ S3 夜班脱敏手术：记忆过 REM 门拔刺 ============ */
const THORN_PT: [number, number, number][] = [
  [-30, -46, 0], [14, -46, 0], [-30, 46, 180], [14, 46, 180],
  [-62, -24, -90], [-62, 8, -90], [62, -24, 90], [62, 8, 90],
];

const MemCard = ({x, y, thorn, s = 1}: {x: number; y: number; thorn: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    {thorn > 0.02 && THORN_PT.map(([tx, ty, rot], i) => (
      <g key={i} transform={`translate(${tx},${ty}) rotate(${rot}) scale(${thorn})`}
        opacity={clamp01(thorn * 1.4)}>
        <polygon points="0,-22 -9,0 9,0" fill={P.coral} />
      </g>
    ))}
    <rect x={-58} y={-42} width={116} height={84} rx={12} fill={P.cream}
      stroke={thorn > 0.4 ? P.coral : P.teal} strokeWidth={5} />
    <circle cx={-18} cy={-8} r={5} fill={P.ink} />
    <circle cx={18} cy={-8} r={5} fill={P.ink} />
    {thorn > 0.4 ? (
      <path d="M -14 20 Q 0 8 14 20" stroke={P.ink} strokeWidth={5} fill="none" strokeLinecap="round" />
    ) : (
      <path d="M -14 12 Q 0 24 14 12" stroke={P.ink} strokeWidth={5} fill="none" strokeLinecap="round" />
    )}
    <rect x={-58} y={-42} width={116} height={20} rx={10} fill={thorn > 0.4 ? P.coral : P.teal} opacity={0.85} />
    <text x={0} y={-27} textAnchor="middle" fontSize={17} fill="#fff"
      fontFamily={FONT_HEAD} fontWeight={800}>{thorn > 0.4 ? '糟心记忆' : '已脱敏'}</text>
  </g>
);

const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const brainPop = Math.max(0.001, spring({frame: f - 12, fps, config: {damping: 12, stiffness: 110}}));
  const cardK = clamp01((f - b1 - 40) / 140);
  const cardX = 1000 + 700 * cardK;
  const thorn = 1 - clamp01((cardX - 1230) / 110);
  const lvl = 0.88 - 0.84 * clamp01((f - b1 - 50) / 110);
  const gateGlow = cardX > 1240 && cardX < 1440 ? 1 : 0;
  const tailK = clamp01((f - b1 - 200) / 24);
  const amygPulse = 1 + Math.sin(f * 0.35) * 0.12;

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={3} name={sec.name} accent={P.vio} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：大脑剖面 */}
        <rect x={120} y={140} width={720} height={560} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <text x={156} y={196} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
          REM 期 · 大脑在夜班（假说：过夜疗法）
        </text>
        <g transform={`translate(415,430) scale(${brainPop * 1.3})`} opacity={clamp01(brainPop)}>
          {/* 重放弧箭头 */}
          <path d="M -135 -128 A 190 130 0 0 1 135 -128" fill="none" stroke={P.amber}
            strokeWidth={6} strokeDasharray="10 12" opacity={0.9} />
          <polygon points="135,-128 112,-138 116,-114" fill={P.amber} />
          {/* 脑 */}
          <path d="M -120 30 Q -140 -40 -90 -82 Q -50 -116 20 -106 Q 95 -96 110 -30 Q 118 12 90 34 Q 60 56 0 54 L -60 52 Q -105 50 -120 30 Z"
            fill={P.vio} opacity={0.95} />
          <path d="M -96 -32 q 26 -26 56 -14 M -62 -72 q 30 -10 52 8 M 0 -90 q 34 4 48 30 M 28 -42 q 30 -8 44 14"
            stroke={P.vioD} strokeWidth={6} fill="none" strokeLinecap="round" />
          <path d="M 84 30 Q 110 42 104 72 L 88 88 L 64 74 L 70 46 Z" fill={P.vioD} />
          {/* 杏仁核 */}
          <g transform={`translate(-14,14) scale(${amygPulse})`}>
            <ellipse rx={24} ry={15} fill={P.coral} transform="rotate(-18)" />
          </g>
          <text x={-150} y={-140} fontSize={20} fill={P.amber} fontFamily={FONT_HEAD} fontWeight={800}>
            重放白天的情绪记忆 ↻
          </text>
        </g>
        {/* 杏仁核标注 */}
        <line x1={430} y1={452} x2={330} y2={560} stroke={P.coral} strokeWidth={3.4} />
        <text x={150} y={600} fontSize={23} fill={P.coral} fontFamily={FONT_HEAD} fontWeight={800}>
          杏仁核 · 情绪中枢
        </text>
        <text x={150} y={632} fontSize={21} fill={P.muted} fontFamily={FONT_HEAD}>REM 期本来就很活跃</text>
        {/* 去甲肾上腺素量表 */}
        <g>
          <rect x={700} y={330} width={48} height={250} rx={16} fill={P.bgDeep} stroke={P.card} strokeWidth={4} />
          <rect x={708} y={573 - 232 * Math.max(0.04, lvl)} width={32} height={232 * Math.max(0.04, lvl)}
            rx={10} fill={lvl > 0.3 ? P.coral : P.teal} />
          <text x={724} y={306} textAnchor="middle" fontSize={20} fill={P.muted} fontFamily={FONT_HEAD}>
            去甲肾上腺素
          </text>
          <text x={724} y={630} textAnchor="middle" fontSize={26}
            fill={lvl > 0.3 ? P.coral : P.teal} fontFamily={FONT_HEAD} fontWeight={900}>
            {lvl > 0.3 ? '压力调质' : '≈ 0'}
          </text>
        </g>
        <text x={156} y={668} fill={P.cream} fontSize={24} fontFamily={FONT_HEAD} fontWeight={800}>
          记忆被调出来 · 但情绪冲击被关了 ≈ 脱敏
        </text>

        {/* 右：REM 清洗门流水线 */}
        <rect x={880} y={140} width={920} height={560} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <text x={916} y={196} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
          把事件留下 · 把情绪的刺拔掉
        </text>
        {/* 传送带 */}
        <line x1={916} y1={492} x2={1764} y2={492} stroke={P.card} strokeWidth={8} strokeDasharray="2 22"
          strokeLinecap="round" />
        {[960, 1080, 1200, 1320, 1440, 1560, 1680].map((rx) => (
          <circle key={rx} cx={rx} cy={492} r={9} fill={P.card} />
        ))}
        {/* REM 门 */}
        <g transform="translate(1350,420)">
          <rect x={-90} y={-150} width={24} height={212} rx={11}
            fill={P.teal} opacity={gateGlow ? 1 : 0.85} />
          <rect x={66} y={-150} width={24} height={212} rx={11}
            fill={P.teal} opacity={gateGlow ? 1 : 0.85} />
          <rect x={-104} y={-172} width={208} height={40} rx={17} fill={P.teal} />
          <text x={0} y={-143} textAnchor="middle" fontSize={26} fill={P.ink}
            fontFamily={FONT_HEAD} fontWeight={900}>REM</text>
          {gateGlow > 0 && [0, 1, 2].map((i) => (
            <polygon key={i}
              points="0,-14 4,-4 14,0 4,4 0,14 -4,4 -14,0 -4,-4"
              fill={P.cream} transform={`translate(${96 + i * 30},${-70 + i * 62}) scale(${0.7 + 0.3 * Math.sin(f * 0.7 + i)})`} />
          ))}
          <text x={0} y={-204} textAnchor="middle" fontSize={22} fill={P.teal}
            fontFamily={FONT_HEAD} fontWeight={800}>脱敏处理门</text>
        </g>
        {/* 记忆卡 */}
        {cardK > 0 && <MemCard x={cardX} y={446} thorn={thorn} s={1.05} />}
        <text x={1000} y={560} textAnchor="middle" fontSize={23} fill={P.coral}
          fontFamily={FONT_HEAD} fontWeight={800}>带刺的情绪记忆</text>
        <text x={1700} y={560} textAnchor="middle" fontSize={23} fill={P.teal}
          fontFamily={FONT_HEAD} fontWeight={800}>事件留下 · 刺拔掉</text>
        {/* 假说诚实标注 */}
        {tailK > 0 && (
          <g opacity={tailK}>
            <rect x={916} y={596} width={848} height={76} rx={16} fill={P.bgDeep} stroke={P.vio} strokeWidth={3.4} />
            <text x={1340} y={645} textAnchor="middle" fontSize={24} fill={P.vio}
              fontFamily={FONT_HEAD} fontWeight={800}>
              「过夜疗法」仍是假说 · 证据在累积 · 但说得通
            </text>
          </g>
        )}
      </svg>
      <Chip text="Walker · 伯克利" x={1390} y={66} at={b1 + 10} accent={P.vio} />
      <Stamp text="大脑在赶工消化情绪" x={520} y={86} at={b1 + 130} color={P.vioD} size={40} rot={-3} />
    </FadeIn>
  );
};

/* ============ S4 焦虑变形版 + 恶性循环 + 酒精 ============ */
const CycleNode = ({x, y, col, k, children, label, lx, ly, anchor}: {
  x: number; y: number; col: string; k: number; children: React.ReactNode;
  label: string; lx: number; ly: number; anchor: 'middle' | 'start' | 'end';
}) => (
  <g transform={`translate(${x},${y}) scale(${Math.max(0.001, k)})`} opacity={clamp01(k * 2)}>
    <circle r={44} fill={P.bgDeep} stroke={col} strokeWidth={5} />
    {children}
    <text x={lx} y={ly} textAnchor={anchor} fontSize={23} fill={col}
      fontFamily={FONT_HEAD} fontWeight={800}>{label}</text>
  </g>
);

const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);
  const chipPops = [0, 1, 2].map((i) =>
    Math.max(0.001, spring({frame: f - 22 - i * 16, fps, config: {damping: 12, stiffness: 130}})));
  const arrowK = clamp01((f - 80) / 24);
  const bubPop = Math.max(0.001, spring({frame: f - 120, fps, config: {damping: 12, stiffness: 110}}));
  const nodeK = [0, 1, 2, 3].map((i) =>
    Math.max(0.001, spring({frame: f - b1 - 6 - i * 12, fps, config: {damping: 11, stiffness: 140}})));
  const wineK = clamp01((f - b1 - 90) / 26);
  const fixK = clamp01((f - b2 - 8) / 26);
  const ringFlow = (f - b1) * 1.2;

  const rad = (d: number) => (d * Math.PI) / 180;
  const arcArrow = (a0: number, a1: number, col: string, k: number) => {
    if (k <= 0) return null;
    const r = 106, cx = 560, cy = 700;
    const ex = cx + r * Math.cos(rad(a1)), ey = cy + r * Math.sin(rad(a1));
    return (
      <g opacity={k}>
        <path d={`M ${cx + r * Math.cos(rad(a0))} ${cy + r * Math.sin(rad(a0))} A ${r} ${r} 0 0 1 ${ex} ${ey}`}
          fill="none" stroke={col} strokeWidth={5} strokeDasharray={k < 1 ? '5 9' : undefined} />
        <g transform={`translate(${ex},${ey}) rotate(${a1 + 90})`}>
          <polygon points="0,0 -13,-9 -13,9" fill={col} />
        </g>
      </g>
    );
  };

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={4} name={sec.name} accent={P.coral} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 上：连续性假说 */}
        <rect x={120} y={140} width={1680} height={330} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <text x={156} y={192} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
          连续性假说 · 梦是白天焦虑的变形版
        </text>
        {[
          {x: 300, pop: chipPops[0], label: '赶 DDL', ico: <IcoDoc x={0} y={0} s={0.82} />},
          {x: 580, pop: chipPops[1], label: '当众讲话', ico: <IcoMic x={0} y={0} s={0.82} />},
          {x: 860, pop: chipPops[2], label: '赶不上车', ico: <IcoTrain x={0} y={-4} s={0.8} />},
        ].map((c, i) => (
          <g key={i} transform={`translate(${c.x},310) scale(${c.pop})`} opacity={clamp01(c.pop * 2)}>
            <circle r={58} fill={P.bgDeep} stroke={P.card} strokeWidth={4} />
            {c.ico}
            <text x={0} y={112} textAnchor="middle" fontSize={23} fill={P.cream}
              fontFamily={FONT_HEAD} fontWeight={800}>{c.label}</text>
          </g>
        ))}
        <text x={580} y={440} textAnchor="middle" fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>
          白天缠着你的素材
        </text>
        {/* 变形箭头 */}
        {arrowK > 0 && (
          <g opacity={arrowK}>
            <line x1={960} y1={310} x2={960 + 150 * arrowK} y2={310} stroke={P.coral}
              strokeWidth={9} strokeLinecap="round" />
            {arrowK > 0.9 && <polygon points="1124,310 1098,296 1098,324" fill={P.coral} />}
            <text x={1035} y={282} textAnchor="middle" fontSize={23} fill={P.coral}
              fontFamily={FONT_HEAD} fontWeight={800}>夜里变形</text>
          </g>
        )}
        {/* 梦泡：焦虑素材的变形版 = 巨钟追小人 */}
        {bubPop > 0.02 && (
          <g transform={`translate(1465,300) scale(${bubPop})`} opacity={clamp01(bubPop)}>
            <Bub x={0} y={0} w={580} h={230} s={1} tail="left">
              <IcoClock x={-150} y={10} s={1.15} col={P.coral} />
              {/* 追人的动感线 */}
              {[0, 1].map((i) => (
                <line key={i} x1={-62 - i * 18} y1={-6 + i * 18} x2={-96 - i * 18} y2={-6 + i * 18}
                  stroke={P.cream} strokeWidth={6} strokeLinecap="round" opacity={0.6 - i * 0.25} />
              ))}
              <Runner x={-8} y={44} s={0.78} />
              <text x={36} y={-36} fontSize={34} fill={P.coral} fontFamily={FONT_HEAD}
                fontWeight={900}>!</text>
              <IcoDoc x={150} y={30} s={0.6} />
              <text x={150} y={-38} textAnchor="middle" fontSize={22} fill={P.muted}
                fontFamily={FONT_HEAD}>稿子飞了</text>
              <text x={0} y={92} textAnchor="middle" fontSize={22} fill={P.rose}
                fontFamily={FONT_HEAD} fontWeight={800}>被 deadline 追 · 一个都跑不掉</text>
            </Bub>
          </g>
        )}

        {/* 下：恶性循环 + 酒精搅局 */}
        <rect x={120} y={500} width={1680} height={380} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <text x={156} y={552} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
          双向循环 · 梦多会反过来加重压力
        </text>
        {/* 四节点环 */}
        {arcArrow(-66, -26, P.muted, clamp01(nodeK[0] * 2 - 0.6))}
        {arcArrow(24, 64, P.muted, clamp01(nodeK[1] * 2 - 0.6))}
        {arcArrow(114, 154, P.muted, clamp01(nodeK[2] * 2 - 0.6))}
        {arcArrow(204, 244, P.muted, clamp01(nodeK[3] * 2 - 0.6))}
        <CycleNode x={560} y={594} col={P.coral} k={nodeK[0]} label="压力大" lx={-58} ly={-2} anchor="end">
          <polygon points="6,-22 -10,4 0,4 -6,22 12,-4 2,-4" fill={P.coral} />
        </CycleNode>
        <CycleNode x={666} y={700} col={P.vio} k={nodeK[1]} label="梦变多" lx={62} ly={8} anchor="start">
          <text x={0} y={11} textAnchor="middle" fontSize={30} fill={P.vio}
            fontFamily={FONT_HEAD} fontWeight={900}>梦</text>
        </CycleNode>
        <CycleNode x={560} y={806} col={P.amber} k={nodeK[2]} label="睡眠碎" lx={58} ly={24} anchor="start">
          <circle r={17} fill={P.cream} />
          <path d="M -6 -14 l 8 9 l -10 7 l 9 9" stroke={P.coral} strokeWidth={4} fill="none"
            strokeLinecap="round" />
        </CycleNode>
        <CycleNode x={454} y={700} col={P.sky} k={nodeK[3]} label="白天更累" lx={-62} ly={8} anchor="end">
          <circle r={16} fill={P.amber} opacity={0.55} />
          <path d="M -8 4 q 5 5 10 0 M -2 14 q 5 4 10 0" stroke={P.ink} strokeWidth={3.6}
            fill="none" strokeLinecap="round" transform="translate(4,-4)" />
        </CycleNode>
        {nodeK[3] > 0.5 && (
          <g>
            <text x={560} y={684} textAnchor="middle" fontSize={26} fill={P.cream}
              fontFamily={FONT_HEAD} fontWeight={900}>恶性</text>
            <text x={560} y={720} textAnchor="middle" fontSize={26} fill={P.cream}
              fontFamily={FONT_HEAD} fontWeight={900}>循环</text>
            {/* 环上流动的小点 */}
            {[0, 90, 180, 270].map((a0, i) => {
              const a = rad(((ringFlow * 3 + a0) % 360) - 90);
              return (
                <circle key={i} cx={560 + 106 * Math.cos(a)} cy={700 + 106 * Math.sin(a)}
                  r={7} fill={P.coral} opacity={0.85} />
              );
            })}
          </g>
        )}
        {/* 右：酒精搅局 */}
        {wineK > 0 && (
          <g opacity={wineK} transform={`translate(0,${(1 - wineK) * 40})`}>
            <Bottle x={1120} y={640} s={1.05} rot={-12 * wineK + Math.sin(f * 0.4) * 4 * wineK} />
            <text x={1120} y={740} textAnchor="middle" fontSize={24} fill={P.cream}
              fontFamily={FONT_HEAD} fontWeight={800}>喝酒助眠</text>
            <text x={1120} y={772} textAnchor="middle" fontSize={21} fill={P.coral}
              fontFamily={FONT_HEAD} fontWeight={800}>搅局者</text>
            {/* REM 先压后弹小图 */}
            <Curve pts={[[1250, 690], [1340, 700], [1410, 730], [1470, 725], [1530, 640], [1600, 610], [1700, 626]]}
              prog={wineK} col={P.coral} lw={6} />
            <text x={1400} y={782} textAnchor="middle" fontSize={20} fill={P.muted}
              fontFamily={FONT_HEAD}>前半夜压住 REM</text>
            <text x={1640} y={584} textAnchor="middle" fontSize={20} fill={P.coral}
              fontFamily={FONT_HEAD} fontWeight={800}>后半夜反弹更凶</text>
            <line x1={1200} y1={640} x2={1236} y2={668} stroke={P.muted} strokeWidth={4}
              strokeDasharray="4 8" />
          </g>
        )}
        {/* 结论：破局点 */}
        {fixK > 0 && (
          <g opacity={fixK} transform={`translate(0,${(1 - fixK) * 30})`}>
            <rect x={900} y={800} width={800} height={64} rx={17} fill={P.bgDeep} stroke={P.teal} strokeWidth={4} />
            <text x={1300} y={843} textAnchor="middle" fontSize={27} fill={P.teal}
              fontFamily={FONT_HEAD} fontWeight={900}>
              破局点：不是不做梦 · 是保住睡眠的连续性
            </text>
          </g>
        )}
      </svg>
      <Chip text="连续性假说" x={1440} y={66} at={60} accent={P.coral} />
      <Stamp text="睡个整觉·梦就回落" x={520} y={86} at={b2 + 40} color={P.tealD} size={40} rot={-3} />
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
        <Stars n={20} from={0} dur={22} />
        <Moon x={260} y={170} scale={0.55} opacity={0.9} />
        <g transform={`translate(960,660) scale(${pop}) translate(-960,-660)`} opacity={clamp01(pop)}>
          <ellipse cx={960} cy={760} rx={500} ry={56} fill={P.bgLite} opacity={0.7} />
          <Sleeper x={960} y={660} s={0.95} />
          <Bub x={1150} y={430} w={360} h={170} s={0.8} tail="left">
            <Moon x={-70} y={-6} scale={0.42} />
            <path d="M -20 30 Q 20 12 60 30" stroke={P.teal} strokeWidth={7} fill="none"
              strokeLinecap="round" />
            <path d="M -10 52 Q 26 36 62 52" stroke={P.teal} strokeWidth={7} fill="none"
              strokeLinecap="round" opacity={0.6} />
            <text x={80} y={-30} fontSize={26} fill={P.vio} fontFamily={FONT_HEAD} fontWeight={800}>平静版</text>
          </Bub>
          <Zzz x={700} y={520} k={0.75} col={P.teal} />
        </g>
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -330}}>
        <div style={{fontSize: 56, fontWeight: 800, color: C.text}}>梦多不是故障 · 是大脑在值夜班</div>
        <div style={{fontSize: 36, color: P.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E238 = ({data = tl}: {data?: any}) => {
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

      <Sequence from={Math.round((data.totalSec - data.endCardSec) * fps)}
        durationInFrames={Math.round(data.endCardSec * fps)}>
        <EndCard />
      </Sequence>

      <Subtitles data={data} />
      <ProgressBar />
    </AbsoluteFill>
  );
};
