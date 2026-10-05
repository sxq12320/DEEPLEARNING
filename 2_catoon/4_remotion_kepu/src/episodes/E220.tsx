import {AbsoluteFill, Audio, Sequence, interpolate, interpolateColors, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01, rand} from '../lib/style';
import {Backdrop, Ripple, Stamp, SceneTag, Chip, Subtitles, ProgressBar, FadeIn, Sun} from '../lib/atoms';
import tl from '../timeline_220.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：雨林深绿 × 气生根棕 × 大地色（纯色平涂） ============ */
const P = {
  bg: '#236647',       // 雨林绿
  bgDeep: '#174a33',   // 深绿
  bgLite: '#2f7d55',   // 浅绿
  card: '#1b5238',     // 卡片底
  plate: '#f5eed3',    // 米盘
  ink: '#2a422e',      // 盘上深字
  canopy: '#63b36b',   // 树冠
  canopyD: '#3f8a4d',
  canopyHi: '#95d68b',
  leaf: '#79c77e',
  leafD: '#4a9954',
  bark: '#a4713e',     // 枝干棕
  barkD: '#71491f',
  root: '#cf9f63',     // 气生根浅棕
  rootD: '#9a7343',
  soil: '#5a422a',     // 大地棕
  soilD: '#3a2a18',
  host: '#8a9a6e',     // 宿主树冠（灰绿）
  hostD: '#5f6d49',
  hostBark: '#8c7a5c', // 宿主树干（灰棕）
  dead: '#9a8f7d',     // 枯死灰
  deadD: '#6e6557',
  fig: '#f2a356',      // 榕果橙
  hole: '#0e2418',     // 树洞深处
  sun: '#ffc94d',
  cream: '#f7f1dd',
  coral: '#ff7b6b',
  sky: '#5cc8ff',
  water: '#57c4f7',
  dark: '#122b1e',     // 剪影
};

/* ---------- 垂下气生根的曲线：k=下探进度 ---------- */
const hangPath = (x: number, y: number, len: number, k: number, wig: number, seed: number) => {
  const kk = clamp01(k);
  if (kk <= 0.002) return '';
  const n = 16;
  let d = `M ${x.toFixed(1)} ${y.toFixed(1)}`;
  for (let i = 1; i <= n; i++) {
    const t = (i / n) * kk;
    const px = x + Math.sin(t * 6.2 + seed * 2.4) * wig * t;
    d += ` L ${px.toFixed(1)} ${(y + t * len).toFixed(1)}`;
  }
  return d;
};
const hangTip = (x: number, y: number, len: number, k: number, wig: number, seed: number): [number, number] => {
  const kk = clamp01(k);
  return [x + Math.sin(kk * 6.2 + seed * 2.4) * wig * kk, y + kk * len];
};

/* ---------- 气生根：下垂线 + 根尖黏液滴 + 触土后可加粗 ---------- */
const AirRoot = ({x, y, len, k = 1, wig = 9, seed = 0, w = 7, thick = 0, mucus = true}: {
  x: number; y: number; len: number; k?: number; wig?: number; seed?: number; w?: number; thick?: number; mucus?: boolean;
}) => {
  const f = useCurrentFrame();
  const kk = clamp01(k);
  if (kk <= 0.01) return null;
  const [tx, ty] = hangTip(x, y, len, kk, wig, seed);
  const sw = w + thick * 22;
  const col = thick > 0.4 ? P.bark : P.root;
  const drip = Math.sin(f * 0.25 + seed) * 2.5;
  return (
    <g>
      <path d={hangPath(x, y, len, kk, wig, seed)} stroke={col} strokeWidth={sw} fill="none" strokeLinecap="round" />
      {kk < 0.995 && <circle cx={tx} cy={ty} r={w * 0.75} fill={col} />}
      {mucus && kk < 0.995 && <circle cx={tx} cy={ty + 10 + drip} r={3.4} fill={P.water} opacity={0.85} />}
    </g>
  );
};

/* ---------- 支柱根：落地后从土里向上长粗的柱子 ---------- */
const PropRoot = ({x, gy, h, w = 30, k = 1, col = P.bark}: {
  x: number; gy: number; h: number; w?: number; k?: number; col?: string;
}) => {
  const kk = Math.max(0.001, clamp01(k));
  return (
    <g transform={`translate(${x},${gy}) scale(1,${kk})`} opacity={kk > 0.02 ? 1 : 0}>
      <path d={`M ${-w * 0.66} 0 L ${-w * 0.3} ${-h} Q 0 ${-h - 12} ${w * 0.3} ${-h} L ${w * 0.66} 0 Q 0 ${w * 0.3} ${-w * 0.66} 0 Z`} fill={col} />
      <path d={`M ${-w * 0.14} ${-h * 0.12} L ${-w * 0.09} ${-h * 0.78} M ${w * 0.15} ${-h * 0.08} L ${w * 0.1} ${-h * 0.66}`}
        stroke={P.barkD} strokeWidth={4} strokeLinecap="round" opacity={0.5} />
    </g>
  );
};

/* ---------- 主干（带根盘） ---------- */
const Trunk = ({x, gy, h, w = 60, col = P.barkD}: {x: number; gy: number; h: number; w?: number; col?: string}) => (
  <g transform={`translate(${x},${gy})`}>
    <path d={`M ${-w * 0.9} 0 Q ${-w * 0.55} ${-h * 0.05} ${-w * 0.34} ${-h} L ${w * 0.34} ${-h} Q ${w * 0.55} ${-h * 0.05} ${w * 0.9} 0 Q 0 ${w * 0.36} ${-w * 0.9} 0 Z`} fill={col} />
    <path d={`M ${-w * 0.12} ${-h * 0.9} L ${-w * 0.05} ${-h * 0.22} M ${w * 0.14} ${-h * 0.82} L ${w * 0.07} ${-h * 0.3}`}
      stroke={P.bark} strokeWidth={6} strokeLinecap="round" opacity={0.55} />
  </g>
);

/* ---------- 树冠团：几团椭圆拼的平涂云 ---------- */
const Canopy = ({x, y, r, k = 1, col = P.canopy, colD = P.canopyD, seed = 0, fruits = false}: {
  x: number; y: number; r: number; k?: number; col?: string; colD?: string; seed?: number; fruits?: boolean;
}) => {
  const kk = Math.max(0.001, clamp01(k));
  const blobs: [number, number, number][] = [
    [0, -0.06, 1.0], [-0.78, 0.14, 0.58], [0.78, 0.12, 0.6],
    [-0.4, -0.36, 0.52], [0.42, -0.36, 0.5],
  ];
  return (
    <g transform={`translate(${x},${y}) scale(${kk})`} opacity={kk > 0.02 ? 1 : 0}>
      <ellipse cx={0} cy={r * 0.3} rx={r * 1.34} ry={r * 0.6} fill={colD} />
      {blobs.map((b, i) => (
        <ellipse key={i} cx={b[0] * r * 1.35} cy={b[1] * r} rx={b[2] * r} ry={b[2] * r * 0.72} fill={col} />
      ))}
      {fruits && [0, 1, 2, 3, 4].map((i) => (
        <circle key={i} cx={(rand(i, seed + 3) - 0.5) * r * 1.7} cy={r * 0.3 + rand(i, seed + 5) * r * 0.24} r={9} fill={P.fig} />
      ))}
    </g>
  );
};

/* ---------- 小苗/附生嫩芽 ---------- */
const Sprout = ({x, y, s = 1, k = 1}: {x: number; y: number; s?: number; k?: number}) => {
  const kk = Math.max(0.001, clamp01(k));
  return (
    <g transform={`translate(${x},${y}) scale(${s * kk})`} opacity={kk > 0.02 ? 1 : 0}>
      <path d="M 0 0 C -3 -18 3 -30 0 -44" stroke={P.leafD} strokeWidth={6} fill="none" strokeLinecap="round" />
      <path d="M 0 -30 C -22 -42 -34 -36 -40 -22 C -26 -14 -10 -18 0 -30 Z" fill={P.canopyHi} />
      <path d="M 0 -38 C 20 -52 34 -46 40 -32 C 26 -22 8 -26 0 -38 Z" fill={P.leaf} />
    </g>
  );
};

/* ---------- 小鸟（带翅膀扇动） ---------- */
const Bird = ({x, y, s = 1, flip = 1, k = 1}: {x: number; y: number; s?: number; flip?: number; k?: number}) => {
  const f = useCurrentFrame();
  const flap = Math.sin(f * 0.55) * 30;
  const kk = Math.max(0.001, k);
  if (kk <= 0.01) return null;
  return (
    <g transform={`translate(${x},${y}) scale(${flip * s * kk},${s * kk})`}>
      <path d="M -30 -2 Q -52 -10 -58 -2 Q -50 8 -30 6 Z" fill={P.canopyD} />
      <ellipse rx={30} ry={19} fill={P.cream} />
      <circle cx={25} cy={-8} r={13} fill={P.cream} />
      <path d="M 36 -12 L 54 -7 L 36 -2 Z" fill={P.fig} />
      <circle cx={29} cy={-11} r={3.2} fill={P.ink} />
      <ellipse cx={-3} cy={-5} rx={21} ry={11} fill={P.leafD} transform={`rotate(${flap} -3 -5)`} />
    </g>
  );
};

/* ---------- 小人剪影（比例尺担当） ---------- */
const Person = ({x, y, s = 1, col = P.dark}: {x: number; y: number; s?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} fill={col}>
    <circle cy={-58} r={11} />
    <rect x={-9} y={-47} width={18} height={34} rx={9} />
    <rect x={-8} y={-16} width={7} height={18} rx={3.5} />
    <rect x={1} y={-16} width={7} height={18} rx={3.5} />
  </g>
);

/* ---------- 小房子 ---------- */
const Hut = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-26} y={-30} width={52} height={30} rx={4} fill={P.plate} />
    <path d="M -34 -30 L 0 -52 L 34 -30 Z" fill={P.coral} />
    <rect x={-8} y={-18} width={16} height={18} rx={2} fill={P.ink} />
  </g>
);

/* ---------- 勾选 / 叉号 ---------- */
const Check = ({x, y, r = 24, k = 1, col = P.canopyHi}: {x: number; y: number; r?: number; k?: number; col?: string}) => {
  const kk = Math.max(0.001, clamp01(k));
  return (
    <g transform={`translate(${x},${y}) scale(${kk})`} opacity={kk > 0.02 ? 1 : 0}>
      <circle r={r} fill={col} />
      <path d={`M ${-r * 0.45} 0 L ${-r * 0.1} ${r * 0.38} L ${r * 0.5} ${-r * 0.36}`} stroke={P.ink} strokeWidth={6} fill="none" strokeLinecap="round" strokeLinejoin="round" />
    </g>
  );
};
const Cross = ({x, y, r = 24, k = 1, col = P.coral}: {x: number; y: number; r?: number; k?: number; col?: string}) => {
  const kk = Math.max(0.001, clamp01(k));
  return (
    <g transform={`translate(${x},${y}) scale(${kk})`} opacity={kk > 0.02 ? 1 : 0} stroke={col} strokeWidth={8} strokeLinecap="round">
      <circle r={r} fill="none" />
      <line x1={-r * 0.5} y1={-r * 0.5} x2={r * 0.5} y2={r * 0.5} />
      <line x1={r * 0.5} y1={-r * 0.5} x2={-r * 0.5} y2={r * 0.5} />
    </g>
  );
};

/* ---------- 向下箭头 ---------- */
const ArrowDown = ({x, y, h, col = P.sun, w = 8}: {x: number; y: number; h: number; col?: string; w?: number}) => (
  <g stroke={col} strokeWidth={w} strokeLinecap="round" fill={col}>
    <line x1={x} y1={y} x2={x} y2={y + h} stroke={col} />
    <path d={`M ${x - 14} ${y + h - 14} L ${x} ${y + h + 4} L ${x + 14} ${y + h - 14} Z`} stroke="none" />
  </g>
);

/* ============ 标题卡 ============ */
const TitleCard = ({data}: {data: any}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 13, mass: 0.7}});
  const pop = spring({frame: f - 6, fps, config: {damping: 10, stiffness: 140}});
  const rootK = clamp01((f - 12) / 30);
  const o = interpolate(f, [0, 8], [0, 1]);
  return (
    <AbsoluteFill>
      <Backdrop top={P.bg} bottom={P.bg} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <circle cx={960} cy={348} r={212} fill={P.plate} />
        <g transform={`translate(960,348) scale(${clamp01(pop)})`}>
          {/* 迷你榕树：树冠 + 枝干 + 垂根帘 */}
          <ellipse cx={0} cy={108} rx={120} ry={16} fill={P.soil} opacity={0.9} />
          <path d="M -16 108 L -9 -30 L 9 -30 L 16 108 Z" fill={P.barkD} />
          <path d="M -8 -28 C -60 -20 -96 -6 -120 16 M 8 -28 C 60 -22 96 -8 118 12"
            stroke={P.bark} strokeWidth={13} fill="none" strokeLinecap="round" />
          <AirRoot x={-96} y={4} len={96} k={rootK} wig={6} seed={1} w={5} mucus={false} />
          <AirRoot x={-40} y={-2} len={104} k={clamp01(rootK * 1.15)} wig={5} seed={2} w={5} mucus={false} />
          <AirRoot x={52} y={-4} len={100} k={clamp01(rootK * 0.9)} wig={6} seed={3} w={5} mucus={false} />
          <AirRoot x={104} y={6} len={92} k={clamp01(rootK * 1.05)} wig={5} seed={4} w={5} mucus={false} />
          <Canopy x={0} y={-88} r={92} fruits />
        </g>
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 268}}>
          <div style={{color: P.canopyHi, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 26}}>
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

/* ============ S0 一棵还是一片林：群柱假象 → 全是同一棵树 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1); // ≈325f “这就是榕树…独木成林”

  const canK = spring({frame: f - 14, fps, config: {damping: 13, stiffness: 120}});
  const pillarK = (i: number) => clamp01(spring({frame: f - 26 - i * 13, fps, config: {damping: 12, stiffness: 120}}));
  const hangK = (i: number) => clamp01((f - 70 - i * 30) / 70) * 0.62;
  const revealK = clamp01((f - b1 - 8) / 22);
  const personX = interpolate(f, [30, 530], [90, 1480], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});

  // 五根柱：x 位置与粗细（主干最粗）
  const pillars = [
    {x: 380, w: 52, h: 500},
    {x: 640, w: 30, h: 490},
    {x: 850, w: 26, h: 495},
    {x: 1070, w: 32, h: 490},
    {x: 1300, w: 24, h: 495},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={0} name={sec.name} accent={P.canopyHi} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <rect x={0} y={872} width={1920} height={64} fill={P.soil} />
        <rect x={0} y={872} width={1920} height={11} fill={P.soilD} />

        {/* 共享大冠 */}
        <g opacity={clamp01(canK)}>
          <Canopy x={470} y={275} r={165} k={1} />
          <Canopy x={850} y={225} r={195} k={1} />
          <Canopy x={1230} y={280} r={160} k={1} />
        </g>

        {/* 主干与支柱群 */}
        {pillars.map((p, i) => (
          <PropRoot key={i} x={p.x} gy={872} h={p.h} w={p.w} k={pillarK(i)} col={i === 0 ? P.barkD : P.bark} />
        ))}
        {/* 还在半空的气生根 */}
        <AirRoot x={510} y={360} len={512} k={hangK(0)} wig={12} seed={2} w={6} />
        <AirRoot x={745} y={352} len={520} k={hangK(1)} wig={10} seed={5} w={5} />
        <AirRoot x={965} y={358} len={514} k={hangK(2)} wig={11} seed={7} w={6} />
        <AirRoot x={1190} y={365} len={507} k={hangK(3)} wig={9} seed={9} w={5} />

        {/* 走路的小人：比例尺 */}
        <g opacity={clamp01((f - 30) / 12) * (1 - clamp01((f - 520) / 14))}>
          <Person x={personX} y={866} s={1.15} />
        </g>

        {/* 揭示：全部连着同一棵 */}
        {revealK > 0 && (
          <g opacity={revealK}>
            <rect x={300} y={120} width={1120} height={770} rx={60} fill="none"
              stroke={P.sun} strokeWidth={5} strokeDasharray="18 16" />
            {[0, 1, 2, 3, 4].map((i) => (
              <line key={i} x1={pillars[i].x} y1={380} x2={pillars[i].x} y2={430}
                stroke={P.sun} strokeWidth={4} strokeDasharray="6 10" />
            ))}
          </g>
        )}
      </svg>
      <Chip text="远看：一片小树林" x={130} y={560} at={beat(0) + 40} accent={P.canopyHi} />
      <Chip text="走近：全连着一个头" x={1330} y={620} at={beat(0) + 150} accent={P.root} />
      <Stamp text="同一棵树！" x={660} y={95} at={b1 + 15} color={P.sun} size={54} rot={-4} />
      <Stamp text="独木成林·基本操作" x={1260} y={170} at={b1 + 95} color={P.fig} size={46} rot={5} />
    </FadeIn>
  );
};

/* ============ S1 出生就住高层：鸟粪落种 → 树杈附生 → 赢在高起点 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1); // ≈307f “别的树从地面起跑…”

  const treeK = spring({frame: f - 8, fps, config: {damping: 12, stiffness: 120}});
  // 鸟：飞入 → 悬停投递 → 飞走
  const birdX = interpolate(f, [20, 95, 150, 225], [-120, 640, 640, 2050], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const birdY = 240 + Math.sin(f * 0.12) * 10;
  const birdK = clamp01((f - 20) / 10);
  // 种子下落
  const seedP = clamp01((f - 118) / 34);
  const seedX = 640 + seedP * 18;
  const seedY = 252 + seedP * seedP * 168; // 落到树杈 (658,420)
  const seedOk = f >= 118 && seedP < 1;
  const seedIn = f >= 152; // 落进杈里
  // 发芽 + 首批气生根
  const sprK = spring({frame: f - 168, fps, config: {damping: 10, stiffness: 150}});
  const r1K = clamp01((f - 210) / 110) * 0.5;
  const r2K = clamp01((f - 280) / 110) * 0.34;
  // 右卡：起跑线对照
  const cardK = spring({frame: f - b1 - 6, fps, config: {damping: 12, stiffness: 130}});

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={1} name={sec.name} accent={P.fig} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <rect x={0} y={872} width={1920} height={64} fill={P.soil} />
        <rect x={0} y={872} width={1920} height={11} fill={P.soilD} />

        {/* ---- 左：宿主树 ---- */}
        <g opacity={clamp01(treeK)}>
          <Trunk x={430} gy={872} h={480} w={86} col={P.hostBark} />
          <path d="M 430 420 C 510 400 570 400 640 430" stroke={P.hostBark} strokeWidth={22} fill="none" strokeLinecap="round" />
          <path d="M 640 430 L 690 396 M 640 430 L 676 452" stroke={P.hostBark} strokeWidth={12} strokeLinecap="round" />
          <Canopy x={400} y={300} r={150} col={P.host} colD={P.hostD} />
          <Canopy x={700} y={360} r={72} col={P.host} colD={P.hostD} />
        </g>

        {/* 种子 */}
        {seedOk && <circle cx={seedX} cy={seedY} r={8} fill={P.fig} stroke={P.barkD} strokeWidth={2} />}
        {seedIn && <circle cx={658} cy={418} r={7} fill={P.fig} opacity={1 - clamp01(sprK)} />}

        {/* 树杈里的嫩芽 + 垂下的第一根气生根 */}
        <Sprout x={656} y={424} s={1.05} k={clamp01(sprK)} />
        <AirRoot x={662} y={430} len={430} k={r1K} wig={10} seed={3} w={6} />
        <AirRoot x={646} y={428} len={432} k={r2K} wig={8} seed={8} w={5} />

        {/* ---- 鸟 ---- */}
        <Bird x={birdX} y={birdY} s={1.15} k={birdK} />
        {f > 100 && f < 160 && (
          <text x={640} y={196} textAnchor="middle" fill={P.cream} fontSize={28} fontFamily={FONT_HEAD} opacity={0.9}>
            投递中…
          </text>
        )}

        {/* ---- 右：起跑线对照卡 ---- */}
        <g transform={`translate(${(1 - clamp01(cardK)) * 60},0)`} opacity={clamp01(cardK)}>
          <rect x={1120} y={170} width={690} height={570} rx={40} fill={P.card} stroke={P.canopyHi} strokeWidth={4} />
          <text x={1465} y={228} textAnchor="middle" fill={P.cream} fontSize={34} fontFamily={FONT_HEAD} fontWeight={800}>
            起跑线差十几米
          </text>
          {/* 高度标尺 */}
          <line x1={1210} y1={660} x2={1210} y2={300} stroke={P.cream} strokeWidth={4} strokeDasharray="4 12" opacity={0.6} />
          {[0, 1, 2, 3].map((i) => (
            <g key={i}>
              <line x1={1198} y1={660 - i * 120} x2={1222} y2={660 - i * 120} stroke={P.cream} strokeWidth={4} />
              <text x={1180} y={668 - i * 120} textAnchor="end" fill={P.cream} fontSize={24} fontFamily={FONT_HEAD} opacity={0.75}>
                {i * 5}m
              </text>
            </g>
          ))}
          {/* 地面起跑的小树苗 */}
          <line x1={1300} y1={660} x2={1740} y2={660} stroke={P.soil} strokeWidth={10} strokeLinecap="round" />
          <Sprout x={1420} y={656} s={0.8} k={1} />
          <text x={1420} y={706} textAnchor="middle" fill={P.cream} fontSize={26} fontFamily={FONT_HEAD} opacity={0.85}>
            普通树·地面起跑
          </text>
          {/* 高层起跑的榕树 */}
          <line x1={1300} y1={300} x2={1740} y2={300} stroke={P.hostBark} strokeWidth={9} strokeLinecap="round" strokeDasharray="2 14" />
          <Sprout x={1560} y={296} s={0.9} k={1} />
          <Sun x={1700} y={262} r={26} />
          {clamp01(cardK) > 0.6 && (
            <g opacity={clamp01((f - b1 - 40) / 20)}>
              <path d="M 1680 296 L 1576 300 M 1696 300 L 1600 306" stroke={P.sun} strokeWidth={4} strokeLinecap="round" opacity={0.8} />
              <text x={1560} y={252} textAnchor="middle" fill={P.sun} fontSize={26} fontFamily={FONT_HEAD} fontWeight={700}>
                出生就晒到太阳
              </text>
            </g>
          )}
          <text x={1465} y={392} textAnchor="middle" fill={P.canopyHi} fontSize={25} fontFamily={FONT_HEAD} opacity={0.9}>
            榕树·树杈上发芽
          </text>
          {/* 树荫罩着地面苗 */}
          <rect x={1340} y={560} width={200} height={100} rx={18} fill={P.dark} opacity={0.35} />
          <text x={1440} y={552} textAnchor="middle" fill={P.cream} fontSize={22} fontFamily={FONT_HEAD} opacity={0.7}>阴影里排队</text>
        </g>
      </svg>
      <Chip text="鸟粪快递·精准入杈" x={140} y={140} at={125} accent={P.fig} />
      <Chip text="附生：先借住别人家" x={130} y={770} at={beat(0) + 160} accent={P.canopyHi} />
      <Stamp text="出生就住高层" x={1150} y={95} at={b1 + 10} color={P.fig} size={50} rot={-4} />
      <Stamp text="起点赢一大截" x={1500} y={790} at={b1 + 95} color={P.canopyD} size={46} rot={4} />
    </FadeIn>
  );
};

/* ============ S2 天上掉下承重柱：气生根下探 → 触土木质化 → 外接支架 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1); // ≈234f “一旦触土就扎进去吸水…”

  const canK = spring({frame: f - 10, fps, config: {damping: 13, stiffness: 110}});
  const trunkK = spring({frame: f - 4, fps, config: {damping: 13, stiffness: 120}});

  // 三条主根：起点 x / 下探进度 / 落地时刻 / 柱化进度
  const GY = 872, BY = 332, ROOT_LEN = GY - BY - 8;
  const roots = [
    {x: 560, seed: 2, t0: 25, land: 150},
    {x: 780, seed: 5, t0: 90, land: 232},
    {x: 1000, seed: 8, t0: 168, land: 330},
  ];
  const rk = (i: number) => clamp01((f - roots[i].t0) / (roots[i].land - roots[i].t0));
  const pk = (i: number) => clamp01((f - roots[i].land - 8) / 62);
  // 两根悬空的短根
  const stubK = (i: number) => clamp01((f - 120 - i * 60) / 90) * (i === 0 ? 0.52 : 0.4);

  // 力学卡
  const cardK = spring({frame: f - b1 - 18, fps, config: {damping: 12, stiffness: 130}});
  // 尾声：树冠继续外扩 + 新根尖
  const growK = clamp01((f - 470) / 90);

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={2} name={sec.name} accent={P.root} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <rect x={0} y={872} width={1920} height={64} fill={P.soil} />
        <rect x={0} y={872} width={1920} height={11} fill={P.soilD} />

        {/* ---- 大榕树 ---- */}
        <g opacity={clamp01(trunkK)}>
          <Trunk x={300} gy={GY} h={560} w={96} />
          <path d="M 330 360 C 500 330 760 312 1160 306" stroke={P.barkD} strokeWidth={34} fill="none" strokeLinecap="round" />
          <path d="M 620 330 C 660 300 680 286 712 268 M 900 316 C 930 292 950 282 986 272"
            stroke={P.bark} strokeWidth={16} fill="none" strokeLinecap="round" />
        </g>
        <g transform={`translate(700,240) scale(${1 + growK * 0.05}) translate(-700,-240)`} opacity={clamp01(canK)}>
          <Canopy x={390} y={215} r={140} />
          <Canopy x={700} y={178} r={165} fruits />
          <Canopy x={1030} y={215} r={145} />
        </g>

        {/* 气生根下探 */}
        {roots.map((r, i) => (
          <AirRoot key={i} x={r.x} y={BY} len={ROOT_LEN} k={rk(i)} wig={10 + i * 2} seed={r.seed} w={7} thick={pk(i)} />
        ))}
        {/* 悬空短根 */}
        <AirRoot x={660} y={330} len={ROOT_LEN} k={stubK(0)} wig={9} seed={11} w={5} />
        <AirRoot x={895} y={326} len={ROOT_LEN} k={stubK(1)} wig={8} seed={13} w={5} />
        {/* 触土涟漪 + 支柱化 */}
        {roots.map((r, i) => {
          const [tx] = hangTip(r.x, BY, ROOT_LEN, 1, 10 + i * 2, r.seed);
          return (
            <g key={`p${i}`}>
              {f >= r.land && <Ripple x={tx} y={GY - 4} startFrame={r.land} color={P.root} />}
              <PropRoot x={tx} gy={GY} h={536} w={30 + i * 3} k={pk(i)} col={P.rootD} />
            </g>
          );
        })}

        {/* ---- 右上：外接支架力学卡 ---- */}
        <g transform={`translate(${(1 - clamp01(cardK)) * 60},0)`} opacity={clamp01(cardK)}>
          <rect x={1250} y={170} width={600} height={470} rx={38} fill={P.card} stroke={P.root} strokeWidth={4} />
          <text x={1550} y={226} textAnchor="middle" fill={P.cream} fontSize={33} fontFamily={FONT_HEAD} fontWeight={800}>
            支柱根 = 外接支架
          </text>
          {/* 横梁（枝干）+ 载荷箭头 + 支柱 */}
          <line x1={1330} y1={330} x2={1700} y2={330} stroke={P.barkD} strokeWidth={20} strokeLinecap="round" />
          <ArrowDown x={1560} y={262} h={56} col={P.sun} w={7} />
          <text x={1560} y={252} textAnchor="middle" fill={P.sun} fontSize={24} fontFamily={FONT_HEAD}>树冠重量</text>
          <PropRoot x={1660} gy={560} h={230} w={30} k={1} col={P.rootD} />
          <line x1={1300} y1={560} x2={1780} y2={560} stroke={P.soil} strokeWidth={9} strokeLinecap="round" />
          {/* 普通树对照：压弯 */}
          <g transform="translate(1390,480)">
            <path d="M 0 70 C -6 30 -14 8 -34 -12" stroke={P.deadD} strokeWidth={13} fill="none" strokeLinecap="round" />
            <ellipse cx={-40} cy={-22} rx={42} ry={24} fill={P.dead} />
            <path d="M -14 6 L -4 20 M -20 22 L -8 34" stroke={P.coral} strokeWidth={4} strokeLinecap="round" />
            <Cross x={34} y={-30} r={16} k={1} col={P.coral} />
          </g>
          <text x={1360} y={612} textAnchor="middle" fill={P.cream} fontSize={24} fontFamily={FONT_HEAD} opacity={0.75}>普通树会压断</text>
          <text x={1660} y={612} textAnchor="middle" fill={P.canopyHi} fontSize={24} fontFamily={FONT_HEAD}>每扩一圈·垂一批新柱</text>
        </g>
      </svg>
      <Chip text="气生根·垂下的麻绳" x={120} y={600} at={beat(0) + 50} accent={P.root} />
      <Chip text="根尖黏液保命" x={880} y={740} at={beat(0) + 140} accent={P.water} />
      <Stamp text="触土→转正" x={1050} y={95} at={roots[0].land + 12} color={P.root} size={50} rot={-5} />
      <Stamp text="上限？拆了" x={1420} y={95} at={b1 + 285} color={P.sun} size={50} rot={4} />
    </FadeIn>
  );
};

/* ============ S3 顺手绞个杀：根网包裹宿主 → 三线齐输 → 空心树洞 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1); // ≈316f “绞杀不是谋杀…”

  const treeK = spring({frame: f - 6, fps, config: {damping: 13, stiffness: 120}});

  // 缠网：6 条根从上方拢向主干
  const HX = 600; // 宿主中心
  const strands = [
    {x0: 380, side: -1}, {x0: 480, side: -1}, {x0: 540, side: -1},
    {x0: 660, side: 1}, {x0: 730, side: 1}, {x0: 830, side: 1},
  ];
  const strandK = (i: number) => clamp01((f - 20 - i * 16) / 130);
  // 缠绕根的贝塞尔轨道（在主干外侧收拢）
  const wrapPath = (x0: number, side: number, k: number) => {
    const p0 = [x0, 130], p1 = [x0 + side * 40, 330], p2 = [HX + side * 105, 610], p3 = [HX + side * 44, 866];
    const n = 16;
    let d = `M ${p0[0]} ${p0[1]}`;
    for (let i = 1; i <= n; i++) {
      const t = (i / n) * clamp01(k);
      const mt = 1 - t;
      const bx = mt * mt * mt * p0[0] + 3 * mt * mt * t * p1[0] + 3 * mt * t * t * p2[0] + t * t * t * p3[0];
      const by = mt * mt * mt * p0[1] + 3 * mt * mt * t * p1[1] + 3 * mt * t * t * p2[1] + t * t * t * p3[1];
      d += ` L ${bx.toFixed(1)} ${by.toFixed(1)}`;
    }
    return d;
  };
  const webK = clamp01((f - 210) / 70);            // 横向网线
  const healthK = 1 - clamp01((f - 250) / 130);    // 宿主衰败
  const trunkFade = 1 - clamp01((f - 372) / 55);   // 主干消失 → 树洞
  const holeK = clamp01((f - 350) / 60);           // 树洞成形
  const hostCanCol = interpolateColors(1 - healthK, [0, 1], [P.host, P.dead]);
  const hostTrunkCol = interpolateColors(1 - healthK, [0, 1], [P.hostBark, P.deadD]);

  return (
    <FadeIn>
      <Backdrop top={P.bgDeep} bottom={P.bgDeep} />
      <SceneTag index={3} name={sec.name} accent={P.coral} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <rect x={0} y={872} width={1920} height={64} fill={P.soil} />
        <rect x={0} y={872} width={1920} height={11} fill={P.soilD} />

        {/* ---- 宿主树 ---- */}
        <g opacity={clamp01(treeK) * (0.35 + trunkFade * 0.65)}>
          <Trunk x={HX} gy={872} h={500} w={88} col={hostTrunkCol} />
          <g transform={`translate(${HX},300) scale(${0.7 + healthK * 0.3})`} opacity={0.4 + healthK * 0.6}>
            <Canopy x={0} y={0} r={150} col={hostCanCol} colD={P.deadD} />
            <Canopy x={-160} y={52} r={80} col={hostCanCol} colD={P.deadD} />
            <Canopy x={160} y={52} r={80} col={hostCanCol} colD={P.deadD} />
          </g>
        </g>
        {/* 落叶 */}
        {[0, 1, 2, 3, 4, 5].map((i) => {
          const t = (f - 268 - i * 16) / 60;
          if (t <= 0 || t >= 1) return null;
          return (
            <ellipse key={i} cx={HX - 140 + rand(i, 31) * 300 + t * 30} cy={330 + t * t * 480}
              rx={13} ry={7} fill={P.dead} opacity={0.9 - t * 0.5} transform={`rotate(${t * 160 + i * 40} ${HX - 140 + rand(i, 31) * 300 + t * 30} ${330 + t * t * 480})`} />
          );
        })}

        {/* ---- 树洞（主干消失后显露） ---- */}
        {holeK > 0 && (
          <g opacity={holeK}>
            <ellipse cx={HX} cy={580} rx={64} ry={190} fill={P.hole} />
            <path d={`M ${HX - 34} 440 Q ${HX - 50} 580 ${HX - 30} 730 M ${HX + 32} 450 Q ${HX + 48} 580 ${HX + 28} 720`}
              stroke={P.deadD} strokeWidth={5} fill="none" opacity={0.7} />
            {/* 洞里的小蝙蝠 */}
            <g transform={`translate(${HX},610)`} fill={P.dark}>
              <path d="M -26 0 Q -14 -18 0 -6 Q 14 -18 26 0 Q 12 10 0 6 Q -12 10 -26 0 Z" />
              <circle cy={-4} r={6} />
            </g>
            <text x={HX} y={806} textAnchor="middle" fill={P.cream} fontSize={27} fontFamily={FONT_HEAD} opacity={0.9}>
              树洞·当年的位置
            </text>
          </g>
        )}

        {/* ---- 绞杀根网 ---- */}
        {strands.map((s, i) => (
          <path key={i} d={wrapPath(s.x0, s.side, strandK(i))} stroke={P.root}
            strokeWidth={8 + clamp01((f - 200) / 120) * 10} fill="none" strokeLinecap="round" />
        ))}
        {/* 横向网线 */}
        {webK > 0 && [500, 620, 740].map((yy, i) => (
          <path key={i} d={`M ${HX - 100 + i * 6} ${yy} Q ${HX} ${yy + 26} ${HX + 100 - i * 6} ${yy}`}
            stroke={P.rootD} strokeWidth={6} fill="none" opacity={webK * 0.9} />
        ))}
        {/* 根部融合裙边 */}
        {webK > 0.3 && (
          <ellipse cx={HX} cy={866} rx={104} ry={20} fill="none" stroke={P.rootD} strokeWidth={9} opacity={webK} />
        )}

        {/* ---- 右卡：三条战线 ---- */}
        <g opacity={clamp01((f - beat(0) - 55) / 20)}>
          <rect x={1150} y={190} width={640} height={360} rx={38} fill={P.card} stroke={P.coral} strokeWidth={4} />
          <text x={1470} y={248} textAnchor="middle" fill={P.cream} fontSize={33} fontFamily={FONT_HEAD} fontWeight={800}>
            宿主三线同时输
          </text>
          {[
            {icon: 'water', label: '土里抢水抢养分'},
            {icon: 'sun', label: '树冠抢阳光'},
            {icon: 'space', label: '空间被包圆'},
          ].map((r, i) => {
            const on = clamp01((f - beat(0) - 80 - i * 30) / 20);
            return (
              <g key={i} opacity={0.3 + on * 0.7}>
                {r.icon === 'water' && <path d={`M 1235 ${318 + i * 74} c -14 18 -14 32 0 40 c 14 -8 14 -22 0 -40`} fill={P.water} />}
                {r.icon === 'sun' && (
                  <g transform={`translate(1235,${336 + i * 74})`}>
                    <circle r={16} fill={P.sun} />
                    {[0, 1, 2, 3].map((j) => (
                      <line key={j} x1={Math.cos(j * Math.PI / 2) * 22} y1={Math.sin(j * Math.PI / 2) * 22}
                        x2={Math.cos(j * Math.PI / 2) * 30} y2={Math.sin(j * Math.PI / 2) * 30} stroke={P.sun} strokeWidth={4} strokeLinecap="round" />
                    ))}
                  </g>
                )}
                {r.icon === 'space' && (
                  <g transform={`translate(1235,${336 + i * 74})`} stroke={P.fig} strokeWidth={5} strokeLinecap="round" fill="none">
                    <rect x={-16} y={-14} width={32} height={28} rx={6} />
                    <path d="M -26 0 L -34 0 M 26 0 L 34 0 M -34 0 l 6 -5 M -34 0 l 6 5 M 34 0 l -6 -5 M 34 0 l -6 5" />
                  </g>
                )}
                <text x={1290} y={348 + i * 74} fill={P.cream} fontSize={30} fontFamily={FONT_HEAD}>{r.label}</text>
              </g>
            );
          })}
        </g>
      </svg>
      <Chip text="气生根·越长越多的网" x={120} y={140} at={beat(0) + 40} accent={P.root} />
      <Stamp text="活活挤死" x={890} y={150} at={250} color={P.coral} size={50} rot={-5} />
      <Chip text="不是谋杀·是副产品" x={1200} y={600} at={b1 + 25} accent={P.fig} />
      <Stamp text="作案现场" x={760} y={700} at={b1 + 115} color={P.dead} size={48} rot={5} />
    </FadeIn>
  );
};

/* ============ S4 树的极限在哪：俯瞰一树成林 → 主干可弃 → 只在南方灵 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1); // ≈308f “疯狂长腿只为抢阳光…”

  const CX = 620, CY = 460, R = 355;
  const discK = spring({frame: f - 8, fps, config: {damping: 14, stiffness: 110}});
  const dotK = (i: number) => clamp01((f - 55 - i * 6) / 22);
  const ghostK = clamp01((f - beat(0) - 140) / 24);   // 主干退役标记
  const southK = spring({frame: f - b1 - 6, fps, config: {damping: 12, stiffness: 130}});
  const northK = spring({frame: f - b1 - 36, fps, config: {damping: 12, stiffness: 130}});

  // 环带支柱根点（俯视图）
  const dots = Array.from({length: 34}).map((_, i) => {
    const a = rand(i, 41) * Math.PI * 2;
    const rr = R * (0.72 + rand(i, 42) * 0.22);
    return {x: CX + Math.cos(a) * rr * 1.12, y: CY + Math.sin(a) * rr * 0.82, r: 8 + rand(i, 43) * 5, k: dotK(i)};
  });
  // 冠内叶簇点
  const tufts = Array.from({length: 20}).map((_, i) => {
    const a = rand(i, 51) * Math.PI * 2;
    const rr = R * rand(i, 52) * 0.72;
    return {x: CX + Math.cos(a) * rr * 1.12, y: CY + Math.sin(a) * rr * 0.8, r: 26 + rand(i, 53) * 34, c: i % 3};
  });

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={4} name={sec.name} accent={P.sun} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* ---- 俯瞰大冠 ---- */}
        <g opacity={clamp01(discK)}>
          <ellipse cx={CX} cy={CY} rx={R * 1.14 * clamp01(discK)} ry={R * 0.84 * clamp01(discK)} fill={P.canopyD} />
          {tufts.map((t, i) => (
            <circle key={i} cx={t.x} cy={t.y} r={t.r}
              fill={t.c === 0 ? P.canopy : t.c === 1 ? P.canopyHi : P.leafD} opacity={0.92} />
          ))}
          {/* 林地边界虚线 */}
          <ellipse cx={CX} cy={CY} rx={R * 1.14 + 14} ry={R * 0.84 + 14} fill="none"
            stroke={P.cream} strokeWidth={4} strokeDasharray="14 14" opacity={0.55} />
        </g>
        {/* 支柱根点阵 */}
        {dots.map((d, i) => (
          <circle key={i} cx={d.x} cy={d.y} r={d.r * clamp01(d.k)} fill={P.rootD} stroke={P.plate} strokeWidth={2.5}
            opacity={d.k > 0.02 ? 1 : 0} />
        ))}
        {/* 主干幽灵位 */}
        {ghostK > 0 && (
          <g opacity={ghostK}>
            <circle cx={CX} cy={CY} r={42} fill="none" stroke={P.coral} strokeWidth={5} strokeDasharray="8 10" />
            <path d={`M ${CX - 14} ${CY + 16} L ${CX - 10} ${CY - 12} L ${CX + 10} ${CY - 12} L ${CX + 14} ${CY + 16} Z`} fill={P.deadD} />
            <line x1={CX - 22} y1={CY - 22} x2={CX + 22} y2={CY + 22} stroke={P.coral} strokeWidth={6} strokeLinecap="round" />
            <line x1={CX + 22} y1={CY - 22} x2={CX - 22} y2={CY + 22} stroke={P.coral} strokeWidth={6} strokeLinecap="round" />
            <text x={CX} y={CY + 84} textAnchor="middle" fill={P.cream} fontSize={26} fontFamily={FONT_HEAD} fontWeight={700}>
              主干·1925 年已砍掉
            </text>
          </g>
        )}
        {/* 比例尺：村口 + 打谷场小人 */}
        <g opacity={clamp01((f - 40) / 16)}>
          <Hut x={268} y={806} s={0.9} />
          <Person x={352} y={830} s={0.75} />
          <Person x={386} y={834} s={0.7} />
        </g>
        <text x={CX} y={CY - R * 0.84 - 34} textAnchor="middle" fill={P.cream} fontSize={30} fontFamily={FONT_HEAD}
          fontWeight={700} opacity={clamp01((f - 60) / 20)}>
          一棵树 ≈ 两公顷
        </text>
        {/* 新根还在落地：闪烁的新点 */}
        {f > 420 && [0, 1].map((i) => {
          const p = (f * 0.5 + i * 40) % 60;
          return <circle key={i} cx={CX - 200 + i * 380} cy={CY + R * 0.5} r={10} fill={P.root}
            opacity={p < 40 ? 0.4 + Math.sin(f / 5) * 0.3 : 0} />;
        })}

        {/* ---- 右卡：南方 vs 北方 ---- */}
        <g transform={`translate(${(1 - clamp01(southK)) * 50},0)`} opacity={clamp01(southK)}>
          <rect x={1130} y={160} width={690} height={290} rx={36} fill={P.card} stroke={P.canopyHi} strokeWidth={4} />
          <text x={1475} y={216} textAnchor="middle" fill={P.cream} fontSize={31} fontFamily={FONT_HEAD} fontWeight={800}>
            湿热南方 · 根能落地
          </text>
          {/* 雨滴 + 根触地 */}
          {[0, 1, 2].map((i) => (
            <path key={i} d={`M ${1260 + i * 60} ${268} c -11 14 -11 26 0 32 c 11 -6 11 -18 0 -32`} fill={P.water} opacity={0.9} />
          ))}
          <line x1={1340} y1={400} x2={1560} y2={400} stroke={P.soil} strokeWidth={8} strokeLinecap="round" />
          <AirRoot x={1400} y={300} len={100} k={1} wig={5} seed={4} w={6} mucus={false} />
          <PropRoot x={1400} gy={400} h={26} w={16} k={clamp01((f - b1 - 60) / 40)} col={P.rootD} />
          <Check x={1630} y={330} r={26} k={clamp01((f - b1 - 70) / 20)} />
          <text x={1475} y={426} textAnchor="middle" fill={P.canopyHi} fontSize={24} fontFamily={FONT_HEAD}>
            气生根顺利入土转正
          </text>
        </g>
        <g transform={`translate(${(1 - clamp01(northK)) * 50},0)`} opacity={clamp01(northK)}>
          <rect x={1130} y={490} width={690} height={290} rx={36} fill={P.card} stroke={P.coral} strokeWidth={4} />
          <text x={1475} y={546} textAnchor="middle" fill={P.cream} fontSize={31} fontFamily={FONT_HEAD} fontWeight={800}>
            干燥北方 · 半路渴死
          </text>
          <Sun x={1260} y={620} r={30} />
          <line x1={1340} y1={730} x2={1560} y2={730} stroke={P.soil} strokeWidth={8} strokeLinecap="round" />
          {/* 干瘪的半截根 */}
          <path d="M 1400 620 Q 1394 664 1398 692" stroke={P.deadD} strokeWidth={6} fill="none" strokeLinecap="round" />
          <path d="M 1398 692 q -8 10 -2 16 M 1398 692 q 8 10 2 16" stroke={P.deadD} strokeWidth={4} fill="none" strokeLinecap="round" />
          <Cross x={1630} y={660} r={26} k={clamp01((f - b1 - 100) / 20)} />
          <text x={1475} y={756} textAnchor="middle" fill={P.coral} fontSize={24} fontFamily={FONT_HEAD}>
            没落地就风干成柴火
          </text>
        </g>
      </svg>
      <Chip text="支柱根 ×3000+" x={150} y={160} at={beat(0) + 90} accent={P.root} />
      <Stamp text="脚手架接管全场" x={880} y={130} at={beat(0) + 210} color={P.fig} size={48} rot={-4} />
      <Chip text="一棵树盖住打谷场" x={140} y={700} at={beat(0) + 40} accent={P.canopyHi} />
      <Stamp text="这招只在南方灵" x={1470} y={95} at={b1 + 110} color={P.coral} size={48} rot={4} />
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
        <ellipse cx={960} cy={1010} rx={1080} ry={96} fill={P.dark} opacity={0.8} />
        <g transform="translate(960,980)">
          <path d="M -22 0 L -12 -190 L 12 -190 L 22 0 Z" fill={P.barkD} />
          <path d="M -10 -180 C -120 -160 -200 -120 -250 -60 M 10 -180 C 120 -160 200 -120 250 -60"
            stroke={P.barkD} strokeWidth={16} fill="none" strokeLinecap="round" />
          {[-200, -120, -45, 55, 130, 205].map((rx, i) => (
            <AirRoot key={i} x={rx} y={-70 - (i % 3) * 10} len={70 + (i % 3) * 14}
              k={clamp01(pop)} wig={6} seed={i + 2} w={5} mucus={false} />
          ))}
          <Canopy x={0} y={-270} r={210} k={clamp01(pop)} />
        </g>
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -200}}>
        <div style={{fontSize: 66, fontWeight: 800, color: C.text}}>一棵树，也能活成一片林</div>
        <div style={{fontSize: 38, color: C.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E220 = ({data = tl}: {data?: any}) => {
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
