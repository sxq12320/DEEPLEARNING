import {AbsoluteFill, Audio, Sequence, interpolate, interpolateColors, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01, rand} from '../lib/style';
import {Backdrop, Ripple, Stamp, SceneTag, Chip, Subtitles, ProgressBar, FadeIn} from '../lib/atoms';
import tl from '../timeline_211.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：竹青 × 稻草 × 土棕（纯色平涂） ============ */
const P = {
  bg: '#2c4020',      // 主底：竹林橄榄绿
  bgDeep: '#213017',
  bgLite: '#38512a',
  plate: '#f3e8c4',   // 米色圆盘
  card: '#35502a',    // 卡片底
  culm: '#83cb57',    // 竹秆
  culmD: '#53963a',
  culmHi: '#b9e88d',
  leaf: '#5dbd67',
  leafD: '#3c8a48',
  spike: '#f0d98e',   // 小花穗
  spikeD: '#d1b066',
  rice: '#f8edc5',    // 竹米
  riceD: '#d9bd7e',
  dry: '#c79f5a',     // 枯竹
  dryD: '#8e6b34',
  soil: '#453321',
  soilD: '#2e2113',
  rhiz: '#e2ba70',    // 地下茎
  land: '#8fb56f',    // 大陆块
  rat: '#a2917f',
  ratD: '#6f6053',
  crop: '#e5c96e',
  pot: '#c9804f',
  potD: '#a35f36',
  amber: '#ffc94d',
  coral: '#ff7b6b',
  mint: '#8fe3c0',
  cream: '#f7efd9',
  ink: '#37401e',     // 浅盘上的深字
};

/* ---------- 披针竹叶 ---------- */
const Leaf = ({x = 0, y = 0, ang = 0, len = 54, col = P.leaf, op = 1}: {
  x?: number; y?: number; ang?: number; len?: number; col?: string; op?: number;
}) => (
  <g transform={`translate(${x},${y}) rotate(${ang})`} opacity={op}>
    <path d={`M0 0 Q ${len * 0.52} ${-len * 0.2} ${len} -2 Q ${len * 0.52} ${len * 0.17} 0 0 Z`} fill={col} />
  </g>
);

/* ---------- 竹秆：节环外凸，侧枝带叶簇和花穗；dryK 0→1 枯黄 ---------- */
const Culm = ({x, y, h = 400, s = 1, rot = 0, dryK = 0, leafK = 1, bloomK = 0, sway = 0, seed = 0}: {
  x: number; y: number; h?: number; s?: number; rot?: number;
  dryK?: number; leafK?: number; bloomK?: number; sway?: number; seed?: number;
}) => {
  const f = useCurrentFrame();
  const seg = 58;
  const n = Math.max(3, Math.round(h / seg));
  const sw = Math.sin(f * 0.05 + seed * 2.3) * sway;
  const c0 = interpolateColors(dryK, [0, 1], [P.culm, P.dry]);
  const c1 = interpolateColors(dryK, [0, 1], [P.culmD, P.dryD]);
  const c2 = interpolateColors(dryK, [0, 1], [P.culmHi, '#e3c08a']);
  const l0 = interpolateColors(dryK, [0, 1], [P.leaf, '#bd9350']);
  const l1 = interpolateColors(dryK, [0, 1], [P.leafD, '#9a743c']);
  return (
    <g transform={`translate(${x},${y}) rotate(${rot + sw}) scale(${Math.max(0.001, s)})`}>
      {Array.from({length: n}).map((_, i) => {
        const w = 26 - i * 1.7;
        const yTop = -(i + 1) * seg + 5;
        return (
          <g key={i}>
            <rect x={-w / 2} y={yTop} width={w} height={seg - 3} rx={w * 0.46} fill={c0} />
            <rect x={-w / 2 + 3} y={yTop + 4} width={w * 0.3} height={seg - 13} rx={w * 0.15} fill={c2} opacity={0.55} />
            {/* 节环 */}
            <rect x={-w / 2 - 3.5} y={-i * seg - 5} width={w + 7} height={9.5} rx={4.5} fill={c1} />
          </g>
        );
      })}
      {/* 叶簇：中部节侧出小枝 */}
      {leafK > 0.02 && [1, 2, 3].map((node) => {
        if (node >= n) return null;
        const yy = -node * seg;
        const side = node % 2 === 1 ? 1 : -1;
        return (
          <g key={node} transform={`translate(${side * 12},${yy})`} opacity={leafK}>
            <line x1={0} y1={0} x2={side * 52} y2={-16} stroke={c1} strokeWidth={4.5} strokeLinecap="round" />
            <Leaf x={side * 30} y={-13} ang={side * (20 + node * 7)} len={54} col={l0} />
            <Leaf x={side * 42} y={-18} ang={side * (54 + node * 6)} len={47} col={l1} />
            <Leaf x={side * 16} y={-5} ang={side * (-12 + node * 4)} len={42} col={l0} />
          </g>
        );
      })}
      {/* 花穗：上部节逐节点亮 */}
      {bloomK > 0.02 && Array.from({length: Math.min(n, 5)}).map((_, i) => {
        const g = clamp01(bloomK * 1.7 - i * 0.2);
        if (g <= 0.02) return null;
        const yy = -(n - 1 - i) * seg;
        const side = i % 2 === 0 ? -1 : 1;
        return (
          <g key={`b${i}`} transform={`translate(${side * 11},${yy}) scale(${g})`} opacity={g}>
            <line x1={0} y1={0} x2={side * 30} y2={-26} stroke={P.spikeD} strokeWidth={3.5} strokeLinecap="round" />
            {[0, 1, 2, 3].map((j) => (
              <ellipse key={j} cx={side * (16 + j * 9)} cy={-13 - j * 9} rx={7.5} ry={4.8} fill={P.spike}
                transform={`rotate(${side * (-32 - j * 14)} ${side * (16 + j * 9)} ${-13 - j * 9})`} />
            ))}
          </g>
        );
      })}
    </g>
  );
};

/* ---------- 竹笋：锥形 + 箨片弧纹 ---------- */
const Shoot = ({x, y, s = 1, k = 1}: {x: number; y: number; s?: number; k?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s * Math.max(0.001, k)})`} opacity={k > 0.02 ? 1 : 0}>
    <path d="M -30 0 Q -24 -78 0 -132 Q 24 -78 30 0 Z" fill={P.culmD} />
    <path d="M -21 -40 Q 0 -52 21 -40" stroke={P.culmHi} strokeWidth={4.5} fill="none" strokeLinecap="round" />
    <path d="M -14 -74 Q 0 -84 14 -74" stroke={P.culmHi} strokeWidth={4} fill="none" strokeLinecap="round" />
    <path d="M -7 -104 Q 0 -112 7 -104" stroke={P.culmHi} strokeWidth={3.5} fill="none" strokeLinecap="round" />
    <ellipse cx={0} cy={-136} rx={9} ry={12} fill={P.culm} />
  </g>
);

/* ---------- 竹米：颖果粒 ---------- */
const Grain = ({x, y, rot = 0, s = 1, k = 1}: {x: number; y: number; rot?: number; s?: number; k?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s * Math.max(0.001, k)})`} opacity={k > 0.02 ? 1 : 0}>
    <ellipse rx={13} ry={7.6} fill={P.rice} />
    <path d="M -7 -2 Q 0 3 7 -2" stroke={P.riceD} strokeWidth={2.2} fill="none" />
  </g>
);

/* ---------- 竹米掉落：抛物线落下后停住 ---------- */
const GrainDrop = ({x0, y0, y1, at, drift = 0, dur = 42, s = 1, hold = 26}: {
  x0: number; y0: number; y1: number; at: number; drift?: number; dur?: number; s?: number; hold?: number;
}) => {
  const f = useCurrentFrame();
  const t = f - at;
  if (t < 0 || t > dur + hold) return null;
  const p = clamp01(t / dur);
  return <Grain x={x0 + drift * p + Math.sin(t * 0.35) * 4} y={y0 + (y1 - y0) * p * p} rot={t * 4} s={s} />;
};

/* ---------- 飘落的枯叶 ---------- */
const FallLeaf = ({x, y, at, len = 46, col = '#bd9350', seed = 0}: {
  x: number; y: number; at: number; len?: number; col?: string; seed?: number;
}) => {
  const f = useCurrentFrame();
  const t = f - at;
  if (t < 0 || t > 85) return null;
  return (
    <g transform={`translate(${x + Math.sin(t * 0.12 + seed) * 26 + t * 0.5 * (seed % 2 ? 1 : -1)},${y + t * t * 0.05}) rotate(${Math.sin(t * 0.16 + seed) * 42})`}
      opacity={1 - clamp01((t - 58) / 26)}>
      <Leaf x={0} y={0} ang={0} len={len} col={col} />
    </g>
  );
};

/* ---------- 粒子迸散 ---------- */
const Burst = ({x, y, at, n = 10, col = P.amber}: {x: number; y: number; at: number; n?: number; col?: string}) => {
  const f = useCurrentFrame();
  const t = f - at;
  if (t < 0) return null;
  const p = clamp01(t / 55);
  if (p >= 1) return null;
  return (
    <g>
      {Array.from({length: n}).map((_, i) => {
        const a = rand(i, 31) * Math.PI * 2;
        const d = p * (46 + rand(i, 32) * 80);
        return <circle key={i} cx={x + Math.cos(a) * d} cy={y + Math.sin(a) * d * 0.8 - p * 30}
          r={5 * (1 - p) + 2} fill={col} opacity={(1 - p) * 0.9} />;
      })}
    </g>
  );
};

/* ---------- 老鼠 ---------- */
const Rat = ({x, y, s = 1, flip = 1, k = 1}: {x: number; y: number; s?: number; flip?: number; k?: number}) => {
  const f = useCurrentFrame();
  const wig = Math.sin(f * 0.5 + x * 0.1) * 7;
  const kk = Math.max(0.001, k);
  return (
    <g transform={`translate(${x},${y}) scale(${flip * s * kk},${s * kk})`} opacity={k > 0.02 ? 1 : 0}>
      <path d={`M -32 6 Q -56 ${12 + wig} -72 ${28 + wig}`} stroke={P.ratD} strokeWidth={5} fill="none" strokeLinecap="round" />
      <ellipse cx={-12} cy={12} rx={13} ry={9} fill={P.ratD} />
      <ellipse rx={33} ry={19} fill={P.rat} />
      <circle cx={29} cy={-7} r={14} fill={P.rat} />
      <circle cx={34} cy={-19} r={6.5} fill={P.ratD} />
      <circle cx={35} cy={-8} r={2.6} fill="#1d1710" />
      <ellipse cx={43} cy={-3} rx={4.2} ry={3} fill={P.ratD} />
      <line x1={14} y1={10} x2={16} y2={24} stroke={P.ratD} strokeWidth={5} strokeLinecap="round" />
    </g>
  );
};

/* ---------- 熊猫脸 ---------- */
const Panda = ({x, y, s = 1, sad = false}: {x: number; y: number; s?: number; sad?: boolean}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <circle cx={-34} cy={-34} r={17} fill="#2b241c" />
    <circle cx={34} cy={-34} r={17} fill="#2b241c" />
    <circle r={46} fill="#f5efe2" />
    <ellipse cx={-17} cy={-8} rx={11} ry={13} fill="#2b241c" transform="rotate(-18 -17 -8)" />
    <ellipse cx={17} cy={-8} rx={11} ry={13} fill="#2b241c" transform="rotate(18 17 -8)" />
    <circle cx={-15} cy={-9} r={3.4} fill="#fff" />
    <circle cx={15} cy={-9} r={3.4} fill="#fff" />
    <ellipse cy={8} rx={7} ry={5} fill="#2b241c" />
    {sad
      ? <path d="M -10 25 Q 0 17 10 25" stroke="#2b241c" strokeWidth={3.5} fill="none" strokeLinecap="round" />
      : <path d="M -9 21 Q 0 28 9 21" stroke="#2b241c" strokeWidth={3.5} fill="none" strokeLinecap="round" />}
  </g>
);

/* ---------- 庄稼秆（麦子状） ---------- */
const Crop = ({x, y, k = 1, lean = 0}: {x: number; y: number; k?: number; lean?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${lean}) scale(1,${Math.max(0.05, k)})`}>
    <line x1={0} y1={0} x2={0} y2={-112} stroke={P.crop} strokeWidth={7} strokeLinecap="round" />
    {[0, 1, 2, 3].map((j) => (
      <ellipse key={j} cx={j % 2 === 0 ? -11 : 11} cy={-62 - j * 14} rx={10} ry={6} fill={P.crop}
        transform={`rotate(${j % 2 === 0 ? -35 : 35} ${j % 2 === 0 ? -11 : 11} ${-62 - j * 14})`} />
    ))}
    <ellipse cx={0} cy={-120} rx={8} ry={12} fill={P.crop} />
  </g>
);

/* ---------- 花盆 ---------- */
const Pot = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-54} y={-14} width={108} height={18} rx={9} fill={P.potD} />
    <path d="M -46 4 L 46 4 L 34 66 L -34 66 Z" fill={P.pot} />
  </g>
);

/* ---------- 基因时钟 ---------- */
const Clock = ({x, y, r = 100, k = 1, spin = 0}: {x: number; y: number; r?: number; k?: number; spin?: number}) => (
  <g transform={`translate(${x},${y}) scale(${Math.max(0.001, k)})`} opacity={k > 0.02 ? 1 : 0}>
    <circle r={r} fill={P.plate} stroke={P.ink} strokeWidth={6} />
    {Array.from({length: 12}).map((_, i) => {
      const a = (i / 12) * Math.PI * 2;
      return <line key={i} x1={Math.cos(a) * r * 0.78} y1={Math.sin(a) * r * 0.78}
        x2={Math.cos(a) * r * 0.9} y2={Math.sin(a) * r * 0.9} stroke={P.ink} strokeWidth={5} strokeLinecap="round" />;
    })}
    <line x1={0} y1={0} x2={0} y2={-r * 0.55} stroke={P.ink} strokeWidth={8} strokeLinecap="round" transform={`rotate(${spin})`} />
    <line x1={0} y1={0} x2={r * 0.38} y2={0} stroke={P.coral} strokeWidth={6} strokeLinecap="round" transform={`rotate(${spin * 6})`} />
    <circle r={8} fill={P.coral} />
  </g>
);

/* ============ 标题卡 ============ */
const TitleCard = ({data}: {data: any}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 13, mass: 0.7}});
  const pop = spring({frame: f - 6, fps, config: {damping: 10, stiffness: 140}});
  const o = interpolate(f, [0, 8], [0, 1]);
  return (
    <AbsoluteFill>
      <Backdrop top={P.bg} bottom={P.bg} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <circle cx={960} cy={350} r={215} fill={P.plate} />
        <g transform="translate(960,350)">
          <Culm x={-18} y={168} h={270} s={0.9 * Math.max(0.001, clamp01(pop))} rot={3} bloomK={clamp01((f - 8) / 30)} seed={2} />
        </g>
        <Shoot x={1060} y={520} s={0.7} k={clamp01(pop)} />
        <Grain x={880} y={528} rot={-20} s={0.9} k={clamp01(pop)} />
        <Grain x={920} y={536} rot={30} s={0.85} k={clamp01(pop)} />
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 250}}>
          <div style={{color: P.spike, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 26}}>
            冷知识 · 第 {data.episode} 期
          </div>
          <div style={{color: C.text, fontSize: 96, fontWeight: 900, fontFamily: FONT_HEAD, letterSpacing: 6}}>
            {data.title}
          </div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ============ S0 怪事：一丛竹几十年才开花，开完整片枯死 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);

  const culms = [
    {x: 430, h: 340, rot: -4, seed: 1},
    {x: 585, h: 440, rot: 2, seed: 2},
    {x: 750, h: 380, rot: -2, seed: 3},
    {x: 945, h: 460, rot: 3, seed: 4},
    {x: 1110, h: 320, rot: -3, seed: 5},
  ];
  const pop = (i: number) => spring({frame: f - 6 - i * 7, fps, config: {damping: 11, stiffness: 130}});

  // 年数表盘：0 → 118 年
  const year = Math.round(interpolate(f, [b1 + 12, b1 + 150], [0, 118],
    {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}));
  const arcP = year / 118;
  const bloomK = clamp01((f - b1 - 160) / 26);
  const dryK = clamp01((f - b1 - 278) / 60);
  const leafK = 1 - clamp01((f - b1 - 268) / 45);
  const R = 118, CXD = 1580, CYD = 420;

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={0} name={sec.name} accent={P.amber} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <ellipse cx={880} cy={968} rx={1080} ry={105} fill={P.bgDeep} />
        {culms.map((c, i) => (
          <Culm key={i} x={c.x} y={952} h={c.h} rot={c.rot + dryK * (i % 2 ? 6 : -7)}
            s={pop(i)} dryK={dryK} leafK={leafK} bloomK={bloomK} sway={1.6} seed={c.seed} />
        ))}
        {/* 枯死时落叶 */}
        {Array.from({length: 8}).map((_, i) => (
          <FallLeaf key={i} x={430 + rand(i, 3) * 700} y={430 + rand(i, 4) * 240}
            at={b1 + 270 + i * 9} len={44} seed={i} />
        ))}
        {/* 开花涟漪 + 枯死迸散 */}
        {bloomK > 0 && <Ripple x={770} y={560} startFrame={b1 + 160} color={P.spike} />}
        <Burst x={760} y={700} at={b2 + 12} col={P.amber} n={12} />

        {/* 年数表盘 */}
        <circle cx={CXD} cy={CYD} r={152} fill={P.plate} />
        <circle cx={CXD} cy={CYD} r={R} fill="none" stroke={P.ink} strokeWidth={4} opacity={0.15} />
        <circle cx={CXD} cy={CYD} r={R} fill="none" stroke={P.coral} strokeWidth={13} strokeLinecap="round"
          strokeDasharray={2 * Math.PI * R} strokeDashoffset={2 * Math.PI * R * (1 - arcP)}
          transform={`rotate(-90 ${CXD} ${CYD})`} />
        <text x={CXD} y={CYD + 12} textAnchor="middle" fill={P.ink} fontSize={86}
          fontFamily={FONT_HEAD} fontWeight={900}>{year}</text>
        <text x={CXD} y={CYD + 66} textAnchor="middle" fill={P.ink} fontSize={32}
          fontFamily={FONT_HEAD} fontWeight={700} opacity={0.75}>年才开一次花</text>
      </svg>
      <Stamp text="不祥之兆？" x={160} y={205} at={beat(0) + 35} color={P.coral} size={52} rot={-6} />
      <Stamp text="同步绽放！" x={1210} y={690} at={b1 + 168} color={P.spikeD} size={50} rot={5} />
      <Stamp text="整片枯死" x={230} y={700} at={b1 + 292} color={P.dryD} size={50} rot={-5} />
      <Stamp text="一辈子一次的大招" x={1130} y={140} at={b2 + 14} color={P.amber} size={46} rot={4} />
      <Chip text="不是不育，是攒着" x={150} y={330} at={b2 + 62} accent={P.mint} />
    </FadeIn>
  );
};

/* ============ S1 攒大招：普通树年年结果 vs 竹子攒一罐；周期时间轴；地下茎扩张 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);

  const cardL = spring({frame: f - 8, fps, config: {damping: 13, stiffness: 120}});
  const cardR = spring({frame: f - 16, fps, config: {damping: 13, stiffness: 120}});

  // 左卡：果树循环结果
  const cyc = (f % 80) / 80;
  const fruitK = cyc < 0.7 ? clamp01(cyc * 6) : 0;
  // 右卡：能量罐在 cue1 前蓄满
  const fillK = clamp01((f - 30) / (b1 - 55));
  const bloomK = clamp01((f - b1 - 8) / 30);
  const fullFlash = clamp01(1 - Math.abs(f - b1 - 8) / 26);
  // 底部时间轴
  const axisP = clamp01((f - b1 - 30) / 24);
  // 右卡土里：地下茎扩张
  const rhizP = clamp01((f - b2 - 12) / 55);
  const sp1 = spring({frame: f - b2 - 50, fps, config: {damping: 10, stiffness: 160}});
  const sp2 = spring({frame: f - b2 - 78, fps, config: {damping: 10, stiffness: 160}});

  const ticks = [
    {t: 0.24, label: '某竹 30年', at: b1 + 45},
    {t: 0.52, label: '毛竹 约65年', at: b1 + 62},
    {t: 0.96, label: '桂竹 120年', at: b1 + 80},
  ];
  const axisX = (t: number) => 280 + 1500 * t;

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={1} name={sec.name} accent={P.culmHi} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左卡：普通树年年结果 */}
        <g transform={`translate(0,${(1 - clamp01(cardL)) * 50})`} opacity={clamp01(cardL)}>
          <rect x={110} y={150} width={760} height={470} rx={40} fill={P.card} stroke={P.culmD} strokeWidth={4} />
          <text x={490} y={222} textAnchor="middle" fill={P.cream} fontSize={44} fontFamily={FONT_HEAD} fontWeight={800}>普通树木</text>
          <text x={490} y={268} textAnchor="middle" fill={P.cream} fontSize={28} fontFamily={FONT_HEAD} opacity={0.7}>年年开花结果</text>
          {/* 循环箭头 */}
          <g transform={`translate(490,418) rotate(${f * 1.4})`}>
            <circle r={138} fill="none" stroke={P.cream} strokeWidth={5} strokeDasharray="14 18" opacity={0.4} />
            <path d="M -10 -152 L -10 -128 L 16 -140 Z" fill={P.cream} opacity={0.6} />
          </g>
          {/* 树 */}
          <rect x={476} y={420} width={28} height={120} rx={10} fill="#7a5230" />
          <circle cx={416} cy={392} r={64} fill={P.leafD} />
          <circle cx={564} cy={392} r={64} fill={P.leafD} />
          <circle cx={490} cy={360} r={92} fill={P.leaf} />
          {Array.from({length: 8}).map((_, i) => {
            const fx = 490 + (rand(i, 41) - 0.5) * 170;
            const fy = 360 + (rand(i, 42) - 0.5) * 130 + 18;
            return <circle key={i} cx={fx} cy={fy} r={13} fill={P.amber} opacity={fruitK}
              transform={`translate(0,0) scale(1)`} />;
          })}
          <text x={490} y={588} textAnchor="middle" fill={P.cream} fontSize={26} fontFamily={FONT_HEAD} opacity={0.65}>每年一轮 · 细水长流</text>
        </g>

        {/* 右卡：竹子攒一罐能量，开一次就枯死 */}
        <g transform={`translate(0,${(1 - clamp01(cardR)) * 50})`} opacity={clamp01(cardR)}>
          <rect x={1050} y={150} width={760} height={470} rx={40} fill={P.card} stroke={P.culmD} strokeWidth={4} />
          <text x={1430} y={222} textAnchor="middle" fill={P.cream} fontSize={44} fontFamily={FONT_HEAD} fontWeight={800}>竹子</text>
          <text x={1430} y={268} textAnchor="middle" fill={P.cream} fontSize={28} fontFamily={FONT_HEAD} opacity={0.7}>开一次花就枯死</text>
          {/* 土槽 + 地下茎 */}
          <rect x={1090} y={545} width={680} height={62} rx={14} fill={P.soil} />
          {rhizP > 0 && (
            <g>
              <rect x={1300} y={566} width={Math.max(0, 400 * rhizP)} height={15} rx={7.5} fill={P.rhiz} />
              {[0.3, 0.62, 0.94].map((t, i) => (
                <circle key={i} cx={1300 + 400 * t} cy={573} r={rhizP > t ? 9 : 0} fill={P.rhiz}
                  opacity={rhizP > t ? 1 : 0} />
              ))}
            </g>
          )}
          <Shoot x={1468} y={550} s={0.42} k={clamp01(sp1)} />
          <Shoot x={1695} y={552} s={0.36} k={clamp01(sp2)} />
          {/* 竹 */}
          <Culm x={1300} y={558} h={290} s={1} rot={2} leafK={1} bloomK={bloomK} sway={0.8} seed={7} />
          {/* 能量罐 */}
          <rect x={1550} y={270} width={96} height={300} rx={22} fill="none" stroke={P.cream} strokeWidth={5} />
          {fillK > 0 && (
            <rect x={1559} y={561 - 282 * fillK} width={78} height={282 * fillK} rx={15}
              fill={fullFlash > 0 ? P.spike : P.amber} />
          )}
          {fillK >= 1 && (
            <rect x={1550} y={270} width={96} height={300} rx={22} fill="none" stroke={P.spike}
              strokeWidth={6} opacity={0.4 + 0.5 * Math.sin(f / 7)} />
          )}
          <Burst x={1300} y={262} at={b1 + 16} col={P.spike} n={12} />
          <text x={1598} y={250} textAnchor="middle" fill={P.cream} fontSize={26} fontFamily={FONT_HEAD} opacity={0.65}>攒一罐</text>
        </g>

        {/* 底部：开花周期时间轴 */}
        <line x1={280} y1={800} x2={280 + 1500 * axisP} y2={800} stroke={P.cream} strokeWidth={5}
          strokeLinecap="round" opacity={axisP > 0 ? 0.85 : 0} />
        {axisP > 0.95 && (
          <path d="M 1780 787 L 1814 800 L 1780 813 Z" fill={P.cream} opacity={0.85} />
        )}
        {ticks.map((tk, i) => {
          const k = spring({frame: f - tk.at, fps, config: {damping: 10, stiffness: 160}});
          const kk = clamp01(k);
          return (
            <g key={i} opacity={kk}>
              <line x1={axisX(tk.t)} y1={784} x2={axisX(tk.t)} y2={816} stroke={P.cream} strokeWidth={5} />
              <Shoot x={axisX(tk.t)} y={784} s={0.32} k={kk} />
              <text x={axisX(tk.t)} y={742} textAnchor="middle" fill={P.spike} fontSize={34}
                fontFamily={FONT_HEAD} fontWeight={800}>{tk.label}</text>
            </g>
          );
        })}
        <text x={1780} y={862} textAnchor="end" fill={P.cream} fontSize={28} fontFamily={FONT_HEAD}
          opacity={axisP * 0.6}>开花周期 · 年</text>
      </svg>
      <Stamp text="单次结实！" x={880} y={70} at={beat(0) + 150} color={P.coral} size={52} rot={-5} />
      <Chip text="几十年攒一罐" x={150} y={668} at={b1 + 70} accent={P.amber} />
      <Chip text="地下茎闷头扩张" x={140} y={790} at={b2 + 100} accent={P.rhiz} />
    </FadeIn>
  );
};

/* ============ S2 同步：扁平地图上，日英美三盆竹同一时刻开花 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b0 = beat(0), b1 = beat(1);

  const BM = b0 + 255;                       // 三地同步开花帧
  const bloomK = clamp01((f - BM) / 26);
  const potA = spring({frame: f - 12, fps, config: {damping: 11, stiffness: 140}});
  const potB = spring({frame: f - 28, fps, config: {damping: 11, stiffness: 140}});
  const potC = spring({frame: f - 44, fps, config: {damping: 11, stiffness: 140}});
  const arcP = (d: number) => clamp01((f - b1 - 10 - d) / 32);
  const crateK = spring({frame: f - b1 - 8, fps, config: {damping: 11, stiffness: 140}});
  const clockK = spring({frame: f - b1 - 150, fps, config: {damping: 10, stiffness: 140}});
  const spin = f > b1 + 150 ? (f - b1 - 150) * 9 : 0;

  const plants = [
    {x: 390, y: 590, label: '美国', ly: 712, k: potA},
    {x: 890, y: 400, label: '英国', ly: 522, k: potB},
    {x: 1690, y: 640, label: '日本', ly: 752, k: potC},
  ];
  const arcs = [
    {d: 'M 390 588 Q 560 780 938 796', at: 0},
    {d: 'M 890 398 Q 810 640 950 788', at: 12},
    {d: 'M 1690 638 Q 1420 820 988 798', at: 24},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={2} name={sec.name} accent={P.mint} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 扁平大陆 */}
        <g fill={P.land} opacity={0.9}>
          <ellipse cx={330} cy={470} rx={190} ry={140} />
          <ellipse cx={390} cy={640} rx={150} ry={120} />
          <ellipse cx={270} cy={380} rx={110} ry={80} />
          <ellipse cx={920} cy={330} rx={140} ry={80} />
          <ellipse cx={1000} cy={390} rx={95} ry={60} />
          <ellipse cx={1520} cy={430} rx={240} ry={130} />
          <ellipse cx={1640} cy={540} rx={150} ry={95} />
          <ellipse cx={1380} cy={530} rx={120} ry={85} />
          <ellipse cx={1720} cy={650} rx={72} ry={42} />
        </g>
        {/* 克隆连线 → 同一批苗 */}
        {arcs.map((a, i) => (
          <path key={i} d={a.d} pathLength={100} stroke={P.cream} strokeWidth={4.5} fill="none"
            strokeDasharray="10 14" strokeDashoffset={100 * (1 - arcP(a.at))} opacity={arcP(a.at) > 0 ? 0.8 : 0} />
        ))}
        {/* 克隆苗箱 */}
        <g transform={`translate(960,810) scale(${Math.max(0.001, clamp01(crateK))})`} opacity={clamp01(crateK)}>
          <rect x={-88} y={-14} width={176} height={22} rx={8} fill={P.potD} />
          <rect x={-80} y={6} width={160} height={62} rx={12} fill={P.pot} />
          {[-44, 0, 44].map((dx, i) => (
            <Shoot key={i} x={dx} y={-12} s={0.26} k={1} />
          ))}
        </g>
        <text x={960} y={918} textAnchor="middle" fill={P.cream} fontSize={30} fontFamily={FONT_HEAD}
          opacity={clamp01((f - b1 - 40) / 12)}>同一批克隆苗的后代</text>

        {/* 三盆竹 */}
        {plants.map((p, i) => (
          <g key={i} opacity={clamp01(p.k)}>
            <Pot x={p.x} y={p.y} s={0.9} />
            <Culm x={p.x} y={p.y - 12} h={195} s={0.8 * Math.max(0.001, clamp01(p.k))} rot={i === 1 ? -2 : 2}
              leafK={1} bloomK={bloomK} sway={1.2} seed={i + 3} />
            <text x={p.x} y={p.ly} textAnchor="middle" fill={P.cream} fontSize={34}
              fontFamily={FONT_HEAD} fontWeight={700}>{p.label}</text>
            <Ripple x={p.x} y={p.y - 130} startFrame={BM} color={P.spike} />
          </g>
        ))}

        {/* 基因时钟 */}
        <Clock x={700} y={160} r={100} k={clamp01(clockK)} spin={spin} />
        <text x={700} y={300} textAnchor="middle" fill={P.cream} fontSize={30} fontFamily={FONT_HEAD}
          fontWeight={700} opacity={clamp01((f - b1 - 160) / 12)}>计时器写在基因里</text>
      </svg>
      <Stamp text="同一时刻开花！" x={150} y={95} at={BM + 8} color={P.coral} size={52} rot={-5} />
      <Chip text="上世纪60年代 · 天各一方" x={140} y={210} at={b0 + 150} accent={P.mint} />
      <Chip text="环境只是微调" x={1440} y={130} at={b1 + 190} accent={P.spike} />
    </FadeIn>
  );
};

/* ============ S3 鼠灾：竹米喂肥老鼠→鼠群暴增→啃光庄稼；熊猫断粮卡 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);

  const heapRx = Math.min(95, 16 + Math.max(0, f - b1 - 10) * 0.55);
  const ratsAt = [
    {x: 810, y: 806, s: 1, flip: 1},
    {x: 890, y: 846, s: 0.92, flip: -1},
    {x: 968, y: 800, s: 0.85, flip: 1},
    {x: 838, y: 752, s: 0.78, flip: -1},
    {x: 936, y: 748, s: 0.72, flip: 1},
    {x: 1010, y: 852, s: 0.9, flip: -1},
    {x: 762, y: 866, s: 0.8, flip: 1},
  ];
  const arrow1 = clamp01((f - b1 - 100) / 18);
  const arrow2 = clamp01((f - b1 - 240) / 18);
  const cardX = interpolate(f, [b2 + 12, b2 + 42], [620, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={3} name={sec.name} accent={P.coral} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <rect x={0} y={850} width={1920} height={130} fill={P.soil} />
        <rect x={0} y={850} width={1920} height={12} fill={P.soilD} />

        {/* 第1幕：开花的梨竹 + 竹米落地成堆 */}
        <Culm x={330} y={848} h={430} s={1} rot={-2} leafK={0.9} bloomK={1} sway={1.2} seed={9} />
        <ellipse cx={345} cy={862} rx={heapRx} ry={heapRx * 0.32} fill={P.riceD} />
        <ellipse cx={345} cy={856} rx={heapRx * 0.82} ry={heapRx * 0.24} fill={P.rice} />
        {[0, 1, 2, 3, 4, 5].map((i) => (
          <GrainDrop key={`a${i}`} x0={300 + rand(i, 51) * 90} y0={440 + i * 26} y1={842}
            at={b1 + 12 + i * 16} drift={(rand(i, 52) - 0.5) * 46} s={0.9} hold={300} />
        ))}
        {[0, 1, 2, 3].map((i) => (
          <GrainDrop key={`b${i}`} x0={280 + rand(i, 53) * 120} y0={460 + i * 30} y1={846}
            at={b1 + 130 + i * 20} drift={(rand(i, 54) - 0.5) * 50} s={0.85} hold={220} />
        ))}

        {/* 箭头1：竹米 → 鼠群 */}
        <g opacity={arrow1}>
          <line x1={545} y1={520} x2={700} y2={520} stroke={P.cream} strokeWidth={7} strokeLinecap="round" />
          <path d="M 690 502 L 726 520 L 690 538 Z" fill={P.cream} />
        </g>

        {/* 第2幕：鼠群暴增 */}
        {ratsAt.map((r, i) => {
          const k = spring({frame: f - b1 - 115 - i * 22, fps, config: {damping: 9, stiffness: 170}});
          return <Rat key={i} x={r.x} y={r.y} s={r.s} flip={r.flip} k={k} />;
        })}

        {/* 箭头2：鼠群 → 庄稼 */}
        <g opacity={arrow2}>
          <line x1={1120} y1={520} x2={1290} y2={520} stroke={P.cream} strokeWidth={7} strokeLinecap="round" />
          <path d="M 1280 502 L 1316 520 L 1280 538 Z" fill={P.cream} />
        </g>

        {/* 第3幕：庄稼被啃 */}
        {[1430, 1540, 1650, 1760].map((cx, i) => {
          const eat = clamp01((f - b1 - 278 - i * 18) / 14);
          return <Crop key={i} x={cx} y={848} k={1 - eat * 0.78} lean={eat * (i % 2 ? 9 : -8)} />;
        })}
        <Rat x={1690} y={816} s={1} flip={-1}
          k={spring({frame: f - b1 - 268, fps, config: {damping: 9, stiffness: 170}})} />

        {/* 熊猫断粮卡：滑入 */}
        <g transform={`translate(${cardX},0)`} opacity={clamp01((f - b2 - 12) / 8)}>
          <rect x={1290} y={170} width={550} height={330} rx={38} fill={P.card} stroke={P.spike} strokeWidth={4} />
          <Panda x={1415} y={310} s={1.15} sad />
          <ellipse cx={1415} cy={432} rx={46} ry={18} fill={P.plate} />
          <ellipse cx={1415} cy={428} rx={34} ry={11} fill={P.soilD} />
          <text x={1530} y={250} fill={P.cream} fontSize={32} fontFamily={FONT_HEAD}>四川岷山 · 冷箭竹枯死</text>
          <text x={1530} y={318} fill={P.amber} fontSize={48} fontFamily={FONT_HEAD} fontWeight={800}>大熊猫断粮</text>
          <text x={1530} y={378} fill={P.cream} fontSize={28} fontFamily={FONT_HEAD} opacity={0.75}>全国捐竹子救急</text>
        </g>
      </svg>
      <Chip text="印度米佐拉姆 · 梨竹48年一开" x={140} y={150} at={b1 + 5} accent={P.spike} />
      <Stamp text="鼠群暴增！" x={790} y={620} at={b1 + 150} color={P.coral} size={50} rot={-4} />
      <Stamp text="啃光粮仓" x={1360} y={620} at={b1 + 300} color={P.dryD} size={48} rot={4} />
      <Chip text="这叫竹灾" x={140} y={330} at={b1 + 200} accent={P.coral} />
    </FadeIn>
  );
};

/* ============ S4 重启：枯死→竹米→新苗循环；家里竹子别慌收竹米 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);

  const nA = spring({frame: f - 15, fps, config: {damping: 11, stiffness: 140}});
  const nB = spring({frame: f - 70, fps, config: {damping: 11, stiffness: 140}});
  const nC = spring({frame: f - 125, fps, config: {damping: 11, stiffness: 140}});
  const ringK = clamp01((f - 8) / 24);
  const sweepRot = f * 1.15;

  const CXR = 560, CYR = 505, R = 222;
  const rowK = (d: number) => spring({frame: f - beat(0) - 15 - d, fps, config: {damping: 11, stiffness: 140}});

  const potK = spring({frame: f - b2 - 10, fps, config: {damping: 11, stiffness: 140}});
  const jarK = spring({frame: f - b2 - 24, fps, config: {damping: 11, stiffness: 140}});

  const nodes = [
    {x: CXR, y: CYR - R, k: nA, label: '集体枯死', lx: CXR, ly: CYR - R - 62},
    {x: CXR + R * 0.86, y: CYR + R * 0.5, k: nB, label: '竹米落地', lx: CXR + R * 0.86 + 96, ly: CYR + R * 0.5 + 66},
    {x: CXR - R * 0.86, y: CYR + R * 0.5, k: nC, label: '重新发芽', lx: CXR - R * 0.86 - 30, ly: CYR + R * 0.5 + 66},
  ];
  const benefits = [
    {y: 205, label: '枯死=清空病害', d: 0},
    {y: 295, label: '结籽多到吃不完', d: 30},
    {y: 385, label: '等于系统重装', d: 60},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={4} name={sec.name} accent={P.spike} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 重启环 */}
        <circle cx={CXR} cy={CYR} r={R} fill="none" stroke={P.culmD} strokeWidth={6}
          strokeDasharray="4 26" strokeLinecap="round" opacity={ringK * 0.6} />
        {/* 旋转高亮弧 */}
        <g transform={`rotate(${sweepRot} ${CXR} ${CYR})`} opacity={ringK}>
          <circle cx={CXR} cy={CYR} r={R} fill="none" stroke={P.amber} strokeWidth={9}
            strokeDasharray={`${0.18 * 2 * Math.PI * R} ${0.82 * 2 * Math.PI * R}`} strokeLinecap="round" />
        </g>
        {/* 环上箭头 */}
        {[-30, 90, 210].map((a, i) => {
          const rad = (a * Math.PI) / 180;
          const ax = CXR + Math.cos(rad) * R, ay = CYR + Math.sin(rad) * R;
          return (
            <path key={i} d="M 0 -14 L 24 0 L 0 14 Z" fill={P.cream} opacity={ringK * 0.85}
              transform={`translate(${ax},${ay}) rotate(${a + 90})`} />
          );
        })}
        {/* 三个节点 */}
        <g transform={`translate(${nodes[0].x},${nodes[0].y}) scale(${Math.max(0.001, clamp01(nA))})`}>
          <g transform="rotate(78)"><Culm x={0} y={0} h={170} s={0.62} dryK={1} leafK={0.4} /></g>
        </g>
        <g transform={`translate(${nodes[1].x},${nodes[1].y}) scale(${Math.max(0.001, clamp01(nB))})`}>
          <ellipse cx={0} cy={26} rx={62} ry={16} fill={P.riceD} />
          <Grain x={-30} y={18} rot={-18} /><Grain x={0} y={12} rot={8} /><Grain x={28} y={20} rot={24} />
          <Grain x={-14} y={6} rot={-40} s={0.9} /><Grain x={14} y={8} rot={44} s={0.9} />
        </g>
        <g transform={`translate(${nodes[2].x},${nodes[2].y}) scale(${Math.max(0.001, clamp01(nC))})`}>
          <Shoot x={0} y={30} s={0.72} k={1} />
        </g>
        {nodes.map((n, i) => (
          <text key={i} x={n.lx} y={n.ly} textAnchor="middle" fill={P.cream} fontSize={30}
            fontFamily={FONT_HEAD} fontWeight={700} opacity={clamp01(n.k)}>{n.label}</text>
        ))}
        <Ripple x={nodes[1].x} y={nodes[1].y} startFrame={b1 + 8} color={P.rice} />
        <Ripple x={nodes[2].x} y={nodes[2].y} startFrame={b1 + 30} color={P.leaf} />

        {/* 右侧好处清单 */}
        {benefits.map((b, i) => {
          const k = clamp01(rowK(b.d));
          return (
            <g key={i} transform={`translate(0,${(1 - k) * 30})`} opacity={k}>
              <circle cx={1090} cy={b.y} r={42} fill={P.plate} />
              {i === 0 && (
                <g transform="translate(1090,205)">
                  <ellipse rx={18} ry={12} fill={P.ratD} />
                  <circle cx={16} cy={-4} r={7} fill={P.ratD} />
                  <line x1={-26} y1={22} x2={26} y2={-22} stroke={P.coral} strokeWidth={6} strokeLinecap="round" />
                </g>
              )}
              {i === 1 && (
                <g transform="translate(1090,295) scale(0.62)">
                  <ellipse cx={0} cy={20} rx={42} ry={11} fill={P.riceD} />
                  <Grain x={-20} y={14} rot={-16} /><Grain x={4} y={10} rot={10} /><Grain x={24} y={16} rot={28} />
                </g>
              )}
              {i === 2 && (
                <g transform="translate(1090,385)" stroke={P.ink} strokeWidth={6} fill="none" strokeLinecap="round">
                  <path d="M -20 -6 A 21 21 0 1 1 -8 18" />
                  <path d="M -26 16 L -8 18 L -14 0 Z" fill={P.ink} stroke="none" />
                </g>
              )}
              <text x={1150} y={b.y + 12} fill={P.cream} fontSize={36} fontFamily={FONT_HEAD} fontWeight={700}>
                {b.label}
              </text>
            </g>
          );
        })}

        {/* 右下：盆栽开花 + 收竹米 */}
        <g transform={`translate(0,${(1 - clamp01(potK)) * 40})`} opacity={clamp01(potK)}>
          <Pot x={1520} y={800} s={1.05} />
          <Culm x={1520} y={788} h={285} s={0.85} rot={2} leafK={0.9} bloomK={1} sway={0.9} seed={12} />
        </g>
        <g transform={`translate(0,${(1 - clamp01(jarK)) * 30})`} opacity={clamp01(jarK)}>
          <rect x={1658} y={780} width={104} height={96} rx={18} fill="none" stroke={P.cream} strokeWidth={5} />
          {[0, 1, 2].map((i) => (
            <Grain key={i} x={1682 + i * 26} y={856 - (i % 2) * 10} rot={-20 + i * 26} s={0.9}
              k={clamp01((f - b2 - 60 - i * 26) / 8)} />
          ))}
        </g>
        {[0, 1, 2].map((i) => (
          <GrainDrop key={i} x0={1600 + i * 8} y0={586 + i * 18} y1={782}
            at={b2 + 34 + i * 26} drift={88 + i * 10} s={0.85} hold={30} />
        ))}
      </svg>
      <Stamp text="别慌！" x={1130} y={480} at={b2 + 10} color={P.mint} size={54} rot={-6} />
      <Chip text="想留种就收竹米" x={1060} y={700} at={b2 + 120} accent={P.rice} />
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
        <ellipse cx={960} cy={1010} rx={1050} ry={110} fill="#18250f" />
        <g transform="translate(700,950) rotate(72)">
          <Culm x={0} y={0} h={240} s={0.75} dryK={1} leafK={0.35} bloomK={0.7} />
        </g>
        {[0, 1, 2, 3].map((i) => (
          <Grain key={i} x={760 + rand(i, 61) * 140} y={975 + rand(i, 62) * 30} rot={rand(i, 63) * 80 - 40} s={0.9} />
        ))}
        <Shoot x={1080} y={965} s={1.05} k={clamp01(pop)} />
        <Culm x={1200} y={960} h={150} s={0.62} rot={4} leafK={0.85} sway={0.6} seed={15} />
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -200}}>
        <div style={{fontSize: 70, fontWeight: 800, color: C.text}}>死亡，只是竹子换了个活法</div>
        <div style={{fontSize: 38, color: C.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E211 = ({data = tl}: {data?: any}) => {
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
