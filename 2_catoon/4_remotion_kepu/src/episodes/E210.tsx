import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01, rand} from '../lib/style';
import {Backdrop, Ripple, Stamp, SceneTag, Chip, Subtitles, ProgressBar, FadeIn, Sun, Moon} from '../lib/atoms';
import tl from '../timeline_210.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：可可底 × 金黄 × 橙红 × 残留绿（纯色平涂） ============ */
const P = {
  bg: '#3d2817',      // 主底：暖可可
  bgDeep: '#2b1c10',
  bgLite: '#4a3020',
  card: '#523318',
  plate: '#f6ead2',   // 奶油盘
  ink: '#3f2410',     // 奶油盘上的深字
  cream: '#f6ead2',
  leafG: '#5ba257',   // 残留叶绿
  leafGD: '#3f7d3f',
  gold: '#f0b53f',    // 银杏黄
  orange: '#ee7f3a',
  red: '#cf3d3d',     // 枫叶红
  redD: '#a52a33',
  antho: '#c03a55',   // 花青素红
  chloro: '#4e9b4e',  // 叶绿素块
  carot: '#f5a623',   // 类胡萝卜素
  sugar: '#fdf3da',   // 糖块
  bark: '#6b4426',    // 枝干
  barkD: '#4a2c15',
  cork: '#d9b07c',    // 软木栓
  vein: '#8a5a33',    // 叶脉
  amber: '#ffc94d',
  mint: '#7fe0c3',
  sky: '#8fd3ff',
  night: '#241708',
};

/* ---------- 枫叶：五裂掌状，本地坐标，叶尖朝上，柄朝下 ---------- */
const mapleD =
  'M 0 -100 L 22 -58 L 62 -78 L 52 -40 L 92 -34 L 60 -4 L 78 34 L 30 30 L 34 78 L 0 60 ' +
  'L -34 78 L -30 30 L -78 34 L -60 -4 L -92 -34 L -52 -40 L -62 -78 L -22 -58 Z';

const Leaf = ({x, y, s = 1, rot = 0, col = P.gold, vein = P.vein, op = 1, stem = 1}: {
  x: number; y: number; s?: number; rot?: number; col?: string; vein?: string; op?: number; stem?: number;
}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`} opacity={op}>
    {stem > 0 && <rect x={-6} y={52} width={12} height={58} rx={6} fill={vein} />}
    <path d={mapleD} fill={col} stroke={col} strokeWidth={12} strokeLinejoin="round" />
    <g stroke={vein} strokeWidth={5} strokeLinecap="round" opacity={0.5}>
      <line x1={0} y1={50} x2={0} y2={-84} />
      <line x1={0} y1={14} x2={64} y2={-70} />
      <line x1={0} y1={14} x2={-64} y2={-70} />
      <line x1={0} y1={26} x2={84} y2={-28} />
      <line x1={0} y1={26} x2={-84} y2={-28} />
    </g>
  </g>
);

/* ---------- 变色叶：绿 → 目标色 交叉叠化（k 0..1） ---------- */
const ColorLeaf = ({x, y, s = 1, rot = 0, from = P.leafG, to = P.gold, k = 1, vein = P.vein}: {
  x: number; y: number; s?: number; rot?: number; from?: string; to?: string; k?: number; vein?: string;
}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`}>
    <rect x={-6} y={52} width={12} height={58} rx={6} fill={vein} />
    <path d={mapleD} fill={to} stroke={to} strokeWidth={12} strokeLinejoin="round" opacity={k} />
    <path d={mapleD} fill={from} stroke={from} strokeWidth={12} strokeLinejoin="round" opacity={1 - k} />
    <g stroke={vein} strokeWidth={5} strokeLinecap="round" opacity={0.5}>
      <line x1={0} y1={50} x2={0} y2={-84} />
      <line x1={0} y1={14} x2={64} y2={-70} />
      <line x1={0} y1={14} x2={-64} y2={-70} />
      <line x1={0} y1={26} x2={84} y2={-28} />
      <line x1={0} y1={26} x2={-84} y2={-28} />
    </g>
  </g>
);

/* ---------- 叶绿素马赛克：下班退场（缩没+轻飘） ---------- */
const GreenTile = ({x, y, at, seed}: {x: number; y: number; at: number; seed: number}) => {
  const f = useCurrentFrame();
  const t = f - at;
  const p = clamp01(t / 26);
  if (p >= 1) return null;
  return (
    <g transform={`translate(${x},${y - p * 22}) rotate(${(rand(seed, 4) - 0.5) * p * 150}) scale(${1 - p})`}>
      <rect x={-15} y={-15} width={30} height={30} rx={8} fill={P.chloro} opacity={1 - p} />
    </g>
  );
};

/* ---------- 绿色颗粒飘走（离场余韵） ---------- */
const FleeDot = ({x, y, at, seed}: {x: number; y: number; at: number; seed: number}) => {
  const f = useCurrentFrame();
  const t = f - at;
  if (t < 0 || t > 60) return null;
  const p = t / 60;
  return (
    <circle cx={x + Math.sin(t * 0.15 + seed) * 26 * p} cy={y - p * 190}
      r={9 * (1 - p) + 2} fill={P.chloro} opacity={0.85 * (1 - p)} />
  );
};

/* ---------- 糖块：跳进叶子里 ---------- */
const SugarCube = ({x, y, at, gone}: {x: number; y: number; at: number; gone: number}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = Math.max(0.001, spring({frame: f - at, fps, config: {damping: 9, stiffness: 170}}));
  return (
    <g transform={`translate(${x},${y}) scale(${s * (1 - gone)})`} opacity={1 - gone}>
      <rect x={-13} y={-13} width={26} height={26} rx={6} fill={P.sugar} stroke="#d9c9a0" strokeWidth={3} />
      <rect x={-6} y={-7} width={9} height={7} rx={2} fill="#fff" opacity={0.9} />
    </g>
  );
};

/* ---------- 风弧 ---------- */
const WindArc = ({d, at, dur = 30, col = P.cream}: {d: string; at: number; dur?: number; col?: string}) => {
  const f = useCurrentFrame();
  const t = f - at;
  if (t < 0) return null;
  const p = clamp01(t / dur);
  const out = clamp01((t - dur - 14) / 20);
  return (
    <path d={d} pathLength={100} stroke={col} strokeWidth={7} fill="none" strokeLinecap="round"
      strokeDasharray={100} strokeDashoffset={100 * (1 - p)} opacity={0.75 * (1 - out)} />
  );
};

/* ============ 标题卡 ============ */
const TitleCard = ({data}: {data: any}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 13, mass: 0.7}});
  const o = interpolate(f, [0, 8], [0, 1]);
  const p0 = Math.max(0.001, spring({frame: f - 4, fps, config: {damping: 10, stiffness: 150}}));
  const p1 = Math.max(0.001, spring({frame: f - 9, fps, config: {damping: 10, stiffness: 150}}));
  const p2 = Math.max(0.001, spring({frame: f - 14, fps, config: {damping: 10, stiffness: 150}}));
  return (
    <AbsoluteFill>
      <Backdrop top={P.bg} bottom={P.bg} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <circle cx={960} cy={352} r={205} fill={P.plate} />
        <g transform={`translate(878,338) scale(${p0})`}><Leaf x={0} y={0} s={1.0} rot={-32} col={P.leafG} /></g>
        <g transform={`translate(1044,338) scale(${p1})`}><Leaf x={0} y={0} s={1.0} rot={30} col={P.red} /></g>
        <g transform={`translate(960,356) scale(${p2})`}><Leaf x={0} y={0} s={1.3} rot={-4} col={P.gold} /></g>
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 250}}>
          <div style={{color: P.gold, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 26}}>
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

/* ============ S0 反转：绿马赛克逐个退场露金黄，红色最后现染 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);

  const LX = 960, LY = 470, LS = 2.3;
  const pop = spring({frame: f - 8, fps, config: {damping: 11, stiffness: 130}});

  // 马赛克网格（叶内坐标），按行退场
  const xs = [-84, -56, -28, 0, 28, 56, 84];
  const ys = [-84, -56, -28, 0, 28, 56];
  const tiles: {x: number; y: number; at: number; seed: number}[] = [];
  ys.forEach((yy, r) => xs.forEach((xx, c) => {
    tiles.push({x: xx, y: yy, at: b1 + 12 + r * 30 + c * 4, seed: r * 10 + c});
  }));

  const sway = Math.sin(f / 22) * 2;
  const redK = clamp01((f - b2 - 12) / 60);

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={0} name={sec.name} accent={P.gold} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <defs>
          <clipPath id="s0leaf"><path d={mapleD} /></clipPath>
        </defs>
        <circle cx={960} cy={480} r={330} fill={P.plate} />
        <g transform={`translate(${LX},${LY}) scale(${LS * Math.max(0.001, pop)}) rotate(${sway})`}>
          {/* 底：金黄素颜 */}
          <path d={mapleD} fill={P.gold} stroke={P.gold} strokeWidth={12} strokeLinejoin="round" />
          {/* 红色"现染"斑 */}
          {[[-32, -38, 34], [30, -14, 40], [-18, 30, 30], [34, -58, 26]].map(([bx, by, br], i) => {
            const kk = clamp01(redK * 1.6 - i * 0.3);
            return <ellipse key={i} cx={bx} cy={by} rx={br * kk} ry={br * kk * 0.8}
              fill={P.red} opacity={kk * 0.95} transform={`rotate(${i * 40 - 40} ${bx} ${by})`} />;
          })}
          {/* 绿马赛克层：叶绿素 */}
          <g clipPath="url(#s0leaf)">
            {tiles.map((t, i) => <GreenTile key={i} x={t.x} y={t.y} at={t.at} seed={t.seed} />)}
          </g>
          {/* 叶脉 */}
          <g stroke={P.vein} strokeWidth={5} strokeLinecap="round" opacity={0.5}>
            <line x1={0} y1={50} x2={0} y2={-84} />
            <line x1={0} y1={14} x2={64} y2={-70} />
            <line x1={0} y1={14} x2={-64} y2={-70} />
            <line x1={0} y1={26} x2={84} y2={-28} />
            <line x1={0} y1={26} x2={-84} y2={-28} />
          </g>
          <rect x={-6} y={52} width={12} height={58} rx={6} fill={P.vein} />
        </g>
        {/* 下班飘走的绿颗粒 */}
        {[0, 1, 2, 3, 4, 5].map((i) => (
          <FleeDot key={i} x={860 + rand(i, 5) * 200} y={300 + rand(i, 6) * 200}
            at={b1 + 60 + i * 34} seed={i} />
        ))}
        <Ripple x={960} y={480} startFrame={b2 + 14} color={P.red} />
      </svg>
      <Stamp text="不是染黄的！" x={150} y={240} at={beat(0) + 45} color={P.red} size={52} rot={-7} />
      <Chip text="绿块 = 叶绿素" x={140} y={140} at={beat(0) + 10} accent={P.leafG} />
      <Chip text="黄橙早就藏在里面" x={140} y={790} at={b1 + 30} accent={P.gold} />
      <Chip text="只有红色是现染的" x={1390} y={140} at={b2 + 12} accent={P.antho} />
      <Stamp text="一半卸妆·一半新妆" x={1310} y={790} at={beat(3) + 8} color={P.antho} size={46} rot={5} />
    </FadeIn>
  );
};

/* ============ S1 卸妆：叶绿素颗粒打包顺枝回仓，叶黄素露脸 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b0 = beat(0), b1 = beat(1), b2 = beat(2);

  const LX = 620, LY = 400, LS = 2.1;
  const TIP = {x: LX, y: 638};          // 叶柄尖（枝条上沿）
  const CRATE = {x: 1575, y: 668};      // 仓库入口

  const pop = spring({frame: f - 8, fps, config: {damping: 11, stiffness: 130}});

  // 叶绿素颗粒：本地坐标 → 飞出 → 顺枝到仓库
  const granules = [
    [0, -70], [-30, -50], [30, -50], [-55, -15], [0, -15],
    [55, -15], [-30, 20], [30, 20], [0, 45], [-58, -45],
  ];
  const carots = [[-18, -32], [38, -34], [-44, 8], [20, 12], [-6, -4], [4, -84]];
  const drainK = clamp01((f - (b1 + 10)) / 150);

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={1} name={sec.name} accent={P.gold} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 枝干横档 */}
        <rect x={380} y={640} width={1260} height={58} rx={29} fill={P.bark} />
        <rect x={380} y={640} width={1260} height={20} rx={10} fill={P.barkD} opacity={0.5} />
        {/* 运输虚线 */}
        <line x1={640} y1={668} x2={1440} y2={668} stroke={P.cream} strokeWidth={4}
          strokeDasharray="4 18" strokeLinecap="round" opacity={0.4} />

        {/* 叶子：金底 + 绿罩渐撤 */}
        <g transform={`translate(${LX},${LY}) scale(${LS * Math.max(0.001, pop)})`}>
          <path d={mapleD} fill={P.gold} stroke={P.gold} strokeWidth={12} strokeLinejoin="round" />
          <path d={mapleD} fill={P.leafG} stroke={P.leafG} strokeWidth={12} strokeLinejoin="round"
            opacity={1 - drainK} />
          <g stroke={P.vein} strokeWidth={5} strokeLinecap="round" opacity={0.5}>
            <line x1={0} y1={50} x2={0} y2={-84} />
            <line x1={0} y1={14} x2={64} y2={-70} />
            <line x1={0} y1={14} x2={-64} y2={-70} />
            <line x1={0} y1={26} x2={84} y2={-28} />
            <line x1={0} y1={26} x2={-84} y2={-28} />
          </g>
          <rect x={-6} y={52} width={12} height={58} rx={6} fill={P.vein} />
          {/* 留下来的类胡萝卜素颗粒 */}
          {carots.map(([gx, gy], i) => (
            <circle key={i} cx={gx} cy={gy} r={9} fill={i % 2 ? P.orange : P.carot}
              opacity={0.35 + drainK * 0.65} />
          ))}
        </g>

        {/* 搬走的叶绿素颗粒（绿块） */}
        {granules.map(([gx, gy], i) => {
          const at = b1 + 8 + i * 9;
          const t = f - at;
          if (t < 0) {
            const pulse = 1 + Math.sin(f / 10 + i) * 0.08;
            return (
              <g key={i} transform={`translate(${LX + gx * LS},${LY + gy * LS}) rotate(45) scale(${pulse})`}>
                <rect x={-11} y={-11} width={22} height={22} rx={6} fill={P.chloro} />
              </g>
            );
          }
          const p1 = clamp01(t / 18);                       // →叶柄尖
          const p2 = clamp01((t - 18) / 55);                // →仓库
          const sx = LX + gx * LS, sy = LY + gy * LS;
          const mx = sx + (TIP.x - sx) * p1;
          const my = sy + (TIP.y - sy) * p1;
          const px = mx + (CRATE.x - mx) * p2;
          const py = my + (CRATE.y - my) * p2;
          const gone = clamp01((t - 73) / 10);
          if (gone >= 1) return null;
          return (
            <g key={i} opacity={1 - gone}>
              <rect x={px - 11} y={py - 11} width={22} height={22} rx={6} fill={P.chloro}
                transform={`rotate(${45 + t * 3} ${px} ${py})`} />
            </g>
          );
        })}
        {/* 养分仓库筐（先画筐，再落袋绿点） */}
        <g>
          <rect x={1440} y={636} width={270} height={118} rx={18} fill={P.barkD} stroke={P.bark} strokeWidth={6} />
          <rect x={1440} y={636} width={270} height={22} rx={11} fill={P.bark} />
          <text x={1575} y={618} textAnchor="middle" fill={P.cream} fontSize={30}
            fontFamily={FONT_HEAD} fontWeight={700}>枝干仓库</text>
        </g>
        {/* 仓库里落袋的绿点 */}
        {granules.map((_, i) => {
          const arrT = b1 + 8 + i * 9 + 74;
          const kk = spring({frame: f - arrT, fps, config: {damping: 10, stiffness: 170}});
          const sx2 = 1490 + (i % 5) * 42, sy2 = 712 - Math.floor(i / 5) * 30;
          return <circle key={i} cx={sx2} cy={sy2} r={10 * Math.max(0.001, kk)} fill={P.chloro} />;
        })}
        <Ripple x={TIP.x} y={TIP.y} startFrame={b1 + 10} color={P.cream} />
      </svg>
      <Chip text="叶绿素 = 最贵的零件" x={140} y={150} at={b0 + 12} accent={P.leafG} />
      <Stamp text="先拆最贵的！" x={150} y={320} at={b1 + 12} color={P.orange} size={48} rot={-6} />
      <Chip text="氮·磷·钾 打包回仓" x={1290} y={150} at={b1 + 45} accent={P.mint} />
      <Stamp text="黄色是素颜" x={1240} y={810} at={b2 + 10} color={P.gold} size={48} rot={4} />
      <Chip text="类胡萝卜素一直在" x={140} y={790} at={b2 + 40} accent={P.carot} />
    </FadeIn>
  );
};

/* ============ S2 新妆：白天产糖夜里堵住，糖堆出花青素红 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);

  const LX = 560, LY = 430, LS = 2.3;
  const pop = spring({frame: f - 8, fps, config: {damping: 11, stiffness: 130}});

  const nightK = clamp01((f - (b1 + 105)) / 25);            // 夜里转冷
  const gateK = clamp01((f - (b1 + 115)) / 14);             // 闸门落下
  const redK = clamp01((f - (b1 + 165)) / 90);              // 花青素染红

  const sugarPos = [
    [0, -66], [-34, -46], [34, -44], [-56, -10], [0, -20], [56, -8],
    [-28, 14], [30, 16], [0, 34], [-46, -34], [44, -64], [12, -4],
  ];

  const cardA = spring({frame: f - b2 - 8, fps, config: {damping: 12, stiffness: 130}});
  const cardB = spring({frame: f - b2 - 26, fps, config: {damping: 12, stiffness: 130}});

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={2} name={sec.name} accent={P.antho} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <defs>
          <clipPath id="s2leaf"><path d={mapleD} /></clipPath>
        </defs>
        <circle cx={560} cy={440} r={300} fill={P.plate} />

        {/* 白天☀ ↔ 夜里❄（盘右侧上空） */}
        <g opacity={1 - nightK}>
          <Sun x={940} y={180} r={55} />
        </g>
        <g opacity={nightK}>
          <Moon x={940} y={180} scale={0.8} />
          {[0, 1, 2].map((i) => (
            <g key={i} transform={`translate(${1040 + i * 52},${150 + (i % 2) * 40}) rotate(${f * 2 + i * 40})`}>
              {[0, 60, 120].map((a) => (
                <rect key={a} x={-2.5} y={-16} width={5} height={32} rx={2.5} fill={P.sky}
                  transform={`rotate(${a})`} />
              ))}
            </g>
          ))}
        </g>
        <text x={960} y={290} textAnchor="middle" fill={P.cream} fontSize={26}
          fontFamily={FONT_HEAD} opacity={0.85}>{nightK > 0.5 ? '夜里一冷' : '白天太阳足'}</text>

        {/* 叶子：金底 → 红罩渐染 */}
        <g transform={`translate(${LX},${LY}) scale(${LS * Math.max(0.001, pop)})`}>
          <path d={mapleD} fill={P.gold} stroke={P.gold} strokeWidth={12} strokeLinejoin="round" />
          {/* 红色斑先起 */}
          {[[-30, -40, 30], [32, -14, 36], [-16, 26, 28], [30, -62, 24]].map(([bx, by, br], i) => {
            const kk = clamp01(redK * 1.7 - i * 0.28);
            return <ellipse key={i} cx={bx} cy={by} rx={br * kk} ry={br * kk * 0.82}
              fill={P.redD} opacity={kk * 0.9} transform={`rotate(${i * 35 - 30} ${bx} ${by})`} />;
          })}
          <path d={mapleD} fill={P.red} stroke={P.red} strokeWidth={12} strokeLinejoin="round"
            opacity={redK * 0.55} />
          <g stroke={P.vein} strokeWidth={5} strokeLinecap="round" opacity={0.5}>
            <line x1={0} y1={50} x2={0} y2={-84} />
            <line x1={0} y1={14} x2={64} y2={-70} />
            <line x1={0} y1={14} x2={-64} y2={-70} />
            <line x1={0} y1={26} x2={84} y2={-28} />
            <line x1={0} y1={26} x2={-84} y2={-28} />
          </g>
          <rect x={-6} y={52} width={12} height={58} rx={6} fill={P.vein} />
          {/* 糖块：跳进叶子，越攒越多 */}
          <g clipPath="url(#s2leaf)">
            {sugarPos.map(([gx, gy], i) => (
              <SugarCube key={i} x={gx} y={gy} at={b1 + 8 + i * 11}
                gone={clamp01((f - (b1 + 168 + i * 5)) / 26)} />
            ))}
          </g>
        </g>

        {/* 叶柄出口闸门：夜里关上 */}
        <g transform={`translate(560,722)`} opacity={gateK}>
          <rect x={-46} y={-8} width={92} height={16} rx={8} fill={P.red} transform={`rotate(${38 * gateK}) scale(${gateK})`} />
          <rect x={-46} y={-8} width={92} height={16} rx={8} fill={P.red} transform={`rotate(${-38 * gateK}) scale(${gateK})`} />
        </g>
        <text x={560} y={790} textAnchor="middle" fill={P.red} fontSize={28}
          fontFamily={FONT_HEAD} fontWeight={700} opacity={gateK}>运输通道 · 关闭</text>

        {/* 右侧对照卡 */}
        <g transform={`translate(0,${(1 - clamp01(cardA)) * 50})`} opacity={clamp01(cardA)}>
          <rect x={1180} y={225} width={400} height={240} rx={30} fill={P.card} stroke={P.gold} strokeWidth={4} />
          <Sun x={1260} y={300} r={34} opacity={0.95} />
          <Leaf x={1400} y={330} s={0.72} rot={8} col={P.red} />
          <text x={1380} y={434} textAnchor="middle" fill={P.cream} fontSize={30}
            fontFamily={FONT_HEAD} fontWeight={700}>晴朗干燥 → 红得艳</text>
        </g>
        <g transform={`translate(0,${(1 - clamp01(cardB)) * 50})`} opacity={clamp01(cardB)}>
          <rect x={1180} y={520} width={400} height={240} rx={30} fill={P.card} stroke={P.sky} strokeWidth={4} />
          <g transform="translate(1265,590)">
            <ellipse cx={0} cy={0} rx={52} ry={26} fill={P.sky} />
            <ellipse cx={-30} cy={8} rx={34} ry={20} fill={P.sky} />
            <ellipse cx={30} cy={10} rx={36} ry={22} fill={P.sky} />
            {[-24, 0, 24].map((dx, i) => (
              <line key={i} x1={dx} y1={34 + ((f * 2 + i * 8) % 14)} x2={dx - 7} y2={52 + ((f * 2 + i * 8) % 14)}
                stroke={P.sky} strokeWidth={4} strokeLinecap="round" />
            ))}
          </g>
          <Leaf x={1400} y={625} s={0.72} rot={-10} col={P.bark} />
          <text x={1380} y={729} textAnchor="middle" fill={P.cream} fontSize={30}
            fontFamily={FONT_HEAD} fontWeight={700}>连阴雨 → 枯黄收场</text>
        </g>
      </svg>
      <Stamp text="红色是现做的！" x={140} y={250} at={beat(0) + 14} color={P.antho} size={50} rot={-6} />
      <Chip text="糖运不走 = 原料" x={140} y={140} at={b1 + 20} accent={P.sugar} />
      <Chip text="糖多 → 花青素暴涨" x={140} y={800} at={b1 + 165} accent={P.antho} />
      <Stamp text="天气定成色" x={1240} y={810} at={b2 + 60} color={P.orange} size={44} rot={5} />
    </FadeIn>
  );
};

/* ============ S3 排班：白天变短拉闸，路灯骗过一枝；五花山接力 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);

  const dayK = clamp01((f - 15) / 140);                     // 白昼条缩短
  // 树冠叶片布局（ellipse rx270 ry170 around 1150,340）
  const crown = Array.from({length: 16}).map((_, i) => {
    const a = rand(i, 21) * Math.PI * 2;
    const rr = 0.35 + rand(i, 22) * 0.65;
    return {
      x: 1150 + Math.cos(a) * 265 * rr,
      y: 335 + Math.sin(a) * 165 * rr,
      s: 0.5 + rand(i, 23) * 0.18,
      rot: (rand(i, 24) - 0.5) * 70,
      to: [P.gold, P.orange, P.red][i % 3],
      at: b1 - 35 + i * 11,
      lit: false as boolean,
    };
  });
  crown.forEach((l) => { l.lit = l.x > 1315 && l.y < 560; });  // 灯照区保持绿
  const litN = crown.filter((l) => l.lit).length;

  // 五花山彩冠
  const ridge = 'M 60 900 L 250 800 L 380 858 L 520 792 L 680 852 L 820 806 L 940 900 Z';
  const dots = [
    {x: 205, y: 838, c: P.leafG}, {x: 300, y: 822, c: P.gold}, {x: 420, y: 852, c: P.orange},
    {x: 520, y: 820, c: P.red}, {x: 640, y: 850, c: P.gold}, {x: 760, y: 838, c: P.red},
    {x: 868, y: 862, c: P.orange},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bgDeep} bottom={P.bgDeep} />
      <SceneTag index={3} name={sec.name} accent={P.sky} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 白昼长度计量条 */}
        <rect x={150} y={150} width={440} height={36} rx={18} fill="none" stroke={P.cream} strokeWidth={4} opacity={0.85} />
        <rect x={156} y={156} width={(428 - 190 * dayK)} height={24} rx={12} fill={P.amber} />
        <text x={150} y={132} fill={P.cream} fontSize={28} fontFamily={FONT_HEAD} fontWeight={700}>白天在变短</text>
        <g transform={`translate(${608 + (1 - dayK) * 0},162)`} opacity={dayK}>
          <Moon x={0} y={0} scale={0.34} />
        </g>
        {/* 大树：树干 + 变色树冠 */}
        <rect x={1122} y={430} width={58} height={350} rx={20} fill={P.bark} />
        <rect x={1140} y={430} width={14} height={350} fill={P.barkD} opacity={0.4} />
        <ellipse cx={1150} cy={812} rx={330} ry={42} fill={P.night} />
        {/* 路灯灯光锥（先画在叶下） */}
        <polygon points="1502,318 1290,585 1690,585" fill={P.amber} opacity={0.16} />
        {crown.map((l, i) => {
          const k = l.lit ? 0 : clamp01((f - l.at) / 22);
          return <ColorLeaf key={i} x={l.x} y={l.y} s={l.s} rot={l.rot} to={l.to} k={k} />;
        })}
        {/* 路灯杆 + 灯头 */}
        <rect x={1632} y={330} width={15} height={460} rx={7} fill="#3a2a1a" />
        <path d="M 1639 340 Q 1639 296 1588 296" stroke="#3a2a1a" strokeWidth={15} fill="none" strokeLinecap="round" />
        <circle cx={1568} cy={302} r={24} fill={P.amber} />
        <circle cx={1568} cy={302} r={36} fill={P.amber} opacity={0.25} />
        <rect x={1548} y={278} width={40} height={14} rx={6} fill="#3a2a1a" />

        {/* 五花山：山脊 + 彩冠点 */}
        <path d={ridge} fill="#1d1207" stroke={P.barkD} strokeWidth={5} />
        {dots.map((d, i) => {
          const sp = spring({frame: f - b2 - 12 - i * 9, fps, config: {damping: 10, stiffness: 160}});
          return (
            <g key={i} transform={`translate(${d.x},${d.y}) scale(${Math.max(0.001, sp)})`}>
              <rect x={-3} y={0} width={6} height={20} fill={P.barkD} />
              <circle cy={-14} r={17} fill={d.c} />
            </g>
          );
        })}
      </svg>
      <Stamp text="开关是日照长短" x={150} y={240} at={beat(0) + 25} color={P.orange} size={46} rot={-5} />
      <Chip text={`灯照枝条 · 晚黄一两周`} x={1290} y={640} at={b1 + 40} accent={P.sky} />
      <Stamp text="被路灯骗了" x={1480} y={140} at={b1 + 20} color={P.sky} size={44} rot={5} />
      <Stamp text="五花山 = 卸妆接力" x={120} y={700} at={b2 + 25} color={P.red} size={46} rot={-4} />
      {litN > 0 && <Chip text="其它部位照常变色" x={140} y={600} at={b1 + 90} accent={P.gold} />}
    </FadeIn>
  );
};

/* ============ S4 离职：离层细胞被酶切薄，风一坠就断，软木栓封口 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);

  const PC = {x: 1280, y: 500, r: 300};
  const bandGlow = clamp01((f - 20) / 15) * (0.6 + 0.4 * Math.sin(f / 7)) * (1 - clamp01((f - (b1 + 130)) / 15));
  const dissolve = (i: number) => clamp01((f - (b1 + 15 + i * 14)) / 20);   // 6个离层细胞
  const enzX = interpolate(f, [b1 + 10, b1 + 100], [1246, 1316], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const enzO = clamp01((f - b1 - 8) / 8) * (1 - clamp01((f - (b1 + 112)) / 12));
  const veinO = clamp01((f - (b1 + 95)) / 15);                              // 维管束露出
  const detT = b1 + 160;                                                    // 断离时刻
  const det = f - detT;
  const leafK = det > 0 ? clamp01(det / 90) : 0;                            // 下落
  const leafDy = det > 0 ? Math.min(340, det * det * 0.045) : 0;
  const leafRot = det > 0 ? det * 1.15 : 0;
  const leafO = det > 0 ? 1 - clamp01((leafDy - 240) / 110) : 1;
  const corkK = clamp01((f - detT - 40) / 25);                              // 软木栓封好

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={4} name={sec.name} accent={P.mint} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <defs>
          <clipPath id="s4zoom"><circle cx={PC.x} cy={PC.y} r={PC.r - 8} /></clipPath>
        </defs>

        {/* 左：枝条实景 + 定位虚线 */}
        <path d="M 120 860 Q 300 620 480 350" stroke={P.bark} strokeWidth={24} fill="none" strokeLinecap="round" />
        <path d="M 250 700 Q 360 660 460 560" stroke={P.bark} strokeWidth={14} fill="none" strokeLinecap="round" />
        <Leaf x={480} y={330} s={0.9} rot={-14} col={P.gold} />
        <Leaf x={465} y={545} s={0.8} rot={55} col={P.orange} />
        <circle cx={480} cy={392} r={62} fill="none" stroke={P.mint} strokeWidth={5}
          strokeDasharray="11 9" opacity={0.95} />
        <path d="M 540 400 C 700 390 850 380 975 390" stroke={P.mint} strokeWidth={4}
          strokeDasharray="10 11" fill="none" opacity={0.75} />

        {/* 右：放大窗 */}
        <circle cx={PC.x} cy={PC.y} r={PC.r} fill="#201205" stroke={P.bark} strokeWidth={6} />
        <text x={PC.x} y={PC.y - PC.r - 26} textAnchor="middle" fill={P.mint} fontSize={30}
          fontFamily={FONT_HEAD} letterSpacing={4}>叶柄根部 · 放大看</text>
        <g clipPath="url(#s4zoom)">
          {/* 枝条段（上方） */}
          <rect x={1245} y={180} width={70} height={382} fill={P.bark} />
          <rect x={1245} y={180} width={18} height={382} fill={P.barkD} opacity={0.5} />
          {[200, 250, 300, 350, 400, 450].map((yy) => (
            <line key={yy} x1={1258} y1={yy} x2={1302} y2={yy + 14} stroke={P.barkD} strokeWidth={4} opacity={0.5} />
          ))}
          {/* 离层细胞带（会被酶溶掉） */}
          {[0, 1, 2, 3, 4, 5].map((i) => {
            const d = dissolve(i);
            const w = 12 * (1 - d), h = 20 * (1 - d);
            if (w <= 0.5) return null;
            return <rect key={i} x={1248 + i * 10.4 + (12 - w) / 2} y={560 - h / 2} width={w} height={h}
              rx={4} fill={P.mint} opacity={0.9} />;
          })}
          {/* 离层高亮圈 */}
          <rect x={1242} y={544} width={76} height={34} rx={16} fill="none"
            stroke={P.mint} strokeWidth={4} opacity={bandGlow} />
          {/* 酶小球：沿线啃过去 */}
          {enzO > 0 && (
            <g opacity={enzO} transform={`translate(${enzX},560)`}>
              <circle r={13} fill={P.mint} />
              <circle cx={4} cy={-3} r={4} fill="#fff" opacity={0.8} />
            </g>
          )}
          {/* 维管束细丝 */}
          {[1266, 1280, 1294].map((vx, i) => {
            const snap = det > 14;
            return (
              <g key={i} opacity={veinO}>
                <line x1={vx} y1={562} x2={vx} y2={snap ? 574 : 640} stroke={P.cream}
                  strokeWidth={4} strokeLinecap="round" opacity={snap ? 0.5 : 0.9} />
              </g>
            );
          })}
          {/* 断口软木栓：封口点 */}
          {[0, 1, 2, 3, 4, 5].map((i) => (
            <circle key={i} cx={1253 + i * 10.4} cy={564} r={5.5} fill={P.cork} opacity={corkK} />
          ))}
          {/* 叶柄+叶（下部，会坠走） */}
          {leafO > 0 && (
            <g transform={`translate(0,${leafDy}) rotate(${leafRot} 1280 570)`} opacity={leafO}>
              <rect x={1258} y={570} width={44} height={74} rx={12} fill={P.bark} />
              <Leaf x={1280} y={700} s={0.85} rot={172} col={P.gold} />
            </g>
          )}
          {/* 风弧 */}
          <WindArc d="M 1020 430 Q 1140 420 1220 470" at={detT - 26} />
          <WindArc d="M 1010 560 Q 1150 555 1235 590" at={detT - 12} />
        </g>
        {/* 落地叶堆（窗外） */}
        <ellipse cx={1280} cy={872} rx={220} ry={26} fill={P.night} />
        {det > 55 && <Leaf x={1290 + Math.sin(det / 8) * 4} y={838} s={0.62} rot={75} col={P.gold} />}
        <Leaf x={1150} y={856} s={0.55} rot={-68} col={P.orange} />
        <Leaf x={1430} y={852} s={0.55} rot={60} col={P.red} />
      </svg>
      <Chip text="离层 = 最后一刀" x={140} y={150} at={beat(0) + 15} accent={P.mint} />
      <Stamp text="酶把细胞壁切薄" x={130} y={290} at={b1 + 15} color={P.mint} size={42} rot={-5} />
      <Stamp text="风一坠 · 啪！" x={170} y={560} at={detT + 6} color={P.red} size={48} rot={6} />
      <Chip text="断口早封好软木栓" x={140} y={700} at={detT + 55} accent={P.cork} />
      <Stamp text="主动盖章离职" x={1360} y={140} at={b2 + 15} color={P.orange} size={46} rot={5} />
    </FadeIn>
  );
};

/* ============ 结尾卡 ============ */
const EndCard = () => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 14}});
  const pop = Math.max(0.001, spring({frame: f - 5, fps, config: {damping: 10, stiffness: 140}}));
  const fall = Math.min(1, f / 55);
  return (
    <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
      <Backdrop top={P.bgDeep} bottom={P.bgDeep} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <ellipse cx={960} cy={1012} rx={1050} ry={120} fill={P.night} />
        <g transform={`translate(700,952) scale(${pop})`}><Leaf x={0} y={0} s={1.0} rot={78} col={P.gold} /></g>
        <g transform={`translate(960,975) scale(${pop})`}><Leaf x={0} y={0} s={0.9} rot={-72} col={P.orange} /></g>
        <g transform={`translate(1210,958) scale(${pop})`}><Leaf x={0} y={0} s={1.0} rot={62} col={P.red} /></g>
        {/* 还在落的叶 */}
        <Leaf x={520 + Math.sin(f / 16) * 40} y={240 + fall * 500} s={0.7}
          rot={-20 + Math.sin(f / 14) * 30} col={P.red} op={1 - clamp01((f - 60) / 20)} />
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -200}}>
        <div style={{fontSize: 70, fontWeight: 800, color: C.text}}>秋叶是树写给冬天的预算表</div>
        <div style={{fontSize: 38, color: C.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E210 = ({data = tl}: {data?: any}) => {
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
