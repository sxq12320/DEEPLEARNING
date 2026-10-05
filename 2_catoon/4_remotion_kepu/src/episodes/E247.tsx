import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01} from '../lib/style';
import {Backdrop, Stamp, SceneTag, Chip, Subtitles, ProgressBar, FadeIn, Stars} from '../lib/atoms';
import tl from '../timeline_247.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：深跑道蓝 × 心肺珊瑚 × 氧气青 × 数据琥珀 ============ */
const P = {
  bg: '#14253f',
  bgDeep: '#0c1930',
  bgLite: '#1d3a5e',
  panel: '#1b3355',
  card: '#2b4b74',
  ink: '#0a1526',
  cream: '#f7f0dd',
  muted: '#9db3d0',
  heart: '#ff6b66',
  heartD: '#d84a48',
  oxy: '#60d4ff',
  oxyD: '#37a3d4',
  amber: '#ffc95e',
  amberD: '#dd9a38',
  leaf: '#7fd98f',
  leafD: '#4aa968',
  lung: '#ffa4b8',
  vio: '#b8a5f0',
  acid: '#e6dc6a',
  track: '#e07b54',
};

/* ---------- 平滑折线 ---------- */
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

const Curve = ({pts, prog, col, lw = 6, dash, smooth = 2}: {
  pts: [number, number][]; prog: number; col: string; lw?: number; dash?: string; smooth?: number;
}) => {
  const sm = chaikin(pts, smooth);
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

const PanelTitle = ({x, y, text, col = P.muted}: {x: number; y: number; text: string; col?: string}) => (
  <text x={x} y={y} fill={col} fontSize={25} fontFamily={FONT_HEAD}>{text}</text>
);

/* ---------- 楼梯剖面 ---------- */
const stairPts = (x: number, y: number, steps: number, dx: number, dy: number): [number, number][] => {
  const pts: [number, number][] = [[x, y], [x + dx, y]];
  for (let i = 1; i <= steps; i++) {
    pts.push([x + dx * i, y - dy * (i - 1)]);
    pts.push([x + dx * i, y - dy * i]);
  }
  pts.push([x + dx * (steps + 1), y - dy * steps]);
  return pts;
};

/* ---------- 爬楼梯小人（侧面，前倾喘） ---------- */
const Climber = ({x, y, s = 1, col = P.cream, wig = 0, tired = 0}: {
  x: number; y: number; s?: number; col?: string; wig?: number; tired?: number;
}) => {
  const f = useCurrentFrame();
  const b = Math.sin(f * 0.45 + wig) * (3 + tired * 3);
  const lean = 10 + tired * 10;
  return (
    <g transform={`translate(${x},${y + b * 0.4}) scale(${s})`}>
      {/* 后腿 */}
      <line x1={0} y1={-2} x2={-24} y2={30} stroke={col} strokeWidth={10} strokeLinecap="round" />
      <line x1={-24} y1={30} x2={-34} y2={50} stroke={col} strokeWidth={9} strokeLinecap="round" />
      {/* 前腿抬上一级 */}
      <line x1={0} y1={-2} x2={22} y2={16} stroke={col} strokeWidth={10} strokeLinecap="round" />
      <line x1={22} y1={16} x2={44} y2={6} stroke={col} strokeWidth={9} strokeLinecap="round" />
      {/* 躯干前倾 */}
      <line x1={0} y1={-2} x2={lean} y2={-48} stroke={col} strokeWidth={13} strokeLinecap="round" />
      {/* 手臂摆动 */}
      <line x1={lean - 1} y1={-40} x2={36 + b} y2={-22} stroke={col} strokeWidth={8} strokeLinecap="round" />
      <line x1={lean - 1} y1={-40} x2={-14 - b} y2={-18} stroke={col} strokeWidth={8} strokeLinecap="round" />
      {/* 头 + 张嘴喘 */}
      <circle cx={lean + 6} cy={-64} r={16} fill={col} />
      <circle cx={lean + 18} cy={-58} r={4.6} fill={P.ink} />
      <circle cx={lean + 4} cy={-68} r={3.2} fill={P.ink} />
    </g>
  );
};

/* ---------- 喘气团 ---------- */
const Puffs = ({x, y, on = 1, dir = 1, n = 3}: {x: number; y: number; on?: number; dir?: number; n?: number}) => {
  const f = useCurrentFrame();
  if (on <= 0) return null;
  return (
    <g opacity={on}>
      {Array.from({length: n}).map((_, i) => {
        const t = (f * 2 + i * 22) % 66;
        const p = t / 66;
        return (
          <circle key={i} cx={x + dir * p * 74} cy={y - p * 40 + Math.sin(t * 0.2) * 5}
            r={6 + p * 12} fill="none" stroke={P.oxy} strokeWidth={3}
            opacity={0.8 * (1 - p)} />
        );
      })}
    </g>
  );
};

/* ---------- 汗滴 ---------- */
const Sweat = ({x, y, s = 1, col = P.oxy}: {x: number; y: number; s?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <path d="M 0 -16 Q 12 2 12 10 A 12 12 0 0 1 -12 10 Q -12 2 0 -16 Z" fill={col} />
  </g>
);

/* ---------- 跳动的心脏 ---------- */
const HeartIco = ({x, y, s = 1, col = P.heart, rate = 1, on = 1, ecg = true}: {
  x: number; y: number; s?: number; col?: string; rate?: number; on?: number; ecg?: boolean;
}) => {
  const f = useCurrentFrame();
  const ph = f * 0.16 * rate;
  const pump = 1 + Math.pow(Math.max(0, Math.sin(ph)), 3) * 0.16;
  return (
    <g transform={`translate(${x},${y}) scale(${s * pump})`} opacity={on}>
      <path d="M 0 -14 C -18 -42 -56 -36 -56 -2 C -56 20 -26 32 0 52 C 26 32 56 20 56 -2 C 56 -36 18 -42 0 -14 Z"
        fill={col} />
      <path d="M 0 -14 C -18 -42 -56 -36 -56 -2 L -40 -2 C -36 -22 -14 -30 0 -14 Z" fill="#ffffff" opacity={0.16} />
      {ecg && (
        <polyline points="-30,4 -18,4 -12,-10 -4,16 4,-18 10,4 30,4" fill="none"
          stroke="#fff" strokeWidth={5} strokeLinecap="round" strokeLinejoin="round" opacity={0.9} />
      )}
    </g>
  );
};

/* ---------- 肺 ---------- */
const LungsIco = ({x, y, s = 1, col = P.lung, rate = 1}: {
  x: number; y: number; s?: number; col?: string; rate?: number;
}) => {
  const f = useCurrentFrame();
  const k = 0.82 + (Math.sin(f * 0.22 * rate) * 0.5 + 0.5) * 0.3;
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      <rect x={-7} y={-56} width={14} height={34} rx={7} fill={col} />
      <line x1={0} y1={-24} x2={-22} y2={-6} stroke={col} strokeWidth={9} strokeLinecap="round" />
      <line x1={0} y1={-24} x2={22} y2={-6} stroke={col} strokeWidth={9} strokeLinecap="round" />
      <g transform={`scale(${k},${k})`}>
        <ellipse cx={-30} cy={16} rx={26} ry={34} fill={col} transform="rotate(10 -30 16)" />
        <ellipse cx={30} cy={16} rx={26} ry={34} fill={col} transform="rotate(-10 30 16)" />
        <ellipse cx={-34} cy={6} rx={10} ry={18} fill="#fff" opacity={0.25} transform="rotate(10 -34 6)" />
        <ellipse cx={34} cy={6} rx={10} ry={18} fill="#fff" opacity={0.25} transform="rotate(-10 34 6)" />
      </g>
    </g>
  );
};

/* ---------- 血滴（携氧） ---------- */
const BloodDrop = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <path d="M 0 -40 Q 34 -2 34 16 A 34 34 0 0 1 -34 16 Q -34 -2 0 -40 Z" fill={P.heartD} />
    <circle cx={-10} cy={12} r={9} fill={P.heart} />
    <circle cx={12} cy={20} r={7} fill={P.heart} />
    <circle cx={6} cy={-2} r={6} fill={P.heart} />
    <circle cx={-10} cy={12} r={3.4} fill={P.heartD} opacity={0.8} />
    <circle cx={12} cy={20} r={2.6} fill={P.heartD} opacity={0.8} />
  </g>
);

/* ---------- 肌肉（屈臂） ---------- */
const MuscleIco = ({x, y, s = 1, col = P.amber}: {x: number; y: number; s?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    {/* 前臂 + 拳 */}
    <rect x={-10} y={-14} width={56} height={26} rx={13} fill={col} />
    <circle cx={46} cy={-2} r={17} fill={col} />
    {/* 上臂 + 肱二头肌峰 */}
    <path d="M -34 34 L -34 -22 Q -34 -44 -12 -44 Q 8 -44 8 -24 L 8 12 Q 8 34 -14 34 Z" fill={col} />
    <circle cx={-14} cy={-40} r={16} fill={col} />
    <path d="M -24 -44 Q -14 -56 -2 -46" fill="none" stroke={P.ink} strokeWidth={4}
      strokeLinecap="round" opacity={0.35} />
  </g>
);

/* ---------- ECG 心率线 ---------- */
const ecgPts = (x0: number, y0: number, w: number, beats: number, h: number): [number, number][] => {
  const pts: [number, number][] = [[x0, y0]];
  const bw = w / beats;
  const shape: [number, number][] = [
    [0.30, 0], [0.38, -0.14], [0.44, 0], [0.50, -0.95], [0.56, 0.34], [0.64, 0], [0.78, -0.18], [1, 0],
  ];
  for (let b = 0; b < beats; b++) {
    for (const [px, py] of shape) pts.push([x0 + b * bw + px * bw, y0 + py * h]);
  }
  return pts;
};

/* ---------- 流动箭头（气流/血流） ---------- */
const FlowArrow = ({x, y, on = 1, dir = 1, col = P.oxy, n = 4, len = 60, spread = 90, vertical = false}: {
  x: number; y: number; on?: number; dir?: number; col?: string; n?: number; len?: number; spread?: number; vertical?: boolean;
}) => {
  const f = useCurrentFrame();
  if (on <= 0) return null;
  return (
    <g opacity={on}>
      {Array.from({length: n}).map((_, i) => {
        const cyc = (f * 1.7 + i * 18) % 54;
        const p = cyc / 54;
        const off = (i - (n - 1) / 2) * (spread / Math.max(1, n - 1));
        const ax = vertical ? x + off : x + dir * p * len;
        const ay = vertical ? y + dir * p * len : y + off;
        const dx = vertical ? 0 : dir * 26;
        const dy = vertical ? dir * 26 : 0;
        return (
          <g key={i} opacity={0.9 * (1 - p)}>
            <line x1={ax} y1={ay} x2={ax + dx * 0.7} y2={ay + dy * 0.7} stroke={col}
              strokeWidth={6} strokeLinecap="round" />
            <polygon points={`${ax + dx},${ay + dy} ${ax + dx * 0.55 - dy * 0.32},${ay + dy * 0.55 - dx * 0.32} ${ax + dx * 0.55 + dy * 0.32},${ay + dy * 0.55 + dx * 0.32}`}
              fill={col} />
          </g>
        );
      })}
    </g>
  );
};

/* ---------- 血容量水箱 ---------- */
const Tank = ({x, y, w = 210, h = 470, lvl = 0.9}: {
  x: number; y: number; w?: number; h?: number; lvl?: number;
}) => {
  const f = useCurrentFrame();
  const wy = -h * lvl;
  const wig = Math.sin(f * 0.2) * 4;
  return (
    <g transform={`translate(${x},${y})`}>
      <rect x={-w / 2} y={-h} width={w} height={h} rx={20} fill={P.bgDeep} stroke={P.card} strokeWidth={4} />
      {/* 血液填充 */}
      <path d={`M ${-w / 2 + 12} ${wy + wig} Q ${-w / 4} ${wy - 10 + wig} 0 ${wy + wig} Q ${w / 4} ${wy + 10 + wig} ${w / 2 - 12} ${wy + wig} L ${w / 2 - 12} -12 L ${-w / 2 + 12} -12 Z`}
        fill={P.heartD} />
      <path d={`M ${-w / 2 + 12} ${wy + wig} Q ${-w / 4} ${wy - 10 + wig} 0 ${wy + wig} Q ${w / 4} ${wy + 10 + wig} ${w / 2 - 12} ${wy + wig}`}
        fill="none" stroke={P.heart} strokeWidth={5} />
      {/* 红细胞 */}
      {[[-40, -60], [30, -110], [-10, -170], [45, -230], [-48, -290], [8, -350], [-32, -410]].map(([dx, dy], i) =>
        dy > wy + 30 ? (
          <circle key={i} cx={dx} cy={dy + wig * 0.4} r={10} fill={P.heart} opacity={0.85} />
        ) : null
      )}
      {/* 刻度 */}
      {[0.25, 0.5, 0.75, 1].map((m) => (
        <line key={m} x1={w / 2 - 30} y1={-h * m} x2={w / 2 - 12} y2={-h * m}
          stroke={P.muted} strokeWidth={3.4} />
      ))}
    </g>
  );
};

/* ---------- 说话气泡 ---------- */
const Bubble = ({x, y, w, text, at, accent = P.leaf, tailX}: {
  x: number; y: number; w: number; text: string; at: number; accent?: string; tailX?: number;
}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = Math.max(0.001, spring({frame: f - at, fps, config: {damping: 11, stiffness: 150}}));
  return (
    <g transform={`translate(${x},${y}) scale(${s})`} opacity={clamp01(s * 2)}>
      <polygon points={`${(tailX ?? -w * 0.3)},34 ${(tailX ?? -w * 0.3) - 26},66 ${(tailX ?? -w * 0.3) + 26},36`}
        fill={P.bgDeep} stroke={accent} strokeWidth={4} />
      <rect x={-w / 2} y={-34} width={w} height={70} rx={20} fill={P.bgDeep} stroke={accent} strokeWidth={4} />
      <text x={0} y={12} textAnchor="middle" fontSize={26} fill={P.cream}
        fontFamily={FONT_HEAD} fontWeight={800}>{text}</text>
    </g>
  );
};

/* ---------- 供应链节点 ---------- */
const ChainNode = ({x, y, w, k, col, icon, top, bottom}: {
  x: number; y: number; w: number; k: number; col: string; icon: React.ReactNode; top: string; bottom: string;
}) => (
  <g transform={`translate(${x},${y}) scale(${Math.max(0.001, k)})`} opacity={clamp01(k * 2)}>
    <rect x={-w / 2} y={-96} width={w} height={192} rx={22} fill={P.bgDeep} stroke={col} strokeWidth={4.4} />
    <g transform="translate(0,-30)">{icon}</g>
    <text x={0} y={52} textAnchor="middle" fontSize={27} fill={col}
      fontFamily={FONT_HEAD} fontWeight={900}>{top}</text>
    <text x={0} y={82} textAnchor="middle" fontSize={20} fill={P.muted}
      fontFamily={FONT_HEAD}>{bottom}</text>
  </g>
);

const NodeArrow = ({x0, x1, y, k}: {x0: number; x1: number; y: number; k: number}) => (
  <g opacity={k}>
    <line x1={x0} y1={y} x2={x0 + (x1 - x0 - 24) * k} y2={y} stroke={P.muted}
      strokeWidth={6} strokeLinecap="round" />
    {k > 0.85 && <polygon points={`${x1},${y} ${x1 - 22},${y - 11} ${x1 - 22},${y + 11}`} fill={P.muted} />}
  </g>
);

/* ============ 标题卡：楼梯 + 喘的小人 + 狂跳的心 ============ */
const TitleCard = ({data}: {data: any}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 13, mass: 0.7}});
  const o = interpolate(f, [0, 8], [0, 1]);
  const pA = Math.max(0.001, spring({frame: f - 3, fps, config: {damping: 12, stiffness: 130}}));
  const pB = Math.max(0.001, spring({frame: f - 12, fps, config: {damping: 10, stiffness: 140}}));
  const stairK = clamp01((f - 2) / 36);
  return (
    <AbsoluteFill>
      <Backdrop top={P.bg} bottom={P.bg} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={16} from={0} dur={22} />
        {/* 顶部心率条 */}
        <g opacity={clamp01(f / 14)}>
          <rect x={330} y={96} width={1260} height={86} rx={22} fill={P.bgDeep} stroke={P.card} strokeWidth={4} />
          <Curve pts={ecgPts(360, 140, 1140, 5, 56)} prog={clamp01((f - 4) / 50)} col={P.heart} lw={5.5} smooth={0} />
          <text x={1560} y={146} textAnchor="end" fontSize={24} fill={P.heart} fontFamily={FONT_HEAD} fontWeight={800}>
            心率飙升
          </text>
        </g>
        {/* 大楼梯 */}
        <g transform={`translate(960,430) scale(${pA}) translate(-960,-430)`} opacity={clamp01(pA)}>
          <Curve pts={stairPts(430, 560, 6, 150, 62)} prog={stairK} col={P.card} lw={14} />
          <line x1={430} y1={576} x2={1620} y2={576} stroke={P.card} strokeWidth={8} strokeLinecap="round" opacity={0.6} />
          {/* 小人爬到一半，喘 */}
          <Climber x={880} y={430} s={1.5} tired={1} wig={2} />
          <Puffs x={920} y={352} on={clamp01((f - 16) / 12)} />
          <Sweat x={846} y={330} s={0.9} />
          <Sweat x={910} y={300} s={0.7} />
          {/* 台阶数标 */}
          <text x={470} y={620} fontSize={24} fill={P.muted} fontFamily={FONT_HEAD}>六层楼 · 突击测试</text>
        </g>
        {/* 大心脏 */}
        <g transform={`translate(1470,360) scale(${pB})`} opacity={clamp01(pB)}>
          <circle cx={0} cy={0} r={126} fill={P.panel} stroke={P.heart} strokeWidth={5} />
          <HeartIco x={0} y={-6} s={1.5} rate={2.2} />
          <text x={0} y={96} textAnchor="middle" fontSize={24} fill={P.heart}
            fontFamily={FONT_HEAD} fontWeight={800}>心脏快蹦出来</text>
        </g>
        {/* 肺部小标 */}
        <g transform={`translate(470,330) scale(${pB})`} opacity={clamp01(pB)}>
          <circle cx={0} cy={0} r={104} fill={P.panel} stroke={P.oxy} strokeWidth={5} />
          <LungsIco x={0} y={8} s={1.1} rate={2.4} />
          <text x={0} y={86} textAnchor="middle" fontSize={22} fill={P.oxy}
            fontFamily={FONT_HEAD} fontWeight={800}>肺先扛不住</text>
        </g>
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 540}}>
          <div style={{color: P.oxy, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 22}}>
            冷知识 · 第 {data.episode} 期
          </div>
          <div style={{color: C.text, fontSize: 74, fontWeight: 900, fontFamily: FONT_HEAD, letterSpacing: 6}}>
            {data.title}
          </div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ============ S0 喘的不是肌肉 · 是心肺 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);
  const popA = Math.max(0.001, spring({frame: f - 8, fps, config: {damping: 12, stiffness: 120}}));
  const popB = Math.max(0.001, spring({frame: f - b1 - 4, fps, config: {damping: 12, stiffness: 120}}));
  const cmpK = clamp01((f - b2 - 2) / 46);
  const stairK = clamp01((f - 12) / 40);

  const VBar = ({x, hMax, k, col, label, tag, tagCol}: {
    x: number; hMax: number; k: number; col: string; label: string; tag: string; tagCol: string;
  }) => (
    <g transform={`translate(${x},0)`} opacity={k}>
      <rect x={-46} y={800 - hMax * k} width={92} height={Math.max(2, hMax * k)} rx={14} fill={col} />
      {k > 0.82 && (
        <text x={0} y={800 - hMax + 48} textAnchor="middle" fontSize={29} fill={tagCol}
          fontFamily={FONT_HEAD} fontWeight={900}>{tag}</text>
      )}
      <text x={0} y={838} textAnchor="middle" fontSize={23} fill={P.cream}
        fontFamily={FONT_HEAD} fontWeight={800}>{label}</text>
    </g>
  );

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={0} name={sec.name} accent={P.heart} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：突击爬楼 */}
        <g transform={`translate(540,330) scale(${popA}) translate(-540,-330)`} opacity={clamp01(popA)}>
          <rect x={120} y={140} width={840} height={380} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <PanelTitle x={160} y={196} text="躺三个月 · 突击爬六楼" />
          <Curve pts={stairPts(190, 480, 6, 106, 52)} prog={stairK} col={P.card} lw={11} />
          <Climber x={560} y={428} s={1.28} tired={1} wig={1} />
          <Puffs x={596} y={360} on={clamp01((f - 24) / 12)} />
          <Sweat x={522} y={350} s={0.85} />
          <HeartIco x={790} y={250} s={0.9} rate={2.4} on={clamp01((f - 30) / 10)} />
          <text x={790} y={330} textAnchor="middle" fontSize={22} fill={P.heart}
            fontFamily={FONT_HEAD} fontWeight={800}>心脏像要蹦出来</text>
          <text x={160} y={498} fontSize={21} fill={P.muted} fontFamily={FONT_HEAD}>
            腿倒还行 · 肺先扛不住
          </text>
        </g>

        {/* 右：VO2max 仪表卡 */}
        <g transform={`translate(1390,330) scale(${popB}) translate(-1390,-330)`} opacity={clamp01(popB)}>
          <rect x={1000} y={140} width={800} height={380} rx={20} fill={P.panel}
            stroke={popB > 0.5 ? P.oxy : P.card} strokeWidth={4} />
          <PanelTitle x={1040} y={196} text="决定喘多凶的指标" col={P.oxy} />
          <text x={1100} y={330} fontSize={86} fill={P.oxy} fontFamily={FONT_HEAD} fontWeight={900}>
            VO₂max
          </text>
          <text x={1100} y={374} fontSize={24} fill={P.cream} fontFamily={FONT_HEAD} fontWeight={800}>
            最大摄氧量
          </text>
          <text x={1100} y={452} fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>
            整条氧气供应链的总能力
          </text>
          <text x={1100} y={486} fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>
            不是某一个器官的事
          </text>
          {/* 迷你链条 */}
          <g transform="translate(1520,310)">
            <LungsIco x={-140} y={6} s={0.56} rate={1.6} />
            <NodeArrow x0={-96} x1={-46} y={0} k={clamp01((f - b1 - 40) / 18)} />
            <BloodDrop x={0} y={2} s={0.72} />
            <NodeArrow x0={46} x1={96} y={0} k={clamp01((f - b1 - 52) / 18)} />
            <HeartIco x={144} y={-2} s={0.6} rate={2} />
            <text x={0} y={112} textAnchor="middle" fontSize={20} fill={P.muted}
              fontFamily={FONT_HEAD}>肺 → 血 → 心 → 肌肉</text>
          </g>
        </g>

        {/* 下：退化速度对比 */}
        <rect x={120} y={550} width={1680} height={330} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <PanelTitle x={160} y={606} text="停训三个月 · 各系统剩余量（示意）" />
        <line x1={960} y1={590} x2={960} y2={816} stroke={P.card} strokeWidth={3} strokeDasharray="6 12" />
        <text x={560} y={656} textAnchor="middle" fontSize={24} fill={P.muted} fontFamily={FONT_HEAD}>
          肌肉力量 · 还剩大头
        </text>
        <text x={1360} y={656} textAnchor="middle" fontSize={24} fill={P.muted} fontFamily={FONT_HEAD}>
          心肺体能 · 先崩一批
        </text>
        <g opacity={cmpK}>
          <VBar x={560} hMax={140} k={cmpK} col={P.amber} label="肌肉力量" tag="≈85%" tagCol={P.ink} />
          <VBar x={1360} hMax={110} k={cmpK} col={P.heart} label="心肺 VO₂max" tag="≈78%" tagCol="#fff" />
          {cmpK > 0.7 && (
            <g>
              <path d="M 700 652 L 700 700" stroke={P.amber} strokeWidth={6} strokeLinecap="round" />
              <path d="M 700 700 L 686 676 M 700 700 L 714 676" stroke={P.amber} strokeWidth={6}
                strokeLinecap="round" />
              <text x={740} y={700} fontSize={23} fill={P.amber} fontFamily={FONT_HEAD} fontWeight={800}>
                掉得慢
              </text>
              <path d="M 1460 660 L 1500 710" stroke={P.heart} strokeWidth={7} strokeLinecap="round" />
              <path d="M 1500 710 L 1490 686 M 1500 710 L 1476 700" stroke={P.heart} strokeWidth={7}
                strokeLinecap="round" />
              <text x={1540} y={700} fontSize={23} fill={P.heart} fontFamily={FONT_HEAD} fontWeight={800}>
                掉得快
              </text>
              <text x={960} y={856} textAnchor="middle" fontSize={23} fill={P.cream}
                fontFamily={FONT_HEAD} fontWeight={800}>
                不是废了 · 是心肺先一步退化
              </text>
            </g>
          )}
        </g>
      </svg>
      <Chip text="喘的是心肺不是腿" x={1330} y={66} at={b1 + 10} accent={P.oxy} />
      <Stamp text="体能掉得比肌肉快" x={520} y={86} at={b2 + 20} color={P.heartD} size={40} rot={-3} />
    </FadeIn>
  );
};

/* ============ S1 金标准掉得有多快 ============ */
const VO2_PTS: [number, number][] = [
  [280, 590], [486, 672], [692, 712], [1100, 748], [1500, 770],
];
const STR_PTS: [number, number][] = [
  [280, 590], [486, 614], [692, 640], [1100, 682], [1500, 704],
];

const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);
  const chainK = [0, 1, 2, 3].map((i) =>
    Math.max(0.001, spring({frame: f - 10 - i * 16, fps, config: {damping: 12, stiffness: 130}})));
  const vo2K = clamp01((f - b1 - 10) / 110);
  const strK = clamp01((f - b1 - 40) / 110);
  const tagK = clamp01((f - b2 - 6) / 26);

  return (
    <FadeIn>
      <Backdrop top={P.bgDeep} bottom={P.bgDeep} />
      <SceneTag index={1} name={sec.name} accent={P.oxy} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 上：供氧链四环节 */}
        <rect x={120} y={140} width={1680} height={300} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <PanelTitle x={160} y={196} text="氧气供应链 · 一环弱全盘弱" />
        <ChainNode x={340} y={300} w={300} k={chainK[0]} col={P.lung}
          icon={<LungsIco x={0} y={0} s={0.72} rate={1.6} />} top="肺换气" bottom="把氧装进来" />
        <NodeArrow x0={500} x1={560} y={300} k={chainK[1]} />
        <ChainNode x={760} y={300} w={300} k={chainK[1]} col={P.heart}
          icon={<BloodDrop x={0} y={-4} s={0.78} />} top="血液携氧" bottom="血红蛋白运输" />
        <NodeArrow x0={920} x1={980} y={300} k={chainK[2]} />
        <ChainNode x={1180} y={300} w={300} k={chainK[2]} col={P.heartD}
          icon={<HeartIco x={0} y={-4} s={0.72} rate={1.8} />} top="心脏泵血" bottom="把氧送到位" />
        <NodeArrow x0={1340} x1={1400} y={300} k={chainK[3]} />
        <ChainNode x={1600} y={300} w={300} k={chainK[3]} col={P.amber}
          icon={<MuscleIco x={-6} y={8} s={0.72} />} top="肌肉提氧" bottom="线粒体烧掉它" />
        <text x={960} y={424} textAnchor="middle" fontSize={21} fill={P.muted} fontFamily={FONT_HEAD}>
          运动生理学公认的心肺耐力金标准 · 普通人每公斤每分钟 35–45 毫升 · 顶尖运动员 80+
        </text>

        {/* 下：停训掉幅曲线 */}
        <rect x={120} y={470} width={1680} height={410} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <PanelTitle x={160} y={526} text="停训后掉幅 · 体能打骨折" />
        {/* 坐标 */}
        <line x1={230} y1={810} x2={1560} y2={810} stroke={P.muted} strokeWidth={3.4} />
        <line x1={230} y1={560} x2={230} y2={810} stroke={P.muted} strokeWidth={3.4} />
        <text x={206} y={594} textAnchor="end" fontSize={19} fill={P.muted} fontFamily={FONT_HEAD}>100%</text>
        <text x={206} y={782} textAnchor="end" fontSize={19} fill={P.muted} fontFamily={FONT_HEAD}>75%</text>
        {['0', '2周', '4周', '12周'].map((t, i) => (
          <text key={t} x={[280, 486, 692, 1500][i]} y={840} textAnchor="middle" fontSize={20}
            fill={P.muted} fontFamily={FONT_HEAD}>停训 {t}</text>
        ))}
        {/* 力量线（慢） */}
        <Curve pts={STR_PTS} prog={strK} col={P.amber} lw={6.5} dash="14 12" />
        {/* VO2max 线（快） */}
        <Curve pts={VO2_PTS} prog={vo2K} col={P.heart} lw={7} />
        {vo2K > 0.4 && (
          <g opacity={vo2K}>
            <text x={430} y={640} fontSize={22} fill={P.heart} fontFamily={FONT_HEAD} fontWeight={900}>
              -10% 两周
            </text>
            <text x={700} y={682} fontSize={22} fill={P.heart} fontFamily={FONT_HEAD} fontWeight={900}>
              -15% 四周
            </text>
            <text x={1330} y={742} fontSize={22} fill={P.heart} fontFamily={FONT_HEAD} fontWeight={900}>
              -20%+ 三个月
            </text>
          </g>
        )}
        {/* 图例 */}
        <g opacity={Math.max(vo2K, strK)}>
          <line x1={1560} y1={756} x2={1612} y2={756} stroke={P.heart} strokeWidth={7} strokeLinecap="round" />
          <text x={1624} y={764} fontSize={21} fill={P.heart} fontFamily={FONT_HEAD} fontWeight={800}>VO₂max</text>
          <line x1={1560} y1={796} x2={1612} y2={796} stroke={P.amber} strokeWidth={6}
            strokeDasharray="12 10" strokeLinecap="round" />
          <text x={1624} y={804} fontSize={21} fill={P.amber} fontFamily={FONT_HEAD} fontWeight={800}>肌肉力量</text>
        </g>
        {tagK > 0 && (
          <g opacity={tagK} transform={`translate(0,${(1 - tagK) * 24})`}>
            <rect x={1050} y={540} width={700} height={60} rx={16} fill={P.bgDeep} stroke={P.heart} strokeWidth={4} />
            <text x={1400} y={580} textAnchor="middle" fontSize={25} fill={P.cream}
              fontFamily={FONT_HEAD} fontWeight={900}>
              力量停一月保八九成 · 体能已打骨折
            </text>
          </g>
        )}
      </svg>
      <Chip text="停训掉幅比想象狠" x={1380} y={66} at={b1 + 16} accent={P.heart} />
      <Stamp text="两周一成 · 三月两成" x={500} y={86} at={b2 + 24} color={P.heartD} size={40} rot={-3} />
    </FadeIn>
  );
};

/* ============ S2 血容量先缩水 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);
  const popA = Math.max(0.001, spring({frame: f - 8, fps, config: {damping: 12, stiffness: 120}}));
  const lvl = 0.95 - 0.26 * clamp01((f - 20) / (b1 + 30));
  const shrinkK = clamp01((f - b1 - 4) / 40);
  const heartS = 1.3 - 0.34 * shrinkK;
  const rateK = clamp01((f - b1 - 30) / 60);
  const bpm = Math.round(68 + 46 * rateK);
  const feelK = clamp01((f - b2 - 4) / 26);

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={2} name={sec.name} accent={P.heart} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：血容量水箱 */}
        <g transform={`translate(360,520) scale(${popA}) translate(-360,-520)`} opacity={clamp01(popA)}>
          <rect x={120} y={140} width={480} height={720} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <PanelTitle x={160} y={196} text="血容量 · 头号撤退项" />
          <Tank x={360} y={800} w={200} h={520} lvl={lvl} />
          <text x={360} y={286} textAnchor="middle" fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>
            血浆量
          </text>
          <g opacity={shrinkK}>
            <text x={360} y={332} textAnchor="middle" fontSize={30} fill={P.heart}
              fontFamily={FONT_HEAD} fontWeight={900}>
              {Math.round(lvl * 100)}%
            </text>
            <text x={540} y={420} textAnchor="end" fontSize={21} fill={P.heart}
              fontFamily={FONT_HEAD} fontWeight={800}>
              第一周就开缩 ↓
            </text>
          </g>
          <text x={160} y={840} fontSize={20} fill={P.muted} fontFamily={FONT_HEAD}>
            血多装的氧多 · 一缩全链缩水
          </text>
        </g>

        {/* 中：心肌缩小 · 每搏变少 */}
        <rect x={640} y={140} width={600} height={720} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <PanelTitle x={680} y={196} text="心肌也缩水 · 每搏输出降" />
        <g transform={`translate(940,430) scale(${popA})`} opacity={clamp01(popA)}>
          {/* 停训前大心（虚线轮廓） */}
          <g opacity={0.6}>
            <path d="M 0 -14 C -18 -42 -56 -36 -56 -2 C -56 20 -26 32 0 52 C 26 32 56 20 56 -2 C 56 -36 18 -42 0 -14 Z"
              transform="scale(1.34)" fill="none" stroke={P.muted} strokeWidth={4.4} strokeDasharray="12 9" />
            <text x={0} y={160} textAnchor="middle" fontSize={21} fill={P.muted}
              fontFamily={FONT_HEAD}>灰影 = 停训前大小</text>
          </g>
          {/* 当前心（缩小，跳更快） */}
          <HeartIco x={0} y={0} s={heartS} rate={1.6 + rateK} />
          {shrinkK > 0.5 && (
            <text x={0} y={104} textAnchor="middle" fontSize={20} fill={P.heart}
              fontFamily={FONT_HEAD} fontWeight={800}>现在 · 小了一号</text>
          )}
          {shrinkK > 0.3 && (
            <g opacity={shrinkK}>
              <line x1={-92} y1={-86} x2={-58} y2={-58} stroke={P.heart} strokeWidth={5}
                strokeLinecap="round" />
              <line x1={92} y1={-86} x2={58} y2={-58} stroke={P.heart} strokeWidth={5}
                strokeLinecap="round" />
              <text x={0} y={-106} textAnchor="middle" fontSize={23} fill={P.heart}
                fontFamily={FONT_HEAD} fontWeight={900}>引擎排量缩小</text>
            </g>
          )}
        </g>
        {/* 每搏输出对比 */}
        <g transform="translate(940,680)" opacity={shrinkK}>
          <text x={-170} y={-26} fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>每搏输出</text>
          <rect x={-170} y={0} width={280} height={30} rx={15} fill={P.card} />
          <rect x={-170} y={46} width={200} height={30} rx={15} fill={P.heart} />
          <text x={-170} y={110} fontSize={21} fill={P.heart} fontFamily={FONT_HEAD} fontWeight={800}>
            泵血天花板 ↓
          </text>
        </g>

        {/* 右：心率代偿 */}
        <rect x={1280} y={140} width={520} height={720} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <PanelTitle x={1320} y={196} text="次数硬凑总数" />
        <text x={1540} y={330} textAnchor="middle" fontSize={92} fill={P.heart}
          fontFamily={FONT_HEAD} fontWeight={900}>
          {bpm}
        </text>
        <text x={1540} y={372} textAnchor="middle" fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>
          心率 bpm · 静止也偏快
        </text>
        {/* ECG */}
        <rect x={1320} y={420} width={440} height={130} rx={18} fill={P.bgDeep}
          stroke={P.card} strokeWidth={3.4} />
        <Curve pts={ecgPts(1340, 486, 400, Math.round(3 + rateK * 2), 60)} prog={0.999}
          col={P.heart} lw={5} smooth={0} />
        <g opacity={rateK}>
          <text x={1540} y={630} textAnchor="middle" fontSize={24} fill={P.cream}
            fontFamily={FONT_HEAD} fontWeight={800}>
            心率 ↑ × 每搏 ↓
          </text>
          <text x={1540} y={668} textAnchor="middle" fontSize={21} fill={P.muted} fontFamily={FONT_HEAD}>
            拼命凑总数 · 还是不够
          </text>
        </g>
        {feelK > 0 && (
          <g opacity={feelK} transform={`translate(0,${(1 - feelK) * 24})`}>
            <rect x={1320} y={710} width={440} height={110} rx={18} fill={P.bgDeep}
              stroke={P.heart} strokeWidth={4} />
            <text x={1540} y={758} textAnchor="middle" fontSize={24} fill={P.heart}
              fontFamily={FONT_HEAD} fontWeight={900}>跳得飞快却使不上劲</text>
            <text x={1540} y={794} textAnchor="middle" fontSize={20} fill={P.muted}
              fontFamily={FONT_HEAD}>喘只是表象</text>
          </g>
        )}
      </svg>
      <Chip text="血容量是地基" x={1400} y={66} at={30} accent={P.heart} />
      <Stamp text="心率飙升在硬凑" x={520} y={86} at={b2 + 20} color={P.heartD} size={40} rot={-3} />
    </FadeIn>
  );
};

/* ============ S3 喘是在填氧债 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);
  const popA = Math.max(0.001, spring({frame: f - 8, fps, config: {damping: 12, stiffness: 120}}));
  const gapK = clamp01((f - 30) / 40);
  const fixK = [0, 1].map((i) =>
    Math.max(0.001, spring({frame: f - b1 - 6 - i * 24, fps, config: {damping: 12, stiffness: 130}})));
  const relK = clamp01((f - b2 - 4) / 50);
  const acidDrop = (f * 1.6) % 60;

  return (
    <FadeIn>
      <Backdrop top={P.bgDeep} bottom={P.bgDeep} />
      <SceneTag index={3} name={sec.name} accent={P.amber} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左上：供需缺口 */}
        <g transform={`translate(470,330) scale(${popA}) translate(-470,-330)`} opacity={clamp01(popA)}>
          <rect x={120} y={140} width={700} height={340} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <PanelTitle x={160} y={196} text="供氧跟不上耗氧" />
          {/* 供给柱（矮） */}
          <g opacity={gapK}>
            <rect x={300} y={330} width={120} height={110} rx={14} fill={P.oxyD} />
            <rect x={300} y={330} width={120} height={110 * (1 - 0.35 * gapK)} rx={14} fill={P.oxy} opacity={0} />
            <text x={360} y={318} textAnchor="middle" fontSize={22} fill={P.oxy}
              fontFamily={FONT_HEAD} fontWeight={800}>供氧能力</text>
            <text x={360} y={466} textAnchor="middle" fontSize={20} fill={P.muted}
              fontFamily={FONT_HEAD}>缩水中</text>
            {/* 需求柱（高） */}
            <rect x={560} y={240} width={120} height={200} rx={14} fill={P.heart} />
            <text x={620} y={228} textAnchor="middle" fontSize={22} fill={P.heart}
              fontFamily={FONT_HEAD} fontWeight={800}>耗氧需求</text>
            <text x={620} y={466} textAnchor="middle" fontSize={20} fill={P.muted}
              fontFamily={FONT_HEAD}>强度定死</text>
            {/* 缺口括号 */}
            {gapK > 0.6 && (
              <g>
                <line x1={430} y1={332} x2={430} y2={240} stroke={P.amber} strokeWidth={5}
                  strokeDasharray="8 8" />
                <line x1={540} y1={332} x2={540} y2={240} stroke={P.amber} strokeWidth={5}
                  strokeDasharray="8 8" />
                <rect x={432} y={256} width={106} height={50} rx={14} fill={P.amber} />
                <text x={485} y={291} textAnchor="middle" fontSize={28} fill={P.ink}
                  fontFamily={FONT_HEAD} fontWeight={900}>氧债</text>
              </g>
            )}
          </g>
          <text x={160} y={466} fontSize={21} fill={P.muted} fontFamily={FONT_HEAD}>
            拆东墙补西墙
          </text>
        </g>

        {/* 右上：两条填债路 */}
        <rect x={860} y={140} width={940} height={340} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <PanelTitle x={900} y={196} text="身体的两条应急路" />
        {/* 无氧酵解 */}
        <g transform={`translate(1070,330) scale(${fixK[0]})`} opacity={clamp01(fixK[0] * 2)}>
          <rect x={-150} y={-120} width={330} height={230} rx={20} fill={P.bgDeep}
            stroke={P.acid} strokeWidth={4} />
          <MuscleIco x={-60} y={-30} s={0.78} />
          {[0, 1, 2].map((i) => {
            const t = (acidDrop + i * 20) % 60;
            return <circle key={i} cx={30 + i * 26} cy={-40 + t * 0.9} r={6}
              fill={P.acid} opacity={0.85 * (1 - t / 60)} />;
          })}
          <text x={15} y={42} textAnchor="middle" fontSize={24} fill={P.acid}
            fontFamily={FONT_HEAD} fontWeight={900}>无氧酵解顶上</text>
          <text x={15} y={78} textAnchor="middle" fontSize={20} fill={P.muted}
            fontFamily={FONT_HEAD}>代价 · 乳酸狂产</text>
        </g>
        {/* 疯狂通气 */}
        <g transform={`translate(1560,330) scale(${fixK[1]})`} opacity={clamp01(fixK[1] * 2)}>
          <rect x={-180} y={-120} width={360} height={230} rx={20} fill={P.bgDeep}
            stroke={P.oxy} strokeWidth={4} />
          <LungsIco x={-40} y={-16} s={0.9} rate={2.6} />
          <FlowArrow x={-96} y={-16} on={fixK[1]} dir={-1} col={P.oxy} n={3} len={46} spread={70} />
          <text x={0} y={42} textAnchor="middle" fontSize={24} fill={P.oxy}
            fontFamily={FONT_HEAD} fontWeight={900}>疯狂通气吸气</text>
          <text x={0} y={78} textAnchor="middle" fontSize={20} fill={P.muted}
            fontFamily={FONT_HEAD}>= 你听到的喘</text>
        </g>

        {/* 下：相对强度 */}
        <rect x={120} y={510} width={1680} height={370} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <PanelTitle x={160} y={566} text="同一座六楼 · 相对强度完全两码事" />
        <Curve pts={stairPts(170, 828, 3, 78, 34)} prog={1} col={P.card} lw={9} />
        <Climber x={300} y={764} s={1.1} wig={3} />
        <g opacity={relK}>
          {/* 以前 */}
          <text x={420} y={676} fontSize={23} fill={P.muted} fontFamily={FONT_HEAD}>三个月前</text>
          <rect x={560} y={648} width={1180} height={44} rx={22} fill={P.bgDeep} />
          <rect x={560} y={648} width={1180 * 0.6 * relK} height={44} rx={22} fill={P.leaf} />
          <text x={560 + 1180 * 0.6 + 20} y={680} fontSize={25} fill={P.leaf}
            fontFamily={FONT_HEAD} fontWeight={900}>≈60% · 还行</text>
          {/* 现在 */}
          <text x={420} y={788} fontSize={23} fill={P.cream} fontFamily={FONT_HEAD} fontWeight={800}>
            现在
          </text>
          <rect x={560} y={760} width={1180} height={44} rx={22} fill={P.bgDeep} />
          <rect x={560} y={760} width={1180 * 0.92 * relK} height={44} rx={22} fill={P.heart} />
          {relK > 0.85 && (
            <text x={560 + 1180 * 0.92 - 90} y={792} fontSize={25} fill="#fff"
              fontFamily={FONT_HEAD} fontWeight={900}>≈90% · 要命</text>
          )}
          <text x={560} y={848} fontSize={21} fill={P.muted} fontFamily={FONT_HEAD}>
            同样的负荷压在缩水的容量上 · 主观体验从「还行」变「要命」
          </text>
        </g>
      </svg>
      <Chip text="喘 = 填氧债的最后手段" x={1290} y={66} at={b1 + 16} accent={P.amber} />
      <Stamp text="发动机排量变小了" x={500} y={86} at={b2 + 26} color={P.amberD} size={40} rot={-3} />
    </FadeIn>
  );
};

/* ============ S4 账得慢慢还 ============ */
const RECOV_PTS: [number, number][] = [
  [1090, 430], [1200, 420], [1330, 380], [1480, 320], [1620, 280], [1740, 268],
];
const RUSH_PTS: [number, number][] = [
  [1090, 430], [1180, 340], [1260, 300], [1340, 380], [1440, 430],
];

const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);
  const popA = Math.max(0.001, spring({frame: f - 8, fps, config: {damping: 12, stiffness: 120}}));
  const stairK = clamp01((f - 14) / 50);
  const recK = clamp01((f - b1 - 8) / 130);
  const rushK = clamp01((f - b1 - 60) / 80);
  const cardK = [0, 1, 2].map((i) =>
    Math.max(0.001, spring({frame: f - b2 - 4 - i * 16, fps, config: {damping: 12, stiffness: 135}})));

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={4} name={sec.name} accent={P.leaf} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：渐进台阶 */}
        <g transform={`translate(540,340) scale(${popA}) translate(-540,-340)`} opacity={clamp01(popA)}>
          <rect x={120} y={140} width={840} height={400} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <PanelTitle x={160} y={196} text="从低强度起步 · 每周加量 ≤10%" />
          <Curve pts={stairPts(190, 490, 8, 80, 30)} prog={stairK} col={P.leafD} lw={10} />
          {['W1', 'W3', 'W6', 'W10'].map((t, i) => (
            <text key={t} x={190 + 80 * [1, 3, 6, 9][i]} y={522 - 30 * [1, 3, 6, 8][i]}
              textAnchor="middle" fontSize={19} fill={P.muted} fontFamily={FONT_HEAD}
              opacity={stairK}>{t}</text>
          ))}
          <Climber x={385} y={378} s={1.12} wig={1} />
          <Bubble x={560} y={300} w={360} text="能聊天 · 不能唱歌" at={b1 - 10}
            accent={P.leaf} tailX={-140} />
          <text x={160} y={522} fontSize={21} fill={P.muted} fontFamily={FONT_HEAD}>
            步行 · 慢骑这类低冲击先行
          </text>
          {stairK > 0.9 && (
            <g opacity={stairK}>
              <rect x={740} y={230} width={190} height={52} rx={16} fill={P.bgDeep}
                stroke={P.leaf} strokeWidth={4} />
              <text x={835} y={265} textAnchor="middle" fontSize={22} fill={P.leaf}
                fontFamily={FONT_HEAD} fontWeight={900}>坡度缓</text>
            </g>
          )}
        </g>

        {/* 右：恢复曲线 */}
        <rect x={1000} y={140} width={800} height={400} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <PanelTitle x={1040} y={196} text="重建有氧系统 · 8–12 周见起色" col={P.leaf} />
        <line x1={1060} y1={460} x2={1770} y2={460} stroke={P.muted} strokeWidth={3.2} />
        <line x1={1060} y1={240} x2={1060} y2={460} stroke={P.muted} strokeWidth={3.2} />
        <text x={1040} y={252} textAnchor="end" fontSize={18} fill={P.muted} fontFamily={FONT_HEAD}>高</text>
        <text x={1040} y={452} textAnchor="end" fontSize={18} fill={P.muted} fontFamily={FONT_HEAD}>低</text>
        {/* 突击冲高回落 */}
        <Curve pts={RUSH_PTS} prog={rushK} col={P.heart} lw={5.5} dash="12 10" />
        {rushK > 0.8 && (
          <text x={1310} y={280} textAnchor="middle" fontSize={21} fill={P.heart}
            fontFamily={FONT_HEAD} fontWeight={800}>突击猛练 · 冲高回落</text>
        )}
        {/* 规律恢复 */}
        <Curve pts={RECOV_PTS} prog={recK} col={P.leaf} lw={7} />
        {recK > 0.75 && (
          <g opacity={recK}>
            <circle cx={1740} cy={268} r={9} fill={P.leaf} />
            <text x={1620} y={240} textAnchor="middle" fontSize={22} fill={P.leaf}
              fontFamily={FONT_HEAD} fontWeight={900}>规律训练慢慢爬回</text>
          </g>
        )}
        {['0', '4周', '8周', '12周'].map((t, i) => (
          <text key={t} x={[1090, 1300, 1520, 1740][i]} y={490} textAnchor="middle" fontSize={19}
            fill={P.muted} fontFamily={FONT_HEAD}>{t}</text>
        ))}
        <text x={1040} y={522} fontSize={20} fill={P.muted} fontFamily={FONT_HEAD}>
          年纪越大恢复越慢 · 更不该完全停
        </text>

        {/* 下：三条硬结论 */}
        <rect x={120} y={570} width={1680} height={310} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <PanelTitle x={160} y={626} text="心肺的账 · 三条还款守则" />
        {[
          {x: 400, k: cardK[0], col: P.leaf, top: '维持 ≪ 重建', sub: '每周两三次就保住大半'},
          {x: 960, k: cardK[1], col: P.amber, top: '渐进不突击', sub: '加量超一成容易崩'},
          {x: 1520, k: cardK[2], col: P.oxy, top: '开始还就回血', sub: '心肺可塑性一直在'},
        ].map((c, i) => (
          <g key={i} transform={`translate(${c.x},740) scale(${c.k})`} opacity={clamp01(c.k * 2)}>
            <rect x={-240} y={-90} width={480} height={180} rx={22} fill={P.bgDeep}
              stroke={c.col} strokeWidth={4.4} />
            <text x={0} y={-16} textAnchor="middle" fontSize={34} fill={c.col}
              fontFamily={FONT_HEAD} fontWeight={900}>{c.top}</text>
            <text x={0} y={38} textAnchor="middle" fontSize={22} fill={P.cream}
              fontFamily={FONT_HEAD}>{c.sub}</text>
          </g>
        ))}
      </svg>
      <Chip text="能聊天不能唱歌 = 有氧区间" x={1230} y={66} at={b1 - 4} accent={P.leaf} />
      <Stamp text="慢就是快" x={560} y={86} at={b2 + 30} color={P.leafD} size={44} rot={-3} />
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
        <Stars n={18} from={0} dur={18} />
        <g transform={`translate(960,640) scale(${pop}) translate(-960,-640)`} opacity={clamp01(pop)}>
          {/* 楼梯口重新出发 */}
          <Curve pts={stairPts(660, 700, 6, 100, 44)} prog={1} col={P.card} lw={11} />
          <line x1={660} y1={716} x2={1370} y2={716} stroke={P.card} strokeWidth={7}
            strokeLinecap="round" opacity={0.6} />
          <Climber x={800} y={692} s={1.25} tired={0} />
          <HeartIco x={1180} y={560} s={1.1} rate={1.1} />
          <Bubble x={1330} y={360} w={330} text="这周先爬三层" at={8} accent={P.leaf} tailX={-100} />
          <text x={960} y={790} textAnchor="middle" fontSize={23} fill={P.oxy}
            fontFamily={FONT_HEAD}>别等喘了才想起心肺</text>
        </g>
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -330}}>
        <div style={{fontSize: 54, fontWeight: 800, color: C.text}}>心肺的账 · 慢慢还</div>
        <div style={{fontSize: 36, color: P.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E247 = ({data = tl}: {data?: any}) => {
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
