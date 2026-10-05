import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01} from '../lib/style';
import {Backdrop, Stamp, SceneTag, Chip, Subtitles, ProgressBar, FadeIn, Stars} from '../lib/atoms';
import tl from '../timeline_243.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：泳池夜蓝 × 冰青 × 饥饿珊瑚 × 高热量琥珀 ============ */
const P = {
  bg: '#0e3049',
  bgDeep: '#092238',
  bgLite: '#17486c',
  panel: '#123d5e',
  card: '#1f5680',
  ink: '#071e2e',
  cream: '#f5efd9',
  muted: '#9dbdd8',
  coral: '#ff7d5c',
  coralD: '#d95f42',
  amber: '#ffc25c',
  amberD: '#e09a38',
  teal: '#63e2c6',
  tealD: '#35a98d',
  sky: '#6fc6f2',
  skyD: '#3b92c6',
  water: '#2b7cb3',
  ice: '#a9e6f7',
  vio: '#c4a8f4',
  rose: '#ff9fb2',
  noodle: '#f2d38c',
};

/* ---------- 水波线 ---------- */
const WaveLine = ({x, y, w, n = 7, amp = 9, col = P.ice, lw = 5, speed = 0.12}: {
  x: number; y: number; w: number; n?: number; amp?: number; col?: string; lw?: number; speed?: number;
}) => {
  const f = useCurrentFrame();
  const dx = Math.sin(f * speed) * 12;
  const seg = w / n;
  let d = `M ${x} ${y}`;
  for (let i = 0; i < n; i++) d += ` q ${seg / 2} ${-amp * 2} ${seg} 0`;
  return <path d={d} transform={`translate(${dx},0)`} fill="none" stroke={col}
    strokeWidth={lw} strokeLinecap="round" />;
};

/* ---------- 游泳小人（俯泳） ---------- */
const Swimmer = ({x, y, s = 1, col = P.cream, wig = 0, flip = 1}: {
  x: number; y: number; s?: number; col?: string; wig?: number; flip?: number;
}) => {
  const f = useCurrentFrame();
  const k = Math.sin(f * 0.5 + wig) * 6;
  const arm = Math.sin(f * 0.42 + wig) * 20;
  return (
    <g transform={`translate(${x},${y + k * 0.4}) scale(${flip * s},${s})`}>
      <line x1={-56} y1={8} x2={14} y2={0} stroke={col} strokeWidth={16} strokeLinecap="round" />
      <circle cx={38} cy={-8} r={17} fill={col} />
      <path d="M 21 -14 A 17 17 0 0 1 55 -11 L 52 -3 Q 38 -12 25 -1 Z" fill={P.coral} />
      <circle cx={44} cy={-6} r={3.4} fill={P.ink} />
      <line x1={10} y1={-6} x2={74 + arm * 0.4} y2={-16 + arm * 0.3} stroke={col}
        strokeWidth={10} strokeLinecap="round" />
      <line x1={4} y1={6} x2={-28} y2={-20 - arm * 0.4} stroke={col} strokeWidth={9} strokeLinecap="round" />
      <line x1={-56} y1={8} x2={-92} y2={-4 + k} stroke={col} strokeWidth={9} strokeLinecap="round" />
      <line x1={-56} y1={10} x2={-88} y2={20 - k} stroke={col} strokeWidth={9} strokeLinecap="round" />
    </g>
  );
};

/* ---------- 跑步小人 ---------- */
const Runner = ({x, y, s = 1, col = P.cream, wig = 0, flip = 1}: {
  x: number; y: number; s?: number; col?: string; wig?: number; flip?: number;
}) => {
  const f = useCurrentFrame();
  const a = Math.sin(f * 0.65 + wig) * 16;
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

/* ---------- 汗滴 ---------- */
const Sweat = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <path d="M 0 -16 Q 12 2 12 10 A 12 12 0 0 1 -12 10 Q -12 2 0 -16 Z" fill={P.sky} />
  </g>
);

/* ---------- 气泡 ---------- */
const Bubbles = ({x, y, n = 4}: {x: number; y: number; n?: number}) => {
  const f = useCurrentFrame();
  return (
    <g>
      {Array.from({length: n}).map((_, i) => {
        const t = (f * 1.4 + i * 30) % 120;
        return (
          <circle key={i} cx={x + Math.sin(t * 0.1 + i) * 14 + i * 6} cy={y - t}
            r={5 + i * 1.6} fill="none" stroke={P.ice} strokeWidth={2.6}
            opacity={0.85 * (1 - t / 120)} />
        );
      })}
    </g>
  );
};

/* ---------- 体温计（竖直，lvl 0~1） ---------- */
const Thermo = ({x, y, h = 250, lvl = 0.75, col = P.coral, label}: {
  x: number; y: number; h?: number; lvl?: number; col?: string; label?: string;
}) => (
  <g transform={`translate(${x},${y})`}>
    <rect x={-17} y={-h} width={34} height={h} rx={17} fill={P.bgDeep} stroke={P.muted} strokeWidth={4} />
    <circle cy={8} r={30} fill={P.bgDeep} stroke={P.muted} strokeWidth={4} />
    <rect x={-8} y={-h * lvl} width={16} height={h * lvl + 8} rx={8} fill={col} />
    <circle cy={8} r={20} fill={col} />
    {[0.25, 0.5, 0.75].map((m) => (
      <line key={m} x1={-17} y1={-h * m} x2={-6} y2={-h * m} stroke={P.muted} strokeWidth={3} />
    ))}
    {label && <text x={0} y={58} textAnchor="middle" fontSize={22} fill={col}
      fontFamily={FONT_HEAD} fontWeight={800}>{label}</text>}
  </g>
);

/* ---------- 食物图标 ---------- */
const Bowl = ({x, y, s = 1, steam = 1}: {x: number; y: number; s?: number; steam?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    {steam > 0 && [0, 1, 2].map((i) => (
      <path key={i} d={`M ${-26 + i * 26} ${-34} q 8 -14 0 -26`} fill="none" stroke={P.cream}
        strokeWidth={5} strokeLinecap="round" opacity={0.55 * steam} />
    ))}
    <path d="M -72 -8 L 72 -8 Q 62 54 0 58 Q -62 54 -72 -8 Z" fill={P.coralD} />
    <path d="M -72 -8 L 72 -8 Q 68 24 40 38 Q -40 38 -68 -8 Z" fill={P.coral} opacity={0.9} />
    <ellipse cy={-8} rx={72} ry={16} fill={P.cream} />
    <ellipse cy={-8} rx={56} ry={10} fill={P.noodle} />
    <line x1={26} y1={-40} x2={86} y2={-70} stroke={P.amberD} strokeWidth={7} strokeLinecap="round" />
    <line x1={34} y1={-30} x2={92} y2={-56} stroke={P.amberD} strokeWidth={7} strokeLinecap="round" />
    <rect x={-26} y={52} width={52} height={10} rx={5} fill={P.coralD} />
  </g>
);

const Burger = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <path d="M -62 -6 A 62 46 0 0 1 62 -6 L 62 -2 L -62 -2 Z" fill={P.amber} />
    {[-30, 0, 30].map((dx) => <circle key={dx} cx={dx} cy={-24} r={4.4} fill={P.cream} />)}
    <rect x={-66} y={-2} width={132} height={16} rx={8} fill={P.tealD} />
    <rect x={-62} y={14} width={124} height={20} rx={9} fill={P.coralD} />
    <rect x={-58} y={34} width={116} height={22} rx={11} fill={P.amberD} />
  </g>
);

const Skewer = ({x, y, s = 1, rot = 0}: {x: number; y: number; s?: number; rot?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`}>
    <line x1={0} y1={-72} x2={0} y2={66} stroke={P.cream} strokeWidth={6} strokeLinecap="round" />
    {[-44, -6, 32].map((dy, i) => (
      <rect key={i} x={-24} y={dy - 15} width={48} height={30} rx={9}
        fill={i % 2 ? P.coral : P.coralD} transform={`rotate(${i % 2 ? 8 : -8})`} />
    ))}
    <circle cy={-72} r={7} fill={P.cream} />
  </g>
);

const Egg = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <ellipse rx={52} ry={40} fill={P.cream} />
    <ellipse cx={-8} cy={-6} rx={44} ry={30} fill="#fff" opacity={0.5} />
    <circle cx={6} cy={0} r={19} fill={P.amber} />
  </g>
);

const Cup = ({x, y, s = 1, col = P.sky}: {x: number; y: number; s?: number; col?: string}) => {
  const f = useCurrentFrame();
  const w = Math.sin(f * 0.3) * 4;
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      <path d={`M -18 -52 q ${8 + w} -14 0 -26 M 8 -52 q ${-8 + w} -14 0 -26`} fill="none"
        stroke={P.cream} strokeWidth={5} strokeLinecap="round" opacity={0.6} />
      <path d="M -48 -40 L 48 -40 L 40 44 Q 36 56 24 56 L -24 56 Q -36 56 -40 44 Z" fill={col} />
      <rect x={48} y={-26} width={22} height={40} rx={11} fill="none" stroke={col} strokeWidth={7} />
      <rect x={-42} y={-30} width={84} height={18} rx={9} fill={P.cream} opacity={0.85} />
    </g>
  );
};

const Bread = ({x, y, s = 1, wig = 0}: {x: number; y: number; s?: number; wig?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-56} y={-34} width={112} height={68} rx={16} fill={P.amber} />
    <rect x={-56} y={-34} width={112} height={20} rx={10} fill={P.amberD} opacity={0.6} />
    {[-30, 0, 30].map((dx) => (
      <path key={dx} d={`M ${dx - 8} -20 q 10 8 16 0`} stroke={P.coralD} strokeWidth={5}
        fill="none" strokeLinecap="round" />
    ))}
    {wig > 0 && <g>{/* keep signature stable */}</g>}
  </g>
);

const Cucumber = ({x, y, s = 1, rot = 0}: {x: number; y: number; s?: number; rot?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`}>
    <rect x={-66} y={-16} width={132} height={32} rx={16} fill={P.tealD} />
    <rect x={-66} y={-16} width={132} height={12} rx={6} fill={P.teal} opacity={0.7} />
    <ellipse cx={-66} cy={0} rx={8} ry={16} fill={P.teal} />
  </g>
);

/* ---------- 老鼠 ---------- */
const Mouse = ({x, y, s = 1, col = '#b9c6d6', eat = 0}: {
  x: number; y: number; s?: number; col?: string; eat?: number;
}) => {
  const f = useCurrentFrame();
  const n = Math.sin(f * 0.9) * 3 * eat;
  return (
    <g transform={`translate(${x},${y + n * 0.4}) scale(${s})`}>
      <path d="M -64 10 Q -98 -8 -110 -36" stroke={col} strokeWidth={8} fill="none" strokeLinecap="round" />
      <ellipse rx={54} ry={36} fill={col} />
      <circle cx={48} cy={-16 - n} r={26} fill={col} />
      <circle cx={34} cy={-40 - n} r={13} fill={col} />
      <circle cx={62} cy={-38 - n} r={13} fill={col} />
      <circle cx={34} cy={-40 - n} r={6.4} fill={P.rose} opacity={0.7} />
      <circle cx={62} cy={-38 - n} r={6.4} fill={P.rose} opacity={0.7} />
      <circle cx={56} cy={-20 - n} r={3.6} fill={P.ink} />
      <circle cx={73} cy={-8 - n} r={4.6} fill={P.rose} />
      {[-1, 0, 1].map((i) => (
        <line key={i} x1={70} y1={-8 - n} x2={98} y2={-16 - n + i * 10} stroke={P.cream}
          strokeWidth={2.6} strokeLinecap="round" opacity={0.8} />
      ))}
    </g>
  );
};

/* ---------- 粮堆 ---------- */
const Pellets = ({x, y, n = 3, s = 1}: {x: number; y: number; n?: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    {Array.from({length: n}).map((_, i) => {
      const row = Math.floor(i / 4);
      const dx = (i % 4) * 34 - Math.min(n - row * 4, 4) * 17 + 17;
      return <circle key={i} cx={dx} cy={-row * 26} r={15} fill={i % 2 ? P.amber : P.amberD} />;
    })}
  </g>
);

/* ---------- 雪花 ---------- */
const Snow = ({x, y, s = 1}: {x: number; y: number; s?: number}) => {
  const f = useCurrentFrame();
  return (
    <g transform={`translate(${x},${y + (f * 0.8) % 40 - 20}) scale(${s}) rotate(${f * 0.6})`} opacity={0.85}>
      {[0, 60, 120].map((a) => (
        <line key={a} x1={-16} y1={0} x2={16} y2={0} stroke={P.ice} strokeWidth={4.4}
          strokeLinecap="round" transform={`rotate(${a})`} />
      ))}
    </g>
  );
};

/* ---------- 散热箭头（从身体向外） ---------- */
const HeatFlow = ({x, y, on = 1, dir = 1, col = P.coral, n = 5, spread = 130, len = 46, ph = 0}: {
  x: number; y: number; on?: number; dir?: 1 | -1; col?: string; n?: number; spread?: number; len?: number; ph?: number;
}) => {
  const f = useCurrentFrame();
  if (on <= 0) return null;
  return (
    <g opacity={on}>
      {Array.from({length: n}).map((_, i) => {
        const cyc = (f * 1.6 + i * 24 + ph) % 60;
        const p = cyc / 60;
        const ay = y + (i - (n - 1) / 2) * (spread / n);
        const ax = x + dir * p * len;
        return (
          <g key={i} opacity={0.9 * (1 - p)}>
            <line x1={ax} y1={ay} x2={ax + dir * 22} y2={ay} stroke={col}
              strokeWidth={6} strokeLinecap="round" />
            <polygon points={`${ax + dir * 34},${ay} ${ax + dir * 20},${ay - 8} ${ax + dir * 20},${ay + 8}`}
              fill={col} />
          </g>
        );
      })}
    </g>
  );
};

/* ---------- 曲线工具 ---------- */
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

/* ---------- 面板标题 ---------- */
const PanelTitle = ({x, y, text, col = P.muted}: {x: number; y: number; text: string; col?: string}) => (
  <text x={x} y={y} fill={col} fontSize={25} fontFamily={FONT_HEAD}>{text}</text>
);

/* ============ 标题卡：泳池 + 泳者 + 饭碗 ============ */
const TitleCard = ({data}: {data: any}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 13, mass: 0.7}});
  const o = interpolate(f, [0, 8], [0, 1]);
  const pA = Math.max(0.001, spring({frame: f - 4, fps, config: {damping: 11, stiffness: 140}}));
  const pB = Math.max(0.001, spring({frame: f - 14, fps, config: {damping: 11, stiffness: 140}}));
  const shake = Math.sin(f * 0.9) * 2.2;
  return (
    <AbsoluteFill>
      <Backdrop top={P.bg} bottom={P.bg} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={18} from={0} dur={22} />
        {/* 泳池横幅 */}
        <g transform={`translate(960,470) scale(${pA}) translate(-960,-470)`} opacity={clamp01(pA)}>
          <rect x={320} y={240} width={1280} height={400} rx={30} fill={P.water} />
          <rect x={320} y={500} width={1280} height={140} rx={30} fill={P.skyD} opacity={0.5} />
          {/* 泳道线 */}
          {[560, 610].map((yy, i) => (
            <line key={i} x1={340} y1={yy} x2={1580} y2={yy} stroke={P.cream} strokeWidth={6}
              strokeDasharray="26 22" opacity={0.55} />
          ))}
          <WaveLine x={330} y={252} w={1260} n={11} amp={10} col={P.ice} lw={6} />
          {/* 泳者 */}
          <Swimmer x={760} y={380} s={1.5} />
          <Bubbles x={700} y={440} n={4} />
          {/* 体温计（低） */}
          <g transform="translate(1270,430)">
            <rect x={-96} y={-140} width={192} height={300} rx={24} fill={P.panel}
              stroke={P.card} strokeWidth={4} />
            <Thermo x={0} y={70} h={170} lvl={0.3} col={P.ice} />
            <text x={0} y={-104} textAnchor="middle" fontSize={24} fill={P.ice}
              fontFamily={FONT_HEAD} fontWeight={800}>体温被偷走</text>
          </g>
        </g>
        {/* 饭碗 + 饿 */}
        <g transform={`translate(1470,690) rotate(${shake * 0.6}) scale(${pB})`}
          opacity={clamp01(pB)}>
          <circle cx={0} cy={0} r={118} fill={P.panel} stroke={P.amber} strokeWidth={5} />
          <Bowl x={0} y={10} s={1.1} steam={1} />
          <text x={84} y={-70} fontSize={52} fill={P.coral} fontFamily={FONT_HEAD}
            fontWeight={900} transform="rotate(12 84 -70)">饿!</text>
        </g>
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 520}}>
          <div style={{color: P.teal, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 22}}>
            冷知识 · 第 {data.episode} 期
          </div>
          <div style={{color: C.text, fontSize: 78, fontWeight: 900, fontFamily: FONT_HEAD, letterSpacing: 6}}>
            {data.title}
          </div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ============ S0 消耗差不多 · 饿得离谱 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);
  const popA = Math.max(0.001, spring({frame: f - 8, fps, config: {damping: 12, stiffness: 120}}));
  const popB = Math.max(0.001, spring({frame: f - 22, fps, config: {damping: 12, stiffness: 120}}));
  const calK = clamp01((f - b1 + 8) / 40);
  const hunK = clamp01((f - b2 + 14) / 46);
  const hunOver = hunK > 0.92 ? Math.sin((f - b2) * 0.5) * 5 : 0;
  const breadPop = Math.max(0.001, spring({frame: f - b2 - 40, fps, config: {damping: 10, stiffness: 150}}));

  const Bar = ({x, h, col, label, wob = 0}: {x: number; h: number; col: string; label: string; wob?: number}) => (
    <g transform={`translate(${x + wob},0)`}>
      <rect x={-42} y={810 - h} width={84} height={Math.max(2, h)} rx={14} fill={col} />
      <text x={0} y={848} textAnchor="middle" fontSize={23} fill={P.cream}
        fontFamily={FONT_HEAD} fontWeight={800}>{label}</text>
    </g>
  );

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={0} name={sec.name} accent={P.sky} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 上：两个运动面板 */}
        <g transform={`translate(530,315) scale(${popA}) translate(-530,-315)`} opacity={clamp01(popA)}>
          <rect x={120} y={140} width={820} height={350} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <PanelTitle x={160} y={196} text="跑步 5 公里 · 陆地" />
          {/* 跑道 */}
          <rect x={160} y={390} width={740} height={56} rx={18} fill={P.bgDeep} />
          {[0, 1, 2, 3].map((i) => (
            <line key={i} x1={190 + i * 180} y1={418} x2={250 + i * 180} y2={418}
              stroke={P.card} strokeWidth={8} strokeLinecap="round" />
          ))}
          <Runner x={540} y={356} s={1.35} />
          <Sweat x={472} y={268} s={0.9} />
          <Sweat x={624} y={250} s={0.72} />
          <text x={850} y={300} textAnchor="end" fontSize={24} fill={P.coral}
            fontFamily={FONT_HEAD} fontWeight={800}>体温 ↑ 出汗散热</text>
          <text x={850} y={336} textAnchor="end" fontSize={21} fill={P.muted} fontFamily={FONT_HEAD}>
            出更衣室 · 有点想吃
          </text>
        </g>
        <g transform={`translate(1390,315) scale(${popB}) translate(-1390,-315)`} opacity={clamp01(popB)}>
          <rect x={980} y={140} width={820} height={350} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <PanelTitle x={1020} y={196} text="游泳 1500 米 · 泡水" />
          {/* 泳池 */}
          <rect x={1020} y={300} width={740} height={146} rx={20} fill={P.water} />
          <WaveLine x={1030} y={304} w={720} n={9} amp={7} col={P.ice} lw={5} />
          <Swimmer x={1360} y={370} s={1.3} />
          <Bubbles x={1300} y={420} n={3} />
          <text x={1750} y={252} textAnchor="end" fontSize={24} fill={P.ice}
            fontFamily={FONT_HEAD} fontWeight={800}>体温 ↓ 持续失热</text>
          <text x={1750} y={336} textAnchor="end" fontSize={21} fill={P.coral}
            fontFamily={FONT_HEAD} fontWeight={800}>饿到手抖 · 见面包走不动</text>
        </g>

        {/* 下：账本对比 */}
        <rect x={120} y={520} width={1680} height={360} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <PanelTitle x={160} y={576} text="算一笔账 · 同时间同强度" />
        <line x1={960} y1={560} x2={960} y2={860} stroke={P.card} strokeWidth={3} strokeDasharray="6 12" />
        {/* 左组：热量消耗打平 */}
        <text x={560} y={576} textAnchor="middle" fontSize={24} fill={P.muted} fontFamily={FONT_HEAD}>
          热量消耗
        </text>
        <g opacity={calK}>
          <Bar x={470} h={150 * calK} col={P.amber} label="跑步" />
          <Bar x={620} h={141 * calK} col={P.sky} label="游泳" />
          {calK > 0.85 && (
            <g>
              <rect x={457} y={588} width={176} height={52} rx={15} fill={P.bgDeep} stroke={P.teal} strokeWidth={4} />
              <text x={545} y={624} textAnchor="middle" fontSize={26} fill={P.teal}
                fontFamily={FONT_HEAD} fontWeight={900}>≈ 打平</text>
            </g>
          )}
        </g>
        {/* 右组：饥饿感爆表 */}
        <text x={1330} y={576} textAnchor="middle" fontSize={24} fill={P.muted} fontFamily={FONT_HEAD}>
          出更衣室时的饥饿感
        </text>
        <g opacity={hunK}>
          <Bar x={1310} h={70 * hunK} col={P.amber} label="跑步" />
          <g transform={`translate(${1460 + hunOver},0)`}>
            <rect x={-42} y={810 - 210 * hunK} width={84} height={Math.max(2, 210 * hunK)} rx={14}
              fill={P.coral} />
            <text x={0} y={848} textAnchor="middle" fontSize={23} fill={P.cream}
              fontFamily={FONT_HEAD} fontWeight={800}>游泳</text>
            {hunK > 0.8 && (
              <text x={0} y={810 - 210 * hunK + 44} textAnchor="middle" fontSize={26}
                fill={P.ink} fontFamily={FONT_HEAD} fontWeight={900}>爆表!</text>
            )}
          </g>
          {hunK > 0.6 && (
            <g transform={`translate(1660,700) scale(${breadPop})`} opacity={clamp01(breadPop)}>
              <Bread x={0} y={0} s={0.9} />
              <text x={0} y={88} textAnchor="middle" fontSize={21} fill={P.amber}
                fontFamily={FONT_HEAD} fontWeight={800}>盯上面包</text>
            </g>
          )}
        </g>
        {hunK > 0.9 && (
          <g>
            <text x={1670} y={600} textAnchor="middle" fontSize={23} fill={P.coral}
              fontFamily={FONT_HEAD} fontWeight={800}>差的是体温</text>
            <text x={1670} y={634} textAnchor="middle" fontSize={23} fill={P.coral}
              fontFamily={FONT_HEAD} fontWeight={800}>不是消耗</text>
          </g>
        )}
      </svg>
      <Chip text="热量账 ≠ 饥饿账" x={1450} y={66} at={b1 + 20} accent={P.amber} />
      <Stamp text="饿得不是一个量级" x={520} y={86} at={b2 + 60} color={P.coralD} size={40} rot={-3} />
    </FadeIn>
  );
};

/* ============ S1 水是热量黑洞 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);
  const poolK = clamp01((f - 8) / 26);
  const heatOn = clamp01((f - b1 + 20) / 20) * (1 + clamp01((f - b2 + 10) / 20) * 0.9);
  const tempK = clamp01((f - b1 - 8) / 30);
  const lvl = 0.86 - 0.3 * clamp01((f - b1 - 6) / 160);
  const balK = clamp01((f - b2 - 6) / 30);
  const tilt = 14 * balK;

  return (
    <FadeIn>
      <Backdrop top={P.bgDeep} bottom={P.bgDeep} />
      <SceneTag index={1} name={sec.name} accent={P.ice} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：泳池剖面 */}
        <rect x={120} y={140} width={1060} height={720} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <PanelTitle x={160} y={196} text="泳池剖面 · 泡在 27°C 水里" />
        <g opacity={poolK}>
          {/* 空气层 */}
          <rect x={160} y={230} width={980} height={90} rx={14} fill={P.bgLite} />
          <text x={200} y={282} fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>
            空气 · 还算隔热
          </text>
          {/* 水体 */}
          <rect x={160} y={320} width={980} height={500} rx={14} fill={P.water} />
          <rect x={160} y={660} width={980} height={160} rx={14} fill={P.skyD} opacity={0.45} />
          <WaveLine x={170} y={322} w={960} n={11} amp={8} col={P.ice} lw={5} />
          <text x={200} y={380} fontSize={22} fill={P.ice} fontFamily={FONT_HEAD}>
            水 · 比热容大导热快
          </text>
          {/* 泳者 */}
          <Swimmer x={620} y={520} s={1.7} wig={1} />
          <Bubbles x={540} y={590} n={4} />
          {/* 散热箭头：体表上下向外 */}
          <HeatFlow x={620} y={440} on={heatOn} dir={1} n={4} spread={120} len={70} ph={0} />
          <HeatFlow x={620} y={600} on={heatOn} dir={1} n={4} spread={110} len={70} ph={30} />
          <HeatFlow x={470} y={480} on={heatOn * 0.8} dir={-1} n={3} spread={100} len={50} ph={15} />
          {/* 靠在池边也失热 */}
          {tempK > 0.4 && (
            <g opacity={tempK}>
              <text x={200} y={770} fontSize={23} fill={P.cream} fontFamily={FONT_HEAD} fontWeight={800}>
                哪怕停下靠池边 · 热量照流不误
              </text>
            </g>
          )}
          <text x={1100} y={770} textAnchor="end" fontSize={23} fill={P.coral}
            fontFamily={FONT_HEAD} fontWeight={800}>
            → 热量持续外流
          </text>
        </g>

        {/* 右：体温计 + 导热对比 + 天平 */}
        <rect x={1210} y={140} width={590} height={720} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <PanelTitle x={1246} y={196} text="差近 10°C · 从入水就在漏" />
        {/* 体温计 */}
        <Thermo x={1400} y={470} h={210} lvl={lvl} col={lvl > 0.5 ? P.coral : P.ice} />
        <g opacity={tempK}>
          <rect x={1490} y={300} width={270} height={60} rx={16} fill={P.bgDeep} stroke={P.coral} strokeWidth={4} />
          <text x={1625} y={340} textAnchor="middle" fontSize={25} fill={P.coral}
            fontFamily={FONT_HEAD} fontWeight={800}>体温 37°C</text>
          <rect x={1490} y={380} width={270} height={60} rx={16} fill={P.bgDeep} stroke={P.ice} strokeWidth={4} />
          <text x={1625} y={420} textAnchor="middle" fontSize={25} fill={P.ice}
            fontFamily={FONT_HEAD} fontWeight={800}>水温 27°C</text>
          <text x={1625} y={490} textAnchor="middle" fontSize={23} fill={P.muted} fontFamily={FONT_HEAD}>
            温差 ≈ 10°C 全程无休
          </text>
        </g>
        {/* 导热对比条 */}
        <text x={1246} y={590} fontSize={23} fill={P.muted} fontFamily={FONT_HEAD}>导热能力对比</text>
        <g opacity={poolK}>
          <text x={1246} y={640} fontSize={21} fill={P.muted} fontFamily={FONT_HEAD}>空气</text>
          <rect x={1340} y={620} width={26} height={26} rx={8} fill={P.muted} />
          <text x={1390} y={641} fontSize={21} fill={P.muted} fontFamily={FONT_HEAD}>×1</text>
          <text x={1246} y={694} fontSize={21} fill={P.ice} fontFamily={FONT_HEAD}>水</text>
          <rect x={1340} y={672} width={420} height={26} rx={8} fill={P.sky} />
          <text x={1748} y={694} textAnchor="end" fontSize={22} fill={P.ink}
            fontFamily={FONT_HEAD} fontWeight={900}>×20+</text>
        </g>
        {/* 产热 vs 散热 小天平 */}
        {balK > 0 && (
          <g opacity={balK} transform={`translate(1505,780) rotate(${tilt})`}>
            <line x1={-180} y1={0} x2={180} y2={0} stroke={P.cream} strokeWidth={8} strokeLinecap="round" />
            <rect x={-206} y={-52} width={120} height={44} rx={12} fill={P.amber} />
            <text x={-146} y={-22} textAnchor="middle" fontSize={22} fill={P.ink}
              fontFamily={FONT_HEAD} fontWeight={900}>产热</text>
            <rect x={86} y={-52} width={120} height={44} rx={12} fill={P.coral} />
            <text x={146} y={-22} textAnchor="middle" fontSize={22} fill={P.ink}
              fontFamily={FONT_HEAD} fontWeight={900}>散热</text>
          </g>
        )}
        {balK > 0.5 && (
          <text x={1505} y={838} textAnchor="middle" fontSize={22} fill={P.coral}
            fontFamily={FONT_HEAD} fontWeight={800}>拼命游也追不平 · 散热赢</text>
        )}
      </svg>
      <Chip text="水 = 热量黑洞" x={1470} y={66} at={30} accent={P.ice} />
      <Stamp text="失热跟动不动无关" x={500} y={86} at={b2 + 30} color={P.skyD} size={40} rot={-3} />
    </FadeIn>
  );
};

/* ============ S2 低温打开食欲 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);
  const warmK = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 12, stiffness: 120}}));
  const coldK = Math.max(0.001, spring({frame: f - b1 - 30, fps, config: {damping: 12, stiffness: 120}}));
  const chainK = [0, 1, 2, 3].map((i) =>
    clamp01((f - b1 - 60 - i * 26) / 22));
  const stompK = Math.max(0.001, spring({frame: f - b2 - 10, fps, config: {damping: 9, stiffness: 130}}));

  const ChainBox = ({x, y, w, col, k, top, bottom}: {
    x: number; y: number; w: number; col: string; k: number; top: string; bottom?: string;
  }) => (
    <g transform={`translate(${x},${y}) scale(${Math.max(0.001, k)})`} opacity={clamp01(k * 2)}>
      <rect x={-w / 2} y={-52} width={w} height={104} rx={18} fill={P.bgDeep} stroke={col} strokeWidth={4.4} />
      <text x={0} y={bottom ? -4 : 10} textAnchor="middle" fontSize={27} fill={col}
        fontFamily={FONT_HEAD} fontWeight={900}>{top}</text>
      {bottom && (
        <text x={0} y={34} textAnchor="middle" fontSize={21} fill={P.cream}
          fontFamily={FONT_HEAD}>{bottom}</text>
      )}
    </g>
  );

  const arrow = (x0: number, x1: number, y: number, k: number) => (
    <g opacity={k}>
      <line x1={x0} y1={y} x2={x1 - 18} y2={y} stroke={P.muted} strokeWidth={6} strokeLinecap="round" />
      <polygon points={`${x1},${y} ${x1 - 20},${y - 10} ${x1 - 20},${y + 10}`} fill={P.muted} />
    </g>
  );

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={2} name={sec.name} accent={P.teal} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 上：老鼠对照实验 */}
        <g transform={`translate(520,320) scale(${warmK}) translate(-520,-320)`} opacity={clamp01(warmK)}>
          <rect x={120} y={140} width={800} height={350} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <PanelTitle x={160} y={196} text="暖舱 · 室温" />
          <rect x={160} y={240} width={720} height={210} rx={18} fill={P.bgLite} />
          <Mouse x={460} y={390} s={1.25} eat={0.25} />
          <Pellets x={660} y={420} n={3} s={0.9} />
          <text x={660} y={280} textAnchor="middle" fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>
            食量正常
          </text>
          <text x={160} y={478} fontSize={21} fill={P.muted} fontFamily={FONT_HEAD}>
            动物实验 · 同一批老鼠
          </text>
        </g>
        <g transform={`translate(1400,320) scale(${coldK}) translate(-1400,-320)`} opacity={clamp01(coldK)}>
          <rect x={1000} y={140} width={800} height={350} rx={20} fill={P.panel}
            stroke={coldK > 0.5 ? P.ice : P.card} strokeWidth={4} />
          <PanelTitle x={1040} y={196} text="冷舱 · 冷暴露" col={P.ice} />
          <rect x={1040} y={240} width={720} height={210} rx={18} fill={P.bgDeep} />
          <Snow x={1100} y={280} s={0.9} />
          <Snow x={1700} y={300} s={0.75} />
          <Snow x={1150} y={380} s={0.6} />
          <Mouse x={1320} y={390} s={1.25} eat={1} />
          <Pellets x={1520} y={428} n={8} s={0.9} />
          <text x={1520} y={280} textAnchor="middle" fontSize={22} fill={P.ice}
            fontFamily={FONT_HEAD} fontWeight={800}>进食量立刻上涨</text>
          <text x={1040} y={478} fontSize={21} fill={P.ice} fontFamily={FONT_HEAD}>
            写在基因里的越冬程序 · 天冷了赶紧囤能量
          </text>
        </g>

        {/* 下：激素链条 */}
        <rect x={120} y={520} width={1680} height={360} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <PanelTitle x={160} y={576} text="冷 → 食欲的激素通路" />
        {/* 链条 */}
        {arrow(330, 480, 690, chainK[0])}
        {arrow(800, 950, 690, chainK[1])}
        {arrow(1270, 1420, 690, chainK[2])}
        <ChainBox x={250} y={690} w={180} col={P.ice} k={chainK[0]} top="冷暴露" bottom="体温往下走" />
        <g transform={`translate(640,690) scale(${Math.max(0.001, chainK[1])})`} opacity={clamp01(chainK[1] * 2)}>
          <rect x={-160} y={-84} width={320} height={72} rx={16} fill={P.bgDeep} stroke={P.coral} strokeWidth={4} />
          <text x={0} y={-38} textAnchor="middle" fontSize={26} fill={P.coral}
            fontFamily={FONT_HEAD} fontWeight={900}>胃饥饿素 ↑</text>
          <rect x={-160} y={6} width={320} height={72} rx={16} fill={P.bgDeep} stroke={P.muted}
            strokeWidth={4} opacity={0.85} />
          <text x={0} y={52} textAnchor="middle" fontSize={26} fill={P.muted}
            fontFamily={FONT_HEAD} fontWeight={900}>饱腹信号 ↓</text>
        </g>
        <ChainBox x={1110} y={690} w={200} col={P.vio} k={chainK[2]} top="下丘脑" bottom="算总账" />
        {/* 指令牌压在理性上 */}
        <g transform={`translate(1600,690) scale(${Math.max(0.001, chainK[3])})`} opacity={clamp01(chainK[3] * 2)}>
          <rect x={-160} y={-52} width={320} height={104} rx={18} fill={P.coral} />
          <text x={0} y={-6} textAnchor="middle" fontSize={30} fill={P.ink}
            fontFamily={FONT_HEAD} fontWeight={900}>必须进料!</text>
          <text x={0} y={32} textAnchor="middle" fontSize={20} fill={P.ink}
            fontFamily={FONT_HEAD} fontWeight={800}>优先级拉满</text>
        </g>
        {/* 游泳双开关 + 压过理性 */}
        {stompK > 0.02 && (
          <g opacity={clamp01(stompK)} transform={`translate(0,${(1 - stompK) * 36})`}>
            <rect x={540} y={796} width={880} height={64} rx={17} fill={P.bgDeep} stroke={P.teal} strokeWidth={4} />
            <text x={980} y={838} textAnchor="middle" fontSize={25} fill={P.teal}
              fontFamily={FONT_HEAD} fontWeight={900}>
              游泳双开关一起踩：消耗 + 产热成本 · 理性管不了温度计
            </text>
            <g transform="translate(1600,560) rotate(-8)">
              <rect x={-120} y={-26} width={240} height={52} rx={14} fill={P.bgDeep}
                stroke={P.muted} strokeWidth={3} opacity={0.9} />
              <text x={0} y={9} textAnchor="middle" fontSize={21} fill={P.muted}
                fontFamily={FONT_HEAD}>理性：没消耗多少</text>
            </g>
            <line x1={1500} y1={566} x2={1700} y2={560} stroke={P.coral} strokeWidth={7}
              strokeLinecap="round" />
            <line x1={1510} y1={586} x2={1690} y2={586} stroke={P.coral} strokeWidth={7}
              strokeLinecap="round" opacity={0.6} />
          </g>
        )}
      </svg>
      <Chip text="冷 = 最强催饿开关之一" x={1330} y={66} at={b1 + 20} accent={P.teal} />
      <Stamp text="越冬程序启动" x={560} y={86} at={b2 + 40} color={P.tealD} size={42} rot={-3} />
    </FadeIn>
  );
};

/* ============ S3 跑步有刹车 ============ */
const RUN_TEMP: [number, number][] = [[180, 470], [320, 430], [480, 360], [640, 310], [800, 300]];
const RUN_APP: [number, number][] = [[180, 370], [330, 410], [500, 445], [660, 452], [800, 450]];
const SWIM_TEMP: [number, number][] = [[1040, 320], [1180, 372], [1330, 425], [1490, 458], [1640, 466]];
const SWIM_APP: [number, number][] = [[1040, 460], [1170, 430], [1310, 372], [1470, 316], [1640, 292]];

const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);
  const runK = clamp01((f - 12) / 70);
  const runAppK = clamp01((f - 50) / 70);
  const swimK = clamp01((f - b1 - 4) / 70);
  const swimAppK = clamp01((f - b1 - 30) / 70);
  const pillK = clamp01((f - 90) / 24);
  const expK = clamp01((f - b2 - 4) / 30);

  const Chart = ({x0, title, col}: {x0: number; title: string; col: string}) => (
    <g>
      <rect x={x0} y={140} width={800} height={430} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
      <PanelTitle x={x0 + 40} y={196} text={title} col={col} />
      <line x1={x0 + 60} y1={250} x2={x0 + 60} y2={500} stroke={P.muted} strokeWidth={3.4} />
      <line x1={x0 + 60} y1={500} x2={x0 + 740} y2={500} stroke={P.muted} strokeWidth={3.4} />
      <text x={x0 + 36} y={280} textAnchor="end" fontSize={19} fill={P.muted} fontFamily={FONT_HEAD}>高</text>
      <text x={x0 + 36} y={490} textAnchor="end" fontSize={19} fill={P.muted} fontFamily={FONT_HEAD}>低</text>
      <text x={x0 + 700} y={536} textAnchor="end" fontSize={19} fill={P.muted}
        fontFamily={FONT_HEAD}>运动后时间 →</text>
    </g>
  );

  return (
    <FadeIn>
      <Backdrop top={P.bgDeep} bottom={P.bgDeep} />
      <SceneTag index={3} name={sec.name} accent={P.coral} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：跑步 */}
        <Chart x0={120} title="跑步 · 升温自带刹车" col={P.amber} />
        <Curve pts={RUN_TEMP} prog={runK} col={P.coral} lw={6.5} />
        <Curve pts={RUN_APP} prog={runAppK} col={P.teal} lw={6.5} dash="2 0" />
        {runAppK > 0.7 && (
          <g>
            <text x={806} y={296} textAnchor="end" fontSize={20} fill={P.coral}
              fontFamily={FONT_HEAD} fontWeight={800}>体温↑</text>
            <text x={806} y={482} textAnchor="end" fontSize={20} fill={P.teal}
              fontFamily={FONT_HEAD} fontWeight={800}>食欲被压住↓</text>
          </g>
        )}
        {pillK > 0 && (
          <g opacity={pillK}>
            <rect x={300} y={216} width={180} height={46} rx={23} fill={P.bgDeep} stroke={P.teal} strokeWidth={3.6} />
            <text x={390} y={248} textAnchor="middle" fontSize={22} fill={P.teal}
              fontFamily={FONT_HEAD} fontWeight={900}>肽YY · GLP1</text>
            <text x={190} y={534} fontSize={21} fill={P.teal} fontFamily={FONT_HEAD} fontWeight={800}>
              跑完一两小时 · 不饿黄金窗
            </text>
          </g>
        )}
        {/* 右：游泳 */}
        <Chart x0={980} title="游泳 · 刹车缺席" col={P.ice} />
        <Curve pts={SWIM_TEMP} prog={swimK} col={P.ice} lw={6.5} />
        <Curve pts={SWIM_APP} prog={swimAppK} col={P.coral} lw={6.5} />
        {swimAppK > 0.7 && (
          <g>
            <text x={1660} y={498} textAnchor="end" fontSize={20} fill={P.ice}
              fontFamily={FONT_HEAD} fontWeight={800}>体温上不去↓</text>
            <text x={1660} y={272} textAnchor="end" fontSize={20} fill={P.coral}
              fontFamily={FONT_HEAD} fontWeight={800}>饥饿排队进场↑</text>
          </g>
        )}
        {swimK > 0.6 && (
          <g opacity={swimK}>
            <rect x={1170} y={216} width={240} height={46} rx={23} fill={P.bgDeep}
              stroke={P.muted} strokeWidth={3.6} strokeDasharray="6 7" />
            <text x={1290} y={248} textAnchor="middle" fontSize={22} fill={P.muted}
              fontFamily={FONT_HEAD}>抑制激素 · 没来齐</text>
          </g>
        )}
        {/* 底部实验条 */}
        <rect x={120} y={600} width={1680} height={280} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <PanelTitle x={160} y={656} text="对照实验 · 同样运动时长 · 不同水温" />
        {expK > 0 && (
          <g opacity={expK} transform={`translate(0,${(1 - expK) * 30})`}>
            {/* 冷水组 */}
            <g transform="translate(560,720)">
              <rect x={-190} y={-70} width={380} height={130} rx={18} fill={P.bgDeep}
                stroke={P.ice} strokeWidth={4} />
              <Bowl x={-90} y={6} s={0.8} steam={0.6} />
              <Bowl x={10} y={10} s={0.62} steam={0.5} />
              <text x={110} y={-30} fontSize={25} fill={P.ice} fontFamily={FONT_HEAD} fontWeight={900}>
                冷水组
              </text>
              <text x={110} y={6} fontSize={22} fill={P.coral} fontFamily={FONT_HEAD} fontWeight={800}>
                吃得明显更多
              </text>
            </g>
            {/* 温水组 */}
            <g transform="translate(1030,720)">
              <rect x={-170} y={-70} width={340} height={130} rx={18} fill={P.bgDeep}
                stroke={P.amber} strokeWidth={4} />
              <Bowl x={-60} y={8} s={0.62} steam={0.8} />
              <text x={70} y={-30} fontSize={25} fill={P.amber} fontFamily={FONT_HEAD} fontWeight={900}>
                温水组
              </text>
              <text x={70} y={6} fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>
                食量平平
              </text>
            </g>
            <text x={1560} y={690} textAnchor="middle" fontSize={24} fill={P.cream}
              fontFamily={FONT_HEAD} fontWeight={800}>光泡冷水不运动也催饿</text>
            <text x={1560} y={730} textAnchor="middle" fontSize={22} fill={P.coral}
              fontFamily={FONT_HEAD} fontWeight={800}>温度才是那把钥匙</text>
            <text x={960} y={850} textAnchor="middle" fontSize={23} fill={P.muted}
              fontFamily={FONT_HEAD}>
              消耗差不多 · 出来的食欲像两个世界
            </text>
          </g>
        )}
      </svg>
      <Chip text="升温本身压食欲" x={1430} y={66} at={40} accent={P.coral} />
      <Stamp text="冷暴露独自就能催饿" x={520} y={86} at={b2 + 30} color={P.skyD} size={38} rot={-3} />
    </FadeIn>
  );
};

/* ============ S4 怎么破这波饿 ============ */
const S4_APP: [number, number][] = [
  [220, 700], [360, 660], [560, 590], [780, 580], [980, 640], [1160, 710], [1360, 766], [1520, 782], [1660, 786],
];

const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);
  const junkPops = [0, 1, 2].map((i) =>
    Math.max(0.001, spring({frame: f - 12 - i * 14, fps, config: {damping: 12, stiffness: 130}})));
  const arrowK = clamp01((f - 70) / 24);
  const fixPops = [0, 1, 2].map((i) =>
    Math.max(0.001, spring({frame: f - b1 - 6 - i * 18, fps, config: {damping: 12, stiffness: 130}})));
  const curveK = clamp01((f - b1 - 60) / 120);
  const tagK = clamp01((f - b2 - 8) / 26);

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={4} name={sec.name} accent={P.amber} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 上：激素点的单 vs 该备的货 */}
        <rect x={120} y={140} width={1680} height={330} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <PanelTitle x={160} y={196} text="冷暴露把偏好拨向高热量 · 不是嘴馋" />
        {/* 左：高热量组 */}
        {[
          {x: 330, pop: junkPops[0], label: '面条', ico: <Bowl x={0} y={0} s={0.8} steam={1} />},
          {x: 560, pop: junkPops[1], label: '汉堡', ico: <Burger x={0} y={-4} s={0.9} />},
          {x: 790, pop: junkPops[2], label: '烤串', ico: <Skewer x={0} y={0} s={0.85} rot={-8} />},
        ].map((c, i) => (
          <g key={i} transform={`translate(${c.x},300) scale(${c.pop})`} opacity={clamp01(c.pop * 2)}>
            <circle r={66} fill={P.bgDeep} stroke={P.amber} strokeWidth={4} />
            {c.ico}
            <text x={0} y={112} textAnchor="middle" fontSize={23} fill={P.amber}
              fontFamily={FONT_HEAD} fontWeight={800}>{c.label}</text>
          </g>
        ))}
        <text x={560} y={446} textAnchor="middle" fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>
          激素替你点的单 · 碳水脂肪
        </text>
        {/* 中间箭头 */}
        {arrowK > 0 && (
          <g opacity={arrowK}>
            <line x1={900} y1={300} x2={900 + 130 * arrowK} y2={300} stroke={P.teal}
              strokeWidth={8} strokeLinecap="round" />
            {arrowK > 0.9 && <polygon points="1046,300 1022,288 1022,312" fill={P.teal} />}
            <text x={965} y={272} textAnchor="middle" fontSize={22} fill={P.teal}
              fontFamily={FONT_HEAD} fontWeight={800}>别硬扛 · 提前备</text>
          </g>
        )}
        {/* 右：健康组 */}
        {[
          {x: 1140, pop: fixPops[0], label: '先喝杯温水', ico: <Cup x={0} y={2} s={0.82} />},
          {x: 1400, pop: fixPops[1], label: '吃点蛋白质', ico: <Egg x={0} y={0} s={0.85} />},
          {x: 1660, pop: fixPops[2], label: '粗粮顶一顶', ico: <Bread x={0} y={0} s={0.8} />},
        ].map((c, i) => (
          <g key={i} transform={`translate(${c.x},300) scale(${c.pop})`} opacity={clamp01(c.pop * 2)}>
            <circle r={66} fill={P.bgDeep} stroke={P.teal} strokeWidth={4} />
            {c.ico}
            <text x={0} y={112} textAnchor="middle" fontSize={23} fill={P.teal}
              fontFamily={FONT_HEAD} fontWeight={800}>{c.label}</text>
          </g>
        ))}
        <text x={1400} y={446} textAnchor="middle" fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>
          温水淋浴回温 · 给身体一个「危险解除」信号
        </text>

        {/* 下：饥饿曲线回落 */}
        <rect x={120} y={500} width={1680} height={380} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <PanelTitle x={160} y={556} text="饥饿感 ≠ 真实缺口 · 扛过最饿半小时" />
        {/* 坐标 */}
        <line x1={200} y1={800} x2={1720} y2={800} stroke={P.muted} strokeWidth={3.6} />
        <line x1={200} y1={590} x2={200} y2={800} stroke={P.muted} strokeWidth={3.6} />
        <text x={168} y={620} textAnchor="end" fontSize={20} fill={P.muted} fontFamily={FONT_HEAD}>饿</text>
        {['0', '15分', '30分', '45分'].map((t, i) => (
          <g key={t}>
            <line x1={220 + i * 480} y1={800} x2={220 + i * 480} y2={812} stroke={P.muted} strokeWidth={3} />
            <text x={220 + i * 480} y={838} textAnchor="middle" fontSize={20} fill={P.muted}
              fontFamily={FONT_HEAD}>{t}</text>
          </g>
        ))}
        {/* 最饿半小时阴影 */}
        <rect x={220} y={600} width={960} height={200} fill={P.coral} opacity={curveK > 0.3 ? 0.1 : 0} />
        {/* 饥饿曲线：先高后落 */}
        <Curve pts={S4_APP} prog={curveK} col={P.coral} lw={7} />
        {curveK > 0.6 && (
          <g>
            <text x={860} y={560} textAnchor="middle" fontSize={24} fill={P.coral}
              fontFamily={FONT_HEAD} fontWeight={900}>激素催出来的峰值</text>
            <line x1={760} y1={576} x2={640} y2={596} stroke={P.coral} strokeWidth={3}
              strokeDasharray="4 7" />
            <text x={700} y={790} textAnchor="middle" fontSize={20} fill={P.muted}
              fontFamily={FONT_HEAD}>← 最饿的半小时 →</text>
            <text x={1450} y={760} textAnchor="middle" fontSize={24} fill={P.teal}
              fontFamily={FONT_HEAD} fontWeight={800}>饿意退一大半</text>
          </g>
        )}
        {tagK > 0 && (
          <g opacity={tagK} transform={`translate(0,${(1 - tagK) * 26})`}>
            <rect x={1050} y={560} width={700} height={64} rx={17} fill={P.bgDeep} stroke={P.amber} strokeWidth={4} />
            <text x={1400} y={602} textAnchor="middle" fontSize={27} fill={P.amber}
              fontFamily={FONT_HEAD} fontWeight={900}>
              游泳是好运动 · 只是别让食欲替你记账
            </text>
          </g>
        )}
      </svg>
      <Chip text="最饿的 30 分钟" x={1440} y={66} at={b1 + 30} accent={P.amber} />
      <Stamp text="账本握在自己手里" x={540} y={86} at={b2 + 46} color={P.tealD} size={40} rot={-3} />
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
        <Stars n={20} from={0} dur={20} />
        <g transform={`translate(960,610) scale(${pop}) translate(-960,-610)`} opacity={clamp01(pop)}>
          {/* 泳池边坐着喝温水的小人 */}
          <rect x={420} y={620} width={1080} height={26} rx={13} fill={P.card} />
          <rect x={420} y={646} width={1080} height={140} rx={20} fill={P.water} />
          <WaveLine x={440} y={652} w={1040} n={10} amp={7} col={P.ice} lw={5} />
          {/* 坐着的小人 */}
          <g transform="translate(830,590)">
            <circle cx={0} cy={-86} r={26} fill={C.skin} />
            <path d="M -26 -92 A 26 26 0 0 1 26 -92 L 24 -78 Q 0 -90 -22 -76 Z" fill={P.coral} />
            <circle cx={8} cy={-88} r={3.6} fill={P.ink} />
            <line x1={0} y1={-60} x2={0} y2={-6} stroke={C.skin} strokeWidth={16} strokeLinecap="round" />
            <line x1={4} y1={-46} x2={52} y2={-30} stroke={C.skin} strokeWidth={10} strokeLinecap="round" />
            <line x1={-2} y1={-6} x2={-30} y2={38} stroke={C.skin} strokeWidth={11} strokeLinecap="round" />
            <line x1={2} y1={-6} x2={34} y2={38} stroke={C.skin} strokeWidth={11} strokeLinecap="round" />
            <Cup x={76} y={-24} s={0.66} col={P.teal} />
          </g>
          <Egg x={1060} y={606} s={0.72} />
          <text x={1060} y={676} textAnchor="middle" fontSize={21} fill={P.cream}
            fontFamily={FONT_HEAD} fontWeight={800}>备好蛋白质</text>
          <text x={960} y={840} textAnchor="middle" fontSize={23} fill={P.ice}
            fontFamily={FONT_HEAD}>先回温 · 再决定吃不吃</text>
        </g>
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -330}}>
        <div style={{fontSize: 54, fontWeight: 800, color: C.text}}>水偷体温 · 激素催饭</div>
        <div style={{fontSize: 36, color: P.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E243 = ({data = tl}: {data?: any}) => {
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
