import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01} from '../lib/style';
import {Backdrop, Ripple, Stamp, SceneTag, Chip, Subtitles, ProgressBar, FadeIn, Stars} from '../lib/atoms';
import tl from '../timeline_227.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：午后焦糖 · 暖阳 × 薄荷回血 × 珊瑚警告（纯色平涂） ============ */
const P = {
  bg: '#3d2b1a',        // 午后深棕
  bgDeep: '#2a1d11',
  bgLite: '#4b3723',
  card: '#332519',
  cardLite: '#473322',
  ink: '#221709',       // 深咖字（压亮色块上）
  cream: '#f8ecd4',
  muted: '#c9b18c',     // 暖沙弱字
  sun: '#ffc94d',
  sunD: '#e89b2e',
  coral: '#ff6b6b',
  coralD: '#d94f4f',
  mint: '#7be0c3',
  mintD: '#45b491',
  sky: '#6fd3ff',
  skin: '#ffcf9e',
  skinD: '#e8a76f',
  wood: '#8a5a3c',
  woodD: '#63402a',
  sheet: '#f2e3cb',
  pillow: '#fff6e2',
  blanket: '#e07840',
  blanketD: '#b85c2f',
  stepTop: '#5a4230',
  stepFace: '#26190f',
  n1: '#ffd97b',        // N1 浅黄
  n2: '#f7a54f',        // N2 橙
  n3: '#8a6bb5',        // N3 深紫
  wave: '#ffe08a',      // 脑电
  coffee: '#8a5127',
  coffeeL: '#c98d52',
  mug: '#eef2f7',
};

/* ---------- 极坐标 / 圆弧 ---------- */
const pol = (cx: number, cy: number, r: number, deg: number): [number, number] =>
  [cx + r * Math.cos(deg * Math.PI / 180), cy + r * Math.sin(deg * Math.PI / 180)];

const arcPath = (cx: number, cy: number, r: number, a0: number, a1: number) => {
  const [x0, y0] = pol(cx, cy, r, a0);
  const [x1, y1] = pol(cx, cy, r, a1);
  const large = a1 - a0 > 180 ? 1 : 0;
  return `M ${x0.toFixed(1)} ${y0.toFixed(1)} A ${r} ${r} 0 ${large} 1 ${x1.toFixed(1)} ${y1.toFixed(1)}`;
};

const minDeg = (m: number) => m * 6 - 90;

/* ---------- 折线波 ---------- */
const wavePath = (x0: number, x1: number, y: number, amp: number, freq: number, ph = 0) => {
  const n = 30;
  let d = '';
  for (let i = 0; i <= n; i++) {
    const px = x0 + (x1 - x0) * (i / n);
    const py = y + Math.sin(i * freq + ph) * amp;
    d += `${i ? 'L' : 'M'} ${px.toFixed(1)} ${py.toFixed(1)} `;
  }
  return d;
};

/* ---------- 午睡闹钟：min 分钟指针，arc0→min 高亮弧，ring 响铃摇摆 ---------- */
const NapClock = ({x, y, r = 110, min = 20, arc0 = 0, arcCol = P.mint, ring = 0}: {
  x: number; y: number; r?: number; min?: number; arc0?: number; arcCol?: string; ring?: number;
}) => {
  const f = useCurrentFrame();
  const wob = Math.sin(f * 0.6) * 6 * ring;
  const a = minDeg(min);
  const [hx, hy] = pol(0, 0, r * 0.58, a);
  return (
    <g transform={`translate(${x},${y}) rotate(${wob})`}>
      <ellipse cx={-r * 0.6} cy={-r * 0.9} rx={r * 0.3} ry={r * 0.18} fill={P.coral}
        transform={`rotate(-28 ${-r * 0.6} ${-r * 0.9})`} />
      <ellipse cx={r * 0.6} cy={-r * 0.9} rx={r * 0.3} ry={r * 0.18} fill={P.coral}
        transform={`rotate(28 ${r * 0.6} ${-r * 0.9})`} />
      <line x1={-r * 0.5} y1={r * 0.8} x2={-r * 0.72} y2={r * 1.06} stroke={P.woodD}
        strokeWidth={r * 0.09} strokeLinecap="round" />
      <line x1={r * 0.5} y1={r * 0.8} x2={r * 0.72} y2={r * 1.06} stroke={P.woodD}
        strokeWidth={r * 0.09} strokeLinecap="round" />
      <circle r={r} fill={P.pillow} stroke={P.woodD} strokeWidth={r * 0.09} />
      {Array.from({length: 12}).map((_, i) => {
        const [tx1, ty1] = pol(0, 0, r * 0.84, i * 30);
        const [tx2, ty2] = pol(0, 0, r * 0.72, i * 30);
        return <line key={i} x1={tx2} y1={ty2} x2={tx1} y2={ty1} stroke={P.ink}
          strokeWidth={r * 0.04} opacity={0.65} />;
      })}
      {min > arc0 && (
        <path d={arcPath(0, 0, r * 0.86, minDeg(arc0), a)} stroke={arcCol}
          strokeWidth={r * 0.14} fill="none" strokeLinecap="round" opacity={0.92} />
      )}
      <line x1={0} y1={0} x2={hx} y2={hy} stroke={P.ink} strokeWidth={r * 0.075} strokeLinecap="round" />
      <line x1={0} y1={0} x2={0} y2={-r * 0.36} stroke={P.ink} strokeWidth={r * 0.06} strokeLinecap="round" />
      <circle r={r * 0.09} fill={P.coral} />
      {/* 响铃声波 */}
      {ring > 0 && (
        <g opacity={ring}>
          {[-1, 1].map((s) => (
            <g key={s} transform={`scale(${s},1)`}>
              {[0, 1].map((i) => {
                const rr = r * (1.16 + i * 0.24 + Math.sin(f * 0.4) * 0.04);
                return (
                  <path key={i} d={arcPath(0, 0, rr, -72, -18)} stroke={P.coral}
                    strokeWidth={6} fill="none" strokeLinecap="round" opacity={0.8 - i * 0.35} />
                );
              })}
            </g>
          ))}
        </g>
      )}
    </g>
  );
};

/* ---------- 瞌睡小人（坐姿） ---------- */
const NapPerson = ({x, y, s = 1, hoodie = P.sky}: {x: number; y: number; s?: number; hoodie?: string}) => {
  const f = useCurrentFrame();
  const nod = Math.sin(f * 0.12) * 5;
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      <rect x={-30} y={38} width={96} height={22} rx={11} fill={hoodie} opacity={0.8} />
      <rect x={-26} y={-8} width={54} height={62} rx={22} fill={hoodie} />
      <g transform={`translate(-2,-40) rotate(${nod})`}>
        <circle r={30} fill={P.skin} />
        <path d="M -30 -12 A 30 30 0 0 1 26 -22 L 26 -30 Q -2 -44 -30 -30 Z" fill={P.ink} />
        <path d="M -14 -4 Q -9 1 -4 -4 M 8 -4 Q 13 1 18 -4" stroke={P.ink} strokeWidth={3.5}
          fill="none" strokeLinecap="round" />
      </g>
    </g>
  );
};

/* ---------- Zzz 飘字 ---------- */
const Zzz = ({x, y, k = 1, col = P.sky}: {x: number; y: number; k?: number; col?: string}) => {
  const f = useCurrentFrame();
  return (
    <g transform={`translate(${x},${y})`}>
      {[0, 1, 2].map((i) => {
        const up = (f * 1.5 + i * 22) % 66;
        return (
          <text key={i} x={i * 46} y={-up * 0.7} fill={col} fontSize={52 - i * 8}
            fontFamily={FONT_HEAD} fontWeight={900}
            opacity={k * clamp01(1 - up / 66) * 0.95}
            transform={`rotate(${-10 + i * 8} ${i * 46} ${-up * 0.7})`}>
            Z
          </text>
        );
      })}
    </g>
  );
};

/* ---------- 枕头 ---------- */
const Pillow = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-160} y={-52} width={320} height={104} rx={46} fill={P.pillow} />
    <rect x={-136} y={-34} width={272} height={68} rx={30} fill={P.sheet} />
    <path d="M -160 -30 q -18 30 0 60 M 160 -30 q 18 30 0 60" stroke={P.pillow}
      strokeWidth={10} fill="none" strokeLinecap="round" />
  </g>
);

/* ---------- 床 ---------- */
const NapBed = ({x, y, s = 1, w = 640}: {x: number; y: number; s?: number; w?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-60} y={-130} width={24} height={140} rx={9} fill={P.woodD} />
    <rect x={-60} y={-140} width={24} height={24} rx={9} fill={P.wood} />
    <rect x={-60} y={-6} width={w} height={30} rx={13} fill={P.wood} />
    <rect x={-36} y={24} width={22} height={46} rx={7} fill={P.woodD} />
    <rect x={w - 100} y={24} width={22} height={46} rx={7} fill={P.woodD} />
    <rect x={-52} y={-40} width={w - 16} height={40} rx={16} fill={P.sheet} />
    <ellipse cx={0} cy={-36} rx={64} ry={26} fill={P.pillow} />
  </g>
);

/* ---------- 迷糊雾云 ---------- */
const FogCloud = ({x, y, s = 1, o = 1}: {x: number; y: number; s?: number; o?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={o}>
    <ellipse rx={78} ry={42} fill={P.muted} opacity={0.8} />
    <ellipse cx={-50} cy={-16} rx={46} ry={32} fill={P.muted} opacity={0.8} />
    <ellipse cx={52} cy={-12} rx={50} ry={34} fill={P.muted} opacity={0.8} />
  </g>
);

/* ---------- 电池/回血条 ---------- */
const Battery = ({x, y, w = 120, h = 54, fill = 0.5, col = P.mint}: {
  x: number; y: number; w?: number; h?: number; fill?: number; col?: string;
}) => (
  <g transform={`translate(${x},${y})`}>
    <rect x={-w / 2} y={-h / 2} width={w} height={h} rx={12} fill="none" stroke={col} strokeWidth={6} />
    <rect x={w / 2 + 5} y={-h * 0.18} width={12} height={h * 0.36} rx={5} fill={col} />
    <rect x={-w / 2 + 8} y={-h / 2 + 8} width={(w - 16) * clamp01(fill)} height={h - 16} rx={7} fill={col} />
  </g>
);

/* ---------- 半圆仪表盘：k 0..1 指针 ---------- */
const Gauge = ({x, y, r = 130, k = 0}: {x: number; y: number; r?: number; k?: number}) => {
  const a = -180 + clamp01(k) * 180;
  const [nx, ny] = pol(0, 0, r * 0.74, a);
  return (
    <g transform={`translate(${x},${y})`}>
      <path d={arcPath(0, 0, r, -180, 0)} stroke={P.cardLite} strokeWidth={20} fill="none" strokeLinecap="round" />
      <path d={arcPath(0, 0, r, -180, -90)} stroke={P.mint} strokeWidth={20} fill="none" strokeLinecap="round" opacity={0.9} />
      <path d={arcPath(0, 0, r, -140, -40)} stroke={P.sun} strokeWidth={20} fill="none" strokeLinecap="round" opacity={0.55} />
      <line x1={0} y1={0} x2={nx} y2={ny} stroke={P.sun} strokeWidth={9} strokeLinecap="round" />
      <circle r={13} fill={P.sun} />
    </g>
  );
};

/* ---------- 咖啡杯 ---------- */
const CoffeeCup = ({x, y, s = 1, k = 1}: {x: number; y: number; s?: number; k?: number}) => {
  const f = useCurrentFrame();
  const kk = Math.max(0.001, clamp01(k));
  return (
    <g transform={`translate(${x},${y}) scale(${s * kk})`} opacity={kk > 0.03 ? 1 : 0}>
      {[0, 1].map((i) => (
        <path key={i} d={`M ${-16 + i * 32} -64 q ${Math.sin(f * 0.12 + i * 2) * 9} -20 0 -38`}
          stroke={P.muted} strokeWidth={6} fill="none" strokeLinecap="round" opacity={0.7} />
      ))}
      <path d="M -52 -32 L -44 44 Q -42 58 -26 58 L 26 58 Q 42 58 44 44 L 52 -32 Z" fill={P.mug} />
      <path d="M -46 -24 L -40 40 L 40 40 L 46 -24 Z" fill={P.coffee} opacity={0.92} />
      <path d="M 52 -18 Q 84 -12 78 16 Q 72 40 46 36" stroke={P.mug} strokeWidth={11} fill="none" />
      <ellipse cx={0} cy={-32} rx={50} ry={10} fill={P.coffeeL} />
    </g>
  );
};

/* ---------- 头像图标（人物卡用） ---------- */
const HeadIcon = ({x, y, s = 1, hair = P.ink, mood = 'sleep'}: {
  x: number; y: number; s?: number; hair?: string; mood?: 'sleep' | 'groggy' | 'wired';
}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <circle r={44} fill={P.skin} />
    <path d="M -44 -16 A 44 44 0 0 1 40 -30 L 40 -40 Q -4 -62 -44 -40 Z" fill={hair} />
    {mood === 'sleep' && (
      <path d="M -22 -4 Q -16 2 -10 -4 M 10 -4 Q 16 2 22 -4" stroke={P.ink} strokeWidth={4.5}
        fill="none" strokeLinecap="round" />
    )}
    {mood === 'groggy' && (
      <g>
        <path d="M -24 -6 Q -16 0 -8 -6 M 8 -6 Q 16 0 24 -6" stroke={P.ink} strokeWidth={4.5}
          fill="none" strokeLinecap="round" />
        <path d="M -22 8 q 6 5 12 0 M 10 8 q 6 5 12 0" stroke={P.ink} strokeWidth={3}
          fill="none" strokeLinecap="round" opacity={0.55} />
        <path d="M -10 24 Q 0 18 10 24" stroke={P.ink} strokeWidth={4} fill="none" strokeLinecap="round" />
      </g>
    )}
    {mood === 'wired' && (
      <g>
        <ellipse cx={-16} cy={-2} rx={9} ry={11} fill="#fff" />
        <ellipse cx={16} cy={-2} rx={9} ry={11} fill="#fff" />
        <circle cx={-15} cy={-1} r={4} fill={P.ink} />
        <circle cx={17} cy={-1} r={4} fill={P.ink} />
        <path d="M -28 -18 Q -18 -25 -8 -19 M 8 -19 Q 18 -25 28 -18" stroke={P.ink}
          strokeWidth={4} fill="none" strokeLinecap="round" />
        <path d="M -12 22 L 12 22" stroke={P.ink} strokeWidth={4.5} strokeLinecap="round" />
      </g>
    )}
  </g>
);

/* ---------- 饭碗 ---------- */
const RiceBowl = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <path d="M -58 0 Q -56 44 -16 48 L 16 48 Q 56 44 58 0 Z" fill={P.mug} />
    <path d="M -60 0 L 60 0" stroke={P.wood} strokeWidth={8} strokeLinecap="round" />
    <path d="M -46 -6 Q -30 -30 0 -32 Q 34 -30 46 -6 Q 20 -14 0 -14 Q -22 -14 -46 -6 Z" fill={P.pillow} />
    <line x1={26} y1={-40} x2={62} y2={-66} stroke={P.wood} strokeWidth={7} strokeLinecap="round" />
    <line x1={36} y1={-34} x2={72} y2={-56} stroke={P.wood} strokeWidth={7} strokeLinecap="round" />
  </g>
);

/* ---------- 吊床 ---------- */
const Hammock = ({x, y, s = 1}: {x: number; y: number; s?: number}) => {
  const f = useCurrentFrame();
  const sway = Math.sin(f * 0.07) * 4;
  return (
    <g transform={`translate(${x},${y}) scale(${s}) rotate(${sway})`}>
      <line x1={-150} y1={-70} x2={-150} y2={40} stroke={P.wood} strokeWidth={10} strokeLinecap="round" />
      <line x1={150} y1={-70} x2={150} y2={40} stroke={P.wood} strokeWidth={10} strokeLinecap="round" />
      <path d="M -150 -66 Q 0 70 150 -66 L 150 -52 Q 0 84 -150 -52 Z" fill={P.coral} />
      <path d="M -150 -66 Q 0 70 150 -66" stroke={P.coralD} strokeWidth={5} fill="none" />
      {/* 躺着的人 */}
      <circle cx={-52} cy={-16} r={22} fill={P.skin} />
      <path d="M -74 -22 A 22 22 0 0 1 -34 -30 L -34 -38 Q -56 -46 -74 -36 Z" fill={P.ink} />
      <path d="M -30 -8 Q 30 14 92 -2" stroke={P.sky} strokeWidth={26} fill="none" strokeLinecap="round" />
      <path d="M -60 -12 Q -56 -8 -52 -12" stroke={P.ink} strokeWidth={3} fill="none" strokeLinecap="round" />
    </g>
  );
};

/* ---------- 对勾 / 叉 ---------- */
const CheckDot = ({x, y, r = 20, k = 1, col = P.mint, ok = true}: {
  x: number; y: number; r?: number; k?: number; col?: string; ok?: boolean;
}) => {
  const kk = Math.max(0.001, clamp01(k));
  return (
    <g transform={`translate(${x},${y}) scale(${kk})`} opacity={kk > 0.02 ? 1 : 0}>
      <circle r={r} fill={col} />
      {ok ? (
        <path d={`M ${-r * 0.45} 0 L ${-r * 0.1} ${r * 0.4} L ${r * 0.5} ${-r * 0.38}`}
          stroke={P.ink} strokeWidth={5.5} fill="none" strokeLinecap="round" strokeLinejoin="round" />
      ) : (
        <path d={`M ${-r * 0.38} ${-r * 0.38} L ${r * 0.38} ${r * 0.38} M ${r * 0.38} ${-r * 0.38} L ${-r * 0.38} ${r * 0.38}`}
          stroke="#fff" strokeWidth={5.5} strokeLinecap="round" />
      )}
    </g>
  );
};

/* ============ 标题卡 ============ */
const TitleCard = ({data}: {data: any}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 13, mass: 0.7}});
  const pop = spring({frame: f - 6, fps, config: {damping: 10, stiffness: 140}});
  const o = interpolate(f, [0, 8], [0, 1]);
  const zK = clamp01((f - 20) / 18);
  return (
    <AbsoluteFill>
      <Backdrop top={P.bg} bottom={P.bg} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <circle cx={960} cy={348} r={212} fill={P.cardLite} />
        <g transform={`translate(960,392) scale(${clamp01(pop)})`}>
          {/* 枕头底托 */}
          <g transform="translate(0,58) scale(0.92)">
            <Pillow x={0} y={0} s={1} />
          </g>
          {/* 午睡闹钟：20 分钟弧 */}
          <g transform="translate(0,-78) scale(0.86)">
            <NapClock x={0} y={0} r={120} min={20} arc0={0} arcCol={P.mint} />
            <text x={0} y={16} textAnchor="middle" fill={P.ink} fontSize={44}
              fontFamily={FONT_HEAD} fontWeight={900}>20'</text>
          </g>
          <g transform="translate(66,-186)">
            <Zzz x={0} y={0} k={zK} />
          </g>
        </g>
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 268}}>
          <div style={{color: P.sun, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 26}}>
            冷知识 · 第 {data.episode} 期
          </div>
          <div style={{color: C.text, fontSize: 80, fontWeight: 900, fontFamily: FONT_HEAD, letterSpacing: 5}}>
            {data.title}
          </div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ============ S0 二十分钟不是玄学：睡眠坡道阶梯 + 回血收益曲线 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1); // ≈157f “差别不在你多累…”坡道解释

  const pairK = (i: number) => clamp01(spring({frame: f - 30 - i * 16, fps, config: {damping: 12, stiffness: 150}}));
  const stairK = spring({frame: f - 40, fps, config: {damping: 13, stiffness: 120}});
  const winK = clamp01((f - b1 - 10) / 18);
  const personK = clamp01(spring({frame: f - b1 - 30, fps, config: {damping: 11, stiffness: 150}}));
  const backK = clamp01((f - b1 - 90) / 20);
  const cardK = spring({frame: f - 70, fps, config: {damping: 13, stiffness: 120}});
  const curveK = clamp01((f - 95) / 55);
  const peakK = clamp01(spring({frame: f - 150, fps, config: {damping: 9, stiffness: 170}}));
  const negK = clamp01((f - 190) / 20);

  // 台阶：x 段与台面 y
  const steps = [
    {x0: 150, x1: 380, y: 445, label: '清 醒', col: P.cream, wAmp: 5, wFreq: 2.2, ty: 620},
    {x0: 380, x1: 610, y: 545, label: 'N1 浅睡', col: P.n1, wAmp: 9, wFreq: 1.5, ty: 700},
    {x0: 610, x1: 840, y: 645, label: 'N2', col: P.n2, wAmp: 15, wFreq: 0.95, ty: 760},
    {x0: 840, x1: 1070, y: 745, label: 'N3 深睡', col: P.n3, wAmp: 24, wFreq: 0.55, ty: 800},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={0} name={sec.name} accent={P.sun} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* ---- 左上：两种体验对照 ---- */}
        <g transform={`translate(170,176) scale(${pairK(0)})`} opacity={pairK(0)}>
          <rect x={-30} y={-40} width={280} height={84} rx={22} fill={P.card} stroke={P.mint} strokeWidth={4} />
          <Battery x={44} y={0} w={86} h={38} fill={0.96} col={P.mint} />
          <text x={116} y={11} fill={P.mint} fontSize={30} fontFamily={FONT_HEAD} fontWeight={800}>
            20' 像重启
          </text>
        </g>
        <g transform={`translate(500,176) scale(${pairK(1)})`} opacity={pairK(1)}>
          <rect x={-30} y={-40} width={290} height={84} rx={22} fill={P.card} stroke={P.coral} strokeWidth={4} />
          {/* 晕圈 */}
          <g transform="translate(42,0)">
            <circle r={24} fill="none" stroke={P.coral} strokeWidth={6} />
            <path d="M -10 -4 L -2 4 M -2 -4 L -10 4 M 4 -4 L 12 4 M 12 -4 L 4 4"
              stroke={P.coral} strokeWidth={4} strokeLinecap="round" />
          </g>
          <text x={96} y={11} fill={P.coral} fontSize={30} fontFamily={FONT_HEAD} fontWeight={800}>
            1小时 像挨揍
          </text>
        </g>

        {/* ---- 左下：睡眠坡道阶梯 ---- */}
        <g opacity={clamp01(stairK)}>
          {/* 阶梯体 */}
          <path d="M 150 830 L 150 445 L 380 445 L 380 545 L 610 545 L 610 645 L 840 645 L 840 745 L 1070 745 L 1070 830 Z"
            fill={P.stepFace} />
          {steps.map((s, i) => (
            <g key={i}>
              <rect x={s.x0} y={s.y - 18} width={s.x1 - s.x0} height={18} fill={s.col} />
              <text x={(s.x0 + s.x1) / 2} y={s.ty} textAnchor="middle" fill={P.cream}
                fontSize={27} fontFamily={FONT_HEAD} fontWeight={800}>
                {s.label}
              </text>
              {/* 每层的脑电波形：越深处越慢越大 */}
              <path d={wavePath(s.x0 + 26, s.x1 - 26, s.ty + 40, s.wAmp, s.wFreq, i)}
                stroke={s.col} strokeWidth={4.5} fill="none" strokeLinecap="round" opacity={0.9} />
            </g>
          ))}
          {/* 深度方向箭头 */}
          <path d="M 190 840 L 1030 840" stroke={P.muted} strokeWidth={4} strokeDasharray="6 12" />
          <path d="M 1030 840 l -22 -9 m 22 9 l -22 9" stroke={P.muted} strokeWidth={4} fill="none" strokeLinecap="round" />
          <text x={960} y={872} textAnchor="middle" fill={P.muted} fontSize={22} fontFamily={FONT_HEAD}>
            越睡越深 · 20-30 分才摸到深睡门槛
          </text>
        </g>

        {/* 时间刻度虚线 */}
        {[
          {x: 265, t: '0分'},
          {x: 495, t: '≈10分'},
          {x: 725, t: '20分'},
          {x: 950, t: '30分+'},
        ].map((m, i) => (
          <g key={i} opacity={clamp01((f - 60 - i * 8) / 14)}>
            <line x1={m.x} y1={400} x2={m.x} y2={838} stroke={P.muted} strokeWidth={3}
              strokeDasharray="4 12" opacity={0.55} />
            <text x={m.x} y={392} textAnchor="middle" fill={P.muted} fontSize={24} fontFamily={FONT_HEAD}>
              {m.t}
            </text>
          </g>
        ))}

        {/* 回血窗口 bracket：20–30 分窗口跨 N1-N2 */}
        <g opacity={winK}>
          <path d="M 640 366 L 640 348 L 840 348 L 840 366" stroke={P.mint} strokeWidth={5}
            fill="none" strokeLinecap="round" />
          <text x={740} y={330} textAnchor="middle" fill={P.mint} fontSize={27}
            fontFamily={FONT_HEAD} fontWeight={800}>
            蹭完浅睡·窗口 20–30 分
          </text>
        </g>

        {/* 坐在 N2 边沿的瞌睡小人 + 掉头箭 */}
        <g transform={`translate(700,578) scale(${personK})`} opacity={personK}>
          <NapPerson x={0} y={0} s={1} />
          <Zzz x={30} y={-86} k={personK} col={P.sun} />
        </g>
        <g opacity={backK}>
          <path d="M 640 540 Q 480 470 330 462" stroke={P.mint} strokeWidth={7} fill="none"
            strokeDasharray="14 12" strokeLinecap="round" />
          <path d="M 330 462 l 30 -14 m -30 14 l 6 30" stroke={P.mint} strokeWidth={7}
            fill="none" strokeLinecap="round" />
          <text x={265} y={500} textAnchor="middle" fill={P.mint} fontSize={26}
            fontFamily={FONT_HEAD} fontWeight={800}>
            到这层就掉头
          </text>
        </g>

        {/* N3 区警告 */}
        <g opacity={negK}>
          <path d="M 960 610 L 960 690" stroke={P.coral} strokeWidth={7} strokeLinecap="round" />
          <path d="M 944 668 L 960 694 L 976 668" stroke={P.coral} strokeWidth={7} fill="none" strokeLinecap="round" />
          <text x={960} y={560} textAnchor="middle" fill={P.coral} fontSize={26}
            fontFamily={FONT_HEAD} fontWeight={800}>
            掉下去 = 唤醒代价
          </text>
        </g>

        {/* ---- 右卡：回血收益曲线 ---- */}
        <g opacity={clamp01(cardK)}>
          <rect x={1140} y={170} width={680} height={650} rx={40} fill={P.card} stroke={P.sun} strokeWidth={4} />
          <text x={1480} y={226} textAnchor="middle" fill={P.cream} fontSize={33}
            fontFamily={FONT_HEAD} fontWeight={800}>
            回血收益：有峰值的曲线
          </text>
          {/* 零线 + 轴 */}
          <line x1={1210} y1={560} x2={1780} y2={560} stroke={P.muted} strokeWidth={3}
            strokeDasharray="8 10" opacity={0.6} />
          <line x1={1210} y1={560} x2={1210} y2={300} stroke={P.muted} strokeWidth={4} />
          <path d="M 1210 300 l -10 20 m 10 -20 l 10 20" stroke={P.muted} strokeWidth={4} fill="none" strokeLinecap="round" />
          <text x={1190} y={292} textAnchor="end" fill={P.muted} fontSize={23} fontFamily={FONT_HEAD}>收益</text>
          {/* x 刻度 */}
          {[{x: 1210, t: "0'"}, {x: 1400, t: "20'"}, {x: 1590, t: "40'"}, {x: 1750, t: "60'"}].map((m, i) => (
            <text key={i} x={m.x} y={730} textAnchor="middle" fill={P.muted} fontSize={23} fontFamily={FONT_HEAD}>
              {m.t}
            </text>
          ))}
          {/* 负收益区 */}
          <rect x={1520} y={560} width={260} height={90} fill={P.coral} opacity={0.13 * negK} />
          <text x={1650} y={696} textAnchor="middle" fill={P.coral} fontSize={25}
            fontFamily={FONT_HEAD} fontWeight={800} opacity={negK}>
            越过越亏
          </text>
          {/* 收益曲线 */}
          <path d="M 1210 560 C 1270 540 1330 400 1400 372 C 1470 350 1500 480 1540 552
            C 1580 620 1650 648 1750 640"
            stroke={P.sun} strokeWidth={9} fill="none" strokeLinecap="round"
            pathLength={1} strokeDasharray="1" strokeDashoffset={1 - curveK} />
          {/* 峰值甜点 */}
          <g transform={`translate(1400,372) scale(${peakK})`} opacity={peakK}>
            <circle r={16} fill={P.mint} />
            <circle r={26 + Math.sin(f * 0.2) * 4} fill="none" stroke={P.mint}
              strokeWidth={4} strokeDasharray="6 8" />
            <text x={0} y={-44} textAnchor="middle" fill={P.mint} fontSize={28}
              fontFamily={FONT_HEAD} fontWeight={800}>甜点 ≈20分</text>
          </g>
        </g>
      </svg>
      <Stamp text="不是玄学·是拐点" x={850} y={92} at={b1 + 8} color={P.sun} size={48} rot={-4} />
      <Chip text="睡眠是坡道·不是开关" x={140} y={286} at={b1 + 40} accent={P.mint} />
    </FadeIn>
  );
};

/* ============ S1 睡过头的代价：闹钟硬拽 + 睡眠惰性面板 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1); // ≈144f “这时候被叫醒…”

  const clockK = spring({frame: f - 10, fps, config: {damping: 11, stiffness: 140}});
  const ringK = clamp01((f - 50) / 20);
  const dragK = clamp01((f - b1 - 6) / 22);
  const personK = clamp01(spring({frame: f - 26, fps, config: {damping: 13, stiffness: 120}}));
  const fogK = 0.55 + Math.sin(f * 0.1) * 0.15;
  const cardK = spring({frame: f - b1 - 14, fps, config: {damping: 12, stiffness: 130}});
  const barK = (i: number) => clamp01((f - b1 - 50 - i * 18) / 26);
  const recK = clamp01((f - b1 - 130) / 24);
  const noteK = clamp01((f - b1 - 190) / 18);

  const stats = [
    {label: '反应速度', v: 0.24},
    {label: '情绪', v: 0.2},
    {label: '判断力', v: 0.3},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={1} name={sec.name} accent={P.coral} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* ---- 左：闹钟 + 被硬拽的人 ---- */}
        <g transform={`translate(300,340) scale(${clockK})`} opacity={clamp01(clockK)}>
          <NapClock x={0} y={0} r={118} min={45} arc0={30} arcCol={P.coral} ring={ringK} />
          <text x={0} y={190} textAnchor="middle" fill={P.coral} fontSize={27}
            fontFamily={FONT_HEAD} fontWeight={800}>
            响在 45 分 · 深睡边上
          </text>
        </g>

        {/* 时长风险条：0-60 分 */}
        <g opacity={personK} transform="translate(0,0)">
          <rect x={120} y={830} width={760} height={26} rx={13} fill={P.card} />
          <rect x={120} y={830} width={253} height={26} rx={13} fill={P.mint} opacity={0.85} />
          <rect x={373} y={830} width={127} height={26} fill={P.sun} opacity={0.85} />
          <rect x={500} y={830} width={380} height={26} rx={13} fill={P.coral} opacity={0.85} />
          <text x={140} y={886} fill={P.mint} fontSize={23} fontFamily={FONT_HEAD} fontWeight={700}>0-20' 安全区</text>
          <text x={392} y={886} fill={P.sun} fontSize={23} fontFamily={FONT_HEAD} fontWeight={700}>20-30'</text>
          <text x={560} y={886} fill={P.coral} fontSize={23} fontFamily={FONT_HEAD} fontWeight={700}>30'+ 惰性区</text>
          {/* 45 分标记 */}
          <g transform="translate(690,820)">
            <rect x={-4} y={0} width={8} height={46} fill={P.cream} />
            <path d="M -12 -4 L 12 -4 L 0 12 Z" fill={P.cream} />
          </g>
        </g>

        {/* 床上迷糊坐起的人 */}
        <g transform={`translate(560,700)`} opacity={personK}>
          <NapBed x={0} y={0} s={1} w={620} />
          {/* 坐起的人 */}
          <g transform="translate(40,-104)">
            <rect x={-34} y={20} width={76} height={80} rx={26} fill={P.blanket} />
            <rect x={-34} y={20} width={76} height={34} rx={16} fill={P.blanketD} opacity={0.7} />
            <g transform="translate(0,-14)">
              <circle r={42} fill={P.skin} />
              <path d="M -42 -12 A 42 42 0 0 1 38 -26 L 38 -36 Q -2 -56 -42 -38 Z" fill={P.ink} />
              {/* 迷糊眼 */}
              <path d="M -20 -2 Q -14 4 -8 -2 M 8 -2 Q 14 4 20 -2" stroke={P.ink} strokeWidth={4.5}
                fill="none" strokeLinecap="round" />
              <path d="M -8 22 L 8 22" stroke={P.ink} strokeWidth={4} strokeLinecap="round" />
            </g>
          </g>
          {/* 脑袋上的雾 */}
          <FogCloud x={40} y={-150} s={1.05} o={fogK} />
          <text x={40} y={-156} textAnchor="middle" fill={P.ink} fontSize={30}
            fontFamily={FONT_HEAD} fontWeight={900}>?!</text>
        </g>

        {/* 硬拽箭头：N3 → 清醒 */}
        <g opacity={dragK}>
          <rect x={706} y={566} width={120} height={52} rx={14} fill={P.n3} />
          <text x={766} y={601} textAnchor="middle" fill="#fff" fontSize={27}
            fontFamily={FONT_HEAD} fontWeight={800}>N3</text>
          <path d="M 736 556 Q 690 470 640 428" stroke={P.coral} strokeWidth={8} fill="none"
            strokeDasharray="16 12" strokeLinecap="round" />
          <path d="M 640 428 l -8 34 m 8 -34 l 30 18" stroke={P.coral} strokeWidth={8}
            fill="none" strokeLinecap="round" />
          <text x={770} y={476} fill={P.coral} fontSize={25} fontFamily={FONT_HEAD} fontWeight={800}>
            硬拽回清醒
          </text>
        </g>

        {/* ---- 右：睡眠惰性面板 ---- */}
        <g transform={`translate(${(1 - clamp01(cardK)) * 60},0)`} opacity={clamp01(cardK)}>
          <rect x={1000} y={170} width={810} height={660} rx={40} fill={P.card} stroke={P.coral} strokeWidth={4} />
          <text x={1405} y={232} textAnchor="middle" fill={P.cream} fontSize={33}
            fontFamily={FONT_HEAD} fontWeight={800}>
            睡眠惰性 = 醒着但不在场
          </text>
          {stats.map((s, i) => (
            <g key={i} transform={`translate(0,${300 + i * 92})`}>
              <text x={1070} y={10} fill={P.cream} fontSize={29} fontFamily={FONT_HEAD} fontWeight={700}>
                {s.label}
              </text>
              <rect x={1240} y={-16} width={440} height={34} rx={17} fill={P.cardLite} />
              <rect x={1240} y={-16} width={440 * s.v * barK(i)} height={34} rx={17} fill={P.coral} />
              <g transform={`translate(1720,0)`} opacity={barK(i)}>
                <line x1={0} y1={-14} x2={0} y2={14} stroke={P.coral} strokeWidth={6} strokeLinecap="round" />
                <path d="M -10 4 L 0 18 L 10 4" stroke={P.coral} strokeWidth={6} fill="none" strokeLinecap="round" />
              </g>
            </g>
          ))}
          {/* 恢复时长 */}
          <g opacity={recK}>
            <text x={1070} y={620} fill={P.cream} fontSize={29} fontFamily={FONT_HEAD} fontWeight={700}>
              缓过来
            </text>
            <rect x={1240} y={596} width={440} height={26} rx={13} fill={P.cardLite} />
            <rect x={1240} y={596} width={440 * 0.7} height={26} rx={13} fill={P.muted} />
            <circle cx={1240 + 440 * 0.7 * clamp01((f - b1 - 130) / 90)} cy={609} r={20} fill={P.sun} />
            <text x={1240} y={664} fill={P.muted} fontSize={24} fontFamily={FONT_HEAD}>
              几分钟 ~ 半小时 · 缺觉的人更久
            </text>
          </g>
          <text x={1405} y={766} textAnchor="middle" fill={P.coral} fontSize={31}
            fontFamily={FONT_HEAD} fontWeight={900} opacity={noteK}>
            睡一小时 · 收益常常是负的
          </text>
        </g>
      </svg>
      <Stamp text="叫醒的是惰性" x={520} y={110} at={b1 + 8} color={P.coral} size={50} rot={-4} />
      <Chip text="不是没睡到·是睡到错的深度" x={480} y={240} at={b1 + 120} accent={P.sun} />
    </FadeIn>
  );
};

/* ============ S2 犯困是节律安排：24h 警觉度曲线 + 午后低谷带 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1); // ≈339f “午休传统本质…”

  const cardK = spring({frame: f - 8, fps, config: {damping: 13, stiffness: 120}});
  const curveK = clamp01((f - 40) / 80);
  const dipK = clamp01((f - 110) / 20);
  const mealK = clamp01((f - 180) / 22);
  const hamK = clamp01(spring({frame: f - b1 - 10, fps, config: {damping: 12, stiffness: 140}}));
  const noteK = clamp01((f - b1 - 70) / 18);

  // 曲线：x 260→1740 对应 9:00→21:00
  const dipX0 = 768, dipX1 = 1022; // 13:00-15:00

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={2} name={sec.name} accent={P.sky} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <g opacity={clamp01(cardK)}>
          <rect x={110} y={160} width={1700} height={660} rx={40} fill={P.card} stroke={P.sky} strokeWidth={4} />
          <text x={960} y={218} textAnchor="middle" fill={P.cream} fontSize={33}
            fontFamily={FONT_HEAD} fontWeight={800}>
            警觉度自带节律 · 下午有个天然低谷
          </text>

          {/* 太阳（午后位置） */}
          <g transform="translate(1660,262) scale(0.62)">
            <g transform={`rotate(${f * 0.5})`}>
              {Array.from({length: 8}).map((_, i) => (
                <rect key={i} x={-7} y={-108} width={14} height={30} rx={7} fill={P.sun}
                  transform={`rotate(${i * 45})`} />
              ))}
            </g>
            <circle r={78} fill={P.sun} />
          </g>

          {/* 低谷带 */}
          <rect x={dipX0} y={290} width={dipX1 - dipX0} height={370} rx={16}
            fill={P.mint} opacity={0.1 * dipK} />
          <rect x={dipX0} y={290} width={dipX1 - dipX0} height={370} rx={16}
            fill="none" stroke={P.mint} strokeWidth={4} strokeDasharray="12 12" opacity={dipK} />
          <text x={(dipX0 + dipX1) / 2} y={322} textAnchor="middle" fill={P.mint}
            fontSize={28} fontFamily={FONT_HEAD} fontWeight={800} opacity={dipK}>
            天然低谷 13:00–15:00
          </text>
          <text x={(dipX0 + dipX1) / 2} y={360} textAnchor="middle" fill={P.mint}
            fontSize={24} fontFamily={FONT_HEAD} opacity={dipK}>
            不吃也困
          </text>

          {/* 坐标轴 */}
          <line x1={220} y1={680} x2={1760} y2={680} stroke={P.muted} strokeWidth={4} />
          <line x1={220} y1={680} x2={220} y2={290} stroke={P.muted} strokeWidth={4} />
          <path d="M 220 290 l -10 22 m 10 -22 l 10 22" stroke={P.muted} strokeWidth={4}
            fill="none" strokeLinecap="round" />
          <text x={196} y={284} textAnchor="end" fill={P.muted} fontSize={23} fontFamily={FONT_HEAD}>
            警觉度
          </text>
          {[260, 640, 1020, 1400, 1760].map((x, i) => {
            const hs = ['9:00', '12:00', '15:00', '18:00', '21:00'];
            return (
              <g key={i}>
                <line x1={x} y1={680} x2={x} y2={692} stroke={P.muted} strokeWidth={4} />
                <text x={x} y={724} textAnchor="middle" fill={P.muted} fontSize={23}
                  fontFamily={FONT_HEAD}>{hs[i]}</text>
              </g>
            );
          })}

          {/* 警觉度曲线 */}
          <path d="M 260 380 C 360 330 470 300 580 320 C 680 350 730 490 800 545
            C 860 590 960 595 1010 550 C 1090 470 1180 420 1270 432
            C 1400 450 1530 540 1620 590 C 1690 628 1730 640 1760 646"
            stroke={P.sky} strokeWidth={9} fill="none" strokeLinecap="round"
            pathLength={1} strokeDasharray="1" strokeDashoffset={1 - curveK} />

          {/* 饭后更深的虚线谷 */}
          <g opacity={mealK}>
            <path d="M 790 548 C 860 636 960 640 1016 560" stroke={P.coral}
              strokeWidth={7} fill="none" strokeDasharray="12 10" strokeLinecap="round" />
            <RiceBowl x={580} y={600} s={0.9} />
            <path d="M 636 548 Q 720 580 796 606" stroke={P.coral} strokeWidth={6}
              fill="none" strokeDasharray="10 9" strokeLinecap="round" />
            <path d="M 802 606 l -26 -8 m 26 8 l -14 24" stroke={P.coral} strokeWidth={6}
              fill="none" strokeLinecap="round" />
            <text x={510} y={676} textAnchor="middle" fill={P.coral} fontSize={25}
              fontFamily={FONT_HEAD} fontWeight={800}>
              吃了更困 · 副交感推一把
            </text>
          </g>

          {/* 午休文化：吊床 */}
          <g transform={`translate(1420,330) scale(${hamK})`} opacity={hamK}>
            <Hammock x={0} y={0} s={0.78} />
            <text x={0} y={92} textAnchor="middle" fill={P.cream} fontSize={25}
              fontFamily={FONT_HEAD} fontWeight={800}>
              午休文化 = 对齐低谷
            </text>
          </g>

          <text x={960} y={792} textAnchor="middle" fill={P.cream} fontSize={27}
            fontFamily={FONT_HEAD} fontWeight={700} opacity={noteK}>
            低谷是真的 · 但低谷 ≠ 需要长睡
          </text>
        </g>
      </svg>
      <Stamp text="生物钟排好的节目" x={1180} y={96} at={90} color={P.sky} size={46} rot={-4} />
      <Chip text="困意是正常信号" x={140} y={856} at={b1 + 20} accent={P.mint} />
    </FadeIn>
  );
};

/* ============ S3 NASA 和咖啡盹：驾驶舱小睡 + 三步咖啡盹流程 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1); // ≈232f “进阶玩法是咖啡盹…”

  const cardLK = spring({frame: f - 8, fps, config: {damping: 13, stiffness: 120}});
  const badgeK = clamp01(spring({frame: f - 60, fps, config: {damping: 9, stiffness: 160}}));
  const gaugeK = clamp01(spring({frame: f - 110, fps, config: {damping: 13, stiffness: 60, mass: 1.4}}));
  const cardRK = spring({frame: f - b1 - 6, fps, config: {damping: 12, stiffness: 130}});
  const stepK = (i: number) => clamp01(spring({frame: f - b1 - 30 - i * 22, fps, config: {damping: 11, stiffness: 160}}));
  const whyK = clamp01((f - b1 - 120) / 18);
  const warnK = clamp01((f - b1 - 220) / 18);

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={3} name={sec.name} accent={P.sun} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* ---- 左卡：NASA 驾驶舱小睡 ---- */}
        <g opacity={clamp01(cardLK)}>
          <rect x={110} y={170} width={820} height={650} rx={40} fill={P.card} stroke={P.sun} strokeWidth={4} />
          <text x={520} y={228} textAnchor="middle" fill={P.cream} fontSize={33}
            fontFamily={FONT_HEAD} fontWeight={800}>
            NASA · 机组小睡研究
          </text>
          {/* 驾驶舱窗 */}
          <rect x={190} y={270} width={400} height={210} rx={26} fill={P.sky} opacity={0.9} />
          <rect x={190} y={270} width={400} height={210} rx={26} fill="none" stroke={P.woodD} strokeWidth={10} />
          <line x1={390} y1={276} x2={390} y2={474} stroke={P.woodD} strokeWidth={8} />
          {/* 窗外云 */}
          <ellipse cx={280} cy={380} rx={70} ry={20} fill="#fff" opacity={0.75} />
          <ellipse cx={470} cy={330} rx={56} ry={16} fill="#fff" opacity={0.6} />
          {/* 操纵杆 */}
          <path d="M 300 560 L 330 470" stroke={P.ink} strokeWidth={14} strokeLinecap="round" />
          <circle cx={330} cy={468} r={22} fill={P.ink} />
          {/* 飞行员：坐椅 + 瞌睡头 */}
          <g transform="translate(660,400)">
            <rect x={-60} y={-10} width={130} height={150} rx={30} fill={P.woodD} />
            <rect x={-46} y={4} width={102} height={120} rx={24} fill={P.wood} />
            <NapPerson x={8} y={-34} s={0.95} hoodie={P.mint} />
            <Zzz x={40} y={-120} k={1} col={P.cream} />
          </g>
          <text x={520} y={566} textAnchor="middle" fill={P.muted} fontSize={24} fontFamily={FONT_HEAD}>
            计划性驾驶舱休息 · 长途飞行员
          </text>
          {/* 26 分大徽章 */}
          <g transform={`translate(310,660) scale(${badgeK})`} opacity={badgeK}>
            <circle r={72} fill={P.sun} />
            <circle r={72} fill="none" stroke={P.cardLite} strokeWidth={5} strokeDasharray="10 10" />
            <text x={0} y={-4} textAnchor="middle" fill={P.ink} fontSize={44}
              fontFamily={FONT_HEAD} fontWeight={900}>26</text>
            <text x={0} y={34} textAnchor="middle" fill={P.ink} fontSize={24}
              fontFamily={FONT_HEAD} fontWeight={800}>分钟</text>
          </g>
          {/* 警觉度仪表 */}
          <g transform="translate(640,700)">
            <Gauge x={0} y={0} r={110} k={gaugeK} />
            <text x={0} y={48} textAnchor="middle" fill={P.mint} fontSize={26}
              fontFamily={FONT_HEAD} fontWeight={800}>警觉度 ↑</text>
            <text x={0} y={86} textAnchor="middle" fill={P.muted} fontSize={21} fontFamily={FONT_HEAD}>
              流传版：+50%（以原始报告为准）
            </text>
          </g>
        </g>

        {/* ---- 右卡：咖啡盹流程 ---- */}
        <g transform={`translate(${(1 - clamp01(cardRK)) * 60},0)`} opacity={clamp01(cardRK)}>
          <rect x={990} y={170} width={820} height={650} rx={40} fill={P.card} stroke={P.mint} strokeWidth={4} />
          <text x={1400} y={228} textAnchor="middle" fill={P.cream} fontSize={33}
            fontFamily={FONT_HEAD} fontWeight={800}>
            咖啡盹 · 进阶玩法
          </text>
          {/* 三步流程 */}
          <g transform="translate(1150,380)">
            <CoffeeCup x={0} y={0} s={1.1} k={stepK(0)} />
            <text x={0} y={110} textAnchor="middle" fill={P.cream} fontSize={26}
              fontFamily={FONT_HEAD} fontWeight={800} opacity={stepK(0)}>① 灌一杯</text>
          </g>
          <g transform="translate(1400,380)" opacity={stepK(1)}>
            <Pillow x={0} y={6} s={0.62} />
            <Zzz x={-16} y={-60} k={1} col={P.sky} />
            <text x={0} y={110} textAnchor="middle" fill={P.cream} fontSize={26}
              fontFamily={FONT_HEAD} fontWeight={800}>② 睡 20 分</text>
          </g>
          <g transform={`translate(1650,372) scale(${stepK(2)})`} opacity={stepK(2)}>
            <path d="M -12 -62 L 30 -62 L 6 -8 L 34 -8 L -18 62 L -4 -8 L -32 -8 Z"
              fill={P.sun} stroke={P.sunD} strokeWidth={4} strokeLinejoin="round" />
            <Battery x={0} y={-4} w={96} h={40} fill={0.92} col={P.mint} />
            <text x={0} y={110} textAnchor="middle" fill={P.cream} fontSize={26}
              fontFamily={FONT_HEAD} fontWeight={800}>③ 双重清醒</text>
          </g>
          {/* 连接箭头 */}
          {[0, 1].map((i) => (
            <g key={i} opacity={stepK(i + 1)} transform={`translate(${1268 + i * 250},376)`}>
              <line x1={0} y1={0} x2={52} y2={0} stroke={P.mint} strokeWidth={7} strokeLinecap="round" />
              <path d="M 52 0 l -18 -11 m 18 11 l -18 11" stroke={P.mint} strokeWidth={7}
                fill="none" strokeLinecap="round" />
            </g>
          ))}
          {/* 原理 */}
          <g opacity={whyK}>
            {[
              '咖啡因吸收 ≈20 分 · 睡着正好等它生效',
              '睡眠清掉腺苷 · 受体腾出来给咖啡因',
            ].map((t, i) => (
              <g key={i} transform={`translate(1060,${560 + i * 64})`}>
                <CheckDot x={0} y={0} r={19} k={1} />
                <text x={40} y={10} fill={P.cream} fontSize={27} fontFamily={FONT_HEAD}>{t}</text>
              </g>
            ))}
          </g>
          {/* 边界警告 */}
          <g opacity={warnK}>
            <rect x={1060} y={706} width={700} height={72} rx={20} fill={P.cardLite}
              stroke={P.coral} strokeWidth={4} />
            <text x={1410} y={752} textAnchor="middle" fill={P.coral} fontSize={27}
              fontFamily={FONT_HEAD} fontWeight={800}>
              咖啡因半衰期 5–6h · 15 点后慎用
            </text>
          </g>
        </g>
      </svg>
      <Stamp text="提神翻倍" x={1230} y={96} at={b1 + 30} color={P.mint} size={50} rot={-4} />
      <Chip text="小睡=工程参数·不是玄学" x={600} y={100} at={120} accent={P.sun} />
    </FadeIn>
  );
};

/* ============ S4 二十分钟不是处方：三类人卡片 + 起始值滑杆 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1); // ≈280f “最需要警惕的是失眠者…”
  const b2 = beat(2); // ≈478f “拿它当起始值…”

  const cardK = (i: number) => clamp01(spring({frame: f - 24 - i * 18, fps, config: {damping: 12, stiffness: 140}}));
  const warnK = clamp01((f - b1 - 8) / 18);
  const slideK = spring({frame: f - b2 - 4, fps, config: {damping: 12, stiffness: 130}});
  const knobWob = Math.sin(f * 0.15) * 8;

  const cards = [
    {
      x: 110, col: P.mint, title: '老人', sub: '深睡份额本来就薄',
      desc: '多睡会儿也未必掉深', pill: "可放宽 ~45'",
    },
    {
      x: 695, col: P.sun, title: '通宵党', sub: '睡眠债没还清',
      desc: "20' 不够还 · 睡久伤晚上", pill: '先补夜里的觉',
    },
    {
      x: 1280, col: P.coral, title: '失眠者', sub: '午睡泄掉睡眠压力',
      desc: '越补觉晚上越睡不着', pill: '建议不午睡',
    },
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={4} name={sec.name} accent={P.mint} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {cards.map((c, i) => (
          <g key={i} transform={`translate(${c.x},${(1 - cardK(i)) * 40})`} opacity={cardK(i)}>
            <rect x={0} y={190} width={530} height={330} rx={36} fill={P.card}
              stroke={c.col} strokeWidth={i === 2 ? 5 + warnK * 2 : 4} />
            <text x={265} y={248} textAnchor="middle" fill={P.cream} fontSize={34}
              fontFamily={FONT_HEAD} fontWeight={900}>{c.title}</text>
            {/* 头像 */}
            <g transform="translate(120,340)">
              <HeadIcon x={0} y={0} s={1.05}
                hair={i === 0 ? '#d8d8e0' : P.ink}
                mood={i === 0 ? 'sleep' : i === 1 ? 'groggy' : 'wired'} />
              {i === 0 && <Zzz x={-20} y={-72} k={cardK(0)} col={P.mint} />}
              {i === 1 && (
                <g transform="translate(44,-62)">
                  <circle r={20} fill={P.pillow} />
                  <circle cx={-8} cy={-4} r={16} fill={P.bg} />
                </g>
              )}
              {i === 2 && warnK > 0.3 && (
                <g transform="translate(66,-74)">
                  <Pillow x={0} y={0} s={0.3} />
                  <line x1={-20} y1={-20} x2={20} y2={20} stroke={P.coral} strokeWidth={7} strokeLinecap="round" />
                  <line x1={20} y1={-20} x2={-20} y2={20} stroke={P.coral} strokeWidth={7} strokeLinecap="round" />
                </g>
              )}
            </g>
            <text x={200} y={316} fill={c.col} fontSize={27} fontFamily={FONT_HEAD} fontWeight={800}>
              {c.sub}
            </text>
            <text x={200} y={362} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
              {c.desc}
            </text>
            {/* 结论胶囊 */}
            <g transform="translate(200,430)">
              <rect x={-14} y={-30} width={c.pill.length * 30 + 34} height={54} rx={27}
                fill={P.cardLite} stroke={c.col} strokeWidth={3.5} />
              <text x={c.pill.length * 15 + 3} y={8} textAnchor="middle" fill={c.col}
                fontSize={27} fontFamily={FONT_HEAD} fontWeight={800}>
                {c.pill}
              </text>
            </g>
          </g>
        ))}

        {/* 失眠者卡片的警示涟漪 */}
        {warnK > 0.4 && <Ripple x={1545} y={355} startFrame={b1 + 8} color={P.coral} />}

        {/* ---- 底条：把 20 分钟当起始值 ---- */}
        <g transform={`translate(0,${(1 - clamp01(slideK)) * 60})`} opacity={clamp01(slideK)}>
          <rect x={110} y={576} width={1700} height={280} rx={36} fill={P.card} stroke={P.mint} strokeWidth={4} />
          <text x={960} y={634} textAnchor="middle" fill={P.cream} fontSize={31}
            fontFamily={FONT_HEAD} fontWeight={800}>
            把 20 分钟当起始值 · 按自己的体感微调
          </text>
          {/* 滑杆 */}
          <rect x={300} y={690} width={1320} height={18} rx={9} fill={P.cardLite} />
          <rect x={520} y={690} width={500} height={18} rx={9} fill={P.mint} opacity={0.4} />
          {[300, 520, 740, 1020, 1620].map((x, i) => {
            const ts = ["10'", "20'", "30'", "45'", "60'"];
            const hot = i === 1;
            return (
              <g key={i}>
                <line x1={x} y1={682} x2={x} y2={716} stroke={hot ? P.mint : P.muted}
                  strokeWidth={hot ? 6 : 4} />
                <text x={x} y={752} textAnchor="middle" fill={hot ? P.mint : P.muted}
                  fontSize={hot ? 30 : 25} fontFamily={FONT_HEAD} fontWeight={hot ? 900 : 400}>
                  {ts[i]}
                </text>
              </g>
            );
          })}
          {/* 旋钮停在 20' */}
          <g transform={`translate(${520 + knobWob * 0.4},699)`}>
            <circle r={30} fill={P.mint} stroke={P.cream} strokeWidth={5} />
            <text x={0} y={9} textAnchor="middle" fill={P.ink} fontSize={24}
              fontFamily={FONT_HEAD} fontWeight={900}>起</text>
          </g>
          <text x={960} y={816} textAnchor="middle" fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            连试一周 · 醒后给自己打清醒分 · 找到你的甜点
          </text>
        </g>
      </svg>
      <Stamp text="甜点≠处方" x={760} y={100} at={b2 + 6} color={P.sun} size={50} rot={-4} />
      <Chip text="失眠的人先修夜里" x={140} y={116} at={b1 + 30} accent={P.coral} />
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
        <Stars n={18} from={0} dur={24} />
        <g transform="translate(960,872) scale(0.72)">
          <g transform={`scale(${clamp01(pop)})`}>
            <g transform="translate(-260,0)">
              <NapBed x={0} y={0} s={1} w={640} />
              <g transform="translate(0,-58)">
                <Pillow x={0} y={0} s={0.6} />
              </g>
            </g>
            <g transform="translate(420,-90)">
              <NapClock x={0} y={0} r={100} min={20} arc0={0} arcCol={P.mint} />
            </g>
            <g transform="translate(-440,-240)">
              <Zzz x={0} y={0} k={1} col={P.sky} />
            </g>
          </g>
        </g>
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -300}}>
        <div style={{fontSize: 62, fontWeight: 800, color: C.text}}>睡对二十分钟，下午像重启</div>
        <div style={{fontSize: 38, color: C.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E227 = ({data = tl}: {data?: any}) => {
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
