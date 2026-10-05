import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01, rand} from '../lib/style';
import {Backdrop, Ripple, Stamp, SceneTag, Chip, Subtitles, ProgressBar, FadeIn} from '../lib/atoms';
import tl from '../timeline_225.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：梦境紫夜 × 屏光青 × 便签黄 × 珊瑚红（纯色平涂） ============ */
const P = {
  bg: '#231f52',
  bgDeep: '#17133a',
  bgLite: '#2e2968',
  card: '#3d3783',
  cardD: '#2c2762',
  cream: '#f6f0dd',
  ink: '#1a163c',
  moon: '#ffd873',
  moonD: '#e0ab40',
  cyan: '#6fe3ff',
  cyanD: '#35a9cc',
  violet: '#b79cf5',
  violetD: '#8467cf',
  coral: '#ff7d6e',
  coralD: '#cf5244',
  mint: '#86e2b0',
  mintD: '#4dab7d',
  amber: '#ffc94d',
  skin: '#f2c79b',
  skinD: '#d99f6b',
  brain: '#f2a9bd',
  brainD: '#d17a94',
  grey: '#9aa3c4',
  pillow: '#eef1fb',
  mattress: '#d9dff2',
  blanket: '#7c68c8',
  blanketD: '#5e4ba3',
  wood: '#8f6240',
  woodD: '#6a4729',
  screen: '#101a3a',
  paper: '#f6f0dd',
};

/* ---------- 月亮 ---------- */
const MoonIcon = ({x, y, s = 1, op = 1, hole = P.bg}: {x: number; y: number; s?: number; op?: number; hole?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
    <circle r={56} fill={P.moon} />
    <circle cx={26} cy={-16} r={46} fill={hole} />
    <circle cx={-18} cy={22} r={8} fill={P.moonD} />
    <circle cx={-34} cy={-6} r={5} fill={P.moonD} />
  </g>
);

/* ---------- 床（圆角软包床头） ---------- */
const Bed = ({x, y, s = 1, op = 1}: {x: number; y: number; s?: number; op?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
    <rect x={-200} y={-170} width={42} height={196} rx={20} fill={P.blanketD} />
    <rect x={-188} y={-158} width={18} height={160} rx={9} fill={P.blanket} opacity={0.7} />
    <rect x={-150} y={26} width={26} height={52} rx={8} fill={P.woodD} />
    <rect x={150} y={26} width={26} height={52} rx={8} fill={P.woodD} />
    <rect x={-168} y={-40} width={356} height={74} rx={18} fill={P.mattress} stroke={P.card} strokeWidth={5} />
    <rect x={-146} y={-68} width={104} height={42} rx={19} fill={P.pillow} stroke={P.card} strokeWidth={5} />
    <rect x={-30} y={-72} width={218} height={80} rx={18} fill={P.blanket} stroke={P.blanketD} strokeWidth={5} />
    <rect x={-30} y={-72} width={218} height={24} rx={12} fill={P.blanketD} opacity={0.55} />
  </g>
);

/* ---------- 睡觉小人头 ---------- */
const Sleeper = ({x, y, s = 1, op = 1, z = 0}: {x: number; y: number; s?: number; op?: number; z?: number}) => {
  const f = useCurrentFrame();
  return (
    <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
      <circle r={46} fill={P.skin} stroke={P.skinD} strokeWidth={5} />
      <path d="M -46 -4 A 46 46 0 0 1 46 -4 Z" fill="#4a3a78" />
      <g stroke={P.ink} strokeWidth={4} fill="none" strokeLinecap="round">
        <path d="M -26 8 Q -17 16 -8 8" />
        <path d="M 10 8 Q 19 16 28 8" />
        <path d="M -2 30 Q 6 35 14 30" />
      </g>
      {z > 0 && [0, 1, 2].map((i) => {
        const t = (f / 30 + i * 0.9) % 3;
        return (
          <text key={i} x={56 + i * 40 + t * 22} y={-36 - t * 52} fill={P.cyan} fontSize={42 - i * 7}
            fontFamily={FONT_HEAD} fontWeight={800} opacity={z * (1 - t / 3)}>Z</text>
        );
      })}
    </g>
  );
};

/* ---------- 一行"文字"：mode 0 清晰直条 / mode 1 糊成波浪 ---------- */
const GlyphRow = ({y, w, seed, mode, color = P.cyan}: {y: number; w: number; seed: number; mode: number; color?: string}) => {
  if (mode === 0) {
    return <rect x={-w / 2} y={y} width={w} height={13} rx={6.5} fill={color} opacity={0.95} />;
  }
  // 波浪乱码段
  const segs = 6;
  const pts: string[] = [];
  for (let i = 0; i <= segs; i++) {
    const px = -w / 2 + (w * i) / segs;
    const py = y + 6 + Math.sin(seed * 3 + i * 1.7) * 7;
    pts.push(`${i === 0 ? 'M' : 'L'} ${px.toFixed(1)} ${py.toFixed(1)}`);
  }
  return <path d={pts.join(' ')} fill="none" stroke={color} strokeWidth={9} strokeLinecap="round" opacity={0.9} />;
};

/* ---------- 梦里的手机：屏幕字会糊/会变 ---------- */
const DreamPhone = ({x, y, s = 1, rot = 0, op = 1, seed = 1, glitch = 1}: {
  x: number; y: number; s?: number; rot?: number; op?: number; seed?: number; glitch?: number;
}) => {
  const f = useCurrentFrame();
  const swap = Math.floor(f / 34); // 每隔一阵自动换一版
  return (
    <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`} opacity={op}>
      <rect x={-82} y={-150} width={164} height={300} rx={26} fill={P.pillow} stroke={P.card} strokeWidth={6} />
      <rect x={-68} y={-134} width={136} height={268} rx={16} fill={P.screen} />
      <circle cy={140} r={0.1} fill="none" />
      <rect x={-22} y={-126} width={44} height={8} rx={4} fill={P.card} />
      {/* 屏光打在壳上的一圈 */}
      <rect x={-68} y={-134} width={136} height={268} rx={16} fill="none" stroke={P.cyan} strokeWidth={3} opacity={0.5} />
      {/* 通知头 */}
      <circle cx={-48} cy={-108} r={9} fill={P.mint} />
      <rect x={-32} y={-114} width={72} height={11} rx={5.5} fill={P.grey} opacity={0.85} />
      {/* 正文行：glitch 时按 swap 换版 + 波浪化 */}
      {[0, 1, 2, 3].map((i) => {
        const v = seed + swap + i * 7;
        const w = 92 - rand(v, 2) * 44;
        return (
          <GlyphRow key={`${i}-${swap}`} y={-76 + i * 36} w={w} seed={v}
            mode={glitch > 0.5 ? 1 : 0} color={i === 0 ? P.cream : P.cyan} />
        );
      })}
      <rect x={-52} y={76} width={104} height={42} rx={14} fill={P.card} />
      <text x={0} y={104} textAnchor="middle" fontSize={24} fill={P.cyan} fontFamily={FONT_HEAD} fontWeight={700}>
        {glitch > 0.5 ? '？？？' : '详情'}
      </text>
    </g>
  );
};

/* ---------- 小城堡（"画面高清"的象征） ---------- */
const Castle = ({x, y, s = 1, op = 1, body = P.mint}: {x: number; y: number; s?: number; op?: number; body?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
    <rect x={-86} y={-20} width={172} height={92} rx={8} fill={body} stroke={P.ink} strokeWidth={5} />
    <rect x={-70} y={-66} width={42} height={52} rx={6} fill={body} stroke={P.ink} strokeWidth={5} />
    <rect x={28} y={-66} width={42} height={52} rx={6} fill={body} stroke={P.ink} strokeWidth={5} />
    <rect x={-22} y={-88} width={44} height={74} rx={6} fill={body} stroke={P.ink} strokeWidth={5} />
    {[-86, -44, 0, 44].map((cx, i) => (
      <rect key={i} x={cx - 10} y={-34} width={20} height={16} fill={body} stroke={P.ink} strokeWidth={4} />
    ))}
    <path d="M -18 72 L -18 30 A 18 18 0 0 1 18 30 L 18 72 Z" fill={P.ink} />
    <rect x={-56} y={0} width={16} height={22} rx={8} fill={P.moon} />
    <rect x={40} y={0} width={16} height={22} rx={8} fill={P.moon} />
    <path d="M 0 -88 L 0 -118 M 0 -118 L 34 -108 L 0 -98" stroke={P.coral} strokeWidth={6} fill={P.coral} strokeLinejoin="round" />
  </g>
);

/* ---------- 大脑轮廓 + 分区亮暗 ---------- */
const BrainMap = ({x, y, s = 1, op = 1, lit}: {
  x: number; y: number; s?: number; op?: number;
  lit: {vc: number; amy: number; pfc: number; tpj: number};
}) => {
  const f = useCurrentFrame();
  const pulse = 0.55 + 0.45 * Math.sin(f / 7);
  const Region = ({cx, cy, on, col}: {cx: number; cy: number; on: number; col: string}) => (
    <g>
      <circle cx={cx} cy={cy} r={36} fill={col} opacity={on * (0.25 + 0.2 * pulse)} />
      <circle cx={cx} cy={cy} r={17} fill={on > 0.5 ? col : P.grey} stroke={P.ink} strokeWidth={4}
        strokeDasharray={on > 0.5 ? undefined : '7 6'} opacity={on > 0.5 ? 1 : 0.85} />
    </g>
  );
  return (
    <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
      <path d="M -96 20 C -122 -22 -104 -72 -66 -86 C -52 -118 -4 -126 26 -104 C 66 -116 104 -90 106 -48
        C 134 -28 130 16 104 34 C 110 70 74 94 42 80 C 16 100 -36 94 -52 66 C -86 66 -106 46 -96 20 Z"
        fill={P.brain} stroke={P.brainD} strokeWidth={6} strokeLinejoin="round" />
      <g stroke={P.brainD} strokeWidth={5} fill="none" strokeLinecap="round">
        <path d="M -78 -42 Q -50 -68 -22 -46" />
        <path d="M -64 18 Q -32 -4 -4 20" />
        <path d="M 8 -74 Q 44 -92 70 -64" />
        <path d="M 30 32 Q 60 12 88 36" />
        <path d="M -30 -8 Q 0 -28 30 -10" />
      </g>
      <Region cx={-72} cy={-30} on={lit.vc} col={P.mint} />
      <Region cx={-18} cy={62} on={lit.amy} col={P.coral} />
      <Region cx={64} cy={-62} on={lit.pfc} col={P.amber} />
      <Region cx={20} cy={-6} on={lit.tpj} col={P.violet} />
    </g>
  );
};

/* ---------- 眼睛（眨眼） ---------- */
const Eye = ({x, y, s = 1, open = 1}: {x: number; y: number; s?: number; open?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s},${Math.max(0.06, open) * s})`}>
    <path d="M -58 0 Q 0 -52 58 0 Q 0 52 -58 0 Z" fill={P.cream} stroke={P.ink} strokeWidth={5} />
    <circle r={21} fill={P.cyan} stroke={P.ink} strokeWidth={4} />
    <circle r={9} fill={P.ink} />
    <circle cx={6} cy={-7} r={4} fill="#fff" />
  </g>
);

/* ---------- 数字时钟牌 ---------- */
const DigiClock = ({x, y, s = 1, txt = '18:37', op = 1}: {x: number; y: number; s?: number; txt?: string; op?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
    <rect x={-92} y={-44} width={184} height={88} rx={14} fill={P.ink} stroke={P.card} strokeWidth={5} />
    <text x={0} y={16} textAnchor="middle" fontSize={46} fill={P.cyan} fontFamily={FONT_HEAD} fontWeight={800}
      letterSpacing={4}>{txt}</text>
  </g>
);

/* ---------- 路牌 ---------- */
const RoadSign = ({x, y, s = 1, txt = '北门路', op = 1}: {x: number; y: number; s?: number; txt?: string; op?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
    <rect x={-8} y={34} width={16} height={70} fill={P.grey} />
    <rect x={-110} y={-52} width={220} height={96} rx={12} fill={P.mintD} stroke={P.ink} strokeWidth={5} />
    <path d="M -88 -4 L -64 -22 L -64 -8 L -18 -8 L -18 10 L -64 10 L -64 24 Z" fill={P.cream} opacity={0.9} />
    <text x={55} y={8} textAnchor="middle" fontSize={34} fill="#fff" fontFamily={FONT_HEAD} fontWeight={800}>{txt}</text>
  </g>
);

/* ---------- 手掌（可画 6 指） ---------- */
const Hand = ({x, y, s = 1, op = 1, extra = false}: {x: number; y: number; s?: number; op?: number; extra?: boolean}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
    {[-2, -1, 0, 1, 2].map((i) => (
      <rect key={i} x={i * 26 - 9} y={-92 - (i === 0 ? 8 : Math.abs(i) === 1 ? 0 : 10)} width={18}
        height={78} rx={9} fill={P.skin} stroke={P.skinD} strokeWidth={4} />
    ))}
    {extra && <rect x={-78} y={-78} width={18} height={62} rx={9} fill={P.skin} stroke={P.skinD} strokeWidth={4}
      transform="rotate(-18 -69 -47)" />}
    <rect x={-72} y={-30} width={132} height={96} rx={34} fill={P.skin} stroke={P.skinD} strokeWidth={5} />
    <rect x={-88} y={-12} width={30} height={66} rx={15} fill={P.skin} stroke={P.skinD} strokeWidth={4}
      transform="rotate(28 -73 21)" />
  </g>
);

/* ---------- 镜子里的怪脸 ---------- */
const Mirror = ({x, y, s = 1, op = 1}: {x: number; y: number; s?: number; op?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
    <ellipse rx={86} ry={112} fill={P.bgLite} stroke={P.amber} strokeWidth={9} />
    <ellipse rx={86} ry={112} fill="none" stroke={P.amber} strokeWidth={4} opacity={0.5} transform="scale(0.88)" />
    <ellipse cy={14} rx={46} ry={52} fill={P.skin} stroke={P.skinD} strokeWidth={4} />
    <circle cx={-18} cy={2} r={7} fill={P.ink} />
    <circle cx={22} cy={-8} r={11} fill={P.ink} />
    <path d="M -14 34 Q 4 26 22 40" stroke={P.ink} strokeWidth={4} fill="none" strokeLinecap="round" />
    <path d="M 36 -44 L 58 -60" stroke={P.violet} strokeWidth={5} strokeLinecap="round" strokeDasharray="7 6" />
  </g>
);

/* ---------- 流水线站点 ---------- */
const Station = ({x, y, icon, name, accent, on, broke}: {
  x: number; y: number; icon: number; name: string; accent: string; on: number; broke: number;
}) => {
  const f = useCurrentFrame();
  const dim = broke > 0.5 ? 0.35 : 1;
  return (
    <g transform={`translate(${x},${y})`} opacity={dim}>
      <rect x={-160} y={-120} width={320} height={180} rx={22} fill={P.card}
        stroke={on > 0.5 ? accent : P.cardD} strokeWidth={on > 0.5 ? 7 : 4} />
      <circle cy={-34} r={46} fill={P.bgLite} stroke={on > 0.5 ? accent : P.cardD} strokeWidth={5} />
      <g transform={`translate(0,-34)`} opacity={on > 0.5 ? 1 : 0.55}>
        {icon === 0 && (
          <g>
            <Eye x={0} y={-6} s={0.42} />
            <text x={0} y={38} textAnchor="middle" fontSize={26} fill={accent} fontFamily={FONT_HEAD} fontWeight={800}>字</text>
          </g>
        )}
        {icon === 1 && (
          <g>
            <rect x={-26} y={-22} width={52} height={40} rx={12} fill={accent} />
            <path d="M -10 18 L -18 34 L 2 18 Z" fill={accent} />
            <line x1={-14} y1={-8} x2={14} y2={-8} stroke={P.ink} strokeWidth={5} strokeLinecap="round" />
            <line x1={-14} y1={6} x2={6} y2={6} stroke={P.ink} strokeWidth={5} strokeLinecap="round" />
          </g>
        )}
        {icon === 2 && (
          <g transform={`rotate(${f * (on > 0.5 ? 1.2 : 0)})`}>
            <circle r={20} fill="none" stroke={accent} strokeWidth={9} />
            {[0, 60, 120, 180, 240, 300].map((a) => {
              const r = a * Math.PI / 180;
              return <rect key={a} x={-6} y={-34} width={12} height={14} rx={4} fill={accent}
                transform={`rotate(${a})`} />;
            })}
            <circle r={7} fill={accent} />
          </g>
        )}
        {icon === 3 && (
          <g stroke={accent} strokeWidth={8} fill="none" strokeLinecap="round">
            <rect x={-34} y={-12} width={40} height={24} rx={12} />
            <rect x={-6} y={-12} width={40} height={24} rx={12} />
          </g>
        )}
      </g>
      <text x={0} y={42} textAnchor="middle" fontSize={30} fill={on > 0.5 ? P.cream : P.grey}
        fontFamily={FONT_HEAD} fontWeight={800}>{name}</text>
      {broke > 0 && (
        <g opacity={broke} stroke={P.coral} strokeWidth={9} strokeLinecap="round">
          <line x1={-30} y1={-64} x2={30} y2={-4} />
          <line x1={30} y1={-64} x2={-30} y2={-4} />
        </g>
      )}
    </g>
  );
};

/* ============ 标题卡 ============ */
const TitleCard = ({data}: {data: any}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 13, mass: 0.7}});
  const o = interpolate(f, [0, 8], [0, 1]);
  const p0 = Math.max(0.001, spring({frame: f - 4, fps, config: {damping: 10, stiffness: 150}}));
  const p1 = Math.max(0.001, spring({frame: f - 12, fps, config: {damping: 10, stiffness: 150}}));
  const bob = Math.sin(f / 15) * 6;
  return (
    <AbsoluteFill>
      <Backdrop top={P.bg} bottom={P.bg} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <circle cx={960} cy={338} r={222} fill={P.cream} />
        <circle cx={960} cy={338} r={200} fill={P.bgLite} />
        <g transform={`translate(960,338) scale(${p0})`}>
          <g transform="translate(-960,-338)">
            <MoonIcon x={838} y={252} s={0.55} hole={P.bgLite} />
            {[0, 1, 2].map((i) => (
              <circle key={i} cx={880 + rand(i, 51) * 200} cy={210 + rand(i, 52) * 60} r={3}
                fill="#fff" opacity={0.8} />
            ))}
            <Sleeper x={886} y={420} s={0.95} />
            <g transform={`translate(1022,${300 + bob * 0.6}) rotate(12)`}>
              <DreamPhone x={0} y={0} s={0.62} seed={3} />
            </g>
            <Castle x={1108} y={428} s={0.5} />
          </g>
        </g>
        <g transform={`translate(762,${452 - bob * 0.4}) scale(${p1 * 0.9})`}>
          <EllipseHint />
        </g>
        <g transform={`translate(1180,${560 + bob * 0.5}) scale(${p1})`}>
          <g transform="rotate(8)">
            <rect x={-92} y={-34} width={184} height={62} rx={14} fill={P.card} stroke={P.cyan} strokeWidth={4} />
            <GlyphRow y={-16} w={120} seed={9} mode={1} />
            <GlyphRow y={6} w={90} seed={11} mode={1} />
          </g>
        </g>
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 320}}>
          <div style={{color: P.cyan, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 26}}>
            冷知识 · 第 {data.episode} 期
          </div>
          <div style={{color: C.text, fontSize: 88, fontWeight: 900, fontFamily: FONT_HEAD, letterSpacing: 6}}>
            {data.title}
          </div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* 标题卡小部件：眼罩 */
const EllipseHint = () => (
  <g transform="rotate(-10)">
    <path d="M -70 0 Q 0 -34 70 0 Q 0 34 -70 0 Z" fill={P.violet} stroke={P.violetD} strokeWidth={5} />
    <line x1={-70} y1={0} x2={-104} y2={-14} stroke={P.violetD} strokeWidth={5} strokeLinecap="round" />
    <line x1={70} y1={0} x2={104} y2={-14} stroke={P.violetD} strokeWidth={5} strokeLinecap="round" />
    <line x1={-34} y1={-4} x2={34} y2={-4} stroke={P.ink} strokeWidth={5} strokeLinecap="round" />
  </g>
);

/* ============ S0 卧室 + "画面高清 / 读字断电" 对照 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b0 = beat(0), b1 = beat(1);

  const glitch = f > b0 + 65 ? 1 : 0;
  const rK = clamp01((f - b1) / 22);
  const cardA = Math.max(0.001, spring({frame: f - b1 - 8, fps, config: {damping: 12, stiffness: 140}}));
  const cardB = Math.max(0.001, spring({frame: f - b1 - 26, fps, config: {damping: 12, stiffness: 140}}));
  const swapFlash = glitch > 0 && f % 34 < 4 ? 1 : 0;

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={0} name={sec.name} accent={P.cyan} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：卧室 */}
        <ellipse cx={420} cy={872} rx={340} ry={30} fill={P.bgDeep} />
        <rect x={88} y={160} width={210} height={250} rx={14} fill={P.bgDeep} stroke={P.card} strokeWidth={6} />
        <MoonIcon x={160} y={240} s={0.5} hole={P.bgDeep} />
        {[0, 1, 2, 3].map((i) => (
          <circle key={i} cx={120 + rand(i, 61) * 150} cy={300 + rand(i, 62) * 80} r={2.6} fill="#fff" opacity={0.7} />
        ))}
        <line x1={193} y1={160} x2={193} y2={410} stroke={P.card} strokeWidth={5} />
        <Bed x={420} y={782} s={1.4} />
        <Sleeper x={288} y={716} s={1.05} z={1} />
        {/* 举着看的手机 */}
        <g transform={`translate(${492},${560 - Math.sin(f / 20) * 6})`}>
          <DreamPhone x={0} y={0} s={0.92} rot={-32} seed={5} glitch={glitch} />
        </g>
        {/* 手臂 */}
        <path d="M 360 730 Q 420 660 470 606" stroke={P.skin} strokeWidth={26} strokeLinecap="round" fill="none" />
        <path d="M 360 730 Q 420 660 470 606" stroke={P.skinD} strokeWidth={26} strokeLinecap="round" fill="none"
          strokeDasharray="4 30" opacity={0.5} />
        {swapFlash && <Ripple x={500} y={545} startFrame={f - (f % 34)} color={P.cyan} />}
        <text x={420} y={510} textAnchor="middle" fill={P.cyan} fontSize={28} fontFamily={FONT_HEAD} fontWeight={700}
          opacity={glitch}>再看一眼，内容就换</text>
        <line x1={858} y1={180} x2={858} y2={850} stroke={P.card} strokeWidth={4} strokeDasharray="4 20" opacity={0.6} />
        {/* 右：对照卡 */}
        <g transform={`translate(${-(1 - cardA) * 70},0)`} opacity={cardA}>
          <rect x={960} y={210} width={820} height={230} rx={24} fill={P.card} />
          <rect x={960} y={210} width={16} height={230} rx={8} fill={P.mint} />
          <Castle x={1100} y={330} s={0.78} />
          <text x={1280} y={300} fill={P.cream} fontSize={38} fontFamily={FONT_HEAD} fontWeight={800}>造景：皮层自嗨</text>
          <text x={1280} y={352} fill={C.muted} fontSize={30} fontFamily={FONT_HEAD}>城堡、脸、大海 · 高清成像 ✓</text>
          <rect x={1280} y={378} width={190} height={8} rx={4} fill={P.mint} />
        </g>
        <g transform={`translate(${-(1 - cardB) * 70},0)`} opacity={cardB}>
          <rect x={960} y={480} width={820} height={230} rx={24} fill={P.card} />
          <rect x={960} y={480} width={16} height={230} rx={8} fill={P.coral} />
          <g transform="translate(1100,592)">
            <rect x={-92} y={-64} width={184} height={128} rx={14} fill={P.screen} stroke={P.coralD} strokeWidth={5} />
            <GlyphRow y={-40} w={120} seed={4} mode={1} />
            <GlyphRow y={-10} w={140} seed={8} mode={1} />
            <GlyphRow y={20} w={100} seed={13} mode={1} />
            <line x1={-70} y1={-78} x2={70} y2={66} stroke={P.coral} strokeWidth={8} strokeLinecap="round" opacity={0.9} />
          </g>
          <text x={1280} y={570} fill={P.cream} fontSize={38} fontFamily={FONT_HEAD} fontWeight={800}>读字：多部门协作</text>
          <text x={1280} y={622} fill={C.muted} fontSize={30} fontFamily={FONT_HEAD}>视觉＋语言＋注意 · 下班断电 ✕</text>
          <rect x={1280} y={648} width={190} height={8} rx={4} fill={P.coral} />
        </g>
        <g opacity={rK}>
          <text x={1370} y={790} textAnchor="middle" fill={P.amber} fontSize={30} fontFamily={FONT_HEAD} fontWeight={700}>
            梦里那条短信只是"该有字"的占位符</text>
        </g>
      </svg>
      <Chip text="梦里看手机 · 糊成一团" x={100} y={140} at={b0 + 20} accent={P.cyan} />
      <Stamp text="阅读断电" x={1560} y={130} at={b1 + 44} color={P.coralD} size={52} rot={4} />
      <Chip text="画面高清 字却读不通" x={100} y={820} at={b1 + 90} accent={P.mint} />
    </FadeIn>
  );
};

/* ============ S1 阅读流水线：四站接力 → 掉链子 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b0 = beat(0), b1 = beat(1), b2 = beat(2);

  const stations = [
    {x: 330, icon: 0, name: '视觉词形区', accent: P.cyan},
    {x: 740, icon: 1, name: '语言区解码', accent: P.violet},
    {x: 1150, icon: 2, name: '前额叶整合', accent: P.amber},
    {x: 1560, icon: 3, name: '工作记忆', accent: P.mint},
  ];
  const t0 = b0 + 30, t1 = b1 + 150;              // 字块行进区间
  const tk = clamp01((f - t0) / (t1 - t0));
  const tokenX = 220 + tk * (1690 - 220);
  const beltMove = (f * 4) % 56;
  const brokeK = clamp01((f - b2 - 20) / 18);
  const doneK = clamp01((f - t1 - 8) / 16) * (1 - brokeK);

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={1} name={sec.name} accent={P.violet} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <text x={960} y={188} textAnchor="middle" fill={P.cream} fontSize={34} fontFamily={FONT_HEAD} fontWeight={800}>
          清醒时读一行字，要过四道岗</text>
        {/* 传送带 */}
        <rect x={140} y={566} width={1640} height={20} rx={10} fill={P.cardD} />
        {Array.from({length: 30}).map((_, i) => (
          <rect key={i} x={150 + i * 56 + beltMove} y={572} width={22} height={8} rx={4} fill={P.card} opacity={0.9} />
        ))}
        {stations.map((st, i) => {
          const on = tk > 0 && tokenX > st.x - 90 ? 1 : 0;
          const broke = brokeK * (i === 2 ? 1 : 0);
          return (
            <g key={i}>
              <line x1={st.x} y1={480} x2={st.x} y2={566} stroke={P.card} strokeWidth={8} strokeDasharray="12 9" />
              <Station x={st.x} y={420} icon={st.icon} name={st.name} accent={st.accent} on={on}
                broke={broke} />
              {on > 0 && brokeK < 0.5 && (
                <circle cx={st.x} cy={566} r={12} fill={st.accent} opacity={0.8} />
              )}
            </g>
          );
        })}
        {/* 注意力钉：跟随字块的锥形光 */}
        {tk > 0 && tk < 1 && (
          <g opacity={0.9}>
            <path d={`M ${tokenX - 70} 300 L ${tokenX} 566 L ${tokenX + 70} 300 Z`} fill={P.amber} opacity={0.13} />
            <g transform={`translate(${tokenX},268)`}>
              <rect x={-44} y={-30} width={88} height={40} rx={12} fill={P.amber} stroke={P.ink} strokeWidth={4} />
              <text x={0} y={-1} textAnchor="middle" fontSize={26} fill={P.ink} fontFamily={FONT_HEAD} fontWeight={800}>注意</text>
              <path d="M 0 10 L 0 30" stroke={P.amber} strokeWidth={6} />
              <circle cy={36} r={7} fill={P.amber} />
            </g>
          </g>
        )}
        {/* 字块 token */}
        {tk > 0 && tk < 1 && (
          <g transform={`translate(${tokenX},520)`}>
            <circle r={44} fill={P.cream} stroke={P.ink} strokeWidth={6} />
            <text y={15} textAnchor="middle" fontSize={46} fill={P.ink} fontFamily={FONT_HEAD} fontWeight={900}>字</text>
          </g>
        )}
        {/* 到达终点：读懂 */}
        {doneK > 0 && (
          <g transform={`translate(1760,430) scale(${doneK})`} opacity={doneK}>
            <circle r={52} fill={P.mint} stroke={P.ink} strokeWidth={6} />
            <path d="M -22 2 L -6 20 L 26 -16" stroke={P.ink} strokeWidth={9} fill="none" strokeLinecap="round" strokeLinejoin="round" />
            <text y={86} textAnchor="middle" fontSize={28} fill={P.mint} fontFamily={FONT_HEAD} fontWeight={800}>读懂了</text>
          </g>
        )}
        {/* 掉链子：一断全崩 */}
        <g opacity={brokeK}>
          <rect x={300} y={700} width={1320} height={120} rx={20} fill={P.bgDeep} stroke={P.coralD} strokeWidth={4} />
          <text x={960} y={748} textAnchor="middle" fill={P.cream} fontSize={32} fontFamily={FONT_HEAD} fontWeight={800}>
            文字才几千年历史 · 这条链是临时拼装的</text>
          <text x={960} y={794} textAnchor="middle" fill={P.coral} fontSize={29} fontFamily={FONT_HEAD}>
            一个环节掉链子，整句话就崩了</text>
        </g>
        {brokeK > 0.4 && <Ripple x={1150} y={386} startFrame={b2 + 20} color={P.coral} />}
      </svg>
      <Chip text="看景是自带硬件" x={120} y={140} at={b0 + 24} accent={P.mint} />
      <Chip text="读字是后期拼装" x={120} y={230} at={b1 + 30} accent={P.violet} />
      <Stamp text="嫩得不能再嫩的新功能" x={1330} y={135} at={b2 + 46} color={P.amber} size={44} rot={-3} />
    </FadeIn>
  );
};

/* ============ S2 REM 脑内明暗图 + 去甲肾上腺素见底 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b0 = beat(0), b1 = beat(1), b2 = beat(2);

  // 视觉皮层/杏仁核一直亮；前额叶/语言区始终"半熄灯"
  const dimOn = {vc: 1, amy: 1, pfc: 0, tpj: 0};
  const neK = clamp01((f - b1 - 10) / 90);            // 去甲肾上腺素下降
  const resK = clamp01((f - b2) / 20);
  const labelK = clamp01((f - b0 - 20) / 20);
  const label2K = clamp01((f - b1 - 5) / 20);

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={2} name={sec.name} accent={P.amber} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <text x={480} y={150} textAnchor="middle" fill={P.cream} fontSize={34} fontFamily={FONT_HEAD} fontWeight={800}>
          REM 期的大脑 · 明暗不均</text>
        <BrainMap x={470} y={480} s={2.5} lit={dimOn} />
        {/* 亮区标注 */}
        <g opacity={labelK} fontFamily={FONT_HEAD}>
          <line x1={290} y1={405} x2={170} y2={330} stroke={P.mint} strokeWidth={4} strokeDasharray="8 7" />
          <text x={80} y={318} fill={P.mint} fontSize={29} fontWeight={800}>视觉皮层</text>
          <text x={80} y={354} fill={C.muted} fontSize={25}>照常甚至兴奋地放电</text>
          <line x1={425} y1={636} x2={330} y2={726} stroke={P.coral} strokeWidth={4} strokeDasharray="8 7" />
          <text x={80} y={740} fill={P.coral} fontSize={29} fontWeight={800}>杏仁核</text>
          <text x={80} y={776} fill={C.muted} fontSize={25}>情绪火力全开</text>
        </g>
        {/* 暗区标注 */}
        <g opacity={label2K} fontFamily={FONT_HEAD}>
          <line x1={630} y1={325} x2={760} y2={250} stroke={P.grey} strokeWidth={4} strokeDasharray="8 7" />
          <text x={770} y={240} fill={P.grey} fontSize={29} fontWeight={800}>背外侧前额叶</text>
          <text x={770} y={276} fill={C.muted} fontSize={25}>工作记忆·逻辑 半熄灯</text>
          <line x1={520} y1={465} x2={760} y2={430} stroke={P.grey} strokeWidth={4} strokeDasharray="8 7" />
          <text x={770} y={418} fill={P.grey} fontSize={29} fontWeight={800}>颞顶语言区</text>
          <text x={770} y={454} fill={C.muted} fontSize={25}>解码部门歇了一半</text>
        </g>
        {/* 右：去甲肾上腺素电量 */}
        <g opacity={label2K}>
          <rect x={1080} y={200} width={720} height={180} rx={22} fill={P.card} />
          <text x={1120} y={252} fill={P.cream} fontSize={30} fontFamily={FONT_HEAD} fontWeight={800}>去甲肾上腺素 · 专注力燃料</text>
          <rect x={1120} y={286} width={560} height={34} rx={17} fill={P.bgDeep} />
          <rect x={1120} y={286} width={560 * (1 - 0.86 * neK)} height={34} rx={17} fill={P.amber} />
          <rect x={1680} y={296} width={20} height={14} rx={6} fill={P.bgDeep} />
          <text x={1120} y={356} fill={P.coral} fontSize={26} fontFamily={FONT_HEAD} fontWeight={700}
            opacity={neK}>REM 期跌到全天最低谷</text>
        </g>
        {/* 右下：结果对照 */}
        <g opacity={resK}>
          <rect x={1080} y={430} width={340} height={230} rx={22} fill={P.card} stroke={P.mint} strokeWidth={4} />
          <Castle x={1250} y={532} s={0.55} />
          <text x={1250} y={636} textAnchor="middle" fill={P.mint} fontSize={30} fontFamily={FONT_HEAD} fontWeight={800}>场景 高清</text>
          <rect x={1460} y={430} width={340} height={230} rx={22} fill={P.card} stroke={P.coral} strokeWidth={4} />
          <g transform="translate(1630,528)">
            <rect x={-100} y={-60} width={200} height={110} rx={14} fill={P.screen} stroke={P.card} strokeWidth={4} />
            <GlyphRow y={-34} w={130} seed={6} mode={1} />
            <GlyphRow y={-6} w={110} seed={9} mode={1} />
            <GlyphRow y={22} w={140} seed={12} mode={1} />
          </g>
          <text x={1630} y={636} textAnchor="middle" fill={P.coral} fontSize={30} fontFamily={FONT_HEAD} fontWeight={800}>文字 占位符</text>
        </g>
        <text x={1440} y={742} textAnchor="middle" fill={C.muted} fontSize={27} fontFamily={FONT_HEAD} opacity={resK}>
          生成画面的全开 · 读解画面的歇一半</text>
      </svg>
      <Chip text="不是全黑 · 是明暗不均" x={100} y={830} at={b0 + 20} accent={P.amber} />
      <Chip text="注意力燃料见底" x={1400} y={830} at={b1 + 60} accent={P.coral} />
      <Stamp text="优先级里没有阅读" x={1380} y={130} at={b2 + 40} color={P.violetD} size={44} rot={-4} />
    </FadeIn>
  );
};

/* ============ S3 看两眼就崩：无底稿 + 三连崩小物 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b0 = beat(0), b1 = beat(1);

  const variants = ['明天下午三点见', '明夭下千三典末', '明天下午鱼点贝', '明夭下千三典未'];
  const cyc = Math.floor((f - b0 - 20) / 95);
  const phase = (f - b0 - 20) % 95;                  // 0-70 睁眼看字 70-95 移开(闭眼)
  const eyeOpen = phase < 70 ? Math.min(1, phase / 8) : Math.max(0.06, 1 - (phase - 70) / 8);
  const vi = Math.max(0, cyc) % variants.length;
  const ghost = phase < 70 ? 0.35 + 0.3 * Math.sin(f / 3) : 0;
  const rowK = clamp01((f - b1) / 20);
  const clockTxt = ['18:37', '18:93', '81:37'][Math.floor(f / 40) % 3];
  const signTxt = ['北门路', '北丁路', '厶门路'][Math.floor((f + 13) / 46) % 3];

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={3} name={sec.name} accent={P.mint} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：醒着——印在纸上 */}
        <rect x={110} y={190} width={470} height={330} rx={22} fill={P.card} />
        <text x={345} y={244} textAnchor="middle" fill={P.cream} fontSize={31} fontFamily={FONT_HEAD} fontWeight={800}>醒着 · 有底稿</text>
        <g transform="translate(345,380)">
          <rect x={-160} y={-96} width={320} height={170} rx={10} fill={P.paper} stroke={P.card} strokeWidth={5}
            transform="rotate(-2)" />
          <text x={0} y={-40} textAnchor="middle" fontSize={33} fill={P.ink} fontFamily={FONT_HEAD} fontWeight={700}
            transform="rotate(-2)">明天下午三点见</text>
          <line x1={-120} y1={0} x2={120} y2={-8} stroke={P.card} strokeWidth={4} transform="rotate(-2)" />
          <line x1={-120} y1={30} x2={100} y2={22} stroke={P.card} strokeWidth={4} transform="rotate(-2)" />
          <g transform="translate(118,-108)">
            <circle r={30} fill={P.mint} stroke={P.ink} strokeWidth={5} />
            <path d="M -13 0 L -3 12 L 15 -9" stroke={P.ink} strokeWidth={6} fill="none" strokeLinecap="round" strokeLinejoin="round" />
          </g>
        </g>
        <text x={345} y={498} textAnchor="middle" fill={C.muted} fontSize={26} fontFamily={FONT_HEAD}>印在纸上 · 看几遍都一样</text>
        {/* 右：梦里——皮层现场编 */}
        <rect x={660} y={190} width={1150} height={330} rx={22} fill={P.card} stroke={P.coralD} strokeWidth={4} />
        <text x={1235} y={244} textAnchor="middle" fill={P.cream} fontSize={31} fontFamily={FONT_HEAD} fontWeight={800}>梦里 · 没有底稿可对照</text>
        <Eye x={830} y={392} s={1.15} open={eyeOpen} />
        <text x={830} y={478} textAnchor="middle" fill={C.muted} fontSize={25} fontFamily={FONT_HEAD}>
          {eyeOpen > 0.5 ? `第 ${Math.min(3, vi + 1)} 眼` : '移开视线…'}</text>
        {/* 梦手机屏幕：每次睁眼换一版 */}
        <g transform="translate(1080,380)">
          <rect x={-170} y={-82} width={340} height={150} rx={16} fill={P.screen} stroke={P.card} strokeWidth={5} />
          {eyeOpen > 0.4 ? (
            <g>
              <text x={3} y={18} textAnchor="middle" fontSize={42} fill={P.cyan} fontFamily={FONT_HEAD} fontWeight={800}
                opacity={ghost} transform="translate(3,-2)">{variants[vi]}</text>
              <text x={0} y={16} textAnchor="middle" fontSize={42} fill={P.cyan} fontFamily={FONT_HEAD} fontWeight={800}>
                {variants[vi]}</text>
            </g>
          ) : (
            <text x={0} y={14} textAnchor="middle" fontSize={36} fill={P.grey} fontFamily={FONT_HEAD}>· · ·</text>
          )}
        </g>
        {/* 版本对不上的标记 */}
        <g transform="translate(1460,330)" opacity={eyeOpen > 0.4 ? 1 : 0.15}>
          <rect x={-160} y={-46} width={330} height={170} rx={16} fill={P.bgDeep} />
          <text x={0} y={-12} textAnchor="middle" fontSize={28} fill={P.grey} fontFamily={FONT_HEAD}>上一版</text>
          <text x={0} y={36} textAnchor="middle" fontSize={34} fill={P.grey} fontFamily={FONT_HEAD}
            textDecoration="line-through">{variants[(vi + variants.length - 1) % variants.length]}</text>
          <text x={0} y={94} textAnchor="middle" fontSize={28} fill={P.coral} fontFamily={FONT_HEAD} fontWeight={700}>对不上 ✕</text>
        </g>
        <line x1={1260} y1={380} x2={1290} y2={380} stroke={P.coral} strokeWidth={5} strokeDasharray="8 7" />
        {/* 底排：同款"看两眼就崩" */}
        <g opacity={rowK}>
          {[
            {cx: 460, name: '数字时钟', el: <DigiClock x={0} y={0} txt={clockTxt} />},
            {cx: 960, name: '路牌', el: <RoadSign x={0} y={-14} txt={signTxt} s={0.9} />},
            {cx: 1460, name: '推送通知', el: (
              <g>
                <rect x={-150} y={-56} width={300} height={112} rx={18} fill={P.card} stroke={P.cardD} strokeWidth={4} />
                <circle cx={-108} cy={0} r={20} fill={P.coral} />
                <GlyphRow y={-30} w={170} seed={Math.floor(f / 38) + 2} mode={1} />
                <GlyphRow y={4} w={140} seed={Math.floor(f / 38) + 5} mode={1} />
              </g>
            )},
          ].map((it, i) => (
            <g key={i} transform={`translate(${it.cx},726)`}>
              <rect x={-210} y={-96} width={420} height={200} rx={20} fill={P.card}
                stroke={P.coralD} strokeWidth={3} strokeDasharray="10 8" />
              {it.el}
              <text x={0} y={-66} textAnchor="middle" fill={P.cream} fontSize={26} fontFamily={FONT_HEAD} fontWeight={700}>
                {it.name} · 也在崩</text>
            </g>
          ))}
        </g>
      </svg>
      <Chip text="字是皮层现场编的" x={700} y={100} at={b0 + 40} accent={P.cyan} />
      <Stamp text="看两眼就崩" x={1560} y={140} at={b0 + 120} color={P.coralD} size={50} rot={4} />
      <Chip text="验梦法：看一眼 · 移开 · 再看" x={100} y={100} at={b1 + 40} accent={P.mint} />
    </FadeIn>
  );
};

/* ============ S4 稳定性测试三件套 + 梦日记 + 默认应用 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b0 = beat(0), b1 = beat(1), b2 = beat(2);

  const tests = [
    {at: b0 + 15, name: '数手指', sub: '梦里多一根', el: <Hand x={0} y={8} s={0.72} extra />},
    {at: b0 + 29, name: '照镜子', sub: '脸有点怪', el: <Mirror x={0} y={4} s={0.62} />},
    {at: b0 + 43, name: '看门牌', sub: '数字在跳舞', el: <DigiClock x={0} y={0} s={0.9} txt={['402', '4O2', '40Z'][Math.floor(f / 42) % 3]} />},
  ];
  const habitK = clamp01((f - b1) / 22);
  const diaryK = clamp01((f - b1 - 40) / 22);
  const appK = clamp01((f - b2) / 22);

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={4} name={sec.name} accent={P.mint} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 上排：清醒梦玩家的稳定性测试 */}
        {tests.map((t, i) => {
          const g = Math.max(0.001, spring({frame: f - t.at, fps, config: {damping: 12, stiffness: 140}}));
          const cx = 330 + i * 630;
          return (
            <g key={i} transform={`translate(${cx},330) scale(${g})`} opacity={g}>
              <rect x={-240} y={-160} width={480} height={300} rx={22} fill={P.card} />
              <rect x={-240} y={-160} width={480} height={60} rx={22} fill={P.cardD} />
              <text x={0} y={-118} textAnchor="middle" fill={P.cream} fontSize={30} fontFamily={FONT_HEAD} fontWeight={800}>
                测试 {i + 1} · {t.name}</text>
              <g transform="translate(0,20)">{t.el}</g>
              <text x={0} y={122} textAnchor="middle" fill={P.coral} fontSize={27} fontFamily={FONT_HEAD} fontWeight={700}>
                {t.sub} ✕</text>
            </g>
          );
        })}
        {/* 左下：白天习惯 → 带进梦里 */}
        <g opacity={habitK} transform={`translate(${-(1 - habitK) * 60},0)`}>
          <rect x={120} y={560} width={830} height={200} rx={22} fill={P.card} stroke={P.mint} strokeWidth={4} />
          <text x={160} y={622} fill={P.cream} fontSize={31} fontFamily={FONT_HEAD} fontWeight={800}>白天养成的习惯会被带进梦</text>
          <g transform="translate(250,690)">
            <Eye x={0} y={0} s={0.6} />
          </g>
          <text x={330} y={700} fill={C.muted} fontSize={28} fontFamily={FONT_HEAD}>看字 → 移开 → 再看一遍</text>
          <path d="M 700 690 L 800 690" stroke={P.mint} strokeWidth={6} strokeDasharray="12 8" />
          <path d="M 800 690 L 782 678 L 782 702 Z" fill={P.mint} />
          <MoonIcon x={852} y={686} s={0.42} hole={P.card} />
        </g>
        {/* 右下：梦日记 */}
        <g opacity={diaryK} transform={`translate(${(1 - diaryK) * 60},0)`}>
          <rect x={1000} y={560} width={800} height={200} rx={22} fill={P.card} />
          <g transform="translate(1130,662) rotate(-4)">
            <rect x={-92} y={-72} width={184} height={140} rx={10} fill={P.paper} stroke={P.card} strokeWidth={5} />
            <line x1={0} y1={-72} x2={0} y2={68} stroke={P.card} strokeWidth={4} />
            <GlyphRow y={-52} w={64} seed={3} mode={0} color={P.grey} />
            <GlyphRow y={-28} w={64} seed={7} mode={1} color={P.grey} />
            <GlyphRow y={-4} w={64} seed={9} mode={1} color={P.grey} />
            <text x={46} y={50} textAnchor="middle" fontSize={22} fill={P.coralD} fontFamily={FONT_HEAD}>？</text>
          </g>
          <text x={1260} y={630} fill={P.cream} fontSize={31} fontFamily={FONT_HEAD} fontWeight={800}>翻翻梦日记</text>
          <text x={1260} y={678} fill={C.muted} fontSize={28} fontFamily={FONT_HEAD}>找不到一句完整读到的话</text>
          <text x={1260} y={722} fill={P.coral} fontSize={26} fontFamily={FONT_HEAD}>"知道写着什么 就是想不起"</text>
        </g>
        {/* 收尾横幅：默认应用里没有阅读 */}
        <g opacity={appK} transform={`translate(0,${(1 - appK) * 40})`}>
          <rect x={330} y={810} width={1260} height={110} rx={24} fill={P.bgDeep} stroke={P.cyan} strokeWidth={4} />
          {[0, 1, 2].map((i) => (
            <g key={i} transform={`translate(${420 + i * 86},864)`}>
              <rect x={-30} y={-30} width={60} height={60} rx={16}
                fill={[P.mint, P.brain, P.cyan][i]} stroke={P.ink} strokeWidth={4} />
              <text x={0} y={11} textAnchor="middle" fontSize={30} fill={P.ink} fontFamily={FONT_HEAD} fontWeight={800}>
                {['山', '脸', '海'][i]}</text>
            </g>
          ))}
          <g transform="translate(678,864)">
            <rect x={-30} y={-30} width={60} height={60} rx={16} fill="none" stroke={P.grey}
              strokeWidth={4} strokeDasharray="8 7" />
            <text x={0} y={11} textAnchor="middle" fontSize={26} fill={P.grey} fontFamily={FONT_HEAD}>阅</text>
          </g>
          <text x={780} y={855} fill={P.cream} fontSize={33} fontFamily={FONT_HEAD} fontWeight={800}>
            梦只给了原生配置</text>
          <text x={780} y={897} fill={P.cyan} fontSize={27} fontFamily={FONT_HEAD}>
            文字从来不在"默认应用"里</text>
        </g>
      </svg>
      <Chip text="钟表·镜子·门牌 同款失真" x={700} y={86} at={b0 + 60} accent={P.coral} />
      <Chip text="两三周习惯带进梦" x={120} y={478} at={b1 + 30} accent={P.mint} />
      <Stamp text="顺手验个梦" x={1540} y={64} at={b2 + 30} color={P.mintD} size={48} rot={5} />
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
        <ellipse cx={960} cy={1015} rx={1100} ry={120} fill={P.bg} />
        <Bed x={880} y={930} s={1.35} />
        <Sleeper x={754} y={870} s={0.95} z={1} />
        <MoonIcon x={620} y={630} s={0.8} hole={P.bgDeep} />
        <g transform={`translate(1420,880) rotate(-8) scale(${pop})`}>
          <DreamPhone x={0} y={0} s={0.9} seed={21} />
          <rect x={-120} y={-230} width={240} height={56} rx={16} fill={P.card} />
          <text x={0} y={-193} textAnchor="middle" fontSize={26} fill={P.cyan} fontFamily={FONT_HEAD}>梦里: 看不清</text>
        </g>
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -300}}>
        <div style={{fontSize: 58, fontWeight: 800, color: C.text}}>梦里看山是山，看字不是字</div>
        <div style={{fontSize: 38, color: C.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E225 = ({data = tl}: {data?: any}) => {
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
