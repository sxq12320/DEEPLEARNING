import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01, rand} from '../lib/style';
import {Backdrop, Stamp, SceneTag, Chip, Subtitles, ProgressBar, FadeIn, Sun} from '../lib/atoms';
import tl from '../timeline_230.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：银发深夜蓝 × 月光黄 × 深睡青 × 警示珊瑚（纯色平涂） ============ */
const P = {
  bg: '#14213a',
  bgDeep: '#0b1526',
  bgLite: '#1c2f4e',
  card: '#223650',
  cardD: '#182a42',
  cream: '#f6efdd',
  ink: '#101b30',
  moon: '#ffd873',
  moonD: '#e0ab40',
  cyan: '#63d6e8',
  cyanD: '#35a9c0',
  sky: '#6fb7ff',
  skyD: '#3d82cc',
  violet: '#a98ef0',
  violetD: '#7c5fc9',
  coral: '#ff7d6e',
  coralD: '#cf5244',
  mint: '#86e2b0',
  mintD: '#4dab7d',
  amber: '#ffc94d',
  grey: '#93a0c2',
  skin: '#f2c79b',
  skinD: '#d99f6b',
  hair: '#e2e9f5',
  hairD: '#a9b4cc',
  brain: '#f2a9bd',
  brainD: '#d17a94',
  pillow: '#eef1fb',
  mattress: '#d9dff2',
  blanket: '#4f7ba6',
  blanketD: '#38597e',
  wood: '#8f6240',
  woodD: '#6a4729',
  screen: '#12203c',
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

/* ---------- 床 ---------- */
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

/* ---------- 老人头：银发 + 皱纹；awake=睁眼失眠 ---------- */
const Elder = ({x, y, s = 1, op = 1, z = 0, awake = 0, rot = 0}: {
  x: number; y: number; s?: number; op?: number; z?: number; awake?: number; rot?: number;
}) => {
  const f = useCurrentFrame();
  return (
    <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`} opacity={op}>
      <circle r={46} fill={P.skin} stroke={P.skinD} strokeWidth={5} />
      {/* 银发盖 */}
      <path d="M -46 -6 A 46 46 0 0 1 46 -6 L 46 -14 A 46 46 0 0 0 -46 -14 Z" fill={P.hair} />
      <path d="M -46 -6 A 46 46 0 0 1 46 -6 L 42 -6 A 42 42 0 0 0 -42 -6 Z" fill={P.hairD} opacity={0.6} />
      <circle cx={-44} cy={-14} r={10} fill={P.hair} />
      <circle cx={44} cy={-14} r={10} fill={P.hair} />
      {/* 抬头纹 */}
      <path d="M -18 -26 Q 0 -30 18 -26" stroke={P.skinD} strokeWidth={3.5} fill="none" strokeLinecap="round" />
      {awake > 0.5 ? (
        <g>
          <circle cx={-17} cy={8} r={5.5} fill={P.ink} />
          <circle cx={19} cy={8} r={5.5} fill={P.ink} />
          <path d="M -24 20 Q -17 24 -10 20" stroke={P.coralD} strokeWidth={3.5} fill="none" strokeLinecap="round" />
          <path d="M 12 20 Q 19 24 26 20" stroke={P.coralD} strokeWidth={3.5} fill="none" strokeLinecap="round" />
          <path d="M -4 34 Q 4 31 12 34" stroke={P.ink} strokeWidth={4} fill="none" strokeLinecap="round" />
        </g>
      ) : (
        <g stroke={P.ink} strokeWidth={4} fill="none" strokeLinecap="round">
          <path d="M -26 8 Q -17 16 -8 8" />
          <path d="M 10 8 Q 19 16 28 8" />
          <path d="M -2 30 Q 6 35 14 30" />
        </g>
      )}
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

/* ---------- 数字时钟 ---------- */
const DigiClock = ({x, y, s = 1, txt = '04:37', op = 1}: {x: number; y: number; s?: number; txt?: string; op?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
    <rect x={-92} y={-44} width={184} height={88} rx={14} fill={P.ink} stroke={P.card} strokeWidth={5} />
    <text x={0} y={16} textAnchor="middle" fontSize={46} fill={P.cyan} fontFamily={FONT_HEAD} fontWeight={800}
      letterSpacing={4}>{txt}</text>
  </g>
);

/* ---------- 大脑轮廓 ---------- */
const BRAIN_D = 'M -96 20 C -122 -22 -104 -72 -66 -86 C -52 -118 -4 -126 26 -104 C 66 -116 104 -90 106 -48 ' +
  'C 134 -28 130 16 104 34 C 110 70 74 94 42 80 C 16 100 -36 94 -52 66 C -86 66 -106 46 -96 20 Z';
const BrainIcon = ({x, y, s = 1, op = 1, dim = 0}: {x: number; y: number; s?: number; op?: number; dim?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
    <path d={BRAIN_D} fill={P.brain} stroke={P.brainD} strokeWidth={6} strokeLinejoin="round"
      strokeDasharray={dim > 0.5 ? '14 9' : undefined} opacity={dim > 0.5 ? 0.55 : 1} />
    <g stroke={P.brainD} strokeWidth={5} fill="none" strokeLinecap="round" opacity={dim > 0.5 ? 0.5 : 1}>
      <path d="M -78 -42 Q -50 -68 -22 -46" />
      <path d="M -64 18 Q -32 -4 -4 20" />
      <path d="M 8 -74 Q 44 -92 70 -64" />
      <path d="M 30 32 Q 60 12 88 36" />
    </g>
    {dim > 0.5 && (
      <g stroke={P.coral} strokeWidth={7} strokeLinecap="round">
        <line x1={-30} y1={-20} x2={-4} y2={6} />
        <line x1={-4} y1={-20} x2={-30} y2={6} />
      </g>
    )}
  </g>
);

/* ---------- 脑电波路径 ---------- */
const wave = (x0: number, y0: number, w: number, amp: number, period: number, jit = 0) => {
  const pts: string[] = [];
  const n = Math.max(8, Math.floor(w / 9));
  for (let i = 0; i <= n; i++) {
    const x = x0 + (w * i) / n;
    const y = y0 + Math.sin(((x - x0) / period) * Math.PI * 2) * amp + (jit ? Math.sin(i * 3.7) * jit : 0);
    pts.push(`${i === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`);
  }
  return pts.join(' ');
};

/* ---------- 电视 ---------- */
const TV = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <line x1={-20} y1={-108} x2={-52} y2={-152} stroke={P.grey} strokeWidth={6} strokeLinecap="round" />
    <line x1={20} y1={-108} x2={52} y2={-152} stroke={P.grey} strokeWidth={6} strokeLinecap="round" />
    <rect x={-105} y={-108} width={210} height={140} rx={16} fill={P.cardD} stroke={P.card} strokeWidth={6} />
    <rect x={-86} y={-90} width={172} height={104} rx={10} fill={P.screen} />
    <rect x={-70} y={-72} width={120} height={12} rx={6} fill={P.grey} opacity={0.5} />
    <rect x={-70} y={-46} width={90} height={12} rx={6} fill={P.grey} opacity={0.35} />
    <rect x={-70} y={-20} width={110} height={12} rx={6} fill={P.grey} opacity={0.25} />
    <rect x={-40} y={32} width={80} height={14} rx={6} fill={P.cardD} />
  </g>
);

/* ---------- 沙发 ---------- */
const Sofa = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-120} y={-118} width={240} height={70} rx={18} fill={P.blanketD} />
    <rect x={-160} y={-96} width={46} height={128} rx={16} fill={P.blanketD} />
    <rect x={114} y={-96} width={46} height={128} rx={16} fill={P.blanketD} />
    <rect x={-150} y={-56} width={300} height={88} rx={20} fill={P.blanket} stroke={P.blanketD} strokeWidth={5} />
  </g>
);

/* ---------- 走路小人 ---------- */
const Walker = ({x, y, s = 1}: {x: number; y: number; s?: number}) => {
  const f = useCurrentFrame();
  const sw = Math.sin(f / 6) * 12;
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      <line x1={-52} y1={-34} x2={-88} y2={-34} stroke={P.mintD} strokeWidth={7} strokeLinecap="round" opacity={0.6} />
      <line x1={-52} y1={-6} x2={-96} y2={-6} stroke={P.mintD} strokeWidth={7} strokeLinecap="round" opacity={0.4} />
      <circle cx={0} cy={-72} r={24} fill={P.skin} stroke={P.skinD} strokeWidth={4} />
      <path d="M -24 -78 A 24 24 0 0 1 24 -78 L 24 -72 A 24 24 0 0 0 -24 -72 Z" fill={P.hair} />
      <rect x={-20} y={-46} width={40} height={66} rx={18} fill={P.mint} stroke={P.mintD} strokeWidth={4} />
      <line x1={-8} y1={18} x2={-26 - sw * 0.4} y2={74} stroke={P.mintD} strokeWidth={11} strokeLinecap="round" />
      <line x1={8} y1={18} x2={26 + sw * 0.4} y2={74} stroke={P.mintD} strokeWidth={11} strokeLinecap="round" />
      <line x1={-14} y1={-30} x2={-44} y2={4 + sw * 0.3} stroke={P.mint} strokeWidth={10} strokeLinecap="round" />
      <line x1={14} y1={-30} x2={44} y2={0 - sw * 0.3} stroke={P.mint} strokeWidth={10} strokeLinecap="round" />
    </g>
  );
};

/* ---------- 午睡计时器 ---------- */
const NapTimer = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <circle r={62} fill={P.bgLite} stroke={P.amber} strokeWidth={8} />
    {Array.from({length: 12}).map((_, i) => {
      const a = (i * 30 * Math.PI) / 180;
      return <circle key={i} cx={Math.sin(a) * 48} cy={-Math.cos(a) * 48} r={i % 3 === 0 ? 4 : 2.5} fill={P.grey} />;
    })}
    <line x1={0} y1={0} x2={0} y2={-38} stroke={P.cream} strokeWidth={7} strokeLinecap="round" />
    <line x1={0} y1={0} x2={30} y2={16} stroke={P.coral} strokeWidth={7} strokeLinecap="round" />
    <circle r={7} fill={P.cream} />
    <rect x={-18} y={-92} width={36} height={18} rx={7} fill={P.amber} />
    <text x={0} y={102} textAnchor="middle" fontSize={34} fill={P.amber} fontFamily={FONT_HEAD} fontWeight={800}>30分内</text>
  </g>
);

/* ---------- 卧室环境小图标 ---------- */
const BedroomIcon = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-92} y={-6} width={150} height={44} rx={12} fill={P.mattress} stroke={P.card} strokeWidth={4} />
    <rect x={-86} y={-26} width={48} height={24} rx={11} fill={P.pillow} stroke={P.card} strokeWidth={4} />
    <rect x={-30} y={-28} width={88} height={46} rx={10} fill={P.blanket} stroke={P.blanketD} strokeWidth={4} />
    <rect x={74} y={-72} width={20} height={76} rx={10} fill={P.bgLite} stroke={P.cyan} strokeWidth={4} />
    <circle cx={84} cy={14} r={15} fill={P.cyan} />
    <rect x={80} y={-48} width={8} height={52} rx={4} fill={P.cyan} />
    <g transform="translate(-60,-58) scale(0.36)">
      <circle r={56} fill={P.moon} />
      <circle cx={26} cy={-16} r={46} fill={P.card} />
    </g>
  </g>
);

/* ============ 标题卡 ============ */
const TitleCard = ({data}: {data: any}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 13, mass: 0.7}});
  const o = interpolate(f, [0, 8], [0, 1]);
  const p0 = Math.max(0.001, spring({frame: f - 4, fps, config: {damping: 10, stiffness: 150}}));
  const p1 = Math.max(0.001, spring({frame: f - 14, fps, config: {damping: 10, stiffness: 150}}));
  return (
    <AbsoluteFill>
      <Backdrop top={P.bg} bottom={P.bg} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <circle cx={960} cy={338} r={222} fill={P.cream} />
        <circle cx={960} cy={338} r={200} fill={P.bgLite} />
        <g transform={`translate(960,338) scale(${p0})`}>
          <g transform="translate(-960,-338)">
            <MoonIcon x={1080} y={222} s={0.5} hole={P.bgLite} />
            {[0, 1, 2].map((i) => (
              <circle key={i} cx={810 + rand(i, 51) * 240} cy={200 + rand(i, 52) * 70} r={3}
                fill="#fff" opacity={0.8} />
            ))}
            <Bed x={978} y={468} s={0.82} />
            <Elder x={898} y={428} s={0.78} z={1} />
            <DigiClock x={832} y={292} s={0.5} txt="04:30" />
          </g>
        </g>
        <g transform={`translate(700,566) rotate(-7) scale(${p1})`}>
          <rect x={-128} y={-34} width={256} height={64} rx={32} fill={P.card} stroke={P.coral} strokeWidth={4} />
          <text x={0} y={12} textAnchor="middle" fontSize={30} fill={P.coral} fontFamily={FONT_HEAD} fontWeight={800}>四五点就醒</text>
        </g>
        <g transform={`translate(1224,566) rotate(6) scale(${p1})`}>
          <rect x={-148} y={-34} width={296} height={64} rx={32} fill={P.card} stroke={P.cyan} strokeWidth={4} />
          <text x={0} y={12} textAnchor="middle" fontSize={30} fill={P.cyan} fontFamily={FONT_HEAD} fontWeight={800}>躺下又睡不着</text>
        </g>
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 330}}>
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

/* ============ S0 卧室失眠 ×「需求还在·产能缩水」车间条 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b0 = beat(0), b1 = beat(1);

  const capK = interpolate(f, [b1 + 8, b1 + 80], [1, 0.42], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const prodW = 540 * capK;
  const cardK = clamp01((f - 10) / 18);
  const dayK = clamp01((f - b0 - 120) / 20);

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={0} name={sec.name} accent={P.amber} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：凌晨卧室 */}
        <ellipse cx={420} cy={880} rx={340} ry={28} fill={P.bgDeep} />
        <rect x={88} y={160} width={210} height={250} rx={14} fill={P.bgDeep} stroke={P.card} strokeWidth={6} />
        <MoonIcon x={158} y={238} s={0.48} hole={P.bgDeep} />
        {[0, 1, 2].map((i) => (
          <circle key={i} cx={120 + rand(i, 61) * 150} cy={300 + rand(i, 62) * 80} r={2.6} fill="#fff" opacity={0.7} />
        ))}
        <line x1={193} y1={160} x2={193} y2={410} stroke={P.card} strokeWidth={5} />
        <Bed x={420} y={790} s={1.35} />
        <Elder x={292} y={722} s={1.02} awake={1} />
        <text x={292} y={640} textAnchor="middle" fill={P.coral} fontSize={27} fontFamily={FONT_HEAD} fontWeight={700}>
          睁着眼等天亮</text>
        {/* 床头柜 + 闹钟 */}
        <rect x={648} y={726} width={112} height={92} rx={10} fill={P.wood} stroke={P.woodD} strokeWidth={5} />
        <rect x={668} y={760} width={72} height={14} rx={7} fill={P.woodD} />
        <DigiClock x={704} y={692} s={0.58} txt="04:37" />
        <line x1={858} y1={180} x2={858} y2={860} stroke={P.card} strokeWidth={4} strokeDasharray="4 20" opacity={0.6} />
        {/* 右：睡眠车间 订单 vs 产出 */}
        <g opacity={cardK} transform={`translate(${(1 - cardK) * 60},0)`}>
          <rect x={960} y={190} width={820} height={300} rx={24} fill={P.card} />
          <text x={1002} y={248} fill={P.cream} fontSize={32} fontFamily={FONT_HEAD} fontWeight={800}>
            睡眠车间 · 订单没变 产线老了</text>
          <text x={1002} y={332} fill={P.cream} fontSize={28} fontFamily={FONT_HEAD}>需求订单</text>
          <rect x={1130} y={304} width={540} height={44} rx={22} fill={P.bgDeep} />
          <rect x={1130} y={304} width={540} height={44} rx={22} fill={P.amber} />
          <text x={1646} y={336} textAnchor="end" fontSize={26} fill={P.ink} fontFamily={FONT_HEAD} fontWeight={800}>7~8小时</text>
          <text x={1002} y={418} fill={P.cream} fontSize={28} fontFamily={FONT_HEAD}>实际产出</text>
          <rect x={1130} y={390} width={540} height={44} rx={22} fill={P.bgDeep} />
          <rect x={1130} y={390} width={Math.max(44, prodW)} height={44} rx={22} fill={P.cyan} />
          <text x={1145 + Math.max(44, prodW)} y={422} fontSize={26} fill={P.coral} fontFamily={FONT_HEAD}
            fontWeight={800} opacity={clamp01((f - b1 - 70) / 15)}>
            5小时·还碎</text>
          <text x={1002} y={468} fill={C.muted} fontSize={25} fontFamily={FONT_HEAD}>
            需求端没降 · 是产能随年龄缩水</text>
        </g>
        {/* 右下：白天沾沙发就睡 */}
        <g opacity={dayK} transform={`translate(${(1 - dayK) * 60},0)`}>
          <rect x={960} y={530} width={820} height={260} rx={24} fill={P.card} stroke={P.mintD} strokeWidth={4} />
          <TV x={1080} y={700} s={0.9} />
          <Sofa x={1330} y={726} s={0.95} />
          <g transform="rotate(14 1296 628)">
            <Elder x={1296} y={628} s={0.82} z={1} />
          </g>
          <text x={1560} y={612} fill={P.cream} fontSize={29} fontFamily={FONT_HEAD} fontWeight={800}>白天沾沙发就睡</text>
          <text x={1560} y={660} fill={P.mint} fontSize={26} fontFamily={FONT_HEAD}>不是觉够了</text>
          <text x={1560} y={704} fill={P.mint} fontSize={26} fontFamily={FONT_HEAD}>是夜里欠的债</text>
        </g>
      </svg>
      <Chip text="四五点醒 · 躺下睡不着" x={100} y={130} at={b0 + 18} accent={P.coral} />
      <Chip text="需求还在 · 产能缩水" x={110} y={836} at={b1 + 80} accent={P.amber} />
      <Stamp text="只对了一半" x={1520} y={120} at={b0 + 200} color={P.coralD} size={52} rot={4} />
    </FadeIn>
  );
};

/* ============ S1 睡眠深度剖面：20岁 vs 70岁 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);

  const revA = clamp01((f - 16) / 120);
  const revB = clamp01((f - b1 - 10) / 130);
  const barA = Math.max(0.001, spring({frame: f - 130, fps, config: {damping: 13, stiffness: 130}}));
  const barB = Math.max(0.001, spring({frame: f - b1 - 140, fps, config: {damping: 13, stiffness: 130}}));
  const alertK = clamp01((f - b2 - 16) / 18);

  const youngPath = 'M 190 322 L 240 402 L 290 542 L 340 502 L 392 402 L 442 546 L 492 472 L 542 432 ' +
    'L 592 484 L 642 402 L 702 362 L 760 384 L 800 342';
  const oldPath = 'M 1110 332 L 1162 392 L 1212 432 L 1262 406 L 1312 446 L 1362 416 L 1412 442 ' +
    'L 1462 412 L 1512 438 L 1562 408 L 1612 428 L 1662 392 L 1720 352';

  const PropBar = ({x, k, deep, rem, light}: {x: number; k: number; deep: number; rem: number; light: number}) => {
    const total = 660;
    const dw = total * deep, rw = total * rem, lw = total * light;
    const aw = total - dw - rw - lw;
    let cx = 0;
    const segs = [
      {w: dw, c: P.sky}, {w: lw, c: P.mint}, {w: rw, c: P.violet}, {w: aw, c: P.grey},
    ];
    return (
      <g transform={`translate(${x},0) scale(${k},1)`} opacity={k}>
        {segs.map((sg, i) => {
          const el = <rect key={i} x={cx} y={612} width={sg.w} height={54} fill={sg.c} />;
          cx += sg.w;
          return el;
        })}
        <rect x={0} y={612} width={total} height={54} rx={10} fill="none" stroke={P.card} strokeWidth={4} />
      </g>
    );
  };

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={1} name={sec.name} accent={P.sky} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <text x={960} y={152} textAnchor="middle" fill={P.cream} fontSize={34} fontFamily={FONT_HEAD} fontWeight={800}>
          一晚的睡眠深度剖面 · 缩得最狠的是深睡</text>
        {/* 左：20岁 */}
        <rect x={120} y={190} width={760} height={460} rx={24} fill={P.card} />
        <text x={500} y={246} textAnchor="middle" fill={P.cream} fontSize={32} fontFamily={FONT_HEAD} fontWeight={800}>年轻时</text>
        <rect x={190} y={466} width={610} height={104} fill={P.sky} opacity={0.15} />
        <rect x={190} y={466} width={610} height={104} fill="none" stroke={P.sky} strokeWidth={3} strokeDasharray="10 8" opacity={0.7} />
        <g fontFamily={FONT_HEAD} fontSize={24} fill={C.muted}>
          <text x={138} y={326}>清醒</text>
          <text x={138} y={424}>浅睡</text>
          <text x={138} y={534} fill={P.sky}>深睡</text>
        </g>
        <clipPath id="s1a"><rect x={190} y={270} width={610 * revA} height={310} /></clipPath>
        <g clipPath="url(#s1a)">
          <path d={youngPath} fill="none" stroke={P.cyan} strokeWidth={9} strokeLinecap="round" strokeLinejoin="round" />
        </g>
        <PropBar x={170} k={barA} deep={0.2} rem={0.22} light={0.45} />
        <text x={170} y={706} fill={P.sky} fontSize={27} fontFamily={FONT_HEAD} fontWeight={800} opacity={barA}>
          深睡 约20%</text>
        <text x={830} y={706} textAnchor="end" fill={C.muted} fontSize={24} fontFamily={FONT_HEAD} opacity={barA}>
          青=深睡 绿=浅睡 紫=REM</text>
        {/* 右：70岁 */}
        <rect x={1040} y={190} width={760} height={460} rx={24} fill={P.card} />
        <text x={1420} y={246} textAnchor="middle" fill={P.cream} fontSize={32} fontFamily={FONT_HEAD} fontWeight={800}>七十岁后</text>
        <rect x={1110} y={466} width={610} height={104} fill={P.sky} opacity={0.15} />
        <rect x={1110} y={466} width={610} height={104} fill="none" stroke={P.sky} strokeWidth={3} strokeDasharray="10 8" opacity={0.7} />
        <g fontFamily={FONT_HEAD} fontSize={24} fill={C.muted}>
          <text x={1058} y={326}>清醒</text>
          <text x={1058} y={424}>浅睡</text>
          <text x={1058} y={534} fill={P.sky}>深睡</text>
        </g>
        <clipPath id="s1b"><rect x={1110} y={270} width={610 * revB} height={310} /></clipPath>
        <g clipPath="url(#s1b)">
          <path d={oldPath} fill="none" stroke={P.cyan} strokeWidth={9} strokeLinecap="round" strokeLinejoin="round" />
        </g>
        <PropBar x={1090} k={barB} deep={0.05} rem={0.2} light={0.58} />
        <text x={1090} y={706} fill={P.coral} fontSize={27} fontFamily={FONT_HEAD} fontWeight={800} opacity={barB}>
          深睡 只剩个位数%</text>
        {/* 警报：深睡薄到看不见 */}
        <g opacity={alertK}>
          <rect x={1110} y={466} width={610} height={104} fill="none" stroke={P.coral} strokeWidth={5} strokeDasharray="12 9" />
          <line x1={1560} y1={466} x2={1660} y2={400} stroke={P.coral} strokeWidth={4} />
          <text x={1668} y={396} fill={P.coral} fontSize={26} fontFamily={FONT_HEAD} fontWeight={800}>几乎薄到看不见</text>
        </g>
        <text x={960} y={768} textAnchor="middle" fill={P.cream} fontSize={30} fontFamily={FONT_HEAD} fontWeight={700}
          opacity={alertK}>
          不是均匀缩水 · 是先把修复力最强的深睡丢了</text>
      </svg>
      <Chip text="慢波睡眠 = 修复担当" x={110} y={130} at={b1 + 24} accent={P.sky} />
      <Stamp text="结构性失血" x={1540} y={104} at={b2 + 42} color={P.coralD} size={50} rot={4} />
    </FadeIn>
  );
};

/* ============ S2 同步放电减弱 + 生物钟前移 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b0 = beat(0), b1 = beat(1), b2 = beat(2);

  const cardA = Math.max(0.001, spring({frame: f - 8, fps, config: {damping: 13, stiffness: 140}}));
  const cardB = Math.max(0.001, spring({frame: f - b1 - 6, fps, config: {damping: 13, stiffness: 140}}));
  const wvA = clamp01((f - 20) / 80);
  const wvB = clamp01((f - b1 - 18) / 80);
  const tlK = clamp01((f - b1 - 60) / 18);
  const oldBlk = Math.max(0.001, spring({frame: f - b2 - 8, fps, config: {damping: 12, stiffness: 140}}));
  const arrK = clamp01((f - b2 - 60) / 20);

  const px = (h: number) => 200 + (h - 18 + (h < 18 ? 24 : 0)) * (1600 / 18);
  const ticks = [18, 20, 22, 0, 2, 4, 6, 8, 10, 12];

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={2} name={sec.name} accent={P.violet} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左上：年轻大脑 深波 */}
        <g opacity={cardA} transform={`translate(${-(1 - cardA) * 60},0)`}>
          <rect x={120} y={180} width={760} height={310} rx={24} fill={P.card} />
          <text x={160} y={236} fill={P.cream} fontSize={31} fontFamily={FONT_HEAD} fontWeight={800}>
            年轻大脑 · 皮层丘脑同步放电</text>
          <BrainIcon x={240} y={360} s={0.85} />
          <rect x={360} y={290} width={480} height={150} rx={14} fill={P.bgDeep} />
          <clipPath id="s2a"><rect x={360} y={290} width={480 * wvA} height={150} /></clipPath>
          <g clipPath="url(#s2a)">
            <path d={wave(370, 365, 460, 44, 150)} fill="none" stroke={P.sky} strokeWidth={8} strokeLinecap="round" />
          </g>
          <text x={600} y={470} textAnchor="middle" fill={P.sky} fontSize={26} fontFamily={FONT_HEAD} fontWeight={800}>
            慢波又深又稳</text>
        </g>
        {/* 右上：老化大脑 深波产不出 */}
        <g opacity={cardB} transform={`translate(${(1 - cardB) * 60},0)`}>
          <rect x={1040} y={180} width={760} height={310} rx={24} fill={P.card} stroke={P.coralD} strokeWidth={4} />
          <text x={1080} y={236} fill={P.cream} fontSize={31} fontFamily={FONT_HEAD} fontWeight={800}>
            老化大脑 · 网络撑不住整晚</text>
          <BrainIcon x={1160} y={360} s={0.85} dim={1} />
          <rect x={1280} y={290} width={480} height={150} rx={14} fill={P.bgDeep} />
          <clipPath id="s2b"><rect x={1280} y={290} width={480 * wvB} height={150} /></clipPath>
          <g clipPath="url(#s2b)">
            <path d={wave(1290, 365, 460, 13, 58, 8)} fill="none" stroke={P.grey} strokeWidth={7} strokeLinecap="round" />
          </g>
          <text x={1520} y={470} textAnchor="middle" fill={P.coral} fontSize={26} fontFamily={FONT_HEAD} fontWeight={800}>
            深波产不出来</text>
        </g>
        {/* 下：生物钟前移 24h 轴 */}
        <g opacity={tlK}>
          <rect x={120} y={530} width={1680} height={320} rx={24} fill={P.card} />
          <text x={170} y={586} fill={P.cream} fontSize={30} fontFamily={FONT_HEAD} fontWeight={800}>
            生物钟前移 · 整个睡眠窗口提前了</text>
          <MoonIcon x={1740} y={582} s={0.34} hole={P.card} />
          {/* 年轻块 */}
          <rect x={px(23)} y={624} width={px(7) - px(23)} height={54} rx={27} fill={P.sky} />
          <text x={(px(23) + px(7)) / 2} y={660} textAnchor="middle" fontSize={26} fill={P.ink}
            fontFamily={FONT_HEAD} fontWeight={800}>年轻人 23:00 到 7:00</text>
          {/* 老人块 */}
          <g opacity={oldBlk}>
            <rect x={px(21) - (1 - oldBlk) * 120} y={712} width={px(4.5) - px(21)} height={54} rx={27} fill={P.coral} />
            <text x={(px(21) + px(4.5)) / 2 - (1 - oldBlk) * 120} y={748} textAnchor="middle" fontSize={26} fill={P.ink}
              fontFamily={FONT_HEAD} fontWeight={800}>老人 21:00 到 4:30</text>
          </g>
          {/* 前移箭头 */}
          <g opacity={arrK} stroke={P.amber} strokeWidth={8} fill="none" strokeLinecap="round">
            <path d={`M ${px(23)} 692 Q ${(px(23) + px(21)) / 2} 734 ${px(21) + 24} 726`} />
            <path d={`M ${px(21) + 24} 726 L ${px(21) + 58} 718 L ${px(21) + 46} 752 Z`} fill={P.amber} stroke="none" />
          </g>
          <text x={1560} y={640} fill={P.amber} fontSize={27} fontFamily={FONT_HEAD} fontWeight={800} opacity={arrK}>
            困得早 · 醒得也早</text>
          {/* 轴 */}
          <line x1={200} y1={800} x2={1800} y2={800} stroke={P.grey} strokeWidth={4} />
          {ticks.map((h) => (
            <g key={h}>
              <line x1={px(h)} y1={800} x2={px(h)} y2={814} stroke={P.grey} strokeWidth={4} />
              <text x={px(h)} y={840} textAnchor="middle" fontSize={22} fill={C.muted} fontFamily={FONT_HEAD}>
                {h}时</text>
            </g>
          ))}
        </g>
      </svg>
      <Chip text="睡不沉 · 易醒 · 难再睡" x={110} y={120} at={b0 + 14} accent={P.coral} />
      <Stamp text="四五点醒 不等于失眠" x={1400} y={112} at={b2 + 100} color={P.violetD} size={44} rot={-3} />
    </FadeIn>
  );
};

/* ============ S3 大脑排污对照：深睡开闸 vs 浅睡细流 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);

  const washK = clamp01((f - b1 - 10) / 25);
  const weakK = clamp01((f - b1 - 150) / 25);
  const resK = clamp01((f - b2 - 10) / 20);

  const washDots = [0, 1, 2, 3, 4, 5].map((i) => {
    const p = ((f * 1.5 + i * 60) % 240) / 240;
    const x = i % 2 === 0 ? 500 + i * 22 : 545 - i * 18;
    const y = 350 + p * 250;
    return {x, y, o: washK * (p > 0.92 ? (1 - p) / 0.08 : 1)};
  });
  const tankK = clamp01((f - b1 - 60) / 200);
  const pileN = Math.min(6, Math.floor((f - b1 - 150) / 45));

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={3} name={sec.name} accent={P.cyan} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <text x={960} y={152} textAnchor="middle" fill={P.cream} fontSize={34} fontFamily={FONT_HEAD} fontWeight={800}>
          深睡是大脑的排污时段 · 类淋巴系统开闸冲洗</text>
        {/* 左：深睡大开闸 */}
        <g opacity={washK}>
          <rect x={100} y={190} width={860} height={570} rx={24} fill={P.card} />
          <text x={140} y={246} fill={P.mint} fontSize={31} fontFamily={FONT_HEAD} fontWeight={800}>深睡 · 大开闸</text>
          {/* 进水口 */}
          <rect x={488} y={252} width={84} height={66} rx={12} fill={P.cardD} stroke={P.cyan} strokeWidth={5} />
          <text x={530} y={294} textAnchor="middle" fontSize={24} fill={P.cyan} fontFamily={FONT_HEAD} fontWeight={800}>脑脊液</text>
          <BrainIcon x={530} y={440} s={1.15} />
          {/* 冲洗流（虚线流动） */}
          <g stroke={P.cyan} strokeLinecap="round" fill="none" opacity={0.85}>
            <line x1={506} y1={330} x2={506} y2={566} strokeWidth={10} strokeDasharray="16 14" strokeDashoffset={-f * 7} />
            <line x1={556} y1={330} x2={556} y2={566} strokeWidth={10} strokeDasharray="16 14" strokeDashoffset={-f * 7 - 15} />
          </g>
          {/* 废物颗粒 */}
          {washDots.map((d, i) => (
            <circle key={i} cx={d.x} cy={d.y} r={10} fill={P.amber} stroke={P.coralD} strokeWidth={3} opacity={d.o} />
          ))}
          {/* 排水管 + 收集罐 */}
          <rect x={505} y={566} width={50} height={66} fill={P.cardD} stroke={P.cyan} strokeWidth={4} />
          <rect x={440} y={632} width={180} height={96} rx={16} fill={P.bgDeep} stroke={P.cyan} strokeWidth={5} />
          <rect x={448} y={632 + 88 - 80 * tankK} width={164} height={80 * tankK} rx={10} fill={P.cyan} opacity={0.4} />
          <text x={660} y={688} fill={P.cyan} fontSize={26} fontFamily={FONT_HEAD} fontWeight={800}>废物被冲走</text>
          {/* 图例 */}
          <circle cx={700} cy={400} r={10} fill={P.amber} stroke={P.coralD} strokeWidth={3} />
          <text x={722} y={410} fill={C.muted} fontSize={24} fontFamily={FONT_HEAD}>代谢废物 · 淀粉样蛋白</text>
        </g>
        {/* 右：老年浅睡细水长流 */}
        <g opacity={weakK}>
          <rect x={1000} y={190} width={820} height={570} rx={24} fill={P.card} stroke={P.coralD} strokeWidth={4} />
          <text x={1040} y={246} fill={P.coral} fontSize={31} fontFamily={FONT_HEAD} fontWeight={800}>老年浅睡 · 细水长流</text>
          <rect x={1368} y={252} width={84} height={66} rx={12} fill={P.cardD} stroke={P.grey} strokeWidth={5} />
          <text x={1410} y={294} textAnchor="middle" fontSize={24} fill={P.grey} fontFamily={FONT_HEAD} fontWeight={800}>脑脊液</text>
          <BrainIcon x={1410} y={440} s={1.15} dim={1} />
          <line x1={1410} y1={330} x2={1410} y2={566} stroke={P.cyan} strokeWidth={6} strokeLinecap="round"
            strokeDasharray="10 18" strokeDashoffset={-f * 2.2} opacity={0.5} />
          {/* 堆积的废物 */}
          {[0, 1, 2, 3, 4, 5].map((i) => i < pileN && (
            <circle key={i} cx={1370 + (i % 3) * 40} cy={528 + Math.floor(i / 3) * 26} r={11}
              fill={P.amber} stroke={P.coralD} strokeWidth={3} />
          ))}
          <text x={1410} y={612} textAnchor="middle" fill={P.coral} fontSize={26} fontFamily={FONT_HEAD} fontWeight={800}>
            冲不干净 · 越积越多</text>
          <rect x={1320} y={632} width={180} height={96} rx={16} fill={P.bgDeep} stroke={P.grey} strokeWidth={5} />
          <rect x={1328} y={706} width={164} height={14} rx={7} fill={P.cyan} opacity={0.3} />
          <text x={1540} y={688} fill={C.muted} fontSize={26} fontFamily={FONT_HEAD}>排废几乎见底</text>
        </g>
        {/* 底条结论 */}
        <g opacity={resK} transform={`translate(0,${(1 - resK) * 30})`}>
          <rect x={330} y={790} width={1260} height={92} rx={20} fill={P.bgDeep} stroke={P.coral} strokeWidth={4} />
          <text x={960} y={848} textAnchor="middle" fill={P.cream} fontSize={32} fontFamily={FONT_HEAD} fontWeight={800}>
            睡不好 不等于一定得病 · 但可能是神经退化的推手之一</text>
        </g>
      </svg>
      <Chip text="记忆巩固也靠深睡" x={110} y={130} at={b1 + 20} accent={P.violet} />
      <Stamp text="值得干预" x={1580} y={110} at={b2 + 60} color={P.mintD} size={50} rot={4} />
    </FadeIn>
  );
};

/* ============ S4 四张补救处方卡 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b0 = beat(0), b1 = beat(1);

  const cards = [
    {name: '多晒上午的太阳', sub: '最强校时信号', accent: P.amber, el: <Sun x={0} y={0} r={52} />},
    {name: '白天保持活动', sub: '疲劳感=睡意原料', accent: P.mint, el: <Walker x={0} y={-14} s={0.95} />},
    {name: '午睡别超半小时', sub: '免得提前泄压', accent: P.cyan, el: <NapTimer x={0} y={-16} s={0.9} />},
    {name: '卧室环境优化', sub: '温度·光线·噪音', accent: P.violet, el: <BedroomIcon x={0} y={-4} s={0.95} />},
  ];
  const bannerK = clamp01((f - b1 - 5) / 20);

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={4} name={sec.name} accent={P.mint} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <text x={960} y={152} textAnchor="middle" fill={P.cream} fontSize={34} fontFamily={FONT_HEAD} fontWeight={800}>
          回不到二十岁 · 但能把落差收小很多</text>
        {cards.map((c, i) => {
          const g = Math.max(0.001, spring({frame: f - b0 - 16 - i * 30, fps, config: {damping: 12, stiffness: 140}}));
          const cx = 285 + i * 450;
          return (
            <g key={i} transform={`translate(${cx},390) scale(${g})`} opacity={g}>
              <rect x={-205} y={-185} width={410} height={360} rx={24} fill={P.card} />
              <rect x={-205} y={-185} width={410} height={62} rx={24} fill={P.cardD} />
              <text x={0} y={-142} textAnchor="middle" fill={c.accent} fontSize={30} fontFamily={FONT_HEAD} fontWeight={800}>
                处方 {i + 1}</text>
              <g transform="translate(0,-32)">{c.el}</g>
              <text x={0} y={108} textAnchor="middle" fill={P.cream} fontSize={30} fontFamily={FONT_HEAD} fontWeight={800}>
                {c.name}</text>
              <text x={0} y={152} textAnchor="middle" fill={C.muted} fontSize={26} fontFamily={FONT_HEAD}>
                {c.sub}</text>
            </g>
          );
        })}
        {/* 底条：需求还在 */}
        <g opacity={bannerK} transform={`translate(0,${(1 - bannerK) * 30})`}>
          <rect x={330} y={640} width={1260} height={120} rx={24} fill={P.bgDeep} stroke={P.amber} strokeWidth={4} />
          <path d="M 400 700 L 430 700 L 444 668 L 462 726 L 476 700 L 506 700" stroke={P.coral} strokeWidth={7}
            fill="none" strokeLinecap="round" strokeLinejoin="round" />
          <text x={620} y={694} fill={P.cream} fontSize={34} fontFamily={FONT_HEAD} fontWeight={800}>
            需求还在 · 干预的窗口一直都在</text>
          <text x={620} y={738} fill={P.amber} fontSize={27} fontFamily={FONT_HEAD}>
            值得像管血压那样认真</text>
        </g>
        <text x={960} y={840} textAnchor="middle" fill={C.muted} fontSize={26} fontFamily={FONT_HEAD} opacity={bannerK}>
          别把「老了」当成放弃干预的答案</text>
      </svg>
      <Chip text="助眠药交给医生评估" x={110} y={840} at={b0 + 150} accent={P.violet} />
      <Stamp text="不只是认命" x={1560} y={108} at={b0 + 120} color={P.mintD} size={50} rot={4} />
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
        <MoonIcon x={640} y={620} s={0.8} hole={P.bgDeep} />
        {[0, 1, 2, 3].map((i) => (
          <circle key={i} cx={520 + rand(i, 71) * 300} cy={500 + rand(i, 72) * 140} r={3} fill="#fff" opacity={0.75} />
        ))}
        <Bed x={880} y={940} s={1.4} />
        <Elder x={742} y={876} s={1} z={1} />
        <g transform={`translate(1420,880) scale(${pop})`}>
          <rect x={-150} y={-64} width={300} height={120} rx={20} fill={P.card} stroke={P.cyan} strokeWidth={4} />
          <text x={0} y={-18} textAnchor="middle" fontSize={30} fill={P.cyan} fontFamily={FONT_HEAD} fontWeight={800}>
            需求还在</text>
          <text x={0} y={30} textAnchor="middle" fontSize={26} fill={C.muted} fontFamily={FONT_HEAD}>
            值得想想办法</text>
        </g>
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -300}}>
        <div style={{fontSize: 58, fontWeight: 800, color: C.text}}>不是不需要，是拿不到</div>
        <div style={{fontSize: 38, color: C.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E230 = ({data = tl}: {data?: any}) => {
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
