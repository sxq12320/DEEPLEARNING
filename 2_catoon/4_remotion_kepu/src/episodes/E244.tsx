import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, FONT_BODY, clamp01} from '../lib/style';
import {Backdrop, Stamp, SceneTag, Chip, Subtitles, ProgressBar, FadeIn} from '../lib/atoms';
import tl from '../timeline_244.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：炉膛可可 × 火苗橙 × 阀门薄荷 × 米白（纯色平涂） ============ */
const P = {
  bg: '#33212c',
  bgLite: '#3d2837',
  bgDeep: '#211420',
  panel: '#2b1b26',
  card: '#52344a',
  ink: '#170d14',
  cream: '#ffeeda',
  muted: '#d3b9ca',
  flame: '#ff9b3d',
  flameD: '#d9711c',
  ember: '#ff6b4a',
  mint: '#6fe3c1',
  mintD: '#39b493',
  amber: '#ffd166',
  amberD: '#e0a83c',
  sky: '#7ecbf0',
  pink: '#ff9ec4',
};

/* ---------- 小叉 / 小勾 ---------- */
const MarkX = ({x, y, s = 1, col = P.ember}: {x: number; y: number; s?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} stroke={col} strokeWidth={7} strokeLinecap="round">
    <line x1={-13} y1={-13} x2={13} y2={13} /><line x1={13} y1={-13} x2={-13} y2={13} />
  </g>
);
const Check = ({x, y, s = 1, col = P.mint}: {x: number; y: number; s?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} stroke={col} strokeWidth={9}
    strokeLinecap="round" strokeLinejoin="round" fill="none">
    <path d="M -16 1 L -5 13 L 18 -13" />
  </g>
);

/* ---------- 小火苗（热量燃烧） ---------- */
const Flame = ({x, y, s = 1, col = P.flame, phase = 0}: {x: number; y: number; s?: number; col?: string; phase?: number}) => {
  const f = useCurrentFrame();
  const w = 1 + Math.sin(f * 0.28 + phase) * 0.12;
  return (
    <g transform={`translate(${x},${y}) scale(${s * w})`}>
      <path d="M 0 -40 C 18 -20 27 -5 27 12 A 27 27 0 1 1 -27 12 C -27 -5 -18 -20 0 -40 Z" fill={col} />
      <path d="M 0 -13 C 9 -2 14 7 14 16 A 14 14 0 1 1 -14 16 C -14 7 -9 -2 0 -13 Z" fill={P.amber} />
    </g>
  );
};

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

/* ---------- 运动弧线（抖动/晃动示意） ---------- */
const MotionArcs = ({x, y, col = P.muted, n = 3, flip = false}: {x: number; y: number; col?: string; n?: number; flip?: boolean}) => {
  const f = useCurrentFrame();
  const k = 0.55 + Math.abs(Math.sin(f * 0.5)) * 0.45;
  return (
    <g transform={`translate(${x},${y}) scale(${flip ? -1 : 1},1)`} opacity={k} stroke={col}
      strokeWidth={5} fill="none" strokeLinecap="round">
      {Array.from({length: n}).map((_, i) => (
        <path key={i} d={`M ${i * 22} ${-18 + i * 4} Q ${18 + i * 22} 0 ${i * 22} ${18 - i * 4}`} />
      ))}
    </g>
  );
};

/* ---------- 圆头小人：站（脚底 y=0） ---------- */
const Stander = ({x, y, s = 1, col = P.mint, lean = 0, fidget = 0}: {
  x: number; y: number; s?: number; col?: string; lean?: number; fidget?: number;
}) => {
  const f = useCurrentFrame();
  const sway = Math.sin(f * 0.18) * 4 * fidget;
  const bob = Math.abs(Math.sin(f * 0.3)) * 4 * fidget;
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      <ellipse cx={0} cy={6} rx={70} ry={10} fill={P.ink} opacity={0.3} />
      <g transform={`translate(${sway},${-bob})`}>
        {/* 腿 */}
        <line x1={-12} y1={-70} x2={-16} y2={-4} stroke={P.ink} strokeWidth={22} strokeLinecap="round" />
        <line x1={12} y1={-70} x2={18} y2={-4} stroke={P.ink} strokeWidth={22} strokeLinecap="round" />
        {/* 鞋 */}
        <ellipse cx={-18} cy={-2} rx={20} ry={9} fill={P.cream} />
        <ellipse cx={20} cy={-2} rx={20} ry={9} fill={P.cream} />
        {/* 躯干 */}
        <g transform={`rotate(${lean} 0 -120)`}>
          <rect x={-30} y={-168} width={60} height={102} rx={28} fill={col} />
          {/* 手臂 */}
          <line x1={-26} y1={-146} x2={-44} y2={-88} stroke={C.skin} strokeWidth={16} strokeLinecap="round" />
          <line x1={26} y1={-146} x2={46} y2={-92} stroke={C.skin} strokeWidth={16} strokeLinecap="round" />
        </g>
        {/* 头 */}
        <g transform={`translate(${lean * 0.5},-196)`}>
          <circle r={30} fill={C.skin} />
          <path d="M -29 -6 Q -20 -38 8 -36 Q 32 -33 29 -4 Q 0 -22 -29 -6 Z" fill={P.ink} />
          <circle cx={11} cy={0} r={4.2} fill={P.ink} />
        </g>
      </g>
    </g>
  );
};

/* ---------- 圆头小人：走（脚底 y=0） ---------- */
const Walker = ({x, y, s = 1, col = P.flame, speed = 0.28}: {x: number; y: number; s?: number; col?: string; speed?: number}) => {
  const f = useCurrentFrame();
  const ph = f * speed;
  const bob = Math.abs(Math.sin(ph)) * 8;
  const legA = Math.sin(ph) * 26;
  const legB = Math.sin(ph + Math.PI) * 26;
  const armA = Math.sin(ph + Math.PI) * 20;
  const armB = Math.sin(ph) * 20;
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      <ellipse cx={0} cy={7} rx={84} ry={11} fill={P.ink} opacity={0.3} />
      <g transform={`translate(0,${-bob})`}>
        <line x1={-8} y1={-74} x2={-8 - legA} y2={-4} stroke={P.ink} strokeWidth={22} strokeLinecap="round" />
        <line x1={8} y1={-74} x2={8 - legB} y2={-4} stroke={P.ink} strokeWidth={22} strokeLinecap="round" />
        <ellipse cx={-10 - legA} cy={-3} rx={20} ry={9} fill={P.cream} />
        <ellipse cx={10 - legB} cy={-3} rx={20} ry={9} fill={P.cream} />
        <g transform="rotate(10 0 -120)">
          <rect x={-30} y={-168} width={60} height={100} rx={28} fill={col} />
          <line x1={-22} y1={-146} x2={-40 - armA} y2={-88} stroke={C.skin} strokeWidth={16} strokeLinecap="round" />
          <line x1={24} y1={-146} x2={42 - armB} y2={-88} stroke={C.skin} strokeWidth={16} strokeLinecap="round" />
        </g>
        <g transform="translate(14,-194)">
          <circle r={30} fill={C.skin} />
          <path d="M -29 -6 Q -20 -38 8 -36 Q 32 -33 29 -4 Q 0 -22 -29 -6 Z" fill={P.ink} />
          <circle cx={13} cy={0} r={4.2} fill={P.ink} />
        </g>
      </g>
    </g>
  );
};

/* ---------- 坐椅小人：抖腿（椅面 y≈-70，脚底 y≈0） ---------- */
const Sitter = ({x, y, s = 1, col = P.sky, fidget = 1}: {x: number; y: number; s?: number; col?: string; fidget?: number}) => {
  const f = useCurrentFrame();
  const jig = Math.sin(f * 0.55) * 16 * fidget;
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      <ellipse cx={10} cy={6} rx={88} ry={11} fill={P.ink} opacity={0.3} />
      {/* 椅子 */}
      <rect x={-64} y={-78} width={92} height={14} rx={7} fill={P.card} />
      <rect x={-70} y={-170} width={14} height={96} rx={7} fill={P.card} />
      <rect x={-58} y={-64} width={10} height={64} rx={5} fill={P.card} />
      <rect x={10} y={-64} width={10} height={64} rx={5} fill={P.card} />
      {/* 大腿（水平） */}
      <line x1={-8} y1={-84} x2={58} y2={-78} stroke={P.ink} strokeWidth={24} strokeLinecap="round" />
      {/* 小腿：膝点 (58,-78)，抖动 */}
      <g transform={`translate(58,-78) rotate(${jig})`}>
        <line x1={0} y1={0} x2={6} y2={72} stroke={P.ink} strokeWidth={20} strokeLinecap="round" />
        <ellipse cx={10} cy={76} rx={19} ry={9} fill={P.cream} />
      </g>
      {fidget > 0 && <MotionArcs x={92} y={-46} col={P.mint} n={3} />}
      {/* 躯干 */}
      <rect x={-52} y={-172} width={62} height={96} rx={28} fill={col} />
      <line x1={-38} y1={-150} x2={30} y2={-104} stroke={C.skin} strokeWidth={15} strokeLinecap="round" />
      {/* 头 */}
      <g transform="translate(-18,-200)">
        <circle r={30} fill={C.skin} />
        <path d="M -29 -6 Q -20 -38 8 -36 Q 32 -33 29 -4 Q 0 -22 -29 -6 Z" fill={P.ink} />
        <circle cx={13} cy={0} r={4.2} fill={P.ink} />
      </g>
    </g>
  );
};

/* ---------- 体重秤 ---------- */
const Scale = ({x, y, s = 1, needle = 0, num = '63'}: {x: number; y: number; s?: number; needle?: number; num?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-76} y={-34} width={152} height={46} rx={14} fill={P.bgDeep} stroke={P.muted} strokeWidth={4} />
    <path d="M -52 -34 A 52 52 0 0 1 52 -34" fill={P.card} stroke={P.muted} strokeWidth={4} />
    {[-40, -20, 0, 20, 40].map((a) => (
      <line key={a} x1={Math.sin(a * Math.PI / 180) * 40} y1={-34 - Math.cos(a * Math.PI / 180) * 40}
        x2={Math.sin(a * Math.PI / 180) * 48} y2={-34 - Math.cos(a * Math.PI / 180) * 48}
        stroke={P.cream} strokeWidth={3.5} />
    ))}
    <line x1={0} y1={-34} x2={Math.sin(needle * Math.PI / 180) * 36} y2={-34 - Math.cos(needle * Math.PI / 180) * 36}
      stroke={P.ember} strokeWidth={5} strokeLinecap="round" />
    <circle cx={0} cy={-34} r={5} fill={P.cream} />
    <text x={0} y={0} textAnchor="middle" fontSize={21} fill={P.mint} fontFamily={FONT_HEAD} fontWeight={800}>{num}kg</text>
  </g>
);

/* ---------- 汉堡 ---------- */
const Burger = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <path d="M -64 -6 A 64 44 0 0 1 64 -6 Z" fill={P.amber} />
    <circle cx={-26} cy={-32} r={4} fill={P.cream} /><circle cx={4} cy={-40} r={4} fill={P.cream} />
    <circle cx={30} cy={-28} r={4} fill={P.cream} />
    <path d="M -68 -4 Q -40 -18 -14 -4 Q 12 -18 38 -4 Q 56 -14 68 -4 L 68 6 L -68 6 Z" fill={P.mintD} />
    <rect x={-66} y={8} width={132} height={18} rx={9} fill={P.ember} />
    <path d="M -64 26 L 64 26 L 56 44 L -56 44 Z" fill={P.amberD} />
  </g>
);

/* ---------- 奶茶杯 ---------- */
const MilkTea = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <path d="M -34 -62 L 34 -62 L 26 34 L -26 34 Z" fill={P.bgDeep} stroke={P.muted} strokeWidth={4} />
    <path d="M -29 -40 L 29 -40 L 24 28 L -24 28 Z" fill={P.pink} opacity={0.9} />
    {[0, 1, 2, 3].map((i) => (
      <circle key={i} cx={-14 + (i % 2) * 24} cy={16 - Math.floor(i / 2) * 14} r={7} fill={P.ink} />
    ))}
    <line x1={8} y1={-62} x2={24} y2={-96} stroke={P.muted} strokeWidth={7} strokeLinecap="round" />
  </g>
);

/* ---------- 阀门 ---------- */
const Valve = ({x, y, s = 1, open = 0.5}: {x: number; y: number; s?: number; open?: number}) => {
  const f = useCurrentFrame();
  const rot = -40 + open * 80;
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      <circle r={64} fill={P.bgDeep} stroke={P.mint} strokeWidth={7} />
      <circle r={44} fill="none" stroke={P.card} strokeWidth={5} />
      {[0, 60, 120].map((a) => (
        <line key={a} x1={0} y1={0} x2={Math.sin((a + rot) * Math.PI / 180) * 56}
          y2={-Math.cos((a + rot) * Math.PI / 180) * 56}
          stroke={P.mint} strokeWidth={10} strokeLinecap="round" />
      ))}
      {[0, 60, 120].map((a) => (
        <line key={`b${a}`} x1={0} y1={0} x2={-Math.sin((a + rot) * Math.PI / 180) * 56}
          y2={Math.cos((a + rot) * Math.PI / 180) * 56}
          stroke={P.mint} strokeWidth={10} strokeLinecap="round" />
        ))}
      <circle r={14} fill={P.amber} stroke={P.ink} strokeWidth={3} />
      {/* 开度指针 */}
      <g transform="rotate(110)">
        <line x1={0} y1={-70} x2={0} y2={-82} stroke={P.ember} strokeWidth={7} strokeLinecap="round"
          transform={`rotate(${rot - 110})`} opacity={0} />
      </g>
    </g>
  );
};

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
        <g transform={`translate(960,520) scale(${pA}) translate(-960,-520)`} opacity={clamp01(pA)}>
          {/* 地面 */}
          <rect x={560} y={586} width={800} height={26} rx={13} fill={P.card} />
          {/* 站上秤的人 */}
          <Scale x={960} y={586} s={1.25} needle={-6} num="63" />
          <Stander x={960} y={566} s={1.3} col={P.mint} fidget={1} />
          {/* 两侧食物 */}
          <Burger x={680} y={540} s={1.1} />
          <MilkTea x={1250} y={520} s={1.15} />
          {/* 小出口漏热：火苗 */}
          <Flame x={1064} y={430} s={0.5} phase={1} />
          <Flame x={880} y={470} s={0.4} phase={3} />
          <MotionArcs x={1050} y={540} col={P.mint} n={3} />
        </g>
        <g transform={`translate(560,300) scale(${pB})`} opacity={clamp01(pB)}>
          <TagPill x={0} y={0} t="学名 NEAT" col={P.flame} w={200} />
        </g>
        <g transform={`translate(1380,320) scale(${pB})`} opacity={clamp01(pB)}>
          <TagPill x={0} y={0} t="小动作也在烧" col={P.mint} w={230} />
        </g>
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 460}}>
          <div style={{color: P.flame, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 24}}>
            冷知识 · 第 {data.episode} 期
          </div>
          <div style={{color: C.text, fontSize: 76, fontWeight: 900, fontFamily: FONT_HEAD, letterSpacing: 4}}>
            {data.title}
          </div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ============ S0 秘密藏在零碎小动作 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const b2 = beat(2);
  const popL = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const neatK = clamp01((f - b1 - 5) / 26);
  const sumK = clamp01((f - b2 - 5) / 26);

  const tics = [
    {t: '抖腿', icon: 0},
    {t: '站立', icon: 1},
    {t: '踱步', icon: 2},
    {t: '换姿势', icon: 3},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={0} name={sec.name} accent={P.flame} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：办公室那个人 */}
        <g transform={`translate(90,140) scale(${popL}) translate(-90,-140)`} opacity={clamp01(popL * 2)}>
          <rect x={90} y={140} width={800} height={560} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={126} y={196} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            办公室总有这么一个人 · 吃最多却不涨秤</text>
          {/* 桌子 */}
          <rect x={430} y={470} width={400} height={26} rx={13} fill={P.card} />
          <rect x={470} y={496} width={14} height={120} rx={7} fill={P.card} />
          <rect x={776} y={496} width={14} height={120} rx={7} fill={P.card} />
          {/* 大碗 + 奶茶 */}
          <g transform="translate(560,452)">
            <path d="M -66 0 A 66 40 0 0 0 66 0 Z" fill={P.cream} />
            <path d="M -56 2 A 56 30 0 0 0 56 2 Z" fill={P.amber} opacity={0.9} />
            <line x1={30} y1={-36} x2={58} y2={-4} stroke={P.cream} strokeWidth={6} strokeLinecap="round" />
          </g>
          <MilkTea x={740} y={440} s={0.9} />
          {/* 坐着吃的人 */}
          <Sitter x={300} y={620} s={1.05} col={P.sky} fidget={0} />
          {/* 想下一顿的气泡 */}
          <g transform="translate(220,300)">
            <ellipse rx={86} ry={64} fill={P.bgDeep} stroke={P.card} strokeWidth={4} />
            <circle cx={30} cy={66} r={12} fill={P.bgDeep} stroke={P.card} strokeWidth={3.5} />
            <circle cx={52} cy={92} r={7} fill={P.bgDeep} stroke={P.card} strokeWidth={3} />
            <Burger x={0} y={-6} s={0.62} />
          </g>
          {/* 体重秤：指针不动 */}
          <Scale x={680} y={630} s={0.95} needle={0} num="63" />
          <g opacity={0.9}>
            <line x1={586} y1={566} x2={800} y2={566} stroke={P.mint} strokeWidth={5}
              strokeDasharray="10 9" strokeLinecap="round" />
            <text x={693} y={552} textAnchor="middle" fontSize={22} fill={P.mint}
              fontFamily={FONT_HEAD} fontWeight={800}>体重十年不动</text>
          </g>
        </g>
        {/* 右：NEAT 名片 + 小动作清单（cue1） */}
        {neatK > 0 && (
          <g opacity={neatK} transform={`translate(0,${(1 - neatK) * 40})`}>
            <rect x={930} y={140} width={900} height={560} rx={20} fill={P.panel}
              stroke={P.flame} strokeWidth={4} />
            <text x={966} y={196} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
              别急着归功于代谢快 · 真凶是这套零碎消耗</text>
            {/* 大名卡 */}
            <rect x={966} y={228} width={828} height={118} rx={18} fill={P.bgDeep}
              stroke={P.flame} strokeWidth={4} />
            <text x={1380} y={288} textAnchor="middle" fontSize={58} fill={P.flame}
              fontFamily={FONT_HEAD} fontWeight={900} letterSpacing={10}>N E A T</text>
            <text x={1380} y={328} textAnchor="middle" fontSize={25} fill={P.cream}
              fontFamily={FONT_HEAD} fontWeight={800}>非运动性活动产热 · 本人完全无感</text>
            {/* 四个小动作 */}
            {tics.map((tc, i) => (
              <g key={i} transform={`translate(${1080 + i * 200},480)`}>
                <rect x={-86} y={-92} width={172} height={184} rx={18} fill={P.bgDeep}
                  stroke={P.card} strokeWidth={3.5} />
                {tc.icon === 0 && (
                  <g transform="translate(0,-14)">
                    <line x1={-26} y1={-34} x2={-4} y2={-8} stroke={P.mint} strokeWidth={14}
                      strokeLinecap="round" />
                    <g transform={`translate(-4,-8) rotate(${Math.sin(f * 0.55) * 22})`}>
                      <line x1={0} y1={0} x2={4} y2={42} stroke={P.mint} strokeWidth={12}
                        strokeLinecap="round" />
                    </g>
                    <MotionArcs x={26} y={6} col={P.mint} n={2} />
                  </g>
                )}
                {tc.icon === 1 && <Stander x={0} y={30} s={0.42} col={P.flame} fidget={0} />}
                {tc.icon === 2 && <Walker x={6} y={32} s={0.4} col={P.amber} speed={0.2} />}
                {tc.icon === 3 && (
                  <g transform="translate(0,0)">
                    <path d={`M -34 30 Q ${Math.sin(f * 0.3) * 14} -20 34 30`} fill="none"
                      stroke={P.pink} strokeWidth={13} strokeLinecap="round" />
                    <circle cx={Math.sin(f * 0.3) * 14} cy={-30} r={15} fill={C.skin} />
                  </g>
                )}
                <text x={0} y={70} textAnchor="middle" fontSize={24} fill={P.cream}
                  fontFamily={FONT_HEAD} fontWeight={800}>{tc.t}</text>
              </g>
            ))}
          </g>
        )}
        {/* 底条：一天差几百大卡 ≈ 多跑五公里（cue2） */}
        {sumK > 0 && (
          <g opacity={sumK} transform={`translate(0,${(1 - sumK) * 50})`}>
            <rect x={90} y={740} width={1740} height={130} rx={20} fill={P.panel}
              stroke={P.mint} strokeWidth={4} />
            <Flame x={170} y={806} s={0.7} />
            <text x={240} y={796} fontSize={30} fill={P.cream} fontFamily={FONT_HEAD} fontWeight={800}>
              零碎加起来 · 一天差出</text>
            <text x={240} y={840} fontSize={30} fill={P.cream} fontFamily={FONT_HEAD} fontWeight={800}>
              几百大卡</text>
            <text x={760} y={822} textAnchor="middle" fontSize={52} fill={P.amber}
              fontFamily={FONT_HEAD} fontWeight={900}>≈</text>
            <Walker x={1010} y={830} s={0.52} col={P.mint} />
            <text x={1150} y={822} fontSize={30} fill={P.mint} fontFamily={FONT_HEAD} fontWeight={900}>
              悄悄多跑 5 公里 · 不用迈腿去健身房</text>
          </g>
        )}
      </svg>
      <Chip text="抖腿站立走动 · 全算" x={1330} y={72} at={beat(0) + 30} accent={P.flame} />
      <Stamp text="差距不在健身房" x={600} y={86} at={b1 + 40} color={P.flameD} size={38} rot={-3} />
    </FadeIn>
  );
};

/* ============ S1 热量账本有四栏 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const b2 = beat(2);
  const pop = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const rowsK = clamp01((f - b1 - 5) / 26);
  const gapK = clamp01((f - b2 - 5) / 26);

  const rows = [
    {t: '基础代谢', s: '心跳呼吸体温', v: 0.65, col: P.mint},
    {t: '食物热效应', s: '消化本身耗能', v: 0.1, col: P.amber},
    {t: '正经运动', s: '摊到全天没几个点', v: 0.05, col: P.sky},
    {t: 'NEAT', s: '剩下所有小动作', v: 0.2, col: P.flame},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={1} name={sec.name} accent={P.amber} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：账本 */}
        <g transform={`translate(90,140) scale(${pop}) translate(-90,-140)`} opacity={clamp01(pop * 2)}>
          <rect x={90} y={140} width={830} height={600} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={126} y={196} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            多数人算账只看两栏 · 其实支出有四栏</text>
          {/* 两栏旧账：被划掉 */}
          <g opacity={0.85}>
            <rect x={126} y={226} width={200} height={70} rx={14} fill={P.bgDeep}
              stroke={P.card} strokeWidth={3.5} />
            <text x={226} y={272} textAnchor="middle" fontSize={27} fill={P.muted}
              fontFamily={FONT_HEAD} fontWeight={800}>吃了多少</text>
            <rect x={346} y={226} width={200} height={70} rx={14} fill={P.bgDeep}
              stroke={P.card} strokeWidth={3.5} />
            <text x={446} y={272} textAnchor="middle" fontSize={27} fill={P.muted}
              fontFamily={FONT_HEAD} fontWeight={800}>练了多少</text>
            <MarkX x={576} y={262} s={1.3} col={P.ember} />
            <text x={640} y={272} fontSize={24} fill={P.ember} fontFamily={FONT_HEAD} fontWeight={800}>
              只算这两栏 = 漏账</text>
          </g>
          {/* 四栏真实支出（cue1） */}
          <g opacity={rowsK} transform={`translate(0,${(1 - rowsK) * 40})`}>
            {rows.map((r, i) => {
              const w = 560 * r.v;
              const y = 340 + i * 96;
              const grow = clamp01((f - b1 - 8 - i * 10) / 22);
              return (
                <g key={i} transform={`translate(126,${y})`}>
                  <text x={0} y={32} fontSize={26} fill={r.col} fontFamily={FONT_HEAD} fontWeight={900}>
                    {r.t}</text>
                  <text x={0} y={62} fontSize={19} fill={P.muted} fontFamily={FONT_BODY}>{r.s}</text>
                  <rect x={210} y={0} width={560} height={46} rx={23} fill={P.bgDeep} />
                  <rect x={210} y={0} width={Math.max(40, w * grow)} height={46} rx={23} fill={r.col}
                    opacity={grow} />
                  <text x={230 + Math.max(40, w * grow)} y={32} fontSize={24} fill={P.cream}
                    fontFamily={FONT_HEAD} fontWeight={900}>{Math.round(r.v * 100)}%</text>
                  {i === 3 && <Flame x={410} y={22} s={0.4} />}
                </g>
              );
            })}
          </g>
        </g>
        {/* 右：NEAT 是巨大变量池（cue2） */}
        <g transform={`translate(950,140) scale(${pop}) translate(-950,-140)`} opacity={clamp01(pop * 2)}>
          <rect x={950} y={140} width={880} height={600} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={986} y={196} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            走路倒水站着接电话 · 全进 NEAT 这个池子</text>
          {/* 两个小人：安静 vs 闲不住 */}
          <g transform="translate(1120,560)">
            <Sitter x={0} y={0} s={0.95} col={P.muted} fidget={0} />
            <text x={0} y={56} textAnchor="middle" fontSize={24} fill={P.muted}
              fontFamily={FONT_HEAD} fontWeight={800}>安静型</text>
            <rect x={-88} y={76} width={176} height={52} rx={14} fill={P.bgDeep}
              stroke={P.card} strokeWidth={3.5} />
            <text x={0} y={112} textAnchor="middle" fontSize={23} fill={P.muted}
              fontFamily={FONT_HEAD} fontWeight={800}>≈200 大卡</text>
          </g>
          <g transform="translate(1660,560)">
            <Sitter x={0} y={0} s={0.95} col={P.flame} fidget={1} />
            <text x={0} y={56} textAnchor="middle" fontSize={24} fill={P.flame}
              fontFamily={FONT_HEAD} fontWeight={800}>闲不住型</text>
            <rect x={-88} y={76} width={176} height={52} rx={14} fill={P.bgDeep}
              stroke={P.flame} strokeWidth={3.5} />
            <text x={0} y={112} textAnchor="middle" fontSize={23} fill={P.flame}
              fontFamily={FONT_HEAD} fontWeight={900}>≈900 大卡</text>
          </g>
          {/* 差额拉条（cue2） */}
          {gapK > 0 && (
            <g opacity={gapK}>
              <line x1={1240} y1={520} x2={1540} y2={520} stroke={P.amber} strokeWidth={10}
                strokeLinecap="round" />
              <polygon points="1556,520 1532,506 1532,534" fill={P.amber} />
              <polygon points="1224,520 1248,506 1248,534" fill={P.amber} />
              <rect x={1280} y={392} width={220} height={96} rx={18} fill={P.bgDeep}
                stroke={P.amber} strokeWidth={4} transform={`scale(${0.6 + gapK * 0.4}) translate(${1390 * (1 / (0.6 + gapK * 0.4) - 1)},0)`} />
              <text x={1390} y={436} textAnchor="middle" fontSize={32} fill={P.amber}
                fontFamily={FONT_HEAD} fontWeight={900}>差 700+</text>
              <text x={1390} y={470} textAnchor="middle" fontSize={20} fill={P.muted}
                fontFamily={FONT_HEAD}>大卡 / 天</text>
              <text x={1390} y={596} textAnchor="middle" fontSize={23} fill={P.muted}
                fontFamily={FONT_HEAD}>足以解释：吃一样的饭 · 一个胖一个瘦</text>
            </g>
          )}
        </g>
      </svg>
      <Chip text="不止吃和练两栏" x={1380} y={72} at={beat(0) + 30} accent={P.amber} />
      <Stamp text="NEAT 是大变量池" x={600} y={86} at={b2 + 20} color={P.mintD} size={38} rot={-3} />
    </FadeIn>
  );
};

/* ============ S2 喂胖实验见分晓 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const b2 = beat(2);
  const pop = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const resK = clamp01((f - b1 - 5) / 26);
  const valveK = clamp01((f - b2 - 5) / 26);
  const drop = (f * 3) % 90;

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={2} name={sec.name} accent={P.ember} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 顶：实验装置 */}
        <g transform={`translate(90,130) scale(${pop}) translate(-90,-130)`} opacity={clamp01(pop * 2)}>
          <rect x={90} y={130} width={1740} height={200} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <TagPill x={270} y={196} t="梅奥诊所 · 90年代" col={P.muted} w={300} fs={25} />
          {/* 漏斗：每天 +1000 大卡 */}
          <g transform="translate(640,230)">
            <path d="M -80 -56 L 80 -56 L 26 6 L 26 46 L -26 46 L -26 6 Z" fill={P.bgDeep}
              stroke={P.amber} strokeWidth={4} />
            {[0, 1, 2].map((i) => (
              <circle key={i} cx={0} cy={-60 + ((drop + i * 30) % 90)} r={7}
                fill={P.amber} opacity={clamp01(1 - ((drop + i * 30) % 90) / 90)} />
            ))}
            <text x={0} y={86} textAnchor="middle" fontSize={24} fill={P.amber}
              fontFamily={FONT_HEAD} fontWeight={900}>每天 +1000 大卡 · 连吃 8 周</text>
          </g>
          {/* 日历 */}
          <g transform="translate(960,230)">
            <rect x={-56} y={-56} width={112} height={104} rx={14} fill={P.bgDeep}
              stroke={P.sky} strokeWidth={4} />
            <rect x={-56} y={-56} width={112} height={26} rx={13} fill={P.sky} />
            <text x={0} y={24} textAnchor="middle" fontSize={34} fill={P.sky}
              fontFamily={FONT_HEAD} fontWeight={900}>8周</text>
          </g>
          {/* 一组志愿者 */}
          {[0, 1, 2].map((i) => (
            <Stander key={i} x={1180 + i * 130} y={290} s={0.5}
              col={[P.mint, P.amber, P.pink][i]} fidget={0} />
          ))}
          <text x={1560} y={196} fontSize={24} fill={P.muted} fontFamily={FONT_HEAD}>
            所有人保持原有锻炼量</text>
          <text x={1560} y={230} fontSize={24} fill={P.muted} fontFamily={FONT_HEAD}>
            基础代谢变化也差不多</text>
        </g>
        {/* 中左：结果 A 胖四公斤 */}
        {resK > 0 && (
          <g opacity={resK} transform={`translate(0,${(1 - resK) * 40})`}>
            <rect x={90} y={360} width={850} height={300} rx={20} fill={P.panel}
              stroke={P.ember} strokeWidth={4} />
            <Stander x={300} y={580} s={1.05} col={P.muted} fidget={0} />
            <ellipse cx={300} cy={486} rx={58} ry={64} fill={P.muted} opacity={0.9} />
            <Stander x={300} y={580} s={1.05} col={P.muted} fidget={0} />
            <Scale x={300} y={602} s={0.9} needle={38} num="67" />
            <g transform="translate(560,470)">
              <text x={0} y={0} fontSize={30} fill={P.muted} fontFamily={FONT_HEAD}>有人</text>
              <text x={0} y={72} fontSize={72} fill={P.ember} fontFamily={FONT_HEAD}
                fontWeight={900}>+4kg</text>
              <text x={0} y={118} fontSize={23} fill={P.muted} fontFamily={FONT_HEAD}>
                热量照单全收存进仓库</text>
            </g>
          </g>
        )}
        {/* 中右：结果 B 不到一公斤 */}
        {resK > 0 && (
          <g opacity={resK} transform={`translate(0,${(1 - resK) * 40})`}>
            <rect x={970} y={360} width={860} height={300} rx={20} fill={P.panel}
              stroke={P.mint} strokeWidth={4} />
            <Stander x={1180} y={580} s={1.05} col={P.mint} fidget={1} />
            <Scale x={1180} y={602} s={0.9} needle={4} num="63.8" />
            <MotionArcs x={1260} y={540} col={P.mint} n={3} />
            <g transform="translate(1450,470)">
              <text x={0} y={0} fontSize={30} fill={P.muted} fontFamily={FONT_HEAD}>有人</text>
              <text x={0} y={72} fontSize={72} fill={P.mint} fontFamily={FONT_HEAD}
                fontWeight={900}>&lt;1kg</text>
              <text x={0} y={118} fontSize={23} fill={P.muted} fontFamily={FONT_HEAD}>
                小动作消耗自动上调烧掉了</text>
            </g>
          </g>
        )}
        {/* 底：自动平衡阀机制（cue2） */}
        {valveK > 0 && (
          <g opacity={valveK} transform={`translate(0,${(1 - valveK) * 50})`}>
            <rect x={90} y={700} width={1740} height={160} rx={20} fill={P.panel}
              stroke={P.flame} strokeWidth={4} />
            <text x={140} y={752} fontSize={26} fill={P.muted} fontFamily={FONT_HEAD}>
              差别全在 NEAT · 身体装了</text>
            <text x={140} y={794} fontSize={30} fill={P.flame} fontFamily={FONT_HEAD} fontWeight={900}>
              自动平衡阀</text>
            {/* 管道：多余热量 → 阀门 → 火苗 */}
            <line x1={470} y1={780} x2={660} y2={780} stroke={P.amber} strokeWidth={16}
              strokeLinecap="round" />
            <text x={560} y={752} textAnchor="middle" fontSize={21} fill={P.amber}
              fontFamily={FONT_HEAD}>多余热量</text>
            <Valve x={740} y={780} s={0.72} open={valveK} />
            <line x1={820} y1={780} x2={960} y2={780} stroke={P.flame} strokeWidth={16}
              strokeLinecap="round" />
            {[0, 1, 2].map((i) => (
              <Flame key={i} x={1010 + i * 66} y={776} s={0.45} phase={i * 2} />
            ))}
            <text x={1300} y={758} fontSize={25} fill={P.cream} fontFamily={FONT_HEAD} fontWeight={800}>
              阀门灵的人：多吃 → 站得多坐得直</text>
            <text x={1300} y={806} fontSize={23} fill={P.muted} fontFamily={FONT_HEAD}>
              阀门不灵的人：多吃 → 全存进仓库</text>
          </g>
        )}
      </svg>
      <Chip text="多吃一千大卡 × 8周" x={1330} y={72} at={beat(0) + 30} accent={P.ember} />
      <Stamp text="不是天赋吃不胖" x={600} y={86} at={b2 + 30} color={P.flameD} size={38} rot={-3} />
    </FadeIn>
  );
};

/* ============ S3 坐不住是出厂设置 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const b2 = beat(2);
  const geneK = clamp01((f - b1 - 5) / 26);
  const baseK = clamp01((f - b2 - 5) / 26);
  const glow = 0.6 + Math.sin(f * 0.2) * 0.4;

  const traits = [
    {t: '坐着抖腿', col: P.mint},
    {t: '站着换重心', col: P.flame},
    {t: '走路步频快', col: P.amber},
    {t: '坐不住爱起身', col: P.pink},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={3} name={sec.name} accent={P.mint} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 顶行：四个特征卡 */}
        {traits.map((tr, i) => {
          const at = 8 + i * 12;
          const sp = Math.max(0.001, spring({frame: f - at, fps, config: {damping: 11, stiffness: 130}}));
          const sx = 90 + i * 450;
          return (
            <g key={i} transform={`translate(${sx},130) scale(${sp}) translate(${-sx},-130)`}
              opacity={clamp01(sp * 2)}>
              <rect x={sx} y={130} width={410} height={230} rx={20} fill={P.panel}
                stroke={tr.col} strokeWidth={4} />
              <g transform={`translate(${sx + 205},240)`}>
                {i === 0 && (
                  <g>
                    <line x1={-30} y1={-40} x2={-4} y2={-10} stroke={tr.col} strokeWidth={16}
                      strokeLinecap="round" />
                    <g transform={`translate(-4,-10) rotate(${Math.sin(f * 0.6) * 22})`}>
                      <line x1={0} y1={0} x2={6} y2={52} stroke={tr.col} strokeWidth={13}
                        strokeLinecap="round" />
                    </g>
                    <MotionArcs x={36} y={8} col={tr.col} n={3} />
                  </g>
                )}
                {i === 1 && <Stander x={0} y={52} s={0.5} col={tr.col} fidget={1} lean={6} />}
                {i === 2 && <Walker x={0} y={54} s={0.48} col={tr.col} speed={0.45} />}
                {i === 3 && (
                  <g>
                    <rect x={-56} y={8} width={70} height={12} rx={6} fill={P.card} />
                    <rect x={-60} y={-52} width={10} height={66} rx={5} fill={P.card} />
                    <g transform={`translate(20,${-Math.abs(Math.sin(f * 0.25)) * 18})`}>
                      <Stander x={0} y={50} s={0.42} col={tr.col} fidget={0} />
                    </g>
                    <line x1={4} y1={-64} x2={4} y2={-84} stroke={tr.col} strokeWidth={6}
                      strokeLinecap="round" />
                    <polygon points="4,-96 -6,-78 14,-78" fill={tr.col} />
                  </g>
                )}
              </g>
              <text x={sx + 205} y={332} textAnchor="middle" fontSize={27} fill={tr.col}
                fontFamily={FONT_HEAD} fontWeight={900}>{tr.t}</text>
            </g>
          );
        })}
        {/* 左下：出厂设置（cue1） */}
        {geneK > 0 && (
          <g opacity={geneK} transform={`translate(0,${(1 - geneK) * 40})`}>
            <rect x={90} y={390} width={860} height={330} rx={20} fill={P.panel}
              stroke={P.sky} strokeWidth={4} />
            <text x={126} y={446} fill={P.sky} fontSize={28} fontFamily={FONT_HEAD} fontWeight={900}>
              「坐不住」部分是出厂设置</text>
            {/* 脑 + 冲动拨盘 */}
            <g transform="translate(280,552)">
              <path d="M -70 -10 Q -78 -68 -20 -76 Q -6 -96 26 -84 Q 74 -80 72 -30 Q 92 -8 70 16 Q 60 46 16 44 Q -40 50 -62 22 Q -76 10 -70 -10 Z"
                fill={P.pink} opacity={0.95} />
              <path d="M -40 -40 Q -20 -52 -2 -40 M -34 -6 Q -10 -18 14 -8 M -20 24 Q 4 14 30 22"
                stroke={P.bgDeep} strokeWidth={6} fill="none" strokeLinecap="round" />
              {/* 多巴胺分子点 */}
              {[0, 1, 2].map((i) => (
                <circle key={i} cx={-36 + i * 38} cy={-100 - (i % 2) * 16} r={9}
                  fill={P.amber} opacity={glow} />
              ))}
              <text x={0} y={106} textAnchor="middle" fontSize={22} fill={P.muted}
                fontFamily={FONT_HEAD}>多巴胺 · 甲状腺素</text>
              <text x={0} y={134} textAnchor="middle" fontSize={22} fill={P.muted}
                fontFamily={FONT_HEAD}>决定活动冲动</text>
            </g>
            {/* 冲动拨盘：高 */}
            <g transform="translate(600,540)">
              <path d="M -110 40 A 110 110 0 0 1 110 40" fill="none" stroke={P.bgDeep}
                strokeWidth={26} strokeLinecap="round" />
              <path d="M -110 40 A 110 110 0 0 1 96 -10" fill="none" stroke={P.flame}
                strokeWidth={26} strokeLinecap="round" opacity={0.25 + geneK * 0.75} />
              <line x1={0} y1={40} x2={72} y2={-26} stroke={P.cream} strokeWidth={8}
                strokeLinecap="round" />
              <circle cx={0} cy={40} r={12} fill={P.cream} />
              <text x={-116} y={86} fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>天生蔫</text>
              <text x={60} y={86} fontSize={22} fill={P.flame} fontFamily={FONT_HEAD}
                fontWeight={800}>闲不住</text>
              <text x={0} y={146} textAnchor="middle" fontSize={22} fill={P.cream}
                fontFamily={FONT_HEAD} fontWeight={800}>活动冲动拨盘 · 儿童期能看出</text>
            </g>
          </g>
        )}
        {/* 右下：底子也差一截（cue2） */}
        {baseK > 0 && (
          <g opacity={baseK} transform={`translate(0,${(1 - baseK) * 40})`}>
            <rect x={980} y={390} width={850} height={330} rx={20} fill={P.panel}
              stroke={P.flame} strokeWidth={4} />
            <text x={1016} y={446} fill={P.flame} fontSize={28} fontFamily={FONT_HEAD} fontWeight={900}>
              底子也差一截</text>
            {/* 静息代谢 ±10% 双表 */}
            <g transform="translate(1140,540)">
              <circle r={64} fill={P.bgDeep} stroke={P.muted} strokeWidth={5} />
              <line x1={0} y1={0} x2={-30} y2={-40} stroke={P.muted} strokeWidth={7}
                strokeLinecap="round" />
              <circle r={8} fill={P.muted} />
              <text x={0} y={106} textAnchor="middle" fontSize={21} fill={P.muted}
                fontFamily={FONT_HEAD}>怠速偏低</text>
            </g>
            <g transform="translate(1320,540)">
              <circle r={64} fill={P.bgDeep} stroke={P.flame} strokeWidth={5} />
              <line x1={0} y1={0} x2={32} y2={-38} stroke={P.flame} strokeWidth={7}
                strokeLinecap="round" />
              <circle r={8} fill={P.flame} />
              <text x={0} y={106} textAnchor="middle" fontSize={21} fill={P.flame}
                fontFamily={FONT_HEAD} fontWeight={800}>怠速偏高</text>
            </g>
            {/* 棕色脂肪小火炉：肩颈发热 */}
            <g transform="translate(1620,535)">
              <rect x={-34} y={-56} width={68} height={104} rx={30} fill={P.card} />
              <circle cx={0} cy={-86} r={26} fill={P.card} />
              {/* 肩颈棕脂 */}
              <ellipse cx={-30} cy={-58} rx={17} ry={11} fill={P.flame} opacity={glow}
                transform="rotate(24 -30 -58)" />
              <ellipse cx={30} cy={-58} rx={17} ry={11} fill={P.flame} opacity={glow}
                transform="rotate(-24 30 -58)" />
              <Flame x={0} y={-134} s={0.34} />
              <text x={0} y={110} textAnchor="middle" fontSize={21} fill={P.flame}
                fontFamily={FONT_HEAD} fontWeight={800}>肩颈棕脂</text>
            </g>
            <text x={1405} y={702} textAnchor="middle" fontSize={22} fill={P.muted}
              fontFamily={FONT_HEAD}>同体重静息代谢差 ±10% · 棕色脂肪 = 天生多开一台暖气</text>
          </g>
        )}
      </svg>
      <Chip text="不全是意志力问题" x={1350} y={72} at={beat(0) + 40} accent={P.mint} />
      <Stamp text="出厂设置不同" x={600} y={86} at={b1 + 30} color={P.mintD} size={38} rot={-3} />
    </FadeIn>
  );
};

/* ============ S4 把日常调高一档 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const b2 = beat(2);
  const pop = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const listK = clamp01((f - b1 - 5) / 26);
  const sumK = clamp01((f - b2 - 5) / 26);

  const acts = [
    {t: '每小时起身走 2 分钟', s: '起身 > 久坐', col: P.mint},
    {t: '电话能站就站', s: '站姿多烧 50%', col: P.sky},
    {t: '饭后散步 10 分钟', s: '别瘫在沙发上', col: P.amber},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={4} name={sec.name} accent={P.flame} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 顶：一天 24h 的时间条 */}
        <g transform={`translate(90,130) scale(${pop}) translate(-90,-130)`} opacity={clamp01(pop * 2)}>
          <rect x={90} y={130} width={1740} height={190} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={126} y={186} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            别光盯健身房那一小时 · 大头在剩下十五个小时</text>
          {/* 时间条：睡8h + 清醒15h + 健身1h */}
          <g transform="translate(150,230)">
            <rect x={0} y={0} width={540} height={52} rx={12} fill={P.bgDeep} stroke={P.card}
              strokeWidth={3} />
            <text x={270} y={35} textAnchor="middle" fontSize={22} fill={P.muted}
              fontFamily={FONT_HEAD}>睡觉 8h</text>
            <rect x={548} y={0} width={1012} height={52} rx={12} fill={P.mint} opacity={0.9} />
            <text x={1054} y={35} textAnchor="middle" fontSize={23} fill={P.ink}
              fontFamily={FONT_HEAD} fontWeight={900}>醒着的 15h = NEAT 的主场</text>
            <rect x={1568} y={0} width={64} height={52} rx={10} fill={P.flame} />
            <text x={1600} y={-14} textAnchor="middle" fontSize={22} fill={P.flame}
              fontFamily={FONT_HEAD} fontWeight={800}>健身 1h</text>
            <line x1={1600} y1={0} x2={1600} y2={-8} stroke={P.flame} strokeWidth={4} />
          </g>
        </g>
        {/* 中：三条微习惯（cue1） */}
        {listK > 0 && acts.map((a, i) => {
          const grow = clamp01((f - b1 - 8 - i * 12) / 22);
          const sx = 90 + i * 590;
          return (
            <g key={i} opacity={grow} transform={`translate(0,${(1 - grow) * 40})`}>
              <rect x={sx} y={360} width={550} height={300} rx={20} fill={P.panel}
                stroke={a.col} strokeWidth={4} />
              <circle cx={sx + 54} cy={414} r={26} fill={a.col} />
              <text x={sx + 54} y={424} textAnchor="middle" fontSize={26} fill={P.ink}
                fontFamily={FONT_HEAD} fontWeight={900}>{i + 1}</text>
              <text x={sx + 96} y={424} fontSize={28} fill={a.col} fontFamily={FONT_HEAD}
                fontWeight={900}>{a.t}</text>
              {/* 图标区 */}
              <g transform={`translate(${sx + 275},530)`}>
                {i === 0 && (
                  <g>
                    <rect x={-120} y={30} width={90} height={14} rx={7} fill={P.card} />
                    <rect x={-126} y={-40} width={14} height={76} rx={7} fill={P.card} />
                    <Walker x={50} y={44} s={0.44} col={a.col} speed={0.22} />
                    <line x1={-20} y1={-52} x2={-20} y2={-76} stroke={a.col} strokeWidth={6}
                      strokeLinecap="round" />
                    <polygon points="-20,-88 -30,-70 -10,-70" fill={a.col} />
                  </g>
                )}
                {i === 1 && (
                  <g>
                    <Stander x={-10} y={50} s={0.52} col={a.col} fidget={1} />
                    <rect x={34} y={-108} width={24} height={40} rx={7} fill={P.bgDeep}
                      stroke={a.col} strokeWidth={4} />
                    <MotionArcs x={76} y={-88} col={a.col} n={2} />
                  </g>
                )}
                {i === 2 && (
                  <g>
                    <Walker x={30} y={46} s={0.44} col={a.col} speed={0.18} />
                    <g transform="translate(-90,-40)">
                      <path d="M -34 0 A 34 20 0 0 0 34 0 Z" fill={P.cream} />
                      <path d="M -26 2 A 26 14 0 0 0 26 2 Z" fill={P.amber} opacity={0.9} />
                    </g>
                    {[0, 1, 2].map((d) => (
                      <path key={d} d={`M ${-96 + d * 12} ${-62 - d * 6} q 6 -10 0 -18`}
                        stroke={P.muted} strokeWidth={4} fill="none" strokeLinecap="round" />
                    ))}
                  </g>
                )}
              </g>
              <text x={sx + 275} y={628} textAnchor="middle" fontSize={22} fill={P.muted}
                fontFamily={FONT_HEAD}>{a.s}</text>
            </g>
          );
        })}
        {/* 底：累加效应（cue2） */}
        {sumK > 0 && (
          <g opacity={sumK} transform={`translate(0,${(1 - sumK) * 50})`}>
            <rect x={90} y={700} width={1740} height={160} rx={20} fill={P.panel}
              stroke={P.amber} strokeWidth={4} />
            {/* 小硬币叠加大数字 */}
            {[0, 1, 2, 3].map((i) => (
              <g key={i} transform={`translate(${230 + i * 74},${798 - i * 14})`}>
                <circle r={24} fill={P.amber} stroke={P.amberD} strokeWidth={4}
                  opacity={0.5 + i * 0.15} />
                <text x={0} y={8} textAnchor="middle" fontSize={19} fill={P.ink}
                  fontFamily={FONT_HEAD} fontWeight={900}>+{30 + i * 10}</text>
              </g>
            ))}
            <text x={180} y={738} fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>
              单条只值几十大卡</text>
            <text x={600} y={796} fontSize={56} fill={P.amber} fontFamily={FONT_HEAD}
              fontWeight={900}>× 365 天</text>
            <text x={900} y={796} fontSize={34} fill={P.cream} fontFamily={FONT_HEAD} fontWeight={900}>
              ≈ 一年几公斤的差别</text>
            <g transform="translate(1420,780)">
              <rect x={0} y={-44} width={380} height={88} rx={16} fill={P.bgDeep}
                stroke={P.mint} strokeWidth={4} />
              <Check x={36} y={-6} s={0.9} />
              <text x={70} y={-10} fontSize={25} fill={P.mint} fontFamily={FONT_HEAD}
                fontWeight={900}>后台自动运行</text>
              <text x={70} y={26} fontSize={21} fill={P.muted} fontFamily={FONT_HEAD}>
                不用跟饥饿硬扛</text>
            </g>
          </g>
        )}
      </svg>
      <Chip text="微习惯可以训练" x={1360} y={72} at={beat(0) + 30} accent={P.flame} />
      <Stamp text="把小动作调高一档" x={590} y={86} at={b2 + 30} color={P.amberD} size={38} rot={-3} />
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
        <g transform={`translate(960,720) scale(${pop})`} opacity={clamp01(pop)}>
          <rect x={-420} y={0} width={840} height={24} rx={12} fill={P.card} />
          <Sitter x={-160} y={0} s={1.0} col={P.mint} fidget={1} />
          <Flame x={-40} y={-60} s={0.4} phase={1} />
          <Flame x={-230} y={-180} s={0.34} phase={3} />
          <g transform="translate(160,-170)">
            <rect x={-190} y={-52} width={380} height={104} rx={18} fill={P.panel}
              stroke={P.flame} strokeWidth={4} />
            <Flame x={-152} y={-16} s={0.4} />
            <text x={10} y={-12} textAnchor="middle" fontSize={25} fill={P.flame}
              fontFamily={FONT_HEAD} fontWeight={900}>热量从小出口漏掉</text>
            <text x={10} y={26} textAnchor="middle" fontSize={20} fill={P.muted}
              fontFamily={FONT_HEAD}>多开几个出口就行</text>
          </g>
        </g>
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -330}}>
        <div style={{fontSize: 50, fontWeight: 800, color: C.text}}>吃不胖不是魔法 · 是无数小动作在帮你烧</div>
        <div style={{fontSize: 36, color: P.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E244 = ({data = tl}: {data?: any}) => {
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
