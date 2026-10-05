import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, FONT_BODY, clamp01} from '../lib/style';
import {Backdrop, Stamp, SceneTag, Chip, Subtitles, ProgressBar, FadeIn} from '../lib/atoms';
import tl from '../timeline_248.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：深海蓝 × 肌纤维珊瑚红 × 肌核明黄 × DNA紫（纯色平涂） ============ */
const P = {
  bg: '#1c2a47',
  bgLite: '#223255',
  bgDeep: '#121d33',
  panel: '#162238',
  card: '#2e4270',
  ink: '#0d1524',
  cream: '#f4f7ff',
  muted: '#9fb2d8',
  muscle: '#ff7a6b',
  muscleD: '#d95a50',
  nuc: '#ffd166',
  nucD: '#e8b04a',
  dna: '#b18cff',
  dnaD: '#8a63e0',
  mint: '#5fe3b3',
  mintD: '#37b58c',
  sky: '#6ec6ff',
  pink: '#ff9ec4',
};

/* ---------- 小叉 / 小勾 ---------- */
const MarkX = ({x, y, s = 1, col = P.muscleD}: {x: number; y: number; s?: number; col?: string}) => (
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

/* ---------- 圆角标签牌 ---------- */
const TagPill = ({x, y, t, col, s = 1, w = 220, fs = 30}: {
  x: number; y: number; t: string; col: string; s?: number; w?: number; fs?: number;
}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-w / 2} y={-30} width={w} height={60} rx={16} fill={P.panel} stroke={col} strokeWidth={4} />
    <text x={0} y={11} textAnchor="middle" fontSize={fs} fill={col}
      fontFamily={FONT_HEAD} fontWeight={800}>{t}</text>
  </g>
);

/* ---------- 肌核（椭圆带核点） ---------- */
const Nuc = ({x, y, s = 1, on = true}: {x: number; y: number; s?: number; on?: boolean}) => {
  const f = useCurrentFrame();
  const tw = on ? 0.82 + Math.sin(f * 0.22 + x * 0.05) * 0.18 : 0.35;
  return (
    <g transform={`translate(${x},${y}) scale(${s})`} opacity={tw}>
      <ellipse rx={16} ry={11} fill={on ? P.nuc : P.muted} stroke={on ? P.nucD : P.card} strokeWidth={3} />
      <circle r={4} fill={P.ink} />
    </g>
  );
};

/* ---------- 肌纤维：长胶囊 + 横纹 + 肌核 ---------- */
const Fiber = ({x, y, w = 520, h = 110, n = 8, s = 1, lit = true, dead = []}: {
  x: number; y: number; w?: number; h?: number; n?: number; s?: number; lit?: boolean; dead?: number[];
}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-w / 2} y={-h / 2} width={w} height={h} rx={h / 2} fill={P.muscle}
      stroke={P.muscleD} strokeWidth={5} />
    {Array.from({length: Math.max(1, Math.floor(w / 54) - 1)}).map((_, i) => (
      <line key={i} x1={-w / 2 + 34 + i * 54} y1={-h / 2 + 15} x2={-w / 2 + 34 + i * 54}
        y2={h / 2 - 15} stroke={P.muscleD} strokeWidth={4} opacity={0.5} />
    ))}
    {Array.from({length: n}).map((_, i) => {
      const nx = -w / 2 + (w / (n + 1)) * (i + 1);
      const ny = i % 2 === 0 ? -h * 0.24 : h * 0.24;
      return <Nuc key={i} x={nx} y={ny} s={0.9} on={lit && !dead.includes(i)} />;
    })}
  </g>
);

/* ---------- 卫星细胞（肌肉干细胞） ---------- */
const SatCell = ({x, y, s = 1, rot = 0}: {x: number; y: number; s?: number; rot?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`}>
    <ellipse rx={32} ry={22} fill={P.mint} opacity={0.92} />
    <ellipse cx={30} cy={-14} rx={13} ry={9} fill={P.mint} opacity={0.7} />
    <ellipse cx={-26} cy={12} rx={11} ry={8} fill={P.mint} opacity={0.7} />
    <circle cx={-4} cy={0} r={9} fill={P.nuc} />
  </g>
);

/* ---------- 大脑 ---------- */
const Brain = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <path d="M -70 -10 Q -78 -68 -20 -76 Q -6 -96 26 -84 Q 74 -80 72 -30 Q 92 -8 70 16 Q 60 46 16 44 Q -40 50 -62 22 Q -76 10 -70 -10 Z"
      fill={P.pink} opacity={0.95} />
    <path d="M -40 -40 Q -20 -52 -2 -40 M -34 -6 Q -10 -18 14 -8 M -20 24 Q 4 14 30 22"
      stroke={P.bgDeep} strokeWidth={6} fill="none" strokeLinecap="round" />
    <line x1={-14} y1={-78} x2={-14} y2={-98} stroke={P.dna} strokeWidth={6} strokeLinecap="round" />
    <circle cx={-14} cy={-104} r={8} fill={P.dna} />
  </g>
);

/* ---------- 自行车 ---------- */
const Bicycle = ({x, y, s = 1, col = P.sky}: {x: number; y: number; s?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} stroke={col} strokeWidth={6}
    fill="none" strokeLinecap="round" strokeLinejoin="round">
    <circle cx={-52} cy={18} r={32} />
    <circle cx={52} cy={18} r={32} />
    <path d="M -52 18 L -24 -22 L 34 -22 L 52 18 M -24 -22 L -6 18 L -52 18 M -6 18 L 34 -22" />
    <line x1={-24} y1={-22} x2={-28} y2={-36} />
    <line x1={-40} y1={-36} x2={-16} y2={-36} />
    <line x1={34} y1={-22} x2={40} y2={-38} />
    <line x1={40} y1={-38} x2={58} y2={-34} />
  </g>
);

/* ---------- 钢琴键盘 ---------- */
const Piano = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-72} y={-26} width={144} height={52} rx={8} fill={P.cream} stroke={P.ink} strokeWidth={4} />
    {[0, 1, 2, 3, 4].map((i) => (
      <rect key={i} x={-58 + i * 28} y={-26} width={14} height={30} rx={3} fill={P.ink} />
    ))}
  </g>
);

/* ---------- 举杠铃小人（脚底 y=0） ---------- */
const Lifter = ({x, y, s = 1, col = P.sky, press = 0}: {
  x: number; y: number; s?: number; col?: string; press?: number;
}) => {
  const f = useCurrentFrame();
  const barY = -228 + Math.sin(f * 0.22) * 12 * press;
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      <ellipse cx={0} cy={6} rx={72} ry={10} fill={P.ink} opacity={0.3} />
      <line x1={-14} y1={-72} x2={-20} y2={-4} stroke={P.ink} strokeWidth={22} strokeLinecap="round" />
      <line x1={14} y1={-72} x2={22} y2={-4} stroke={P.ink} strokeWidth={22} strokeLinecap="round" />
      <ellipse cx={-22} cy={-2} rx={20} ry={9} fill={P.cream} />
      <ellipse cx={24} cy={-2} rx={20} ry={9} fill={P.cream} />
      <rect x={-30} y={-166} width={60} height={98} rx={28} fill={col} />
      {/* 手臂向上举 */}
      <line x1={-24} y1={-144} x2={-56} y2={barY + 6} stroke={C.skin} strokeWidth={15}
        strokeLinecap="round" />
      <line x1={24} y1={-144} x2={56} y2={barY + 6} stroke={C.skin} strokeWidth={15}
        strokeLinecap="round" />
      {/* 杠铃 */}
      <line x1={-110} y1={barY} x2={110} y2={barY} stroke={P.cream} strokeWidth={9}
        strokeLinecap="round" />
      {[-1, 1].map((sgn) => (
        <g key={sgn}>
          <rect x={sgn * 96 - 9} y={barY - 30} width={18} height={60} rx={8} fill={P.muscleD} />
          <rect x={sgn * 76 - 7} y={barY - 22} width={14} height={44} rx={7} fill={P.muscle} />
        </g>
      ))}
      {/* 头 */}
      <g transform="translate(0,-194)">
        <circle r={30} fill={C.skin} />
        <path d="M -29 -6 Q -20 -38 8 -36 Q 32 -33 29 -4 Q 0 -22 -29 -6 Z" fill={P.ink} />
        <circle cx={11} cy={0} r={4.2} fill={P.ink} />
      </g>
    </g>
  );
};

/* ---------- DNA 双链 + 甲基化标记 ---------- */
const DNA = ({x, y, s = 1, w = 640, methyl = true}: {
  x: number; y: number; s?: number; w?: number; methyl?: boolean;
}) => {
  const pts = (ph: number) => Array.from({length: 33}).map((_, i) => {
    const px = -w / 2 + i * (w / 32);
    return `${px},${Math.sin((i / 32) * Math.PI * 4 + ph) * 30}`;
  }).join(' ');
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      {[0, 4, 8, 12, 16, 20, 24, 28].map((i) => {
        const px = -w / 2 + i * (w / 32);
        const py = Math.sin((i / 32) * Math.PI * 4) * 30;
        return <line key={i} x1={px} y1={py} x2={px} y2={-py} stroke={P.card} strokeWidth={5} />;
      })}
      <polyline points={pts(0)} stroke={P.dna} strokeWidth={7} fill="none" strokeLinecap="round" />
      <polyline points={pts(Math.PI)} stroke={P.sky} strokeWidth={7} fill="none" strokeLinecap="round" />
      {methyl && [3, 11, 19, 27].map((i) => {
        const px = -w / 2 + i * (w / 32);
        const py = Math.sin((i / 32) * Math.PI * 4) * 30;
        return (
          <g key={i} transform={`translate(${px},${py - 26}) rotate(45)`}>
            <rect x={-11} y={-11} width={22} height={22} rx={5} fill={P.dnaD}
              stroke={P.dna} strokeWidth={3} />
          </g>
        );
      })}
    </g>
  );
};

/* ---------- 肌核账户保险柜 ---------- */
const Vault = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-140} y={-110} width={280} height={220} rx={24} fill={P.card}
      stroke={P.nuc} strokeWidth={6} />
    {/* 顶部投币槽 */}
    <rect x={-52} y={-118} width={104} height={16} rx={8} fill={P.bgDeep}
      stroke={P.nuc} strokeWidth={4} />
    {/* 圆形库门 */}
    <circle r={64} fill={P.bgDeep} stroke={P.nuc} strokeWidth={6} />
    {[0, 60, 120].map((a) => (
      <line key={a} x1={Math.sin(a * Math.PI / 180) * 18} y1={-Math.cos(a * Math.PI / 180) * 18}
        x2={Math.sin(a * Math.PI / 180) * 54} y2={-Math.cos(a * Math.PI / 180) * 54}
        stroke={P.nuc} strokeWidth={8} strokeLinecap="round" />
    ))}
    {[0, 60, 120].map((a) => (
      <line key={`b${a}`} x1={-Math.sin(a * Math.PI / 180) * 18} y1={Math.cos(a * Math.PI / 180) * 18}
        x2={-Math.sin(a * Math.PI / 180) * 54} y2={Math.cos(a * Math.PI / 180) * 54}
        stroke={P.nuc} strokeWidth={8} strokeLinecap="round" />
    ))}
    <circle r={14} fill={P.nuc} stroke={P.ink} strokeWidth={3} />
  </g>
);

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
        <g transform={`translate(960,440) scale(${pA}) translate(-960,-440)`} opacity={clamp01(pA)}>
          {/* 主图标：一条装着肌核的肌纤维 */}
          <Fiber x={960} y={430} w={720} h={150} n={9} s={1} />
          {/* 左侧被划掉的：大脑技能记忆 */}
          <Brain x={440} y={440} s={0.9} />
          <MarkX x={440} y={440} s={2.4} col={P.muscleD} />
          {/* 右侧举铃小人 */}
          <Lifter x={1460} y={640} s={0.9} col={P.mint} press={1} />
        </g>
        <g transform={`translate(390,240) scale(${pB})`} opacity={clamp01(pB)}>
          <TagPill x={0} y={0} t="弹琴骑车归大脑" col={P.pink} w={270} fs={27} />
        </g>
        <g transform={`translate(1530,240) scale(${pB})`} opacity={clamp01(pB)}>
          <TagPill x={0} y={0} t="真记忆在肌核里" col={P.nuc} w={280} fs={27} />
        </g>
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 470}}>
          <div style={{color: P.nuc, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 24}}>
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

/* ============ S0 两种肌肉记忆 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const b2 = beat(2);
  const popL = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const musK = clamp01((f - b1 - 5) / 26);
  const sumK = clamp01((f - b2 - 5) / 26);

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={0} name={sec.name} accent={P.pink} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：记在脑子里的技能记忆 */}
        <g transform={`translate(90,160) scale(${popL}) translate(-90,-160)`} opacity={clamp01(popL * 2)}>
          <rect x={90} y={160} width={840} height={520} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <TagPill x={240} y={220} t="被用烂的那种" col={P.pink} w={230} fs={27} />
          <text x={395} y={232} fontSize={24} fill={P.cream} fontFamily={FONT_HEAD}
            fontWeight={800}>技能记忆 · 肌肉只是执行器</text>
          <Brain x={505} y={400} s={1.35} />
          <text x={505} y={520} textAnchor="middle" fontSize={24} fill={P.muted}
            fontFamily={FONT_HEAD}>存在大脑皮层·小脑</text>
          {/* 技能图标排 */}
          <Bicycle x={330} y={590} s={0.85} />
          <Piano x={560} y={590} s={0.95} />
          <g transform="translate(750,585)" stroke={P.muted} strokeWidth={6}
            fill="none" strokeLinecap="round">
            <path d="M -50 0 Q -25 -26 0 0 Q 25 -26 50 0" />
            <path d="M -40 18 Q -20 -2 0 18 Q 20 -2 40 18" />
            <circle cx={58} cy={-30} r={14} fill={C.skin} stroke="none" />
            <path d="M 58 -16 L 74 -2 M 58 -16 L 44 0" />
          </g>
          <text x={330} y={660} textAnchor="middle" fontSize={21} fill={P.sky}
            fontFamily={FONT_HEAD}>骑车</text>
          <text x={560} y={660} textAnchor="middle" fontSize={21} fill={P.cream}
            fontFamily={FONT_HEAD}>弹琴</text>
          <text x={750} y={660} textAnchor="middle" fontSize={21} fill={P.muted}
            fontFamily={FONT_HEAD}>游泳</text>
        </g>
        {/* 右：长在肌肉里的真记忆（cue1） */}
        {musK > 0 && (
          <g opacity={musK} transform={`translate(0,${(1 - musK) * 40})`}>
            <rect x={990} y={160} width={840} height={520} rx={20} fill={P.panel}
              stroke={P.nuc} strokeWidth={4} />
            <TagPill x={1120} y={220} t="真的那种" col={P.nuc} w={190} fs={27} />
            <text x={1240} y={232} fontSize={24} fill={P.cream}
              fontFamily={FONT_HEAD} fontWeight={800}>长在肌纤维里 · 载体是肌核</text>
            {/* 萎缩→复练对比 */}
            <g transform="translate(0,0)">
              <Fiber x={1290} y={390} w={340} h={66} n={6} s={1} />
              <text x={1290} y={462} textAnchor="middle" fontSize={22} fill={P.muted}
                fontFamily={FONT_HEAD}>练过 · 瘪回去的纤维</text>
              {/* 箭头：重新练 */}
              <line x1={1490} y1={390} x2={1580} y2={390} stroke={P.mint} strokeWidth={8}
                strokeLinecap="round" />
              <polygon points="1596,390 1572,376 1572,404" fill={P.mint} />
              <text x={1543} y={370} textAnchor="middle" fontSize={20} fill={P.mint}
                fontFamily={FONT_HEAD} fontWeight={800}>重新练</text>
              <Fiber x={1710} y={390} w={180} h={110} n={6} s={1} />
              <text x={1710} y={462} textAnchor="middle" fontSize={22} fill={P.nuc}
                fontFamily={FONT_HEAD} fontWeight={800}>长得更快</text>
            </g>
            {/* 速度条对比 */}
            <g transform="translate(1060,540)">
              <text x={0} y={20} fontSize={23} fill={P.muted} fontFamily={FONT_HEAD}>零基础长肌肉</text>
              <rect x={0} y={34} width={300} height={34} rx={17} fill={P.bgDeep} />
              <rect x={0} y={34} width={110} height={34} rx={17} fill={P.muted} />
              <text x={430} y={20} fontSize={23} fill={P.nuc} fontFamily={FONT_HEAD}
                fontWeight={800}>练过的再练</text>
              <rect x={430} y={34} width={300} height={34} rx={17} fill={P.bgDeep} />
              <rect x={430} y={34} width={300 * musK} height={34} rx={17} fill={P.nuc} />
            </g>
          </g>
        )}
        {/* 底条：有细胞学证据（cue2） */}
        {sumK > 0 && (
          <g opacity={sumK} transform={`translate(0,${(1 - sumK) * 50})`}>
            <rect x={90} y={740} width={1740} height={128} rx={20} fill={P.panel}
              stroke={P.mint} strokeWidth={4} />
            <Check x={160} y={804} s={1.2} />
            <text x={215} y={792} fontSize={30} fill={P.cream} fontFamily={FONT_HEAD} fontWeight={900}>
              不是经验之谈</text>
            <text x={215} y={836} fontSize={25} fill={P.muted} fontFamily={FONT_HEAD}>
              是能在显微镜下数出来的细胞事实</text>
            <g transform="translate(900,804)">
              <Fiber x={0} y={0} w={300} h={60} n={5} s={0.9} />
            </g>
            <text x={1310} y={818} fontSize={32} fill={P.nuc} fontFamily={FONT_HEAD}
              fontWeight={900}>肌核 = 肌肉里留下的记忆</text>
          </g>
        )}
      </svg>
      <Chip text="技能记忆 vs 肌核记忆" x={1330} y={72} at={beat(0) + 30} accent={P.nuc} />
      <Stamp text="练过真的不一样" x={610} y={86} at={b1 + 40} color={P.muscleD} size={38} rot={-3} />
    </FadeIn>
  );
};

/* ============ S1 肌核是钉子户 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const pop = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const vsK = clamp01((f - b1 - 5) / 26);
  const tw = 0.75 + Math.sin(f * 0.25) * 0.25;

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={1} name={sec.name} accent={P.nuc} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 顶：肌纤维解剖 */}
        <g transform={`translate(90,140) scale(${pop}) translate(-90,-140)`} opacity={clamp01(pop * 2)}>
          <rect x={90} y={140} width={1740} height={300} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={130} y={196} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            肌肉细胞是个另类 · 一根纤维就是一条几厘米长的多核细胞</text>
          {/* 大纤维 */}
          <Fiber x={920} y={300} w={1080} h={130} n={11} />
          {/* 长度标尺 */}
          <line x1={380} y1={390} x2={1460} y2={390} stroke={P.sky} strokeWidth={4}
            strokeDasharray="12 9" />
          <polygon points="380,390 402,381 402,399" fill={P.sky} />
          <polygon points="1460,390 1438,381 1438,399" fill={P.sky} />
          <text x={920} y={424} textAnchor="middle" fontSize={22} fill={P.sky}
            fontFamily={FONT_HEAD}>能有几厘米长</text>
          {/* 肌核标注 */}
          <line x1={1330} y1={262} x2={1560} y2={196} stroke={P.nuc} strokeWidth={4} />
          <circle cx={1330} cy={262} r={7} fill={P.nuc} />
          <text x={1578} y={190} fontSize={25} fill={P.nuc} fontFamily={FONT_HEAD} fontWeight={900}>
            肌核 ×成百上千</text>
          <text x={1578} y={224} fontSize={21} fill={P.muted} fontFamily={FONT_HEAD}>
            每个核管一小段产能</text>
          {/* 卫星细胞正在融合 */}
          <SatCell x={300} y={330} s={0.9} rot={-15} />
          <line x1={352} y1={316} x2={392} y2={306} stroke={P.mint} strokeWidth={5}
            strokeDasharray="8 7" strokeLinecap="round" />
          <text x={140} y={424} fontSize={21} fill={P.mint} fontFamily={FONT_HEAD} fontWeight={800}>
            卫星细胞融合送新核</text>
        </g>
        {/* 底左：旧观点（cue1） */}
        {vsK > 0 && (
          <g opacity={vsK} transform={`translate(0,${(1 - vsK) * 40})`}>
            <rect x={90} y={480} width={850} height={330} rx={20} fill={P.panel}
              stroke={P.muted} strokeWidth={4} />
            <text x={126} y={536} fontSize={26} fill={P.muted} fontFamily={FONT_HEAD} fontWeight={900}>
              旧观点 · 萎缩 = 核也清零</text>
            {/* 瘪纤维 + 核被清走 */}
            <Fiber x={430} y={640} w={560} h={64} n={7} lit={false} dead={[0, 1, 5, 6]} />
            {[0, 1, 5, 6].map((i) => {
              const nx = 430 - 280 + (560 / 8) * (i + 1);
              return (
                <g key={i} opacity={tw}>
                  <MarkX x={nx} y={600} s={1} col={P.muscleD} />
                </g>
              );
            })}
            {/* 核被扫走的箭头 */}
            {[0, 1].map((i) => (
              <g key={i} transform={`translate(${690 + i * 60},${590 - i * 30})`}>
                <line x1={0} y1={0} x2={44} y2={-26} stroke={P.muted} strokeWidth={5}
                  strokeLinecap="round" opacity={0.6} />
                <polygon points="52,-31 36,-24 44,-12" fill={P.muted} opacity={0.6} />
              </g>
            ))}
            <rect x={660} y={726} width={240} height={56} rx={14} fill={P.bgDeep}
              stroke={P.card} strokeWidth={3.5} />
            <text x={780} y={764} textAnchor="middle" fontSize={24} fill={P.muted}
              fontFamily={FONT_HEAD}>回到出厂设置</text>
          </g>
        )}
        {/* 底右：新发现 */}
        {vsK > 0 && (
          <g opacity={vsK} transform={`translate(0,${(1 - vsK) * 40})`}>
            <rect x={980} y={480} width={850} height={330} rx={20} fill={P.panel}
              stroke={P.nuc} strokeWidth={4} />
            <text x={1016} y={536} fontSize={26} fill={P.nuc} fontFamily={FONT_HEAD} fontWeight={900}>
              新发现 · 块头瘪了核留下</text>
            <Fiber x={1320} y={640} w={560} h={64} n={7} />
            {/* 钉子户挂牌 */}
            <g transform="translate(1660,612)">
              <rect x={-96} y={-30} width={192} height={60} rx={14} fill={P.nuc} />
              <text x={0} y={11} textAnchor="middle" fontSize={27} fill={P.ink}
                fontFamily={FONT_HEAD} fontWeight={900}>钉子户</text>
            </g>
            <Check x={1040} y={750} s={0.95} />
            <text x={1080} y={742} fontSize={25} fill={P.mint} fontFamily={FONT_HEAD} fontWeight={800}>
              核档案还在</text>
            <text x={1080} y={778} fontSize={21} fill={P.muted} fontFamily={FONT_HEAD}>
              只等重新开工的信号</text>
          </g>
        )}
      </svg>
      <Chip text="萎缩 ≠ 清零" x={1420} y={72} at={beat(0) + 30} accent={P.nuc} />
      <Stamp text="肌核是钉子户" x={620} y={86} at={b1 + 30} color={P.nucD} size={38} rot={-3} />
    </FadeIn>
  );
};

/* ============ S2 现成的厂房 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const b2 = beat(2);
  const pop = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const buildK = clamp01((f - b1 - 5) / 26);
  const methK = clamp01((f - b2 - 5) / 26);
  const flow = (f * 2.2) % 140;

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={2} name={sec.name} accent={P.mint} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：没练过 从零铺基建 */}
        <g transform={`translate(90,140) scale(${pop}) translate(-90,-140)`} opacity={clamp01(pop * 2)}>
          <rect x={90} y={140} width={850} height={420} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <TagPill x={255} y={204} t="没练过" col={P.muted} w={170} fs={27} />
          <text x={380} y={208} fontSize={26} fill={P.muted} fontFamily={FONT_HEAD} fontWeight={800}>
            先招募卫星细胞 · 从零造核</text>
          {/* 纤维 + 慢慢靠近的卫星细胞 */}
          <Fiber x={480} y={400} w={500} h={104} n={4} lit={false} />
          <SatCell x={290 - buildK * 40} y={540} s={0.85} rot={-20} />
          <SatCell x={480 - buildK * 20} y={560} s={0.75} rot={10} />
          <SatCell x={660 - buildK * 50} y={536} s={0.8} rot={-8} />
          <line x1={300} y1={510} x2={360} y2={478} stroke={P.mint} strokeWidth={5}
            strokeDasharray="8 8" strokeLinecap="round" />
          <text x={705} y={536} fontSize={21} fill={P.mint} fontFamily={FONT_HEAD}>
            卫星细胞慢吞吞报到</text>
          <g transform="translate(170,452)" opacity={0.9}>
            <rect x={0} y={0} width={36} height={26} rx={6} fill={P.muscleD} />
            <rect x={44} y={0} width={36} height={26} rx={6} fill={P.muscleD} opacity={0.6} />
            <rect x={22} y={-30} width={36} height={26} rx={6} fill={P.muscleD} opacity={0.35} />
          </g>
          <text x={300} y={478} fontSize={21} fill={P.muted} fontFamily={FONT_HEAD}>
            铺基建占掉大半时间</text>
        </g>
        {/* 右：练过 直接开足马力（cue1） */}
        <g transform={`translate(980,140) scale(${pop}) translate(-980,-140)`} opacity={clamp01(pop * 2)}>
          <rect x={980} y={140} width={850} height={420} rx={20} fill={P.panel}
            stroke={P.mint} strokeWidth={4} />
          <TagPill x={1110} y={204} t="练过" col={P.nuc} w={150} fs={27} />
          <text x={1210} y={208} fontSize={26} fill={P.nuc} fontFamily={FONT_HEAD} fontWeight={800}>
            基建早铺好 · 直接下增产指令</text>
          <Fiber x={1370} y={400} w={500} h={118} n={8} />
          {/* 增产产出：蛋白砖块流出 */}
          {[0, 1, 2].map((i) => {
            const px = 1650 + ((flow + i * 46) % 140);
            return (
              <rect key={i} x={px} y={388 + (i % 2) * 16} width={30} height={22} rx={5}
                fill={P.nuc} opacity={clamp01(1.2 - ((flow + i * 46) % 140) / 140)} />
            );
          })}
          <polygon points="1800,399 1782,389 1782,409" fill={P.mint} />
          <g transform="translate(1050,460)" opacity={0.95}>
            <rect x={0} y={0} width={36} height={26} rx={6} fill={P.nuc} />
            <rect x={44} y={0} width={36} height={26} rx={6} fill={P.nuc} />
            <rect x={88} y={0} width={36} height={26} rx={6} fill={P.nuc} />
            <rect x={22} y={-30} width={36} height={26} rx={6} fill={P.nuc} />
            <rect x={66} y={-30} width={36} height={26} rx={6} fill={P.nuc} />
          </g>
          <text x={1200} y={490} fontSize={21} fill={P.nuc} fontFamily={FONT_HEAD}>
            现有肌核接到指令</text>
          <text x={1450} y={530} textAnchor="middle" fontSize={22} fill={P.cream}
            fontFamily={FONT_HEAD} fontWeight={800}>现成厂房重启 · 蛋白质产量拉满</text>
        </g>
        {/* 底：甲基化记号（cue2） */}
        {methK > 0 && (
          <g opacity={methK} transform={`translate(0,${(1 - methK) * 50})`}>
            <rect x={90} y={600} width={1740} height={270} rx={20} fill={P.panel}
              stroke={P.dna} strokeWidth={4} />
            <text x={130} y={656} fontSize={26} fill={P.dna} fontFamily={FONT_HEAD} fontWeight={900}>
              分子层面还留了一手 · DNA 甲基化记号</text>
            <DNA x={700} y={740} w={760} methyl={true} />
            <g transform="translate(1330,700)">
              <rect x={0} y={0} width={440} height={120} rx={16} fill={P.bgDeep}
                stroke={P.dna} strokeWidth={4} />
              <g transform="translate(44,42) rotate(45)">
                <rect x={-9} y={-9} width={18} height={18} rx={4} fill={P.dnaD}
                  stroke={P.dna} strokeWidth={3} />
              </g>
              <text x={76} y={44} fontSize={24} fill={P.dna} fontFamily={FONT_HEAD} fontWeight={800}>
                甲基化标记</text>
              <text x={76} y={86} fontSize={21} fill={P.muted} fontFamily={FONT_HEAD}>
                把生长基因的开关拨得更松</text>
            </g>
            <text x={700} y={830} textAnchor="middle" fontSize={22} fill={P.muted}
              fontFamily={FONT_HEAD}>下次训练刺激一来 · 基因响应更灵敏</text>
          </g>
        )}
      </svg>
      <Chip text="产能备案还在" x={1390} y={72} at={beat(0) + 30} accent={P.mint} />
      <Stamp text="重建是快装" x={630} y={86} at={b1 + 30} color={P.mintD} size={38} rot={-3} />
    </FadeIn>
  );
};

/* ============ S3 帮的是长回来 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const b2 = beat(2);
  const b3 = beat(3);
  const pop = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const tlK = clamp01((f - b1 - 5) / 30);
  const helpK = clamp01((f - b2 - 5) / 26);
  const skillK = clamp01((f - b3 - 5) / 26);

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={3} name={sec.name} accent={P.sky} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 顶：记忆能留多久的时间轴 */}
        <g transform={`translate(90,140) scale(${pop}) translate(-90,-140)`} opacity={clamp01(pop * 2)}>
          <rect x={90} y={140} width={1740} height={300} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={130} y={196} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            这份记忆能存多久 · 停训之后肌核还在岗</text>
          {/* 时间轴 */}
          <line x1={260} y1={330} x2={1660} y2={330} stroke={P.card} strokeWidth={10}
            strokeLinecap="round" />
          <line x1={260} y1={330} x2={260 + 1400 * tlK} y2={330} stroke={P.nuc}
            strokeWidth={10} strokeLinecap="round" />
          {/* 节点 */}
          {[
            {x: 340, t: '停训开始', sub: '块头慢慢瘪', col: P.muted},
            {x: 800, t: '大半年', sub: '动物实验：核还在', col: P.sky},
            {x: 1230, t: '几年', sub: '人类：优势仍在', col: P.nuc},
            {x: 1600, t: '甚至终身', sub: '部分研究支持', col: P.mint},
          ].map((m, i) => (
            <g key={i} transform={`translate(${m.x},330)`} opacity={clamp01(tlK * 4 - i * 0.7)}>
              <circle r={13} fill={m.col} stroke={P.bgDeep} strokeWidth={4} />
              <Nuc x={0} y={-52} s={1.05} on={i > 0} />
              <text x={0} y={46} textAnchor="middle" fontSize={25} fill={m.col}
                fontFamily={FONT_HEAD} fontWeight={900}>{m.t}</text>
              <text x={0} y={78} textAnchor="middle" fontSize={19} fill={P.muted}
                fontFamily={FONT_HEAD}>{m.sub}</text>
            </g>
          ))}
        </g>
        {/* 底左：帮的是块头和力量（cue2） */}
        {helpK > 0 && (
          <g opacity={helpK} transform={`translate(0,${(1 - helpK) * 40})`}>
            <rect x={90} y={480} width={850} height={330} rx={20} fill={P.panel}
              stroke={P.mint} strokeWidth={4} />
            <text x={126} y={536} fontSize={26} fill={P.mint} fontFamily={FONT_HEAD} fontWeight={900}>
              帮得上 · 块头和力量长回来快</text>
            <Lifter x={300} y={760} s={0.78} col={P.mint} press={1} />
            <g transform="translate(470,590)">
              <Check x={0} y={0} s={0.9} />
              <text x={36} y={10} fontSize={25} fill={P.cream} fontFamily={FONT_HEAD} fontWeight={800}>
                重建速度快一截</text>
              <Check x={0} y={62} s={0.9} />
              <text x={36} y={72} fontSize={25} fill={P.cream} fontFamily={FONT_HEAD} fontWeight={800}>
                复出运动员的隐形资产</text>
            </g>
            <rect x={470} y={700} width={380} height={72} rx={16} fill={P.bgDeep}
              stroke={P.card} strokeWidth={3.5} />
            <text x={660} y={732} textAnchor="middle" fontSize={23} fill={P.muted}
              fontFamily={FONT_HEAD}>注意 · 不是永远不掉</text>
            <text x={660} y={762} textAnchor="middle" fontSize={21} fill={P.muted}
              fontFamily={FONT_HEAD}>该退还是退 · 只是重建更省</text>
          </g>
        )}
        {/* 底右：技能归大脑（cue3） */}
        {skillK > 0 && (
          <g opacity={skillK} transform={`translate(0,${(1 - skillK) * 40})`}>
            <rect x={980} y={480} width={850} height={330} rx={20} fill={P.panel}
              stroke={P.pink} strokeWidth={4} />
            <text x={1016} y={536} fontSize={26} fill={P.pink} fontFamily={FONT_HEAD} fontWeight={900}>
              帮不上 · 技能耐力得另练</text>
            <Brain x={1180} y={640} s={0.95} />
            <Bicycle x={1360} y={660} s={0.7} col={P.muted} />
            <MarkX x={1360} y={660} s={1.6} col={P.muscleD} />
            <g transform="translate(1490,590)">
              <MarkX x={0} y={0} s={0.9} col={P.pink} />
              <text x={36} y={10} fontSize={25} fill={P.cream} fontFamily={FONT_HEAD} fontWeight={800}>
                发力时序·动作轨迹</text>
              <MarkX x={0} y={62} s={0.9} col={P.pink} />
              <text x={36} y={72} fontSize={25} fill={P.cream} fontFamily={FONT_HEAD} fontWeight={800}>
                心肺耐力的记忆</text>
            </g>
            <text x={1400} y={772} textAnchor="middle" fontSize={22} fill={P.muted}
              fontFamily={FONT_HEAD}>神经熟练度得重新刷 · 肌核管不了</text>
          </g>
        )}
      </svg>
      <Chip text="记很多年甚至终身" x={1360} y={72} at={beat(0) + 30} accent={P.sky} />
      <Stamp text="只管长回来" x={630} y={86} at={b2 + 20} color={P.muscleD} size={38} rot={-3} />
    </FadeIn>
  );
};

/* ============ S4 年轻时练的是存款 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const b2 = beat(2);
  const pop = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const optK = clamp01((f - b1 - 5) / 26);
  const sumK = clamp01((f - b2 - 5) / 26);
  const drop = (f * 2.4) % 120;

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={4} name={sec.name} accent={P.nuc} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：肌核账户 */}
        <g transform={`translate(90,140) scale(${pop}) translate(-90,-140)`} opacity={clamp01(pop * 2)}>
          <rect x={90} y={140} width={900} height={560} rx={20} fill={P.panel} stroke={P.nuc} strokeWidth={4} />
          <TagPill x={250} y={204} t="肌核账户" col={P.nuc} w={210} fs={29} />
          <text x={390} y={208} fontSize={24} fill={P.muted} fontFamily={FONT_HEAD}>
            年轻时认真练 · 就是存本钱</text>
          {/* 举铃小人往账户里存核 */}
          <Lifter x={270} y={640} s={0.9} col={P.sky} press={1} />
          <text x={270} y={672} textAnchor="middle" fontSize={21} fill={P.sky}
            fontFamily={FONT_HEAD}>20岁 · 系统练一两年</text>
          {/* 下落的肌核硬币 */}
          {[0, 1, 2].map((i) => {
            const dy = ((drop + i * 40) % 120);
            const py = 300 + dy;
            return (
              <g key={i} opacity={clamp01(1.15 - dy / 120)}>
                <Nuc x={640} y={py} s={0.85} on={true} />
              </g>
            );
          })}
          <Vault x={640} y={520} s={0.95} />
          <text x={640} y={668} textAnchor="middle" fontSize={23} fill={P.nuc}
            fontFamily={FONT_HEAD} fontWeight={900}>账户几十年不清零</text>
          {/* 时间箭头 */}
          <line x1={400} y1={330} x2={520} y2={330} stroke={P.muted} strokeWidth={5}
            strokeDasharray="9 8" strokeLinecap="round" />
          <polygon points="536,330 516,321 516,339" fill={P.muted} />
          <text x={460} y={310} textAnchor="middle" fontSize={20} fill={P.muted}
            fontFamily={FONT_HEAD}>每次训练存一点</text>
          {/* 账本卡 */}
          <g transform="translate(160,200)">
            <rect x={620} y={-40} width={180} height={120} rx={14} fill={P.bgDeep}
              stroke={P.card} strokeWidth={3.5} />
            <text x={710} y={0} textAnchor="middle" fontSize={22} fill={P.muted}
              fontFamily={FONT_HEAD}>肌核余额</text>
            <text x={710} y={52} textAnchor="middle" fontSize={40} fill={P.nuc}
              fontFamily={FONT_HEAD} fontWeight={900}>≠ 0</text>
          </g>
        </g>
        {/* 右：两个选择（cue1） */}
        {optK > 0 && (
          <g opacity={optK} transform={`translate(0,${(1 - optK) * 40})`}>
            <rect x={1020} y={140} width={810} height={560} rx={20} fill={P.panel}
              stroke={P.card} strokeWidth={4} />
            <text x={1060} y={200} fontSize={26} fill={P.cream} fontFamily={FONT_HEAD} fontWeight={900}>
              实在忙怎么办</text>
            {/* 选项A */}
            <g transform="translate(1060,240)">
              <rect x={0} y={0} width={730} height={140} rx={18} fill={P.bgDeep}
                stroke={P.card} strokeWidth={3.5} />
              <MarkX x={52} y={70} s={1.3} col={P.muscleD} />
              <text x={96} y={60} fontSize={26} fill={P.muted} fontFamily={FONT_HEAD} fontWeight={800}>
                彻底停练</text>
              <text x={96} y={102} fontSize={21} fill={P.muted} fontFamily={FONT_HEAD}>
                核档案还在 · 但体能白白退坡</text>
              <Fiber x={560} y={72} w={260} h={44} n={4} lit={false} s={0.9} />
            </g>
            {/* 选项B */}
            <g transform="translate(1060,420)">
              <rect x={0} y={0} width={730} height={140} rx={18} fill={P.bgDeep}
                stroke={P.mint} strokeWidth={4} />
              <Check x={52} y={70} s={1.1} />
              <text x={96} y={60} fontSize={26} fill={P.mint} fontFamily={FONT_HEAD} fontWeight={900}>
                减量维持</text>
              <text x={96} y={102} fontSize={21} fill={P.muted} fontFamily={FONT_HEAD}>
                每周留一点量 · 重启时全程占便宜</text>
              <Fiber x={560} y={72} w={260} h={60} n={6} s={0.9} />
            </g>
            {/* 重启对比小条 */}
            <g transform="translate(1060,600)">
              <text x={0} y={24} fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>
                同龄重启速度</text>
              <rect x={220} y={4} width={220} height={30} rx={15} fill={P.bgDeep} />
              <rect x={220} y={4} width={90} height={30} rx={15} fill={P.muted} />
              <text x={220} y={64} fontSize={19} fill={P.muted} fontFamily={FONT_HEAD}>
                从没练过</text>
              <rect x={480} y={4} width={260} height={30} rx={15} fill={P.bgDeep} />
              <rect x={480} y={4} width={260 * optK} height={30} rx={15} fill={P.nuc} />
              <text x={480} y={64} fontSize={19} fill={P.nuc} fontFamily={FONT_HEAD}
                fontWeight={800}>年轻时练过的你</text>
            </g>
          </g>
        )}
        {/* 底条：汗都记着（cue2） */}
        {sumK > 0 && (
          <g opacity={sumK} transform={`translate(0,${(1 - sumK) * 50})`}>
            <rect x={90} y={740} width={1740} height={128} rx={20} fill={P.panel}
              stroke={P.nuc} strokeWidth={4} />
            {[0, 1, 2].map((i) => (
              <path key={i} d={`M ${170 + i * 40} ${768} q 14 22 0 44 q -14 -22 0 -44`}
                fill={P.sky} opacity={0.9 - i * 0.2} />
            ))}
            <text x={330} y={792} fontSize={30} fill={P.cream} fontFamily={FONT_HEAD} fontWeight={900}>
              你流过的汗 · 肌肉都偷偷记着</text>
            <text x={330} y={836} fontSize={25} fill={P.muted} fontFamily={FONT_HEAD}>
              等你想回来 · 它比你想象的更欢迎</text>
            <g transform="translate(1400,804)">
              <rect x={-190} y={-46} width={380} height={92} rx={18} fill={P.bgDeep}
                stroke={P.nuc} strokeWidth={4} />
              <text x={0} y={-8} textAnchor="middle" fontSize={27} fill={P.nuc}
                fontFamily={FONT_HEAD} fontWeight={900}>练过 ≠ 白练</text>
              <text x={0} y={28} textAnchor="middle" fontSize={20} fill={P.muted}
                fontFamily={FONT_HEAD}>中断不可怕 · 从没开始才亏</text>
            </g>
          </g>
        )}
      </svg>
      <Chip text="别怕中断 · 怕没开始" x={1340} y={72} at={beat(0) + 30} accent={P.nuc} />
      <Stamp text="练过是存款" x={640} y={86} at={b1 + 30} color={P.nucD} size={38} rot={-3} />
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
        <g transform={`translate(960,660) scale(${pop})`} opacity={clamp01(pop)}>
          <Fiber x={0} y={0} w={760} h={140} n={9} />
          <SatCell x={-440} y={-60} s={0.8} rot={-20} />
          <SatCell x={450} y={50} s={0.75} rot={15} />
          <g transform="translate(0,130)">
            <rect x={-260} y={-40} width={520} height={80} rx={18} fill={P.panel}
              stroke={P.nuc} strokeWidth={4} />
            <Nuc x={-216} y={0} s={0.8} />
            <text x={22} y={10} textAnchor="middle" fontSize={27} fill={P.nuc}
              fontFamily={FONT_HEAD} fontWeight={900}>肌核在岗 · 随时重启</text>
          </g>
        </g>
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -360}}>
        <div style={{fontSize: 50, fontWeight: 800, color: C.text}}>练过的每一分 · 肌肉都替你记着</div>
        <div style={{fontSize: 36, color: P.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E248 = ({data = tl}: {data?: any}) => {
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
