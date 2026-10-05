import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01} from '../lib/style';
import {Backdrop, Stamp, SceneTag, Chip, Subtitles, ProgressBar, FadeIn, Moon, Stars} from '../lib/atoms';
import tl from '../timeline_237.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：午夜靛蓝 × 深睡蓝 × REM紫 × 月黄 × 珊瑚警示 ============ */
const P = {
  bg: '#1c2140',
  bgDeep: '#131730',
  bgLite: '#262c54',
  panel: '#1f2547',
  card: '#3a4270',
  ink: '#151a36',
  cream: '#f5eeda',
  muted: '#a9b2d8',
  moon: '#ffd166',
  moonD: '#e0a53d',
  deep: '#4da3ff',
  deepD: '#2f6fd0',
  rem: '#c792ff',
  remD: '#9a63e0',
  shallow: '#7d87b8',
  coral: '#ff7a6e',
  coralD: '#d8564c',
  teal: '#5fd9b3',
  tealD: '#2f9e6e',
  wine: '#ff8fa3',
  wineD: '#d9567a',
  sky: '#5cc8ff',
  hair: '#3a3050',
};

/* ---------- 小叉 / 小勾 ---------- */
const MarkX = ({x, y, s = 1, col = P.coral}: {x: number; y: number; s?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} stroke={col} strokeWidth={7} strokeLinecap="round">
    <line x1={-13} y1={-13} x2={13} y2={13} /><line x1={13} y1={-13} x2={-13} y2={13} />
  </g>
);
const MarkV = ({x, y, s = 1, col = P.teal}: {x: number; y: number; s?: number; col?: string}) => (
  <path d="M -13 0 L -3 11 L 15 -11" transform={`translate(${x},${y}) scale(${s})`}
    stroke={col} strokeWidth={7} fill="none" strokeLinecap="round" strokeLinejoin="round" />
);

/* ---------- 飘z ---------- */
const Zzz = ({x, y, k = 1, col = P.deep}: {x: number; y: number; k?: number; col?: string}) => {
  const f = useCurrentFrame();
  return (
    <g transform={`translate(${x},${y})`} opacity={k}>
      {[0, 1, 2].map((i) => {
        const t = (f * 0.9 + i * 20) % 60;
        const p = t / 60;
        return (
          <text key={i} x={i * 24 + Math.sin(p * 3) * 9} y={-p * 70}
            fontSize={24 + i * 13} fill={col} fontFamily={FONT_HEAD}
            fontWeight={700} opacity={0.95 * (1 - p)}>z</text>
        );
      })}
    </g>
  );
};

/* ---------- 警示火花 ---------- */
const Spark = ({x, y, s = 1, col = P.coral, spin = 1}: {x: number; y: number; s?: number; col?: string; spin?: number}) => {
  const f = useCurrentFrame();
  const pulse = 0.75 + Math.abs(Math.sin(f * 0.35)) * 0.45;
  return (
    <g transform={`translate(${x},${y}) scale(${s * pulse}) rotate(${f * spin})`} stroke={col}
      strokeWidth={5} strokeLinecap="round">
      {[0, 1, 2, 3, 4, 5].map((i) => {
        const a = i * 60 * Math.PI / 180;
        return <line key={i} x1={Math.cos(a) * 12} y1={Math.sin(a) * 12}
          x2={Math.cos(a) * 30} y2={Math.sin(a) * 30} />;
      })}
      <circle r={6} fill={col} stroke="none" />
    </g>
  );
};

/* ---------- 圆角标签牌 ---------- */
const TagPill = ({x, y, t, col, s = 1, w = 200, fs = 30, tcol}: {
  x: number; y: number; t: string; col: string; s?: number; w?: number; fs?: number; tcol?: string;
}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-w / 2} y={-30} width={w} height={60} rx={16} fill={P.panel} stroke={col} strokeWidth={4} />
    <text x={0} y={11} textAnchor="middle" fontSize={fs} fill={tcol ?? col}
      fontFamily={FONT_HEAD} fontWeight={800}>{t}</text>
  </g>
);

/* ---------- 床上小人（简洁平涂版） ---------- */
const BedSleeper = ({x, y, s = 1, zzz = true, blanket = P.deep}: {
  x: number; y: number; s?: number; zzz?: boolean; blanket?: string;
}) => {
  const f = useCurrentFrame();
  const br = Math.sin(f * 0.14) * 3;
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      <rect x={-250} y={0} width={500} height={34} rx={16} fill={P.card} />
      <rect x={-230} y={34} width={24} height={50} rx={10} fill={P.card} />
      <rect x={206} y={34} width={24} height={50} rx={10} fill={P.card} />
      <rect x={-222} y={-54} width={140} height={56} rx={22} fill={P.cream} />
      <circle cx={-140} cy={-80} r={34} fill={C.skin} />
      <path d="M -174 -84 Q -140 -122 -106 -84 L -106 -74 Q -140 -102 -174 -74 Z" fill={P.hair} />
      <path d="M -154 -76 q 7 8 14 0" stroke={P.ink} strokeWidth={3.4} fill="none" strokeLinecap="round" />
      <path d={`M -84 0 Q 10 ${-98 + br} 196 -8 L 246 -2 L 246 8 L -84 8 Z`} fill={blanket} />
      <path d={`M -84 0 Q 10 ${-98 + br} 196 -8`} fill="none" stroke={P.cream}
        strokeWidth={5} strokeLinecap="round" opacity={0.5} />
      {zzz && <Zzz x={-84} y={-118} k={0.9} />}
    </g>
  );
};

/* ---------- 打卡闹钟 ---------- */
const AlarmClock = ({x, y, s = 1, label = '8h'}: {x: number; y: number; s?: number; label?: string}) => {
  const f = useCurrentFrame();
  const wob = Math.sin(f * 0.08) * 2;
  return (
    <g transform={`translate(${x},${y + wob}) scale(${s})`}>
      <line x1={-52} y1={66} x2={-68} y2={96} stroke={P.moonD} strokeWidth={10} strokeLinecap="round" />
      <line x1={52} y1={66} x2={68} y2={96} stroke={P.moonD} strokeWidth={10} strokeLinecap="round" />
      <circle cx={-54} cy={-86} r={24} fill={P.moonD} />
      <circle cx={54} cy={-86} r={24} fill={P.moonD} />
      <line x1={-30} y1={-98} x2={30} y2={-98} stroke={P.moonD} strokeWidth={9} strokeLinecap="round" />
      <circle r={92} fill={P.cream} stroke={P.moonD} strokeWidth={8} />
      {Array.from({length: 12}).map((_, i) => (
        <line key={i} x1={0} y1={-78} x2={0} y2={-66} stroke={P.ink} strokeWidth={4.5}
          strokeLinecap="round" transform={`rotate(${i * 30})`} />
      ))}
      {/* 8:00 指针 */}
      <line x1={0} y1={0} x2={0} y2={-44} stroke={P.ink} strokeWidth={9} strokeLinecap="round"
        transform="rotate(240)" />
      <line x1={0} y1={0} x2={0} y2={-62} stroke={P.ink} strokeWidth={6} strokeLinecap="round" />
      <circle r={8} fill={P.ink} />
      <text x={0} y={58} textAnchor="middle" fontSize={30} fill={P.coral}
        fontFamily={FONT_HEAD} fontWeight={900}>{label}</text>
    </g>
  );
};

/* ---------- 电量电池 ---------- */
const Battery = ({x, y, s = 1, k = 0.6, col = P.coral, label = ''}: {
  x: number; y: number; s?: number; k?: number; col?: string; label?: string;
}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-95} y={-52} width={190} height={104} rx={16} fill="none" stroke={P.cream} strokeWidth={7} />
    <rect x={97} y={-20} width={16} height={40} rx={7} fill={P.cream} />
    <rect x={-86} y={-43} width={172 * k} height={86} rx={10} fill={col} />
    <text x={0} y={13} textAnchor="middle" fontSize={34} fill={P.cream}
      fontFamily={FONT_HEAD} fontWeight={900}>{label}</text>
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
        <Stars n={22} from={0} dur={24} />
        <Moon x={250} y={150} scale={0.7} />
        {/* 中央：床 + 打卡钟 + 低电量 */}
        <g transform={`translate(960,340) scale(${pA}) translate(-960,-340)`} opacity={clamp01(pA)}>
          <AlarmClock x={480} y={300} s={0.95} />
          <BedSleeper x={960} y={400} s={0.95} zzz />
          <Battery x={1420} y={300} s={1} k={0.6} label="≈5h" />
          <text x={1420} y={398} textAnchor="middle" fontSize={23} fill={P.muted}
            fontFamily={FONT_HEAD}>实际充到的电</text>
          {/* 打卡满勤小牌 */}
          <g transform="translate(480,160) rotate(-8)">
            <rect x={-120} y={-30} width={240} height={56} rx={14} fill={P.tealD} />
            <text x={0} y={8} textAnchor="middle" fontSize={26} fill={P.cream}
              fontFamily={FONT_HEAD} fontWeight={900}>睡满 8h ✓</text>
          </g>
          {/* ? 气泡 */}
          <g transform="translate(1580,160)">
            <circle r={34} fill={P.coral} />
            <text y={12} textAnchor="middle" fontSize={36} fill={P.cream}
              fontFamily={FONT_HEAD} fontWeight={900}>?</text>
          </g>
        </g>
        {/* 两侧小标签 */}
        <g transform={`translate(560,560) scale(${pB})`} opacity={clamp01(pB)}>
          <TagPill x={0} y={0} t="时长账 及格" col={P.teal} w={260} />
          <MarkV x={150} y={-6} s={0.9} />
        </g>
        <g transform={`translate(1360,560) scale(${pB})`} opacity={clamp01(pB)}>
          <TagPill x={0} y={0} t="质量账 挂科" col={P.coral} w={260} />
          <MarkX x={152} y={-6} s={0.9} />
        </g>
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 440}}>
          <div style={{color: P.rem, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 24}}>
            冷知识 · 第 {data.episode} 期
          </div>
          <div style={{color: C.text, fontSize: 78, fontWeight: 900, fontFamily: FONT_HEAD, letterSpacing: 4}}>
            {data.title}
          </div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ============ S0 睡眠结账小票 + 皮馅包子 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);
  const popL = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const popR = Math.max(0.001, spring({frame: f - 35, fps, config: {damping: 11, stiffness: 120}}));
  const rowK = [0, 1, 2].map((i) => clamp01((f - 60 - i * 26) / 18));
  const hangK = clamp01((f - b1 - 20) / 22);
  const bunK = clamp01((f - b2) / 24);

  // 小票撕边
  const teeth: string[] = [];
  for (let i = 0; i <= 18; i++) teeth.push(`${1140 + i * 30},${i % 2 ? 578 : 560}`);

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={0} name={sec.name} accent={P.moon} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={10} from={0} dur={26} />
        {/* 左：打卡机只看时长 */}
        <g transform={`translate(120,150) scale(${popL}) translate(-120,-150)`} opacity={clamp01(popL * 2)}>
          <rect x={120} y={150} width={740} height={440} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={156} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>打卡机只看得见时长</text>
          <AlarmClock x={360} y={368} s={0.82} />
          <BedSleeper x={640} y={452} s={0.62} zzz />
          <g transform="translate(560,268) rotate(-8)">
            <rect x={-110} y={-28} width={220} height={52} rx={14} fill={P.tealD} />
            <text x={0} y={8} textAnchor="middle" fontSize={25} fill={P.cream}
              fontFamily={FONT_HEAD} fontWeight={900}>满勤 ✓</text>
          </g>
          <text x={490} y={548} textAnchor="middle" fill={P.teal} fontSize={30}
            fontFamily={FONT_HEAD} fontWeight={900}>时长账 · 账面及格</text>
        </g>
        {/* 右：睡眠结账小票 */}
        <g transform={`translate(1040,150) scale(${popR}) translate(-1040,-150)`} opacity={clamp01(popR * 2)}>
          <rect x={1040} y={150} width={740} height={440} rx={20} fill={P.panel}
            stroke={hangK > 0.3 ? P.coral : P.card} strokeWidth={4} />
          <text x={1076} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>但睡眠按质量结账</text>
          {/* 小票纸 */}
          <rect x={1140} y={232} width={540} height={332} rx={10} fill={P.cream} />
          <polygon points={`1140,560 ${teeth.join(' ')} 1680,560`} fill={P.cream} />
          <text x={1410} y={280} textAnchor="middle" fill={P.ink} fontSize={30}
            fontFamily={FONT_HEAD} fontWeight={900}>睡 眠 结 账 小 票</text>
          <line x1={1180} y1={300} x2={1640} y2={300} stroke={P.ink} strokeWidth={2.5}
            strokeDasharray="8 7" opacity={0.4} />
          {/* 票面条目 */}
          <g opacity={rowK[0]}>
            <text x={1180} y={346} fill={P.ink} fontSize={25} fontFamily={FONT_HEAD}>时长账 · 8.0 小时</text>
            <rect x={1470} y={318} width={180} height={42} rx={12} fill={P.tealD} />
            <text x={1560} y={348} textAnchor="middle" fontSize={24} fill={P.cream}
              fontFamily={FONT_HEAD} fontWeight={900}>及格 ✓</text>
          </g>
          <g opacity={rowK[1]}>
            <text x={1180} y={404} fill={P.ink} fontSize={25} fontFamily={FONT_HEAD}>浅睡 + 微觉醒</text>
            <rect x={1470} y={376} width={180} height={42} rx={12} fill={P.shallow} />
            <text x={1560} y={406} textAnchor="middle" fontSize={24} fill={P.cream}
              fontFamily={FONT_HEAD} fontWeight={900}>≈5 小时</text>
          </g>
          <g opacity={rowK[2]}>
            <text x={1180} y={462} fill={P.ink} fontSize={25} fontFamily={FONT_HEAD}>深睡 + REM</text>
            <rect x={1470} y={434} width={180} height={42} rx={12} fill={P.coral} />
            <text x={1560} y={464} textAnchor="middle" fontSize={24} fill={P.cream}
              fontFamily={FONT_HEAD} fontWeight={900}>被切碎</text>
          </g>
          {/* 质量账挂科 */}
          {hangK > 0 && (
            <g opacity={hangK}>
              <line x1={1180} y1={492} x2={1640} y2={492} stroke={P.ink} strokeWidth={2.5}
                strokeDasharray="8 7" opacity={0.4} />
              <text x={1180} y={536} fill={P.ink} fontSize={27} fontFamily={FONT_HEAD} fontWeight={900}>质量账</text>
              <rect x={1470} y={504} width={180} height={46} rx={12} fill={P.coralD} />
              <text x={1560} y={537} textAnchor="middle" fontSize={26} fill={P.cream}
                fontFamily={FONT_HEAD} fontWeight={900}>挂科 ✗</text>
            </g>
          )}
          {/* 大印章：实效（骑在小票撕边上） */}
          {hangK > 0.4 && (
            <g transform="translate(1260,578) rotate(-9)" opacity={clamp01((hangK - 0.4) * 2)}>
              <rect x={-150} y={-34} width={300} height={64} rx={12} fill={P.panel}
                stroke={P.coralD} strokeWidth={5} />
              <text x={0} y={10} textAnchor="middle" fontSize={32} fill={P.coralD}
                fontFamily={FONT_HEAD} fontWeight={900}>实效 ≈5h</text>
            </g>
          )}
        </g>
        {/* 底部：皮馅包子（cue2） */}
        {bunK > 0 && (
          <g opacity={bunK} transform={`translate(0,${(1 - bunK) * 60})`}>
            <rect x={120} y={648} width={1680} height={192} rx={20} fill={P.panel} stroke={P.moon} strokeWidth={4} />
            {/* 包子剖面 */}
            <g transform="translate(370,744)">
              <circle r={84} fill={P.cream} />
              <circle r={84} fill="none" stroke={P.moonD} strokeWidth={4} opacity={0.6} />
              <circle r={34} fill={P.deep} />
              <circle cx={-10} cy={-8} r={14} fill={P.rem} />
              <circle cx={12} cy={10} r={10} fill={P.rem} />
            </g>
            <line x1={330} y1={680} x2={262} y2={672} stroke={P.cream} strokeWidth={3} strokeDasharray="6 6" />
            <text x={246} y={678} textAnchor="end" fill={P.cream} fontSize={21} fontFamily={FONT_HEAD}>皮=时长</text>
            <line x1={396} y1={726} x2={470} y2={694} stroke={P.deep} strokeWidth={3} strokeDasharray="6 6" />
            <text x={482} y={690} fill={P.deep} fontSize={21} fontFamily={FONT_HEAD}>馅=深睡+REM</text>
            <text x={560} y={722} fill={P.moon} fontSize={34} fontFamily={FONT_HEAD} fontWeight={900}>
              时长是皮 · 质量是馅</text>
            <text x={560} y={780} fill={P.cream} fontSize={29} fontFamily={FONT_HEAD} fontWeight={700}>
              皮够馅空 → 躺够 8 小时，实效只有 5 小时</text>
            {/* 右侧迷你结构条 */}
            <g transform="translate(1330,716)">
              <text x={160} y={-16} textAnchor="middle" fill={P.muted} fontSize={22}
                fontFamily={FONT_HEAD}>你那晚的干活份额</text>
              <rect x={0} y={0} width={211} height={44} rx={12} fill={P.shallow} />
              <rect x={211} y={0} width={64} height={44} fill={P.deep} />
              <rect x={275} y={0} width={45} height={44} rx={12} fill={P.rem} />
              <rect x={211} y={0} width={4} height={44} fill={P.bg} />
              <text x={160} y={76} textAnchor="middle" fill={P.muted} fontSize={20}
                fontFamily={FONT_HEAD}>浅睡一大半 · 干活的不到一半</text>
            </g>
          </g>
        )}
      </svg>
      <Chip text="睡够 ≠ 睡好" x={1420} y={72} at={beat(0) + 25} accent={P.moon} />
      <Stamp text="按质量结账" x={620} y={88} at={b1 + 40} color={P.coralD} size={38} rot={-3} />
    </FadeIn>
  );
};

/* ============ S1 睡眠分期条 + 三个值班员 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);
  const barPop = Math.max(0.001, spring({frame: f - 8, fps, config: {damping: 11, stiffness: 120}}));
  const segK = [0, 1, 2].map((i) => clamp01((f - 18 - i * 20) / 22));
  const cardPop = [0, 1, 2].map((i) => Math.max(0.001, spring({frame: f - b1 - 10 - i * 16, fps, config: {damping: 11, stiffness: 130}})));
  const crackK = clamp01((f - b2) / 20);

  const segs = [
    {x: 180, w: 858, col: P.shallow, t: '浅睡 N1+N2 · 约一半', tc: P.cream, sub: '过渡层 · 躺了没完全睡', subCol: P.muted},
    {x: 1038, w: 390, col: P.deep, t: '深睡 ≈25%', tc: P.ink, sub: '修身体 · 调免疫', subCol: P.deep},
    {x: 1428, w: 312, col: P.rem, t: 'REM ≈20%', tc: P.ink, sub: '理记忆 · 抚情绪', subCol: P.rem},
  ];
  const cards = [
    {x: 120, col: P.shallow, dash: true, title: '浅睡 · 过渡层', tag: '低价值', tagCol: P.muted,
      lines: ['躺了但没完全睡', '占掉半晚 · 恢复贡献小'], icon: 'cloud'},
    {x: 690, col: P.deep, dash: false, title: '深睡 · 夜班修理工', tag: '高价值', tagCol: P.deep,
      lines: ['修身体 · 分泌生长激素', '调免疫 · 恢复主力'], icon: 'heart'},
    {x: 1260, col: P.rem, dash: false, title: 'REM · 夜班档案员', tag: '高价值', tagCol: P.rem,
      lines: ['整理记忆 · 安抚情绪', '做梦主要在这一段'], icon: 'brain'},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={1} name={sec.name} accent={P.rem} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 顶部：分期结构条 */}
        <g transform={`translate(120,140) scale(${barPop}) translate(-120,-140)`} opacity={clamp01(barPop * 2)}>
          <rect x={120} y={140} width={1680} height={290} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={156} y={196} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>一夜 8 小时的内部结构</text>
          <Moon x={1730} y={196} scale={0.3} />
          {segs.map((sg, i) => (
            <g key={i}>
              <rect x={sg.x} y={250} width={sg.w * segK[i]} height={100} rx={i === 0 ? 18 : 0}
                fill={sg.col} />
              {segK[i] > 0.7 && (
                <text x={sg.x + sg.w / 2} y={312} textAnchor="middle" fontSize={30} fill={sg.tc}
                  fontFamily={FONT_HEAD} fontWeight={900}>{sg.t}</text>
              )}
              <text x={sg.x + sg.w / 2} y={396} textAnchor="middle" fontSize={23} fill={sg.subCol}
                fontFamily={FONT_HEAD}>{sg.sub}</text>
            </g>
          ))}
          {/* cue2：结构被切碎 → 裂缝 */}
          {crackK > 0 && (
            <g opacity={crackK} stroke={P.bg} strokeWidth={8} strokeLinecap="round">
              {[400, 640, 880, 1180, 1360, 1560].map((cx, i) => (
                <path key={i} d={`M ${cx} 250 l 12 30 l -14 28 l 14 30 l -10 14`}
                  fill="none" />
              ))}
            </g>
          )}
          {crackK > 0.3 && (
            <g transform="translate(960,300) rotate(-6)" opacity={clamp01((crackK - 0.3) * 2)}>
              <rect x={-190} y={-36} width={380} height={68} rx={14} fill={P.coralD} />
              <text x={0} y={10} textAnchor="middle" fontSize={34} fill={P.cream}
                fontFamily={FONT_HEAD} fontWeight={900}>切碎 = 白睡</text>
            </g>
          )}
        </g>
        {/* 三张值班卡（cue1） */}
        {cards.map((cd, i) => (
          <g key={i} transform={`translate(${cd.x},462) scale(${cardPop[i]}) translate(${-cd.x},-462)`}
            opacity={clamp01(cardPop[i] * 2)}>
            <rect x={cd.x} y={462} width={540} height={330} rx={20} fill={P.panel}
              stroke={cd.col} strokeWidth={4} strokeDasharray={cd.dash ? '14 10' : '0'} opacity={cd.dash ? 0.75 : 1} />
            <TagPill x={cd.x + 430} y={512} t={cd.tag} col={cd.tagCol} w={150} fs={22}
              tcol={cd.dash ? P.muted : P.cream} />
            {/* 图标 */}
            <g transform={`translate(${cd.x + 78},556)`}>
              <circle r={46} fill={P.bgDeep} stroke={cd.col} strokeWidth={4} />
              {cd.icon === 'cloud' && (
                <g fill={P.shallow}>
                  <ellipse cx={0} cy={8} rx={30} ry={16} />
                  <circle cx={-12} cy={-4} r={14} /><circle cx={8} cy={-8} r={17} />
                  <text x={34} y={-16} fontSize={24} fill={P.muted} fontFamily={FONT_HEAD}
                    fontWeight={700}>z</text>
                </g>
              )}
              {cd.icon === 'heart' && (
                <g>
                  <path d="M 0 18 C -26 -4 -16 -28 0 -14 C 16 -28 26 -4 0 18 Z" fill={P.cream} />
                  <polyline points="-20,0 -10,0 -5,-8 2,6 7,0 20,0" fill="none"
                    stroke={P.deep} strokeWidth={4} strokeLinecap="round" strokeLinejoin="round" />
                </g>
              )}
              {cd.icon === 'brain' && (
                <g>
                  <circle r={22} fill={P.cream} />
                  <path d="M -12 -4 a 6 6 0 0 1 12 -2 M -2 -12 a 6 6 0 0 1 12 3 M -11 7 a 7 7 0 0 0 11 5"
                    stroke={P.remD} strokeWidth={3.4} fill="none" strokeLinecap="round" />
                  <circle cx={19} cy={-19} r={6} fill={P.rem} />
                </g>
              )}
            </g>
            <text x={cd.x + 146} y={568} fill={cd.dash ? P.muted : cd.col} fontSize={32}
              fontFamily={FONT_HEAD} fontWeight={900}>{cd.title}</text>
            {cd.lines.map((ln, k) => (
              <text key={k} x={cd.x + 42} y={648 + k * 42} fill={P.cream} fontSize={24}
                fontFamily={FONT_HEAD}>{ln}</text>
            ))}
            {/* 卡的角标序号 */}
            <text x={cd.x + 40} y={770} fill={P.muted} fontSize={22} fontFamily={FONT_HEAD}>
              {i === 0 ? '过渡值班 · 没干什么活' : i === 1 ? '前半夜高产期' : '后半夜高产期'}</text>
          </g>
        ))}
      </svg>
      <Chip text="干活的是深睡和REM" x={1290} y={72} at={beat(0) + 25} accent={P.rem} />
      <Stamp text="时长及格 · 质量挂科" x={600} y={88} at={b1 + 30} color={P.coralD} size={36} rot={-2} />
    </FadeIn>
  );
};

/* ============ S2 呼吸暂停：气道塌陷 + 深睡切碎曲线 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);
  const popL = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const popR = Math.max(0.001, spring({frame: f - 30, fps, config: {damping: 11, stiffness: 120}}));
  const drawK = clamp01((f - 25) / 70);
  // cue1 后气道反复塌陷（脉冲式）
  const collapse = f > b1 ? Math.abs(Math.sin((f - b1) * 0.07)) : 0;
  const arousalN = Math.min(4, Math.floor(drawK * 4.6));
  const pillPop = [0, 1, 2].map((i) => Math.max(0.001, spring({frame: f - 30 - i * 16, fps, config: {damping: 12, stiffness: 140}})));
  const tipK = clamp01((f - b1 - 20) / 20);

  // 被切碎的深睡曲线（反复弹回浅睡）
  const jagPts: string[] = ['1170,302', '1250,398', '1310,446'];
  for (let k = 0; k < 4; k++) {
    const sx = 1345 + k * 95;
    jagPts.push(`${sx},302`);
    jagPts.push(`${sx + 38},452`);
  }
  jagPts.push('1710,452');
  const spikes = [1345, 1440, 1535, 1630];

  return (
    <FadeIn>
      <Backdrop top={P.bgDeep} bottom={P.bgDeep} />
      <SceneTag index={2} name={sec.name} accent={P.coral} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={8} from={0} dur={26} />
        {/* 左：气道剖面 */}
        <g transform={`translate(120,150) scale(${popL}) translate(-120,-150)`} opacity={clamp01(popL * 2)}>
          <rect x={120} y={150} width={740} height={470} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={156} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>气道剖面 · 睡着以后</text>
          {/* 头轮廓 */}
          <circle cx={330} cy={330} r={96} fill={C.skin} />
          <polygon points="236,320 262,344 236,356" fill={C.skin} />
          <path d="M 300 260 Q 330 236 366 252" stroke={P.hair} strokeWidth={16}
            fill="none" strokeLinecap="round" />
          <path d="M 300 330 q 8 9 16 0" stroke={P.ink} strokeWidth={3.4} fill="none" strokeLinecap="round" />
          {/* 气道：鼻→咽→下 */}
          <path d="M 262 336 Q 470 330 470 380 L 470 548" fill="none" stroke={P.bg}
            strokeWidth={56} strokeLinecap="round" />
          <path d="M 262 336 Q 470 330 470 380 L 470 548" fill="none" stroke={P.muted}
            strokeWidth={56} strokeLinecap="round" opacity={0.25} />
          {/* 咽壁 */}
          <path d="M 420 356 Q 420 420 420 548" stroke={C.skinDark} strokeWidth={16}
            fill="none" strokeLinecap="round" opacity={0.8} />
          {/* 舌根 + 塌陷压板 */}
          <ellipse cx={566 - collapse * 26} cy={500} rx={72} ry={44} fill={C.skinDark} />
          <rect x={496 - collapse * 42} y={428} width={46} height={60} rx={14} fill={C.skinDark}
            opacity={0.2 + collapse * 0.8} />
          <text x={630} y={560} fill={C.skinDark} fontSize={21} fontFamily={FONT_HEAD}
            fontWeight={700}>舌根后坠</text>
          {/* 气流点：塌陷时堵在压点上方 */}
          {[0, 1, 2].map((i) => {
            const t = (f * 0.018 + i * 0.33) % 1;
            let cy = 300 + t * 230;
            let fade = 1;
            if (collapse > 0.5 && cy > 420) {cy = 420; fade = clamp01((0.5 - (cy - 420) / 40) * 2 + 0.3);}
            return <circle key={i} cx={452 - t * 6} cy={cy} r={9} fill={P.sky}
              opacity={0.9 * fade} />;
          })}
          <text x={368} y={292} textAnchor="middle" fontSize={20} fill={P.muted}
            fontFamily={FONT_HEAD}>气流</text>
          {/* 塌陷警示 */}
          {collapse > 0.5 && (
            <g opacity={collapse}>
              <Spark x={470} y={452} s={0.7} />
              <text x={560} y={430} fontSize={24} fill={P.coral} fontFamily={FONT_HEAD}
                fontWeight={900}>塌陷断气!</text>
            </g>
          )}
          <text x={490} y={578} textAnchor="middle" fill={P.coral} fontSize={26}
            fontFamily={FONT_HEAD} fontWeight={900}>气道塌 → 断气 → 强制微觉醒</text>
          <text x={490} y={608} textAnchor="middle" fill={P.muted} fontSize={21}
            fontFamily={FONT_HEAD}>一晚几十上百次 · 你完全不知道</text>
        </g>
        {/* 右：深睡曲线被切碎 */}
        <g transform={`translate(1040,150) scale(${popR}) translate(-1040,-150)`} opacity={clamp01(popR * 2)}>
          <rect x={1040} y={150} width={740} height={470} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={1076} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>睡眠深度记录仪</text>
          {/* 微觉醒计数 */}
          <rect x={1480} y={176} width={260} height={50} rx={14} fill={P.bgDeep} stroke={P.coral} strokeWidth={3.5} />
          <text x={1610} y={209} textAnchor="middle" fontSize={25} fill={P.coral}
            fontFamily={FONT_HEAD} fontWeight={900}>微觉醒 ×{arousalN}</text>
          {/* 坐标 */}
          <line x1={1150} y1={250} x2={1150} y2={540} stroke={P.muted} strokeWidth={4} />
          <polygon points="1150,556 1142,538 1158,538" fill={P.muted} />
          <line x1={1150} y1={540} x2={1720} y2={540} stroke={P.muted} strokeWidth={4} />
          <polygon points="1734,540 1718,532 1718,548" fill={P.muted} />
          <text x={1140} y={272} textAnchor="end" fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>浅</text>
          <text x={1140} y={534} textAnchor="end" fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>深</text>
          <text x={1700} y={575} textAnchor="end" fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>时间</text>
          {/* 健康曲线（虚线对照） */}
          <path d="M 1170 302 C 1330 310 1360 470 1470 482 C 1570 492 1660 468 1710 482"
            fill="none" stroke={P.muted} strokeWidth={4} strokeDasharray="10 9" opacity={0.55} />
          <text x={1450} y={516} fontSize={20} fill={P.muted} fontFamily={FONT_HEAD}>健康睡眠</text>
          {/* 实际曲线：反复被拽回浅睡 */}
          <polyline points={jagPts.slice(0, Math.max(2, Math.floor(jagPts.length * drawK))).join(' ')}
            fill="none" stroke={P.coral} strokeWidth={7} strokeLinecap="round" strokeLinejoin="round" />
          {/* 尖峰 = 微觉醒 */}
          {spikes.slice(0, arousalN).map((sx, i) => (
            <g key={i}>
              <text x={sx} y={284} textAnchor="middle" fontSize={30} fill={P.coral}
                fontFamily={FONT_HEAD} fontWeight={900}>!</text>
              <circle cx={sx} cy={302} r={16} fill="none" stroke={P.coral} strokeWidth={4}
                strokeDasharray="5 5" transform={`rotate(${f * 2} ${sx} 302)`} />
            </g>
          ))}
          <text x={1435} y={420} fontSize={20} fill={P.coral} fontFamily={FONT_HEAD}
            fontWeight={800} opacity={0.9}>你的夜晚</text>
          <text x={1430} y={592} textAnchor="middle" fill={P.cream} fontSize={26}
            fontFamily={FONT_HEAD} fontWeight={800}>深睡被切得稀碎 · 睡到的是碎片</text>
        </g>
        {/* 底部症状三连 + 指向（cue0/cue1） */}
        {[0, 1, 2].map((i) => (
          <g key={i} transform={`translate(${400 + i * 560},668) scale(${pillPop[i]}) translate(${-(400 + i * 560)},-668)`}
            opacity={clamp01(pillPop[i] * 2)}>
            <rect x={400 + i * 560 - 160} y={640} width={320} height={56} rx={28}
              fill={P.panel} stroke={P.coral} strokeWidth={3.5} />
            <circle cx={400 + i * 560 - 124} cy={668} r={8} fill={P.coral} />
            <text x={400 + i * 560 + 10} y={678} textAnchor="middle" fontSize={27} fill={P.cream}
              fontFamily={FONT_HEAD} fontWeight={800}>
              {i === 0 ? '打鼾很响' : i === 1 ? '早起头痛' : '白天嗜睡'}</text>
          </g>
        ))}
        {tipK > 0 && (
          <g opacity={tipK}>
            <rect x={560} y={726} width={800} height={64} rx={16} fill={P.bgDeep}
              stroke={P.coral} strokeWidth={3.5} />
            <text x={960} y={768} textAnchor="middle" fontSize={27} fill={P.coral}
              fontFamily={FONT_HEAD} fontWeight={900}>睡够还累 + 这三样 → 优先排查呼吸暂停</text>
          </g>
        )}
      </svg>
      <Chip text="头号嫌犯" x={1450} y={72} at={beat(0) + 25} accent={P.coral} />
      <Stamp text="常见 · 伤得重 · 能治" x={600} y={90} at={b2 + 8} color={P.tealD} size={36} rot={-2} />
    </FadeIn>
  );
};

/* ============ S3 酒精刺客 + 深睡小偷团伙 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const popL = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const popR = Math.max(0.001, spring({frame: f - b1 - 5, fps, config: {damping: 11, stiffness: 120}}));
  const thiefPop = [0, 1, 2, 3].map((i) => Math.max(0.001, spring({frame: f - b1 - 16 - i * 13, fps, config: {damping: 12, stiffness: 140}})));
  const remK = clamp01((f - 70) / 40);
  const robK = clamp01((f - b1 - 30) / 24);
  const sway = Math.sin(f * 0.1) * 3;

  const thieves = [
    {cap: '慢性压力', sub: '皮质醇压深睡', col: P.coral, icon: 'cloud'},
    {cap: '太热太冷', sub: '温度不对', col: P.moon, icon: 'thermo'},
    {cap: '噪音光亮', sub: '环境打断', col: P.sky, icon: 'noise'},
    {cap: '作息乱', sub: '深睡被错峰', col: P.rem, icon: 'clock'},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={3} name={sec.name} accent={P.wine} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：睡前酒刺客 */}
        <g transform={`translate(120,150) scale(${popL}) translate(-120,-150)`} opacity={clamp01(popL * 2)}>
          <rect x={120} y={150} width={800} height={470} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={156} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>睡前酒 · 温柔刺客</text>
          {/* 刺客酒杯 */}
          <g transform={`translate(330,398) rotate(${sway})`}>
            <rect x={-7} y={14} width={14} height={68} fill={P.cream} />
            <ellipse cx={0} cy={88} rx={54} ry={10} fill={P.cream} />
            <path d="M -64 -88 C -64 -26 -36 14 0 14 C 36 14 64 -26 64 -88 Z"
              fill={P.bgDeep} stroke={P.cream} strokeWidth={5} />
            <path d="M -56 -40 C -56 -16 -32 8 0 8 C 32 8 56 -16 56 -40 Z" fill={P.wineD} />
            {/* 刺客眼罩 */}
            <rect x={-58} y={-76} width={116} height={28} rx={14} fill={P.ink} opacity={0.95} />
            <ellipse cx={-22} cy={-62} rx={8} ry={4.5} fill={P.cream} />
            <ellipse cx={22} cy={-62} rx={8} ry={4.5} fill={P.cream} />
            {/* 小匕首 */}
            <g transform="translate(76,-16) rotate(24)">
              <polygon points="0,-30 8,2 0,34 -8,2" fill="#dfe4ff" />
              <rect x={-6} y={30} width={12} height={18} rx={4} fill={P.moonD} />
            </g>
          </g>
          <text x={330} y={548} textAnchor="middle" fill={P.wine} fontSize={25}
            fontFamily={FONT_HEAD} fontWeight={900}>帮你睡着 ✓ 拆你结构 ✗</text>
          {/* REM 一夜曲线小图 */}
          <rect x={520} y={252} width={340} height={220} rx={14} fill={P.bgDeep} />
          <text x={540} y={232} fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>一夜 REM 曲线</text>
          <line x1={540} y1={436} x2={840} y2={436} stroke={P.card} strokeWidth={3} />
          <line x1={690} y1={262} x2={690} y2={436} stroke={P.card} strokeWidth={3} strokeDasharray="7 7" />
          {/* 前半夜 REM 被压平 */}
          <polyline points="552,420 590,414 630,420 668,416"
            fill="none" stroke={P.rem} strokeWidth={6} strokeLinecap="round"
            opacity={0.4 + remK * 0.6} />
          <text x={604} y={288} textAnchor="middle" fontSize={20} fill={P.rem}
            fontFamily={FONT_HEAD} fontWeight={800} opacity={remK}>REM被压</text>
          <text x={604} y={468} textAnchor="middle" fontSize={20} fill={P.muted}
            fontFamily={FONT_HEAD}>前半夜·假沉稳</text>
          {/* 后半夜 REM 反弹碎裂 */}
          {remK > 0 && (
            <polyline points="700,420 722,310 744,420 766,320 788,420 810,330 832,420"
              fill="none" stroke={P.coral} strokeWidth={6} strokeLinecap="round"
              strokeLinejoin="round" opacity={remK}
              strokeDasharray="500" strokeDashoffset={500 * (1 - remK)} />
          )}
          {remK > 0.6 && (
            <text x={776} y={288} textAnchor="middle" fontSize={20} fill={P.coral}
              fontFamily={FONT_HEAD} fontWeight={800}>REM反弹·醒</text>
          )}
          <text x={776} y={468} textAnchor="middle" fontSize={20} fill={P.muted}
            fontFamily={FONT_HEAD}>后半夜·稀碎</text>
          <text x={690} y={548} textAnchor="middle" fill={P.wine} fontSize={23}
            fontFamily={FONT_HEAD} fontWeight={800}>还放松喉肌 → 鼾更响</text>
          <text x={690} y={584} textAnchor="middle" fill={P.muted} fontSize={20}
            fontFamily={FONT_HEAD}>和呼吸暂停是队友</text>
        </g>
        {/* 右：小偷团伙（cue1） */}
        <g transform={`translate(960,150) scale(${popR}) translate(-960,-150)`} opacity={clamp01(popR * 2)}>
          <rect x={960} y={150} width={840} height={470} rx={20} fill={P.panel}
            stroke={popR > 0.5 ? P.coral : P.card} strokeWidth={4} />
          <text x={996} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>深睡小偷团伙 · 分头作案</text>
          {/* 被咬的深睡条 */}
          <text x={1030} y={268} fill={P.deep} fontSize={22} fontFamily={FONT_HEAD}
            fontWeight={800}>深睡存量</text>
          <g>
            <rect x={1030} y={290} width={640} height={84} rx={20} fill={P.deep} />
            {robK > 0 && [1120, 1260, 1410, 1560].map((bx, i) => (
              <circle key={i} cx={bx} cy={i % 2 ? 290 : 374} r={30 * robK} fill={P.panel} />
            ))}
            {robK > 0.5 && (
              <text x={1350} y={344} textAnchor="middle" fontSize={24} fill={P.cream}
                fontFamily={FONT_HEAD} fontWeight={900} opacity={robK}>被啃薄了</text>
            )}
          </g>
          {/* 四个小偷 */}
          {thieves.map((th, i) => (
            <g key={i} transform={`translate(${1080 + i * 210},472) scale(${thiefPop[i]}) translate(${-(1080 + i * 210)},-472)`}
              opacity={clamp01(thiefPop[i] * 2)}>
              <g transform={`translate(${1080 + i * 210},472)`}>
                <circle r={46} fill={P.bgDeep} stroke={th.col} strokeWidth={4} />
                {/* 小偷眼罩 */}
                <rect x={-34} y={-26} width={68} height={18} rx={9} fill={P.ink} />
                <circle cx={-12} cy={-17} r={3.5} fill={P.cream} />
                <circle cx={12} cy={-17} r={3.5} fill={P.cream} />
                {th.icon === 'cloud' && (
                  <g fill={th.col} transform="translate(0,14)">
                    <ellipse cx={0} cy={6} rx={24} ry={12} />
                    <circle cx={-9} cy={-3} r={11} /><circle cx={7} cy={-6} r={13} />
                  </g>
                )}
                {th.icon === 'thermo' && (
                  <g transform="translate(0,10)">
                    <rect x={-7} y={-16} width={14} height={34} rx={7} fill="none"
                      stroke={th.col} strokeWidth={4} />
                    <circle cy={20} r={11} fill={th.col} />
                  </g>
                )}
                {th.icon === 'noise' && (
                  <g transform="translate(0,12)" stroke={th.col} strokeWidth={4}
                    strokeLinecap="round" fill="none">
                    <polygon points="-20,-6 -8,-6 4,-16 4,16 -8,6 -20,6" fill={th.col} stroke="none" />
                    <path d="M 10 -10 q 10 10 0 20" />
                    <path d="M 18 -16 q 16 16 0 32" />
                  </g>
                )}
                {th.icon === 'clock' && (
                  <g transform="translate(0,10)" stroke={th.col} strokeWidth={4}
                    strokeLinecap="round" fill="none">
                    <circle r={22} />
                    <line x1={0} y1={0} x2={0} y2={-14} />
                    <line x1={0} y1={0} x2={11} y2={6} />
                  </g>
                )}
                <text x={0} y={86} textAnchor="middle" fontSize={24} fill={th.col}
                  fontFamily={FONT_HEAD} fontWeight={900}>{th.cap}</text>
                <text x={0} y={116} textAnchor="middle" fontSize={19} fill={P.muted}
                  fontFamily={FONT_HEAD}>{th.sub}</text>
              </g>
            </g>
          ))}
        </g>
        {/* 底部结论（cue1 后段） */}
        {robK > 0 && (
          <g opacity={robK} transform={`translate(0,${(1 - robK) * 60})`}>
            <rect x={120} y={662} width={1680} height={168} rx={20} fill={P.panel} stroke={P.coral} strokeWidth={4} />
            <text x={160} y={726} fill={P.coral} fontSize={30} fontFamily={FONT_HEAD} fontWeight={900}>
              作案时间：你睡着以后</text>
            <text x={160} y={782} fill={P.cream} fontSize={28} fontFamily={FONT_HEAD} fontWeight={700}>
              微觉醒你自己毫无察觉 —— 单看不致命，叠加起来深睡就被稀释没了</text>
          </g>
        )}
      </svg>
      <Chip text="助眠是假 · 拆结构是真" x={1150} y={72} at={beat(0) + 25} accent={P.wine} />
      <Stamp text="一窝贼分头作案" x={620} y={90} at={b1 + 55} color={P.coralD} size={38} rot={-3} />
    </FadeIn>
  );
};

/* ============ S4 倒U甜点区间 + 自查三招 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);
  const popL = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const curveK = clamp01((f - 20) / 60);
  const cardPop = [0, 1, 2].map((i) => Math.max(0.001, spring({frame: f - b1 - 10 - i * 14, fps, config: {damping: 12, stiffness: 140}})));
  const bannerK = clamp01((f - b2) / 24);

  const cards = [
    {col: P.coral, title: '录音听鼾声', sub: '鼾声停几秒再猛吸气 = 呼吸暂停信号', icon: 'mic'},
    {col: P.moon, title: '疲劳日记找关联', sub: '喝没喝酒 · 熬没熬夜 · 压力大不大', icon: 'note'},
    {col: P.teal, title: '作息稳不稳', sub: '入睡起床时间每天飘 → 累的是节律', icon: 'clock'},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={4} name={sec.name} accent={P.teal} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：倒U曲线 */}
        <g transform={`translate(120,150) scale(${popL}) translate(-120,-150)`} opacity={clamp01(popL * 2)}>
          <rect x={120} y={150} width={760} height={480} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={156} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            睡多久 ↔ 白天精力 · 倒U形</text>
          {/* 甜点区间 */}
          <rect x={470} y={250} width={170} height={290} fill={P.teal} opacity={0.13} />
          <text x={555} y={276} textAnchor="middle" fontSize={23} fill={P.teal}
            fontFamily={FONT_HEAD} fontWeight={900}>甜点 ≈8h</text>
          {/* 坐标轴 */}
          <line x1={200} y1={540} x2={200} y2={250} stroke={P.muted} strokeWidth={4} />
          <polygon points="200,238 192,256 208,256" fill={P.muted} />
          <line x1={200} y1={540} x2={822} y2={540} stroke={P.muted} strokeWidth={4} />
          <polygon points="836,540 820,532 820,548" fill={P.muted} />
          <text x={186} y={240} textAnchor="end" fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>精力</text>
          {['6h', '7h', '8h', '9h', '10h'].map((t, i) => (
            <text key={i} x={200 + i * 155} y={572} textAnchor="middle" fontSize={20}
              fill={P.muted} fontFamily={FONT_HEAD}>{t}</text>
          ))}
          {/* 倒U曲线 */}
          <path d="M 230 495 C 330 470 380 300 470 290 C 540 282 600 282 640 300 C 720 338 770 440 800 498"
            fill="none" stroke={P.moon} strokeWidth={7} strokeLinecap="round"
            strokeDasharray="1400" strokeDashoffset={1400 * (1 - curveK)} />
          {curveK > 0.3 && (
            <g opacity={clamp01((curveK - 0.3) * 2)}>
              <circle cx={230} cy={495} r={9} fill={P.coral} />
              <text x={230} y={462} textAnchor="middle" fontSize={22} fill={P.coral}
                fontFamily={FONT_HEAD} fontWeight={800}>睡少了·累</text>
            </g>
          )}
          {curveK > 0.85 && (
            <g opacity={clamp01((curveK - 0.85) * 4)}>
              <circle cx={800} cy={498} r={9} fill={P.coral} />
              <text x={800} y={534} textAnchor="middle" fontSize={22} fill={P.coral}
                fontFamily={FONT_HEAD} fontWeight={800}>睡多了·也累</text>
            </g>
          )}
          <text x={510} y={612} textAnchor="middle" fill={P.cream} fontSize={24}
            fontFamily={FONT_HEAD} fontWeight={700}>疲劳和睡太久相关 → 该管规律 不是管更久</text>
        </g>
        {/* 右：自查三招（cue1） */}
        {cards.map((cd, i) => (
          <g key={i} transform={`translate(920,${150 + i * 160}) scale(${cardPop[i]}) translate(-920,${-(150 + i * 160)})`}
            opacity={clamp01(cardPop[i] * 2)}>
            <rect x={920} y={150 + i * 160} width={880} height={140} rx={18} fill={P.panel}
              stroke={cd.col} strokeWidth={4} />
            <circle cx={986} cy={220 + i * 160} r={30} fill={cd.col} />
            <text x={986} y={230 + i * 160} textAnchor="middle" fontSize={28} fill={P.ink}
              fontFamily={FONT_HEAD} fontWeight={900}>{i + 1}</text>
            <text x={1040} y={210 + i * 160} fill={cd.col} fontSize={30}
              fontFamily={FONT_HEAD} fontWeight={900}>{cd.title}</text>
            <text x={1040} y={252 + i * 160} fill={P.muted} fontSize={23}
              fontFamily={FONT_HEAD}>{cd.sub}</text>
            {/* 图标 */}
            <g transform={`translate(1705,${220 + i * 160})`}>
              <circle r={44} fill={P.bgDeep} stroke={cd.col} strokeWidth={3.5} />
              {cd.icon === 'mic' && (
                <g stroke={cd.col} strokeWidth={4} strokeLinecap="round" fill={cd.col}>
                  <rect x={-11} y={-26} width={22} height={42} rx={11} />
                  <path d="M -20 -6 a 20 20 0 0 0 40 0" fill="none" />
                  <line x1={0} y1={14} x2={0} y2={26} />
                  <line x1={-14} y1={28} x2={14} y2={28} />
                </g>
              )}
              {cd.icon === 'note' && (
                <g>
                  <rect x={-18} y={-26} width={36} height={52} rx={6} fill="none"
                    stroke={cd.col} strokeWidth={4} />
                  <line x1={-8} y1={-12} x2={10} y2={-12} stroke={cd.col} strokeWidth={3.5} />
                  <line x1={-8} y1={0} x2={10} y2={0} stroke={cd.col} strokeWidth={3.5} />
                  <line x1={-8} y1={12} x2={4} y2={12} stroke={cd.col} strokeWidth={3.5} />
                </g>
              )}
              {cd.icon === 'clock' && (
                <g stroke={cd.col} strokeWidth={4} strokeLinecap="round" fill="none">
                  <circle r={24} />
                  <line x1={0} y1={0} x2={0} y2={-15} />
                  <line x1={0} y1={0} x2={12} y2={7} />
                </g>
              )}
            </g>
          </g>
        ))}
        {/* 底部：该去门诊（cue2） */}
        {bannerK > 0 && (
          <g opacity={bannerK} transform={`translate(0,${(1 - bannerK) * 60})`}>
            <rect x={120} y={662} width={1680} height={168} rx={20} fill={P.panel} stroke={P.teal} strokeWidth={4} />
            <text x={160} y={726} fill={P.teal} fontSize={30} fontFamily={FONT_HEAD} fontWeight={900}>
              白天总犯困 + 打鼾 + 晨起头痛</text>
            <text x={160} y={782} fill={P.cream} fontSize={28} fontFamily={FONT_HEAD} fontWeight={700}>
              别硬扛 → 睡眠门诊做多导监测 · 这是真能治的病</text>
            {/* 门诊小楼 */}
            <g transform="translate(1690,746)">
              <rect x={-44} y={-50} width={88} height={100} rx={12} fill={P.bgDeep}
                stroke={P.teal} strokeWidth={4} />
              <rect x={-10} y={-32} width={20} height={64} fill={P.teal} />
              <rect x={-32} y={-10} width={64} height={20} fill={P.teal} />
            </g>
          </g>
        )}
      </svg>
      <Chip text="睡得多 ≠ 睡得好" x={1330} y={72} at={beat(0) + 25} accent={P.teal} />
      <Stamp text="白天状态说了算" x={620} y={90} at={b2 + 15} color={P.tealD} size={38} rot={-2} />
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
        <Stars n={20} from={0} dur={22} />
        <Moon x={330} y={220} scale={0.8} />
        <g transform={`translate(960,790) scale(${pop})`} opacity={clamp01(pop)}>
          <BedSleeper x={0} y={0} s={1.05} zzz={false} />
          <Zzz x={140} y={-140} k={0.9} />
        </g>
        {/* 质量达标小牌 */}
        <g transform={`translate(1470,740) scale(${pop})`} opacity={clamp01(pop)}>
          <rect x={-170} y={-52} width={340} height={104} rx={18} fill={P.panel} stroke={P.teal} strokeWidth={4} />
          <MarkV x={-128} y={0} s={1.2} />
          <text x={20} y={-4} textAnchor="middle" fontSize={26} fill={P.cream}
            fontFamily={FONT_HEAD} fontWeight={800}>今晚睡个</text>
          <text x={20} y={34} textAnchor="middle" fontSize={26} fill={P.teal}
            fontFamily={FONT_HEAD} fontWeight={900}>高质量的觉</text>
        </g>
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -300}}>
        <div style={{fontSize: 54, fontWeight: 800, color: C.text}}>别数睡了几小时 · 数数睡得好不好</div>
        <div style={{fontSize: 36, color: P.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E237 = ({data = tl}: {data?: any}) => {
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
