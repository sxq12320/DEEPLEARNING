import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01} from '../lib/style';
import {Backdrop, Stamp, SceneTag, Chip, Subtitles, ProgressBar, FadeIn, Moon, Stars} from '../lib/atoms';
import tl from '../timeline_236.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：凌晨靛夜 × 闹钟暖黄 × 皮质橙 × 腺苷青 ============ */
const P = {
  bg: '#1a1d40',
  bgDeep: '#10122c',
  bgLite: '#252a56',
  panel: '#141736',
  card: '#2f3468',
  ink: '#101226',
  cream: '#f4eeda',
  muted: '#a9b0da',
  amber: '#ffd166',
  amberD: '#e0a83c',
  coral: '#ff7d6e',
  coralD: '#d8564c',
  teal: '#5fdcb4',
  tealD: '#37a886',
  sky: '#62c6ff',
  lav: '#b79cff',
  hair: '#39304e',
};

/* ---------- 小叉 / 小勾 ---------- */
const MarkX = ({x, y, s = 1, col = P.coral}: {x: number; y: number; s?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} stroke={col} strokeWidth={7} strokeLinecap="round">
    <line x1={-13} y1={-13} x2={13} y2={13} /><line x1={13} y1={-13} x2={-13} y2={13} />
  </g>
);


/* ---------- 飘z ---------- */
const Zzz = ({x, y, k = 1, col = P.sky}: {x: number; y: number; k?: number; col?: string}) => {
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

/* ---------- 闹钟（指针 3:40） ---------- */
const Clock340 = ({x, y, r = 62, ring = true}: {x: number; y: number; r?: number; ring?: boolean}) => {
  const f = useCurrentFrame();
  const shake = ring ? Math.sin(f * 0.8) * 2.5 : 0;
  const hA = ((3 + 40 / 60) * 30 - 90) * Math.PI / 180;
  const mA = (40 * 6 - 90) * Math.PI / 180;
  return (
    <g transform={`translate(${x},${y}) rotate(${shake})`}>
      {/* 铃铛 + 脚 */}
      <circle cx={-r * 0.55} cy={-r * 0.92} r={r * 0.3} fill={P.amberD} />
      <circle cx={r * 0.55} cy={-r * 0.92} r={r * 0.3} fill={P.amberD} />
      <line x1={-r * 0.5} y1={r * 0.88} x2={-r * 0.72} y2={r * 1.18} stroke={P.amberD} strokeWidth={7} strokeLinecap="round" />
      <line x1={r * 0.5} y1={r * 0.88} x2={r * 0.72} y2={r * 1.18} stroke={P.amberD} strokeWidth={7} strokeLinecap="round" />
      <circle r={r} fill={P.cream} stroke={P.amberD} strokeWidth={8} />
      {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11].map((i) => (
        <line key={i} x1={0} y1={-r + 10} x2={0} y2={-r + (i % 3 === 0 ? 22 : 16)}
          stroke={P.ink} strokeWidth={i % 3 === 0 ? 5 : 3} transform={`rotate(${i * 30})`} strokeLinecap="round" />
      ))}
      <line x1={0} y1={0} x2={Math.cos(hA) * r * 0.48} y2={Math.sin(hA) * r * 0.48}
        stroke={P.ink} strokeWidth={8} strokeLinecap="round" />
      <line x1={0} y1={0} x2={Math.cos(mA) * r * 0.68} y2={Math.sin(mA) * r * 0.68}
        stroke={P.coralD} strokeWidth={6} strokeLinecap="round" />
      <circle r={6} fill={P.coralD} />
    </g>
  );
};

/* ---------- 床上坐起的人（醒了） ---------- */
const Sitter = ({x, y, s = 1}: {x: number; y: number; s?: number}) => {
  const f = useCurrentFrame();
  const br = Math.sin(f * 0.14) * 3;
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      {/* 床 */}
      <rect x={-280} y={-8} width={560} height={32} rx={16} fill={P.card} />
      <rect x={-262} y={24} width={22} height={60} rx={9} fill={P.card} />
      <rect x={240} y={24} width={22} height={60} rx={9} fill={P.card} />
      {/* 立着的枕头 */}
      <rect x={-252} y={-122} width={120} height={66} rx={26} fill={P.cream}
        transform="rotate(-18 -192 -89)" />
      {/* 盖腿的被子 */}
      <path d={`M -52 -6 Q 40 ${-78 + br} 258 -10 L 258 8 L -52 8 Z`} fill={P.sky} opacity={0.92} />
      <path d={`M -52 -6 Q 40 ${-78 + br} 258 -10`} fill="none" stroke={P.cream}
        strokeWidth={5} strokeLinecap="round" opacity={0.5} />
      {/* 坐着的人：睡衣躯干 */}
      <rect x={-162} y={-172} width={106} height={174} rx={44} fill={P.lav} />
      <rect x={-162} y={-172} width={106} height={26} rx={13} fill={P.card} opacity={0.35} />
      {/* 搭在被子上的手臂 */}
      <rect x={-56} y={-96} width={150} height={26} rx={13} fill={C.skin} transform="rotate(-5 20 -83)" />
      <circle cx={98} cy={-104} r={16} fill={C.skin} />
      {/* 头：眼睛睁着 */}
      <g transform="translate(-110,-212)">
        <circle r={37} fill={C.skin} />
        <path d="M -36 -10 Q 0 -52 36 -10 L 36 0 Q 0 -24 -36 0 Z" fill={P.hair} />
        <circle cx={-14} cy={-2} r={4.6} fill={P.ink} />
        <circle cx={14} cy={-2} r={4.6} fill={P.ink} />
        <path d="M -20 -14 l 11 -3 M 9 -17 l 11 3" stroke={P.hair} strokeWidth={3.4} strokeLinecap="round" />
        <path d="M -8 16 q 8 5 16 0" stroke={P.ink} strokeWidth={3.4} fill="none" strokeLinecap="round" />
        {/* 眼袋 */}
        <path d="M -19 6 q 5 4 10 0 M 9 6 q 5 4 10 0" stroke={C.skinDark} strokeWidth={2.6}
          fill="none" strokeLinecap="round" opacity={0.8} />
      </g>
    </g>
  );
};

/* ---------- 躺着睡的人 ---------- */
const Sleeper = ({x, y, s = 1, awake = false}: {x: number; y: number; s?: number; awake?: boolean}) => {
  const f = useCurrentFrame();
  const br = Math.sin(f * 0.14) * 3;
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      <rect x={-280} y={-8} width={560} height={32} rx={16} fill={P.card} />
      <rect x={-262} y={24} width={22} height={60} rx={9} fill={P.card} />
      <rect x={240} y={24} width={22} height={60} rx={9} fill={P.card} />
      <rect x={-244} y={-64} width={150} height={56} rx={24} fill={P.cream} />
      <path d={`M -70 -8 Q 20 ${-102 + br} 216 -12 L 258 -8 L 258 8 L -70 8 Z`} fill={P.lav} opacity={0.92} />
      <path d={`M -70 -8 Q 20 ${-102 + br} 216 -12`} fill="none" stroke={P.cream}
        strokeWidth={5} strokeLinecap="round" opacity={0.5} />
      <g transform="translate(-158,-92)">
        <circle r={34} fill={C.skin} />
        <path d="M -33 -8 Q 0 -46 33 -8 L 33 2 Q 0 -22 -33 2 Z" fill={P.hair} />
        {awake ? (
          <g>
            <circle cx={-13} cy={2} r={4.4} fill={P.ink} />
            <circle cx={13} cy={2} r={4.4} fill={P.ink} />
            <path d="M -18 -10 l 10 -3 M 8 -13 l 10 3" stroke={P.hair} strokeWidth={3} strokeLinecap="round" />
          </g>
        ) : (
          <path d="M -16 4 q 7 7 14 0 M 6 4 q 7 7 14 0" stroke={P.ink} strokeWidth={3.2}
            fill="none" strokeLinecap="round" />
        )}
      </g>
      {!awake && <Zzz x={-96} y={-140} k={0.85} />}
    </g>
  );
};

/* ---------- 床头柜 ---------- */
const Nightstand = ({x, y}: {x: number; y: number}) => (
  <g transform={`translate(${x},${y})`}>
    <rect x={-58} y={-84} width={116} height={84} rx={12} fill={P.card} />
    <rect x={-58} y={-84} width={116} height={16} rx={8} fill={P.muted} opacity={0.3} />
    <circle cy={-38} r={8} fill={P.amberD} />
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
        <Moon x={250} y={160} scale={0.7} />
        {/* 中央：半夜坐起来的人 + 床头柜上的闹钟 3:40 */}
        <g transform={`translate(960,470) scale(${pA}) translate(-960,-470)`} opacity={clamp01(pA)}>
          <Sitter x={830} y={470} s={1.06} />
          <Nightstand x={1400} y={494} />
          <Clock340 x={1400} y={368} r={58} />
          <text x={1400} y={272} textAnchor="middle" fontSize={34} fill={P.amber}
            fontFamily={FONT_HEAD} fontWeight={900}>3:40</text>
          {/* 头上感叹号 */}
          <g transform="translate(660,196)">
            <circle r={30} fill={P.coral} />
            <text y={12} textAnchor="middle" fontSize={34} fill={P.cream}
              fontFamily={FONT_HEAD} fontWeight={900}>!</text>
          </g>
        </g>
        {/* 两侧小标签 */}
        <g transform={`translate(500,600) scale(${pB})`} opacity={clamp01(pB)}>
          <TagPill x={0} y={0} t="想睡" col={P.teal} w={150} />
        </g>
        <g transform={`translate(1420,600) scale(${pB})`} opacity={clamp01(pB)}>
          <TagPill x={0} y={0} t="睡不回去" col={P.coral} w={220} />
          <MarkX x={132} y={-6} s={0.85} />
        </g>
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 470}}>
          <div style={{color: P.sky, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 24}}>
            冷知识 · 第 {data.episode} 期
          </div>
          <div style={{color: C.text, fontSize: 74, fontWeight: 900, fontFamily: FONT_HEAD, letterSpacing: 4}}>
            {data.title}
          </div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ============ S0 睁眼到天亮：正式名 + 两股力夹缝 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const popL = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const popR = Math.max(0.001, spring({frame: f - 45, fps, config: {damping: 11, stiffness: 120}}));
  const nameK = clamp01((f - 70) / 24);
  const conK = clamp01((f - b1 - 5) / 24);

  const subtypes = [
    {t: '入睡困难', on: false},
    {t: '睡眠维持 · 凌晨醒', on: true},
    {t: '多梦 / 早醒', on: false},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={0} name={sec.name} accent={P.amber} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={10} from={0} dur={26} />
        {/* 左：半夜坐起来的人 */}
        <g transform={`translate(120,150) scale(${popL}) translate(-120,-150)`} opacity={clamp01(popL * 2)}>
          <rect x={120} y={150} width={800} height={480} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={156} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>凌晨三四点 · 又醒了</text>
          <Moon x={840} y={230} scale={0.4} opacity={0.9} />
          <Sitter x={370} y={540} s={0.92} />
          <Nightstand x={742} y={562} />
          <Clock340 x={742} y={438} r={54} />
          <text x={742} y={348} textAnchor="middle" fontSize={32} fill={P.amber}
            fontFamily={FONT_HEAD} fontWeight={900}>3:40</text>
          <TagPill x={300} y={290} t="眼睁睁看天亮" col={P.coral} w={260} fs={26} s={0.9} />
        </g>
        {/* 右：它有正式名字 */}
        <g transform={`translate(960,150) scale(${popR}) translate(-960,-150)`} opacity={clamp01(popR * 2)}>
          <rect x={960} y={150} width={840} height={480} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={996} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>不是「想多了」 · 它有正式名字</text>
          {/* 大名 */}
          <g opacity={nameK} transform={`translate(0,${(1 - nameK) * 30})`}>
            <rect x={992} y={236} width={776} height={112} rx={18} fill={P.bgDeep} stroke={P.amber} strokeWidth={4} />
            <text x={1380} y={308} textAnchor="middle" fontSize={54} fill={P.amber}
              fontFamily={FONT_HEAD} fontWeight={900}>睡眠维持性失眠</text>
          </g>
          <text x={1380} y={392} textAnchor="middle" fill={P.muted} fontSize={24} fontFamily={FONT_HEAD}>
            失眠的一种正式亚型 · 和入睡困难是亲戚不是一回事</text>
          {/* 三个亚型胶囊 */}
          {subtypes.map((s2, i) => (
            <g key={i} transform={`translate(${1130 + i * 255},456)`}>
              <rect x={-118} y={-30} width={236} height={60} rx={30}
                fill={s2.on ? P.coral : 'transparent'} stroke={s2.on ? P.coral : P.card} strokeWidth={4} />
              <text x={0} y={10} textAnchor="middle" fontSize={24}
                fill={s2.on ? P.cream : P.muted} fontFamily={FONT_HEAD} fontWeight={s2.on ? 900 : 600}>{s2.t}</text>
            </g>
          ))}
          <text x={1380} y={560} textAnchor="middle" fill={P.cream} fontSize={27}
            fontFamily={FONT_HEAD} fontWeight={700}>卡在「不想醒」和「睡不动」中间的尴尬时点</text>
        </g>
        {/* 底部：两股力夹缝（cue1） */}
        {conK > 0 && (
          <g opacity={conK} transform={`translate(0,${(1 - conK) * 60})`}>
            <rect x={120} y={668} width={1680} height={190} rx={20} fill={P.panel} stroke={P.amber} strokeWidth={4} />
            <text x={160} y={726} fill={P.amber} fontSize={30} fontFamily={FONT_HEAD} fontWeight={900}>
              两股力的夹缝</text>
            {/* 左力：睡眠压力撤了 */}
            <g transform="translate(160,756)">
              <rect x={0} y={0} width={500} height={76} rx={16} fill={P.bgDeep} stroke={P.teal} strokeWidth={3.5} />
              <rect x={18} y={22} width={180} height={32} rx={16} fill={P.card} />
              <rect x={18} y={22} width={30} height={32} rx={15} fill={P.teal} />
              <text x={216} y={47} fill={P.teal} fontSize={24} fontFamily={FONT_HEAD} fontWeight={800}>
                睡眠压力 · 快还清了</text>
            </g>
            {/* 中间小人被夹 */}
            <g transform="translate(960,794)">
              <line x1={-240} y1={0} x2={-70} y2={0} stroke={P.teal} strokeWidth={6} strokeLinecap="round" strokeDasharray="10 9" />
              <line x1={240} y1={0} x2={70} y2={0} stroke={P.coral} strokeWidth={6} strokeLinecap="round" />
              <polygon points="70,0 92,-10 92,10" fill={P.coral} />
              <circle cy={-26} r={20} fill={C.skin} />
              <rect x={-22} y={-6} width={44} height={40} rx={16} fill={P.lav} />
              <text y={66} textAnchor="middle" fontSize={22} fill={P.cream} fontFamily={FONT_HEAD} fontWeight={800}>
                你 · 卡在中间</text>
            </g>
            {/* 右力：生物钟爬坡 */}
            <g transform="translate(1300,756)">
              <rect x={0} y={0} width={500} height={76} rx={16} fill={P.bgDeep} stroke={P.coral} strokeWidth={3.5} />
              <path d="M 20 54 L 130 54 L 200 22" fill="none" stroke={P.coral} strokeWidth={6} strokeLinecap="round" />
              <circle cx={200} cy={22} r={7} fill={P.amber} />
              <text x={222} y={47} fill={P.coral} fontSize={24} fontFamily={FONT_HEAD} fontWeight={800}>
                生物钟 · 往清晨爬</text>
            </g>
          </g>
        )}
      </svg>
      <Chip text="眼睁睁看天亮" x={1290} y={72} at={beat(0) + 25} accent={P.amber} />
      <Stamp text="失眠亚型 · 不是觉轻" x={620} y={86} at={b1 + 30} color={P.coralD} size={38} rot={-3} />
    </FadeIn>
  );
};

/* ============ S1 后半夜本来就浅：整夜睡眠深度曲线 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const pop = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const drawK = clamp01(f / 80);
  const zoneK = clamp01((f - 90) / 24);
  const hitK = clamp01((f - b1 + 20) / 22);
  const conK = clamp01((f - b1) / 24);

  // 整夜深度阶梯：16 列，y 越大越深
  const yVals = [560, 575, 545, 565, 470, 520, 425, 465, 390, 425, 375, 412, 365, 398, 370, 392];
  const x0 = 250, colW = 90, base = 636;
  let d = `M ${x0} ${yVals[0]}`;
  for (let i = 1; i < yVals.length; i++) d += ` L ${x0 + i * colW} ${yVals[i - 1]} L ${x0 + i * colW} ${yVals[i]}`;
  d += ` L ${x0 + yVals.length * colW} ${yVals[yVals.length - 1]}`;
  const fillD = `${d} L ${x0 + yVals.length * colW} ${base} L ${x0} ${base} Z`;
  const shown = Math.floor((drawK * (x0 + yVals.length * colW - x0)) / colW) * colW + x0;

  const times = ['23点', '0点', '1点', '2点', '3点', '4点', '5点', '6点', '7点'];
  const nuis = [
    {t: '一点声音', col: P.sky},
    {t: '一丝亮光', col: P.amber},
    {t: '一个念头', col: P.lav},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={1} name={sec.name} accent={P.sky} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <g transform={`translate(120,150) scale(${pop}) translate(-120,-150)`} opacity={clamp01(pop * 2)}>
          <rect x={120} y={150} width={1680} height={520} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={156} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            整夜睡眠深度 · 深睡几乎全压在前半夜</text>
          {/* 凌晨高亮带 */}
          <g opacity={zoneK}>
            <rect x={1258} y={240} width={360} height={396} rx={14} fill={P.coral} opacity={0.12} />
            <rect x={1258} y={240} width={360} height={44} rx={14} fill={P.coral} opacity={0.85} />
            <text x={1438} y={270} textAnchor="middle" fontSize={24} fill={P.cream}
              fontFamily={FONT_HEAD} fontWeight={900}>凌晨三四点 · 浅睡为主</text>
          </g>
          {/* 吵醒门槛 */}
          <line x1={x0} y1={352} x2={1690} y2={352} stroke={P.coral} strokeWidth={5}
            strokeDasharray="14 12" opacity={0.9} />
          <text x={x0 + 8} y={338} fill={P.coral} fontSize={24} fontFamily={FONT_HEAD} fontWeight={800}>
            吵醒门槛</text>
          {/* 深度曲线 */}
          <clipPath id="s1clip"><rect x={x0} y={300} width={shown - x0} height={340} /></clipPath>
          <g clipPath="url(#s1clip)">
            <path d={fillD} fill={P.sky} opacity={0.85} />
            {/* 深睡带：底部染色 */}
            <rect x={x0} y={490} width={yVals.length * colW} height={base - 490} fill={P.lav} opacity={0.28} />
          </g>
          <path d={d} fill="none" stroke={P.cream} strokeWidth={6} strokeLinejoin="round"
            clipPath="url(#s1clip)" />
          {/* 阶段标签 */}
          <text x={330} y={600} fill={P.ink} fontSize={26} fontFamily={FONT_HEAD} fontWeight={900}>深睡</text>
          <text x={1180} y={452} fill={P.ink} fontSize={24} fontFamily={FONT_HEAD} fontWeight={800}>浅睡</text>
          <text x={1510} y={420} fill={P.ink} fontSize={24} fontFamily={FONT_HEAD} fontWeight={800}>做梦期</text>
          {/* 前半夜：离门槛很远 */}
          <g opacity={zoneK}>
            <line x1={560} y1={366} x2={560} y2={520} stroke={P.teal} strokeWidth={5}
              strokeLinecap="round" strokeDasharray="2 12" />
            <polygon points="560,352 548,374 572,374" fill={P.teal} />
            <polygon points="560,534 548,512 572,512" fill={P.teal} />
            <text x={588} y={452} fill={P.teal} fontSize={23} fontFamily={FONT_HEAD} fontWeight={800}>
              前半夜 · 动静够不着</text>
          </g>
          {/* 后半夜：小动作越线 */}
          {hitK > 0 && (
            <g opacity={hitK}>
              {[0, 1, 2].map((i) => {
                const cx = 1330 + i * 120;
                return (
                  <g key={i} transform={`translate(${cx},300)`}>
                    <circle r={22} fill={nuis[i].col} />
                    <text y={8} textAnchor="middle" fontSize={20} fill={P.ink}
                      fontFamily={FONT_HEAD} fontWeight={900}>{['声', '光', '想'][i]}</text>
                    <line x1={0} y1={26} x2={0} y2={46} stroke={P.coral} strokeWidth={5} strokeLinecap="round" />
                    <polygon points="0,58 -9,44 9,44" fill={P.coral} />
                  </g>
                );
              })}
              <circle cx={1450} cy={368} r={18} fill={P.coral} opacity={0.9} />
              <text x={1450} y={376} textAnchor="middle" fontSize={18} fill={P.cream}
                fontFamily={FONT_HEAD} fontWeight={900}>醒</text>
            </g>
          )}
          {/* 时间轴 */}
          {times.map((t, i) => (
            <text key={i} x={x0 + i * (colW * 2)} y={base + 38} fill={P.muted}
              fontSize={22} fontFamily={FONT_HEAD}>{t}</text>
          ))}
        </g>
        {/* 底部结论（cue1） */}
        {conK > 0 && (
          <g opacity={conK} transform={`translate(0,${(1 - conK) * 60})`}>
            <rect x={120} y={710} width={1680} height={150} rx={20} fill={P.panel} stroke={P.sky} strokeWidth={4} />
            <text x={160} y={768} fill={P.sky} fontSize={30} fontFamily={FONT_HEAD} fontWeight={900}>
              凌晨醒不算突然 —— 你本来就躺在浅层</text>
            {nuis.map((n, i) => (
              <g key={i} transform={`translate(${980 + i * 250},780)`}>
                <rect x={-104} y={-30} width={208} height={60} rx={30}
                  fill={P.bgDeep} stroke={n.col} strokeWidth={3.5} />
                <text x={0} y={9} textAnchor="middle" fontSize={24} fill={n.col}
                  fontFamily={FONT_HEAD} fontWeight={800}>{n.t}</text>
              </g>
            ))}
            <text x={160} y={824} fill={P.cream} fontSize={26} fontFamily={FONT_HEAD} fontWeight={700}>
              一点动静都够把你拽出来 · 这是结构特性不是神经衰弱</text>
          </g>
        )}
      </svg>
      <Chip text="后半夜 = 浅睡+做梦期" x={1260} y={72} at={beat(0) + 25} accent={P.sky} />
      <Stamp text="门槛低了 · 一碰就醒" x={620} y={86} at={b1 + 40} color={P.coralD} size={38} rot={-3} />
    </FadeIn>
  );
};

/* ============ S2 两股力都松手了：腺苷见底 + 皮质醇爬坡 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const popL = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const popR = Math.max(0.001, spring({frame: f - b1 - 5, fps, config: {damping: 11, stiffness: 120}}));
  const drainK = clamp01(f / 110);
  const level = 0.95 - drainK * 0.82;
  const rampK = clamp01((f - b1 - 25) / 40);
  const stressK = clamp01((f - b1 - 70) / 26);
  const conK = clamp01((f - b1 - 90) / 24);

  const cort = `M 1010 502 L 1360 502 C 1430 502 1470 470 1522 428 C 1580 382 1646 320 1700 300`;
  const cortLen = 900;
  const stress = `M 1360 502 C 1420 452 1480 356 1560 312`;

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={2} name={sec.name} accent={P.teal} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：睡眠压力账户见底 */}
        <g transform={`translate(120,150) scale(${popL}) translate(-120,-150)`} opacity={clamp01(popL * 2)}>
          <rect x={120} y={150} width={790} height={480} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={156} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            睡眠压力 · 腺苷攒的账</text>
          {/* 竖罐 */}
          <rect x={270} y={250} width={140} height={320} rx={26} fill={P.bgDeep} stroke={P.teal} strokeWidth={5} />
          <rect x={286} y={266 + 288 * (1 - level)} width={108} height={288 * level} rx={18}
            fill={P.teal} opacity={0.9} />
          <text x={340} y={606} textAnchor="middle" fontSize={24} fill={P.teal}
            fontFamily={FONT_HEAD} fontWeight={800}>腺苷存量</text>
          {/* 刻度 */}
          <line x1={424} y1={266} x2={470} y2={266} stroke={P.muted} strokeWidth={3} strokeDasharray="6 6" />
          <text x={478} y={274} fill={P.muted} fontSize={21} fontFamily={FONT_HEAD}>入睡时 · 满</text>
          <line x1={424} y1={530} x2={470} y2={530} stroke={P.coral} strokeWidth={3} strokeDasharray="6 6" />
          <text x={478} y={522} fill={P.coral} fontSize={21} fontFamily={FONT_HEAD} fontWeight={800}>
            凌晨 · 见底</text>
          {/* 状态切换 */}
          <g transform="translate(640,300)">
            <rect x={-120} y={-30} width={240} height={60} rx={30} fill={P.bgDeep}
              stroke={P.teal} strokeWidth={3.5} opacity={1 - drainK * 0.5} />
            <text x={0} y={10} textAnchor="middle" fontSize={26} fill={P.teal}
              fontFamily={FONT_HEAD} fontWeight={900} opacity={1 - drainK * 0.4}>必须睡</text>
            {drainK > 0.6 && (
              <line x1={-86} y1={0} x2={86} y2={0} stroke={P.coral} strokeWidth={6}
                strokeLinecap="round" opacity={(drainK - 0.6) * 2.5} />
            )}
            <line x1={0} y1={36} x2={0} y2={76} stroke={P.muted} strokeWidth={4} strokeDasharray="6 6" />
            <rect x={-140} y={84} width={280} height={60} rx={30}
              fill={drainK > 0.75 ? P.coral : P.bgDeep} stroke={P.coral} strokeWidth={3.5} />
            <text x={0} y={122} textAnchor="middle" fontSize={26}
              fill={drainK > 0.75 ? P.cream : P.coral} fontFamily={FONT_HEAD} fontWeight={900}>可睡可不睡</text>
          </g>
          <text x={650} y={592} textAnchor="middle" fill={P.muted} fontSize={23} fontFamily={FONT_HEAD}>
            按住你睡的那股劲 · 撤了</text>
        </g>
        {/* 右：皮质醇爬坡 */}
        <g transform={`translate(950,150) scale(${popR}) translate(-950,-150)`} opacity={clamp01(popR * 2)}>
          <rect x={950} y={150} width={850} height={480} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={986} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            皮质醇 · 不管你睡没睡够到点爬坡</text>
          <line x1={1010} y1={502} x2={1760} y2={502} stroke={P.card} strokeWidth={4} />
          <line x1={1010} y1={502} x2={1010} y2={280} stroke={P.card} strokeWidth={4} />
          {/* 爬坡主线（描线动画） */}
          <path d={cort} fill="none" stroke={P.coral} strokeWidth={8} strokeLinecap="round"
            strokeDasharray={cortLen} strokeDashoffset={cortLen * (1 - rampK)} />
          {/* 虚线标记：三四点开爬 */}
          <line x1={1360} y1={502} x2={1360} y2={288} stroke={P.amber} strokeWidth={3}
            strokeDasharray="7 8" opacity={rampK} />
          <text x={1360} y={540} textAnchor="middle" fontSize={22} fill={P.amber}
            fontFamily={FONT_HEAD} fontWeight={800} opacity={rampK}>凌晨三四点</text>
          {/* 小太阳露头 */}
          {rampK > 0.6 && (
            <g opacity={clamp01((rampK - 0.6) * 2.5)} transform="translate(1706,286)">
              {[0, 1, 2, 3, 4, 5, 6, 7].map((i) => (
                <line key={i} x1={0} y1={-30} x2={0} y2={-44} stroke={P.amber} strokeWidth={5}
                  strokeLinecap="round" transform={`rotate(${i * 45})`} />
              ))}
              <circle r={24} fill={P.amber} />
            </g>
          )}
          {/* 压力大的更陡坡 */}
          {stressK > 0 && (
            <g opacity={stressK}>
              <path d={stress} fill="none" stroke={P.amber} strokeWidth={5}
                strokeDasharray="10 9" strokeLinecap="round" />
              <rect x={1440} y={336} width={250} height={52} rx={14} fill={P.bgDeep}
                stroke={P.amber} strokeWidth={3.5} />
              <text x={1565} y={370} textAnchor="middle" fontSize={22} fill={P.amber}
                fontFamily={FONT_HEAD} fontWeight={800}>压力大 · 坡更陡</text>
            </g>
          )}
          <text x={1010} y={540} fill={P.muted} fontSize={21} fontFamily={FONT_HEAD}>23点</text>
          <text x={1700} y={540} fill={P.muted} fontSize={21} fontFamily={FONT_HEAD}>早上</text>
          <text x={1380} y={596} textAnchor="middle" fill={P.cream} fontSize={26}
            fontFamily={FONT_HEAD} fontWeight={700}>这个坡本来是叫你起床的 · 却先把你拽醒了</text>
        </g>
        {/* 底部合力（cue1 后段） */}
        {conK > 0 && (
          <g opacity={conK} transform={`translate(0,${(1 - conK) * 60})`}>
            <rect x={120} y={668} width={1680} height={180} rx={20} fill={P.panel} stroke={P.teal} strokeWidth={4} />
            <text x={160} y={730} fill={P.teal} fontSize={30} fontFamily={FONT_HEAD} fontWeight={900}>
              一边不拦 · 一边拽 —— 可不就醒了</text>
            <g transform="translate(160,760)">
              <rect width={460} height={62} rx={16} fill={P.bgDeep} stroke={P.teal} strokeWidth={3.5} />
              <text x={230} y={40} textAnchor="middle" fontSize={25} fill={P.teal}
                fontFamily={FONT_HEAD} fontWeight={800}>睡眠压力 ≈ 0 · 不拦你</text>
            </g>
            <text x={700} y={802} fontSize={40} fill={P.muted} fontFamily={FONT_HEAD}>+</text>
            <g transform="translate(760,760)">
              <rect width={460} height={62} rx={16} fill={P.bgDeep} stroke={P.coral} strokeWidth={3.5} />
              <text x={230} y={40} textAnchor="middle" fontSize={25} fill={P.coral}
                fontFamily={FONT_HEAD} fontWeight={800}>皮质醇爬坡 · 拽你醒</text>
            </g>
            <text x={1300} y={802} fontSize={40} fill={P.muted} fontFamily={FONT_HEAD}>=</text>
            <g transform="translate(1380,748)">
              <rect width={380} height={86} rx={18} fill={P.coral} />
              <text x={190} y={55} textAnchor="middle" fontSize={30} fill={P.cream}
                fontFamily={FONT_HEAD} fontWeight={900}>凌晨准时睁眼</text>
            </g>
          </g>
        )}
      </svg>
      <Chip text="腺苷还清 + 皮质醇爬坡" x={1240} y={72} at={beat(0) + 25} accent={P.teal} />
      <Stamp text="可睡可不睡的尴尬态" x={600} y={86} at={b1 + 40} color={P.coralD} size={38} rot={-3} />
    </FadeIn>
  );
};

/* ============ S3 焦虑是放大器：看表算账 + 恶性循环 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const popL = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const popR = Math.max(0.001, spring({frame: f - 40, fps, config: {damping: 11, stiffness: 120}}));
  const bubK = clamp01((f - 70) / 24);
  const ampK = clamp01((f - 100) / 26);
  const conK = clamp01((f - b1) / 24);
  const ringA = f * 1.4;

  return (
    <FadeIn>
      <Backdrop top={P.bgDeep} bottom={P.bgDeep} />
      <SceneTag index={3} name={sec.name} accent={P.coral} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={8} from={0} dur={26} />
        {/* 左：躺着看表的人 */}
        <g transform={`translate(120,150) scale(${popL}) translate(-120,-150)`} opacity={clamp01(popL * 2)}>
          <rect x={120} y={150} width={790} height={480} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={156} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            醒来第一件事 · 看表</text>
          <Sleeper x={400} y={498} s={0.92} awake={true} />
          {/* 举着的闹钟 */}
          <Clock340 x={660} y={350} r={56} ring={true} />
          <text x={660} y={256} textAnchor="middle" fontSize={32} fill={P.amber}
            fontFamily={FONT_HEAD} fontWeight={900}>3:40</text>
          {/* 念头泡 */}
          {bubK > 0 && (
            <g opacity={bubK}>
              <circle cx={262} cy={382} r={10} fill={P.cream} />
              <circle cx={232} cy={344} r={15} fill={P.cream} />
              <rect x={136} y={222} width={300} height={104} rx={40} fill={P.cream} />
              <text x={286} y={264} textAnchor="middle" fontSize={23} fill={P.ink}
                fontFamily={FONT_HEAD} fontWeight={800}>距闹钟还有 3 小时</text>
              <text x={286} y={298} textAnchor="middle" fontSize={23} fill={P.coralD}
                fontFamily={FONT_HEAD} fontWeight={900}>明天要废了</text>
            </g>
          )}
          {/* 心率监护条 */}
          <rect x={150} y={578} width={720} height={48} rx={10} fill={P.bgDeep} stroke={P.card} strokeWidth={3} />
          <polyline points="170,602 280,602 316,586 352,614 388,584 424,612 470,602 640,602"
            fill="none" stroke={P.coral} strokeWidth={4.5} strokeLinecap="round" strokeLinejoin="round" />
          <text x={852} y={610} textAnchor="end" fill={P.coral} fontSize={22}
            fontFamily={FONT_HEAD} fontWeight={800}>心率↑ 警觉↑</text>
        </g>
        {/* 右：焦虑放大器 */}
        <g transform={`translate(950,150) scale(${popR}) translate(-950,-150)`} opacity={clamp01(popR * 2)}>
          <rect x={950} y={150} width={850} height={480} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={986} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            这笔账本身 · 就是最猛的清醒剂</text>
          {/* 放大器 */}
          <g transform="translate(0,0)">
            <rect x={986} y={272} width={220} height={64} rx={32} fill={P.bgDeep}
              stroke={P.muted} strokeWidth={3.5} />
            <text x={1096} y={312} textAnchor="middle" fontSize={24} fill={P.muted}
              fontFamily={FONT_HEAD} fontWeight={800}>一点担心</text>
            <line x1={1210} y1={304} x2={1268} y2={304} stroke={P.muted} strokeWidth={5} strokeLinecap="round" />
            <polygon points="1282,304 1264,294 1264,314" fill={P.muted} />
            {ampK > 0 && (
              <g opacity={ampK}>
                <polygon points="1300,240 1300,368 1440,304" fill={P.bgDeep} stroke={P.coral} strokeWidth={5}
                  strokeLinejoin="round" />
                <text x={1350} y={316} textAnchor="middle" fontSize={26} fill={P.coral}
                  fontFamily={FONT_HEAD} fontWeight={900}>焦虑</text>
                <line x1={1448} y1={304} x2={1506} y2={304} stroke={P.coral} strokeWidth={6} strokeLinecap="round" />
                <polygon points="1522,304 1502,292 1502,316" fill={P.coral} />
                <rect x={1536} y={258} width={230} height={92} rx={18} fill={P.coral} />
                <text x={1651} y={296} textAnchor="middle" fontSize={30} fill={P.cream}
                  fontFamily={FONT_HEAD} fontWeight={900}>彻底清醒</text>
                <text x={1651} y={330} textAnchor="middle" fontSize={20} fill={P.cream}
                  fontFamily={FONT_HEAD}>交感兴奋 · 越努力越醒</text>
              </g>
            )}
          </g>
          {/* 恶性循环环 */}
          <g transform="translate(1210,500)">
            <circle r={82} fill="none" stroke={P.card} strokeWidth={5} strokeDasharray="10 10" />
            <g transform={`rotate(${ringA})`}>
              <polygon points="0,-92 15,-70 -15,-70" fill={P.coral} />
            </g>
            <g transform="translate(0,-82)">
              <circle r={26} fill={P.bgDeep} stroke={P.coral} strokeWidth={4} />
              <text y={8} textAnchor="middle" fontSize={20} fill={P.coral} fontFamily={FONT_HEAD} fontWeight={900}>醒</text>
            </g>
            <g transform="translate(72,40)">
              <circle r={26} fill={P.bgDeep} stroke={P.amber} strokeWidth={4} />
              <text y={8} textAnchor="middle" fontSize={20} fill={P.amber} fontFamily={FONT_HEAD} fontWeight={900}>算</text>
            </g>
            <g transform="translate(-72,40)">
              <circle r={26} fill={P.bgDeep} stroke={P.sky} strokeWidth={4} />
              <text y={8} textAnchor="middle" fontSize={20} fill={P.sky} fontFamily={FONT_HEAD} fontWeight={900}>慌</text>
            </g>
            <text y={8} textAnchor="middle" fontSize={23} fill={P.cream} fontFamily={FONT_HEAD} fontWeight={800}>
              恶性循环</text>
          </g>
          {/* 条件反射 */}
          <g transform="translate(1600,470)" opacity={conK}>
            <rect x={-180} y={-46} width={170} height={66} rx={16} fill={P.bgDeep} stroke={P.sky} strokeWidth={3.5} />
            <text x={-95} y={-4} textAnchor="middle" fontSize={26} fill={P.sky}
              fontFamily={FONT_HEAD} fontWeight={900}>床</text>
            <path d="M -6 -22 q 14 -12 26 0 q 12 12 26 0" fill="none" stroke={P.coral}
              strokeWidth={5} strokeLinecap="round" />
            <rect x={10} y={-46} width={170} height={66} rx={16} fill={P.bgDeep} stroke={P.coral} strokeWidth={3.5} />
            <text x={95} y={-4} textAnchor="middle" fontSize={26} fill={P.coral}
              fontFamily={FONT_HEAD} fontWeight={900}>失眠</text>
            <text x={0} y={66} textAnchor="middle" fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>
              几次下来 · 绑成了条件反射</text>
          </g>
        </g>
        {/* 底部结论（cue1） */}
        {conK > 0 && (
          <g opacity={conK} transform={`translate(0,${(1 - conK) * 60})`}>
            <rect x={120} y={668} width={1680} height={180} rx={20} fill={P.panel} stroke={P.coral} strokeWidth={4} />
            <text x={160} y={730} fill={P.coral} fontSize={30} fontFamily={FONT_HEAD} fontWeight={900}>
              睡不回去的症结不在「醒」 · 在醒来后开始和自己较劲</text>
            <text x={160} y={788} fill={P.cream} fontSize={27} fontFamily={FONT_HEAD} fontWeight={700}>
              睡眠压力在退 · 焦虑在放大 —— 两股力合起来把你钉在清醒里</text>
            <TagPill x={1600} y={762} t="到点就醒" col={P.amber} w={220} fs={28} />
          </g>
        )}
      </svg>
      <Chip text="看表算账 = 最强清醒剂" x={1260} y={72} at={beat(0) + 25} accent={P.coral} />
      <Stamp text="床 → 焦虑触发器" x={620} y={86} at={b1 + 40} color={P.coralD} size={38} rot={-3} />
    </FadeIn>
  );
};

/* ============ S4 别躺在床上硬睡：二十分钟规矩 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const pop = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const stepPop = [0, 1, 2, 3].map((i) => Math.max(0.001, spring({frame: f - 20 - i * 16, fps, config: {damping: 12, stiffness: 140}})));
  const conK = clamp01((f - b1) / 24);
  const docK = clamp01((f - b1 - 50) / 24);

  const steps = [
    {t: '醒了超 20 分钟', s: '还睡不着', icon: 'clock', col: P.amber},
    {t: '离开床', s: '换个昏暗的房间', icon: 'door', col: P.sky},
    {t: '做无聊的事', s: '纸书·叠衣服·播客', icon: 'book', col: P.lav},
    {t: '困了再回来', s: '床只跟困意绑定', icon: 'bed', col: P.teal},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={4} name={sec.name} accent={P.teal} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 顶部流程 */}
        <g transform={`translate(120,150) scale(${pop}) translate(-120,-150)`} opacity={clamp01(pop * 2)}>
          <rect x={120} y={150} width={1680} height={360} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={156} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            失眠认知行为疗法 · 反直觉但实证很强的规矩</text>
          {steps.map((st, i) => {
            const sx = 190 + i * 420;
            return (
              <g key={i} transform={`translate(${sx},250) scale(${stepPop[i]}) translate(${-sx},-250)`}
                opacity={clamp01(stepPop[i] * 2)}>
                <rect x={sx} y={250} width={350} height={220} rx={18} fill={P.bgDeep}
                  stroke={st.col} strokeWidth={4} />
                {/* 图标 */}
                <g transform={`translate(${sx + 60},316)`}>
                  {st.icon === 'clock' && (
                    <g>
                      <circle r={30} fill={P.panel} stroke={st.col} strokeWidth={5} />
                      <line x1={0} y1={0} x2={0} y2={-18} stroke={st.col} strokeWidth={5} strokeLinecap="round" />
                      <line x1={0} y1={0} x2={13} y2={6} stroke={st.col} strokeWidth={5} strokeLinecap="round" />
                    </g>
                  )}
                  {st.icon === 'door' && (
                    <g>
                      <rect x={-26} y={-32} width={52} height={64} rx={6} fill="none" stroke={st.col} strokeWidth={5} />
                      <line x1={-8} y1={0} x2={16} y2={0} stroke={st.col} strokeWidth={5} strokeLinecap="round" />
                      <polygon points="26,0 12,-9 12,9" fill={st.col} />
                    </g>
                  )}
                  {st.icon === 'book' && (
                    <g>
                      <path d="M -28 -22 L -2 -30 L -2 22 L -28 16 Z" fill={st.col} opacity={0.9} />
                      <path d="M 28 -22 L 2 -30 L 2 22 L 28 16 Z" fill={st.col} opacity={0.65} />
                    </g>
                  )}
                  {st.icon === 'bed' && (
                    <g>
                      <rect x={-32} y={-6} width={64} height={14} rx={7} fill={st.col} />
                      <rect x={-32} y={8} width={8} height={14} rx={4} fill={st.col} />
                      <rect x={24} y={8} width={8} height={14} rx={4} fill={st.col} />
                      <circle cx={-18} cy={-14} r={9} fill={C.skin} />
                    </g>
                  )}
                </g>
                <circle cx={sx + 316} cy={250} r={20} fill={st.col} />
                <text x={sx + 316} y={258} textAnchor="middle" fontSize={22} fill={P.ink}
                  fontFamily={FONT_HEAD} fontWeight={900}>{i + 1}</text>
                <text x={sx + 116} y={308} fill={st.col} fontSize={30}
                  fontFamily={FONT_HEAD} fontWeight={900}>{st.t}</text>
                <text x={sx + 116} y={352} fill={P.muted} fontSize={22} fontFamily={FONT_HEAD}>{st.s}</text>
                <text x={sx + 28} y={440} fill={P.muted} fontSize={20} fontFamily={FONT_HEAD}>
                  {['别硬躺 · 躺着焦虑会加固联结', '把「床+清醒+焦虑」拆开', '无聊到困意重新冒头', '床重新只等于困'][i]}
                </text>
                {i < 3 && (
                  <g stroke={P.muted} strokeWidth={5} strokeLinecap="round">
                    <line x1={sx + 358} y1={360} x2={sx + 404} y2={360} />
                    <polygon points={`${sx + 416},360 ${sx + 398},350 ${sx + 398},370`} fill={P.muted} stroke="none" />
                  </g>
                )}
              </g>
            );
          })}
        </g>
        {/* 底部：别做 + 就医提示（cue1） */}
        {conK > 0 && (
          <g opacity={conK} transform={`translate(0,${(1 - conK) * 60})`}>
            <rect x={120} y={560} width={800} height={270} rx={20} fill={P.panel} stroke={P.coral} strokeWidth={4} />
            <text x={156} y={620} fill={P.coral} fontSize={30} fontFamily={FONT_HEAD} fontWeight={900}>
              凌晨醒后 · 三不做</text>
            {[
              {t: '不看时间', icon: 'clock'},
              {t: '不摸手机', icon: 'phone'},
              {t: '不躺着较劲', icon: 'think'},
            ].map((n, i) => (
              <g key={i} transform={`translate(${170 + i * 250},690)`}>
                <rect x={0} y={0} width={230} height={110} rx={16} fill={P.bgDeep}
                  stroke={P.card} strokeWidth={3} />
                <g transform="translate(46,55)">
                  {n.icon === 'clock' && (
                    <g>
                      <circle r={24} fill="none" stroke={P.muted} strokeWidth={4} />
                      <line x1={0} y1={0} x2={0} y2={-14} stroke={P.muted} strokeWidth={4} strokeLinecap="round" />
                      <line x1={0} y1={0} x2={10} y2={5} stroke={P.muted} strokeWidth={4} strokeLinecap="round" />
                    </g>
                  )}
                  {n.icon === 'phone' && (
                    <g>
                      <rect x={-14} y={-28} width={28} height={56} rx={7} fill="none"
                        stroke={P.muted} strokeWidth={4} />
                      <circle cy={19} r={3} fill={P.muted} />
                    </g>
                  )}
                  {n.icon === 'think' && (
                    <g>
                      <circle r={24} fill="none" stroke={P.muted} strokeWidth={4} />
                      <text y={9} textAnchor="middle" fontSize={24} fill={P.muted}
                        fontFamily={FONT_HEAD} fontWeight={900}>?</text>
                    </g>
                  )}
                  <MarkX x={0} y={0} s={1.5} col={P.coral} />
                </g>
                <text x={92} y={n.icon === 'think' ? 42 : 63} fill={P.cream} fontSize={21}
                  fontFamily={FONT_HEAD} fontWeight={800}>{n.t}</text>
                {n.icon === 'think' && (
                  <text x={92} y={72} fill={P.muted} fontSize={18} fontFamily={FONT_HEAD}>
                    钻牛角尖最醒脑</text>
                )}
              </g>
            ))}
          </g>
        )}
        {docK > 0 && (
          <g opacity={docK} transform={`translate(0,${(1 - docK) * 60})`}>
            <rect x={960} y={560} width={840} height={270} rx={20} fill={P.panel} stroke={P.amber} strokeWidth={4} />
            <text x={996} y={620} fill={P.amber} fontSize={30} fontFamily={FONT_HEAD} fontWeight={900}>
              什么情况要认真对待</text>
            <g transform="translate(1040,700)">
              <rect x={-10} y={-56} width={56} height={56} rx={14} fill={P.coral} />
              <line x1={18} y1={-42} x2={18} y2={-14} stroke={P.cream} strokeWidth={8} strokeLinecap="round" />
              <line x1={4} y1={-28} x2={32} y2={-28} stroke={P.cream} strokeWidth={8} strokeLinecap="round" />
            </g>
            <text x={1120} y={684} fill={P.cream} fontSize={26} fontFamily={FONT_HEAD} fontWeight={800}>
              凌晨醒持续几周 · 白天也垮了</text>
            <text x={1120} y={730} fill={P.muted} fontSize={24} fontFamily={FONT_HEAD}>
              情绪低 · 兴趣减 · 早上状态最差</text>
            <rect x={1010} y={764} width={400} height={52} rx={14} fill={P.bgDeep}
              stroke={P.amber} strokeWidth={3.5} />
            <text x={1210} y={799} textAnchor="middle" fontSize={24} fill={P.amber}
              fontFamily={FONT_HEAD} fontWeight={900}>别硬扛 · 睡眠门诊聊聊</text>
            <text x={1560} y={684} fill={P.cream} fontSize={24} fontFamily={FONT_HEAD} fontWeight={700}>
              接受今晚就这样</text>
            <text x={1560} y={730} fill={P.muted} fontSize={22} fontFamily={FONT_HEAD}>
              平静离床就是治疗</text>
          </g>
        )}
      </svg>
      <Chip text="醒了 20 分钟 → 下床" x={1290} y={72} at={beat(0) + 25} accent={P.teal} />
      <Stamp text="床 = 困意专用" x={620} y={86} at={b1 + 40} color={P.coralD} size={38} rot={-3} />
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
        <g transform={`translate(960,800) scale(${pop})`} opacity={clamp01(pop)}>
          <Sleeper x={0} y={0} s={1.05} />
        </g>
        <g transform={`translate(1520,720) scale(${pop})`} opacity={clamp01(pop)}>
          <rect x={-190} y={-52} width={380} height={104} rx={18} fill={P.panel} stroke={P.teal} strokeWidth={4} />
          <text x={0} y={-8} textAnchor="middle" fontSize={26} fill={P.teal}
            fontFamily={FONT_HEAD} fontWeight={900}>床 = 困意</text>
          <text x={0} y={30} textAnchor="middle" fontSize={22} fill={P.muted}
            fontFamily={FONT_HEAD}>联结拆干净 · 睡得更稳</text>
        </g>
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -300}}>
        <div style={{fontSize: 52, fontWeight: 800, color: C.text}}>凌晨醒不是你的错 · 是睡眠在换挡</div>
        <div style={{fontSize: 36, color: P.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E236 = ({data = tl}: {data?: any}) => {
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
