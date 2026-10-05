import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01, rand} from '../lib/style';
import {Backdrop, Stamp, SceneTag, Chip, Subtitles, ProgressBar, FadeIn, Moon, Stars} from '../lib/atoms';
import tl from '../timeline_239.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：深靛夜 × 梦境紫 × 记忆琥珀 × 肥皂青 ============ */
const P = {
  bg: '#1c1e3e',
  bgDeep: '#141630',
  bgLite: '#272b52',
  panel: '#191c38',
  card: '#353b66',
  ink: '#181b36',
  cream: '#f4edda',
  muted: '#a9b1dd',
  dream: '#c9a4ff',
  dreamD: '#9d7de0',
  memo: '#ffd166',
  memoD: '#e0a93f',
  sig: '#6fe3ff',
  coral: '#ff7d70',
  coralD: '#d85a4e',
  mint: '#5fdcb0',
  pink: '#ff9ecb',
  soap: '#a9e8f5',
};

/* ---------- 小叉 / 小勾 ---------- */
const MarkX = ({x, y, s = 1, col = P.coral}: {x: number; y: number; s?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} stroke={col} strokeWidth={7} strokeLinecap="round">
    <line x1={-13} y1={-13} x2={13} y2={13} /><line x1={13} y1={-13} x2={-13} y2={13} />
  </g>
);
const MarkV = ({x, y, s = 1, col = P.mint}: {x: number; y: number; s?: number; col?: string}) => (
  <path d="M -13 0 L -3 11 L 15 -11" transform={`translate(${x},${y}) scale(${s})`}
    stroke={col} strokeWidth={7} fill="none" strokeLinecap="round" strokeLinejoin="round" />
);

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

/* ---------- 右箭头 ---------- */
const ArrowR = ({x, y, len, col, w = 7, dash}: {x: number; y: number; len: number; col: string; w?: number; dash?: string}) => (
  <g>
    <line x1={x} y1={y} x2={x + len} y2={y} stroke={col} strokeWidth={w}
      strokeLinecap="round" strokeDasharray={dash} />
    <polygon points={`${x + len + 18},${y} ${x + len - 2},${y - 11} ${x + len - 2},${y + 11}`} fill={col} />
  </g>
);

/* ---------- 飘 z ---------- */
const Zzz = ({x, y, k = 1, col = P.sig}: {x: number; y: number; k?: number; col?: string}) => {
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

/* ---------- 文件小纸片：dream=虚线梦文件 ---------- */
const DocSheet = ({x, y, s = 1, dream = false, rot = 0, sealed = false}: {
  x: number; y: number; s?: number; dream?: boolean; rot?: number; sealed?: boolean;
}) => (
  <g transform={`translate(${x},${y}) scale(${s}) rotate(${rot})`}>
    <rect x={-34} y={-44} width={68} height={88} rx={8}
      fill={dream ? P.bgLite : P.cream}
      stroke={dream ? P.dream : P.card} strokeWidth={dream ? 4 : 3}
      strokeDasharray={dream ? '8 7' : '0'} />
    <polygon points="14,-44 34,-24 14,-24" fill={dream ? P.dream : P.card} opacity={0.7} />
    <line x1={-22} y1={-14} x2={16} y2={-14} stroke={dream ? P.dream : P.card} strokeWidth={5} strokeLinecap="round" />
    <line x1={-22} y1={4} x2={20} y2={4} stroke={dream ? P.dream : P.card} strokeWidth={5} strokeLinecap="round" />
    <line x1={-22} y1={22} x2={8} y2={22} stroke={dream ? P.dream : P.card} strokeWidth={5} strokeLinecap="round" />
    {sealed && (
      <g transform="translate(8,26) rotate(-14)">
        <circle r={20} fill="none" stroke={P.coral} strokeWidth={5} />
        <text y={7} textAnchor="middle" fontSize={18} fill={P.coral}
          fontFamily={FONT_HEAD} fontWeight={900}>留</text>
      </g>
    )}
  </g>
);

/* ---------- 存档抽屉柜 ---------- */
const DrawerBox = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-95} y={-80} width={190} height={168} rx={16} fill={P.card} />
    {/* 顶上探头的文件 */}
    <rect x={-70} y={-104} width={30} height={40} rx={6} fill={P.cream} transform="rotate(-6 -55 -84)" />
    <rect x={-28} y={-110} width={30} height={46} rx={6} fill={P.memo} />
    <rect x={14} y={-104} width={30} height={40} rx={6} fill={P.mint} transform="rotate(6 29 -84)" />
    {/* 抽屉 */}
    {[0, 1].map((i) => (
      <g key={i}>
        <rect x={-78} y={-56 + i * 76} width={156} height={58} rx={10} fill={P.bgDeep} opacity={0.7} />
        <rect x={-24} y={-34 + i * 76} width={48} height={12} rx={6} fill={P.muted} />
      </g>
    ))}
    <rect x={-95} y={88} width={190} height={14} rx={7} fill={P.card} />
  </g>
);

/* ---------- 临时缓存云（用完即焚，顶部冒消散的点） ---------- */
const CloudBin = ({x, y, s = 1, fade = 0}: {x: number; y: number; s?: number; fade?: number}) => {
  const f = useCurrentFrame();
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      <g fill={P.bgLite} stroke={P.dream} strokeWidth={4} strokeDasharray="9 7">
        <circle cx={-56} cy={6} r={44} />
        <circle cx={0} cy={-22} r={52} />
        <circle cx={58} cy={4} r={46} />
        <rect x={-92} y={-2} width={184} height={46} rx={23} stroke="none" />
      </g>
      <rect x={-92} y={-2} width={184} height={46} rx={23} fill="none"
        stroke={P.dream} strokeWidth={4} strokeDasharray="9 7" />
      {/* 蒸发的记忆碎点 */}
      {[0, 1, 2, 3, 4].map((i) => {
        const t = (f * 0.016 + i * 0.21) % 1;
        return (
          <circle key={i} cx={-60 + i * 30 + Math.sin(f * 0.1 + i) * 8}
            cy={-60 - t * 110} r={6 - t * 3} fill={P.dream} opacity={(1 - t) * (0.8 - fade * 0.3)} />
        );
      })}
      <text x={0} y={14} textAnchor="middle" fontSize={22} fill={P.dream}
        fontFamily={FONT_HEAD} fontWeight={800}>临时缓存</text>
    </g>
  );
};

/* ---------- 仰面睡觉的小人（区别于 235 的侧睡压臂） ---------- */
const Dreamer = ({x, y, s = 1, zzz = true}: {x: number; y: number; s?: number; zzz?: boolean}) => {
  const f = useCurrentFrame();
  const br = Math.sin(f * 0.14) * 3;
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      {/* 床 */}
      <rect x={-250} y={-6} width={540} height={30} rx={15} fill={P.card} />
      <rect x={-234} y={24} width={22} height={52} rx={9} fill={P.card} />
      <rect x={232} y={24} width={22} height={52} rx={9} fill={P.card} />
      {/* 枕头 */}
      <rect x={-224} y={-60} width={150} height={56} rx={24} fill={P.cream} />
      {/* 仰面的头 */}
      <g transform="translate(-148,-84)">
        <circle r={33} fill={C.skin} />
        <path d="M -33 0 Q 0 -50 33 0 L 33 -8 Q 0 -34 -33 -8 Z" fill="#4a3f66" />
        <path d="M -15 2 q 6 6 12 0 M 5 2 q 6 6 12 0" stroke={P.ink} strokeWidth={3.2}
          fill="none" strokeLinecap="round" />
        <circle cx={0} cy={-14} r={4} fill={C.skinDark} opacity={0.8} />
      </g>
      {/* 被子（随呼吸起伏） */}
      <path d={`M -100 -58 Q 20 ${-116 + br} 214 -22 L 262 -6 L 262 8 L -100 8 Z`}
        fill={P.dream} opacity={0.92} />
      <path d={`M -100 -58 Q 20 ${-116 + br} 214 -22`} fill="none" stroke={P.cream}
        strokeWidth={5} strokeLinecap="round" opacity={0.5} />
      {zzz && <Zzz x={-88} y={-140} k={0.85} />}
    </g>
  );
};

/* ---------- 梦境泡泡：小圆链 + 大云朵 + 右缘正在消散 ---------- */
const DreamBubble = ({x, y, s = 1, dissolving = true}: {x: number; y: number; s?: number; dissolving?: boolean}) => {
  const f = useCurrentFrame();
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      {/* 云体 */}
      <g fill={P.cream}>
        <circle cx={-64} cy={8} r={46} />
        <circle cx={-8} cy={-26} r={56} />
        <circle cx={54} cy={0} r={48} />
        <rect x={-100} y={-4} width={196} height={52} rx={26} />
      </g>
      {/* 云里的小梦境：月亮 + 山 */}
      <circle cx={-30} cy={-16} r={17} fill={P.memo} />
      <circle cx={-23} cy={-20} r={14} fill={P.cream} />
      <path d="M -62 30 L -20 -6 L 22 30 Z" fill={P.mint} opacity={0.85} />
      <path d="M 6 30 L 40 2 L 74 30 Z" fill={P.dreamD} opacity={0.9} />
      {/* 右缘消散成点 */}
      {dissolving && [0, 1, 2, 3, 4, 5].map((i) => {
        const t = (f * 0.02 + i * 0.17) % 1;
        return (
          <circle key={i} cx={110 + t * 130 + rand(i, 3) * 30} cy={-30 + rand(i, 5) * 70 - t * 46}
            r={7 - t * 5} fill={P.dream} opacity={(1 - t) * 0.85} />
        );
      })}
    </g>
  );
};

/* ---------- 海马体（小海马造型） ---------- */
const Seahorse = ({x, y, s = 1, col = P.memo}: {x: number; y: number; s?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    {/* 身尾弧线 */}
    <path d="M 4 -30 C 34 -26 44 -4 36 16 C 28 36 6 42 -8 34 C -18 28 -20 16 -12 10"
      fill="none" stroke={col} strokeWidth={24} strokeLinecap="round" />
    {/* 背鳍刺 */}
    <path d="M 30 -20 l 16 -8 M 36 -4 l 16 -4 M 36 12 l 14 2" stroke={col}
      strokeWidth={6} strokeLinecap="round" />
    {/* 头 */}
    <circle cx={0} cy={-36} r={19} fill={col} />
    {/* 吻 */}
    <rect x={8} y={-44} width={28} height={14} rx={7} fill={col} />
    {/* 冠 */}
    <circle cx={-10} cy={-50} r={8} fill={col} />
    <circle cx={2} cy={-40} r={3.6} fill={P.ink} />
  </g>
);

/* ---------- 头部剖面 + 海马体 ---------- */
const HeadBrain = ({x, y, s = 1, replay = 0}: {x: number; y: number; s?: number; replay?: number}) => {
  const f = useCurrentFrame();
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      {/* 轮廓 */}
      <rect x={10} y={66} width={70} height={110} rx={26} fill={P.card} />
      <circle r={105} fill={P.card} />
      {/* 颅腔 */}
      <circle cx={-6} cy={-8} r={82} fill={P.bgDeep} />
      {/* 脑回 */}
      <path d="M -62 -34 q 20 -18 40 -6 M -66 -6 q 22 -14 44 -4 M -56 24 q 20 -10 38 0"
        stroke={P.muted} strokeWidth={5} fill="none" strokeLinecap="round" opacity={0.75} />
      <path d="M 8 -58 q 24 6 30 26 M 30 -34 q 22 12 20 34" stroke={P.muted}
        strokeWidth={5} fill="none" strokeLinecap="round" opacity={0.75} />
      {/* 海马体 */}
      <Seahorse x={-6} y={34} s={0.66} />
      {/* 回放模式：旧文件绕圈飞 */}
      {replay > 0 && (
        <g opacity={replay}>
          <circle cx={-6} cy={-8} r={56} fill="none" stroke={P.dream} strokeWidth={4}
            strokeDasharray="10 9" transform={`rotate(${f * 2.4} -6 -8)`} />
          {[0, 1, 2].map((i) => {
            const a = (f * 0.08 + i * 2.09) % (Math.PI * 2);
            return (
              <rect key={i} x={-6 + Math.cos(a) * 56 - 9} y={-8 + Math.sin(a) * 56 - 12}
                width={18} height={24} rx={4} fill={P.cream} opacity={0.9} />
            );
          })}
        </g>
      )}
    </g>
  );
};

/* ---------- 去甲肾上腺素罐 ---------- */
const Tank = ({x, y, level, label}: {x: number; y: number; level: number; label: string}) => (
  <g transform={`translate(${x},${y})`}>
    <rect x={-48} y={-116} width={96} height={226} rx={22} fill={P.bgDeep}
      stroke={level > 0.3 ? P.memo : P.coral} strokeWidth={5} />
    <rect x={-36} y={-100 + 200 * (1 - level)} width={72} height={Math.max(6, 200 * level)}
      rx={10} fill={level > 0.3 ? P.memo : P.coral} opacity={0.9} />
    {/* 刻度 */}
    {[0.25, 0.5, 0.75].map((k) => (
      <line key={k} x1={-48} y1={-100 + 200 * k} x2={-30} y2={-100 + 200 * k}
        stroke={P.muted} strokeWidth={4} />
    ))}
    <text x={0} y={142} textAnchor="middle" fontSize={21} fill={P.muted}
      fontFamily={FONT_HEAD} fontWeight={700}>{label}</text>
    <text x={0} y={170} textAnchor="middle" fontSize={19}
      fill={level > 0.3 ? P.memo : P.coral} fontFamily={FONT_HEAD} fontWeight={800}>
      {level > 0.3 ? '在线' : '≈ 0'}
    </text>
  </g>
);

/* ---------- 盖章机 ---------- */
const Stamper = ({x, y, up = false}: {x: number; y: number; up?: boolean}) => {
  const f = useCurrentFrame();
  const bob = up ? -46 : Math.abs(Math.sin(f * 0.35)) * 8;
  return (
    <g transform={`translate(${x},${y})`}>
      {/* 立柱 */}
      <rect x={-88} y={-140} width={26} height={150} rx={10} fill={P.card} />
      {/* 悬臂 */}
      <g transform={`translate(-30,-110) rotate(${up ? -26 : 0})`}>
        <rect x={-64} y={-14} width={150} height={28} rx={14} fill={P.card} />
      </g>
      {/* 章头 */}
      <g transform={`translate(24,${-78 + bob}) rotate(${up ? -26 : 0})`}>
        <rect x={-13} y={-40} width={26} height={44} rx={8} fill={P.card} />
        <rect x={-34} y={0} width={68} height={34} rx={10} fill={up ? P.card : P.coral} />
        <text x={0} y={24} textAnchor="middle" fontSize={18} fill={P.cream}
          fontFamily={FONT_HEAD} fontWeight={900}>留</text>
      </g>
      {/* 台面 */}
      <rect x={-52} y={20} width={130} height={18} rx={9} fill={P.card} />
    </g>
  );
};

/* ---------- 遗忘层回收桶 ---------- */
const Bin = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <path d="M -62 -50 L 62 -50 L 48 66 L -48 66 Z" fill={P.card} />
    <rect x={-74} y={-64} width={148} height={20} rx={10} fill={P.muted} opacity={0.7} />
    <line x1={-20} y1={-32} x2={-14} y2={48} stroke={P.bgDeep} strokeWidth={8} strokeLinecap="round" />
    <line x1={14} y1={-32} x2={8} y2={48} stroke={P.bgDeep} strokeWidth={8} strokeLinecap="round" />
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
        <Moon x={265} y={160} scale={0.7} />
        {/* 睡觉的小人 + 正在消散的梦泡泡 */}
        <g transform={`translate(740,430) scale(${pA}) translate(-740,-430)`} opacity={clamp01(pA)}>
          <Dreamer x={740} y={430} s={1.05} />
        </g>
        <g transform={`translate(1230,240) scale(${pA})`} opacity={clamp01(pA)}>
          {/* 泡泡链 */}
          <circle cx={-415} cy={262} r={7} fill={P.cream} opacity={0.9} />
          <circle cx={-360} cy={222} r={10} fill={P.cream} opacity={0.9} />
          <circle cx={-300} cy={178} r={14} fill={P.cream} opacity={0.9} />
          <circle cx={-238} cy={128} r={19} fill={P.cream} opacity={0.9} />
          <circle cx={-190} cy={82} r={25} fill={P.cream} opacity={0.9} />
          <DreamBubble x={0} y={0} s={1.1} />
        </g>
        {/* 大问号 */}
        <g transform={`translate(1620,330) scale(${pB})`} opacity={clamp01(pB)}>
          <circle r={40} fill={P.coral} />
          <text y={14} textAnchor="middle" fontSize={42} fill={P.cream}
            fontFamily={FONT_HEAD} fontWeight={900}>?</text>
        </g>
        {/* 两侧小标签 */}
        <g transform={`translate(560,600) scale(${pB})`} opacity={clamp01(pB)}>
          <TagPill x={0} y={0} t="昨晚的梦还在" col={P.memo} w={260} />
        </g>
        <g transform={`translate(1360,600) scale(${pB})`} opacity={clamp01(pB)}>
          <TagPill x={0} y={0} t="刷牙回来秒没" col={P.coral} w={260} />
          <MarkX x={152} y={-6} s={0.9} />
        </g>
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 480}}>
          <div style={{color: P.sig, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 24}}>
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

/* ============ S0 梦是旁路文件：标准通道 vs 临时缓存 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const popL = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const popR = Math.max(0.001, spring({frame: f - 40, fps, config: {damping: 11, stiffness: 120}}));
  const barK = clamp01((f - b1 - 5) / 24);
  const wob = Math.sin(f * 0.4) * 4;

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={0} name={sec.name} accent={P.memo} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={8} from={0} dur={26} />
        {/* 左：白天经历的标准通道 */}
        <g transform={`translate(120,155) scale(${popL}) translate(-120,-155)`} opacity={clamp01(popL * 2)}>
          <rect x={120} y={155} width={760} height={430} rx={20} fill={P.panel} stroke={P.mint} strokeWidth={4} />
          <text x={156} y={210} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            白天的经历 · 标准记忆通道</text>
          {/* 文件 → 编码盖章 → 抽屉柜 */}
          <DocSheet x={250} y={330} s={1.05} />
          <ArrowR x={310} y={330} len={88} col={P.mint} />
          {/* 编码章机 */}
          <g transform="translate(510,330)">
            <circle r={46} fill={P.memo} />
            <text y={-2} textAnchor="middle" fontSize={24} fill={P.ink}
              fontFamily={FONT_HEAD} fontWeight={900}>编码</text>
            <text y={24} textAnchor="middle" fontSize={18} fill={P.ink}
              fontFamily={FONT_HEAD} fontWeight={800}>盖章</text>
          </g>
          <ArrowR x={580} y={330} len={80} col={P.mint} />
          <DrawerBox x={745} y={360} s={0.95} />
          <DocSheet x={505} y={452} s={0.8} sealed={true} rot={-8} />
          <text x={500} y={545} textAnchor="middle" fill={P.cream} fontSize={30}
            fontFamily={FONT_HEAD} fontWeight={800}>盖完章 · 存进长期仓库</text>
        </g>
        {/* 右：梦的旁路 */}
        <g transform={`translate(1040,155) scale(${popR}) translate(-1040,-155)`} opacity={clamp01(popR * 2)}>
          <rect x={1040} y={155} width={760} height={430} rx={20} fill={P.panel} stroke={P.dream} strokeWidth={4} />
          <text x={1076} y={210} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            梦 · 大脑没打算给你存档</text>
          <DocSheet x={1170} y={330} s={1.05} dream={true} rot={wob} />
          <ArrowR x={1236} y={330} len={92} col={P.dream} dash="12 10" />
          <CloudBin x={1560} y={330} s={1.05} />
          <TagPill x={1560} y={452} t="用完即焚" col={P.coral} w={190} fs={26} s={0.92} />
          <text x={1420} y={545} textAnchor="middle" fill={P.cream} fontSize={30}
            fontFamily={FONT_HEAD} fontWeight={800}>没编码没盖章 · 醒了自动清空</text>
        </g>
        {/* 中央 VS */}
        <g transform="translate(960,370)">
          <circle r={40} fill={P.dream} />
          <text y={12} textAnchor="middle" fontSize={30} fill={P.ink}
            fontFamily={FONT_HEAD} fontWeight={900}>VS</text>
        </g>
        {/* 底部结论（cue1） */}
        {barK > 0 && (
          <g opacity={barK} transform={`translate(0,${(1 - barK) * 60})`}>
            <rect x={120} y={648} width={1680} height={210} rx={20} fill={P.panel} stroke={P.memo} strokeWidth={4} />
            <text x={160} y={710} fill={P.memo} fontSize={30} fontFamily={FONT_HEAD} fontWeight={900}>
              秒忘不是缺陷 · 是默认设置</text>
            <text x={160} y={770} fill={P.cream} fontSize={29} fontFamily={FONT_HEAD} fontWeight={700}>
              每晚做四五个梦、累计一两小时 —— 大脑只在自我维护时顺便让你瞥一眼</text>
            {/* 右侧小统计 */}
            <g transform="translate(1500,700)">
              <rect x={-190} y={-36} width={240} height={60} rx={14} fill={P.bgDeep} stroke={P.dream} strokeWidth={3.5} />
              <text x={-70} y={4} textAnchor="middle" fontSize={23} fill={P.dream}
                fontFamily={FONT_HEAD} fontWeight={800}>4~5 场 / 晚</text>
              <rect x={70} y={-36} width={240} height={60} rx={14} fill={P.bgDeep} stroke={P.coral} strokeWidth={3.5} />
              <text x={190} y={4} textAnchor="middle" fontSize={23} fill={P.coral}
                fontFamily={FONT_HEAD} fontWeight={800}>只记得 1~2 个</text>
              <text x={60} y={86} textAnchor="middle" fontSize={21} fill={P.muted}
                fontFamily={FONT_HEAD}>不是你丢了 · 是根本没发存档权限</text>
            </g>
          </g>
        )}
      </svg>
      <Chip text="梦是旁路文件" x={1300} y={72} at={beat(0) + 25} accent={P.dream} />
      <Stamp text="留不住是设计好的" x={620} y={86} at={b1 + 40} color={P.dreamD} size={38} rot={-3} />
    </FadeIn>
  );
};

/* ============ S1 海马体不登记：编码模式 vs 回放模式 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const headPop = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const card1Pop = Math.max(0.001, spring({frame: f - 30, fps, config: {damping: 11, stiffness: 120}}));
  const card2Pop = Math.max(0.001, spring({frame: f - b1 - 5, fps, config: {damping: 11, stiffness: 120}}));
  const replayK = clamp01((f - b1) / 24);

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={1} name={sec.name} accent={P.dream} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：头部剖面 + 海马体 */}
        <g transform={`translate(120,150) scale(${headPop}) translate(-120,-150)`} opacity={clamp01(headPop * 2)}>
          <rect x={120} y={150} width={620} height={640} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={156} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            记忆门卫 · 海马体</text>
          <HeadBrain x={430} y={440} s={1.55} replay={replayK} />
          {/* 引线 + 标注 */}
          <line x1={350} y1={480} x2={240} y2={560} stroke={P.memo} strokeWidth={3.5} strokeDasharray="7 6" />
          <TagPill x={250} y={610} t="海马体" col={P.memo} w={170} fs={28} />
          {replayK > 0 && (
            <g opacity={replayK}>
              <TagPill x={560} y={640} t="回放旧记忆" col={P.dream} w={220} fs={26} />
            </g>
          )}
          <text x={430} y={756} textAnchor="middle" fill={P.muted} fontSize={23}
            fontFamily={FONT_HEAD}>做梦时它忙着整理白天的旧文件</text>
        </g>
        {/* 右上：清醒·编码模式 */}
        <g transform={`translate(780,150) scale(${card1Pop}) translate(-780,-150)`} opacity={clamp01(card1Pop * 2)}>
          <rect x={780} y={150} width={1020} height={270} rx={20} fill={P.panel}
            stroke={replayK > 0.5 ? P.card : P.mint} strokeWidth={4} />
          <TagPill x={980} y={206} t="清醒 · 编码模式" col={P.mint} w={280} fs={26} s={0.95} />
          {/* 新信息 → 编码 → 长期仓库 */}
          <text x={830} y={268} fill={P.muted} fontSize={22} fontFamily={FONT_HEAD}>新信息进来</text>
          <DocSheet x={880} y={330} s={0.85} />
          <ArrowR x={940} y={330} len={90} col={P.mint} />
          <g transform="translate(1140,330)">
            <circle r={42} fill={P.memo} />
            <text y={-1} textAnchor="middle" fontSize={23} fill={P.ink}
              fontFamily={FONT_HEAD} fontWeight={900}>编码</text>
            <text y={23} textAnchor="middle" fontSize={17} fill={P.ink}
              fontFamily={FONT_HEAD} fontWeight={800}>打卡</text>
          </g>
          <ArrowR x={1210} y={330} len={90} col={P.mint} />
          <g transform="translate(1480,330)">
            <rect x={-160} y={-60} width={320} height={120} rx={16} fill={P.bgDeep} stroke={P.mint} strokeWidth={4} />
            <text x={0} y={-12} textAnchor="middle" fontSize={25} fill={P.mint}
              fontFamily={FONT_HEAD} fontWeight={900}>大脑皮层</text>
            <text x={0} y={28} textAnchor="middle" fontSize={22} fill={P.cream}
              fontFamily={FONT_HEAD}>长期仓库 · 慢慢移交</text>
          </g>
          <text x={1710} y={390} textAnchor="end" fill={P.mint} fontSize={22}
            fontFamily={FONT_HEAD} fontWeight={700}>登记在册 ✓</text>
        </g>
        {/* 右下：REM·回放模式 */}
        <g transform={`translate(780,470) scale(${card2Pop}) translate(-780,-470)`} opacity={clamp01(card2Pop * 2)}>
          <rect x={780} y={470} width={1020} height={320} rx={20} fill={P.panel}
            stroke={replayK > 0.5 ? P.dream : P.card} strokeWidth={4} />
          <TagPill x={985} y={526} t="做梦 REM · 回放模式" col={P.dream} w={330} fs={26} s={0.95} />
          {/* 旧文件循环回皮层 */}
          <g transform="translate(950,640)">
            <circle r={64} fill="none" stroke={P.dream} strokeWidth={5} strokeDasharray="11 9"
              transform={`rotate(${f * 2})`} />
            <DocSheet x={-10} y={-14} s={0.62} />
            <text x={0} y={102} textAnchor="middle" fontSize={21} fill={P.dream}
              fontFamily={FONT_HEAD} fontWeight={800}>重放白天旧事</text>
          </g>
          <ArrowR x={1040} y={640} len={90} col={P.dream} dash="10 9" />
          <g transform="translate(1280,640)">
            <rect x={-140} y={-56} width={280} height={112} rx={16} fill={P.bgDeep} stroke={P.dream} strokeWidth={4} />
            <text x={0} y={-8} textAnchor="middle" fontSize={24} fill={P.dream}
              fontFamily={FONT_HEAD} fontWeight={900}>巩固旧记忆</text>
            <text x={0} y={30} textAnchor="middle" fontSize={20} fill={P.muted}
              fontFamily={FONT_HEAD}>自我维护 · 不对外办公</text>
          </g>
          {/* 正在做的梦被弹回 */}
          <g transform="translate(1520,600)">
            <DocSheet x={0} y={0} s={0.8} dream={true} rot={Math.sin(f * 0.4) * 6} />
            <text x={0} y={-62} textAnchor="middle" fontSize={19} fill={P.dream}
              fontFamily={FONT_HEAD} fontWeight={700}>正在做的梦</text>
          </g>
          <g transform="translate(1650,640)">
            <rect x={-30} y={-86} width={60} height={172} rx={14} fill={P.coral} opacity={0.85} />
            <text x={0} y={-10} textAnchor="middle" fontSize={21} fill={P.cream}
              fontFamily={FONT_HEAD} fontWeight={900} transform="rotate(-90 0 -8)">不登记</text>
          </g>
          <MarkX x={1600} y={540} s={1.1} />
          <text x={1690} y={752} textAnchor="end" fill={P.coral} fontSize={22}
            fontFamily={FONT_HEAD} fontWeight={800}>体验了 · 但没打卡</text>
        </g>
      </svg>
      <Chip text="编码模式 vs 回放模式" x={1230} y={72} at={beat(0) + 25} accent={P.mint} />
      <Stamp text="梦没进编码通道" x={600} y={86} at={b1 + 40} color={P.dreamD} size={38} rot={3} />
    </FadeIn>
  );
};

/* ============ S2 没盖章的文件：去甲肾上腺素见底 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const popL = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const popR = Math.max(0.001, spring({frame: f - 40, fps, config: {damping: 11, stiffness: 120}}));
  const barK = clamp01((f - b1 - 5) / 24);
  const drop = interpolate(f, [b1 - 30, b1 + 20], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});

  const docs = (baseX: number, sealed: boolean) =>
    [0, 1, 2].map((i) => {
      const t = (f * 0.014 + i * 0.36) % 1;
      const x = baseX + t * 430;
      return <DocSheet key={i} x={x} y={500} s={0.8} sealed={sealed && t > 0.5} />;
    });

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={2} name={sec.name} accent={P.memo} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：清醒·盖章流水线 */}
        <g transform={`translate(120,150) scale(${popL}) translate(-120,-150)`} opacity={clamp01(popL * 2)}>
          <rect x={120} y={150} width={800} height={440} rx={20} fill={P.panel} stroke={P.memo} strokeWidth={4} />
          <text x={156} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            清醒 · 盖章机全速运转</text>
          <Tank x={230} y={380} level={0.85} label="去甲肾上腺素" />
          {/* 传送带 */}
          <line x1={340} y1={540} x2={830} y2={540} stroke={P.card} strokeWidth={10} strokeLinecap="round" />
          {[0, 1, 2, 3, 4, 5].map((i) => (
            <rect key={i} x={350 + i * 82} y={548} width={40} height={10} rx={5} fill={P.muted} opacity={0.5} />
          ))}
          {docs(360, true)}
          <Stamper x={620} y={430} />
          {/* 存档出口 */}
          <g transform="translate(830,452)">
            <DrawerBox x={0} y={0} s={0.78} />
            <text x={0} y={120} textAnchor="middle" fontSize={22} fill={P.mint}
              fontFamily={FONT_HEAD} fontWeight={800}>稳固存档</text>
          </g>
          <TagPill x={620} y={330} t="盖「值得留」" col={P.coral} w={220} fs={24} s={0.85} />
        </g>
        {/* 右：做梦·没墨了 */}
        <g transform={`translate(1000,150) scale(${popR}) translate(-1000,-150)`} opacity={clamp01(popR * 2)}>
          <rect x={1000} y={150} width={800} height={440} rx={20} fill={P.panel}
            stroke={popR > 0.5 ? P.coral : P.card} strokeWidth={4} />
          <text x={1036} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            做梦 REM · 化学环境低配</text>
          <Tank x={1110} y={380} level={0.05 + drop * 0} label="去甲肾上腺素" />
          <MarkX x={1190} y={280} s={0.9} />
          {/* 传送带 */}
          <line x1={1220} y1={540} x2={1710} y2={540} stroke={P.card} strokeWidth={10} strokeLinecap="round" />
          {[0, 1, 2, 3, 4, 5].map((i) => (
            <rect key={i} x={1230 + i * 82} y={548} width={40} height={10} rx={5} fill={P.muted} opacity={0.5} />
          ))}
          {docs(1240, false)}
          <Stamper x={1500} y={430} up={true} />
          <TagPill x={1500} y={330} t="章头抬着 · 没墨" col={P.coral} w={240} fs={24} s={0.85} />
          {/* 掉进遗忘层 */}
          <Bin x={1730} y={470} s={0.9} />
          <g transform={`translate(1690,${420 + drop * 40}) rotate(${drop * 30})`} opacity={0.9 - drop * 0.4}>
            <DocSheet x={0} y={0} s={0.7} />
          </g>
          <text x={1730} y={566} textAnchor="middle" fontSize={22} fill={P.coral}
            fontFamily={FONT_HEAD} fontWeight={800}>遗忘层</text>
        </g>
        {/* 中央 VS */}
        <g transform="translate(960,370)">
          <circle r={40} fill={P.memo} />
          <text y={12} textAnchor="middle" fontSize={30} fill={P.ink}
            fontFamily={FONT_HEAD} fontWeight={900}>VS</text>
        </g>
        {/* 底部结论（cue1） */}
        {barK > 0 && (
          <g opacity={barK} transform={`translate(0,${(1 - barK) * 60})`}>
            <rect x={120} y={648} width={1680} height={210} rx={20} fill={P.panel} stroke={P.coral} strokeWidth={4} />
            <text x={160} y={710} fill={P.coral} fontSize={30} fontFamily={FONT_HEAD} fontWeight={900}>
              没盖章 = 低优先级文件</text>
            <text x={160} y={770} fill={P.cream} fontSize={29} fontFamily={FONT_HEAD} fontWeight={700}>
              存得潦草、取得飞快 —— 不是你忘了，是它本来就没被好好存过</text>
            {/* 对照小图标 */}
            <g transform="translate(1500,720)">
              <DocSheet x={-140} y={-10} s={0.72} sealed={true} />
              <text x={-140} y={66} textAnchor="middle" fontSize={20} fill={P.mint}
                fontFamily={FONT_HEAD} fontWeight={800}>盖过章</text>
              <DocSheet x={90} y={-10} s={0.72} dream={true} />
              <text x={90} y={66} textAnchor="middle" fontSize={20} fill={P.coral}
                fontFamily={FONT_HEAD} fontWeight={800}>没盖章 · 秒没</text>
            </g>
          </g>
        )}
      </svg>
      <Chip text="去甲肾上腺素 ≈ 0" x={1300} y={72} at={beat(0) + 25} accent={P.memo} />
      <Stamp text="没打算让你找到" x={620} y={86} at={b1 + 40} color={P.coralD} size={38} rot={-3} />
    </FadeIn>
  );
};

/* ============ S3 醒来那一刻就冲散：换台 + 肥皂 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const popL = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const popR = Math.max(0.001, spring({frame: f - 35, fps, config: {damping: 11, stiffness: 120}}));
  const switchK = clamp01((f - 110) / 22);
  const slipK = clamp01((f - b1 - 8) / 26);
  const tlK = clamp01((f - b1 - 34) / 26);
  const nodePop = [0, 1, 2].map((i) => Math.max(0.001, spring({frame: f - b1 - 40 - i * 12, fps, config: {damping: 12, stiffness: 140}})));

  const knobX = interpolate(switchK, [0, 1], [308, 672]);
  const soapRot = interpolate(slipK, [0, 1], [0, 70]);
  const soapX = interpolate(slipK, [0, 1], [0, 130]);
  const soapY = interpolate(slipK, [0, 1], [0, 96]);

  /* 梦境碎片：换台时被吹散 */
  const shards = [
    {k: 'moon', x: 250, y: 250, dx: -120, dy: -60},
    {k: 'star', x: 420, y: 220, dx: -60, dy: -110},
    {k: 'blob', x: 580, y: 260, dx: 40, dy: -130},
    {k: 'zig', x: 330, y: 330, dx: -140, dy: 30},
    {k: 'star', x: 540, y: 360, dx: 80, dy: -40},
    {k: 'blob', x: 700, y: 300, dx: 160, dy: -80},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bgDeep} bottom={P.bgDeep} />
      <SceneTag index={3} name={sec.name} accent={P.sig} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={8} from={0} dur={26} />
        {/* 左：大脑换台开关 */}
        <g transform={`translate(120,150) scale(${popL}) translate(-120,-150)`} opacity={clamp01(popL * 2)}>
          <rect x={120} y={150} width={800} height={440} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={156} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            醒来瞬间 · 化学环境换台</text>
          {/* 滑动开关 */}
          <rect x={250} y={390} width={480} height={86} rx={43} fill={P.bgDeep}
            stroke={P.card} strokeWidth={4} />
          <text x={330} y={358} textAnchor="middle" fontSize={24} fill={P.dream}
            fontFamily={FONT_HEAD} fontWeight={900} opacity={1 - switchK * 0.5}>REM 梦境态</text>
          <text x={650} y={358} textAnchor="middle" fontSize={24} fill={P.sig}
            fontFamily={FONT_HEAD} fontWeight={900} opacity={0.4 + switchK * 0.6}>清醒态</text>
          {/* 轨道上的碎梦被吹散 */}
          {shards.map((s, i) => {
            const px = s.x + s.dx * switchK;
            const py = s.y + s.dy * switchK;
            const op = 0.95 - switchK * 0.85;
            return (
              <g key={i} transform={`translate(${px},${py}) rotate(${switchK * (i % 2 ? 90 : -90)})`} opacity={op}>
                {s.k === 'moon' && (
                  <g><circle r={17} fill={P.memo} /><circle cx={6} cy={-4} r={13} fill={P.panel} /></g>
                )}
                {s.k === 'star' && (
                  <polygon points="0,-18 5,-5 18,-5 8,4 12,17 0,9 -12,17 -8,4 -18,-5 -5,-5" fill={P.dream} />
                )}
                {s.k === 'blob' && <circle r={14} fill={P.dream} />}
                {s.k === 'zig' && (
                  <polyline points="-16,-12 0,0 -16,12 4,24" fill="none" stroke={P.sig}
                    strokeWidth={6} strokeLinecap="round" />
                )}
              </g>
            );
          })}
          {/* 滑块 */}
          <g transform={`translate(${knobX},433)`}>
            <circle r={58} fill={switchK > 0.5 ? P.sig : P.dream} />
            {switchK < 0.5 ? (
              <text y={14} textAnchor="middle" fontSize={40} fill={P.ink}
                fontFamily={FONT_HEAD} fontWeight={900}>z</text>
            ) : (
              <g>
                <circle r={16} cy={-4} fill={P.ink} />
                <circle cy={-4} r={7} fill={P.sig} />
                <path d="M -22 14 q 22 14 44 0" stroke={P.ink} strokeWidth={5}
                  fill="none" strokeLinecap="round" />
              </g>
            )}
          </g>
          <text x={490} y={556} textAnchor="middle" fill={P.cream} fontSize={27}
            fontFamily={FONT_HEAD} fontWeight={800}>
            没来得及转存稳固仓库的梦 · 被切换直接冲散</text>
        </g>
        {/* 右：手里的肥皂 */}
        <g transform={`translate(1000,150) scale(${popR}) translate(-1000,-150)`} opacity={clamp01(popR * 2)}>
          <rect x={1000} y={150} width={800} height={440} rx={20} fill={P.panel} stroke={P.soap} strokeWidth={4} />
          <text x={1036} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            像洗澡时手里那块肥皂</text>
          {/* 手：拳头握住 */}
          <g transform="translate(1280,400)">
            <rect x={-120} y={-56} width={150} height={112} rx={52} fill={C.skin} />
            {[0, 1, 2, 3].map((i) => (
              <path key={i} d={`M ${-56 + i * 34} -50 q 14 50 0 100`} stroke={C.skinDark}
                strokeWidth={5} fill="none" opacity={0.6} />
            ))}
            <ellipse cx={-96} cy={0} rx={26} ry={40} fill={C.skinDark} opacity={0.45} />
            <text x={-46} y={106} textAnchor="middle" fontSize={21} fill={P.muted}
              fontFamily={FONT_HEAD}>攥得越紧</text>
          </g>
          {/* 肥皂滑出去 */}
          <g transform={`translate(${1420 + soapX},${392 + soapY}) rotate(${soapRot})`}>
            <rect x={-85} y={-38} width={170} height={76} rx={34} fill={P.soap} />
            <rect x={-85} y={-38} width={170} height={30} rx={15} fill="#d8f4fb" />
            <text x={0} y={13} textAnchor="middle" fontSize={34} fill={P.ink}
              fontFamily={FONT_HEAD} fontWeight={900}>梦</text>
          </g>
          {/* 滑出的速度线 + 泡泡 */}
          {slipK > 0.15 && (
            <g stroke={P.soap} strokeWidth={5} strokeLinecap="round" opacity={slipK * 0.8}>
              <line x1={1380} y1={360} x2={1300} y2={330} />
              <line x1={1390} y1={420} x2={1290} y2={410} />
              <line x1={1400} y1={470} x2={1330} y2={490} />
            </g>
          )}
          {[0, 1, 2, 3].map((i) => {
            const t = (f * 0.02 + i * 0.25) % 1;
            return <circle key={i} cx={1560 + i * 46} cy={330 - t * 60} r={8 - t * 4}
              fill="none" stroke={P.soap} strokeWidth={3} opacity={(1 - t) * 0.7} />;
          })}
          <text x={1190} y={556} textAnchor="middle" fill={P.soap} fontSize={27}
            fontFamily={FONT_HEAD} fontWeight={800}>「醒来」就是那滑出去的一下</text>
        </g>
        {/* 底部遗忘时间线（cue1） */}
        {tlK > 0 && (
          <g opacity={tlK} transform={`translate(0,${(1 - tlK) * 60})`}>
            <rect x={120} y={648} width={1680} height={210} rx={20} fill={P.panel} stroke={P.sig} strokeWidth={4} />
            {[
              {x: 420, t: '刚醒几秒', cap: '残留环境还在 · 画面清晰', op: 1, col: P.dream},
              {x: 960, t: '几分钟后', cap: '换成清醒态 · 只剩碎片残影', op: 0.55, col: P.muted},
              {x: 1500, t: '半小时后', cap: '查无此梦 · 连接已断', op: 0.22, col: P.coral},
            ].map((n, i) => (
              <g key={i} transform={`translate(${n.x},720) scale(${nodePop[i]}) translate(${-n.x},-720)`}
                opacity={clamp01(nodePop[i] * 2)}>
                {/* 迷你梦云 */}
                <g transform={`translate(${n.x},726)`} opacity={n.op}>
                  <circle cx={-34} cy={4} r={24} fill={P.cream} />
                  <circle cx={0} cy={-12} r={30} fill={P.cream} />
                  <circle cx={32} cy={2} r={25} fill={P.cream} />
                  <rect x={-56} y={-4} width={112} height={30} rx={15} fill={P.cream} />
                  <circle cx={-10} cy={-8} r={9} fill={P.memo} />
                  <circle cx={-6} cy={-11} r={7} fill={P.cream} />
                </g>
                {i === 2 && (
                  <g transform={`translate(${n.x},726)`}>
                    <circle cx={-34} cy={4} r={24} fill="none" stroke={P.muted} strokeWidth={3.5} strokeDasharray="7 6" />
                    <circle cx={0} cy={-12} r={30} fill="none" stroke={P.muted} strokeWidth={3.5} strokeDasharray="7 6" />
                    <circle cx={32} cy={2} r={25} fill="none" stroke={P.muted} strokeWidth={3.5} strokeDasharray="7 6" />
                  </g>
                )}
                <text x={n.x} y={800} textAnchor="middle" fontSize={27} fill={n.col}
                  fontFamily={FONT_HEAD} fontWeight={900}>{n.t}</text>
                <text x={n.x} y={838} textAnchor="middle" fontSize={21} fill={P.muted}
                  fontFamily={FONT_HEAD}>{n.cap}</text>
                {i < 2 && (
                  <ArrowR x={n.x + 130} y={726} len={240} col={P.card} />
                )}
              </g>
            ))}
          </g>
        )}
      </svg>
      <Chip text="换台 = 连接掐断" x={1290} y={72} at={beat(0) + 30} accent={P.sig} />
      <Stamp text="不是慢慢忘 · 是秒断" x={560} y={86} at={b1 + 45} color={P.coralD} size={37} rot={-3} />
    </FadeIn>
  );
};

/* ============ S4 想留住梦有办法：天生难忘 + 练习三招 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const bandPop = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const cardPop = [
    Math.max(0.001, spring({frame: f - 40, fps, config: {damping: 12, stiffness: 130}})),
    Math.max(0.001, spring({frame: f - b1 - 10, fps, config: {damping: 12, stiffness: 130}})),
    Math.max(0.001, spring({frame: f - b1 - 26, fps, config: {damping: 12, stiffness: 130}})),
  ];
  const footK = clamp01((f - b1 - 60) / 24);

  const cards = [
    {t: '醒来躺一分钟', sub: '别睁眼 · 脑子里默念一遍', cap: '或立刻写下来', col: P.mint, icon: 'bed'},
    {t: '睡前立个 flag', sub: '告诉自己「我要记住梦」', cap: '意向设置 · 多留个神', col: P.dream, icon: 'flag'},
    {t: '先别碰手机', sub: '清醒任务会瞬间覆盖梦', cap: '安静回想 · 梦会回笼', col: P.coral, icon: 'phone'},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={4} name={sec.name} accent={P.mint} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 顶条：哪种梦天生难忘 */}
        <g transform={`translate(120,140) scale(${bandPop}) translate(-120,-140)`} opacity={clamp01(bandPop * 2)}>
          <rect x={120} y={140} width={1680} height={120} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={160} y={214} fill={P.memo} fontSize={28} fontFamily={FONT_HEAD} fontWeight={900}>
            哪种梦天生难忘</text>
          {/* 情绪爆表 */}
          <g transform="translate(620,200)">
            <rect x={-140} y={-36} width={330} height={72} rx={36} fill={P.bgDeep} stroke={P.coral} strokeWidth={4} />
            <path d="M -104 6 C -116 -10 -134 2 -104 24 C -74 2 -92 -10 -104 6 Z" fill={P.coral} />
            <text x={-66} y={10} fontSize={26} fill={P.cream} fontFamily={FONT_HEAD} fontWeight={800}>
              情绪爆表 · 噩喜乐怪</text>
          </g>
          {/* 情节完整 */}
          <g transform="translate(1150,200)">
            <rect x={-160} y={-36} width={330} height={72} rx={36} fill={P.bgDeep} stroke={P.sig} strokeWidth={4} />
            <rect x={-128} y={-18} width={36} height={30} rx={6} fill="none" stroke={P.sig} strokeWidth={4} />
            <line x1={-128} y1={-6} x2={-92} y2={-6} stroke={P.sig} strokeWidth={4} />
            <text x={-70} y={10} fontSize={26} fill={P.cream} fontFamily={FONT_HEAD} fontWeight={800}>
              有头有尾 · 情节完整</text>
          </g>
          <text x={1760} y={214} textAnchor="end" fill={P.muted} fontSize={22}
            fontFamily={FONT_HEAD}>被强行记住 →</text>
        </g>
        {/* 练习三招 */}
        {cards.map((c, i) => (
          <g key={i} transform={`translate(${120 + i * 566},300) scale(${cardPop[i]}) translate(${-(120 + i * 566)},-300)`}
            opacity={clamp01(cardPop[i] * 2)}>
            <rect x={120 + i * 566} y={300} width={536} height={440} rx={20}
              fill={P.panel} stroke={c.col} strokeWidth={4} />
            <circle cx={176 + i * 566} cy={356} r={26} fill={c.col} />
            <text x={176 + i * 566} y={366} textAnchor="middle" fontSize={28} fill={P.ink}
              fontFamily={FONT_HEAD} fontWeight={900}>{i + 1}</text>
            <text x={216 + i * 566} y={366} fill={c.col} fontSize={32}
              fontFamily={FONT_HEAD} fontWeight={900}>{c.t}</text>
            {/* 图标区 */}
            <g transform={`translate(${388 + i * 566},500)`}>
              {c.icon === 'bed' && (
                <g>
                  <rect x={-150} y={10} width={300} height={22} rx={11} fill={P.card} />
                  <rect x={-140} y={32} width={16} height={30} rx={7} fill={P.card} />
                  <rect x={124} y={32} width={16} height={30} rx={7} fill={P.card} />
                  <rect x={-138} y={-28} width={86} height={36} rx={16} fill={P.cream} />
                  <circle cx={-76} cy={-44} r={24} fill={C.skin} />
                  <path d="M -52 -8 Q 20 -46 120 -6 L 140 10 L -52 10 Z" fill={P.mint} opacity={0.9} />
                  {/* 默念泡泡 */}
                  <circle cx={30} cy={-70} r={8} fill={P.cream} />
                  <circle cx={52} cy={-94} r={12} fill={P.cream} />
                  <g transform="translate(96,-124)">
                    <circle r={34} fill={P.cream} />
                    <text y={10} textAnchor="middle" fontSize={26} fill={P.dreamD}
                      fontFamily={FONT_HEAD} fontWeight={900}>梦</text>
                  </g>
                </g>
              )}
              {c.icon === 'flag' && (
                <g>
                  <Moon x={-88} y={-20} scale={0.62} />
                  {/* 便签 */}
                  <g transform="translate(50,-10) rotate(-4)">
                    <rect x={-76} y={-64} width={170} height={120} rx={12} fill={P.memo} />
                    <circle cx={9} cy={-56} r={9} fill={P.coralD} />
                    <text x={9} y={-18} textAnchor="middle" fontSize={26} fill={P.ink}
                      fontFamily={FONT_HEAD} fontWeight={900}>我要记住</text>
                    <text x={9} y={20} textAnchor="middle" fontSize={30} fill={P.ink}
                      fontFamily={FONT_HEAD} fontWeight={900}>今晚的梦</text>
                  </g>
                </g>
              )}
              {c.icon === 'phone' && (
                <g>
                  <rect x={-58} y={-92} width={116} height={184} rx={22} fill={P.bgDeep}
                    stroke={P.muted} strokeWidth={6} />
                  <rect x={-42} y={-72} width={84} height={126} rx={8} fill={P.card} />
                  <circle cx={0} cy={72} r={8} fill={P.muted} />
                  <line x1={-30} y1={-46} x2={30} y2={-46} stroke={P.sig} strokeWidth={6} strokeLinecap="round" opacity={0.7} />
                  <line x1={-30} y1={-22} x2={14} y2={-22} stroke={P.sig} strokeWidth={6} strokeLinecap="round" opacity={0.7} />
                  <MarkX x={0} y={-6} s={2.1} />
                </g>
              )}
            </g>
            <text x={160 + i * 566} y={636} fill={P.cream} fontSize={24}
              fontFamily={FONT_HEAD} fontWeight={700}>{c.sub}</text>
            <text x={160 + i * 566} y={682} fill={P.muted} fontSize={22}
              fontFamily={FONT_HEAD}>{c.cap}</text>
          </g>
        ))}
        {/* 底注：实验室证据 */}
        {footK > 0 && (
          <g opacity={footK} transform={`translate(0,${(1 - footK) * 40})`}>
            <rect x={430} y={786} width={1060} height={70} rx={35} fill={P.bgDeep} stroke={P.memo} strokeWidth={4} />
            <text x={960} y={832} textAnchor="middle" fontSize={26} fill={P.cream}
              fontFamily={FONT_HEAD} fontWeight={700}>
              睡眠实验室实测：<tspan fill={P.memo} fontWeight={900}>REM 期把人叫醒 · 八成以上能报出梦</tspan>
            </text>
          </g>
        )}
      </svg>
      <Chip text="趁痕迹没散 · 补个章" x={1220} y={72} at={beat(0) + 25} accent={P.mint} />
      <Stamp text="记梦是能练的本事" x={580} y={86} at={b1 + 50} color={P.dreamD} size={38} rot={-2} />
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
        <g transform={`translate(880,830) scale(${pop})`} opacity={clamp01(pop)}>
          <Dreamer x={0} y={0} s={1.0} />
        </g>
        {/* 梦泡泡轻轻飘走 */}
        <g transform={`translate(1330,700) scale(${pop})`} opacity={clamp01(pop)}>
          <circle cx={-120} cy={60} r={10} fill={P.cream} opacity={0.8} />
          <circle cx={-88} cy={32} r={15} fill={P.cream} opacity={0.8} />
          <DreamBubble x={0} y={0} s={0.8} />
        </g>
        {/* 小牌子 */}
        <g transform={`translate(1470,880) scale(${pop})`} opacity={clamp01(pop)}>
          <rect x={-190} y={-48} width={380} height={96} rx={18} fill={P.panel} stroke={P.dream} strokeWidth={4} />
          <text x={0} y={-6} textAnchor="middle" fontSize={24} fill={P.dream}
            fontFamily={FONT_HEAD} fontWeight={800}>临时文件 · 用完即焚</text>
          <text x={0} y={30} textAnchor="middle" fontSize={21} fill={P.muted}
            fontFamily={FONT_HEAD}>想留就趁醒来的第一分钟</text>
        </g>
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -280}}>
        <div style={{fontSize: 54, fontWeight: 800, color: C.text}}>忘梦不是缺陷 · 是大脑的默认设置</div>
        <div style={{fontSize: 36, color: P.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E239 = ({data = tl}: {data?: any}) => {
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
