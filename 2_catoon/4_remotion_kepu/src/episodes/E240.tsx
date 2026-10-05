import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01, rand} from '../lib/style';
import {Backdrop, Stamp, SceneTag, Chip, Subtitles, ProgressBar, FadeIn, Moon, Stars} from '../lib/atoms';
import tl from '../timeline_240.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：深靛档案室 × 旧照片琥珀 × 神经青 × 情绪珊瑚 ============ */
const P = {
  bg: '#1d2040',
  bgDeep: '#151834',
  bgLite: '#262a4e',
  panel: '#212547',
  card: '#3a3f6e',
  ink: '#171a33',
  cream: '#f3ecda',
  muted: '#a7aede',
  amber: '#ffc766',
  amberD: '#e09f42',
  teal: '#64e0c3',
  tealD: '#40b398',
  coral: '#ff7f72',
  coralD: '#d95d50',
  lilac: '#b3a2f2',
  lilacD: '#8d7bd8',
  blue: '#74ccff',
  sepia: '#b98d5e',
};

/* ---------- 小叉 / 小勾 ---------- */
const MarkX = ({x, y, s = 1, col = P.coral}: {x: number; y: number; s?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} stroke={col} strokeWidth={7} strokeLinecap="round">
    <line x1={-13} y1={-13} x2={13} y2={13} /><line x1={13} y1={-13} x2={-13} y2={13} />
  </g>
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
const Zzz = ({x, y, k = 1, col = P.lilac}: {x: number; y: number; k?: number; col?: string}) => {
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

/* ---------- 旧照片人脸卡：库存里的「人」 ---------- */
const FaceCard = ({x, y, s = 1, rot = 0, lit = false, dim = false, label, tag}: {
  x: number; y: number; s?: number; rot?: number; lit?: boolean; dim?: boolean; label?: string; tag?: string;
}) => (
  <g transform={`translate(${x},${y}) scale(${s}) rotate(${rot})`}>
    <rect x={-44} y={-58} width={88} height={116} rx={10}
      fill={dim ? P.bgLite : P.cream}
      stroke={lit ? P.amber : P.card} strokeWidth={lit ? 5 : 3}
      opacity={dim ? 0.8 : 1} />
    {/* 人像剪影 */}
    <circle cx={0} cy={-16} r={19} fill={dim ? P.muted : P.sepia} opacity={dim ? 0.45 : 0.95} />
    <path d="M -25 30 Q 0 2 25 30 L 25 34 L -25 34 Z" fill={dim ? P.muted : P.sepia} opacity={dim ? 0.45 : 0.95} />
    <line x1={-26} y1={44} x2={26} y2={44} stroke={dim ? P.muted : P.card} strokeWidth={5}
      strokeLinecap="round" opacity={dim ? 0.35 : 0.8} />
    {lit && (
      <rect x={-52} y={-66} width={104} height={132} rx={14} fill="none"
        stroke={P.amber} strokeWidth={3} strokeDasharray="8 7" opacity={0.85} />
    )}
    {label && (
      <g transform="translate(0,78)">
        <rect x={-62} y={-19} width={124} height={38} rx={10} fill={P.bgDeep} stroke={P.amber} strokeWidth={3} />
        <text x={0} y={8} textAnchor="middle" fontSize={21} fill={P.amber}
          fontFamily={FONT_HEAD} fontWeight={800}>{label}</text>
      </g>
    )}
    {tag && (
      <g transform="translate(30,-44) rotate(12)">
        <rect x={-22} y={-16} width={44} height={32} rx={8} fill={P.coral} />
        <text x={0} y={7} textAnchor="middle" fontSize={19} fill={P.cream}
          fontFamily={FONT_HEAD} fontWeight={900}>{tag}</text>
      </g>
    )}
  </g>
);

/* ---------- 蜷缩侧睡的小人（胎儿式，区别于 239 仰面 / 235 侧压臂） ---------- */
const Sleeper = ({x, y, s = 1, zzz = true}: {x: number; y: number; s?: number; zzz?: boolean}) => {
  const f = useCurrentFrame();
  const br = Math.sin(f * 0.13) * 4;
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      {/* 床 */}
      <rect x={-270} y={-6} width={580} height={30} rx={15} fill={P.card} />
      <rect x={-252} y={24} width={22} height={52} rx={9} fill={P.card} />
      <rect x={250} y={24} width={22} height={52} rx={9} fill={P.card} />
      {/* 枕头 */}
      <rect x={-242} y={-58} width={150} height={54} rx={24} fill={P.cream} />
      {/* 蜷起的被子（呼吸起伏） */}
      <path d={`M -128 -50 Q -60 ${-112 + br} 62 ${-98 + br} Q 168 ${-88 + br} 212 -30 L 256 -6 L -128 -6 Z`}
        fill={P.lilac} opacity={0.95} />
      <path d={`M -128 -50 Q -60 ${-112 + br} 62 ${-98 + br} Q 168 ${-88 + br} 212 -30`}
        fill="none" stroke={P.cream} strokeWidth={5} strokeLinecap="round" opacity={0.45} />
      {/* 朝向右侧的脸 */}
      <g transform="translate(-172,-82)">
        <circle r={33} fill={C.skin} />
        <path d="M -33 4 Q -34 -34 0 -35 Q 34 -34 33 4 L 30 -6 Q 0 -30 -30 -6 Z" fill="#4a3f66" />
        <path d="M 4 0 q 6 6 12 0" stroke={P.ink} strokeWidth={3.2} fill="none" strokeLinecap="round" />
        <circle cx={22} cy={10} r={4} fill={C.skinDark} opacity={0.8} />
      </g>
      {/* 被口露出的小手 */}
      <circle cx={-112} cy={-56} r={12} fill={C.skin} />
      {zzz && <Zzz x={-96} y={-146} k={0.85} />}
    </g>
  );
};

/* ---------- 梦云（内容物自定义，右缘消散） ---------- */
const DreamCloud = ({x, y, s = 1, children}: {x: number; y: number; s?: number; children?: React.ReactNode}) => {
  const f = useCurrentFrame();
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      <g fill={P.cream}>
        <circle cx={-64} cy={8} r={46} />
        <circle cx={-8} cy={-26} r={56} />
        <circle cx={54} cy={0} r={48} />
        <rect x={-100} y={-4} width={196} height={52} rx={26} />
      </g>
      {children}
      {[0, 1, 2, 3].map((i) => {
        const t = (f * 0.02 + i * 0.24) % 1;
        return (
          <circle key={i} cx={104 + t * 120 + rand(i, 3) * 26} cy={-26 + rand(i, 5) * 60 - t * 40}
            r={7 - t * 5} fill={P.lilac} opacity={(1 - t) * 0.8} />
        );
      })}
    </g>
  );
};

/* ---------- 库存卡片盒 ---------- */
const CardCrate = ({x, y, s = 1, open = true}: {x: number; y: number; s?: number; open?: boolean}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    {/* 探出的卡 */}
    <g transform="translate(-46,-70) rotate(-10)"><FaceCard x={0} y={0} s={0.62} /></g>
    <g transform="translate(0,-80)"><FaceCard x={0} y={0} s={0.62} /></g>
    <g transform="translate(46,-70) rotate(10)"><FaceCard x={0} y={0} s={0.62} /></g>
    {/* 盒身 */}
    <rect x={-112} y={-50} width={224} height={156} rx={14} fill={P.card} />
    <rect x={-112} y={-50} width={224} height={30} rx={14} fill={P.bgDeep} opacity={0.6} />
    {/* 铭牌 */}
    <rect x={-70} y={16} width={140} height={52} rx={10} fill={P.bgDeep} stroke={P.amber} strokeWidth={3.5} />
    <text x={0} y={51} textAnchor="middle" fontSize={25} fill={P.amber}
      fontFamily={FONT_HEAD} fontWeight={900}>记忆库存</text>
    {/* 盒脚 */}
    <rect x={-100} y={106} width={18} height={16} rx={6} fill={P.card} />
    <rect x={82} y={106} width={18} height={16} rx={6} fill={P.card} />
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
        <Moon x={270} y={165} scale={0.7} />
        {/* 蜷缩睡姿 + 梦云里的旧人脸卡 */}
        <g transform={`translate(700,430) scale(${pA}) translate(-700,-430)`} opacity={clamp01(pA)}>
          <Sleeper x={700} y={430} s={1.05} />
        </g>
        <g transform={`translate(1150,220) scale(${pA})`} opacity={clamp01(pA)}>
          <circle cx={-330} cy={240} r={7} fill={P.cream} opacity={0.9} />
          <circle cx={-280} cy={198} r={10} fill={P.cream} opacity={0.9} />
          <circle cx={-224} cy={152} r={14} fill={P.cream} opacity={0.9} />
          <circle cx={-166} cy={104} r={19} fill={P.cream} opacity={0.9} />
          <circle cx={-110} cy={62} r={25} fill={P.cream} opacity={0.9} />
          <DreamCloud x={0} y={0} s={1.05}>
            <FaceCard x={0} y={-10} s={0.82} lit={true} />
            <text x={0} y={42} textAnchor="middle" fontSize={20} fill={P.ink}
              fontFamily={FONT_HEAD} fontWeight={800}>多年没联系的人</text>
          </DreamCloud>
        </g>
        {/* 右侧库存盒：随机抽出一张 */}
        <g transform={`translate(1560,380) scale(${pB})`} opacity={clamp01(pB)}>
          <CardCrate x={0} y={0} s={1.0} />
          <g transform={`translate(-96,-140) rotate(-14)`}>
            <FaceCard x={0} y={0} s={0.72} lit={true} />
          </g>
          <g transform="translate(120,-170)">
            <circle r={40} fill={P.coral} />
            <text y={14} textAnchor="middle" fontSize={42} fill={P.cream}
              fontFamily={FONT_HEAD} fontWeight={900}>?</text>
          </g>
        </g>
        {/* 两侧小标签 */}
        <g transform={`translate(380,560) scale(${pB})`} opacity={clamp01(pB)}>
          <TagPill x={0} y={0} t="他在想我吗？" col={P.coral} w={280} />
        </g>
        <g transform={`translate(1560,600) scale(${pB})`} opacity={clamp01(pB)}>
          <TagPill x={0} y={0} t="库存随机翻牌" col={P.amber} w={280} />
        </g>
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 480}}>
          <div style={{color: P.blue, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 24}}>
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

/* ============ S0 抽到一张旧卡片：三连猜 vs 随机翻卡 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const b2 = beat(2);
  const popL = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const popR = Math.max(0.001, spring({frame: f - 40, fps, config: {damping: 11, stiffness: 120}}));
  const barK = clamp01((f - b1 - 5) / 24);
  const guess = [0, 1, 2].map((i) => Math.max(0.001, spring({frame: f - 30 - i * 14, fps, config: {damping: 12, stiffness: 150}})));
  const cardBob = Math.sin(f * 0.16) * 8;

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={0} name={sec.name} accent={P.coral} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={8} from={0} dur={26} />
        {/* 左：醒来三连猜 */}
        <g transform={`translate(120,150) scale(${popL}) translate(-120,-150)`} opacity={clamp01(popL * 2)}>
          <rect x={120} y={150} width={760} height={440} rx={20} fill={P.panel} stroke={P.coral} strokeWidth={4} />
          <text x={156} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            醒来第一反应 · 三连猜</text>
          {/* 刚坐起的小人 */}
          <g transform="translate(300,440)">
            <rect x={-170} y={60} width={340} height={26} rx={13} fill={P.card} />
            <circle cx={0} cy={-88} r={40} fill={C.skin} />
            <path d="M -40 -84 Q -36 -132 4 -130 Q 40 -128 40 -84 L 36 -94 Q 0 -114 -36 -94 Z" fill="#4a3f66" />
            <path d="M -16 -84 q 5 -7 12 0 M 10 -84 q 5 -7 12 0" stroke={P.ink} strokeWidth={3.2}
              fill="none" strokeLinecap="round" />
            <rect x={-52} y={-48} width={104} height={112} rx={40} fill={P.teal} />
            {/* 挠头的手 */}
            <g transform={`translate(46,-104) rotate(${Math.sin(f * 0.3) * 8})`}>
              <circle r={16} fill={C.skin} />
            </g>
            <text x={0} y={104} textAnchor="middle" fontSize={21} fill={P.muted}
              fontFamily={FONT_HEAD}>他怎么冒出来了</text>
          </g>
          {/* 三个猜测泡泡 */}
          {[
            {t: '他在想我？', x: 620, y: 250},
            {t: '有什么寓意？', x: 640, y: 370},
            {t: '潜意识提醒？', x: 620, y: 490},
          ].map((g0, i) => (
            <g key={i} transform={`translate(${g0.x},${g0.y}) scale(${guess[i]}) translate(${-g0.x},${-g0.y})`}
              opacity={clamp01(guess[i] * 2)}>
              <rect x={g0.x - 130} y={g0.y - 34} width={260} height={68} rx={34}
                fill={P.bgDeep} stroke={P.coral} strokeWidth={3.5} />
              <circle cx={g0.x - 160} cy={g0.y + 40} r={8} fill={P.coral} opacity={0.7} />
              <circle cx={g0.x - 140} cy={g0.y + 52} r={5} fill={P.coral} opacity={0.5} />
              <text x={g0.x} y={g0.y + 10} textAnchor="middle" fontSize={27} fill={P.coral}
                fontFamily={FONT_HEAD} fontWeight={800}>{g0.t}</text>
            </g>
          ))}
        </g>
        {/* 右：大脑翻库存 */}
        <g transform={`translate(1000,150) scale(${popR}) translate(-1000,-150)`} opacity={clamp01(popR * 2)}>
          <rect x={1000} y={150} width={800} height={440} rx={20} fill={P.panel} stroke={P.amber} strokeWidth={4} />
          <text x={1036} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            神经科学的平淡答案 · 夜里在盘库</text>
          <CardCrate x={1260} y={420} s={1.05} />
          {/* 被抽中的旧卡片 */}
          <g transform={`translate(1530,${340 + cardBob})`}>
            <FaceCard x={0} y={0} s={1.15} lit={true} label="小学同桌" tag="旧" />
          </g>
          {/* 抽取虚线轨迹 */}
          <path d="M 1370 330 Q 1440 270 1480 300" fill="none" stroke={P.amber}
            strokeWidth={4} strokeDasharray="8 8" opacity={0.8} />
          <g transform="translate(1700,300)">
            <circle r={34} fill={P.bgDeep} stroke={P.amber} strokeWidth={3.5} />
            <text y={10} textAnchor="middle" fontSize={26} fill={P.amber}
              fontFamily={FONT_HEAD} fontWeight={900}>抽</text>
          </g>
          <text x={1400} y={556} textAnchor="middle" fill={P.cream} fontSize={27}
            fontFamily={FONT_HEAD} fontWeight={800}>整理库存时 · 随机翻到一张旧卡片</text>
        </g>
        {/* 中央 ≠ */}
        <g transform="translate(960,370)">
          <circle r={40} fill={P.coral} />
          <text y={14} textAnchor="middle" fontSize={34} fill={P.cream}
            fontFamily={FONT_HEAD} fontWeight={900}>≠</text>
        </g>
        {/* 底部结论（cue2 后浮出） */}
        {barK > 0 && (
          <g opacity={barK} transform={`translate(0,${(1 - barK) * 60})`}>
            <rect x={120} y={648} width={1680} height={210} rx={20} fill={P.panel} stroke={P.amber} strokeWidth={4} />
            <text x={160} y={710} fill={P.amber} fontSize={30} fontFamily={FONT_HEAD} fontWeight={900}>
              它只说一句 ·「这条记忆还在」</text>
            <text x={160} y={770} fill={P.cream} fontSize={29} fontFamily={FONT_HEAD} fontWeight={700}>
              不是跨越时空的信息，不是对方在想你 —— 碰巧翻到而已，碰巧才是主角</text>
            {/* 右侧小图 */}
            <g transform="translate(1500,712)">
              <FaceCard x={-110} y={0} s={0.72} lit={true} />
              <g transform="translate(60,0)">
                <rect x={-110} y={-30} width={220} height={60} rx={14} fill={P.bgDeep} stroke={P.amber} strokeWidth={3.5} />
                <text x={0} y={9} textAnchor="middle" fontSize={24} fill={P.amber}
                  fontFamily={FONT_HEAD} fontWeight={800}>碰巧翻到</text>
              </g>
              <text x={-10} y={76} textAnchor="middle" fontSize={21} fill={P.muted}
                fontFamily={FONT_HEAD}>库存盘点 · 不是信号</text>
            </g>
          </g>
        )}
      </svg>
      <Chip text="梦到旧人 ≠ 他在想你" x={1240} y={72} at={beat(0) + 25} accent={P.coral} />
      <Stamp text="碰巧才是主角" x={620} y={86} at={Math.min(b2, b1 + 50)} color={P.amberD} size={38} rot={-3} />
    </FadeIn>
  );
};

/* ============ S1 库存比你想的大：深层卡片墙 + 随机点亮 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const b2 = beat(2);
  const popL = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const popR = Math.max(0.001, spring({frame: f - 35, fps, config: {damping: 11, stiffness: 120}}));
  const barK = clamp01((f - b2 - 5) / 24);
  const sinkT = (f * 0.008) % 1;

  /* 卡片墙 6×3 */
  const cols = [720, 910, 1100, 1290, 1480, 1670];
  const rows = [295, 435, 560];
  const N = 18;
  const sweep = f < b1 ? Math.floor(rand(Math.floor(f / 26), 5) * N) : 10;
  const pickX = cols[10 % 6];
  const pickY = rows[Math.floor(10 / 6)];

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={1} name={sec.name} accent={P.amber} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：数量与分层 */}
        <g transform={`translate(120,150) scale(${popL}) translate(-120,-150)`} opacity={clamp01(popL * 2)}>
          <rect x={120} y={150} width={460} height={470} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={156} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            这辈子见过的人</text>
          <text x={350} y={292} textAnchor="middle" fontSize={72} fill={P.amber}
            fontFamily={FONT_HEAD} fontWeight={900}>几千个</text>
          {/* 三层库存 */}
          {[
            {y: 340, t: '浅层 · 常联系', col: P.teal, op: 1},
            {y: 432, t: '中层 · 几年没见', col: P.muted, op: 0.75},
            {y: 524, t: '深层 · 以为早忘了', col: P.card, op: 0.55},
          ].map((L, i) => (
            <g key={i} opacity={L.op}>
              <rect x={156} y={L.y} width={388} height={72} rx={12} fill={P.bgDeep}
                stroke={L.col} strokeWidth={3.5} />
              <text x={186} y={L.y + 45} fontSize={24} fill={L.col}
                fontFamily={FONT_HEAD} fontWeight={800}>{L.t}</text>
              <FaceCard x={508} y={L.y + 36} s={0.42} dim={i > 0} />
            </g>
          ))}
          {/* 下沉箭头 */}
          <g transform={`translate(382,${340 + sinkT * 190})`} opacity={0.9 - sinkT * 0.5}>
            <polygon points="0,0 -12,-16 12,-16" fill={P.amber} />
          </g>
          <text x={350} y={612} textAnchor="middle" fill={P.cream} fontSize={23}
            fontFamily={FONT_HEAD} fontWeight={700}>没被删除 · 只是压深了</text>
        </g>
        {/* 右：卡片墙 */}
        <g transform={`translate(620,150) scale(${popR}) translate(-620,-150)`} opacity={clamp01(popR * 2)}>
          <rect x={620} y={150} width={1180} height={470} rx={20} fill={P.panel} stroke={P.amber} strokeWidth={4} />
          <text x={656} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            记忆库存墙 · 每张都还在</text>
          {rows.map((ry, r) =>
            cols.map((cx, c) => {
              const idx = r * 6 + c;
              const isPick = idx === sweep;
              return (
                <FaceCard key={idx} x={cx} y={ry} s={0.78}
                  dim={!isPick} lit={isPick}
                  rot={isPick ? Math.sin(f * 0.2) * 4 : 0} />
              );
            })
          )}
          {/* 被选中的那张 */}
          {f >= b1 && (
            <g>
              <circle cx={pickX} cy={pickY} r={76} fill="none" stroke={P.amber}
                strokeWidth={4} strokeDasharray="10 8" transform={`rotate(${f * 2} ${pickX} ${pickY})`} />
              <TagPill x={pickX} y={pickY + 108} t="恰好被点亮" col={P.amber} w={200} fs={24} s={0.9} />
            </g>
          )}
          <text x={1210} y={206} textAnchor="end" fill={P.amber} fontSize={22}
            fontFamily={FONT_HEAD} fontWeight={800} opacity={0.8}>
            {f < b1 ? '随机扫过…' : '抽中一张！'}</text>
        </g>
        {/* 底部结论（cue2） */}
        {barK > 0 && (
          <g opacity={barK} transform={`translate(0,${(1 - barK) * 60})`}>
            <rect x={120} y={648} width={1680} height={210} rx={20} fill={P.panel} stroke={P.teal} strokeWidth={4} />
            <text x={160} y={710} fill={P.teal} fontSize={30} fontFamily={FONT_HEAD} fontWeight={900}>
              抓谁不看谁重要 · 只看痕迹恰好被点亮</text>
            <text x={160} y={770} fill={P.cream} fontSize={29} fontFamily={FONT_HEAD} fontWeight={700}>
              想不起 ≠ 不存在 —— 夜里整理巩固时随手抓材料，抽到谁纯属碰巧</text>
            <g transform="translate(1520,714)">
              <rect x={-250} y={-32} width={230} height={62} rx={14} fill={P.bgDeep} stroke={P.card} strokeWidth={3.5} />
              <text x={-135} y={7} textAnchor="middle" fontSize={23} fill={P.muted}
                fontFamily={FONT_HEAD} fontWeight={800}>重要性排序</text>
              <MarkX x={-20} y={-2} s={0.85} />
              <rect x={20} y={-32} width={230} height={62} rx={14} fill={P.bgDeep} stroke={P.teal} strokeWidth={3.5} />
              <text x={135} y={7} textAnchor="middle" fontSize={23} fill={P.teal}
                fontFamily={FONT_HEAD} fontWeight={800}>随机点亮</text>
            </g>
          </g>
        )}
      </svg>
      <Chip text="几千张脸都还在" x={1300} y={72} at={beat(0) + 25} accent={P.amber} />
      <Stamp text="想不起 ≠ 不存在" x={600} y={86} at={b1 + 40} color={P.tealD} size={38} rot={3} />
    </FadeIn>
  );
};

/* ============ S2 翻的不是通讯录：人名索引× vs 联结图谱√ ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const popL = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const popR = Math.max(0.001, spring({frame: f - 35, fps, config: {damping: 11, stiffness: 120}}));
  const barK = clamp01((f - b1 - 5) / 24);
  const pkg = (f * 0.011) % 1; /* 沿边跑的激活包 */

  /* 联结图谱：情绪节点 → 五张脸卡 */
  const emo = {x: 850, y: 430};
  const nodes = [
    {x: 1230, y: 250}, {x: 1450, y: 290}, {x: 1560, y: 430},
    {x: 1420, y: 570}, {x: 1180, y: 565},
  ];
  const hot = 1; /* 被联结点亮的那条边 */

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={2} name={sec.name} accent={P.teal} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：通讯录被否掉 */}
        <g transform={`translate(120,150) scale(${popL}) translate(-120,-150)`} opacity={clamp01(popL * 2)}>
          <rect x={120} y={150} width={560} height={470} rx={20} fill={P.panel} stroke={P.coral} strokeWidth={4} />
          <text x={156} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            不是按人名翻通讯录</text>
          {/* 通讯录本子 */}
          <g transform="translate(400,392)">
            <rect x={-160} y={-185} width={320} height={370} rx={16} fill={P.bgLite}
              stroke={P.muted} strokeWidth={4} />
            <rect x={-160} y={-185} width={34} height={370} rx={16} fill={P.bgDeep} />
            {[0, 1, 2, 3, 4, 5].map((i) => (
              <circle key={i} cx={-143} cy={-140 + i * 56} r={7} fill={P.muted} />
            ))}
            {/* 字母标签 */}
            {['A', 'C', 'K', 'W', 'Z'].map((L, i) => (
              <g key={i}>
                <rect x={160} y={-160 + i * 70} width={44} height={50} rx={8}
                  fill={P.card} opacity={0.85} />
                <text x={182} y={-126 + i * 70} textAnchor="middle" fontSize={22}
                  fill={P.muted} fontFamily={FONT_HEAD} fontWeight={800}>{L}</text>
              </g>
            ))}
            {/* 人名行 */}
            {[0, 1, 2, 3, 4].map((i) => (
              <g key={i} opacity={0.75}>
                <circle cx={-96} cy={-140 + i * 66} r={15} fill={P.card} />
                <line x1={-64} y1={-146 + i * 66} x2={100 - rand(i, 4) * 50} y2={-146 + i * 66}
                  stroke={P.muted} strokeWidth={7} strokeLinecap="round" />
                <line x1={-64} y1={-128 + i * 66} x2={60 - rand(i, 6) * 40} y2={-128 + i * 66}
                  stroke={P.card} strokeWidth={6} strokeLinecap="round" />
              </g>
            ))}
          </g>
          <MarkX x={400} y={392} s={2.4} />
          <text x={400} y={606} textAnchor="middle" fill={P.coral} fontSize={24}
            fontFamily={FONT_HEAD} fontWeight={800}>大脑不按字母表找人</text>
        </g>
        {/* 右：神经联结图谱 */}
        <g transform={`translate(720,150) scale(${popR}) translate(-720,-150)`} opacity={clamp01(popR * 2)}>
          <rect x={720} y={150} width={1080} height={470} rx={20} fill={P.panel} stroke={P.teal} strokeWidth={4} />
          <text x={756} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            而是按联结图谱 · 谁连着就拉谁</text>
          {/* 边 */}
          {nodes.map((n, i) => {
            const active = i === hot;
            return (
              <line key={i} x1={emo.x + 60} y1={emo.y} x2={n.x - 50} y2={n.y}
                stroke={active ? P.teal : P.card} strokeWidth={active ? 6 : 4}
                strokeDasharray={active ? '14 12' : '6 10'}
                strokeDashoffset={active ? -f * 2.2 : 0}
                opacity={active ? 0.95 : 0.5} />
            );
          })}
          {/* 激活包沿 hot 边走 */}
          {(() => {
            const n = nodes[hot];
            const px = emo.x + 60 + (n.x - 50 - emo.x - 60) * pkg;
            const py = emo.y + (n.y - emo.y) * pkg;
            return <circle cx={px} cy={py} r={11} fill={P.teal} />;
          })()}
          {/* 脸卡节点 */}
          {nodes.map((n, i) => (
            <g key={i}>
              <FaceCard x={n.x} y={n.y} s={0.62} dim={i !== hot} lit={i === hot} />
              {i === hot && f > b1 - 60 && (
                <g transform={`translate(${n.x},${n.y - 78})`}>
                  <rect x={-86} y={-22} width={172} height={44} rx={12} fill={P.bgDeep}
                    stroke={P.teal} strokeWidth={3.5} />
                  <text y={8} textAnchor="middle" fontSize={22} fill={P.teal}
                    fontFamily={FONT_HEAD} fontWeight={800}>被拉进梦里</text>
                </g>
              )}
            </g>
          ))}
          {/* 情绪节点 */}
          <g transform={`translate(${emo.x},${emo.y})`}>
            <circle r={66} fill={P.coral} opacity={0.22} />
            <circle r={52} fill={P.coral} />
            <path d="M 0 -12 C -26 -32 -50 -8 0 30 C 50 -8 26 -32 0 -12 Z" fill={P.cream} />
            <text x={0} y={100} textAnchor="middle" fontSize={22} fill={P.coral}
              fontFamily={FONT_HEAD} fontWeight={800}>正在加工的情绪</text>
          </g>
          <text x={1760} y={592} textAnchor="end" fill={P.teal} fontSize={22}
            fontFamily={FONT_HEAD} fontWeight={800}>和当前情绪连着 → 顺带出场</text>
        </g>
        {/* 底部结论（cue1） */}
        {barK > 0 && (
          <g opacity={barK} transform={`translate(0,${(1 - barK) * 60})`}>
            <rect x={120} y={648} width={1680} height={210} rx={20} fill={P.panel} stroke={P.teal} strokeWidth={4} />
            <text x={160} y={710} fill={P.teal} fontSize={30} fontFamily={FONT_HEAD} fontWeight={900}>
              像翻相册的某个分区 · 照片里的人只是顺带出场</text>
            <text x={160} y={770} fill={P.cream} fontSize={29} fontFamily={FONT_HEAD} fontWeight={700}>
              大脑要加工的是那段记忆和情绪 —— 不是那个人本身</text>
            {/* 相册小图 */}
            <g transform="translate(1530,724)">
              <rect x={-120} y={-56} width={240} height={112} rx={14} fill={P.bgDeep} stroke={P.amber} strokeWidth={3.5} />
              <FaceCard x={-66} y={-6} s={0.5} dim={true} />
              <FaceCard x={0} y={-6} s={0.5} lit={true} />
              <FaceCard x={66} y={-6} s={0.5} dim={true} />
              <text x={0} y={82} textAnchor="middle" fontSize={20} fill={P.muted}
                fontFamily={FONT_HEAD}>相册分区 · 顺带被翻出</text>
            </g>
          </g>
        )}
      </svg>
      <Chip text="联结图谱 · 不按人名" x={1260} y={72} at={beat(0) + 25} accent={P.teal} />
      <Stamp text="顺带出场而已" x={600} y={86} at={b1 + 40} color={P.tealD} size={38} rot={-3} />
    </FadeIn>
  );
};

/* ============ S3 旧人是借来的道具：似曾相识 → 梦的三格跳剪 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const b2 = beat(2);
  const bandPop = Math.max(0.001, spring({frame: f - 8, fps, config: {damping: 11, stiffness: 120}}));
  const linkK = clamp01((f - 50) / 26);
  const stripPop = Math.max(0.001, spring({frame: f - 90, fps, config: {damping: 11, stiffness: 120}}));
  const barK = clamp01((f - b2 - 5) / 24);
  const wavePhase = f * 0.09;

  /* 情绪波：横贯三格分镜 */
  const waveY = (x: number) => 560 + Math.sin(x / 190 + wavePhase) * 22;
  const wavePts = Array.from({length: 57}, (_, i) => `${170 + i * 28},${waveY(170 + i * 28)}`).join(' ');

  /* 道具小人：剪影 + 借来的脸牌 */
  const PropMan = ({x, y, s = 1, pose = 0}: {x: number; y: number; s?: number; pose?: number}) => (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      <ellipse cx={0} cy={-64} rx={16} ry={20} fill={P.ink} opacity={0.85} />
      <path d={pose === 0 ? 'M -20 -44 L 20 -44 L 26 30 L -26 30 Z' : 'M -24 -44 L 18 -48 L 30 26 L -20 34 Z'}
        fill={P.ink} opacity={0.85} />
      {/* 插在地上的脸牌 */}
      <line x1={18} y1={-30} x2={30} y2={-66} stroke={P.card} strokeWidth={5} strokeLinecap="round" />
      <FaceCard x={34} y={-78} s={0.44} lit={true} />
    </g>
  );

  return (
    <FadeIn>
      <Backdrop top={P.bgDeep} bottom={P.bgDeep} />
      <SceneTag index={3} name={sec.name} accent={P.lilac} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={8} from={0} dur={26} />
        {/* 顶条：白天埋的钩子 */}
        <g transform={`translate(120,140) scale(${bandPop}) translate(-120,-140)`} opacity={clamp01(bandPop * 2)}>
          <rect x={120} y={140} width={1680} height={140} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={160} y={200} fill={P.muted} fontSize={24} fontFamily={FONT_HEAD}>
            白天根本没想他 · 但一个场景感觉像当年</text>
          {/* 场景照片 A：街角小店 */}
          <g transform="translate(790,208)">
            <rect x={-110} y={-56} width={220} height={112} rx={12} fill={P.bgLite} stroke={P.muted} strokeWidth={3.5} />
            <rect x={-90} y={-30} width={80} height={66} rx={6} fill={P.card} />
            <rect x={-90} y={-46} width={180} height={18} rx={7} fill={P.coral} opacity={0.8} />
            <line x1={30} y1={-20} x2={30} y2={36} stroke={P.muted} strokeWidth={5} />
            <text x={0} y={76} textAnchor="middle" fontSize={19} fill={P.muted}
              fontFamily={FONT_HEAD}>今天路过的小店</text>
          </g>
          {/* 关联虚线 */}
          <path d={`M 900 208 L ${900 + 180 * linkK} 208`} stroke={P.teal} strokeWidth={4}
            strokeDasharray="9 8" />
          <g transform="translate(990,208)" opacity={linkK}>
            <circle r={26} fill={P.teal} />
            <text y={9} textAnchor="middle" fontSize={21} fill={P.ink}
              fontFamily={FONT_HEAD} fontWeight={900}>像</text>
          </g>
          {/* 场景照片 B：老教室 */}
          <g transform="translate(1220,208)" opacity={0.35 + linkK * 0.65}>
            <rect x={-110} y={-56} width={220} height={112} rx={12} fill={P.bgLite}
              stroke={linkK > 0.6 ? P.amber : P.muted} strokeWidth={3.5} />
            <rect x={-84} y={-36} width={110} height={52} rx={6} fill={P.card} />
            <line x1={-76} y1={-20} x2={-20} y2={-20} stroke={P.muted} strokeWidth={4} />
            <line x1={-76} y1={-4} x2={-34} y2={-4} stroke={P.muted} strokeWidth={4} />
            <rect x={48} y={-10} width={44} height={46} rx={6} fill={P.card} />
            <text x={0} y={76} textAnchor="middle" fontSize={19} fill={P.amber}
              fontFamily={FONT_HEAD}>当年的教室</text>
          </g>
          <text x={1770} y={216} textAnchor="end" fill={P.cream} fontSize={26}
            fontFamily={FONT_HEAD} fontWeight={800} opacity={0.35 + linkK * 0.65}>
            夜里借那张脸当符号</text>
        </g>
        {/* 主区：梦的三格胶片 */}
        <g transform={`translate(120,320) scale(${stripPop}) translate(-120,-320)`} opacity={clamp01(stripPop * 2)}>
          <text x={160} y={352} fill={P.lilac} fontSize={25} fontFamily={FONT_HEAD} fontWeight={900}>
            梦的剪辑 · 场景乱跳</text>
          {/* 胶片框 */}
          {[0, 1, 2].map((c) => {
            const x0 = 150 + c * 556;
            return (
              <g key={c}>
                <rect x={x0} y={380} width={500} height={230} rx={14} fill={P.bgLite}
                  stroke={P.card} strokeWidth={4} />
                {[0, 1, 2, 3, 4, 5].map((h) => (
                  <g key={h}>
                    <rect x={x0 + 18 + h * 80} y={386} width={34} height={10} rx={5} fill={P.bgDeep} />
                    <rect x={x0 + 18 + h * 80} y={594} width={34} height={10} rx={5} fill={P.bgDeep} />
                  </g>
                ))}
              </g>
            );
          })}
          {/* 格 1：教室 */}
          <g transform="translate(150,380)">
            <rect x={70} y={50} width={150} height={80} rx={6} fill={P.card} />
            <line x1={86} y1={74} x2={180} y2={74} stroke={P.muted} strokeWidth={4} />
            <line x1={86} y1={92} x2={160} y2={92} stroke={P.muted} strokeWidth={4} />
            <rect x={300} y={130} width={120} height={60} rx={8} fill={P.card} />
            <PropMan x={300} y={170} s={0.8} />
            <text x={250} y={222} textAnchor="middle" fontSize={20} fill={P.muted}
              fontFamily={FONT_HEAD}>教室</text>
          </g>
          {/* 格 2：海边 */}
          <g transform="translate(706,380)">
            <path d="M 20 170 Q 80 140 140 170 Q 200 200 260 170 Q 320 140 380 170 Q 440 200 480 170 L 480 190 L 20 190 Z"
              fill={P.blue} opacity={0.5} />
            <circle cx={400} cy={60} r={26} fill={P.amber} />
            <path d="M 330 96 L 330 60 L 372 84 Z" fill={P.cream} />
            <line x1={330} y1={60} x2={330} y2={96} stroke={P.muted} strokeWidth={4} />
            <PropMan x={140} y={170} s={0.8} pose={1} />
            <text x={250} y={222} textAnchor="middle" fontSize={20} fill={P.muted}
              fontFamily={FONT_HEAD}>海边</text>
          </g>
          {/* 格 3：陌生城市 */}
          <g transform="translate(1262,380)">
            {[0, 1, 2, 3].map((b) => (
              <rect key={b} x={50 + b * 100} y={70 - (b % 2) * 24} width={76}
                height={120 + (b % 2) * 24} rx={6} fill={P.card} />
            ))}
            {[0, 1, 2, 3].map((b) => (
              <rect key={b} x={66 + b * 100} y={88 - (b % 2) * 24} width={14} height={14}
                fill={P.amber} opacity={0.8} />
            ))}
            <circle cx={430} cy={46} r={20} fill={P.cream} />
            <circle cx={436} cy={42} r={16} fill={P.bgLite} />
            <PropMan x={220} y={170} s={0.8} />
            <text x={250} y={222} textAnchor="middle" fontSize={20} fill={P.muted}
              fontFamily={FONT_HEAD}>陌生城市</text>
          </g>
          {/* 情绪贯穿波 */}
          {b1 <= f && (
            <g>
              <polyline points={wavePts} fill="none" stroke={P.coral} strokeWidth={6}
                strokeLinecap="round" opacity={0.9} />
              <g transform="translate(1730,560)">
                <rect x={-30} y={-26} width={180} height={52} rx={12} fill={P.bgDeep}
                  stroke={P.coral} strokeWidth={3.5} />
                <text x={60} y={9} textAnchor="middle" fontSize={22} fill={P.coral}
                  fontFamily={FONT_HEAD} fontWeight={800}>情绪一条线</text>
              </g>
            </g>
          )}
        </g>
        {/* 底部结论（cue2） */}
        {barK > 0 && (
          <g opacity={barK} transform={`translate(0,${(1 - barK) * 60})`}>
            <rect x={120} y={648} width={1680} height={210} rx={20} fill={P.panel} stroke={P.lilac} strokeWidth={4} />
            <text x={160} y={710} fill={P.lilac} fontSize={30} fontFamily={FONT_HEAD} fontWeight={900}>
              旧素材是借来的道具 · 新故事才是主角</text>
            <text x={160} y={770} fill={P.cream} fontSize={29} fontFamily={FONT_HEAD} fontWeight={700}>
              别追问「他为什么出现」—— 该问的是那段情绪为什么借了他的脸</text>
            <g transform="translate(1540,716)">
              <FaceCard x={-120} y={-4} s={0.66} lit={true} tag="借" />
              <text x={60} y={4} fontSize={24} fill={P.muted} fontFamily={FONT_HEAD} fontWeight={700}>
                道具脸牌 · 用完就还</text>
              <text x={60} y={46} fontSize={21} fill={P.muted} fontFamily={FONT_HEAD}>
                情绪线索上的一个节点</text>
            </g>
          </g>
        )}
      </svg>
      <Chip text="借脸当符号" x={1330} y={72} at={beat(0) + 25} accent={P.lilac} />
      <Stamp text="场景乱跳 · 情绪连贯" x={540} y={86} at={b1 + 30} color={P.coralD} size={36} rot={-3} />
    </FadeIn>
  );
};

/* ============ S4 不是神谕是盘点：随机抽样球 + 情绪强度计 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const b2 = beat(2);
  const popL = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const popR = Math.max(0.001, spring({frame: f - 35, fps, config: {damping: 11, stiffness: 120}}));
  const barK = clamp01((f - b1 - 5) / 24);
  const doneK = clamp01((f - b2 - 5) / 24);

  /* 摇奖球内翻滚的卡 */
  const tumbling = [0, 1, 2, 3, 4, 5].map((i) => {
    const a = f * (0.024 + i * 0.004) + i * 1.05;
    const rx = 62 + Math.sin(f * 0.05 + i * 2.2) * 26;
    const ry = 44 + Math.cos(f * 0.04 + i * 1.7) * 20;
    return {x: Math.cos(a) * rx, y: Math.sin(a) * ry - 20, r: Math.sin(f * 0.11 + i) * 16};
  });

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={4} name={sec.name} accent={P.coral} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：神经噪声摇奖球 */}
        <g transform={`translate(120,150) scale(${popL}) translate(-120,-150)`} opacity={clamp01(popL * 2)}>
          <rect x={120} y={150} width={780} height={440} rx={20} fill={P.panel} stroke={P.amber} strokeWidth={4} />
          <text x={156} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            机制一 · 神经噪声随机抽样</text>
          {/* 摇奖球 */}
          <g transform="translate(360,390)">
            <circle r={130} fill={P.bgDeep} stroke={P.teal} strokeWidth={5} />
            {tumbling.map((c, i) => (
              <FaceCard key={i} x={c.x} y={c.y} s={0.4} rot={c.r} dim={i % 3 === 2} />
            ))}
            {/* 球口 */}
            <rect x={-30} y={126} width={60} height={34} rx={10} fill={P.teal} />
            {/* 底座 */}
            <rect x={-96} y={160} width={192} height={26} rx={12} fill={P.card} />
            <rect x={-66} y={186} width={132} height={18} rx={9} fill={P.card} opacity={0.7} />
          </g>
          {/* 出口滑出一张 */}
          <g transform={`translate(${560 + Math.min(90, f * 1.2 % 90)},545) rotate(-8)`}>
            <FaceCard x={0} y={0} s={0.62} lit={true} />
          </g>
          <path d="M 490 545 L 545 545" stroke={P.teal} strokeWidth={4} strokeDasharray="7 6" />
          <TagPill x={720} y={470} t="抽到哪张是哪张" col={P.teal} w={240} fs={24} s={0.9} />
          <text x={510} y={300} fill={P.cream} fontSize={24} fontFamily={FONT_HEAD} fontWeight={700}
            textAnchor="middle">抽样不按重要性排序</text>
        </g>
        {/* 右：情绪强度计 */}
        <g transform={`translate(980,150) scale(${popR}) translate(-980,-150)`} opacity={clamp01(popR * 2)}>
          <rect x={980} y={150} width={820} height={440} rx={20} fill={P.panel} stroke={P.coral} strokeWidth={4} />
          <text x={1016} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            机制二 · 情绪强度够格才容易被翻</text>
          {/* 竖向刻度计 */}
          <g transform="translate(1120,390)">
            <rect x={-46} y={-140} width={92} height={280} rx={20} fill={P.bgDeep}
              stroke={P.coral} strokeWidth={4.5} />
            <rect x={-32} y={-140 + 264 * (1 - 0.72)} width={64} height={264 * 0.72}
              rx={12} fill={P.coral} opacity={0.9} />
            <line x1={-58} y1={-66} x2={58} y2={-66} stroke={P.amber} strokeWidth={4}
              strokeDasharray="8 6" />
            <text x={78} y={-58} fontSize={21} fill={P.amber} fontFamily={FONT_HEAD} fontWeight={800}>够格线</text>
            {[0.25, 0.5, 0.75].map((k) => (
              <line key={k} x1={-46} y1={-140 + 264 * k} x2={-28} y2={-140 + 264 * k}
                stroke={P.muted} strokeWidth={4} />
            ))}
            <text x={0} y={172} textAnchor="middle" fontSize={21} fill={P.coral}
              fontFamily={FONT_HEAD} fontWeight={800}>情绪强度</text>
          </g>
          {/* 两张记忆卡：高强 vs 低强 */}
          <g transform="translate(1400,330)">
            <FaceCard x={0} y={0} s={0.95} lit={true} label="那段关系" />
            {/* 三心强度 */}
            {[-1, 0, 1].map((h) => (
              <path key={h} d="M 0 0 C -8 -8 -18 0 0 12 C 18 0 8 -8 0 0 Z"
                transform={`translate(${h * 30},-96) scale(0.9)`} fill={P.coral} />
            ))}
            <text x={0} y={130} textAnchor="middle" fontSize={21} fill={P.coral}
              fontFamily={FONT_HEAD} fontWeight={800}>强度够 · 常被翻</text>
          </g>
          <g transform="translate(1650,360)">
            <FaceCard x={0} y={0} s={0.8} dim={true} />
            <path d="M 0 0 C -8 -8 -18 0 0 12 C 18 0 8 -8 0 0 Z"
              transform="translate(0,-78) scale(0.8)" fill={P.muted} opacity={0.6} />
            <text x={0} y={108} textAnchor="middle" fontSize={19} fill={P.muted}
              fontFamily={FONT_HEAD}>强度低 · 躺着不动</text>
          </g>
          <ArrowR x={1320} y={470} len={60} col={P.coral} dash="8 7" />
          <text x={1560} y={556} textAnchor="middle" fill={P.cream} fontSize={23}
            fontFamily={FONT_HEAD} fontWeight={700}>梦到前任 ≠ 还爱 · 是强度标记还在</text>
        </g>
        {/* 中央 + */}
        <g transform="translate(950,370)">
          <circle r={38} fill={P.amber} />
          <text y={13} textAnchor="middle" fontSize={32} fill={P.ink}
            fontFamily={FONT_HEAD} fontWeight={900}>+</text>
        </g>
        {/* 底部结论（cue1） */}
        {barK > 0 && (
          <g opacity={barK} transform={`translate(0,${(1 - barK) * 60})`}>
            <rect x={120} y={648} width={1680} height={210} rx={20} fill={P.panel} stroke={P.coral} strokeWidth={4} />
            <text x={160} y={710} fill={P.coral} fontSize={30} fontFamily={FONT_HEAD} fontWeight={900}>
              这不是神谕 · 是大脑在定期盘点库存</text>
            <text x={160} y={770} fill={P.cream} fontSize={29} fontFamily={FONT_HEAD} fontWeight={700}>
              梦到只说明那段记忆还占着位置 —— 看一眼就收回去，不必赋予深意</text>
            <g transform="translate(1530,724)">
              <rect x={-180} y={-38} width={360} height={76} rx={16} fill={P.bgDeep}
                stroke={P.amber} strokeWidth={4} />
              <text x={0} y={-4} textAnchor="middle" fontSize={25} fill={P.amber}
                fontFamily={FONT_HEAD} fontWeight={900}>库存盘点 · 完成</text>
              <text x={0} y={28} textAnchor="middle" fontSize={19} fill={P.muted}
                fontFamily={FONT_HEAD}>看一眼就收 · 不必行动</text>
            </g>
          </g>
        )}
        {/* cue3 时的对号 */}
        {doneK > 0 && (
          <g opacity={doneK} transform="translate(1560,228)">
            <circle r={54} fill={P.teal} />
            <path d="M -22 0 L -6 16 L 24 -14" stroke={P.ink} strokeWidth={9}
              fill="none" strokeLinecap="round" strokeLinejoin="round" />
          </g>
        )}
      </svg>
      <Chip text="情绪强度 × 随机抽样" x={1240} y={72} at={beat(0) + 25} accent={P.amber} />
      <Stamp text="看一眼就收" x={620} y={86} at={Math.min(b2, b1 + 55)} color={P.tealD} size={38} rot={-3} />
    </FadeIn>
  );
};

/* ============ 结尾卡 ============ */
const EndCard = () => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 14}});
  const pop = Math.max(0.001, spring({frame: f - 5, fps, config: {damping: 10, stiffness: 140}}));
  const slide = clamp01(f / 30);
  return (
    <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
      <Backdrop top={P.bgDeep} bottom={P.bgDeep} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={20} from={0} dur={22} />
        <Moon x={330} y={220} scale={0.8} />
        <g transform={`translate(880,830) scale(${pop})`} opacity={clamp01(pop)}>
          <Sleeper x={0} y={0} s={1.0} zzz={true} />
        </g>
        {/* 卡片收回库存盒 */}
        <g transform={`translate(1420,780) scale(${pop})`} opacity={clamp01(pop)}>
          <CardCrate x={0} y={0} s={1.0} />
          <g transform={`translate(${-60 - slide * 20},${-150 + slide * 60}) rotate(${-14 + slide * 10})`}
            opacity={1 - slide * 0.75}>
            <FaceCard x={0} y={0} s={0.66} lit={true} />
          </g>
        </g>
        <g transform={`translate(1560,950) scale(${pop})`} opacity={clamp01(pop)}>
          <rect x={-200} y={-48} width={400} height={96} rx={18} fill={P.panel} stroke={P.amber} strokeWidth={4} />
          <text x={0} y={-6} textAnchor="middle" fontSize={24} fill={P.amber}
            fontFamily={FONT_HEAD} fontWeight={800}>库存盘点 · 不是神谕</text>
          <text x={0} y={30} textAnchor="middle" fontSize={21} fill={P.muted}
            fontFamily={FONT_HEAD}>那段记忆还在 · 仅此而已</text>
        </g>
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -280}}>
        <div style={{fontSize: 52, fontWeight: 800, color: C.text}}>梦到旧人不是信号 · 是库存盘点</div>
        <div style={{fontSize: 36, color: P.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E240 = ({data = tl}: {data?: any}) => {
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
