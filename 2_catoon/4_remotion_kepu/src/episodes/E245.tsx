import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01} from '../lib/style';
import {Backdrop, Stamp, SceneTag, Chip, Subtitles, ProgressBar, FadeIn} from '../lib/atoms';
import tl from '../timeline_245.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：铁馆深青 × 镁粉白 × 喊声橙 × 气压青 ============ */
const P = {
  bg: '#12333d',
  bgLite: '#17414d',
  bgDeep: '#0a242e',
  panel: '#163a45',
  card: '#2a5a68',
  ink: '#07222b',
  cream: '#fff3de',
  muted: '#a9c7d4',
  shout: '#ff7a45',
  shoutD: '#d8552a',
  air: '#7ee0d2',
  airD: '#45b3a4',
  amber: '#ffd166',
  amberD: '#e0a83c',
  steel: '#cfd8e2',
  steelD: '#8fa2b5',
  belt: '#c08552',
  beltD: '#8f5a2e',
  red: '#ff5d73',
  lime: '#8fe06a',
  sky: '#5cc8ff',
};

/* ---------- 小叉 / 小勾 ---------- */
const MarkX = ({x, y, s = 1, col = P.red}: {x: number; y: number; s?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} stroke={col} strokeWidth={7} strokeLinecap="round">
    <line x1={-13} y1={-13} x2={13} y2={13} /><line x1={13} y1={-13} x2={-13} y2={13} />
  </g>
);
const Check = ({x, y, s = 1, col = P.lime}: {x: number; y: number; s?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} stroke={col} strokeWidth={9}
    strokeLinecap="round" strokeLinejoin="round" fill="none">
    <path d="M -16 1 L -5 13 L 18 -13" />
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

/* ---------- 向右扩散的喊声弧线 ---------- */
const ShoutArcs = ({x, y, s = 1, col = P.shout, rot = 0}: {x: number; y: number; s?: number; col?: string; rot?: number}) => {
  const f = useCurrentFrame();
  return (
    <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`}>
      {[34, 62, 90].map((r, i) => {
        const wob = Math.sin(f * 0.35 + i * 1.3) * 0.12;
        const x0 = Math.cos(-0.9) * r, y0 = Math.sin(-0.9) * r;
        const x1 = Math.cos(0.9) * r, y1 = Math.sin(0.9) * r;
        return (
          <path key={i} d={`M ${x0} ${y0} A ${r} ${r} 0 0 1 ${x1} ${y1}`}
            fill="none" stroke={col} strokeWidth={8} strokeLinecap="round"
            opacity={clamp01(0.9 - i * 0.25 + wob)} />
        );
      })}
    </g>
  );
};

/* ---------- 喷气短线 ---------- */
const Jets = ({x, y, s = 1, col = P.air, rot = 0}: {x: number; y: number; s?: number; col?: string; rot?: number}) => {
  const f = useCurrentFrame();
  const o = 0.62 + Math.sin(f * 0.5) * 0.28;
  return (
    <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`} opacity={o}>
      {[-18, 0, 18].map((dx, i) => (
        <line key={i} x1={dx} y1={0} x2={dx + (i - 1) * 10} y2={-40 - i * 8}
          stroke={col} strokeWidth={7} strokeLinecap="round" />
      ))}
    </g>
  );
};

/* ---------- 杠铃（正面横杠 + 铃片） ---------- */
const Barbell = ({x, y, s = 1, w = 460}: {x: number; y: number; s?: number; w?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-w / 2} y={-7} width={w} height={14} rx={7} fill={P.steel} />
    {[-1, 1].map((side) => (
      <g key={side} transform={`scale(${side},1)`}>
        <rect x={w / 2 - 34} y={-64} width={26} height={128} rx={10} fill={P.shout} stroke={P.ink} strokeWidth={4} />
        <rect x={w / 2 - 64} y={-50} width={22} height={100} rx={9} fill={P.shoutD} stroke={P.ink} strokeWidth={4} />
        <rect x={w / 2 - 88} y={-36} width={18} height={72} rx={8} fill={P.steelD} stroke={P.ink} strokeWidth={4} />
      </g>
    ))}
  </g>
);

/* ---------- 举重小人（正面挺举，脚底 y=0） ---------- */
const Lifter = ({x, y, s = 1, mode = 'lift', jersey = P.shout, shout = false}: {
  x: number; y: number; s?: number; mode?: 'lift' | 'strain' | 'fold'; jersey?: string; shout?: boolean;
}) => {
  const f = useCurrentFrame();
  const trem = mode === 'strain' ? Math.sin(f * 0.8) * 2.6 : 0;
  if (mode === 'fold') {
    return (
      <g transform={`translate(${x},${y}) scale(${s})`}>
        <ellipse cx={0} cy={8} rx={120} ry={12} fill={P.ink} opacity={0.32} />
        {/* 下压箭头 */}
        <line x1={0} y1={-236} x2={0} y2={-204} stroke={P.red} strokeWidth={9} strokeLinecap="round" />
        <polygon points="0,-190 -13,-212 13,-212" fill={P.red} />
        {/* 压在低位的杠铃 */}
        <Barbell x={0} y={-172} s={1} w={380} />
        {/* 撑不住的腿 */}
        <path d="M -24 -100 L -88 -44 L -84 -6" fill="none" stroke={P.ink}
          strokeWidth={26} strokeLinecap="round" strokeLinejoin="round" />
        <path d="M 24 -100 L 88 -44 L 84 -6" fill="none" stroke={P.ink}
          strokeWidth={26} strokeLinecap="round" strokeLinejoin="round" />
        <ellipse cx={-88} cy={0} rx={24} ry={10} fill={P.cream} />
        <ellipse cx={88} cy={0} rx={24} ry={10} fill={P.cream} />
        {/* 被压扁的躯干 */}
        <rect x={-48} y={-132} width={96} height={60} rx={24} fill={jersey} />
        {/* 抖着撑杠的手臂 */}
        <path d="M -34 -124 L -70 -152 L -96 -158" fill="none" stroke={C.skin}
          strokeWidth={18} strokeLinecap="round" strokeLinejoin="round" />
        <path d="M 34 -124 L 70 -152 L 96 -158" fill="none" stroke={C.skin}
          strokeWidth={18} strokeLinecap="round" strokeLinejoin="round" />
        {/* 痛苦头 */}
        <g transform="translate(0,-140)">
          <circle r={24} fill={C.skin} />
          <path d="M -24 -5 Q -20 -32 0 -32 Q 20 -32 24 -5 Q 8 -16 0 -16 Q -8 -16 -24 -5 Z" fill={P.ink} />
          <g stroke={P.ink} strokeWidth={4} strokeLinecap="round">
            <path d="M -12 0 L -4 6 M -4 0 L -12 6" fill="none" />
            <path d="M 4 0 L 12 6 M 12 0 L 4 6" fill="none" />
          </g>
        </g>
      </g>
    );
  }
  return (
    <g transform={`translate(${x + trem},${y}) scale(${s})`}>
      <ellipse cx={0} cy={8} rx={132} ry={13} fill={P.ink} opacity={0.32} />
      {/* 腿：宽站距深蹲 */}
      <path d="M -20 -118 L -80 -66 L -88 -6" fill="none" stroke={P.ink}
        strokeWidth={26} strokeLinecap="round" strokeLinejoin="round" />
      <path d="M 20 -118 L 80 -66 L 88 -6" fill="none" stroke={P.ink}
        strokeWidth={26} strokeLinecap="round" strokeLinejoin="round" />
      <ellipse cx={-92} cy={0} rx={26} ry={11} fill={P.cream} />
      <ellipse cx={92} cy={0} rx={26} ry={11} fill={P.cream} />
      {/* 短裤 */}
      <rect x={-48} y={-140} width={96} height={34} rx={15} fill={P.steelD} />
      {/* 躯干 */}
      <rect x={-40} y={-218} width={80} height={86} rx={26} fill={jersey} />
      {/* 举重腰带 */}
      <rect x={-44} y={-146} width={88} height={20} rx={9} fill={P.belt} stroke={P.beltD} strokeWidth={3} />
      {/* 手臂：上举锁死 */}
      <path d="M -32 -204 L -78 -254 L -112 -288" fill="none" stroke={C.skin}
        strokeWidth={19} strokeLinecap="round" strokeLinejoin="round" />
      <path d="M 32 -204 L 78 -254 L 112 -288" fill="none" stroke={C.skin}
        strokeWidth={19} strokeLinecap="round" strokeLinejoin="round" />
      <circle cx={-112} cy={-290} r={13} fill={C.skin} />
      <circle cx={112} cy={-290} r={13} fill={C.skin} />
      {/* 杠铃 */}
      <Barbell x={0} y={-298} s={1} w={430} />
      {/* 头 */}
      <g transform="translate(0,-240)">
        <circle r={30} fill={C.skin} />
        <path d="M -30 -6 Q -26 -38 0 -38 Q 26 -38 30 -6 Q 12 -20 0 -20 Q -12 -20 -30 -6 Z" fill={P.ink} />
        <circle cx={-10} cy={-2} r={3.6} fill={P.ink} />
        <circle cx={11} cy={-2} r={3.6} fill={P.ink} />
        {shout ? (
          <ellipse cx={4} cy={15} rx={7.5} ry={10} fill={P.ink} />
        ) : (
          <path d="M -5 13 Q 3 18 11 12" fill="none" stroke={P.ink} strokeWidth={3.5} strokeLinecap="round" />
        )}
      </g>
      {shout && <ShoutArcs x={46} y={-228} s={0.9} />}
    </g>
  );
};

/* ---------- 指向圆心的压力箭头 ---------- */
const InArrow = ({cx, cy, a, rx1, ry1, rx2, ry2, col = P.amber, sw = 8}: {
  cx: number; cy: number; a: number; rx1: number; ry1: number; rx2: number; ry2: number;
  col?: string; sw?: number;
}) => {
  const rad = (a * Math.PI) / 180;
  const x1 = cx + Math.cos(rad) * rx1, y1 = cy + Math.sin(rad) * ry1;
  const x2 = cx + Math.cos(rad) * rx2, y2 = cy + Math.sin(rad) * ry2;
  return (
    <g>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={col} strokeWidth={sw} strokeLinecap="round" />
      <g transform={`translate(${x2},${y2}) rotate(${a + 180})`}>
        <polygon points="10,0 -6,-9 -6,9" fill={col} />
      </g>
    </g>
  );
};

/* ---------- 躯干正面剖面：肺 + 膈肌 + 高压气囊顶住脊柱 ---------- */
const TorsoPressure = ({x, y, s = 1}: {x: number; y: number; s?: number}) => {
  const f = useCurrentFrame();
  const pulse = 1 + Math.sin(f * 0.14) * 0.035;
  const wob = Math.sin(f * 0.2) * 6;
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      {/* 躯干外壁 */}
      <path d="M -150 -230 Q -176 -60 -142 60 L -126 140 L 126 140 L 142 60 Q 176 -60 150 -230 Q 75 -262 0 -262 Q -75 -262 -150 -230 Z"
        fill={P.card} />
      <rect x={-26} y={-280} width={52} height={26} rx={13} fill={P.card} />
      {/* 脊柱 */}
      <rect x={-13} y={-228} width={26} height={356} rx={12} fill={P.bgDeep} />
      {[0, 1, 2, 3, 4].map((i) => (
        <line key={i} x1={-13} y1={-196 + i * 68} x2={13} y2={-196 + i * 68}
          stroke={P.steel} strokeWidth={5} opacity={0.6} />
      ))}
      {/* 肺 */}
      <ellipse cx={-64} cy={-172} rx={46} ry={54} fill={P.air} opacity={0.85} />
      <ellipse cx={64} cy={-172} rx={46} ry={54} fill={P.air} opacity={0.85} />
      {/* 声门（关闭的闸板） */}
      <g transform="translate(0,-252)">
        <rect x={-22} y={-8} width={44} height={16} rx={8} fill={P.amber} stroke={P.ink} strokeWidth={3} />
        <text x={0} y={40} textAnchor="middle" fontSize={19} fill={P.cream} fontFamily={FONT_HEAD}>声门关</text>
      </g>
      {/* 膈肌穹顶（向上顶住） */}
      <path d="M -112 -62 Q 0 -126 112 -62" stroke={P.airD} strokeWidth={12} fill="none" strokeLinecap="round" />
      {/* 腹腔高压气囊 */}
      <g transform={`translate(0,42) scale(${pulse})`}>
        <ellipse rx={102} ry={76} fill={P.air} opacity={0.3} />
        <ellipse rx={102} ry={76} fill="none" stroke={P.air} strokeWidth={5} strokeDasharray="12 10" opacity={0.9} />
      </g>
      {/* 气囊表面肌肉收缩记号 */}
      {[-1, 1].map((sd) => (
        <g key={sd} stroke={P.cream} strokeWidth={6} strokeLinecap="round" opacity={0.8}>
          <line x1={sd * 118} y1={-6} x2={sd * 134} y2={-6} />
          <line x1={sd * 122} y1={28} x2={sd * 138} y2={28} />
          <line x1={sd * 118} y1={62} x2={sd * 134} y2={62} />
        </g>
      ))}
      {/* 四面向脊柱加压的箭头 */}
      {[15, 75, 105, 165, 195, 255, 285, 345].map((a) => (
        <InArrow key={a} cx={0} cy={40} a={a}
          rx1={150 + wob} ry1={112 + wob * 0.7} rx2={104} ry2={76} col={P.amber} sw={8} />
      ))}
      {/* 脊柱被顶住的高亮 */}
      <rect x={-19} y={-20} width={38} height={130} rx={16} fill="none" stroke={P.amber} strokeWidth={5} opacity={0.85} />
    </g>
  );
};

/* ---------- 阀门（管道 + 阀板，open=0 全闭 / 1 全开） ---------- */
const Valve = ({x, y, s = 1, open = 0.5, jets = false}: {
  x: number; y: number; s?: number; open?: number; jets?: boolean;
}) => {
  const gate = open * 78;
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      <rect x={-150} y={-18} width={300} height={36} rx={18} fill={P.steel} opacity={0.85} />
      <rect x={-150} y={-18} width={300} height={36} rx={18} fill="none" stroke={P.steelD} strokeWidth={4} />
      <circle r={46} fill={P.bgDeep} stroke={P.steel} strokeWidth={6} />
      <rect x={-7} y={-36} width={14} height={72} rx={7} fill={P.air} transform={`rotate(${gate})`} />
      <circle r={8} fill={P.cream} />
      {jets && <Jets x={0} y={-58} s={1} />}
    </g>
  );
};

/* ---------- 腹内压液位条 ---------- */
const PressBar = ({x, y, level, col, h = 150}: {x: number; y: number; level: number; col: string; h?: number}) => (
  <g transform={`translate(${x},${y})`}>
    <rect x={-17} y={-h} width={34} height={h} rx={17} fill={P.bgDeep} stroke={P.card} strokeWidth={3} />
    <rect x={-17} y={-h * level} width={34} height={h * level} rx={17} fill={col} />
    <text x={0} y={30} textAnchor="middle" fontSize={19} fill={P.muted} fontFamily={FONT_HEAD}>腹内压</text>
  </g>
);

/* ---------- 右向箭头 ---------- */
const Chev = ({x, y, col = P.muted}: {x: number; y: number; col?: string}) => (
  <g transform={`translate(${x},${y})`} stroke={col} strokeWidth={7} fill="none"
    strokeLinecap="round" strokeLinejoin="round">
    <path d="M -8 -14 L 8 0 L -8 14" />
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
        {/* 顶部装饰点 */}
        {Array.from({length: 9}).map((_, i) => (
          <circle key={i} cx={140 + i * 205} cy={86 + (i % 2) * 26} r={4} fill={P.cream} opacity={0.3} />
        ))}
        {/* 举重台 + 挺举小人 */}
        <g transform={`translate(960,545) scale(${pA}) translate(-960,-545)`} opacity={clamp01(pA)}>
          <rect x={470} y={560} width={980} height={64} rx={20} fill={P.card} />
          <rect x={470} y={560} width={980} height={64} rx={20} fill="none" stroke={P.steel} strokeWidth={5} />
          <rect x={470} y={560} width={980} height={14} rx={7} fill={P.shout} opacity={0.85} />
          <Lifter x={960} y={556} s={1.15} mode="lift" shout />
          <TagPill x={700} y={330} t="哈！" col={P.shout} w={130} fs={34} s={1.05} />
        </g>
        <g transform={`translate(480,300) scale(${pB})`} opacity={clamp01(pB)}>
          <TagPill x={0} y={0} t="瓦尔萨尔瓦动作" col={P.air} w={260} />
        </g>
        <g transform={`translate(1460,330) scale(${pB})`} opacity={clamp01(pB)}>
          <TagPill x={0} y={0} t="峰值泄压阀" col={P.amber} w={220} />
        </g>
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 470}}>
          <div style={{color: P.air, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 24}}>
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

/* ============ S0 那一喊不是表演：外行看气势 · 内行看泄压 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const popL = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const popR = Math.max(0.001, spring({frame: f - b1 - 6, fps, config: {damping: 11, stiffness: 120}}));

  const steps = [
    {t: '憋气封死', s: '气关在肺里 · 声门关闭', col: P.air},
    {t: '气密桶', s: '腹腔高压气囊 · 顶住脊柱', col: P.amber},
    {t: '泄压阀', s: '喊出来 = 峰值放气', col: P.shout},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={0} name={sec.name} accent={P.shout} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：外行看气势 */}
        <g transform={`translate(90,150) scale(${popL}) translate(-90,-150)`} opacity={clamp01(popL * 2)}>
          <rect x={90} y={150} width={800} height={560} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={126} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            外行看 · 发力瞬间吼一嗓子，纯气势？</text>
          {/* 举重台 */}
          <rect x={130} y={600} width={720} height={16} rx={8} fill={P.card} />
          <Lifter x={450} y={598} s={1.1} mode="lift" shout />
          <TagPill x={700} y={330} t="嘿！" col={P.shout} w={120} fs={32} s={0.95} />
          <TagPill x={220} y={395} t="纯属表演？" col={P.muted} w={200} fs={25} s={0.95} />
          <text x={490} y={680} textAnchor="middle" fontSize={24} fill={P.muted} fontFamily={FONT_HEAD}>
            短促 · 尖锐 · 像要掀翻训练馆的房顶</text>
        </g>
        {/* 右：内行看力学设计（cue1） */}
        {popR > 0.02 && (
          <g transform={`translate(930,150) scale(${popR}) translate(-930,-150)`} opacity={clamp01(popR * 2)}>
            <rect x={930} y={150} width={900} height={560} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
            <text x={966} y={206} fill={P.cream} fontSize={26} fontFamily={FONT_HEAD} fontWeight={900}>
              内行看 · 一整套力学设计</text>
            {steps.map((st, i) => {
              const bx = 986 + i * 270;
              const at = b1 + 10 + i * 12;
              const sp = Math.max(0.001, spring({frame: f - at, fps, config: {damping: 11, stiffness: 130}}));
              return (
                <g key={i} transform={`translate(${bx},290) scale(${sp}) translate(${-bx},-290)`}
                  opacity={clamp01(sp * 2)}>
                  <rect x={bx} y={290} width={240} height={300} rx={18} fill={P.bgDeep}
                    stroke={st.col} strokeWidth={3.5} />
                  <circle cx={bx + 40} cy={332} r={20} fill={st.col} />
                  <text x={bx + 40} y={340} textAnchor="middle" fontSize={22} fill={P.ink}
                    fontFamily={FONT_HEAD} fontWeight={900}>{i + 1}</text>
                  <text x={bx + 70} y={340} fontSize={26} fill={st.col}
                    fontFamily={FONT_HEAD} fontWeight={900}>{st.t}</text>
                  {/* 图标 */}
                  <g transform={`translate(${bx + 120},440)`}>
                    {i === 0 && (
                      <g>
                        <ellipse cx={-32} cy={-6} rx={30} ry={40} fill={P.air} opacity={0.85} />
                        <ellipse cx={32} cy={-6} rx={30} ry={40} fill={P.air} opacity={0.85} />
                        <line x1={0} y1={-58} x2={0} y2={-24} stroke={P.cream} strokeWidth={8}
                          strokeLinecap="round" />
                        <rect x={-20} y={-76} width={40} height={18} rx={9} fill={P.amber}
                          stroke={P.ink} strokeWidth={3} />
                      </g>
                    )}
                    {i === 1 && (
                      <g>
                        <rect x={-58} y={-64} width={116} height={130} rx={22} fill={P.card}
                          stroke={P.air} strokeWidth={4} />
                        <rect x={-8} y={-50} width={16} height={102} rx={8} fill={P.bgDeep} />
                        <ellipse rx={38} ry={26} fill={P.air} opacity={0.35} />
                        <InArrow cx={0} cy={0} a={0} rx1={82} ry1={0.1} rx2={48} ry2={0.1} col={P.amber} sw={7} />
                        <InArrow cx={0} cy={0} a={180} rx1={82} ry1={0.1} rx2={48} ry2={0.1} col={P.amber} sw={7} />
                      </g>
                    )}
                    {i === 2 && (
                      <g>
                        <Valve x={0} y={0} s={0.62} open={0.55} jets />
                      </g>
                    )}
                  </g>
                  <text x={bx + 120} y={552} textAnchor="middle" fontSize={20} fill={P.muted}
                    fontFamily={FONT_HEAD}>{st.s}</text>
                  {i < 2 && <Chev x={bx + 255} y={440} />}
                </g>
              );
            })}
            <text x={1380} y={668} textAnchor="middle" fontSize={26} fill={P.cream}
              fontFamily={FONT_HEAD} fontWeight={800}>
              喊之前封死加压 · 喊出来是精确计算的泄压</text>
          </g>
        )}
      </svg>
      <Chip text="那一嗓子 = 力学程序" x={1150} y={72} at={beat(0) + 25} accent={P.shout} />
      <Stamp text="不是情绪发泄" x={620} y={86} at={b1 + 70} color={P.shoutD} size={38} rot={-3} />
    </FadeIn>
  );
};

/* ============ S1 憋气那一下：躯干变高压气囊 + 腰带外箍 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const pop = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const beltK = clamp01((f - b1 - 8) / 26);
  const strap = beltK * (1 - Math.exp(-(f - b1) * 0.04));

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={1} name={sec.name} accent={P.air} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：躯干剖面 */}
        <g transform={`translate(90,150) scale(${pop}) translate(-90,-150)`} opacity={clamp01(pop * 2)}>
          <rect x={90} y={150} width={800} height={560} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={126} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            把躯干变成刚性气密桶 · 保住脊柱</text>
          <TorsoPressure x={400} y={500} s={1.0} />
          <TagPill x={720} y={300} t="声门一关 · 气憋在肺" col={P.air} w={280} fs={23} s={0.92} />
          <TagPill x={720} y={390} t="腹内压顶住脊柱" col={P.amber} w={240} fs={23} s={0.92} />
          <TagPill x={720} y={480} t="刚性立刻上来" col={P.lime} w={220} fs={23} s={0.92} />
        </g>
        {/* 右：瓦尔萨尔瓦 + 腰带外箍（cue1） */}
        {beltK > 0 && (
          <g opacity={beltK} transform={`translate(0,${(1 - beltK) * 40})`}>
            <rect x={930} y={150} width={900} height={560} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
            {/* 大名卡 */}
            <rect x={966} y={186} width={828} height={108} rx={18} fill={P.bgDeep} stroke={P.air} strokeWidth={4} />
            <text x={1380} y={238} textAnchor="middle" fontSize={44} fill={P.air}
              fontFamily={FONT_HEAD} fontWeight={900}>瓦尔萨尔瓦动作</text>
            <text x={1380} y={276} textAnchor="middle" fontSize={23} fill={P.muted}
              fontFamily={FONT_HEAD} letterSpacing={4}>Valsalva Maneuver · 体液+气体当内部支撑</text>
            {/* 腰带箍住腹腔 */}
            <g transform="translate(1395,480)">
              {/* 简化躯干 */}
              <rect x={-130} y={-140} width={260} height={240} rx={60} fill={P.card} />
              <rect x={-9} y={-120} width={18} height={200} rx={9} fill={P.bgDeep} />
              <ellipse cy={30} rx={88} ry={60} fill={P.air} opacity={0.3} />
              {/* 腰带（收紧动画：宽度随 strap 收拢） */}
              <g transform={`scale(${1 + (1 - strap) * 0.12},1)`}>
                <rect x={-158} y={-6} width={316} height={64} rx={18} fill={P.belt}
                  stroke={P.beltD} strokeWidth={5} />
                <rect x={-34} y={6} width={68} height={40} rx={10} fill={P.amber}
                  stroke={P.amberD} strokeWidth={4} />
              </g>
              {/* 两侧约束箭头 */}
              <InArrow cx={0} cy={26} a={0} rx1={216} ry1={0.1} rx2={172} ry2={0.1} col={P.belt} sw={10} />
              <InArrow cx={0} cy={26} a={180} rx1={216} ry1={0.1} rx2={172} ry2={0.1} col={P.belt} sw={10} />
              <text x={0} y={164} textAnchor="middle" fontSize={24} fill={P.cream}
                fontFamily={FONT_HEAD} fontWeight={800}>硬腰带 = 给气囊加一圈外箍</text>
            </g>
            <TagPill x={1660} y={380} t="高压锅同款" col={P.amber} w={210} fs={23} s={0.9} />
            <text x={1395} y={680} textAnchor="middle" fontSize={23} fill={P.muted} fontFamily={FONT_HEAD}>
              不让腹部往外鼓 · 压力全部导向脊柱</text>
          </g>
        )}
      </svg>
      <Chip text="腹肌+膈肌+盆底肌齐收缩" x={1090} y={72} at={beat(0) + 25} accent={P.air} />
      <Stamp text="内置气压支撑柱" x={620} y={86} at={b1 + 40} color={P.airD} size={38} rot={-3} />
    </FadeIn>
  );
};

/* ============ S2 憋太久的代价：血压爆表 + 峰值开阀时序 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const pop = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const chartK = clamp01((f - b1 - 8) / 26);
  const needleK = clamp01((f - 24) / 50);
  const v = 120 + (265 - 120) * needleK + Math.sin(f * 0.55) * 12 * needleK;
  const nAng = v * 0.6 - 108;

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={2} name={sec.name} accent={P.red} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：血压表盘 */}
        <g transform={`translate(90,150) scale(${pop}) translate(-90,-150)`} opacity={clamp01(pop * 2)}>
          <rect x={90} y={150} width={830} height={560} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={126} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            憋太久的代价 · 胸内压把血压一起顶上去</text>
          {/* 半圆表盘 */}
          <g transform="translate(370,590)">
            <path d="M -185 0 A 185 185 0 0 1 185 0" fill="none" stroke={P.cream}
              strokeWidth={26} opacity={0.22} strokeLinecap="round" />
            {/* 红区：240→300 */}
            <path d="M 108.7 -149.7 A 185 185 0 0 1 176 -57.2" fill="none" stroke={P.red}
              strokeWidth={26} strokeLinecap="round" />
            {/* 刻度 */}
            {[120, 180, 240, 300].map((vv) => {
              const ang = (198 - vv * 0.6) * Math.PI / 180;
              const tx = Math.cos(ang) * 152, ty = -Math.sin(ang) * 152;
              const lx = Math.cos(ang) * 218, ly = -Math.sin(ang) * 218;
              return (
                <g key={vv}>
                  <line x1={tx * 0.88} y1={ty * 0.88} x2={tx} y2={ty} stroke={P.cream}
                    strokeWidth={6} strokeLinecap="round" opacity={0.7} />
                  <text x={lx} y={ly + 8} textAnchor="middle" fontSize={23} fill={P.cream}
                    fontFamily={FONT_HEAD}>{vv}</text>
                </g>
              );
            })}
            {/* 指针 */}
            <g transform={`rotate(${nAng})`}>
              <line x1={0} y1={20} x2={0} y2={-158} stroke={P.shout} strokeWidth={10}
                strokeLinecap="round" />
            </g>
            <circle r={18} fill={P.cream} stroke={P.ink} strokeWidth={4} />
            <text x={0} y={70} textAnchor="middle" fontSize={24} fill={P.muted}
              fontFamily={FONT_HEAD}>收缩压 mmHg</text>
          </g>
          <TagPill x={370} y={672} t="极限试举冲到 200–300" col={P.red} w={360} fs={24} s={0.95} />
          {/* 右侧三个连锁反应 */}
          <g transform="translate(645,270)">
            {/* 大静脉受压 */}
            <rect x={-16} y={-56} width={32} height={42} rx={14} fill={P.red} opacity={0.8} />
            <rect x={-8} y={-14} width={16} height={28} rx={7} fill={P.red} opacity={0.8} />
            <rect x={-16} y={14} width={32} height={42} rx={14} fill={P.red} opacity={0.8} />
            <line x1={-52} y1={0} x2={-24} y2={0} stroke={P.amber} strokeWidth={7} strokeLinecap="round" />
            <line x1={52} y1={0} x2={24} y2={0} stroke={P.amber} strokeWidth={7} strokeLinecap="round" />
            <text x={78} y={8} fontSize={24} fill={P.cream} fontFamily={FONT_HEAD} fontWeight={800}>
              大静脉被压住</text>
          </g>
          <g transform="translate(645,390)">
            <path d="M 0 -14 Q -30 -40 -54 -14 Q -76 8 -48 38 L 0 66 L 48 38 Q 76 8 54 -14 Q 30 -40 0 -14 Z"
              fill={P.red} opacity={0.9} transform="scale(0.72)" />
            <line x1={10} y1={52} x2={10} y2={30} stroke={P.cream} strokeWidth={6} strokeLinecap="round" />
            <polygon points="10,20 2,36 18,36" fill={P.cream} transform="rotate(180 10 28)" />
            <text x={78} y={8} fontSize={24} fill={P.cream} fontFamily={FONT_HEAD} fontWeight={800}>
              回心血量掉下去</text>
          </g>
          <g transform="translate(645,510)">
            <circle r={30} fill={C.skin} />
            <path d="M -16 -6 q 8 -10 16 0 q -8 10 -16 0" fill="none" stroke={P.ink} strokeWidth={4} />
            <path d="M 2 -6 q 8 -10 16 0 q -8 10 -16 0" fill="none" stroke={P.ink} strokeWidth={4} transform="translate(-6,0)" />
            {[0, 1, 2].map((i) => (
              <circle key={i} cx={-34 + i * 30} cy={-44} r={4} fill={P.amber}
                transform={`rotate(${i * 40} 0 0)`} />
            ))}
            <text x={78} y={8} fontSize={24} fill={P.cream} fontFamily={FONT_HEAD} fontWeight={800}>
              头晕 · 眼前发黑</text>
          </g>
        </g>
        {/* 右：峰值开阀时序（cue1） */}
        {chartK > 0 && (
          <g opacity={chartK} transform={`translate(0,${(1 - chartK) * 40})`}>
            <rect x={960} y={150} width={870} height={560} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
            <text x={996} y={206} fill={P.cream} fontSize={26} fontFamily={FONT_HEAD} fontWeight={900}>
              时序 = 精密气压控制系统</text>
            {/* 坐标轴 */}
            <line x1={1020} y1={560} x2={1780} y2={560} stroke={P.muted} strokeWidth={5}
              strokeLinecap="round" opacity={0.6} />
            <line x1={1020} y1={560} x2={1020} y2={280} stroke={P.muted} strokeWidth={5}
              strokeLinecap="round" opacity={0.6} />
            <text x={1002} y={292} textAnchor="end" fontSize={20} fill={P.muted}
              fontFamily={FONT_HEAD}>腹内压</text>
            {/* 腹内压曲线：憋气升压→平台→喊=放气→完成后半程 */}
            <path d="M 1040 540 L 1140 540 L 1260 340 L 1440 340 L 1500 440 L 1760 430"
              fill="none" stroke={P.air} strokeWidth={9} strokeLinecap="round" strokeLinejoin="round" />
            {/* 如果继续憋：血压顶穿的虚线 */}
            <path d="M 1440 340 Q 1600 330 1690 250" fill="none" stroke={P.red}
              strokeWidth={7} strokeDasharray="12 10" strokeLinecap="round" />
            <text x={1690} y={226} textAnchor="middle" fontSize={22} fill={P.red}
              fontFamily={FONT_HEAD} fontWeight={800}>继续憋 → 血压顶穿</text>
            {/* 喊点 */}
            <circle cx={1440} cy={340} r={12} fill={P.shout} stroke={P.cream} strokeWidth={4} />
            <ShoutArcs x={1472} y={330} s={0.55} col={P.shout} rot={-30} />
            <TagPill x={1545} y={300} t="喊！开阀放气" col={P.shout} w={220} fs={24} s={0.9} />
            {/* 阶段标注 */}
            <text x={1120} y={604} textAnchor="middle" fontSize={23} fill={P.air}
              fontFamily={FONT_HEAD} fontWeight={800}>封死加压</text>
            <text x={1440} y={604} textAnchor="middle" fontSize={23} fill={P.amber}
              fontFamily={FONT_HEAD} fontWeight={800}>峰值</text>
            <text x={1660} y={604} textAnchor="middle" fontSize={23} fill={P.air}
              fontFamily={FONT_HEAD} fontWeight={800}>刚性完成动作</text>
            <text x={1395} y={672} textAnchor="middle" fontSize={24} fill={P.cream}
              fontFamily={FONT_HEAD} fontWeight={800}>
              在「压力支撑」和「血压爆表」之间走钢丝</text>
          </g>
        )}
      </svg>
      <Chip text="收缩压冲到两三百" x={1150} y={72} at={beat(0) + 25} accent={P.red} />
      <Stamp text="峰值一到就开阀" x={620} y={86} at={b1 + 40} color={P.red} size={38} rot={-3} />
    </FadeIn>
  );
};

/* ============ S3 半开阀门：全闭/半开/全开对照 + 力量小加成 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const b2 = beat(2);
  const b3 = beat(3);
  const qK = clamp01((f - 8) / 24);
  const popL = Math.max(0.001, spring({frame: f - b1 - 6, fps, config: {damping: 11, stiffness: 120}}));
  const popR = Math.max(0.001, spring({frame: f - b1 - 16, fps, config: {damping: 11, stiffness: 120}}));
  const popC = Math.max(0.001, spring({frame: f - b2 - 6, fps, config: {damping: 10, stiffness: 140}}));
  const gainK = clamp01((f - b3 - 4) / 26);

  const cards = [
    {cx: 360, pop: popL, title: '全闭 · 憋到底', col: P.red, open: 0, jets: false, level: 0.98,
      verdict: '压力拉满 · 血压也拉满', icon: 'warn'},
    {cx: 960, pop: popC, title: '半开 · 喊出来', col: P.air, open: 0.55, jets: true, level: 0.68,
      verdict: '泄了压 · 还保住核心', icon: 'check'},
    {cx: 1560, pop: popR, title: '全开 · 张嘴呼气', col: P.muted, open: 1, jets: true, level: 0.08,
      verdict: '压力掉底 · 躯干塌', icon: 'flat'},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={3} name={sec.name} accent={P.air} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 顶部提问（cue0） */}
        {qK > 0 && (
          <g opacity={qK} transform={`translate(0,${(1 - qK) * 20})`}>
            <rect x={90} y={116} width={470} height={60} rx={30} fill={P.bgDeep} stroke={P.amber} strokeWidth={4} />
            <text x={325} y={157} textAnchor="middle" fontSize={29} fill={P.amber}
              fontFamily={FONT_HEAD} fontWeight={900}>为什么不直接张嘴呼气？</text>
          </g>
        )}
        {/* 三态阀门对照 */}
        {cards.map((cd, i) => {
          const cw = 540;
          return (
            <g key={i} transform={`translate(${cd.cx},190) scale(${cd.pop}) translate(${-cd.cx},-190)`}
              opacity={clamp01(cd.pop * 2)}>
              <rect x={cd.cx - cw / 2} y={190} width={cw} height={400} rx={20} fill={P.panel}
                stroke={cd.col} strokeWidth={i === 1 ? 5 : 4} />
              <text x={cd.cx} y={244} textAnchor="middle" fontSize={30} fill={cd.col}
                fontFamily={FONT_HEAD} fontWeight={900}>{cd.title}</text>
              {/* 阀门 */}
              <Valve x={cd.cx - 40} y={360} s={0.85} open={cd.open} jets={cd.jets} />
              {/* 液位条 */}
              <PressBar x={cd.cx + 180} y={430} level={cd.level} col={cd.col} h={150} />
              {/* 底部判定 */}
              {cd.icon === 'warn' && (
                <g>
                  <TagPill x={cd.cx} y={512} t="血压飙升 ⚠" col={P.red} w={240} fs={24} s={0.95} />
                </g>
              )}
              {cd.icon === 'check' && (
                <g>
                  <TagPill x={cd.cx} y={512} t="泄压 + 保刚性 ✓" col={P.air} w={280} fs={24} s={0.95} />
                </g>
              )}
              {cd.icon === 'flat' && (
                <g>
                  {/* 被压扁的躯干 */}
                  <ellipse cx={cd.cx - 110} cy={505} rx={62} ry={26} fill={P.card}
                    stroke={P.steelD} strokeWidth={4} />
                  <line x1={cd.cx - 110} y1={430} x2={cd.cx - 110} y2={462} stroke={P.red}
                    strokeWidth={8} strokeLinecap="round" />
                  <polygon points={`${cd.cx - 110},476 ${cd.cx - 122},456 ${cd.cx - 98},456`} fill={P.red} />
                  <TagPill x={cd.cx + 60} y={512} t="核心塌了 ✗" col={P.red} w={220} fs={24} s={0.95} />
                </g>
              )}
            </g>
          );
        })}
        {/* 底部：力量小加成（cue3） */}
        {gainK > 0 && (
          <g opacity={gainK} transform={`translate(0,${(1 - gainK) * 50})`}>
            <rect x={90} y={640} width={1740} height={190} rx={20} fill={P.panel}
              stroke={P.amber} strokeWidth={4} />
            <ShoutArcs x={180} y={700} s={0.7} col={P.shout} rot={-20} />
            <text x={330} y={745} textAnchor="middle" fontSize={72} fill={P.amber}
              fontFamily={FONT_HEAD} fontWeight={900}>+3~5%</text>
            <text x={330} y={795} textAnchor="middle" fontSize={24} fill={P.muted}
              fontFamily={FONT_HEAD}>实测发力输出加成</text>
            <line x1={560} y1={680} x2={560} y2={790} stroke={P.card} strokeWidth={4} />
            <text x={1180} y={722} textAnchor="middle" fontSize={32} fill={P.cream}
              fontFamily={FONT_HEAD} fontWeight={900}>
              小加成 · 不是大力丸 · 也不全是心理作用</text>
            <text x={1180} y={772} textAnchor="middle" fontSize={24} fill={P.muted}
              fontFamily={FONT_HEAD}>
              神经兴奋？注意力聚焦？—— 哪条占主导还没定论</text>
          </g>
        )}
      </svg>
      <Chip text="半开阀门 = 折中正解" x={1150} y={72} at={b1 + 20} accent={P.air} />
      <Stamp text="泄压不塌腰" x={620} y={86} at={b2 + 40} color={P.airD} size={38} rot={-3} />
    </FadeIn>
  );
};

/* ============ S4 喊法分项目：爆发专利 + 耐力别喊 + 量力而憋 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const b2 = beat(2);
  const warnK = clamp01((f - b2 - 4) / 26);
  const pop4 = Math.max(0.001, spring({frame: f - b1 - 6, fps, config: {damping: 11, stiffness: 130}}));

  const cards = [
    {t: '举重 · 铅球', col: P.shout, l1: '喊得短促尖锐', l2: '贴着发力节拍'},
    {t: '网球', col: P.lime, l1: '盖住击球音色', l2: '干扰对手判断'},
    {t: '武术', col: P.amber, l1: '一声「喝」', l2: '协同 + 心理威慑'},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={4} name={sec.name} accent={P.amber} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 上排三张爆发项目卡（cue0） */}
        {cards.map((cd, i) => {
          const cx = 290 + i * 450;
          const at = 8 + i * 14;
          const sp = Math.max(0.001, spring({frame: f - at, fps, config: {damping: 11, stiffness: 130}}));
          return (
            <g key={i} transform={`translate(${cx},170) scale(${sp}) translate(${-cx},-170)`}
              opacity={clamp01(sp * 2)}>
              <rect x={cx - 200} y={170} width={400} height={330} rx={20} fill={P.panel}
                stroke={cd.col} strokeWidth={4} />
              <text x={cx} y={228} textAnchor="middle" fontSize={30} fill={cd.col}
                fontFamily={FONT_HEAD} fontWeight={900}>{cd.t}</text>
              {/* 图标区 */}
              <g transform={`translate(${cx},330)`}>
                {i === 0 && (
                  <g>
                    <Barbell x={-20} y={-30} s={0.52} w={360} />
                    <Lifter x={-20} y={76} s={0.42} mode="lift" shout={false} />
                    <ShoutArcs x={56} y={-24} s={0.5} col={cd.col} rot={-20} />
                  </g>
                )}
                {i === 1 && (
                  <g>
                    <g transform="rotate(-28)">
                      <ellipse cx={-30} cy={-10} rx={38} ry={54} fill="none"
                        stroke={P.steel} strokeWidth={8} />
                      <line x1={-30} y1={44} x2={-30} y2={92} stroke={P.steel}
                        strokeWidth={9} strokeLinecap="round" />
                    </g>
                    <circle cx={46} cy={-30} r={13} fill={P.lime} />
                    <ShoutArcs x={30} y={-52} s={0.5} col={cd.col} rot={-10} />
                    <line x1={-70} y1={60} x2={70} y2={60} stroke={P.card} strokeWidth={6}
                      strokeLinecap="round" />
                  </g>
                )}
                {i === 2 && (
                  <g>
                    <rect x={-44} y={-26} width={72} height={56} rx={20} fill={C.skin}
                      stroke={P.ink} strokeWidth={4} />
                    {[0, 1, 2].map((j) => (
                      <line key={j} x1={-28 + j * 22} y1={-26} x2={-28 + j * 22} y2={-40}
                        stroke={P.ink} strokeWidth={5} strokeLinecap="round" />
                    ))}
                    {/* 「喝」星芒 */}
                    <g transform={`translate(52,-30) scale(${1 + Math.sin(f * 0.3) * 0.08})`}>
                      {Array.from({length: 8}).map((_, j) => (
                        <line key={j} x1={0} y1={-14} x2={0} y2={-30} stroke={cd.col}
                          strokeWidth={7} strokeLinecap="round" transform={`rotate(${j * 45})`} />
                      ))}
                      <circle r={13} fill={P.amber} stroke={P.ink} strokeWidth={3} />
                    </g>
                    <text x={56} y={62} textAnchor="middle" fontSize={30} fill={cd.col}
                      fontFamily={FONT_HEAD} fontWeight={900}>喝！</text>
                  </g>
                )}
              </g>
              <text x={cx} y={442} textAnchor="middle" fontSize={23} fill={P.cream}
                fontFamily={FONT_HEAD} fontWeight={800}>{cd.l1}</text>
              <text x={cx} y={478} textAnchor="middle" fontSize={21} fill={P.muted}
                fontFamily={FONT_HEAD}>{cd.l2}</text>
            </g>
          );
        })}
        {/* 第四张：耐力项目（cue1） */}
        {pop4 > 0.02 && (
          <g transform={`translate(1640,170) scale(${pop4}) translate(${-1640},-170)`}
            opacity={clamp01(pop4 * 2)}>
            <rect x={1440} y={170} width={400} height={330} rx={20} fill={P.panel}
              stroke={P.muted} strokeWidth={4} />
            <text x={1640} y={228} textAnchor="middle" fontSize={30} fill={P.muted}
              fontFamily={FONT_HEAD} fontWeight={900}>马拉松 · 游泳</text>
            <g transform="translate(1640,330)">
              {/* 泳者 */}
              <circle cx={-40} cy={-30} r={16} fill={C.skin} />
              <path d="M -64 -12 L -10 -2 L 30 -20" fill="none" stroke={C.skin}
                strokeWidth={14} strokeLinecap="round" strokeLinejoin="round" />
              {[0, 1, 2].map((j) => (
                <path key={j} d={`M ${-90 + j * 4} ${22 + j * 22} Q ${-50 + j * 4} ${8 + j * 22} ${-10 + j * 4} ${22 + j * 22} Q ${30 + j * 4} ${36 + j * 22} ${70 + j * 4} ${22 + j * 22}`}
                  fill="none" stroke={P.sky} strokeWidth={7} strokeLinecap="round" opacity={0.9 - j * 0.25} />
              ))}
              {/* 喊声气泡打叉 */}
              <g transform="translate(52,-52)">
                <ellipse rx={36} ry={26} fill={P.bgDeep} stroke={P.muted} strokeWidth={4} />
                <text y={9} textAnchor="middle" fontSize={22} fill={P.muted}
                  fontFamily={FONT_HEAD}>喊</text>
                <MarkX x={0} y={0} s={1.5} col={P.red} />
              </g>
            </g>
            <text x={1640} y={442} textAnchor="middle" fontSize={23} fill={P.cream}
              fontFamily={FONT_HEAD} fontWeight={800}>瓶颈是供氧 · 不是爆发</text>
            <text x={1640} y={478} textAnchor="middle" fontSize={21} fill={P.muted}
              fontFamily={FONT_HEAD}>憋气 = 给自己添堵</text>
          </g>
        )}
        {/* 底部：量力而憋（cue2） */}
        {warnK > 0 && (
          <g opacity={warnK} transform={`translate(0,${(1 - warnK) * 50})`}>
            <rect x={90} y={540} width={1740} height={280} rx={20} fill={P.panel}
              stroke={P.red} strokeWidth={4} />
            {/* 左：警告 */}
            <g transform="translate(150,610)">
              <path d="M 0 -34 L 38 26 L -38 26 Z" fill={P.bgDeep} stroke={P.red}
                strokeWidth={6} strokeLinejoin="round" />
              <line x1={0} y1={-16} x2={0} y2={6} stroke={P.red} strokeWidth={7}
                strokeLinecap="round" />
              <circle cy={17} r={4.5} fill={P.red} />
            </g>
            <text x={210} y={624} fontSize={30} fill={P.red} fontFamily={FONT_HEAD} fontWeight={900}>
              高血压 · 心血管 → 别硬学憋气</text>
            <text x={150} y={676} fontSize={23} fill={P.muted} fontFamily={FONT_HEAD}>
              瓦尔萨尔瓦对他们是风险不是技巧 · 任何憋气前先评估健康</text>
            <rect x={150} y={706} width={560} height={56} rx={14} fill={P.bgDeep}
              stroke={P.air} strokeWidth={3.5} />
            <text x={430} y={743} textAnchor="middle" fontSize={25} fill={P.air}
              fontFamily={FONT_HEAD} fontWeight={800}>轻重量 → 正常呼吸就够</text>
            <line x1={960} y1={580} x2={960} y2={780} stroke={P.card} strokeWidth={4} />
            {/* 右：本质 */}
            <text x={1400} y={640} textAnchor="middle" fontSize={34} fill={P.amber}
              fontFamily={FONT_HEAD} fontWeight={900}>那声喊是泄压 · 不是音量比赛</text>
            <text x={1400} y={700} textAnchor="middle" fontSize={24} fill={P.muted}
              fontFamily={FONT_HEAD}>高手喊得短促精准 · 分寸是多年练出来的</text>
            <g transform="translate(1400,762)">
              <TagPill x={-190} y={0} t="该封死时封死" col={P.air} w={240} fs={23} s={0.9} />
              <TagPill x={190} y={0} t="该放掉时放掉" col={P.shout} w={240} fs={23} s={0.9} />
            </g>
          </g>
        )}
      </svg>
      <Chip text="喊声是爆发项目的专利" x={1120} y={72} at={beat(0) + 30} accent={P.amber} />
      <Stamp text="泄压 ≠ 音量比赛" x={620} y={86} at={b2 + 40} color={P.shoutD} size={38} rot={-3} />
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
        <g transform={`translate(960,770) scale(${pop})`} opacity={clamp01(pop)}>
          <rect x={-560} y={0} width={1120} height={64} rx={20} fill={P.card} />
          <rect x={-560} y={0} width={1120} height={14} rx={7} fill={P.shout} opacity={0.85} />
          <Lifter x={-140} y={-2} s={0.95} mode="lift" jersey={P.air} shout={false} />
          <g transform="translate(200,-190)">
            <rect x={-190} y={-50} width={380} height={100} rx={18} fill={P.panel}
              stroke={P.air} strokeWidth={4} />
            <text x={0} y={-12} textAnchor="middle" fontSize={26} fill={P.air}
              fontFamily={FONT_HEAD} fontWeight={900}>泄压 · 不是失态</text>
            <text x={0} y={28} textAnchor="middle" fontSize={21} fill={P.muted}
              fontFamily={FONT_HEAD}>是身体在执行力学程序</text>
          </g>
        </g>
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -330}}>
        <div style={{fontSize: 52, fontWeight: 800, color: C.text}}>该封死时封死 · 该放掉时放掉</div>
        <div style={{fontSize: 36, color: P.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E245 = ({data = tl}: {data?: any}) => {
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
