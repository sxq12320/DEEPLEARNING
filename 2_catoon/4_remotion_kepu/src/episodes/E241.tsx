import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01} from '../lib/style';
import {Backdrop, Stamp, SceneTag, Chip, Subtitles, ProgressBar, FadeIn} from '../lib/atoms';
import tl from '../timeline_241.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：赛道藏蓝 × 跑道橙红 × 草地青柠 ============ */
const P = {
  bg: '#1a3a6e',
  bgLite: '#22468a',
  bgDeep: '#102550',
  panel: '#16305e',
  card: '#2c4f92',
  ink: '#0d1e40',
  cream: '#fff2dc',
  muted: '#a9b8e2',
  track: '#ff6a4d',
  trackD: '#d8472e',
  lane: '#ffd9cf',
  lime: '#8fe06a',
  limeD: '#54b045',
  amber: '#ffd166',
  amberD: '#e0a83c',
  sky: '#6fd2ff',
  pink: '#ff9ec4',
  organ: '#ff9d5c',
  organD: '#e0763c',
  dia: '#7ef0d4',
  diaD: '#46c9ab',
};

/* ---------- 小叉 / 小勾 ---------- */
const MarkX = ({x, y, s = 1, col = P.track}: {x: number; y: number; s?: number; col?: string}) => (
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

/* ---------- 岔气疼痛星芒 ---------- */
const PainBurst = ({x, y, s = 1}: {x: number; y: number; s?: number}) => {
  const f = useCurrentFrame();
  const puls = 1 + Math.sin(f * 0.35) * 0.1;
  return (
    <g transform={`translate(${x},${y}) scale(${s * puls})`}>
      {Array.from({length: 8}).map((_, i) => (
        <line key={i} x1={0} y1={-18} x2={0} y2={-40} stroke={P.track} strokeWidth={9}
          strokeLinecap="round" transform={`rotate(${i * 45})`} />
      ))}
      <circle r={15} fill={P.amber} stroke={P.ink} strokeWidth={4} />
    </g>
  );
};

/* ---------- 闪电 ---------- */
const Bolt = ({x, y, s = 1, col = P.amber, rot = 0}: {x: number; y: number; s?: number; col?: string; rot?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`}>
    <path d="M 6 -40 L 26 -40 L 8 -10 L 28 -10 L -8 40 L 0 4 L -20 4 Z"
      fill={col} stroke={P.ink} strokeWidth={3.5} strokeLinejoin="round" />
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

/* ---------- 韧带弹簧 ---------- */
const Lig = ({x, y1, y2, col = P.dia, w = 12}: {x: number; y1: number; y2: number; col?: string; w?: number}) => {
  const n = 4;
  const seg = (y2 - y1) / n;
  let d = `M ${x} ${y1}`;
  for (let i = 0; i < n; i++) {
    d += ` L ${x + (i % 2 ? -w : w)} ${y1 + seg * (i + 0.5)} L ${x} ${y1 + seg * (i + 1)}`;
  }
  return <path d={d} fill="none" stroke={col} strokeWidth={7} strokeLinecap="round" strokeLinejoin="round" />;
};

/* ---------- 跑步小人（面向右，脚底 y≈0） ---------- */
const Runner = ({x, y, s = 1, mode = 'run', jersey = P.track}: {
  x: number; y: number; s?: number; mode?: 'run' | 'hurt' | 'easy'; jersey?: string;
}) => {
  const f = useCurrentFrame();
  const hurt = mode === 'hurt';
  const run = mode === 'run';
  const bob = hurt ? 0 : Math.abs(Math.sin(f * (run ? 0.3 : 0.12))) * (run ? 9 : 4);
  const lean = hurt ? 34 : run ? 15 : 8;
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      <ellipse cx={4} cy={8} rx={102} ry={13} fill={P.ink} opacity={0.32} />
      <g transform={`translate(0,${-bob})`}>
        {/* 后腿 */}
        <path d={hurt ? 'M -8 -124 L -34 -66 L -32 -4' : run ? 'M -8 -124 L -54 -72 L -84 -10' : 'M -8 -124 L -30 -64 L -28 -4'}
          fill="none" stroke={P.ink} strokeWidth={26} strokeLinecap="round" strokeLinejoin="round" />
        {/* 前腿 */}
        <path d={hurt ? 'M 10 -122 L 40 -66 L 64 -10' : run ? 'M 10 -122 L 58 -78 L 98 -46' : 'M 10 -122 L 36 -66 L 58 -8'}
          fill="none" stroke={P.ink} strokeWidth={26} strokeLinecap="round" strokeLinejoin="round" />
        {/* 鞋 */}
        <ellipse cx={hurt ? -38 : run ? -92 : -34} cy={hurt ? 2 : run ? -4 : 2} rx={24} ry={11} fill={P.cream} />
        <ellipse cx={hurt ? 70 : run ? 104 : 64} cy={hurt ? -4 : run ? -42 : -2} rx={24} ry={11} fill={P.cream}
          transform={run ? 'rotate(-20 104 -42)' : ''} />
        {/* 躯干 + 短裤 */}
        <g transform={`rotate(${lean} 0 -160)`}>
          <rect x={-34} y={-224} width={74} height={106} rx={30} fill={jersey} />
          <rect x={-38} y={-140} width={82} height={34} rx={16} fill={P.sky} />
        </g>
        {/* 手臂 */}
        {hurt ? (
          <g>
            {/* 前手捂住右肋 */}
            <path d="M 10 -192 L 58 -168 L 52 -130" fill="none" stroke={C.skin}
              strokeWidth={20} strokeLinecap="round" strokeLinejoin="round" />
            <circle cx={52} cy={-126} r={14} fill={C.skin} />
            <path d="M -6 -190 L -46 -158 L -42 -124" fill="none" stroke={C.skinDark}
              strokeWidth={20} strokeLinecap="round" strokeLinejoin="round" />
          </g>
        ) : (
          <g>
            <path d="M 12 -196 L 58 -164 L 32 -124" fill="none" stroke={C.skin}
              strokeWidth={20} strokeLinecap="round" strokeLinejoin="round" />
            <circle cx={32} cy={-120} r={13} fill={C.skin} />
            <path d="M -4 -194 L -48 -162 L -72 -196" fill="none" stroke={C.skinDark}
              strokeWidth={20} strokeLinecap="round" strokeLinejoin="round" />
          </g>
        )}
        {/* 头 */}
        <g transform={`translate(${hurt ? 54 : 46},${hurt ? -212 : -250})`}>
          <circle r={33} fill={C.skin} />
          <path d="M -32 -4 Q -22 -42 12 -38 Q 36 -34 33 -4 Q 0 -24 -32 -4 Z" fill={P.ink} />
          {hurt ? (
            <g stroke={P.ink} strokeWidth={4.5} strokeLinecap="round">
              <path d="M 6 -6 L 18 2 M 18 -6 L 6 2" fill="none" />
              <path d="M -12 -4 q 6 -4 12 0" fill="none" />
            </g>
          ) : (
            <circle cx={15} cy={-2} r={4.6} fill={P.ink} />
          )}
          {hurt && (
            <g transform="translate(44,-16)">
              <path d="M 0 0 Q -9 13 0 21 Q 9 13 0 0 Z" fill={P.sky} />
            </g>
          )}
        </g>
      </g>
    </g>
  );
};

/* ---------- 正面躯干剖面：肋骨 + 膈肌穹顶 + 肝/胃 ---------- */
const TorsoFront = ({x, y, s = 1, liverDrop = 0, showLig = false}: {
  x: number; y: number; s?: number; liverDrop?: number; showLig?: boolean;
}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    {/* 躯干轮廓 */}
    <path d="M -160 -238 Q -186 -60 -146 58 L -132 152 L 132 152 L 146 58 Q 186 -60 160 -238 Q 80 -272 0 -272 Q -80 -272 -160 -238 Z"
      fill={P.card} />
    <rect x={-30} y={-286} width={60} height={26} rx={13} fill={P.card} />
    {/* 脊柱 */}
    <line x1={0} y1={-244} x2={0} y2={140} stroke={P.bg} strokeWidth={13} opacity={0.55} />
    {/* 肋骨（4 对） */}
    {[0, 1, 2, 3].map((i) => (
      <g key={i} stroke={P.cream} strokeWidth={8} fill="none" strokeLinecap="round" opacity={0.85}>
        <path d={`M 0 ${-214 + i * 40} Q ${-72 - i * 5} ${-212 + i * 40} ${-114 + i * 4} ${-190 + i * 40}`} />
        <path d={`M 0 ${-214 + i * 40} Q ${72 + i * 5} ${-212 + i * 40} ${114 - i * 4} ${-190 + i * 40}`} />
      </g>
    ))}
    {/* 膈肌穹顶 */}
    <path d="M -118 -12 Q 0 -86 118 -12" stroke={P.dia} strokeWidth={13} fill="none" strokeLinecap="round" />
    {/* 肝：画面右侧，挂在膈肌下 */}
    <g transform={`translate(58,${44 + liverDrop})`}>
      {showLig && <Lig x={0} y1={-52} y2={-34} col={P.diaD} w={8} />}
      <path d="M -62 -6 Q -70 -36 -30 -42 L 36 -36 Q 72 -28 66 4 Q 58 32 8 34 L -32 32 Q -60 26 -62 -6 Z"
        fill={P.organ} stroke={P.organD} strokeWidth={4} />
    </g>
    {/* 胃：画面左侧 */}
    <ellipse cx={-66} cy={54} rx={40} ry={30} fill={P.pink} opacity={0.95} />
    <ellipse cx={-74} cy={46} rx={16} ry={11} fill={P.cream} opacity={0.35} />
  </g>
);

/* ---------- 脚图标（侧视，脚尖向右） ---------- */
const Foot = ({x, y, s = 1, col = P.cream}: {x: number; y: number; s?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <path d="M -56 -8 L -56 -56 L -20 -56 L 2 -34 Q 54 -22 58 -2 L 58 8 L -56 8 Z" fill={col} />
  </g>
);

/* ---------- 肺叶 ---------- */
const Lungs = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <path d="M -14 -70 L -14 10 L -80 22 Q -104 24 -104 -6 L -104 -52 Q -104 -80 -70 -78 L -40 -74 Q -14 -72 -14 -70 Z"
      fill={P.sky} opacity={0.95} transform="scale(-1,1)" />
    <path d="M -14 -70 L -14 10 L -80 22 Q -104 24 -104 -6 L -104 -52 Q -104 -80 -70 -78 L -40 -74 Q -14 -72 -14 -70 Z"
      fill={P.sky} opacity={0.8} />
    <line x1={0} y1={-96} x2={0} y2={10} stroke={P.cream} strokeWidth={9} strokeLinecap="round" />
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
        {/* 顶部装饰：跑道条 + 小白点 */}
        {Array.from({length: 9}).map((_, i) => (
          <circle key={i} cx={140 + i * 205} cy={86 + (i % 2) * 26} r={4} fill={P.cream} opacity={0.3} />
        ))}
        {/* 跑道 */}
        <g transform={`translate(960,545) scale(${pA}) translate(-960,-545)`} opacity={clamp01(pA)}>
          <rect x={430} y={545} width={1060} height={84} rx={42} fill={P.track} />
          <rect x={430} y={545} width={1060} height={84} rx={42} fill="none" stroke={P.trackD} strokeWidth={6} />
          {Array.from({length: 7}).map((_, i) => (
            <rect key={i} x={520 + i * 140} y={579} width={64} height={14} rx={7} fill={P.lane} />
          ))}
          {/* 捂着肋下的跑者 */}
          <Runner x={830} y={545} s={1.22} mode="hurt" />
          <PainBurst x={912} y={400} s={1.15} />
          {/* 终点旗 */}
          <g transform="translate(1400,395)">
            <line x1={0} y1={0} x2={0} y2={150} stroke={P.cream} strokeWidth={8} strokeLinecap="round" />
            <path d="M 0 0 L 76 18 L 0 38 Z" fill={P.lime} />
          </g>
        </g>
        <g transform={`translate(560,300) scale(${pB})`} opacity={clamp01(pB)}>
          <TagPill x={0} y={0} t="学名 ETAP" col={P.dia} w={210} />
        </g>
        <g transform={`translate(1450,330) scale(${pB})`} opacity={clamp01(pB)}>
          <TagPill x={0} y={0} t="每步一扎" col={P.track} w={190} />
        </g>
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 470}}>
          <div style={{color: P.dia, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 24}}>
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

/* ============ S0 肋下像顶了根钉：疼法 + 正式名 + 凶手未定 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const popL = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const popR = Math.max(0.001, spring({frame: f - 40, fps, config: {damping: 11, stiffness: 120}}));
  const nameK = clamp01((f - b1 - 5) / 24);
  const susK = clamp01((f - b1 - 55) / 24);

  const suspects = [
    {t: '膈肌', col: P.dia},
    {t: '腹膜', col: P.pink},
    {t: '韧带', col: P.amber},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={0} name={sec.name} accent={P.track} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：跑着跑着被扎 */}
        <g transform={`translate(90,150) scale(${popL}) translate(-90,-150)`} opacity={clamp01(popL * 2)}>
          <rect x={90} y={150} width={800} height={500} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={126} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            跑得好好的 · 肋下突然顶了根烧红的钉子</text>
          {/* 地面 */}
          <rect x={130} y={560} width={720} height={18} rx={9} fill={P.track} />
          {Array.from({length: 5}).map((_, i) => (
            <rect key={i} x={170 + i * 140} y={564} width={56} height={10} rx={5} fill={P.lane} />
          ))}
          {/* 速度线（变弱） */}
          {[0, 1, 2].map((i) => (
            <line key={i} x1={180 + i * 40} y1={300 + i * 46} x2={250 + i * 40} y2={300 + i * 46}
              stroke={P.muted} strokeWidth={7} strokeLinecap="round" opacity={0.5 - i * 0.13} />
          ))}
          <Runner x={520} y={552} s={1.15} mode="hurt" />
          <PainBurst x={616} y={412} s={1.05} />
          <TagPill x={300} y={330} t="每跑一步 · 扎一下" col={P.track} w={300} fs={26} s={0.95} />
        </g>
        {/* 右：它有学名 · 凶手没定 */}
        <g transform={`translate(930,150) scale(${popR}) translate(-930,-150)`} opacity={clamp01(popR * 2)}>
          <rect x={930} y={150} width={900} height={500} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={966} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            不是「娇气」 · 它有正式名字</text>
          {/* 大名卡 */}
          <g opacity={nameK} transform={`translate(0,${(1 - nameK) * 26})`}>
            <rect x={966} y={236} width={828} height={118} rx={18} fill={P.bgDeep} stroke={P.dia} strokeWidth={4} />
            <text x={1380} y={290} textAnchor="middle" fontSize={46} fill={P.dia}
              fontFamily={FONT_HEAD} fontWeight={900}>运动相关短暂腹痛</text>
            <text x={1380} y={332} textAnchor="middle" fontSize={26} fill={P.amber}
              fontFamily={FONT_HEAD} fontWeight={800} letterSpacing={6}>ETAP · 俗称「岔气」</text>
          </g>
          {/* 三个嫌疑人 */}
          <text x={1380} y={412} textAnchor="middle" fill={P.muted} fontSize={24} fontFamily={FONT_HEAD}>
            医学界吵到今天 · 桌面上摆着好几个假说</text>
          <g opacity={susK} transform={`translate(0,${(1 - susK) * 26})`}>
            {suspects.map((sp, i) => (
              <g key={i} transform={`translate(${1120 + i * 265},510)`}>
                <rect x={-100} y={-72} width={200} height={144} rx={18} fill={P.bgDeep}
                  stroke={sp.col} strokeWidth={3.5} />
                {/* 嫌疑人剪影头 */}
                <circle cx={0} cy={-22} r={22} fill={sp.col} opacity={0.9} />
                <path d="M -30 22 Q -30 -2 0 -2 Q 30 -2 30 22 Z" fill={sp.col} opacity={0.9} />
                <text x={0} y={58} textAnchor="middle" fontSize={24} fill={P.cream}
                  fontFamily={FONT_HEAD} fontWeight={800}>{sp.t}</text>
                <g transform="translate(72,-66)">
                  <circle r={18} fill={P.track} />
                  <text y={9} textAnchor="middle" fontSize={22} fill={P.cream}
                    fontFamily={FONT_HEAD} fontWeight={900}>?</text>
                </g>
              </g>
            ))}
          </g>
        </g>
        {/* 底部一句话 */}
        <g opacity={susK}>
          <rect x={90} y={700} width={1740} height={130} rx={20} fill={P.panel} stroke={P.dia} strokeWidth={4} />
          <text x={960} y={756} textAnchor="middle" fontSize={30} fill={P.cream}
            fontFamily={FONT_HEAD} fontWeight={800}>
            它不挑新手 · 高水平运动员照岔不误 · 只是概率低一点</text>
          <text x={960} y={800} textAnchor="middle" fontSize={24} fill={P.muted} fontFamily={FONT_HEAD}>
            呼吸力学 × 内脏悬挂 × 腹膜润滑 × 进食时机 —— 比「抽筋」复杂得多</text>
        </g>
      </svg>
      <Chip text="肋骨下缘 · 尖锐刺痛" x={1290} y={72} at={beat(0) + 25} accent={P.track} />
      <Stamp text="不是一个凶手" x={620} y={86} at={b1 + 70} color={P.diaD} size={38} rot={-3} />
    </FadeIn>
  );
};

/* ============ S1 疼点就是线索：右肋下 + 肝是最沉的摆锤 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const pop = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const statK = clamp01((f - 60) / 24);
  const liverK = clamp01((f - b1 - 8) / 26);
  const swing = liverK * Math.sin(f * 0.12) * 14;

  const bars = [
    {t: '右侧肋下', v: 1, col: P.track, note: '明显更多'},
    {t: '左侧肋下', v: 0.45, col: P.sky, note: '也有'},
    {t: '正中', v: 0.22, col: P.muted, note: '少数'},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={1} name={sec.name} accent={P.amber} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：躯干剖面 + 疼点 */}
        <g transform={`translate(90,150) scale(${pop}) translate(-90,-150)`} opacity={clamp01(pop * 2)}>
          <rect x={90} y={150} width={780} height={560} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={126} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            疼的位置固定得可疑 · 就在肋骨下缘</text>
          <TorsoFront x={480} y={480} s={1.12} />
          <PainBurst x={600} y={430} s={0.95} />
          <g opacity={statK}>
            <rect x={150} y={596} width={280} height={56} rx={28} fill={P.bgDeep} stroke={P.amber} strokeWidth={3.5} />
            <text x={290} y={634} textAnchor="middle" fontSize={26} fill={P.amber}
              fontFamily={FONT_HEAD} fontWeight={900}>80%+ 扎在肋下</text>
            <TagPill x={620} y={624} t="尖锐刺痛 ≠ 酸胀" col={P.track} w={300} fs={25} s={0.95} />
          </g>
        </g>
        {/* 右：左右对照 + 肝摆锤 */}
        <g transform={`translate(910,150) scale(${pop}) translate(-910,-150)`} opacity={clamp01(pop * 2)}>
          <rect x={910} y={150} width={920} height={560} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={946} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            全身疲劳解释不了 · 为什么专挑这里</text>
          {/* 左右对比条 */}
          {bars.map((b, i) => (
            <g key={i} transform={`translate(946,${262 + i * 92})`} opacity={statK}>
              <text x={0} y={32} fill={P.cream} fontSize={26} fontFamily={FONT_HEAD} fontWeight={800} width={140}>
                {b.t}</text>
              <rect x={170} y={0} width={540} height={44} rx={22} fill={P.bgDeep} />
              <rect x={170} y={0} width={540 * b.v * statK} height={44} rx={22} fill={b.col} />
              <text x={734} y={32} fill={b.col} fontSize={23} fontFamily={FONT_HEAD} fontWeight={800}>{b.note}</text>
            </g>
          ))}
          {/* 肝 = 最沉的摆锤（cue1） */}
          {liverK > 0 && (
            <g opacity={liverK} transform={`translate(0,${(1 - liverK) * 40})`}>
              <rect x={946} y={540} width={848} height={140} rx={18} fill={P.bgDeep}
                stroke={P.organ} strokeWidth={4} />
              {/* 膈肌线 + 肝摆锤 */}
              <line x1={1000} y1={566} x2={1150} y2={566} stroke={P.dia} strokeWidth={10} strokeLinecap="round" />
              <g transform={`translate(1075,566) rotate(${swing})`}>
                <Lig x={0} y1={4} y2={44} col={P.diaD} w={8} />
                <path d="M -46 44 Q -54 16 -20 12 L 30 16 Q 58 24 52 50 Q 44 74 4 76 L -24 74 Q -48 68 -46 44 Z"
                  fill={P.organ} stroke={P.organD} strokeWidth={4} transform="translate(0,26)" />
                {/* 摆动弧线 */}
                <path d="M -60 96 Q 0 122 60 96" fill="none" stroke={P.amber} strokeWidth={5}
                  strokeDasharray="8 8" opacity={0.85} />
              </g>
              <g transform="translate(1230,566)">
                <rect x={-64} y={-26} width={128} height={52} rx={14} fill={P.organ} />
                <text x={0} y={10} textAnchor="middle" fontSize={26} fill={P.ink}
                  fontFamily={FONT_HEAD} fontWeight={900}>≈1.5kg</text>
              </g>
              <text x={1300} y={582} fill={P.cream} fontSize={25} fontFamily={FONT_HEAD} fontWeight={800}>
                肝脏吊在右上腹</text>
              <text x={1300} y={622} fill={P.muted} fontSize={23} fontFamily={FONT_HEAD}>
                跑步一颠 · 它就是最沉的摆锤</text>
            </g>
          )}
        </g>
      </svg>
      <Chip text="地址固定 → 肇事结构也固定" x={1140} y={72} at={beat(0) + 30} accent={P.amber} />
      <Stamp text="嫌疑锁定这片区域" x={600} y={86} at={b1 + 40} color={P.organD} size={38} rot={-3} />
    </FadeIn>
  );
};

/* ============ S2 头号嫌疑膈肌：供血排队 + 右脚呼气拽肝 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const pop = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const flowK = clamp01((f - 70) / 26);
  const footK = clamp01((f - b1 - 8) / 26);
  const stamp = Math.max(0, footK);
  const pull = Math.sin(f * 0.16) * 10 * stamp;

  const queue = [
    {t: '四肢肌肉', col: P.track, big: true},
    {t: '膈肌', col: P.dia, big: false},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={2} name={sec.name} accent={P.dia} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：膈肌缺血说 */}
        <g transform={`translate(90,150) scale(${pop}) translate(-90,-150)`} opacity={clamp01(pop * 2)}>
          <rect x={90} y={150} width={830} height={570} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={126} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            嫌疑犯① · 膈肌缺血抽筋说（流传最广）</text>
          {/* 肺 + 膈肌穹顶 */}
          <Lungs x={340} y={400} s={1.05} />
          <path d={`M ${340 - 128} ${472 + pull * 0.4} Q 340 ${398 + pull * 0.4} ${340 + 128} ${472 + pull * 0.4}`}
            stroke={P.dia} strokeWidth={14} fill="none" strokeLinecap="round" />
          {/* 膈肌痉挛抖动 */}
          {flowK > 0 && (
            <g opacity={flowK}>
              {[0, 1, 2].map((i) => (
                <Bolt key={i} x={240 + i * 96} y={502 + (i % 2) * 18} s={0.5}
                  col={P.amber} rot={-14 + i * 14} />
              ))}
              <TagPill x={340} y={566} t="缺血 → 痉挛" col={P.track} w={220} fs={25} s={0.95} />
            </g>
          )}
          {/* 供血排队 */}
          <g transform="translate(556,310)" opacity={flowK}>
            <text x={0} y={-18} fill={P.cream} fontSize={24} fontFamily={FONT_HEAD} fontWeight={800}>
              剧烈运动 · 供血排队</text>
            {/* 心脏 */}
            <path d="M 0 26 Q -26 0 -48 22 Q -68 44 -40 70 L 0 96 L 40 70 Q 68 44 48 22 Q 26 0 0 26 Z"
              fill={P.track} transform="scale(0.8)" />
            {/* 粗箭头到四肢 */}
            <line x1={56} y1={40} x2={180} y2={40} stroke={P.track} strokeWidth={14}
              strokeLinecap="round" />
            <polygon points="196,40 172,26 172,54" fill={P.track} />
            <rect x={200} y={4} width={120} height={72} rx={14} fill={P.bgDeep}
              stroke={P.track} strokeWidth={3.5} />
            <text x={260} y={50} textAnchor="middle" fontSize={24} fill={P.track}
              fontFamily={FONT_HEAD} fontWeight={900}>四肢</text>
            {/* 细虚线到膈肌 */}
            <line x1={56} y1={86} x2={180} y2={110} stroke={P.dia} strokeWidth={6}
              strokeLinecap="round" strokeDasharray="8 9" />
            <polygon points="196,114 174,100 176,124" fill={P.dia} />
            <rect x={200} y={88} width={120} height={56} rx={14} fill={P.bgDeep}
              stroke={P.dia} strokeWidth={3.5} />
            <text x={260} y={125} textAnchor="middle" fontSize={23} fill={P.dia}
              fontFamily={FONT_HEAD} fontWeight={900}>膈肌</text>
            <text x={0} y={190} fill={P.muted} fontSize={22} fontFamily={FONT_HEAD}>
              内部员工被排到队尾 → 缺血痉挛</text>
          </g>
        </g>
        {/* 右：右脚落地呼气 → 拽肝（cue1） */}
        {footK > 0 && (
          <g opacity={footK} transform={`translate(0,${(1 - footK) * 40})`}>
            <rect x={960} y={150} width={870} height={570} rx={20} fill={P.panel}
              stroke={P.dia} strokeWidth={4} />
            <text x={996} y={206} fill={P.dia} fontSize={26} fontFamily={FONT_HEAD} fontWeight={900}>
              漂亮细节 · 右脚落地时呼气 = 拽肝</text>
            {/* 三步机制 */}
            {/* 1 呼气：膈肌上抬 */}
            <g transform="translate(1070,420)">
              <rect x={-96} y={-110} width={200} height={220} rx={18} fill={P.bgDeep}
                stroke={P.card} strokeWidth={3.5} />
              <text x={4} y={-78} textAnchor="middle" fontSize={23} fill={P.muted}
                fontFamily={FONT_HEAD}>呼气瞬间</text>
              <path d="M -66 30 Q 4 -22 74 30" stroke={P.dia} strokeWidth={11}
                fill="none" strokeLinecap="round" />
              <line x1={4} y1={-8} x2={4} y2={-46} stroke={P.dia} strokeWidth={7}
                strokeLinecap="round" />
              <polygon points="4,-58 -8,-38 16,-38" fill={P.dia} />
              <text x={4} y={86} textAnchor="middle" fontSize={22} fill={P.dia}
                fontFamily={FONT_HEAD} fontWeight={800}>膈肌上抬放松</text>
            </g>
            {/* 2 右脚落地：冲击 */}
            <g transform="translate(1310,420)">
              <rect x={-96} y={-110} width={200} height={220} rx={18} fill={P.bgDeep}
                stroke={P.card} strokeWidth={3.5} />
              <text x={4} y={-78} textAnchor="middle" fontSize={23} fill={P.muted}
                fontFamily={FONT_HEAD}>右脚落地</text>
              <g transform={`translate(4,${Math.abs(Math.sin(f * 0.3)) * -14})`}>
                <Foot x={0} y={30} s={0.8} col={P.cream} />
              </g>
              <rect x={-70} y={44} width={150} height={12} rx={6} fill={P.track} />
              {[0, 1, 2].map((i) => (
                <line key={i} x1={-40 + i * 40} y1={62} x2={-40 + i * 40} y2={82}
                  stroke={P.amber} strokeWidth={6} strokeLinecap="round" />
              ))}
              <text x={4} y={106} textAnchor="middle" fontSize={22} fill={P.amber}
                fontFamily={FONT_HEAD} fontWeight={800}>冲击往下传</text>
            </g>
            {/* 3 肝被拽下 */}
            <g transform="translate(1560,420)">
              <rect x={-96} y={-110} width={200} height={220} rx={18} fill={P.bgDeep}
                stroke={P.track} strokeWidth={3.5} />
              <text x={4} y={-78} textAnchor="middle" fontSize={23} fill={P.muted}
                fontFamily={FONT_HEAD}>肝韧带</text>
              <g transform={`translate(4,${pull * 0.6})`}>
                <Lig x={0} y1={-40} y2={-6} col={P.diaD} w={9} />
                <ellipse cx={0} cy={26} rx={44} ry={30} fill={P.organ}
                  stroke={P.organD} strokeWidth={4} />
                <line x1={0} y1={58} x2={0} y2={82} stroke={P.track} strokeWidth={7}
                  strokeLinecap="round" />
                <polygon points="0,94 -10,74 10,74" fill={P.track} />
              </g>
              <text x={4} y={106} textAnchor="middle" fontSize={22} fill={P.track}
                fontFamily={FONT_HEAD} fontWeight={800}>把膈肌往下拽</text>
            </g>
            {/* 连接箭头 */}
            <g stroke={P.muted} strokeWidth={5} strokeLinecap="round">
              <line x1={1178} y1={420} x2={1200} y2={420} />
              <polygon points="1212,420 1196,411 1196,429" fill={P.muted} stroke="none" />
              <line x1={1420} y1={420} x2={1446} y2={420} />
              <polygon points="1458,420 1442,411 1442,429" fill={P.muted} stroke="none" />
            </g>
            {/* 结论 */}
            <rect x={996} y={560} width={798} height={120} rx={16} fill={P.bgDeep}
              stroke={P.amber} strokeWidth={3.5} />
            <text x={1395} y={612} textAnchor="middle" fontSize={27} fill={P.cream}
              fontFamily={FONT_HEAD} fontWeight={800}>
              膈肌被反复下拽成千上万次 → 右侧多发</text>
            <text x={1395} y={652} textAnchor="middle" fontSize={23} fill={P.amber}
              fontFamily={FONT_HEAD}>
              所以土办法「换只脚落地呼气」常常真的管用</text>
          </g>
        )}
      </svg>
      <Chip text="呼吸节奏 × 脚步" x={1350} y={72} at={beat(0) + 30} accent={P.dia} />
      <Stamp text="头号嫌疑 · 非唯一真凶" x={580} y={86} at={b1 + 60} color={P.diaD} size={38} rot={-3} />
    </FadeIn>
  );
};

/* ============ S3 还有俩嫌疑犯：腹膜摩擦 + 韧带牵拉 + 多因并存 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const b2 = beat(2);
  const popA = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const popB = Math.max(0.001, spring({frame: f - b1 - 6, fps, config: {damping: 11, stiffness: 120}}));
  const allK = clamp01((f - b2 - 4) / 26);
  const rub = Math.sin(f * 0.5) * 6;
  const bounce = Math.sin(f * 0.28) * 12 * clamp01((f - b1) / 30);

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={3} name={sec.name} accent={P.pink} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：腹膜摩擦 */}
        <g transform={`translate(90,140) scale(${popA}) translate(-90,-140)`} opacity={clamp01(popA * 2)}>
          <rect x={90} y={140} width={830} height={500} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={126} y={196} fill={P.pink} fontSize={26} fontFamily={FONT_HEAD} fontWeight={900}>
            嫌疑犯② · 腹膜被摩擦</text>
          {/* 两层腹膜 */}
          <path d={`M 150 ${300 + rub * 0.3} Q 505 ${268 + rub * 0.3} 860 ${300 + rub * 0.3}`}
            stroke={P.pink} strokeWidth={10} fill="none" strokeLinecap="round" opacity={0.9} />
          <path d={`M 150 ${560 - rub * 0.3} Q 505 ${592 - rub * 0.3} 860 ${560 - rub * 0.3}`}
            stroke={P.pink} strokeWidth={10} fill="none" strokeLinecap="round" opacity={0.9} />
          <text x={150} y={262} fill={P.pink} fontSize={21} fontFamily={FONT_HEAD}>壁层腹膜 · 布满痛觉神经</text>
          {/* 痛觉神经点 */}
          {[0, 1, 2, 3, 4].map((i) => (
            <circle key={i} cx={240 + i * 130} cy={290 + (i % 2) * 8} r={7}
              fill={i % 2 ? P.track : P.amber} />
          ))}
          {/* 吃饱的胃往下坠 */}
          <g transform={`translate(505,${430 + rub})`}>
            <path d="M -110 -60 Q -130 30 -60 66 L 60 66 Q 130 30 110 -60 Q 60 -88 0 -88 Q -60 -88 -110 -60 Z"
              fill={P.pink} stroke={P.trackD} strokeWidth={4} />
            {/* 内容物 */}
            <rect x={-58} y={-40} width={44} height={30} rx={8} fill={P.amber} />
            <rect x={-6} y={-48} width={40} height={28} rx={8} fill={P.lime} />
            <rect x={-30} y={-6} width={52} height={26} rx={8} fill={P.sky} />
            <circle cx={52} cy={-16} r={13} fill={P.amberD} />
          </g>
          <TagPill x={700} y={252} t="吃饱 / 高糖饮料 → 胃往下坠" col={P.pink} w={380} fs={23} s={0.92} />
          {/* 摩擦火花 */}
          {[0, 1, 2].map((i) => (
            <Bolt key={i} x={300 + i * 210} y={310 + (i % 2) * 240} s={0.45}
              col={P.amber} rot={i * 30 - 20} />
          ))}
          <text x={505} y={610} textAnchor="middle" fontSize={23} fill={P.muted} fontFamily={FONT_HEAD}>
            润滑被打乱 · 颠簸中两层膜互相挤压摩擦 → 刺痛</text>
        </g>
        {/* 右：韧带牵拉 */}
        {popB > 0.02 && (
          <g transform={`translate(960,140) scale(${popB}) translate(-960,-140)`} opacity={clamp01(popB * 2)}>
            <rect x={960} y={140} width={870} height={500} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
            <text x={996} y={196} fill={P.amber} fontSize={26} fontFamily={FONT_HEAD} fontWeight={900}>
              嫌疑犯③ · 内脏韧带牵拉</text>
            {/* 膈肌天花板 */}
            <line x1={1010} y1={268} x2={1780} y2={268} stroke={P.dia} strokeWidth={12}
              strokeLinecap="round" />
            <text x={1780} y={252} textAnchor="end" fill={P.dia} fontSize={21}
              fontFamily={FONT_HEAD}>膈肌 / 腹壁 = 天花板</text>
            {/* 三个悬挂器官 */}
            {[
              {x: 1130, r: 46, col: P.organ, colD: P.organD, t: '肝'},
              {x: 1395, r: 40, col: P.pink, colD: P.trackD, t: '胃'},
              {x: 1650, r: 34, col: '#b79cff', colD: '#8a6fe0', t: '脾'},
            ].map((o, i) => (
              <g key={i} transform={`translate(${o.x},${bounce * (i === 1 ? -1 : 1)})`}>
                <Lig x={-16} y1={274} y2={352} col={P.diaD} w={10} />
                <Lig x={16} y1={274} y2={352} col={P.diaD} w={10} />
                <ellipse cx={0} cy={392} rx={o.r + 14} ry={o.r}
                  fill={o.col} stroke={o.colD} strokeWidth={4} />
                <text x={0} y={402} textAnchor="middle" fontSize={26} fill={P.ink}
                  fontFamily={FONT_HEAD} fontWeight={900}>{o.t}</text>
              </g>
            ))}
            {/* 落地冲击记号 */}
            {[0, 1, 2, 3].map((i) => (
              <g key={i} transform={`translate(${1120 + i * 180},540)`}>
                <rect x={-34} y={0} width={68} height={12} rx={6} fill={P.track} />
                <line x1={-20} y1={-8} x2={-20} y2={-26} stroke={P.amber} strokeWidth={5}
                  strokeLinecap="round" />
                <line x1={20} y1={-8} x2={20} y2={-26} stroke={P.amber} strokeWidth={5}
                  strokeLinecap="round" />
              </g>
            ))}
            <text x={1395} y={596} textAnchor="middle" fontSize={23} fill={P.muted} fontFamily={FONT_HEAD}>
              几公斤内脏吊着 · 每步落地都被抻一下</text>
          </g>
        )}
        {/* 底部：多因并存（cue2） */}
        {allK > 0 && (
          <g opacity={allK} transform={`translate(0,${(1 - allK) * 50})`}>
            <rect x={90} y={680} width={1740} height={160} rx={20} fill={P.panel}
              stroke={P.amber} strokeWidth={4} />
            <text x={960} y={742} textAnchor="middle" fontSize={32} fill={P.cream}
              fontFamily={FONT_HEAD} fontWeight={900}>
              三条假说不打架 —— 不同人岔的可能是不同的气</text>
            <g transform="translate(960,792)">
              <TagPill x={-560} y={0} t="饭后立刻跑 → 腹膜型" col={P.pink} w={360} fs={24} s={0.9} />
              <TagPill x={0} y={0} t="呼吸乱 → 膈肌型" col={P.dia} w={300} fs={24} s={0.9} />
              <TagPill x={560} y={0} t="颠得凶 → 韧带型" col={P.amber} w={320} fs={24} s={0.9} />
            </g>
          </g>
        )}
      </svg>
      <Chip text="骑马也会岔 → 震动本身就在揉腹膜" x={1010} y={72} at={b1 + 30} accent={P.pink} />
      <Stamp text="多因并存" x={620} y={86} at={b2 + 30} color={P.amberD} size={40} rot={-3} />
    </FadeIn>
  );
};

/* ============ S4 岔气自救指南：四招 + 预防 + 警示 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const b2 = beat(2);
  const prevK = clamp01((f - b1 - 4) / 26);
  const warnK = clamp01((f - b2 - 4) / 26);

  const steps = [
    {t: '减速', s: '步频降 · 颠簸立减', col: P.track},
    {t: '深呼吸', s: '加深呼气 · 重置膈肌', col: P.dia},
    {t: '换脚呼气', s: '右脚 → 左脚落地呼', col: P.sky},
    {t: '按压弯腰', s: '压疼点 · 弯腰减负', col: P.amber},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={4} name={sec.name} accent={P.lime} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 顶部：四招卡片 */}
        {steps.map((st, i) => {
          const at = 8 + i * 14;
          const sp = Math.max(0.001, spring({frame: f - at, fps, config: {damping: 11, stiffness: 130}}));
          const sx = 150 + i * 445;
          return (
            <g key={i} transform={`translate(${sx},170) scale(${sp}) translate(${-sx},-170)`}
              opacity={clamp01(sp * 2)}>
              <rect x={sx} y={170} width={390} height={260} rx={20} fill={P.panel}
                stroke={st.col} strokeWidth={4} />
              <circle cx={sx + 52} cy={222} r={26} fill={st.col} />
              <text x={sx + 52} y={232} textAnchor="middle" fontSize={27} fill={P.ink}
                fontFamily={FONT_HEAD} fontWeight={900}>{i + 1}</text>
              <text x={sx + 195} y={232} textAnchor="middle" fontSize={32} fill={st.col}
                fontFamily={FONT_HEAD} fontWeight={900}>{st.t}</text>
              {/* 图标区 */}
              <g transform={`translate(${sx + 195},320)`}>
                {i === 0 && (
                  <g>
                    <Runner x={-30} y={46} s={0.34} mode="easy" jersey={st.col} />
                    <line x1={-96} y1={-6} x2={-60} y2={-6} stroke={st.col} strokeWidth={7}
                      strokeLinecap="round" opacity={0.7} />
                    <line x1={-104} y1={16} x2={-62} y2={16} stroke={st.col} strokeWidth={7}
                      strokeLinecap="round" opacity={0.45} />
                  </g>
                )}
                {i === 1 && (
                  <g>
                    {/* 膈肌穹顶 + 上下呼吸箭头 */}
                    <path d="M -46 18 Q 0 -30 46 18" fill="none" stroke={st.col}
                      strokeWidth={9} strokeLinecap="round" />
                    <line x1={0} y1={-8} x2={0} y2={-40} stroke={P.cream} strokeWidth={7}
                      strokeLinecap="round" />
                    <polygon points="0,-52 -11,-32 11,-32" fill={P.cream} />
                    <line x1={0} y1={34} x2={0} y2={54} stroke={P.cream} strokeWidth={7}
                      strokeLinecap="round" />
                    <polygon points="0,66 -11,46 11,46" fill={P.cream} />
                  </g>
                )}
                {i === 2 && (
                  <g>
                    <Foot x={-66} y={-18} s={0.62} col={P.muted} />
                    <Foot x={30} y={18} s={0.62} col={st.col} />
                    <path d="M -30 -46 Q 20 -62 56 -44" fill="none" stroke={st.col}
                      strokeWidth={6} strokeLinecap="round" />
                    <polygon points="66,-40 48,-52 50,-30" fill={st.col} />
                  </g>
                )}
                {i === 3 && (
                  <g>
                    {/* 弯腰小人 */}
                    <circle cx={-40} cy={-30} r={17} fill={C.skin} />
                    <path d="M -30 -22 L 30 -6 L 10 34" fill="none" stroke={st.col}
                      strokeWidth={16} strokeLinecap="round" strokeLinejoin="round" />
                    <line x1={10} y1={34} x2={-8} y2={52} stroke={P.ink} strokeWidth={12}
                      strokeLinecap="round" />
                    <line x1={10} y1={34} x2={30} y2={52} stroke={P.ink} strokeWidth={12}
                      strokeLinecap="round" />
                    {/* 按压的手 */}
                    <circle cx={52} cy={-2} r={12} fill={C.skin} />
                    <PainBurst x={56} y={-8} s={0.4} />
                  </g>
                )}
              </g>
              <text x={sx + 195} y={404} textAnchor="middle" fontSize={23} fill={P.muted}
                fontFamily={FONT_HEAD}>{st.s}</text>
            </g>
          );
        })}
        {/* 左下：预防（cue1） */}
        {prevK > 0 && (
          <g opacity={prevK} transform={`translate(0,${(1 - prevK) * 50})`}>
            <rect x={90} y={480} width={860} height={350} rx={20} fill={P.panel}
              stroke={P.lime} strokeWidth={4} />
            <text x={126} y={540} fill={P.lime} fontSize={30} fontFamily={FONT_HEAD} fontWeight={900}>
              预防 · 让胃里轻一点</text>
            {/* 时钟 2h */}
            <g transform="translate(190,640)">
              <circle r={48} fill={P.bgDeep} stroke={P.lime} strokeWidth={6} />
              <line x1={0} y1={0} x2={0} y2={-30} stroke={P.cream} strokeWidth={7} strokeLinecap="round" />
              <line x1={0} y1={0} x2={22} y2={12} stroke={P.cream} strokeWidth={7} strokeLinecap="round" />
              <text x={0} y={92} textAnchor="middle" fontSize={25} fill={P.cream}
                fontFamily={FONT_HEAD} fontWeight={800}>饭后 ≥2 小时再跑</text>
              <text x={0} y={126} textAnchor="middle" fontSize={20} fill={P.muted}
                fontFamily={FONT_HEAD}>大餐隔 3 小时</text>
            </g>
            {/* 别灌甜饮 */}
            <g transform="translate(430,640)">
              <path d="M -32 -50 L 32 -50 L 22 40 L -22 40 Z" fill={P.bgDeep}
                stroke={P.muted} strokeWidth={5} />
              <path d="M -27 -30 L 27 -30 L 22 34 L -22 34 Z" fill={P.pink} opacity={0.85} />
              <line x1={10} y1={-50} x2={30} y2={-88} stroke={P.muted} strokeWidth={7}
                strokeLinecap="round" />
              <MarkX x={44} y={-44} s={1.2} col={P.track} />
              <text x={0} y={92} textAnchor="middle" fontSize={25} fill={P.cream}
                fontFamily={FONT_HEAD} fontWeight={800}>跑前别灌甜饮料</text>
              <text x={0} y={126} textAnchor="middle" fontSize={20} fill={P.muted}
                fontFamily={FONT_HEAD}>小口润喉就行</text>
            </g>
            {/* 热身 */}
            <g transform="translate(670,640)">
              <path d="M -60 34 L -16 34 L 4 -6 L 28 34 L 62 34" fill="none"
                stroke={P.lime} strokeWidth={8} strokeLinecap="round" strokeLinejoin="round" />
              <circle cx={4} cy={-26} r={12} fill={P.lime} />
              <text x={0} y={92} textAnchor="middle" fontSize={25} fill={P.cream}
                fontFamily={FONT_HEAD} fontWeight={800}>热身先把呼吸练顺</text>
              <text x={0} y={126} textAnchor="middle" fontSize={20} fill={P.muted}
                fontFamily={FONT_HEAD}>慢跑到配速 · 逐步过渡</text>
            </g>
            <text x={520} y={796} textAnchor="middle" fontSize={23} fill={P.muted}
              fontFamily={FONT_HEAD}>胃沉是所有机制的帮凶 · 老手岔得少就因为颠得轻、吃得巧</text>
          </g>
        )}
        {/* 右下：警示（cue2） */}
        {warnK > 0 && (
          <g opacity={warnK} transform={`translate(0,${(1 - warnK) * 50})`}>
            <rect x={990} y={480} width={840} height={350} rx={20} fill={P.panel}
              stroke={P.track} strokeWidth={5} />
            <text x={1026} y={540} fill={P.track} fontSize={30} fontFamily={FONT_HEAD} fontWeight={900}>
              但不一定是岔气 · 要会鉴别</text>
            <g transform="translate(1090,650)">
              <Check x={0} y={-8} s={1.1} col={P.lime} />
              <text x={36} y={0} fill={P.cream} fontSize={25} fontFamily={FONT_HEAD} fontWeight={800}>
                停下几分钟就消 → 岔气</text>
            </g>
            <g transform="translate(1090,724)">
              <MarkX x={0} y={-8} s={1.2} col={P.track} />
              <text x={36} y={0} fill={P.cream} fontSize={25} fontFamily={FONT_HEAD} fontWeight={800}>
                停了还疼 · 疼在胸口或下腹</text>
            </g>
            <rect x={1030} y={756} width={400} height={56} rx={14} fill={P.bgDeep}
              stroke={P.track} strokeWidth={3.5} />
            <text x={1230} y={793} textAnchor="middle" fontSize={25} fill={P.track}
              fontFamily={FONT_HEAD} fontWeight={900}>别硬扛 · 去查</text>
            {/* 心脏示意：上腹痛可能是心脏 */}
            <g transform="translate(1620,660)">
              <path d="M 0 -14 Q -34 -44 -62 -16 Q -88 10 -54 44 L 0 76 L 54 44 Q 88 10 62 -16 Q 34 -44 0 -14 Z"
                fill={P.track} opacity={0.95} />
              <path d="M -44 18 L -18 18 L -8 2 L 4 30 L 14 12 L 44 12" fill="none"
                stroke={P.cream} strokeWidth={6} strokeLinecap="round" strokeLinejoin="round" />
              <text x={0} y={120} textAnchor="middle" fontSize={21} fill={P.muted}
                fontFamily={FONT_HEAD}>中老年上腹痛 · 警惕心脏</text>
            </g>
          </g>
        )}
      </svg>
      <Chip text="四招顺序试 · 记下最管用的" x={1140} y={72} at={beat(0) + 30} accent={P.lime} />
      <Stamp text="停止即愈 = 岔气" x={620} y={86} at={b2 + 40} color={P.trackD} size={38} rot={-3} />
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
        {/* 跑道 */}
        <g transform={`translate(960,760) scale(${pop})`} opacity={clamp01(pop)}>
          <rect x={-560} y={0} width={1120} height={76} rx={38} fill={P.track} />
          {Array.from({length: 6}).map((_, i) => (
            <rect key={i} x={-460 + i * 170} y={30} width={70} height={14} rx={7} fill={P.lane} />
          ))}
          <Runner x={-80} y={0} s={1.05} mode="run" jersey={P.lime} />
          <g transform="translate(160,-190)">
            <rect x={-170} y={-48} width={340} height={96} rx={18} fill={P.panel}
              stroke={P.lime} strokeWidth={4} />
            <Check x={-128} y={-16} s={0.9} />
            <text x={0} y={-12} textAnchor="middle" fontSize={25} fill={P.lime}
              fontFamily={FONT_HEAD} fontWeight={900}>读懂信号</text>
            <text x={0} y={26} textAnchor="middle" fontSize={21} fill={P.muted}
              fontFamily={FONT_HEAD}>调整工况比硬顶聪明</text>
          </g>
        </g>
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -330}}>
        <div style={{fontSize: 52, fontWeight: 800, color: C.text}}>岔气不是伤 · 是身体在说「工况不适应」</div>
        <div style={{fontSize: 36, color: P.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E241 = ({data = tl}: {data?: any}) => {
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
