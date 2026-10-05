import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_BODY, FONT_HEAD, clamp01, rand} from '../lib/style';
import {Backdrop, Stamp, SceneTag, Chip, Subtitles, ProgressBar, Moon, Stars, FadeIn} from '../lib/atoms';
import tl from '../timeline_223.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：深夜紫蓝 × 脑粉 × 值班灯琥珀（纯色平涂） ============ */
const V = {
  bg0: '#1a1634',        // S0 深紫夜·值班表
  bg1: '#131f3c',        // S1 深海军蓝·质检车间
  bg2: '#251736',        // S2 深葡紫·片场
  bg3: '#10262f',        // S3 深青夜·碎片工坊
  bg4: '#1d1830',        // S4 深靛紫·存档室
  panel: '#232a52',      // 面板
  panel2: '#1b2044',     // 面板内衬
  brain: '#f2a9bc',      // 脑粉
  brainShade: '#d9869d',
  groove: '#c9758f',
  on: '#ffc94d',         // 在岗灯·琥珀
  onDeep: '#e8a83a',
  amy: '#ff7a6e',        // 杏仁核·珊瑚
  amyDeep: '#d95a52',
  hippo: '#63e2c6',      // 海马·青
  hippoDeep: '#3aa892',
  stem: '#9b7ede',       // 脑干·紫
  off: '#3d4266',        // 熄灯区
  offHi: '#5a6089',
  rem: '#5cc8ff',        // REM 眼蓝
  paper: '#f6ecd8',      // 纸
  paperSh: '#e2d0ae',
  ink: '#2a2e4a',
  cream: '#f6ecd8',
  wall: '#478ba4',       // 走廊墙
  wallD: '#356c83',
  floor: '#2c5a74',
  ceil: '#1f4258',
  backwall: '#3a7590',
  seal: '#ff6b6b',       // 盖章红
  ok: '#7dffa8',
  belt: '#39406e',
  roll: '#566099',
  lamp: '#ffd166',
};

/* ---------- 大脑侧视图（头朝左）：lit 控制各脑区亮灭 ---------- */
const Brain = ({x, y, s = 1, lit}: {
  x: number; y: number; s?: number;
  lit: {vis: boolean; amy: boolean; hip: boolean; pre: boolean};
}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    {/* 脑干 */}
    <path d="M -34 100 C -34 138 -20 162 8 174 L 44 150 C 24 138 16 118 20 98 Z" fill={V.brain} />
    {/* 脑体 */}
    <path d="M -170 55 C -185 -10 -160 -80 -95 -115 C -50 -138 40 -140 100 -110
      C 155 -82 180 -30 168 30 C 158 80 110 105 55 110 C 20 114 -10 112 -40 118
      C -90 128 -150 105 -170 55 Z" fill={V.brain} />
    {/* 脑区：背外侧前额叶(前上) / 视觉皮层(后) / 杏仁核(中) / 海马旁回(下) */}
    <ellipse cx={96} cy={-28} rx={58} ry={52} fill={lit.vis ? V.on : V.off} />
    <ellipse cx={-4} cy={34} rx={36} ry={30} fill={lit.amy ? V.amy : V.off} />
    <ellipse cx={72} cy={70} rx={46} ry={30} fill={lit.hip ? V.hippo : V.off} />
    <ellipse cx={-106} cy={-44} rx={54} ry={46} fill={lit.pre ? V.on : V.off} />
    {/* 沟回短线 */}
    <path d="M -40 -118 C -30 -86 -52 -60 -44 -28" fill="none" stroke={V.groove} strokeWidth={7} strokeLinecap="round" />
    <path d="M 148 -50 C 160 -20 158 12 146 40" fill="none" stroke={V.groove} strokeWidth={7} strokeLinecap="round" />
    <path d="M -160 30 C -150 60 -130 84 -96 96" fill="none" stroke={V.groove} strokeWidth={7} strokeLinecap="round" />
    {/* 外轮廓 */}
    <path d="M -170 55 C -185 -10 -160 -80 -95 -115 C -50 -138 40 -140 100 -110
      C 155 -82 180 -30 168 30 C 158 80 110 105 55 110 C 20 114 -10 112 -40 118
      C -90 128 -150 105 -170 55 Z" fill="none" stroke={V.brainShade} strokeWidth={9} />
  </g>
);

/* ---------- REM 扫动的眼睛 ---------- */
const RemEye = ({x, y}: {x: number; y: number}) => {
  const f = useCurrentFrame();
  const px = Math.sin(f * 0.16) * 36;
  return (
    <g transform={`translate(${x},${y})`}>
      <ellipse rx={104} ry={60} fill="#eef3ff" />
      <circle cx={px} r={33} fill={V.rem} />
      <circle cx={px} r={15} fill={V.ink} />
      <circle cx={px - 10} cy={-11} r={6.5} fill="#fff" />
      <path d="M -104 0 Q 0 -76 104 0" fill="none" stroke={V.ink} strokeWidth={6} />
      <path d="M -104 0 Q 0 76 104 0" fill="none" stroke={V.ink} strokeWidth={6} />
    </g>
  );
};

/* ---------- 值班表行 ---------- */
const DutyRow = ({x, y, name, on, status, delay, color}: {
  x: number; y: number; name: string; on: boolean; status: string; delay: number; color: string;
}) => {
  const f = useCurrentFrame();
  const o = interpolate(f, [delay, delay + 14], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const sx = interpolate(f, [delay, delay + 14], [-26, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  if (o <= 0) return null;
  return (
    <g opacity={o} transform={`translate(${x + sx},${y})`}>
      <circle r={14} fill={on ? color : V.off} />
      {on && <circle r={6} fill="#fff" opacity={0.7} />}
      <text x={34} y={11} fill={C.text} fontSize={35} fontFamily={FONT_HEAD} fontWeight={700}>{name}</text>
      <text x={700} y={11} textAnchor="end" fill={on ? color : C.muted} fontSize={30} fontFamily={FONT_HEAD}>{status}</text>
    </g>
  );
};

/* ---------- 悬浮 Zzz ---------- */
const Zzz = ({x, y, delay = 0}: {x: number; y: number; delay?: number}) => {
  const f = useCurrentFrame();
  const t = f - delay;
  if (t < 0) return null;
  const rise = (t * 0.9) % 90;
  const o = clamp01(t / 12) * (1 - rise / 100);
  return (
    <g transform={`translate(${x + rise * 0.4},${y - rise})`} opacity={o}>
      <text fill={C.muted} fontSize={44} fontFamily={FONT_HEAD} fontWeight={900}>Z</text>
      <text x={30} y={-26} fill={C.muted} fontSize={32} fontFamily={FONT_HEAD} fontWeight={900} opacity={0.8}>z</text>
      <text x={54} y={-46} fill={C.muted} fontSize={24} fontFamily={FONT_HEAD} fontWeight={900} opacity={0.6}>z</text>
    </g>
  );
};

/* ============ 标题卡 ============ */
const TitleCard = ({data}: {data: any}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 13, mass: 0.7}});
  const o = interpolate(f, [0, 8], [0, 1]);
  const pop = spring({frame: f - 4, fps, config: {damping: 11, stiffness: 150}});
  return (
    <AbsoluteFill>
      <Backdrop top={V.bg0} bottom={V.bg0} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={16} from={0} dur={30} />
        <Moon x={1660} y={190} scale={0.9} />
        {/* 顶部图标：夜班大脑徽章 */}
        <g transform={`translate(960,290) scale(${Math.max(0.001, pop)})`}>
          <circle r={158} fill={V.panel} stroke={V.rem} strokeWidth={7} />
          <Brain x={0} y={6} s={0.62} lit={{vis: true, amy: true, hip: true, pre: false}} />
        </g>
        <Zzz x={1108} y={216} delay={10} />
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 420}}>
          <div style={{color: V.rem, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 26}}>
            冷知识 · 第 {data.episode} 期
          </div>
          <div style={{color: C.text, fontSize: 92, fontWeight: 900, fontFamily: FONT_HEAD, letterSpacing: 5}}>
            {data.title}
          </div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ============ S0 值班表：脑区明暗 + REM 扫动眼 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);

  const rowAt = (i: number) => beat(1) + i * 16;
  const lit = {vis: f > rowAt(0), amy: f > rowAt(1), hip: f > rowAt(2), pre: false};
  const rows = [
    {name: '视觉皮层', on: true, status: '在岗 · 灯火通明', color: V.on, blob: [688, 429]},
    {name: '杏仁核', on: true, status: '在岗 · 情绪拉满', color: V.amy, blob: [560, 517]},
    {name: '海马旁回', on: true, status: '在岗 · 重播记忆', color: V.hippo, blob: [654, 564]},
    {name: '背外侧前额叶', on: false, status: '熄灯 · 休假中', color: V.off, blob: [418, 409]},
  ];

  return (
    <FadeIn>
      <Backdrop top={V.bg0} bottom={V.bg0} />
      <SceneTag index={0} name={sec.name} accent={V.rem} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：大脑 + 脑区亮灭 */}
        <Brain x={560} y={470} s={1.35} lit={lit} />
        <text x={560} y={718} textAnchor="middle" fill={C.muted} fontSize={28} fontFamily={FONT_HEAD}>
          做梦中的大脑（REM 期）
        </text>

        {/* REM 扫动眼 */}
        <RemEye x={560} y={806} />
        <text x={560} y={872} textAnchor="middle" fill={V.rem} fontSize={26} fontFamily={FONT_BODY}>
          眼球在追看不见的画面
        </text>

        {/* 右：值班表面板 */}
        <rect x={950} y={220} width={790} height={570} rx={26} fill={V.panel} />
        <rect x={950} y={220} width={790} height={90} rx={26} fill={V.panel2} />
        <text x={1000} y={280} fill={V.cream} fontSize={38} fontFamily={FONT_HEAD} fontWeight={700}>
          做梦时 · 大脑值班表
        </text>
        {rows.map((r, i) => (
          <g key={i}>
            {/* 引导虚线：脑区 → 行 */}
            {f > rowAt(i) && (
              <line x1={r.blob[0]} y1={r.blob[1]} x2={985} y2={390 + i * 92}
                stroke={r.on ? r.color : V.offHi} strokeWidth={4} strokeDasharray="9 10"
                opacity={clamp01((f - rowAt(i)) / 16) * 0.8} />
            )}
            <DutyRow x={1000} y={390 + i * 92} name={r.name} on={r.on}
              status={r.status} delay={rowAt(i)} color={r.color} />
          </g>
        ))}
      </svg>
      <Chip text="REM 快速眼动期" x={720} y={110} at={beat(0) + 10} accent={V.rem} />
      <Stamp text="明暗极不均匀" x={200} y={150} at={beat(2) + 8} color={V.onDeep} size={50} rot={-4} />
      <Stamp text="审查部门集体休假" x={1040} y={830} at={beat(2) + 40} color={V.seal} size={48} rot={3} />
    </FadeIn>
  );
};

/* ---------- 半成品梦卡（荒诞剧情） ---------- */
const DreamCard = ({type, seal}: {type: number; seal: number}) => (
  <g>
    <rect width={264} height={200} rx={14} fill="#f2ecdc" />
    <rect x={10} y={10} width={244} height={180} rx={10} fill="#d8cfb8" />
    {type === 0 && (
      <g>
        {/* 客厅一步跨进教室 */}
        <rect x={16} y={16} width={112} height={168} fill="#8ec9d8" />
        <rect x={30} y={110} width={84} height={46} rx={12} fill={V.onDeep} />
        <rect x={30} y={96} width={84} height={22} rx={10} fill="#e8b03a" />
        <rect x={136} y={16} width={112} height={168} fill="#3d5a80" />
        <rect x={150} y={40} width={84} height={60} rx={6} fill="#2e4568" />
        <line x1={158} y1={58} x2={222} y2={58} stroke="#cfe0f0" strokeWidth={5} />
        <line x1={158} y1={76} x2={210} y2={76} stroke="#cfe0f0" strokeWidth={5} />
        <path d="M 122 96 L 146 96 M 136 86 L 148 96 L 136 106" stroke={V.ink} strokeWidth={7}
          fill="none" strokeLinecap="round" />
      </g>
    )}
    {type === 1 && (
      <g>
        {/* 逝去的长辈和昨天的客户同桌 */}
        <ellipse cx={132} cy={160} rx={96} ry={22} fill="#b39ddb" />
        <circle cx={92} cy={86} r={30} fill={C.skin} />
        <path d="M 62 78 Q 92 44 122 78" fill="none" stroke="#d9d9e8" strokeWidth={10} />
        <circle cx={176} cy={86} r={30} fill={C.skin} />
        <rect x={168} y={112} width={16} height={30} fill="#4a5f8f" />
        <rect x={120} y={140} width={20} height={16} rx={4} fill={V.seal} />
      </g>
    )}
    {type === 2 && (
      <g>
        {/* 通向云里的楼梯 */}
        {[0, 1, 2, 3].map(i => (
          <rect key={i} x={30 + i * 40} y={150 - i * 32} width={52} height={18} rx={6} fill="#7fc4d8" />
        ))}
        <ellipse cx={200} cy={60} rx={52} ry={24} fill="#eef3ff" />
        <ellipse cx={166} cy={70} rx={34} ry={18} fill="#eef3ff" />
      </g>
    )}
    {/* 红印章：真实 */}
    {seal > 0 && (
      <g transform={`translate(212,50) rotate(-14) scale(${0.5 + 0.5 * seal})`} opacity={seal}>
        <circle r={40} fill="none" stroke={V.seal} strokeWidth={7} />
        <circle r={30} fill="none" stroke={V.seal} strokeWidth={2.5} />
        <text y={9} textAnchor="middle" fill={V.seal} fontSize={25} fontWeight={900} fontFamily={FONT_HEAD}>真实</text>
      </g>
    )}
  </g>
);

/* ============ S1 质检员下班：传送带半成品照盖章 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);

  const gone = clamp01((f - beat(1) - 14) / 16);            // 质检员离岗
  const cardX = (i: number) => -200 + ((f * 3.1 + i * 700) % 2300);
  const press = Math.max(...[0, 1, 2].map(i => Math.max(0, 1 - Math.abs(cardX(i) + 132 - 1560) / 170)));

  return (
    <FadeIn>
      <Backdrop top={V.bg1} bottom={V.bg1} />
      <SceneTag index={1} name={sec.name} accent={V.seal} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 质检亭 */}
        <rect x={110} y={215} width={560} height={370} rx={24} fill={V.panel} />
        <text x={150} y={272} fill={V.cream} fontSize={34} fontFamily={FONT_HEAD} fontWeight={700}>
          质检亭 · 背外侧前额叶
        </text>
        {/* 桌子 + 台灯 */}
        <rect x={150} y={470} width={400} height={24} rx={8} fill={V.belt} />
        <rect x={150} y={494} width={18} height={70} fill={V.belt} />
        <rect x={532} y={494} width={18} height={70} fill={V.belt} />
        <rect x={196} y={420} width={10} height={50} fill={V.offHi} />
        <path d="M 176 420 L 226 420 L 214 392 L 188 392 Z" fill={gone > 0.5 ? V.off : V.lamp} />
        {gone < 0.5 && <circle cx={201} cy={428} r={12} fill={V.lamp} opacity={0.8} />}
        {/* 椅子 */}
        <rect x={560} y={400} width={34} height={120} rx={10} fill={V.roll} />
        <rect x={500} y={440} width={80} height={20} rx={8} fill={V.roll} />
        {/* 质检员（离岗后淡出） */}
        <g opacity={1 - gone}>
          <rect x={430} y={400} width={76} height={80} rx={20} fill={V.rem} />
          <circle cx={468} cy={366} r={34} fill={C.skin} />
          <rect x={438} y={338} width={60} height={18} rx={9} fill={V.ink} />
          {/* 手里的小旗「查」 */}
          <line x1={506} y1={410} x2={548} y2={386} stroke={C.skin} strokeWidth={10} strokeLinecap="round" />
          <rect x={546} y={352} width={54} height={40} rx={6} fill={V.ok} />
          <text x={573} y={380} textAnchor="middle" fill={V.ink} fontSize={26} fontWeight={900} fontFamily={FONT_HEAD}>查</text>
        </g>
        {/* 离岗标记 */}
        {gone > 0.4 && (
          <g opacity={gone}>
            <rect x={420} y={330} width={120} height={52} rx={12} fill={V.seal} transform="rotate(-8 480 356)" />
            <text x={480} y={366} textAnchor="middle" fill="#fff" fontSize={30} fontWeight={900}
              fontFamily={FONT_HEAD} transform="rotate(-8 480 356)">离岗</text>
            <Zzz x={560} y={320} />
          </g>
        )}
        <text x={150} y={552} fill={C.muted} fontSize={26} fontFamily={FONT_BODY}>
          职责：查逻辑 · 查合理 · 查连贯
        </text>

        {/* 传送带 */}
        <rect x={60} y={648} width={1800} height={26} rx={13} fill={V.belt} />
        {Array.from({length: 16}).map((_, i) => (
          <circle key={i} cx={110 + i * 118} cy={700} r={24} fill={V.roll}
            transform={`rotate(${f * 2} ${110 + i * 118} 700)`} />
        ))}
        {/* 盖章机 */}
        <g transform={`translate(1560,0)`}>
          <rect x={-16} y={300} width={32} height={270 - press * 0 + 46 * press} fill={V.offHi} />
          <rect x={-52} y={330 + 46 * press} width={104} height={44} rx={10} fill={V.seal} />
          <text x={0} y={360 + 46 * press} textAnchor="middle" fill="#fff" fontSize={24}
            fontWeight={900} fontFamily={FONT_HEAD}>真实</text>
          <rect x={-70} y={290} width={140} height={30} rx={8} fill={V.belt} />
        </g>
        {/* 半成品梦卡 */}
        {[0, 1, 2].map(i => {
          const cx = cardX(i);
          const seal = clamp01((cx + 132 - 1440) / 46);
          return (
            <g key={i} transform={`translate(${cx},446)`}>
              <DreamCard type={i} seal={seal} />
            </g>
          );
        })}
        <text x={1560} y={272} textAnchor="middle" fill={C.muted} fontSize={26} fontFamily={FONT_BODY}>
          自动盖章机
        </text>
      </svg>
      <Stamp text="质检员离岗" x={720} y={140} at={beat(1) + 30} color={V.seal} size={52} rot={-5} />
      <Stamp text="半成品照过" x={1150} y={150} at={beat(2) + 8} color={V.amyDeep} size={48} rot={4} />
      <Chip text="挑毛病的系统没开机" x={300} y={830} at={beat(2) + 50} accent={V.seal} />
    </FadeIn>
  );
};

/* ---------- 剧组值班行 ---------- */
const CrewRow = ({x, y, name, role, ok, delay}: {
  x: number; y: number; name: string; role: string; ok: boolean; delay: number;
}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f - delay, fps, config: {damping: 13, stiffness: 150}});
  const o = interpolate(f, [delay, delay + 10], [0, 1], {extrapolateLeft: 'clamp'});
  if (o <= 0) return null;
  return (
    <g transform={`translate(${x},${y}) scale(${Math.max(0.001, s)})`} opacity={o}>
      <circle r={20} fill={ok ? '#1e3a2e' : '#3a1e28'} stroke={ok ? V.ok : V.seal} strokeWidth={4} />
      {ok
        ? <path d="M -8 0 L -2 7 L 10 -7" stroke={V.ok} strokeWidth={5} fill="none" strokeLinecap="round" />
        : <g stroke={V.seal} strokeWidth={5} strokeLinecap="round">
            <line x1={-7} y1={-7} x2={7} y2={7} /><line x1={7} y1={-7} x2={-7} y2={7} />
          </g>}
      <text x={40} y={6} fill={C.text} fontSize={34} fontFamily={FONT_HEAD} fontWeight={700}>{name}</text>
      <text x={40} y={44} fill={C.muted} fontSize={24} fontFamily={FONT_BODY}>{role}</text>
      <text x={620} y={10} textAnchor="end" fill={ok ? V.ok : V.seal} fontSize={30}
        fontFamily={FONT_HEAD} fontWeight={700}>{ok ? '在岗' : '缺席'}</text>
    </g>
  );
};

/* ============ S2 画面高清：片场剧组 编剧剪辑缺席 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);

  return (
    <FadeIn>
      <Backdrop top={V.bg2} bottom={V.bg2} />
      <SceneTag index={2} name={sec.name} accent={V.on} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：成片画面·走廊 */}
        <rect x={140} y={240} width={880} height={560} rx={22} fill={V.panel2} stroke={V.cream} strokeWidth={6} />
        <g>
          {/* 走廊透视 */}
          <path d="M 170 270 L 470 460 L 470 560 L 170 770 Z" fill={V.wall} />
          <path d="M 990 270 L 690 460 L 690 560 L 990 770 Z" fill={V.wallD} />
          <path d="M 170 270 L 990 270 L 690 460 L 470 460 Z" fill={V.ceil} />
          <path d="M 170 770 L 990 770 L 690 560 L 470 560 Z" fill={V.floor} />
          <rect x={470} y={460} width={220} height={100} fill={V.backwall} />
          {/* 尽头的门透出光 */}
          <rect x={540} y={472} width={80} height={88} fill={V.lamp} />
          {/* 吊灯 */}
          <line x1={580} y1={270} x2={580} y2={330} stroke={V.offHi} strokeWidth={6} />
          <circle cx={580} cy={346} r={24} fill={V.lamp} />
          {/* 墙面装饰线汇聚到灭点 */}
          <line x1={170} y1={400} x2={470} y2={492} stroke={V.wallD} strokeWidth={6} />
          <line x1={990} y1={400} x2={690} y2={492} stroke={V.wall} strokeWidth={6} />
          <line x1={170} y1={640} x2={470} y2={536} stroke={V.wallD} strokeWidth={6} />
          <line x1={990} y1={640} x2={690} y2={536} stroke={V.wall} strokeWidth={6} />
        </g>
        {/* 杏仁核往下灌情绪 */}
        <g transform={`translate(880,286)`}>
          <ellipse rx={66} ry={46} fill={V.amy} />
          <ellipse cx={-14} cy={-12} rx={34} ry={22} fill="#ff9a8e" />
          <text x={0} y={8} textAnchor="middle" fill="#fff" fontSize={26} fontWeight={900}
            fontFamily={FONT_HEAD}>杏仁核</text>
        </g>
        {clamp01((f - beat(0) - 60) / 20) > 0 && [0, 1, 2, 3].map(i => {
          const dy = ((f * 3.4 + i * 78) % 300);
          return <circle key={i} cx={852 + (i % 2) * 46} cy={330 + dy} r={10} fill={V.amy}
            opacity={clamp01((f - beat(0) - 60) / 20) * (1 - dy / 320)} />;
        })}
        <text x={170} y={838} fill={V.cream} fontSize={28} fontFamily={FONT_HEAD}>
          成片画面：颜色质感俱全 · 异常真切
        </text>

        {/* 右：剧组值班表 */}
        <rect x={1080} y={240} width={740} height={560} rx={26} fill={V.panel} />
        <text x={1120} y={300} fill={V.cream} fontSize={38} fontFamily={FONT_HEAD} fontWeight={700}>
          梦的剧组 · 谁在岗
        </text>
        <CrewRow x={1140} y={380} name="摄影 · 视觉皮层" role="放电比清醒时还自由" ok delay={beat(0) + 30} />
        <CrewRow x={1140} y={470} name="灯光" role="颜色光线质感全拉满" ok delay={beat(0) + 48} />
        <CrewRow x={1140} y={560} name="配乐 · 杏仁核" role="情绪直接灌进场景" ok delay={beat(0) + 66} />
        <CrewRow x={1140} y={656} name="编剧" role="没有时间线没有因果链" ok={false} delay={beat(1) + 10} />
        <CrewRow x={1140} y={746} name="剪辑" role="谁先谁后说不清" ok={false} delay={beat(1) + 30} />
      </svg>
      <Stamp text="不是像素问题" x={160} y={140} at={beat(1) + 60} color={V.onDeep} size={50} rot={-4} />
      <Stamp text="是导演问题" x={620} y={120} at={beat(1) + 90} color={V.seal} size={52} rot={3} />
      <Chip text="情绪配乐火力全开" x={1170} y={140} at={beat(0) + 100} accent={V.amy} />
    </FadeIn>
  );
};

/* ---------- 记忆碎片小卡 ---------- */
const FRAG_KIND = ['sun', 'face', 'tree', 'bars', 'house', 'fish'] as const;
const Frag = ({kind, t, from, to, seed}: {
  kind: string; t: number; from: [number, number]; to: [number, number]; seed: number;
}) => {
  if (t <= 0) return null;
  const k = clamp01(t);
  const x = from[0] + (to[0] - from[0]) * k;
  const y = from[1] + (to[1] - from[1]) * k - Math.sin(k * Math.PI) * 150;
  const rot = (1 - k) * (seed % 2 ? 22 : -22) - 3;
  return (
    <g transform={`translate(${x},${y}) rotate(${rot})`} opacity={0.4 + k * 0.6}>
      <rect x={-48} y={-36} width={96} height={72} rx={10} fill={V.paper} stroke={V.paperSh} strokeWidth={4} />
      {kind === 'sun' && <g><circle cy={-4} r={16} fill={V.on} /><rect x={-2} y={-32} width={4} height={10} fill={V.on} /></g>}
      {kind === 'face' && <g><circle cy={-2} r={20} fill={C.skin} /><circle cx={-7} cy={-6} r={3} fill={V.ink} /><circle cx={7} cy={-6} r={3} fill={V.ink} /><path d="M -8 6 Q 0 13 8 6" stroke={V.ink} strokeWidth={3} fill="none" /></g>}
      {kind === 'tree' && <g><path d="M 0 -26 L 16 4 L -16 4 Z" fill={V.hippoDeep} /><rect x={-4} y={4} width={8} height={16} fill="#8a6a4a" /></g>}
      {kind === 'bars' && <g>{[0, 1, 2].map(i => <rect key={i} x={-26 + i * 20} y={10 - i * 14} width={12} height={18 + i * 14} fill={V.rem} />)}</g>}
      {kind === 'house' && <g><rect x={-16} y={-4} width={32} height={24} fill={V.seal} /><path d="M -22 -4 L 0 -24 L 22 -4 Z" fill={V.amyDeep} /></g>}
      {kind === 'fish' && <g><ellipse cy={-2} rx={20} ry={12} fill={V.rem} /><path d="M 18 -2 L 32 -12 L 32 8 Z" fill={V.rem} /><circle cx={-10} cy={-5} r={3} fill={V.ink} /></g>}
    </g>
  );
};

/* ============ S3 碎片拼信：海马重播 + 脑干随机信号 → 皮层缝故事 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);

  const slots: [number, number][] = [[1210, 368], [1424, 389], [1278, 466], [1539, 523], [1345, 603], [1588, 660]];
  const froms: [number, number][] = [[360, 420], [290, 660]];
  const scraps = [
    [-260, -150, 150], [-95, -150, 120], [40, -150, 170],
    [-240, -80, 110], [-110, -80, 180], [90, -80, 100],
    [-250, -10, 170], [-60, -10, 130], [100, -10, 150],
    [-230, 60, 140], [-70, 60, 160], [110, 60, 120],
    [-250, 130, 160], [-60, 130, 110],
  ];
  const scrapCols = ['#c9d6f2', '#f2c9c9', '#d9e8c9', '#f0e0b8'];

  return (
    <FadeIn>
      <Backdrop top={V.bg3} bottom={V.bg3} />
      <SceneTag index={3} name={sec.name} accent={V.hippo} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：海马（重播白天片段） */}
        <g transform={`translate(360,420)`}>
          <ellipse rx={130} ry={86} fill={V.hippo} />
          <path d="M -80 -30 C -30 -60 40 -60 80 -20" fill="none" stroke={V.hippoDeep} strokeWidth={12} strokeLinecap="round" />
          <path d="M -70 20 C -20 -10 40 -10 78 26" fill="none" stroke={V.hippoDeep} strokeWidth={12} strokeLinecap="round" />
          <text x={0} y={12} textAnchor="middle" fill={V.ink} fontSize={34} fontWeight={900} fontFamily={FONT_HEAD}>海马</text>
        </g>
        <text x={360} y={536} textAnchor="middle" fill={C.muted} fontSize={26} fontFamily={FONT_BODY}>
          重播白天的记忆片段
        </text>

        {/* 左下：脑干（随机冲动） */}
        <g transform={`translate(280,660)`}>
          <rect x={-60} y={-90} width={120} height={180} rx={40} fill={V.stem} />
          <rect x={-30} y={-60} width={60} height={120} rx={24} fill="#b49ae8" />
          <text x={0} y={132} textAnchor="middle" fill={C.muted} fontSize={26} fontFamily={FONT_BODY}>脑干</text>
        </g>
        <text x={280} y={836} textAnchor="middle" fill={C.muted} fontSize={26} fontFamily={FONT_BODY}>
          随机神经冲动
        </text>
        {/* 闪电脉冲 */}
        {[0, 1].map(i => (
          <path key={i} d={`M ${360 + i * 56} ${600 - i * 30} l 22 26 l -14 4 l 26 30`}
            fill="none" stroke={V.on} strokeWidth={7} strokeLinecap="round"
            opacity={Math.sin(f * 0.55 + i * 2.1) > 0.2 ? 0.95 : 0.12} />
        ))}

        {/* 右：皮层缝出来的信 */}
        <g transform={`translate(1400,510) rotate(-3)`}>
          <rect x={-330} y={-290} width={660} height={580} rx={14} fill={V.paper} />
          <rect x={-330} y={-290} width={660} height={580} rx={14} fill="none" stroke={V.paperSh} strokeWidth={6} />
          {/* 缝线虚框 */}
          <rect x={-308} y={-268} width={616} height={536} rx={10} fill="none"
            stroke={V.amyDeep} strokeWidth={4} strokeDasharray="14 12" opacity={0.7} />
          <rect x={-280} y={-252} width={240} height={52} rx={8} fill={V.panel} />
          <text x={-160} y={-216} textAnchor="middle" fill={V.cream} fontSize={30} fontFamily={FONT_HEAD}>
            皮层缝的故事
          </text>
          {/* 剪报条 */}
          {scraps.map((s2, i) => (
            <rect key={i} x={s2[0]} y={s2[1]} width={s2[2]} height={44} rx={6}
              fill={scrapCols[i % 4]} transform={`rotate(${rand(i, 3) * 6 - 3} ${s2[0] + s2[2] / 2} ${s2[1] + 22})`}
              opacity={clamp01((f - beat(1) - i * 4) / 20)} />
          ))}
        </g>

        {/* 飞入的碎片 */}
        {slots.map((sl, i) => (
          <Frag key={i} kind={FRAG_KIND[i]} seed={i}
            from={froms[i % 2]} to={sl}
            t={(f - beat(0) - 26 - i * 20) / 52} />
        ))}

        {/* 引线标注 */}
        <text x={150} y={268} fill={V.cream} fontSize={30} fontFamily={FONT_HEAD} fontWeight={700}>
          原料：碎片 + 随机信号
        </text>
      </svg>
      <Chip text="激活合成假说" x={150} y={130} at={beat(1) + 16} accent={V.hippo} />
      <Stamp text="就地取材缝故事" x={620} y={140} at={beat(0) + 120} color={V.hippoDeep} size={48} rot={-4} />
      <Stamp text="字都真·连起来不通" x={1140} y={838} at={beat(1) + 60} color={V.seal} size={46} rot={3} />
    </FadeIn>
  );
};

/* ---------- 迷糊/清晰小景（窗景） ---------- */
const DreamView = ({x, y, w, h, crisp}: {x: number; y: number; w: number; h: number; crisp: boolean}) => {
  const inner = (key: number, dx: number, dy: number, o: number, dash?: string) => (
    <g key={key} transform={`translate(${dx},${dy})`} opacity={o}>
      <circle cx={x + w * 0.72} cy={y + h * 0.34} r={26} fill={crisp ? V.lamp : '#8a8fae'}
        strokeDasharray={dash} stroke={crisp ? 'none' : '#8a8fae'} strokeWidth={3} />
      <path d={`M ${x + 20} ${y + h - 24} Q ${x + w * 0.35} ${y + h * 0.42} ${x + w * 0.62} ${y + h - 24}
        Q ${x + w * 0.85} ${y + h * 0.55} ${x + w - 16} ${y + h - 24}`}
        fill={crisp ? V.wall : 'none'} stroke={crisp ? 'none' : '#8a8fae'}
        strokeWidth={4} strokeDasharray={dash} />
      <circle cx={x + w * 0.3} cy={y + h * 0.45} r={16} fill={crisp ? C.skin : '#8a8fae'} />
      <rect x={x + w * 0.3 - 14} y={y + h * 0.45 + 16} width={28} height={30} rx={10}
        fill={crisp ? V.rem : '#8a8fae'} />
    </g>
  );
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={14}
        fill={crisp ? '#2b3f66' : '#262b47'}
        stroke={crisp ? V.cream : '#565c85'} strokeWidth={crisp ? 5 : 4}
        strokeDasharray={crisp ? 'none' : '12 10'} />
      {crisp ? inner(0, 0, 0, 1) : <>{inner(0, 0, 0, 0.85, '8 7')}{inner(1, 9, 6, 0.3, '8 7')}</>}
    </g>
  );
};

/* ============ S4 存档淡 + 清醒梦反证 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);

  const level = 1 - clamp01((f - beat(0) - 40) / 60) * 0.92;    // 去甲肾上腺素 1→0.08
  const ang = -180 + level * 180;
  const popLucid = spring({frame: f - beat(1) - 6, fps, config: {damping: 12, stiffness: 140}});
  const fadeStart = beat(0) + 90;

  return (
    <FadeIn>
      <Backdrop top={V.bg4} bottom={V.bg4} />
      <SceneTag index={4} name={sec.name} accent={V.on} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：记忆写入工序 */}
        <rect x={110} y={220} width={620} height={630} rx={26} fill={V.panel} />
        <text x={150} y={282} fill={V.cream} fontSize={36} fontFamily={FONT_HEAD} fontWeight={700}>
          记忆写入工序
        </text>
        {/* 仪表 */}
        <g transform={`translate(300,540)`}>
          <path d="M -140 0 A 140 140 0 0 1 140 0" fill="none" stroke={V.offHi} strokeWidth={18} strokeLinecap="round" />
          <path d="M 100 -98 A 140 140 0 0 1 140 0" fill="none" stroke={V.seal} strokeWidth={18} strokeLinecap="round" />
          {[-180, -135, -90, -45, 0].map(a => (
            <line key={a} x1={Math.cos(a * Math.PI / 180) * 118} y1={Math.sin(a * Math.PI / 180) * 118}
              x2={Math.cos(a * Math.PI / 180) * 132} y2={Math.sin(a * Math.PI / 180) * 132}
              stroke={V.cream} strokeWidth={5} />
          ))}
          <line x1={0} y1={0} x2={Math.cos(ang * Math.PI / 180) * 105} y2={Math.sin(ang * Math.PI / 180) * 105}
            stroke={V.cream} strokeWidth={9} strokeLinecap="round" />
          <circle r={14} fill={V.cream} />
        </g>
        <text x={300} y={592} textAnchor="middle" fill={C.text} fontSize={28} fontFamily={FONT_HEAD}>
          去甲肾上腺素
        </text>
        {level < 0.2 && (
          <text x={300} y={636} textAnchor="middle" fill={V.seal} fontSize={30}
            fontFamily={FONT_HEAD} fontWeight={900}>全天最低</text>
        )}
        {/* 变淡的笔记 */}
        <g transform={`translate(545,380) rotate(-5)`}>
          <rect x={-70} y={-110} width={150} height={220} rx={10} fill={V.paper} />
          {[0, 1, 2, 3].map(i => {
            const o = 1 - clamp01((f - fadeStart - i * 14) / 34) * 0.8;
            return <rect key={i} x={-52} y={-84 + i * 46} width={110 - (i % 2) * 30} height={18}
              rx={6} fill={V.ink} opacity={o} />;
          })}
          <text x={4} y={136} textAnchor="middle" fill={C.muted} fontSize={24} fontFamily={FONT_BODY}>梦的笔迹</text>
        </g>
        {/* 时钟：三分钟 */}
        <g transform={`translate(185,758)`}>
          <circle r={42} fill={V.panel2} stroke={V.cream} strokeWidth={5} />
          <line x1={0} y1={0} x2={0} y2={-28} stroke={V.cream} strokeWidth={5} strokeLinecap="round" />
          <line x1={0} y1={0} x2={20} y2={0} stroke={V.seal} strokeWidth={5} strokeLinecap="round" />
        </g>
        <text x={245} y={768} fill={V.cream} fontSize={26} fontFamily={FONT_BODY}>
          醒来三分钟不记，痕迹就散
        </text>

        {/* 右上：普通梦 */}
        <rect x={790} y={220} width={1020} height={300} rx={24} fill={V.panel} />
        <text x={830} y={282} fill={V.cream} fontSize={34} fontFamily={FONT_HEAD} fontWeight={700}>普通梦</text>
        <DreamView x={830} y={310} w={430} h={170} crisp={false} />
        <Brain x={1480} y={390} s={0.52} lit={{vis: true, amy: true, hip: true, pre: false}} />
        <text x={1580} y={380} fill={C.muted} fontSize={28} fontFamily={FONT_HEAD}>前额叶 · 熄灯</text>
        <text x={1580} y={428} fill={C.muted} fontSize={28} fontFamily={FONT_HEAD}>场景漂移 · 结构散</text>

        {/* 右下：清醒梦（弹出） */}
        {popLucid > 0 && (
          <g transform={`translate(1300,700) scale(${Math.max(0.001, popLucid)}) translate(-1300,-700)`}
            opacity={clamp01((f - beat(1)) / 12)}>
            <rect x={790} y={555} width={1020} height={290} rx={24} fill={V.panel}
              stroke={V.on} strokeWidth={5} />
            <text x={830} y={617} fill={V.on} fontSize={34} fontFamily={FONT_HEAD} fontWeight={700}>清醒梦</text>
            <DreamView x={830} y={645} w={430} h={160} crisp />
            <Brain x={1480} y={720} s={0.52} lit={{vis: true, amy: true, hip: true, pre: true}} />
            <text x={1580} y={710} fill={V.on} fontSize={28} fontFamily={FONT_HEAD} fontWeight={700}>前额叶 · 回岗</text>
            <text x={1580} y={758} fill={C.text} fontSize={28} fontFamily={FONT_HEAD}>场景稳定 · 逻辑在线</text>
          </g>
        )}
      </svg>
      <Stamp text="笔迹本来就淡" x={170} y={140} at={beat(0) + 60} color={V.offHi} size={48} rot={-4} />
      <Stamp text="回岗即高清" x={850} y={140} at={beat(1) + 50} color={V.onDeep} size={52} rot={3} />
      <Chip text="质检员在不在岗" x={1350} y={140} at={beat(1) + 100} accent={V.on} />
    </FadeIn>
  );
};

/* ============ 结尾卡 ============ */
const EndCard = () => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 14}});
  const pop = spring({frame: f - 4, fps, config: {damping: 10, stiffness: 150}});
  return (
    <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
      <Backdrop top={V.bg0} bottom={V.bg0} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={18} from={0} dur={20} />
        <Moon x={1560} y={230} scale={0.95} />
        {/* 小床 + 熟睡小人 */}
        <g transform={`translate(960,760) scale(${Math.max(0.001, pop)})`}>
          <rect x={-300} y={0} width={600} height={56} rx={16} fill={V.belt} />
          <rect x={-300} y={-86} width={26} height={142} rx={10} fill={V.belt} />
          <rect x={274} y={-60} width={26} height={116} rx={10} fill={V.belt} />
          <rect x={-262} y={-56} width={130} height={56} rx={18} fill="#eef3ff" />
          <circle cx={-190} cy={-66} r={34} fill={C.skin} />
          <path d="M -222 -78 Q -190 -104 -158 -78" fill={V.ink} />
          <rect x={-150} y={-84} width={400} height={90} rx={26} fill={V.rem} />
          <rect x={-150} y={-84} width={400} height={30} rx={15} fill="#7fd4ff" />
          <Zzz x={120} y={-110} />
        </g>
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -320}}>
        <div style={{fontSize: 66, fontWeight: 800, color: C.text}}>梦不缺画质，缺质检员</div>
        <div style={{fontSize: 38, color: C.muted, marginTop: 22, letterSpacing: 6}}>清醒梦 = 质检员回岗 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E223 = ({data = tl}: {data?: any}) => {
  const {fps} = useVideoConfig();
  return (
    <AbsoluteFill style={{background: V.bg0}}>
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
