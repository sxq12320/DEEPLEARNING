import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01, rand} from '../lib/style';
import {Backdrop, Stamp, SceneTag, Chip, Subtitles, ProgressBar, FadeIn, Moon, Stars, Sun} from '../lib/atoms';
import tl from '../timeline_231.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：午夜蓝 × 腺苷紫 × 节律青 × 皮质醇橙 ============ */
const P = {
  bg: '#15203c',
  bgDeep: '#0d1527',
  bgLite: '#1c2a4e',
  panel: '#101a34',
  card: '#27355e',
  paper: '#f2ecdc',
  ink: '#141b32',
  cream: '#f4eedd',
  muted: '#9aa8cf',
  vio: '#b48cf2',
  vioD: '#8f6ad8',
  cyan: '#57d8e8',
  cyanD: '#35aec4',
  amber: '#ffb85c',
  amberD: '#e09a3e',
  coral: '#ff7a6e',
  coralD: '#d8564c',
  teal: '#5fe3b3',
  sky: '#7ab8ff',
};

/* ---------- 飘z ---------- */
const Zzz = ({x, y, k = 1, col = P.cyan}: {x: number; y: number; k?: number; col?: string}) => {
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

/* ---------- 数字时刻牌 ---------- */
const TimeBadge = ({x, y, t, col, s = 1, w = 164}: {x: number; y: number; t: string; col: string; s?: number; w?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-w / 2} y={-32} width={w} height={64} rx={16} fill={P.panel} stroke={col} strokeWidth={4} />
    <text x={0} y={14} textAnchor="middle" fontSize={36} fill={col}
      fontFamily={FONT_HEAD} fontWeight={800}>{t}</text>
  </g>
);

/* ---------- 伏案小人：nod 打盹 / alert 精神 / crash 趴桌 ---------- */
const DeskPerson = ({x, y, s = 1, mode = 'nod'}: {x: number; y: number; s?: number; mode?: 'nod' | 'alert' | 'crash'}) => {
  const f = useCurrentFrame();
  const nod = mode === 'nod' ? Math.abs(Math.sin(f / 10)) : 0;
  const headDy = mode === 'crash' ? 88 : nod * 14;
  const headRot = mode === 'crash' ? 70 : nod * 24;
  const headDx = mode === 'crash' ? 42 : 0;
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      <rect x={-72} y={-48} width={144} height={140} rx={22} fill={P.card} opacity={0.55} />
      <rect x={-54} y={-60} width={108} height={106} rx={44} fill={P.sky} />
      <rect x={-152} y={24} width={304} height={18} rx={9} fill={P.paper} opacity={0.92} />
      <rect x={-134} y={42} width={14} height={66} fill={P.card} />
      <rect x={120} y={42} width={14} height={66} fill={P.card} />
      <g transform={`translate(${headDx},${-96 + headDy}) rotate(${headRot})`}>
        <circle r={34} fill={C.skin} />
        <path d="M -33 -8 Q 0 -46 33 -8 L 33 2 Q 0 -24 -33 2 Z" fill="#3a2c4e" />
        {mode === 'alert' ? (
          <g>
            <circle cx={-12} cy={0} r={7.5} fill="#fff" />
            <circle cx={14} cy={0} r={7.5} fill="#fff" />
            <circle cx={-11} cy={1} r={3.4} fill={P.ink} />
            <circle cx={15} cy={1} r={3.4} fill={P.ink} />
            <path d="M -20 -15 q 7 -5 13 -2 M 8 -17 q 7 -3 13 2" stroke={P.ink}
              strokeWidth={3} fill="none" strokeLinecap="round" />
            <path d="M -8 16 q 9 9 18 0" stroke={P.ink} strokeWidth={3.4} fill="none" strokeLinecap="round" />
          </g>
        ) : (
          <g stroke={P.ink} strokeWidth={3.2} strokeLinecap="round" fill="none">
            <path d="M -18 0 q 5 7 11 0" />
            <path d="M 7 0 q 5 7 11 0" />
            <path d="M -7 18 q 6 5 12 0" />
          </g>
        )}
      </g>
    </g>
  );
};

/* ---------- 腺苷分子（六边形） ---------- */
const AdHex = ({x, y, r = 17, op = 1}: {x: number; y: number; r?: number; op?: number}) => (
  <g transform={`translate(${x},${y})`} opacity={op}>
    <polygon
      points={[0, 1, 2, 3, 4, 5].map((i) => {
        const a = (i * 60 - 90) * Math.PI / 180;
        return `${(r * Math.cos(a)).toFixed(1)},${(r * Math.sin(a)).toFixed(1)}`;
      }).join(' ')}
      fill={P.vio} stroke={P.vioD} strokeWidth={3} />
    <circle cx={-r * 0.2} cy={-r * 0.2} r={r * 0.28} fill={P.vioD} opacity={0.85} />
  </g>
);

/* ---------- 咖啡因分子（菱形） ---------- */
const CafSq = ({x, y, r = 15, op = 1}: {x: number; y: number; r?: number; op?: number}) => (
  <g transform={`translate(${x},${y}) rotate(45)`} opacity={op}>
    <rect x={-r} y={-r} width={r * 2} height={r * 2} rx={7} fill={P.amber} stroke={P.amberD} strokeWidth={3} />
  </g>
);

/* ---------- 绳子上的"你" ---------- */
const You = ({x, y, s = 1, strain = 0}: {x: number; y: number; s?: number; strain?: number}) => {
  const f = useCurrentFrame();
  const wob = Math.sin(f * 0.6) * 2 + strain * Math.sin(f * 1.7) * 5;
  return (
    <g transform={`translate(${x + wob},${y}) scale(${s})`}>
      <rect x={-36} y={-6} width={72} height={76} rx={32} fill={P.cream} />
      <rect x={-36} y={-6} width={72} height={22} rx={11} fill={P.coral} opacity={0.85} />
      <g transform="translate(0,-44)">
        <circle r={27} fill={C.skin} />
        <path d="M -26 -7 Q 0 -36 26 -7 L 26 1 Q 0 -19 -26 1 Z" fill="#3a2c4e" />
        {strain > 0.5 ? (
          <g>
            <path d="M -15 -6 a 6 6 0 1 1 6 6 a 3.2 3.2 0 1 0 -3.2 -3.2" stroke={P.ink} strokeWidth={2.6} fill="none" />
            <path d="M 8 -6 a 6 6 0 1 1 6 6 a 3.2 3.2 0 1 0 -3.2 -3.2" stroke={P.ink} strokeWidth={2.6} fill="none" />
            <ellipse cx={1} cy={15} rx={6} ry={8} fill={P.ink} />
            <path d="M 30 -34 q 7 9 0 14 q -7 -5 0 -14" fill={P.cyan} />
          </g>
        ) : (
          <g>
            <circle cx={-9} cy={-2} r={3.8} fill={P.ink} />
            <circle cx={10} cy={-2} r={3.8} fill={P.ink} />
            <path d="M -7 13 q 7 7 15 0" stroke={P.ink} strokeWidth={3.2} fill="none" strokeLinecap="round" />
          </g>
        )}
      </g>
    </g>
  );
};

/* ---------- 拔河力士（dir=+1 朝右拉 / -1 朝左拉） ---------- */
const Puller = ({x, y, s = 1, col, dark, dir = 1, strain = 0}: {
  x: number; y: number; s?: number; col: string; dark: string; dir?: number; strain?: number;
}) => {
  const f = useCurrentFrame();
  const lean = -dir * (10 + strain * 15);
  const bob = Math.sin(f * 0.5 + x * 0.1) * 3;
  return (
    <g transform={`translate(${x},${y + bob})`}>
      <g transform={`scale(${dir * s},${s}) rotate(${lean})`}>
        <line x1={-12} y1={46} x2={-36} y2={82} stroke={dark} strokeWidth={11} strokeLinecap="round" />
        <line x1={14} y1={46} x2={38} y2={82} stroke={dark} strokeWidth={11} strokeLinecap="round" />
        <line x1={34} y1={-4} x2={92} y2={-34} stroke={dark} strokeWidth={11} strokeLinecap="round" />
        <line x1={34} y1={8} x2={92} y2={-2} stroke={dark} strokeWidth={11} strokeLinecap="round" />
        <circle cx={94} cy={-34} r={9} fill={dark} />
        <circle cx={94} cy={-2} r={9} fill={dark} />
        <rect x={-48} y={-66} width={96} height={122} rx={44} fill={col} />
        <rect x={-48} y={-66} width={96} height={32} rx={16} fill={dark} opacity={0.35} />
        <circle cx={4} cy={-36} r={5.5} fill={P.ink} />
        <circle cx={27} cy={-36} r={5.5} fill={P.ink} />
        {strain > 0.5 ? (
          <path d="M 6 -13 q 10 -7 19 0" stroke={P.ink} strokeWidth={3.4} fill="none" strokeLinecap="round" />
        ) : (
          <path d="M 6 -11 q 10 8 19 0" stroke={P.ink} strokeWidth={3.4} fill="none" strokeLinecap="round" />
        )}
      </g>
    </g>
  );
};

/* ---------- 小叉 / 小勾 ---------- */
const MarkX = ({x, y, s = 1, col = P.coral}: {x: number; y: number; s?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} stroke={col} strokeWidth={6} strokeLinecap="round">
    <line x1={-11} y1={-11} x2={11} y2={11} /><line x1={11} y1={-11} x2={-11} y2={11} />
  </g>
);
const MarkV = ({x, y, s = 1, col = P.teal}: {x: number; y: number; s?: number; col?: string}) => (
  <path d="M -12 0 L -3 10 L 14 -10" transform={`translate(${x},${y}) scale(${s})`}
    stroke={col} strokeWidth={6} fill="none" strokeLinecap="round" strokeLinejoin="round" />
);

/* ============ 标题卡 ============ */
const TitleCard = ({data}: {data: any}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 13, mass: 0.7}});
  const o = interpolate(f, [0, 8], [0, 1]);
  const pA = Math.max(0.001, spring({frame: f - 4, fps, config: {damping: 10, stiffness: 150}}));
  const pB = Math.max(0.001, spring({frame: f - 14, fps, config: {damping: 10, stiffness: 150}}));
  return (
    <AbsoluteFill>
      <Backdrop top={P.bg} bottom={P.bg} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={22} from={0} dur={24} />
        <Moon x={240} y={150} scale={0.7} />
        {/* 中央：两套系统拔河图标 */}
        <g transform={`translate(960,330) scale(${pA}) translate(-960,-330)`} opacity={clamp01(pA)}>
          {/* 左：睡眠压力紫月 */}
          <g transform="translate(700,300)">
            <circle r={58} fill={P.vio} />
            <circle cx={-20} cy={-14} r={46} fill={P.bg} opacity={0.9} />
            <circle cx={16} cy={18} r={7} fill={P.vioD} />
            <text x={0} y={96} textAnchor="middle" fontSize={26} fill={P.vio}
              fontFamily={FONT_HEAD} fontWeight={800}>S·醒了多久</text>
          </g>
          {/* 右：节律青太阳钟 */}
          <g transform="translate(1220,300)">
            {Array.from({length: 8}).map((_, i) => (
              <rect key={i} x={-5} y={-88} width={10} height={24} rx={5} fill={P.cyan}
                transform={`rotate(${i * 45})`} />
            ))}
            <circle r={56} fill={P.cyan} />
            <circle r={44} fill={P.panel} />
            <line x1={0} y1={0} x2={0} y2={-28} stroke={P.cyan} strokeWidth={5} strokeLinecap="round" />
            <line x1={0} y1={0} x2={19} y2={8} stroke={P.cyan} strokeWidth={5} strokeLinecap="round" />
            <text x={0} y={96} textAnchor="middle" fontSize={26} fill={P.cyan}
              fontFamily={FONT_HEAD} fontWeight={800}>C·现在几点</text>
          </g>
          {/* 绳子 + 中间的人 */}
          <path d="M 756 300 Q 960 330 1164 300" stroke={P.cream} strokeWidth={8}
            fill="none" strokeLinecap="round" />
          <You x={960} y={258} s={0.75} />
          <text x={960} y={158} textAnchor="middle" fontSize={52} fill={P.coral}
            fontFamily={FONT_HEAD} fontWeight={900}>?</text>
        </g>
        {/* 两个时刻签 */}
        <g transform={`translate(700,540) scale(${pB})`} opacity={clamp01(pB)}>
          <TimeBadge x={0} y={0} t="4:00 困到极限" col={P.vio} s={1} w={300} />
        </g>
        <g transform={`translate(1220,540) scale(${pB})`} opacity={clamp01(pB)}>
          <TimeBadge x={0} y={0} t="7:00 莫名清醒" col={P.cyan} s={1} w={300} />
        </g>
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 430}}>
          <div style={{color: P.cyan, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 24}}>
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

/* ============ S0 清晨的清醒是假象：4点 vs 7点 对照 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const popL = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const popR = Math.max(0.001, spring({frame: f - 45, fps, config: {damping: 11, stiffness: 120}}));
  const sysK = clamp01((f - b1 - 10) / 24);
  const sysPop = [0, 1].map((i) => Math.max(0.001, spring({frame: f - b1 - 15 - i * 10, fps, config: {damping: 11, stiffness: 130}})));

  const panel = (x0: number, k: number, kids: React.ReactNode) => (
    <g transform={`translate(${x0},150) scale(${k}) translate(${-x0},-150)`} opacity={clamp01(k * 2)}>
      <rect x={x0} y={150} width={760} height={500} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
      {kids}
    </g>
  );

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={0} name={sec.name} accent={P.cyan} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={12} from={0} dur={26} />
        {/* 左：凌晨4点 困到极限 */}
        {panel(120, popL, (
          <g>
            <text x={156} y={206} fill={P.muted} fontSize={24} fontFamily={FONT_HEAD}>同一个你 · 没合眼</text>
            <TimeBadge x={265} y={265} t="凌晨 4:00" col={P.vio} />
            <Moon x={795} y={230} scale={0.35} opacity={0.9} />
            <DeskPerson x={480} y={400} s={1.05} mode="nod" />
            <Zzz x={640} y={340} k={0.85} col={P.vio} />
            <text x={500} y={586} textAnchor="middle" fill={P.vio} fontSize={30}
              fontFamily={FONT_HEAD} fontWeight={800}>眼皮打架 · 困到深渊</text>
          </g>
        ))}
        {/* 右：早7点 莫名清醒 */}
        {panel(1040, popR, (
          <g>
            <text x={1076} y={206} fill={P.muted} fontSize={24} fontFamily={FONT_HEAD}>三小时后 · 还是没睡</text>
            <TimeBadge x={1185} y={265} t="早上 7:00" col={P.cyan} />
            <Sun x={1715} y={230} r={34} />
            <DeskPerson x={1400} y={400} s={1.05} mode="alert" />
            <text x={1612} y={330} fontSize={46} fill={P.cyan} fontFamily={FONT_HEAD} fontWeight={900}>?</text>
            <text x={1420} y={586} textAnchor="middle" fill={P.cyan} fontSize={30}
              fontFamily={FONT_HEAD} fontWeight={800}>反而清醒 · 觉得还能扛</text>
          </g>
        ))}
        {/* 中间反差徽章 */}
        <g transform="translate(960,380)">
          <circle r={38} fill={P.coral} />
          <text y={13} textAnchor="middle" fontSize={34} fill={P.cream}
            fontFamily={FONT_HEAD} fontWeight={900}>≠</text>
        </g>
        {/* 底条：两套系统揭示（cue1） */}
        {sysK > 0 && (
          <g opacity={sysK} transform={`translate(0,${(1 - sysK) * 60})`}>
            <rect x={120} y={692} width={1680} height={172} rx={20} fill={P.panel} stroke={P.cyan} strokeWidth={4} />
            <text x={156} y={742} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
              不是适应了熬夜 —— 是两套睡意系统错位了</text>
            <g transform={`translate(640,802) scale(${sysPop[0]})`} opacity={clamp01(sysPop[0])}>
              <rect x={-300} y={-42} width={600} height={84} rx={20} fill={P.bgDeep} stroke={P.vio} strokeWidth={4} />
              <circle cx={-250} cy={0} r={17} fill={P.vio} />
              <text x={-215} y={-6} fill={P.cream} fontSize={28} fontFamily={FONT_HEAD} fontWeight={800}>系统一</text>
              <text x={-215} y={28} fill={P.vio} fontSize={25} fontFamily={FONT_HEAD}>管「你醒了多久」</text>
            </g>
            <text x={990} y={815} textAnchor="middle" fontSize={40} fill={P.coral}
              fontFamily={FONT_HEAD} fontWeight={900}>⚡错位</text>
            <g transform={`translate(1370,802) scale(${sysPop[1]})`} opacity={clamp01(sysPop[1])}>
              <rect x={-300} y={-42} width={600} height={84} rx={20} fill={P.bgDeep} stroke={P.cyan} strokeWidth={4} />
              <circle cx={-250} cy={0} r={17} fill={P.cyan} />
              <text x={-215} y={-6} fill={P.cream} fontSize={28} fontFamily={FONT_HEAD} fontWeight={800}>系统二</text>
              <text x={-215} y={28} fill={P.cyan} fontSize={25} fontFamily={FONT_HEAD}>管「现在几点」</text>
            </g>
          </g>
        )}
      </svg>
      <Chip text="回光返照式清醒" x={1420} y={70} at={beat(0) + 30} accent={P.cyan} />
      <Stamp text="错觉 · 不是恢复" x={700} y={90} at={b1 + 40} color={P.coralD} size={40} rot={-3} />
    </FadeIn>
  );
};

/* ============ S1 睡意是两股力拔河 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const fightK = clamp01((f - b1) / 22);
  const sLineK = clamp01((f - 40) / 90);
  const cWaveK = clamp01((f - 150) / 90);
  const ropeY = 470;
  const sag = 10 + Math.sin(f * 0.4) * 5 + fightK * (10 + Math.abs(Math.sin(f * 0.9)) * 16);
  const kx = fightK * Math.sin(f * 0.9) * 34;

  // S 单调上升小曲线
  const sPts: string[] = [];
  for (let i = 0; i <= 30; i++) {
    const u = i / 30;
    sPts.push(`${(210 + u * 360).toFixed(1)},${(828 - (0.2 + 0.7 * (1 - Math.exp(-u * 3))) * 120).toFixed(1)}`);
  }
  const sN = Math.max(2, Math.floor(30 * sLineK));
  // C 正弦小曲线
  const cPts: string[] = [];
  for (let i = 0; i <= 36; i++) {
    const u = i / 36;
    cPts.push(`${(1370 + u * 360).toFixed(1)},${(768 - Math.sin(u * Math.PI * 2 - Math.PI / 2) * 46).toFixed(1)}`);
  }
  const cN = Math.max(2, Math.floor(36 * cWaveK));

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={1} name={sec.name} accent={P.vio} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 标题 */}
        <text x={960} y={200} textAnchor="middle" fill={P.cream} fontSize={34}
          fontFamily={FONT_HEAD} fontWeight={800}>睡意 = 两股力在拔河</text>
        {/* 地面线 */}
        <line x1={180} y1={588} x2={1740} y2={588} stroke={P.card} strokeWidth={6} strokeLinecap="round" />
        {/* 左力士：S过程 */}
        <Puller x={400} y={492} s={1.05} col={P.vio} dark={P.vioD} dir={1} strain={fightK} />
        <text x={330} y={300} textAnchor="middle" fill={P.vio} fontSize={32}
          fontFamily={FONT_HEAD} fontWeight={900}>S 过程</text>
        <text x={330} y={338} textAnchor="middle" fill={P.muted} fontSize={23}
          fontFamily={FONT_HEAD}>睡眠压力 · 腺苷</text>
        {/* 右力士：C过程 */}
        <Puller x={1520} y={492} s={1.05} col={P.cyan} dark={P.cyanD} dir={-1} strain={fightK} />
        <text x={1590} y={300} textAnchor="middle" fill={P.cyan} fontSize={32}
          fontFamily={FONT_HEAD} fontWeight={900}>C 过程</text>
        <text x={1590} y={338} textAnchor="middle" fill={P.muted} fontSize={23}
          fontFamily={FONT_HEAD}>昼夜节律 · 准点派信号</text>
        {/* 绳子（带垂度） */}
        <path d={`M 494 ${ropeY - 34} Q 960 ${ropeY + sag} 1426 ${ropeY - 34}`}
          stroke={P.cream} strokeWidth={9} fill="none" strokeLinecap="round" />
        <path d={`M 494 ${ropeY - 34} Q 960 ${ropeY + sag} 1426 ${ropeY - 34}`}
          stroke={P.amberD} strokeWidth={3} fill="none" strokeDasharray="14 18" opacity={0.7} />
        {/* 中心结上的"你" */}
        <You x={960 + kx} y={ropeY + sag - 66} s={0.95} strain={fightK} />
        {/* 中央小旗 */}
        <line x1={960 + kx} y1={ropeY + sag - 4} x2={960 + kx} y2={ropeY + sag - 58}
          stroke={P.coral} strokeWidth={5} />
        <polygon points={`${960 + kx},${ropeY + sag - 58} ${960 + kx + 36},${ropeY + sag - 50} ${960 + kx},${ropeY + sag - 42}`}
          fill={P.coral} />
        {/* 打架时的冲击星 */}
        {fightK > 0.3 && (
          <g opacity={clamp01((fightK - 0.3) * 3)} stroke={P.amber} strokeWidth={5} strokeLinecap="round">
            {[0, 1, 2, 3].map((i) => {
              const a = (i * 90 + f * 3) * Math.PI / 180;
              return <line key={i}
                x1={960 + kx + Math.cos(a) * 86} y1={ropeY - 30 + Math.sin(a) * 56}
                x2={960 + kx + Math.cos(a) * 112} y2={ropeY - 30 + Math.sin(a) * 76} />;
            })}
          </g>
        )}
        {/* 左下小面板：S 曲线 */}
        <g>
          <rect x={160} y={648} width={460} height={220} rx={16} fill={P.panel} stroke={P.vio} strokeWidth={4} />
          <text x={186} y={692} fill={P.vio} fontSize={26} fontFamily={FONT_HEAD} fontWeight={800}>
            S · 醒得越久越高</text>
          <line x1={200} y1={830} x2={580} y2={830} stroke={P.muted} strokeWidth={3} />
          <line x1={200} y1={830} x2={200} y2={712} stroke={P.muted} strokeWidth={3} />
          <polyline points={sPts.slice(0, sN + 1).join(' ')} fill="none" stroke={P.vio}
            strokeWidth={5} strokeLinecap="round" strokeLinejoin="round" />
          {sLineK > 0.9 && <AdHex x={560} y={722} r={14} />}
          <text x={210} y={858} fill={P.muted} fontSize={20} fontFamily={FONT_HEAD}>醒来时长 →</text>
        </g>
        {/* 右下小面板：C 正弦 */}
        <g>
          <rect x={1320} y={648} width={460} height={220} rx={16} fill={P.panel} stroke={P.cyan} strokeWidth={4} />
          <text x={1346} y={692} fill={P.cyan} fontSize={26} fontFamily={FONT_HEAD} fontWeight={800}>
            C · 看钟点不看账</text>
          <rect x={1360} y={706} width={380} height={130} rx={10} fill={P.bgDeep} />
          <polyline points={cPts.slice(0, cN + 1).join(' ')} fill="none" stroke={P.cyan}
            strokeWidth={5} strokeLinecap="round" />
          {cWaveK > 0.9 && (
            <g>
              <circle cx={1550} cy={722} r={8} fill={P.amber} />
              <text x={1616} y={712} textAnchor="middle" fill={P.amber} fontSize={19}
                fontFamily={FONT_HEAD} fontWeight={700}>清醒峰</text>
            </g>
          )}
          <text x={1346} y={858} fill={P.muted} fontSize={20} fontFamily={FONT_HEAD}>视交叉上核 · 每天同一套表</text>
        </g>
        {/* 中下方结论（cue1） */}
        {fightK > 0 && (
          <g opacity={fightK} transform={`translate(0,${(1 - fightK) * 40})`}>
            <rect x={700} y={668} width={520} height={180} rx={18} fill={P.panel} stroke={P.coral} strokeWidth={4} />
            <text x={960} y={730} textAnchor="middle" fill={P.cream} fontSize={30}
              fontFamily={FONT_HEAD} fontWeight={800}>平时配合默契</text>
            <text x={960} y={778} textAnchor="middle" fill={P.coral} fontSize={32}
              fontFamily={FONT_HEAD} fontWeight={900}>通宵 → 当场打架</text>
            <text x={960} y={824} textAnchor="middle" fill={P.muted} fontSize={23}
              fontFamily={FONT_HEAD}>你被夹在中间 以为没事</text>
          </g>
        )}
      </svg>
      <Chip text="双过程模型 · 博尔贝利" x={1330} y={70} at={beat(0) + 30} accent={P.vio} />
      <Stamp text="睡意是两股力拔河" x={640} y={84} at={40} color={P.vioD} size={38} rot={-3} />
    </FadeIn>
  );
};

/* ============ S2 谷底反弹的清晨：S×C 双线机制图 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const sK = clamp01((f - 15) / 100);
  const cK = clamp01((f - 70) / 100);
  const feltK = clamp01((f - b1 - 15) / 90);
  const winK = clamp01((f - b1 - 8) / 20);
  const cortK = clamp01((f - b1 - 40) / 24);
  const crashK = clamp01((f - b1 - 120) / 24);

  const X0 = 215, X1 = 1245, YB = 690, YT = 240, W = X1 - X0;
  const h2x = (h: number) => X0 + ((h - 20) / 19) * W;           // 20点→次日15点
  const y2y = (v: number) => YB - v * (YB - YT);
  const sFn = (h: number) => 0.95 - 0.6 * Math.exp(-(h - 20) / 7);
  const cFn = (h: number) => 0.22 + 0.30 * Math.exp(-(((h - 21) ** 2) / 8)) + 0.62 * Math.exp(-(((h - 33.2) ** 2) / 7));
  const felt = (h: number) => clamp01(0.9 * sFn(h) - 0.62 * cFn(h) + 0.16);

  const draw = (fn: (h: number) => number, k: number, col: string, dash?: string, lw = 6) => {
    const pts: string[] = [];
    const hMax = 20 + 19 * clamp01(k);
    for (let h = 20; h <= hMax; h += 0.2) {
      pts.push(`${h2x(h).toFixed(1)},${y2y(fn(h)).toFixed(1)}`);
    }
    if (pts.length < 2) return null;
    return <polyline points={pts.join(' ')} fill="none" stroke={col} strokeWidth={lw}
      strokeLinecap="round" strokeLinejoin="round" strokeDasharray={dash} />;
  };

  const ticks: [number, string][] = [[20, '20点'], [24, '0点'], [28, '4点'], [33, '9点'], [36, '12点'], [39, '15点']];

  return (
    <FadeIn>
      <Backdrop top={P.bgDeep} bottom={P.bgDeep} />
      <SceneTag index={2} name={sec.name} accent={P.amber} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 主图面板 */}
        <rect x={140} y={150} width={1180} height={640} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <text x={176} y={204} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
          通宵一夜 · 两股力的错位现场</text>
        {/* 夜段底纹：20点→6点半 */}
        <rect x={X0} y={YT} width={h2x(26.5) - X0} height={YB - YT} fill={P.vio} opacity={0.07} />
        <Moon x={265} y={300} scale={0.28} opacity={0.85} />
        {/* 轴与刻度 */}
        <line x1={X0} y1={YB} x2={X1} y2={YB} stroke={P.muted} strokeWidth={3.5} />
        <line x1={X0} y1={YB} x2={X0} y2={YT - 12} stroke={P.muted} strokeWidth={3.5} />
        {ticks.map(([h, lab]) => (
          <g key={h}>
            <line x1={h2x(h)} y1={YB} x2={h2x(h)} y2={YB + 10} stroke={P.muted} strokeWidth={3} />
            <text x={h2x(h)} y={YB + 40} textAnchor="middle" fill={P.muted} fontSize={22}
              fontFamily={FONT_HEAD}>{lab}</text>
          </g>
        ))}
        {/* 4:00 谷底标记 */}
        <line x1={h2x(28)} y1={YB} x2={h2x(28)} y2={YT + 30} stroke={P.vio}
          strokeWidth={3} strokeDasharray="7 8" opacity={0.8} />
        <rect x={h2x(28) - 120} y={YT - 4} width={240} height={44} rx={12} fill={P.vio} opacity={0.92} />
        <text x={h2x(28)} y={YT + 26} textAnchor="middle" fontSize={24} fill={P.ink}
          fontFamily={FONT_HEAD} fontWeight={800}>4点 · 困意深渊</text>
        {/* S 曲线 */}
        {draw(sFn, sK, P.vio)}
        {sK > 0.95 && (
          <g>
            <AdHex x={h2x(38.2)} y={y2y(sFn(38.2))} r={15} />
            <text x={h2x(37.6)} y={y2y(sFn(38.2)) - 34} textAnchor="middle" fill={P.vio}
              fontSize={24} fontFamily={FONT_HEAD} fontWeight={800}>S 压力顶满</text>
          </g>
        )}
        {/* C 曲线 */}
        {draw(cFn, cK, P.cyan)}
        {cK > 0.95 && (
          <text x={1140} y={626} textAnchor="middle" fill={P.cyan}
            fontSize={24} fontFamily={FONT_HEAD} fontWeight={800}>C 节律信号</text>
        )}
        {/* cue1：假清醒窗口 */}
        {winK > 0 && (
          <g opacity={winK}>
            <rect x={h2x(30.5)} y={YT} width={h2x(34.5) - h2x(30.5)} height={YB - YT}
              fill={P.cyan} opacity={0.1} />
            <text x={h2x(32.5)} y={YT + 30} textAnchor="middle" fill={P.cyan} fontSize={23}
              fontFamily={FONT_HEAD} fontWeight={800}>假性清醒窗</text>
          </g>
        )}
        {/* 体感困意虚线 */}
        {draw(felt, feltK, P.cream, '16 12', 5)}
        {feltK > 0.9 && (
          <g>
            <circle cx={h2x(33.2)} cy={y2y(felt(33.2))} r={10} fill={P.amber} stroke={P.cream} strokeWidth={3} />
            <text x={h2x(33.2)} y={y2y(felt(33.2)) + 40} textAnchor="middle" fill={P.cream}
              fontSize={23} fontFamily={FONT_HEAD} fontWeight={700}>体感困意 · 被拽下来了</text>
          </g>
        )}
        {/* C 把你往上拽的箭头 */}
        {winK > 0.5 && (
          <g opacity={clamp01((winK - 0.5) * 2)}>
            <line x1={h2x(31.2)} y1={y2y(0.72)} x2={h2x(31.2)} y2={y2y(0.4)}
              stroke={P.cyan} strokeWidth={7} strokeLinecap="round" />
            <polygon points={`${h2x(31.2)},${y2y(0.36)} ${h2x(31.2) - 13},${y2y(0.44)} ${h2x(31.2) + 13},${y2y(0.44)}`}
              fill={P.cyan} />
          </g>
        )}
        {/* 15点 债主回来（尾段） */}
        {crashK > 0 && (
          <g opacity={crashK}>
            <line x1={h2x(38.6)} y1={YB} x2={h2x(38.6)} y2={y2y(felt(38.6)) - 30}
              stroke={P.coral} strokeWidth={4} strokeDasharray="8 8" />
            <text x={h2x(38.6)} y={YB - 14} textAnchor="middle" fill={P.coral} fontSize={23}
              fontFamily={FONT_HEAD} fontWeight={800}>15点 债主回来</text>
          </g>
        )}
        {/* 右列卡片 */}
        <g>
          <rect x={1350} y={150} width={430} height={280} rx={18} fill={P.panel} stroke={P.amber} strokeWidth={4} />
          <text x={1380} y={202} fill={P.amber} fontSize={27} fontFamily={FONT_HEAD} fontWeight={800}>
            皮质醇 · 跟着上岗</text>
          {/* 小试管 */}
          <g transform="translate(1440,310)">
            <rect x={-13} y={-62} width={26} height={10} rx={5} fill={P.cream} />
            <path d="M -13 -52 L -13 30 A 13 13 0 0 0 13 30 L 13 -52 Z" fill="none" stroke={P.cream} strokeWidth={4} />
            <path d={`M -10 ${30 - 52 * cortK} L -10 26 A 10 10 0 0 0 10 26 L 10 ${30 - 52 * cortK} Z`}
              fill={P.amber} opacity={0.9} />
          </g>
          {['升血糖', '提心率', '拉体温'].map((t, i) => (
            <g key={t} transform={`translate(1520,${272 + i * 48})`} opacity={clamp01((f - b1 - 50 - i * 12) / 16)}>
              <circle r={8} fill={P.amber} />
              <text x={24} y={9} fill={P.cream} fontSize={25} fontFamily={FONT_HEAD} fontWeight={700}>{t}</text>
            </g>
          ))}
          <rect x={1350} y={460} width={430} height={330} rx={18} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={1380} y={512} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>同一时间 · 两台仪表</text>
          {/* 仪表1：压力 */}
          <text x={1380} y={572} fill={P.vio} fontSize={24} fontFamily={FONT_HEAD} fontWeight={700}>睡眠压力</text>
          <rect x={1380} y={588} width={340} height={30} rx={15} fill={P.card} />
          <rect x={1380} y={588} width={340 * clamp01(sK)} height={30} rx={15} fill={P.vio} />
          <text x={1730} y={612} textAnchor="end" fill={P.vio} fontSize={22}
            fontFamily={FONT_HEAD} fontWeight={800}>满格</text>
          {/* 仪表2：信号 */}
          <text x={1380} y={672} fill={P.cyan} fontSize={24} fontFamily={FONT_HEAD} fontWeight={700}>清醒信号</text>
          <rect x={1380} y={688} width={340} height={30} rx={15} fill={P.card} />
          <rect x={1380} y={688} width={Math.max(24, 340 * clamp01((f - b1 - 20) / 60))} height={30} rx={15} fill={P.cyan} />
          <text x={1730} y={712} textAnchor="end" fill={P.cyan} fontSize={22}
            fontFamily={FONT_HEAD} fontWeight={800}>爬坡中</text>
          <text x={1380} y={764} fill={P.cream} fontSize={24} fontFamily={FONT_HEAD}>
            压力没减 · 是信号盖过它</text>
        </g>
      </svg>
      <Chip text="不是你恢复了" x={1420} y={76} at={b1 + 30} accent={P.amber} />
      <Stamp text="节律把你从谷底往上拽" x={330} y={840} at={b1 + 60} color={P.amberD} size={38} rot={-2} />
    </FadeIn>
  );
};

/* ============ S3 腺苷账单一分没少：受体现场 + 咖啡因 + 崩盘 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);
  const dockK = [0, 1, 2].map((i) => clamp01((f - 30 - i * 35) / 22));
  const cafK = clamp01((f - b1 - 10) / 20);
  const cafOut = clamp01((f - b2 - 5) / 26);          // 咖啡因撤退
  const floodK = clamp01((f - b2 - 20) / 30);         // 腺苷反扑
  const debt = Math.round(8 + Math.min(f * 0.09, 26) + floodK * 34);
  const cupP = Math.max(0.001, spring({frame: f - b1 - 5, fps, config: {damping: 11, stiffness: 120}}));

  const slots = [300, 440, 580, 720, 860];
  const slotDock = (i: number) => {
    // 0-2 号早期 docking；3-4 被咖啡因占后被 flood 接管
    if (i < 3) return dockK[i];
    if (cafK > 0.3 && cafOut < 1) return 0;
    return floodK;
  };
  const cafOn = (i: number) => (i >= 3 && cafK > 0.3 ? clamp01((cafK - 0.3) * 3) * (1 - cafOut) : 0);

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={3} name={sec.name} accent={P.vio} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：受体现场 */}
        <rect x={120} y={150} width={860} height={500} rx={20} fill={P.panel}
          stroke={floodK > 0.4 ? P.coral : P.card} strokeWidth={4} />
        <text x={156} y={204} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
          大脑受体现场 · 腺苷一上岗就降转速</text>
        {/* 膜带 */}
        <rect x={170} y={400} width={760} height={44} rx={20} fill={P.card} />
        {slots.map((sx, i) => (
          <g key={i}>
            <path d={`M ${sx - 20} 400 v -30 q 0 -12 12 -12 h 16 q 12 0 12 12 v 30`}
              fill={P.bgDeep} stroke={P.muted} strokeWidth={3} />
            {/* 腺苷飞来 docking */}
            {(() => {
              const k = slotDock(i);
              if (k <= 0) return null;
              const y = interpolate(k, [0, 1], [560 + (i % 3) * 24, 384]);
              return <AdHex x={sx} y={y} r={18} op={k} />;
            })()}
            {/* 咖啡因占位 */}
            {cafOn(i) > 0 && <CafSq x={slots[i]} y={384} r={16} op={cafOn(i)} />}
          </g>
        ))}
        {/* 反扑时多飞一群 */}
        {floodK > 0 && [0, 1, 2, 3, 4].map((i) => (
          <AdHex key={i} x={240 + i * 140 + rand(i, 3) * 40}
            y={interpolate(floodK, [0, 1], [600, 320 + rand(i, 5) * 60])}
            r={15} op={floodK} />
        ))}
        {/* 膜下注释 */}
        <text x={200} y={500} fill={P.muted} fontSize={23} fontFamily={FONT_HEAD}>
          腺苷压在受体上 = 给神经降转速</text>
        {/* 咖啡因被子（cue1 进入） */}
        {cupP > 0.02 && (
          <g transform={`translate(268,296) scale(${cupP})`} opacity={clamp01(cupP)}>
            <path d="M -44 -30 L -36 34 A 16 14 0 0 0 12 34 L 20 -30 Z" fill={P.paper} />
            <path d="M 20 -22 q 34 -4 30 18 q -4 20 -34 14" fill="none" stroke={P.paper} strokeWidth={7} />
            <path d="M -40 -22 L -33 28 A 12 10 0 0 0 8 28 L 15 -22 Z" fill={P.amberD} />
            {[0, 1].map((i) => (
              <path key={i} d={`M ${-14 + i * 18} -44 q 6 -10 0 -18 q -6 -8 0 -16`}
                stroke={P.muted} strokeWidth={4} fill="none" opacity={0.7} />
            ))}
            <text x={-8} y={14} textAnchor="middle" fontSize={19} fill={P.ink}
              fontFamily={FONT_HEAD} fontWeight={900}>咖</text>
          </g>
        )}
        {cafK > 0.3 && cafOut < 0.9 && (
          <g opacity={clamp01((cafK - 0.3) * 3) * (1 - cafOut)}>
            <rect x={430} y={252} width={380} height={52} rx={26} fill={P.amber} />
            <text x={620} y={288} textAnchor="middle" fontSize={25} fill={P.ink}
              fontFamily={FONT_HEAD} fontWeight={800}>咖啡因堵门 · 但账照记</text>
          </g>
        )}
        {floodK > 0 && (
          <g opacity={floodK}>
            <rect x={380} y={252} width={460} height={52} rx={26} fill={P.coral} />
            <text x={610} y={288} textAnchor="middle" fontSize={25} fill={P.cream}
              fontFamily={FONT_HEAD} fontWeight={800}>咖啡因一退 · 连本带利反扑</text>
          </g>
        )}
        {/* 右：腺苷账本 */}
        <rect x={1020} y={150} width={760} height={500} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <text x={1056} y={204} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>腺苷账本 · 一分没少</text>
        {/* 大计数器 */}
        <g transform="translate(1520,400)">
          <rect x={-190} y={-120} width={380} height={180} rx={20} fill={P.bgDeep}
            stroke={floodK > 0.4 ? P.coral : P.vio} strokeWidth={5} />
          <text x={0} y={14} textAnchor="middle" fontSize={104}
            fill={floodK > 0.4 ? P.coral : P.vio} fontFamily={FONT_HEAD} fontWeight={900}>
            {debt}
          </text>
          <text x={0} y={52} textAnchor="middle" fontSize={22} fill={P.muted}
            fontFamily={FONT_HEAD}>欠账分钟数 · 只增不减</text>
        </g>
        {cafK > 0.3 && cafOut < 1 && (
          <g opacity={clamp01((cafK - 0.3) * 3)}>
            <rect x={1580} y={296} width={160} height={56} rx={14} fill={P.amber} />
            <text x={1660} y={334} textAnchor="middle" fontSize={26} fill={P.ink}
              fontFamily={FONT_HEAD} fontWeight={800}>+利息</text>
          </g>
        )}
        {/* 账本刻度 */}
        {[0, 1, 2, 3].map((i) => (
          <g key={i} transform={`translate(1080,${300 + i * 80})`}>
            <rect width={64} height={52} rx={10} fill={P.card} />
            <AdHex x={32} y={26} r={15} op={i * 0.2 < f / 240 ? 1 : 0.25} />
            <text x={86} y={36} fontSize={24} fill={P.muted} fontFamily={FONT_HEAD}>
              {['醒 16h', '醒 20h', '通宵 24h', '次日下午'][i]}
            </text>
          </g>
        ))}
        {/* 底部崩盘条（cue2） */}
        {floodK > 0 && (
          <g opacity={floodK} transform={`translate(0,${(1 - floodK) * 70})`}>
            <rect x={120} y={690} width={1660} height={180} rx={20} fill={P.panel} stroke={P.coral} strokeWidth={4} />
            <TimeBadge x={280} y={780} t="15:00" col={P.coral} s={1.05} />
            <DeskPerson x={620} y={768} s={0.95} mode="crash" />
            <text x={880} y={760} fill={P.coral} fontSize={34} fontFamily={FONT_HEAD} fontWeight={900}>
              下午直接报废</text>
            <text x={880} y={812} fill={P.muted} fontSize={26} fontFamily={FONT_HEAD}>
              节律高潮一过 + 咖啡因一退 = 翻倍奉还</text>
            <text x={1660} y={790} textAnchor="middle" fontSize={44} fill={P.coral}
              fontFamily={FONT_HEAD} fontWeight={900} transform="rotate(-4 1660 790)">连本带利</text>
          </g>
        )}
      </svg>
      <Chip text="咖啡因只是耳塞 · 不是销账" x={1180} y={70} at={b1 + 25} accent={P.amber} />
      <Stamp text="腺苷账单一分没少" x={620} y={86} at={30} color={P.vioD} size={38} rot={-3} />
    </FadeIn>
  );
};

/* ============ S4 硬撑的代价与补救 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const pop = [0, 1, 2].map((i) => Math.max(0.001, spring({frame: f - 15 - i * 16, fps, config: {damping: 11, stiffness: 120}})));
  const gapK = clamp01((f - 120) / 24);
  const listK = clamp01((f - b1 - 8) / 22);
  const listPop = [0, 1, 2, 3].map((i) => Math.max(0.001, spring({frame: f - b1 - 12 - i * 9, fps, config: {damping: 12, stiffness: 140}})));

  const cards = [
    {x: 150, col: P.coral, title: '注意力 ↓', cap: '反应测试全线掉分'},
    {x: 720, col: P.amber, title: '情绪控制 ↓', cap: '杏仁核放大 · 前额叶掉线'},
    {x: 1290, col: P.vio, title: '自我判断失真', cap: '「我感觉没事」≠ 没事'},
  ];

  // 微睡眠断片：一条意识线被打断
  const gaps = [[420, 500], [700, 800], [1020, 1080], [1340, 1440]];

  const items = [
    {icon: 'bed', t: '能睡就睡', d: '20分钟也能清一部分账'},
    {icon: 'wheel', t: '千万别开车', d: '疲劳反应 ≈ 酒驾'},
    {icon: 'cup', t: '咖啡因赶早', d: '放早上用 · 晚上是埋单'},
    {icon: 'clock', t: '晚上按点睡', d: '别补到傍晚 · 拉回节律'},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={4} name={sec.name} accent={P.teal} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 上排：三张代价卡 */}
        {cards.map((c, i) => (
          <g key={i} transform={`translate(${c.x},150) scale(${pop[i]}) translate(${-c.x},-150)`}
            opacity={clamp01(pop[i] * 2)}>
            <rect x={c.x} y={150} width={480} height={300} rx={18} fill={P.panel} stroke={c.col} strokeWidth={4} />
            <text x={c.x + 32} y={204} fill={c.col} fontSize={30} fontFamily={FONT_HEAD} fontWeight={800}>{c.title}</text>
            {i === 0 && (
              <g>
                {[64, 40, 22, 10].map((h, k) => (
                  <rect key={k} x={c.x + 60 + k * 92} y={360 - h} width={56} height={h}
                    rx={8} fill={k === 0 ? P.muted : c.col} opacity={0.9} />
                ))}
                <line x1={c.x + 56} y1={362} x2={c.x + 440} y2={362} stroke={P.muted} strokeWidth={3} />
                <path d={`M ${c.x + 70} 290 L ${c.x + 400} 352`} stroke={c.col} strokeWidth={5}
                  strokeDasharray="10 8" strokeLinecap="round" />
              </g>
            )}
            {i === 1 && (
              <g>
                <circle cx={c.x + 140} cy={310} r={52} fill={C.skin} />
                <path d={`M ${c.x + 88} 296 Q ${c.x + 140} 262 ${c.x + 192} 296 L ${c.x + 192} 306 Q ${c.x + 140} 282 ${c.x + 88} 306 Z`}
                  fill="#3a2c4e" />
                <path d={`M ${c.x + 118} 312 l 12 10 M ${c.x + 150} 312 l 12 10`} stroke={P.ink}
                  strokeWidth={4} strokeLinecap="round" />
                <path d={`M ${c.x + 124} 342 q 16 -8 32 0`} stroke={P.ink} strokeWidth={4}
                  fill="none" strokeLinecap="round" />
                {/* 火气 */}
                <g transform={`translate(${c.x + 330},300)`}>
                  <path d="M 0 -46 Q 22 -20 14 4 Q 30 -10 24 -34 Q 46 -6 30 22 Q 10 44 -12 30 Q -34 14 -22 -12 Q -12 -30 0 -46 Z"
                    fill={P.coral} opacity={0.9} />
                  <text x={0} y={70} textAnchor="middle" fontSize={22} fill={c.col}
                    fontFamily={FONT_HEAD} fontWeight={800}>易燃易爆炸</text>
                </g>
              </g>
            )}
            {i === 2 && (
              <g>
                <text x={c.x + 40} y={290} fill={P.muted} fontSize={23} fontFamily={FONT_HEAD}>我感觉：</text>
                <rect x={c.x + 150} y={262} width={260} height={40} rx={20} fill={P.teal} />
                <text x={c.x + 280} y={292} textAnchor="middle" fontSize={24} fill={P.ink}
                  fontFamily={FONT_HEAD} fontWeight={800}>没事 还能干</text>
                <text x={c.x + 40} y={352} fill={P.muted} fontSize={23} fontFamily={FONT_HEAD}>实际：</text>
                <rect x={c.x + 150} y={324} width={260} height={40} rx={20} fill={P.card} />
                <rect x={c.x + 150} y={324} width={80} height={40} rx={20} fill={P.coral} />
                <text x={c.x + 280} y={354} textAnchor="middle" fontSize={24} fill={P.cream}
                  fontFamily={FONT_HEAD} fontWeight={800}>认知碎成段</text>
              </g>
            )}
            <text x={c.x + 32} y={428} fill={P.muted} fontSize={23} fontFamily={FONT_HEAD}>{c.cap}</text>
          </g>
        ))}
        {/* 微睡眠断片带 */}
        <g opacity={gapK}>
          <rect x={150} y={484} width={1620} height={140} rx={18} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={186} y={534} fill={P.muted} fontSize={24} fontFamily={FONT_HEAD}>
            微睡眠 · 意识在好几秒盲区里离线，你自己察觉不到</text>
          {/* 意识线（有断口） */}
          {(() => {
            const segs: React.ReactNode[] = [];
            let prev = 210;
            gaps.forEach(([a, b], i) => {
              segs.push(<line key={`s${i}`} x1={prev} y1={576} x2={a} y2={576}
                stroke={P.teal} strokeWidth={7} strokeLinecap="round" />);
              segs.push(
                <g key={`g${i}`}>
                  <line x1={a} y1={576} x2={b} y2={576} stroke={P.coral} strokeWidth={7}
                    strokeDasharray="4 12" strokeLinecap="round"
                    opacity={0.4 + 0.5 * Math.abs(Math.sin(f / 7 + i))} />
                  <text x={(a + b) / 2} y={612} textAnchor="middle" fontSize={20}
                    fill={P.coral} fontFamily={FONT_HEAD} fontWeight={700}>断片</text>
                </g>
              );
              prev = b;
            });
            segs.push(<line key="end" x1={prev} y1={576} x2={1700} y2={576}
              stroke={P.teal} strokeWidth={7} strokeLinecap="round" />);
            return segs;
          })()}
        </g>
        {/* 减伤清单（cue1） */}
        {listK > 0 && (
          <g opacity={listK} transform={`translate(0,${(1 - listK) * 60})`}>
            <text x={150} y={688} fill={P.teal} fontSize={30} fontFamily={FONT_HEAD} fontWeight={900}>
              真要通宵 · 减伤清单</text>
            {items.map((it, i) => (
              <g key={i} transform={`translate(${150 + i * 412},712) scale(${listPop[i]}) translate(${-(150 + i * 412)},-712)`}
                opacity={clamp01(listPop[i] * 2)}>
                <rect x={150 + i * 412} y={712} width={396} height={160} rx={16}
                  fill={P.panel} stroke={P.teal} strokeWidth={4} />
                {/* 图标 */}
                <g transform={`translate(${150 + i * 412 + 62},792)`}>
                  {it.icon === 'bed' && (
                    <g>
                      <rect x={-36} y={-8} width={72} height={16} rx={8} fill={P.sky} />
                      <rect x={-36} y={-34} width={12} height={26} rx={5} fill={P.sky} />
                      <rect x={-30} y={-26} width={26} height={14} rx={6} fill={P.cream} />
                      <circle cx={-12} cy={-34} r={7} fill={C.skin} />
                    </g>
                  )}
                  {it.icon === 'wheel' && (
                    <g>
                      <circle r={30} fill="none" stroke={P.cream} strokeWidth={7} />
                      <circle r={9} fill={P.cream} />
                      {[0, 120, 240].map((a) => (
                        <line key={a} x1={0} y1={0} x2={0} y2={-26} stroke={P.cream}
                          strokeWidth={6} transform={`rotate(${a})`} strokeLinecap="round" />
                      ))}
                      <MarkX x={0} y={0} s={1.9} col={P.coral} />
                    </g>
                  )}
                  {it.icon === 'cup' && (
                    <g>
                      <path d="M -22 -20 L -17 18 A 9 8 0 0 0 7 18 L 12 -20 Z" fill={P.paper} />
                      <path d="M 12 -14 q 20 -2 17 10 q -3 11 -20 8" fill="none" stroke={P.paper} strokeWidth={5} />
                      <Sun x={30} y={-42} r={9} />
                    </g>
                  )}
                  {it.icon === 'clock' && (
                    <g>
                      <circle r={28} fill={P.panel} stroke={P.cream} strokeWidth={6} />
                      <line x1={0} y1={0} x2={0} y2={-17} stroke={P.cream} strokeWidth={5} strokeLinecap="round" />
                      <line x1={0} y1={0} x2={12} y2={6} stroke={P.cream} strokeWidth={5} strokeLinecap="round" />
                      <MarkV x={26} y={22} s={1.1} />
                    </g>
                  )}
                </g>
                <text x={150 + i * 412 + 116} y={782} fill={P.cream} fontSize={28}
                  fontFamily={FONT_HEAD} fontWeight={800}>{it.t}</text>
                <text x={150 + i * 412 + 116} y={826} fill={P.muted} fontSize={22}
                  fontFamily={FONT_HEAD}>{it.d}</text>
              </g>
            ))}
          </g>
        )}
      </svg>
      <Chip text="微睡眠 = 几秒离线" x={1400} y={70} at={beat(0) + 30} accent={P.teal} />
      <Stamp text="清醒是债 · 早晚要还" x={620} y={90} at={b1 + 70} color={P.teal} size={40} rot={-2} />
    </FadeIn>
  );
};

/* ============ 结尾卡 ============ */
const EndCard = () => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 14}});
  const pop = Math.max(0.001, spring({frame: f - 5, fps, config: {damping: 10, stiffness: 140}}));
  const sinkK = clamp01(f / 32);
  return (
    <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
      <Backdrop top={P.bgDeep} bottom={P.bgDeep} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={20} from={0} dur={22} />
        {/* 地平线 */}
        <line x1={240} y1={806} x2={1680} y2={806} stroke={P.card} strokeWidth={6} strokeLinecap="round" />
        {/* 月亮落山 */}
        <g transform={`translate(520,${806 - 60 + sinkK * 60})`} opacity={0.9}>
          <circle r={70} fill="#ffe9a8" />
          <circle cx={-26} cy={-18} r={54} fill={P.bgDeep} opacity={0.9} />
        </g>
        {/* 床上的人终于睡了 */}
        <g transform={`translate(960,780) scale(${pop})`} opacity={clamp01(pop)}>
          <rect x={-190} y={-16} width={380} height={24} rx={12} fill={P.card} />
          <rect x={-182} y={-96} width={20} height={80} rx={8} fill={P.card} />
          <rect x={-152} y={-52} width={96} height={36} rx={16} fill={P.cream} />
          <path d="M -56 -16 Q 30 -78 178 -20 L 184 -14 Z" fill={P.cyan} opacity={0.85} />
          <circle cx={-92} cy={-58} r={24} fill={C.skin} />
          <path d="M -116 -66 Q -92 -92 -68 -66 L -68 -60 Q -92 -78 -116 -60 Z" fill="#3a2c4e" />
          <path d="M -104 -54 q 5 6 10 0 M -88 -54 q 5 6 10 0" stroke={P.ink} strokeWidth={3}
            fill="none" strokeLinecap="round" />
        </g>
        <Zzz x={1130} y={700} k={0.9} col={P.cyan} />
        {/* 腺苷债清零牌 */}
        <g transform={`translate(1390,730) scale(${pop})`} opacity={clamp01(pop)}>
          <rect x={-140} y={-46} width={280} height={92} rx={18} fill={P.panel} stroke={P.teal} strokeWidth={4} />
          <text x={0} y={-6} textAnchor="middle" fontSize={26} fill={P.muted} fontFamily={FONT_HEAD}>腺苷账</text>
          <text x={0} y={32} textAnchor="middle" fontSize={34} fill={P.teal}
            fontFamily={FONT_HEAD} fontWeight={900}>一夜清零</text>
        </g>
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -300}}>
        <div style={{fontSize: 56, fontWeight: 800, color: C.text}}>清醒是借来的 · 记得按时归还</div>
        <div style={{fontSize: 36, color: P.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E231 = ({data = tl}: {data?: any}) => {
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
