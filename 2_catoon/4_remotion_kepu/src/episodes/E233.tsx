import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01, rand} from '../lib/style';
import {Backdrop, Stamp, SceneTag, Chip, Subtitles, ProgressBar, FadeIn, Moon, Stars} from '../lib/atoms';
import tl from '../timeline_233.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：梦夜紫 × 月光黄 × REM青 × 猫橘 ============ */
const P = {
  bg: '#232048',
  bgDeep: '#171330',
  bgLite: '#2b2657',
  panel: '#241f4a',
  card: '#372f68',
  ink: '#171232',
  cream: '#f6efdd',
  muted: '#a9aed8',
  vio: '#c39bf5',
  vioD: '#9673de',
  cyan: '#62d9ec',
  cyanD: '#3aaec6',
  amber: '#ffca66',
  amberD: '#e2a03c',
  coral: '#ff7d70',
  coralD: '#d85a4e',
  teal: '#66e0b0',
  cat: '#f5a95c',
  catD: '#d98a3e',
  dog: '#d9a878',
  dogD: '#b8855a',
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

/* ---------- 梦境云泡 ---------- */
const Cloud = ({x, y, s = 1, op = 1, children}: {x: number; y: number; s?: number; op?: number; children?: React.ReactNode}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
    {[[-86, 12, 50], [-22, -20, 64], [52, -12, 56], [96, 16, 42], [10, 26, 60]].map(([cx, cy, r], i) => (
      <circle key={i} cx={cx} cy={cy} r={r + 7} fill={P.vioD} />
    ))}
    {[[-86, 12, 50], [-22, -20, 64], [52, -12, 56], [96, 16, 42], [10, 26, 60]].map(([cx, cy, r], i) => (
      <circle key={i} cx={cx} cy={cy} r={r} fill={P.bgLite} />
    ))}
    {children}
  </g>
);

/* ---------- 蜷缩睡觉的猫 ---------- */
const SleepCat = ({x, y, s = 1}: {x: number; y: number; s?: number}) => {
  const f = useCurrentFrame();
  const br = Math.sin(f * 0.12) * 2.5;                    // 呼吸起伏
  const earT = Math.max(0, Math.sin(f * 0.5)) * Math.sin(f * 1.9) * 6;  // 耳朵抽动
  const whiT = Math.sin(f * 1.7) * 4;                     // 胡须抽搐
  const pawT = Math.max(0, Math.sin(f * 0.45)) * 7;       // 爪子抽动
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      <path d="M 62 34 Q 118 14 100 -34 Q 92 -56 66 -50" fill="none" stroke={P.catD} strokeWidth={22} strokeLinecap="round" />
      <ellipse cx={0} cy={-10 - br} rx={106} ry={80} fill={P.cat} />
      <ellipse cx={-34} cy={34} rx={72} ry={32} fill={P.catD} opacity={0.45} />
      <g transform={`translate(-62,-76) rotate(${-16 + earT})`}>
        <polygon points="0,0 -32,-44 16,-26" fill={P.cat} />
        <polygon points="-5,-11 -23,-36 7,-24" fill={P.catD} />
      </g>
      <g transform="translate(34,-86) rotate(10)">
        <polygon points="0,0 -18,-42 26,-22" fill={P.cat} />
      </g>
      <path d="M -32 -18 q 8 8 16 0 M 10 -20 q 8 8 16 0" stroke={P.ink} strokeWidth={4.4} fill="none" strokeLinecap="round" />
      <polygon points="-5,-2 5,-2 0,5" fill={P.coralD} />
      <g stroke={P.cream} strokeWidth={3.4} strokeLinecap="round" transform={`rotate(${whiT})`}>
        <line x1={-60} y1={-6} x2={-98} y2={-16} />
        <line x1={-60} y1={2} x2={-100} y2={4} />
        <line x1={60} y1={-10} x2={96} y2={-22} />
        <line x1={60} y1={-2} x2={98} y2={0} />
      </g>
      <ellipse cx={-58 + pawT * 0.5} cy={46 - pawT} rx={26} ry={15} fill={P.catD} />
      <ellipse cx={8} cy={50} rx={28} ry={14} fill={P.catD} />
    </g>
  );
};

/* ---------- 侧躺蹬腿的狗 ---------- */
const SleepDog = ({x, y, s = 1, kick = 1}: {x: number; y: number; s?: number; kick?: number}) => {
  const f = useCurrentFrame();
  const legA = Math.sin(f * 0.55) * 13 * kick;
  const legB = Math.sin(f * 0.55 + 1.5) * 15 * kick;
  const tailW = Math.sin(f * 0.42) * 9;
  const br = Math.sin(f * 0.12) * 2;
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      <g transform="translate(122,-34)">
        <path d={`M 0 0 Q 40 ${-16 - tailW} 56 ${-42 - tailW}`} stroke={P.dogD} strokeWidth={17} fill="none" strokeLinecap="round" />
      </g>
      <ellipse cx={0} cy={-br} rx={132} ry={62} fill={P.dog} />
      <ellipse cx={74} cy={-14} rx={50} ry={34} fill={P.cream} opacity={0.5} />
      <g transform={`translate(92,28) rotate(${legA})`}>
        <rect x={-10} y={0} width={100} height={26} rx={13} fill={P.dogD} />
        <circle cx={92} cy={12} r={15} fill={P.dogD} />
      </g>
      <g transform={`translate(54,42) rotate(${legB * 0.8})`}>
        <rect x={-10} y={0} width={90} height={24} rx={12} fill={P.dog} />
        <circle cx={82} cy={10} r={14} fill={P.dog} />
      </g>
      <rect x={-198} y={28} width={102} height={24} rx={12} fill={P.dogD} />
      <rect x={-192} y={4} width={90} height={22} rx={11} fill={P.dog} />
      <g transform="translate(-132,-34) rotate(-8)">
        <circle r={54} fill={P.dog} />
        <ellipse cx={-16} cy={-38} rx={21} ry={32} fill={P.dogD} transform="rotate(-28)" />
        <ellipse cx={-36} cy={16} rx={30} ry={22} fill={P.cream} />
        <circle cx={-54} cy={8} r={8} fill={P.ink} />
        <path d="M -8 -16 q 8 8 16 0" stroke={P.ink} strokeWidth={4.4} fill="none" strokeLinecap="round" />
      </g>
    </g>
  );
};

/* ---------- 站起来"演梦"的猫（扑咬姿态） ---------- */
const ActCat = ({x, y, s = 1, act = 1}: {x: number; y: number; s?: number; act?: number}) => {
  const f = useCurrentFrame();
  const reach = Math.sin(f * 0.7) * 10 * act;
  const bob = Math.sin(f * 0.4) * 3 * act;
  return (
    <g transform={`translate(${x},${y + bob}) scale(${s})`}>
      <path d="M 96 -12 Q 152 -60 132 -112" stroke={P.cat} strokeWidth={18} fill="none" strokeLinecap="round" />
      <path d="M 62 28 L 84 92 L 62 96" stroke={P.catD} strokeWidth={20} fill="none" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M 92 24 L 118 86 L 98 94" stroke={P.cat} strokeWidth={20} fill="none" strokeLinecap="round" strokeLinejoin="round" />
      <ellipse cx={0} cy={0} rx={102} ry={46} fill={P.cat} transform="rotate(-8)" />
      <g transform={`translate(-84,16) rotate(${-34 + reach * 0.5})`}>
        <rect x={-96} y={-11} width={100} height={22} rx={11} fill={P.cat} />
        <circle cx={-98} cy={0} r={13} fill={P.catD} />
      </g>
      <g transform="translate(-70,36) rotate(-6)">
        <rect x={-80} y={-10} width={86} height={20} rx={10} fill={P.catD} />
      </g>
      <g transform="translate(-108,-54) rotate(-6)">
        <circle r={46} fill={P.cat} />
        <polygon points="-32,-26 -46,-66 -8,-46" fill={P.cat} />
        <polygon points="14,-34 22,-70 46,-46" fill={P.cat} />
        <circle cx={-16} cy={-4} r={7} fill="#fff" />
        <circle cx={14} cy={-4} r={7} fill="#fff" />
        <circle cx={-14} cy={-3} r={3.4} fill={P.ink} />
        <circle cx={16} cy={-3} r={3.4} fill={P.ink} />
        <path d="M -4 12 q 8 6 16 0" stroke={P.ink} strokeWidth={3.4} fill="none" strokeLinecap="round" />
        <g stroke={P.cream} strokeWidth={2.8} strokeLinecap="round">
          <line x1={-52} y1={4} x2={-82} y2={-2} />
          <line x1={-52} y1={10} x2={-82} y2={14} />
        </g>
      </g>
    </g>
  );
};

/* ---------- 挂锁 ---------- */
const PadLock = ({x, y, s = 1, open = 0, col = P.teal}: {x: number; y: number; s?: number; open?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <g transform={`rotate(${-open * 55})`}>
      <path d="M -18 -6 L -18 -24 A 18 18 0 0 1 18 -24 L 18 -6" fill="none" stroke={col} strokeWidth={8} strokeLinecap="round" />
    </g>
    <rect x={-30} y={-6} width={60} height={50} rx={12} fill={col} />
    <circle cx={0} cy={14} r={7} fill={P.ink} />
    <rect x={-3} y={16} width={6} height={14} rx={3} fill={P.ink} />
  </g>
);

/* ---------- 眼皮下乱转的眼球 ---------- */
const Eye = ({x, y, s = 1}: {x: number; y: number; s?: number}) => {
  const f = useCurrentFrame();
  const dx = Math.sin(f * 0.8) * 12;
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      <ellipse rx={42} ry={30} fill={P.cream} />
      <circle cx={dx} cy={0} r={14} fill={P.cyanD} />
      <circle cx={dx} cy={0} r={7} fill={P.ink} />
      <circle cx={dx + 4} cy={-4} r={2.6} fill="#fff" />
      <path d="M -42 0 Q 0 -34 42 0" fill="none" stroke={P.cream} strokeWidth={4} opacity={0.7} />
    </g>
  );
};

/* ---------- 小鱼 / 骨头（梦境内容物） ---------- */
const Fish = ({x, y, s = 1, col = P.cyan}: {x: number; y: number; s?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <ellipse rx={30} ry={17} fill={col} />
    <polygon points="24,0 46,-14 46,14" fill={col} />
    <circle cx={-16} cy={-4} r={3.4} fill={P.ink} />
    <path d="M -2 -8 q 8 8 0 16" stroke={P.cyanD} strokeWidth={3} fill="none" />
  </g>
);
const Bone = ({x, y, s = 1, rot = 0}: {x: number; y: number; s?: number; rot?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`} fill={P.cream}>
    <rect x={-26} y={-7} width={52} height={14} rx={7} />
    <circle cx={-30} cy={-8} r={10} /><circle cx={-30} cy={8} r={10} />
    <circle cx={30} cy={-8} r={10} /><circle cx={30} cy={8} r={10} />
  </g>
);

/* ---------- 小鸟（梦的内容） ---------- */
const Bird = ({x, y, s = 1}: {x: number; y: number; s?: number}) => {
  const f = useCurrentFrame();
  const flap = Math.sin(f * 0.9) * 14;
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      <ellipse rx={24} ry={17} fill={'#7ab8ff'} />
      <circle cx={20} cy={-10} r={12} fill={'#7ab8ff'} />
      <polygon points="30,-12 44,-8 30,-4" fill={P.amber} />
      <circle cx={23} cy={-12} r={2.8} fill={P.ink} />
      <g transform={`rotate(${-flap} 0 0)`}>
        <path d="M 0 -4 Q -26 -26 -44 -10 Q -24 -2 0 2 Z" fill={P.cyanD} />
      </g>
      <path d="M -22 4 L -38 10 M -20 8 L -34 16" stroke={P.cyanD} strokeWidth={4} strokeLinecap="round" />
    </g>
  );
};

/* ---------- 小老鼠 ---------- */
const Rat = ({x, y, rot = 0, s = 1}: {x: number; y: number; rot?: number; s?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`}>
    <path d="M -32 8 Q -58 18 -70 4" stroke={P.cream} strokeWidth={5} fill="none" strokeLinecap="round" />
    <ellipse rx={30} ry={18} fill={P.muted} />
    <circle cx={26} cy={-6} r={13} fill={P.muted} />
    <circle cx={20} cy={-18} r={6} fill={P.vioD} />
    <circle cx={30} cy={-9} r={3} fill={P.ink} />
    <circle cx={39} cy={-4} r={3} fill={P.coralD} />
  </g>
);

/* ---------- 伸手的手掌（别摇醒） ---------- */
const Palm = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-156} y={-42} width={86} height={84} rx={20} fill={P.card} />
    <rect x={-80} y={-46} width={114} height={92} rx={28} fill={C.skin} />
    {[0, 1, 2, 3].map((i) => (
      <rect key={i} x={26} y={-42 + i * 23} width={48 - i * 4} height={17} rx={8.5} fill={C.skin} />
    ))}
    <rect x={-8} y={-66} width={42} height={17} rx={8.5} fill={C.skin} transform="rotate(-24 10 -58)" />
  </g>
);

/* ---------- 物种小图标 ---------- */
const PetIcon = ({kind, x, y, col}: {kind: string; x: number; y: number; col: string}) => {
  if (kind === 'cat') return (
    <g transform={`translate(${x},${y})`}>
      <circle r={34} fill={col} />
      <polygon points="-24,-18 -34,-46 -8,-32" fill={col} />
      <polygon points="16,-20 24,-46 40,-28" fill={col} />
      <path d="M -14 -2 q 5 5 10 0 M 6 -2 q 5 5 10 0" stroke={P.ink} strokeWidth={3.4} fill="none" strokeLinecap="round" />
    </g>
  );
  if (kind === 'dog') return (
    <g transform={`translate(${x},${y})`}>
      <circle r={34} fill={col} />
      <ellipse cx={-22} cy={-24} rx={13} ry={20} fill={P.dogD} transform="rotate(-30)" />
      <ellipse cx={24} cy={-22} rx={12} ry={19} fill={P.dogD} transform="rotate(28)" />
      <ellipse cx={0} cy={12} rx={18} ry={13} fill={P.cream} />
      <circle cx={0} cy={4} r={5} fill={P.ink} />
      <path d="M -16 -8 q 5 5 10 0 M 8 -8 q 5 5 10 0" stroke={P.ink} strokeWidth={3.4} fill="none" strokeLinecap="round" />
    </g>
  );
  if (kind === 'bird') return (
    <g transform={`translate(${x},${y})`}>
      <ellipse rx={30} ry={22} fill={col} />
      <circle cx={22} cy={-14} r={15} fill={col} />
      <polygon points="34,-16 50,-11 34,-5" fill={P.amber} />
      <circle cx={26} cy={-17} r={3} fill={P.ink} />
      <path d="M -8 -6 Q -30 -22 -40 -4 Q -22 2 -8 2 Z" fill={P.cyanD} />
    </g>
  );
  return (
    <g transform={`translate(${x},${y})`}>
      <path d="M -28 6 A 28 30 0 0 1 28 6 L 28 10 L -28 10 Z" fill={col} />
      {[-20, -7, 7, 20].map((dx, i) => (
        <path key={i} d={`M ${dx} 8 q ${i % 2 ? 6 : -6} 16 ${i % 2 ? -2 : 2} 26`}
          stroke={col} strokeWidth={9} fill="none" strokeLinecap="round" />
      ))}
      <circle cx={-9} cy={-4} r={4.4} fill={P.ink} />
      <circle cx={11} cy={-4} r={4.4} fill={P.ink} />
    </g>
  );
};

/* ============ 标题卡 ============ */
const TitleCard = ({data}: {data: any}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 13, mass: 0.7}});
  const o = interpolate(f, [0, 8], [0, 1]);
  const pA = Math.max(0.001, spring({frame: f - 4, fps, config: {damping: 11, stiffness: 140}}));
  const pB = Math.max(0.001, spring({frame: f - 16, fps, config: {damping: 11, stiffness: 140}}));
  return (
    <AbsoluteFill>
      <Backdrop top={P.bg} bottom={P.bg} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={22} from={0} dur={24} />
        <Moon x={250} y={145} scale={0.7} />
        {/* 大地毯 + 猫狗同睡 */}
        <g transform={`translate(960,600) scale(${pA}) translate(-960,-600)`} opacity={clamp01(pA)}>
          <ellipse cx={960} cy={600} rx={580} ry={104} fill={P.card} />
          <ellipse cx={960} cy={592} rx={534} ry={88} fill={P.bgLite} />
          <ellipse cx={960} cy={592} rx={534} ry={88} fill="none" stroke={P.vioD}
            strokeWidth={4} strokeDasharray="4 18" opacity={0.7} />
          <SleepDog x={680} y={548} s={0.95} />
          <SleepCat x={1290} y={540} s={0.92} />
          <Zzz x={520} y={430} k={0.8} col={P.vio} />
          <Zzz x={1420} y={420} k={0.8} col={P.cyan} />
        </g>
        {/* 梦泡 */}
        <g transform={`translate(960,300) scale(${pB}) translate(-960,-300)`} opacity={clamp01(pB)}>
          {[0, 1, 2].map((i) => (
            <g key={i}>
              <circle cx={770 + i * 52} cy={455 - i * 40} r={10 + i * 6} fill={P.bgLite}
                stroke={P.vioD} strokeWidth={3} opacity={0.85} />
              <circle cx={1150 - i * 52} cy={455 - i * 40} r={10 + i * 6} fill={P.bgLite}
                stroke={P.vioD} strokeWidth={3} opacity={0.85} />
            </g>
          ))}
          <Cloud x={960} y={290} s={1.05}>
            <Fish x={-64} y={6} s={1.1} />
            <Bone x={62} y={14} s={0.95} rot={-12} />
            <text x={2} y={-32} textAnchor="middle" fontSize={56} fill={P.coral}
              fontFamily={FONT_HEAD} fontWeight={900}>?</text>
          </Cloud>
        </g>
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 500}}>
          <div style={{color: P.cyan, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 22}}>
            冷知识 · 第 {data.episode} 期
          </div>
          <div style={{color: C.text, fontSize: 84, fontWeight: 900, fontFamily: FONT_HEAD, letterSpacing: 6}}>
            {data.title}
          </div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ============ S0 睡着就开始演戏：蹬腿的狗 × 抽胡须的猫 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);
  const popL = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const popR = Math.max(0.001, spring({frame: f - 42, fps, config: {damping: 11, stiffness: 120}}));
  const askPop = Math.max(0.001, spring({frame: f - b1 - 4, fps, config: {damping: 10, stiffness: 130}}));
  const verdictK = clamp01((f - b2 - 4) / 22);
  const whimper = Math.sin(f * 0.5) > 0.55 ? 1 : 0;

  const panel = (x0: number, k: number, kids: React.ReactNode) => (
    <g transform={`translate(${x0},150) scale(${k}) translate(${-x0},-150)`} opacity={clamp01(k * 2)}>
      <rect x={x0} y={150} width={770} height={480} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
      {kids}
    </g>
  );

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={0} name={sec.name} accent={P.cyan} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={10} from={0} dur={26} />
        {/* 左：狗在蹬腿 */}
        {panel(120, popL, (
          <g>
            <text x={156} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>狗 · 睡熟之后</text>
            <SleepDog x={500} y={430} s={1.05} />
            {/* 蹬腿的运动弧 */}
            {[0, 1].map((i) => (
              <path key={i} d={`M ${640 + i * 30} ${470 + i * 26} q 30 ${10 + Math.sin(f * 0.55 + i) * 8} 46 44`}
                stroke={P.cyan} strokeWidth={5} fill="none" strokeLinecap="round" opacity={0.65 - i * 0.25} />
            ))}
            {whimper > 0 && (
              <text x={300} y={330} fontSize={34} fill={P.amber} fontFamily={FONT_HEAD}
                fontWeight={800} transform="rotate(-8 300 330)">呜…</text>
            )}
            <text x={505} y={592} textAnchor="middle" fill={P.cyan} fontSize={28}
              fontFamily={FONT_HEAD} fontWeight={800}>蹬腿 · 摇尾巴 · 呜呜叫</text>
          </g>
        ))}
        {/* 右：猫在抽搐 */}
        {panel(1030, popR, (
          <g>
            <text x={1066} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>猫 · 睡熟之后</text>
            <SleepCat x={1400} y={430} s={1.12} />
            {[0, 1].map((i) => (
              <path key={i} d={`M ${1590 + i * 26} ${360 + i * 22} q ${20 + Math.sin(f * 1.3 + i) * 10} -18 34 -8`}
                stroke={P.vio} strokeWidth={5} fill="none" strokeLinecap="round" opacity={0.65 - i * 0.25} />
            ))}
            <text x={1410} y={592} textAnchor="middle" fill={P.vio} fontSize={28}
              fontFamily={FONT_HEAD} fontWeight={800}>抽胡须 · 抖爪子 · 转耳朵</text>
          </g>
        ))}
        {/* 中间的疑问泡（cue1） */}
        {askPop > 0.02 && (
          <g transform={`translate(960,380) scale(${askPop})`} opacity={clamp01(askPop)}>
            <circle r={92} fill={P.bgLite} stroke={P.coral} strokeWidth={6} />
            <text x={0} y={-6} textAnchor="middle" fontSize={58} fill={P.coral}
              fontFamily={FONT_HEAD} fontWeight={900}>做梦</text>
            <text x={0} y={44} textAnchor="middle" fontSize={50} fill={P.coral}
              fontFamily={FONT_HEAD} fontWeight={900}>吗?</text>
          </g>
        )}
        {/* 底条结论（cue2） */}
        {verdictK > 0 && (
          <g opacity={verdictK} transform={`translate(0,${(1 - verdictK) * 60})`}>
            <rect x={120} y={692} width={1680} height={176} rx={20} fill={P.panel} stroke={P.cyan} strokeWidth={4} />
            <text x={170} y={750} fill={P.muted} fontSize={26} fontFamily={FONT_HEAD}>科学界的态度已经变了</text>
            <text x={170} y={822} fill={P.cream} fontSize={44} fontFamily={FONT_HEAD} fontWeight={900}>
              不是「可能是」<tspan fill={P.cyan}> · 是「几乎肯定」在做梦</tspan>
            </text>
            <g transform="translate(1630,780)">
              <rect x={-130} y={-56} width={260} height={112} rx={16} fill={P.bgDeep} stroke={P.coral} strokeWidth={4} />
              <text x={0} y={-12} textAnchor="middle" fontSize={24} fill={P.muted} fontFamily={FONT_HEAD}>只剩争论</text>
              <text x={0} y={30} textAnchor="middle" fontSize={30} fill={P.coral}
                fontFamily={FONT_HEAD} fontWeight={800}>梦见了什么</text>
            </g>
          </g>
        )}
      </svg>
      <Chip text="蹬腿呜咽 · 全世界同款" x={1360} y={66} at={beat(0) + 40} accent={P.cyan} />
      <Stamp text="它们在做什么?" x={640} y={86} at={b1 + 20} color={P.coralD} size={40} rot={-4} />
    </FadeIn>
  );
};

/* ---------- 脑电波形 ---------- */
const Wave = ({x0, x1, yc, kind, prog, col, amp = 1, lw = 5}: {
  x0: number; x1: number; yc: number; kind: 'awake' | 'slow' | 'rem';
  prog: number; col: string; amp?: number; lw?: number;
}) => {
  const pts: string[] = [];
  const n = 64;
  const xmax = x0 + (x1 - x0) * clamp01(prog);
  for (let i = 0; i <= n; i++) {
    const u = i / n;
    const x = x0 + u * (x1 - x0);
    if (x > xmax) break;
    let yy = 0;
    if (kind === 'awake') yy = Math.sin(u * 44) * 7 + Math.sin(u * 21 + 1.3) * 6;
    if (kind === 'slow') yy = Math.sin(u * Math.PI * 2.6) * 42;
    if (kind === 'rem') yy = Math.sin(u * 40) * 8 + Math.sin(u * 16 + 2.1) * 7;
    pts.push(`${x.toFixed(1)},${(yc - yy * amp).toFixed(1)}`);
  }
  if (pts.length < 2) return null;
  return <polyline points={pts.join(' ')} fill="none" stroke={col} strokeWidth={lw}
    strokeLinecap="round" strokeLinejoin="round" />;
};

/* ============ S1 脑电波先过这一关 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);
  const awakeK = clamp01((f - 20) / 70);
  const slowK = clamp01((f - b1 - 10) / 90);
  const remK = clamp01((f - b1 - 80) / 80);
  const iconK = clamp01((f - b1 - 130) / 22);
  const cmpK = clamp01((f - b2 - 4) / 24);
  const cmpWave = clamp01((f - b2 - 10) / 80);

  return (
    <FadeIn>
      <Backdrop top={P.bgDeep} bottom={P.bgDeep} />
      <SceneTag index={1} name={sec.name} accent={P.vio} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 主图：一夜脑电三段式 */}
        <rect x={120} y={150} width={1680} height={420} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <text x={156} y={204} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
          判断做梦 · 第一关看脑电（人和猫狗通用）</text>
        {/* REM 区底纹 */}
        <rect x={1010} y={232} width={740} height={240} fill={P.vio} opacity={0.08} />
        {/* 分区虚线 */}
        {[610, 1010].map((x) => (
          <line key={x} x1={x} y1={240} x2={x} y2={470} stroke={P.card} strokeWidth={4} strokeDasharray="8 10" />
        ))}
        {/* 三段波 */}
        <Wave x0={170} x1={610} yc={352} kind="awake" prog={awakeK} col={P.cream} />
        <Wave x0={610} x1={1010} yc={352} kind="slow" prog={slowK} col={P.teal} />
        <Wave x0={1010} x1={1450} yc={352} kind="rem" prog={remK} col={P.vio} />
        {/* 区标 */}
        <text x={390} y={518} textAnchor="middle" fill={P.cream} fontSize={26}
          fontFamily={FONT_HEAD} fontWeight={800}>清醒 · 快而乱</text>
        <text x={810} y={518} textAnchor="middle" fill={P.teal} fontSize={26}
          fontFamily={FONT_HEAD} fontWeight={800}>慢波 · 大又慢</text>
        <text x={1200} y={518} textAnchor="middle" fill={P.vio} fontSize={26}
          fontFamily={FONT_HEAD} fontWeight={800}>REM · 做梦期</text>
        {/* REM 区的小部件：眼球乱转 + 肌肉锁 */}
        {iconK > 0 && (
          <g opacity={iconK} transform={`translate(0,${(1 - iconK) * 30})`}>
            <Eye x={1580} y={352} s={0.95} />
            <PadLock x={1700} y={330} s={0.85} col={P.teal} />
            <text x={1580} y={428} textAnchor="middle" fontSize={20} fill={P.cyan}
              fontFamily={FONT_HEAD} fontWeight={700}>眼球乱转</text>
            <text x={1700} y={428} textAnchor="middle" fontSize={20} fill={P.teal}
              fontFamily={FONT_HEAD} fontWeight={700}>肌肉锁死</text>
          </g>
        )}
        {/* cue2：猫狗同款脑电对照条 */}
        {cmpK > 0 && (
          <g opacity={cmpK} transform={`translate(0,${(1 - cmpK) * 60})`}>
            <rect x={120} y={602} width={1680} height={270} rx={20} fill={P.panel} stroke={P.vio} strokeWidth={4} />
            <text x={156} y={656} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
              两条脑电叠在一起 · 几乎逐帧对上</text>
            {/* 人 */}
            <text x={160} y={726} fill={P.cream} fontSize={24} fontFamily={FONT_HEAD} fontWeight={800}>人</text>
            <Wave x0={230} x1={1140} yc={712} kind="rem" prog={cmpWave} col={P.cream} amp={0.85} lw={4.5} />
            {/* 猫狗 */}
            <text x={160} y={818} fill={P.vio} fontSize={24} fontFamily={FONT_HEAD} fontWeight={800}>猫狗</text>
            <Wave x0={230} x1={1140} yc={800} kind="rem" prog={cmpWave} col={P.vio} amp={0.85} lw={4.5} />
            {/* 右侧结论 */}
            <g transform="translate(1450,735)">
              <circle r={64} fill={P.bgDeep} stroke={P.cyan} strokeWidth={5} />
              <text x={0} y={-4} textAnchor="middle" fontSize={30} fill={P.cyan}
                fontFamily={FONT_HEAD} fontWeight={900}>同款</text>
              <text x={0} y={32} textAnchor="middle" fontSize={24} fill={P.cream}
                fontFamily={FONT_HEAD} fontWeight={700}>REM</text>
              <text x={0} y={112} textAnchor="middle" fontSize={23} fill={P.muted}
                fontFamily={FONT_HEAD}>做梦那个状态 · 两物种都有</text>
            </g>
          </g>
        )}
      </svg>
      <Chip text="REM = 快速眼动期" x={1410} y={66} at={b1 + 40} accent={P.vio} />
      <Stamp text="脑电这一关 · 过了" x={600} y={86} at={b2 + 50} color={P.vioD} size={40} rot={-3} />
    </FadeIn>
  );
};

/* ============ S2 把梦演出来的猫：茹韦脑干实验 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const headPop = Math.max(0.001, spring({frame: f - 12, fps, config: {damping: 12, stiffness: 110}}));
  const lesionK = clamp01((f - b1 - 6) / 20);
  const lockOpen = clamp01((f - b1 - 18) / 22);
  const actPop = Math.max(0.001, spring({frame: f - b1 - 30, fps, config: {damping: 11, stiffness: 110}}));
  const bubPop = Math.max(0.001, spring({frame: f - b1 - 55, fps, config: {damping: 11, stiffness: 120}}));
  const flash = lesionK > 0 && lesionK < 1 ? Math.abs(Math.sin(f * 0.9)) : 0;

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={2} name={sec.name} accent={P.coral} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：脑干剖面 */}
        <rect x={120} y={150} width={780} height={730} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <text x={156} y={204} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
          茹韦实验 · 1960s · 脑干里有一把"肌张力锁"</text>
        <g transform={`translate(470,480) scale(${headPop})`} opacity={clamp01(headPop)}>
          {/* 猫头侧影（朝右） */}
          <circle r={160} fill={P.cat} />
          <polygon points="-60,-122 -110,-196 -6,-156" fill={P.cat} />
          <polygon points="30,-142 56,-210 112,-152" fill={P.cat} />
          <polygon points="-66,-140 -96,-184 -26,-156" fill={P.catD} />
          {/* 脑（透视） */}
          <ellipse cx={-30} cy={-52} rx={102} ry={58} fill={P.vio} opacity={0.92} />
          <path d="M -104 -60 q 24 -22 48 -6 q 22 14 44 0 q 20 -12 40 4"
            stroke={P.vioD} strokeWidth={5} fill="none" strokeLinecap="round" />
          <path d="M -96 -34 q 22 -12 44 2" stroke={P.vioD} strokeWidth={5} fill="none" strokeLinecap="round" />
          {/* 脑干 */}
          <g transform="translate(72,22) rotate(26)">
            <rect x={-34} y={-16} width={68} height={32} rx={14}
              fill={lockOpen > 0.5 ? P.coral : P.teal}
              stroke={flash > 0.4 ? P.cream : 'none'} strokeWidth={4} />
          </g>
          {/* 脊髓→肌肉 线 */}
          <path d="M 96 48 L 168 150" stroke={P.muted} strokeWidth={7} strokeLinecap="round" />
          {/* 闭眼睡颜 */}
          <path d="M -8 10 q 9 9 18 0" stroke={P.ink} strokeWidth={4.4} fill="none" strokeLinecap="round" />
          <polygon points="128,2 148,8 130,16" fill={P.coralD} />
          {/* 损毁标记 */}
          {lesionK > 0 && (
            <g opacity={lesionK} stroke={P.coral} strokeWidth={7} strokeLinecap="round">
              <line x1={44} y1={-8} x2={72} y2={20} />
              <line x1={72} y1={-8} x2={44} y2={20} />
            </g>
          )}
        </g>
        {/* 锁 */}
        <PadLock x={672} y={668} s={1.05} open={lockOpen} col={lockOpen > 0.5 ? P.coral : P.teal} />
        {/* 引线标注 */}
        <line x1={580} y1={470} x2={690} y2={424} stroke={P.muted} strokeWidth={3} />
        <text x={700} y={432} fontSize={23} fill={lockOpen > 0.5 ? P.coral : P.teal}
          fontFamily={FONT_HEAD} fontWeight={800}>
          脑干·肌张力锁
        </text>
        <line x1={660} y1={660} x2={560} y2={740} stroke={P.muted} strokeWidth={3} />
        <text x={330} y={772} fontSize={23} fill={P.muted} fontFamily={FONT_HEAD}>
          锁住 → 信号到不了肌肉</text>
        <text x={156} y={840} fill={P.cream} fontSize={27} fontFamily={FONT_HEAD} fontWeight={800}>
          {lockOpen > 0.5 ? '损毁脑干 → 锁没了 → 梦漏到行为层' : '正常 REM：全身肌肉被锁死'}
        </text>
        {/* 右：对照两格 */}
        <rect x={940} y={150} width={860} height={730} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        {/* 上格：正常 */}
        <text x={976} y={204} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>REM 正常 · 被锁住只能眼珠乱转</text>
        <SleepCat x={1330} y={330} s={0.62} />
        <PadLock x={1620} y={300} s={0.8} col={P.teal} />
        <text x={1620} y={390} textAnchor="middle" fontSize={21} fill={P.teal}
          fontFamily={FONT_HEAD} fontWeight={700}>肌张力锁 · 正常</text>
        <line x1={976} y1={452} x2={1764} y2={452} stroke={P.card} strokeWidth={4} strokeDasharray="10 10" />
        {/* 下格：损毁后演梦 */}
        {actPop > 0.02 ? (
          <g opacity={clamp01(actPop)}>
            <text x={976} y={512} fill={P.coral} fontSize={25} fontFamily={FONT_HEAD} fontWeight={800}>
              脑干损毁后 · 睡着站起来把梦演出来</text>
            <g transform={`translate(1370,690) scale(${actPop}) translate(-1370,-690)`}>
              <ActCat x={1290} y={700} s={0.92} />
              {/* 梦泡：追鸟 */}
              <g transform={`translate(1620,590) scale(${bubPop})`} opacity={clamp01(bubPop)}>
                <Cloud x={0} y={0} s={0.62}>
                  <Bird x={0} y={-6} s={0.9} />
                </Cloud>
              </g>
            </g>
            {[0, 1].map((i) => (
              <path key={i} d={`M ${1050 + i * 24} ${720 + i * 20} q -24 ${12 + Math.sin(f * 0.7 + i) * 6} -40 8`}
                stroke={P.coral} strokeWidth={5} fill="none" strokeLinecap="round" opacity={0.6 - i * 0.22} />
            ))}
            <g transform="translate(1650,800)">
              <rect x={-140} y={-38} width={280} height={76} rx={16} fill={P.coral} />
              <text x={0} y={12} textAnchor="middle" fontSize={27} fill={P.cream}
                fontFamily={FONT_HEAD} fontWeight={900}>扑咬·追逐·理毛</text>
            </g>
          </g>
        ) : (
          <g opacity={0.55}>
            <text x={976} y={512} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
              如果把锁拆掉呢？</text>
            <text x={1370} y={680} textAnchor="middle" fontSize={60} fill={P.card}
              fontFamily={FONT_HEAD} fontWeight={900}>?</text>
          </g>
        )}
      </svg>
      <Chip text="行为泄漏 · 直接证据" x={1390} y={66} at={40} accent={P.coral} />
      <Stamp text="不是抽搐 · 是表演" x={560} y={86} at={b1 + 90} color={P.coralD} size={40} rot={-3} />
    </FadeIn>
  );
};

/* ---------- 迷宫路径 ---------- */
const MAZE_PTS: [number, number][] = [
  [220, 302], [578, 302], [578, 412], [212, 412], [212, 532], [578, 532], [578, 662], [300, 662],
];
const pathPos = (u: number): [number, number, number] => {
  const uu = clamp01(u) * (MAZE_PTS.length - 1);
  const seg = Math.min(MAZE_PTS.length - 2, Math.floor(uu));
  const t = uu - seg;
  const [x0, y0] = MAZE_PTS[seg];
  const [x1, y1] = MAZE_PTS[seg + 1];
  return [x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, Math.atan2(y1 - y0, x1 - x0) * 180 / Math.PI];
};

/* ============ S3 老鼠把迷宫重跑一遍 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);
  const mazePop = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 12, stiffness: 110}}));
  const runK = clamp01((f - b1 - 8) / 120);
  const [rx, ry, rr] = pathPos(runK);
  const recK = clamp01((f - b1 - 6) / 24);
  const repK = clamp01((f - b2 - 6) / 24);
  const sweep = repK > 0 ? ((f - b2 - 14) / 60) % 1 : 0;

  const neuronRow = (yc: number, litIdx: number, sweepMode: boolean, col: string) => (
    <g>
      {[0, 1, 2, 3, 4].map((i) => {
        const x = 850 + i * 200;
        const lit = sweepMode ? i === Math.floor(sweep * 5) % 5 : i <= litIdx;
        return (
          <g key={i}>
            {i < 4 && (
              <line x1={x + 40} y1={yc} x2={x + 160} y2={yc} stroke={P.card} strokeWidth={5} />
            )}
            {i < 4 && (
              <polygon points={`${x + 152},${yc - 9} ${x + 168},${yc} ${x + 152},${yc + 9}`}
                fill={P.card} />
            )}
            {/* 放电尖峰 */}
            {lit && (
              <path d={`M ${x} ${yc - 38} l 7 -18 l 7 24 l 7 -14`}
                stroke={P.amber} strokeWidth={4.5} fill="none" strokeLinecap="round" />
            )}
            <circle cx={x} cy={yc} r={36} fill={lit ? col : P.bgDeep}
              stroke={lit ? P.cream : col} strokeWidth={4} />
            <text x={x} y={yc + 10} textAnchor="middle" fontSize={28}
              fill={lit ? P.ink : col} fontFamily={FONT_HEAD} fontWeight={900}>
              {'ABCDE'[i]}
            </text>
          </g>
        );
      })}
    </g>
  );

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={3} name={sec.name} accent={P.amber} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：迷宫 */}
        <g transform={`translate(395,470) scale(${mazePop}) translate(-395,-470)`} opacity={clamp01(mazePop)}>
          <rect x={120} y={150} width={550} height={640} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={156} y={204} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>白天 · 老鼠跑迷宫（俯视）</text>
          {/* 迷宫 */}
          <rect x={170} y={252} width={460} height={460} rx={10} fill={P.bgDeep}
            stroke={P.muted} strokeWidth={5} />
          <rect x={170} y={344} width={372} height={14} fill={P.muted} />
          <rect x={230} y={462} width={400} height={14} fill={P.muted} />
          <rect x={170} y={582} width={372} height={14} fill={P.muted} />
          {/* 路线虚线 */}
          <polyline points={MAZE_PTS.map((p) => p.join(',')).join(' ')} fill="none"
            stroke={P.amber} strokeWidth={5} strokeDasharray="3 14" strokeLinecap="round" opacity={0.85} />
          {/* 起点旗 / 终点奶酪 */}
          <g transform="translate(220,302)">
            <line x1={0} y1={0} x2={0} y2={-34} stroke={P.cream} strokeWidth={5} />
            <polygon points="0,-34 30,-27 0,-18" fill={P.coral} />
          </g>
          <g transform="translate(300,662)">
            <path d="M -20 8 L 22 -8 L 22 14 Z" fill={P.amber} />
            <circle cx={-2} cy={4} r={3.4} fill={P.amberD} />
            <circle cx={10} cy={2} r={2.6} fill={P.amberD} />
          </g>
          {/* 老鼠 */}
          {runK > 0 && <Rat x={rx} y={ry - 14} rot={rr} s={0.8} />}
          {runK >= 1 && (
            <text x={395} y={756} textAnchor="middle" fontSize={24} fill={P.amber}
              fontFamily={FONT_HEAD} fontWeight={800}>跑完 → 记下放电顺序</text>
          )}
        </g>
        {/* 右上：白天记录 */}
        <g opacity={recK} transform={`translate(0,${(1 - recK) * 40})`}>
          <rect x={710} y={150} width={1090} height={310} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={746} y={204} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            海马神经元的放电顺序 · 白天</text>
          {neuronRow(330, Math.floor(runK * 5.99), false, P.teal)}
          <text x={746} y={420} fill={P.muted} fontSize={22} fontFamily={FONT_HEAD}>
            跑过哪段路 → 哪批细胞按序放电</text>
        </g>
        {/* 右下：睡着重放（cue2） */}
        {repK > 0 && (
          <g opacity={repK} transform={`translate(0,${(1 - repK) * 50})`}>
            <rect x={710} y={500} width={1090} height={290} rx={20} fill={P.panel} stroke={P.amber} strokeWidth={4} />
            <text x={746} y={554} fill={P.amber} fontSize={25} fontFamily={FONT_HEAD} fontWeight={800}>
              睡着后 · 同一批神经元按原顺序重放</text>
            {neuronRow(646, 0, true, P.amber)}
            {/* 快进徽标 */}
            <g transform="translate(1696,556)">
              <rect x={-64} y={-34} width={128} height={68} rx={14} fill={P.bgDeep} stroke={P.amber} strokeWidth={4} />
              <text x={0} y={10} textAnchor="middle" fontSize={27} fill={P.amber}
                fontFamily={FONT_HEAD} fontWeight={900}>≈快进</text>
            </g>
            <text x={746} y={756} fill={P.cream} fontSize={26} fontFamily={FONT_HEAD} fontWeight={800}>
              梦不是乱码 · 带着白天路线的结构 <tspan fill={P.muted}>≈ 梦里把迷宫又跑了一遍</tspan>
            </text>
          </g>
        )}
      </svg>
      <Chip text="海马 · 记忆重放" x={1420} y={66} at={b1 + 20} accent={P.amber} />
      <Stamp text="睡着还在跑迷宫" x={600} y={86} at={b2 + 70} color={P.amberD} size={40} rot={-3} />
    </FadeIn>
  );
};

/* ============ S4 确定在做梦 · 但内容是黑箱 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);
  const pops = [0, 1, 2, 3].map((i) => Math.max(0.001, spring({frame: f - 12 - i * 14, fps, config: {damping: 12, stiffness: 130}})));
  const handK = clamp01((f - b1 - 8) / 24);
  const stopPop = Math.max(0.001, spring({frame: f - b1 - 46, fps, config: {damping: 10, stiffness: 150}}));
  const boxK = clamp01((f - b2 - 6) / 24);
  const hx = interpolate(clamp01((f - b1 - 10) / 44), [0, 1], [176, 300]);

  const cards = [
    {kind: 'cat', name: '猫', frac: 0.85, note: 'REM≈两成多 · 幼崽更多', col: P.cat},
    {kind: 'dog', name: '狗', frac: 0.68, note: '比猫少一点', col: P.dog},
    {kind: 'bird', name: '鸟', frac: 0.4, note: '有 · 短而碎', col: '#7ab8ff'},
    {kind: 'octo', name: '章鱼', frac: 0.5, note: '变色 + 触手抽动', col: P.vio},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={4} name={sec.name} accent={P.teal} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 上排：会做梦的物种卡 */}
        {cards.map((c, i) => {
          const x = 180 + i * 400;
          return (
            <g key={i} transform={`translate(${x},160) scale(${pops[i]}) translate(${-x},-160)`}
              opacity={clamp01(pops[i] * 2)}>
              <rect x={x} y={160} width={360} height={186} rx={18} fill={P.panel} stroke={c.col} strokeWidth={4} />
              <PetIcon kind={c.kind} x={x + 66} y={238} col={c.col} />
              <text x={x + 118} y={222} fill={P.cream} fontSize={30} fontFamily={FONT_HEAD} fontWeight={800}>{c.name}</text>
              <rect x={x + 118} y={246} width={200} height={18} rx={9} fill={P.card} />
              <rect x={x + 118} y={246} width={200 * c.frac} height={18} rx={9} fill={c.col} />
              <text x={x + 118} y={304} fill={P.muted} fontSize={21} fontFamily={FONT_HEAD}>{c.note}</text>
            </g>
          );
        })}
        <text x={180} y={392} fill={P.muted} fontSize={24} fontFamily={FONT_HEAD}>
          REM 占比 · 幼崽和小体型普遍更高 —— 会做梦的不只猫狗</text>
        {/* 左下：别摇醒 */}
        <g opacity={handK} transform={`translate(0,${(1 - handK) * 50})`}>
          <rect x={120} y={430} width={840} height={450} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={156} y={484} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            蹬腿呜咽 = 梦里动作漏出来的一小部分</text>
          <SleepDog x={560} y={680} s={0.82} />
          {handK > 0.3 && <Palm x={hx} y={676} s={0.8} />}
          {stopPop > 0.02 && (
            <g transform={`translate(316,596) scale(${stopPop})`} opacity={clamp01(stopPop)}>
              <circle r={58} fill="none" stroke={P.coral} strokeWidth={9} />
              <line x1={-38} y1={38} x2={38} y2={-38} stroke={P.coral} strokeWidth={9} strokeLinecap="round" />
            </g>
          )}
          <text x={156} y={838} fill={P.coral} fontSize={27} fontFamily={FONT_HEAD} fontWeight={800}>
            别猛摇醒它 · 睡梦中被碰会受惊咬人</text>
        </g>
        {/* 右下：黑箱 */}
        <g opacity={boxK} transform={`translate(0,${(1 - boxK) * 50})`}>
          <rect x={1000} y={430} width={800} height={450} rx={20} fill={P.panel} stroke={P.vio} strokeWidth={4} />
          <text x={1036} y={484} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            诚实的边界 · 只能推断不能读取</text>
          <rect x={1120} y={540} width={560} height={230} rx={18} fill="#0d0f24"
            stroke={P.vioD} strokeWidth={4} strokeDasharray="12 10" />
          {[0, 1, 2].map((i) => (
            <text key={i} x={1260 + i * 140} y={680 + Math.sin(f * 0.5 + i * 2.1) * 12}
              textAnchor="middle" fontSize={64 - i * 8} fill={P.vio} fontFamily={FONT_HEAD}
              fontWeight={900} opacity={0.5 + 0.4 * Math.abs(Math.sin(f * 0.4 + i))}>?</text>
          ))}
          <text x={1400} y={586} textAnchor="middle" fontSize={24} fill={P.muted}
            fontFamily={FONT_HEAD}>梦见了什么 · 永久黑箱</text>
          <text x={1036} y={838} fill={P.cream} fontSize={27} fontFamily={FONT_HEAD} fontWeight={800}>
            确定<tspan fill={P.teal}>「在做梦」</tspan> · 不知道<tspan fill={P.vio}>「梦见什么」</tspan>
          </text>
        </g>
      </svg>
      <Chip text="做梦很普遍" x={1470} y={66} at={50} accent={P.teal} />
      <Stamp text="做给自己看的梦" x={600} y={86} at={b2 + 60} color={P.teal} size={40} rot={-3} />
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
        <Moon x={1620} y={180} scale={0.55} opacity={0.9} />
        <g transform={`translate(960,700) scale(${pop}) translate(-960,-700)`} opacity={clamp01(pop)}>
          <ellipse cx={960} cy={700} rx={480} ry={86} fill={P.card} />
          <ellipse cx={960} cy={693} rx={440} ry={72} fill={P.bgLite} />
          <SleepDog x={760} y={656} s={0.72} />
          <SleepCat x={1180} y={650} s={0.7} />
          <Cloud x={970} y={480} s={0.62}>
            <Fish x={-56} y={4} s={0.8} />
            <Bone x={52} y={10} s={0.7} rot={-12} />
          </Cloud>
          <Zzz x={620} y={560} k={0.75} col={P.vio} />
          <Zzz x={1330} y={555} k={0.75} col={P.cyan} />
        </g>
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -320}}>
        <div style={{fontSize: 56, fontWeight: 800, color: C.text}}>做梦不是人类的特权</div>
        <div style={{fontSize: 36, color: P.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E233 = ({data = tl}: {data?: any}) => {
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
