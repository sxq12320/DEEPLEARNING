import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01, rand} from '../lib/style';
import {Backdrop, Ripple, Stamp, SceneTag, Chip, Subtitles, ProgressBar, FadeIn, Moon, Stars} from '../lib/atoms';
import tl from '../timeline_224.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：深夜卧室 · 靛紫 × 暖被 × 珊瑚警报（纯色平涂） ============ */
const P = {
  bg: '#322a5e',        // 夜紫
  bgDeep: '#231d45',    // 更深夜紫
  bgLite: '#403672',    // 浅夜紫
  card: '#2a2350',      // 卡片底
  cardLite: '#372f66',
  ink: '#241d49',       // 深紫字
  cream: '#f7f0dd',     // 米白
  sheet: '#f2e7d3',     // 床单
  blanket: '#5f83e0',   // 被子蓝
  blanketD: '#4160b8',
  pillow: '#fff6e2',
  wood: '#8a5a3c',      // 床架木
  woodD: '#63402a',
  skin: '#ffcf9e',
  skinD: '#e8a76f',
  brain: '#ffb3c7',     // 脑
  brainD: '#e77fa0',
  stem: '#ffc94d',      // 脑干/信号 琥珀
  coral: '#ff6b6b',     // 杏仁核/警报
  coralD: '#d94f4f',
  mint: '#7be0c3',      // 解锁/通过
  metal: '#96a0bd',     // 锁
  metalD: '#6b7494',
  ghost: '#171233',     // 影子
  ghostHi: '#2c2453',
  cat: '#6a63b3',       // 猫
  catD: '#453f80',
  catHi: '#8f88d4',
  moon: '#ffe9a8',
  wave: '#6fd3ff',      // 脑电
  waveD: '#8f9bb5',     // 肌电
  muted: '#a8a4cf',     // 弱字
  sun: '#ffc94d',
};

/* ---------- 挂锁：k 出现进度，open 开锁 ---------- */
const Padlock = ({x, y, s = 1, k = 1, open = 0, col = P.metal}: {
  x: number; y: number; s?: number; k?: number; open?: number; col?: string;
}) => {
  const kk = Math.max(0.001, clamp01(k));
  const op = clamp01(open);
  return (
    <g transform={`translate(${x},${y}) scale(${s * kk})`} opacity={kk > 0.02 ? 1 : 0}>
      <g transform={`rotate(${-75 * op} -16 -4)`}>
        <path d="M -16 -4 L -16 -22 Q -16 -40 0 -40 Q 16 -40 16 -22 L 16 -4"
          stroke={col} strokeWidth={9} fill="none" strokeLinecap="round" />
      </g>
      <rect x={-22} y={-8} width={44} height={38} rx={9} fill={col} />
      <circle cy={9} r={5.5} fill={P.card} />
      <rect x={-2.5} y={12} width={5} height={11} rx={2.5} fill={P.card} />
    </g>
  );
};

/* ---------- 抑制信号闪电：x1,y1 → x2,y2 锯齿 ---------- */
const Bolt = ({x1, y1, x2, y2, k = 1, col = P.stem, w = 6, n = 4}: {
  x1: number; y1: number; x2: number; y2: number; k?: number; col?: string; w?: number; n?: number;
}) => {
  const kk = clamp01(k);
  if (kk <= 0.01) return null;
  let d = `M ${x1} ${y1}`;
  for (let i = 1; i <= n * 2; i++) {
    const t = (i / (n * 2)) * kk;
    const px = x1 + (x2 - x1) * t + (i % 2 ? 14 : -14) * (i < n * 2 ? 1 : 0);
    const py = y1 + (y2 - y1) * t;
    d += ` L ${px.toFixed(1)} ${py.toFixed(1)}`;
  }
  const ex = x1 + (x2 - x1) * kk, ey = y1 + (y2 - y1) * kk;
  return (
    <g>
      <path d={d} stroke={col} strokeWidth={w} fill="none" strokeLinecap="round" strokeLinejoin="round" />
      {kk > 0.85 && <path d={`M ${ex - 11} ${ey - 16} L ${ex} ${ey} L ${ex + 11} ${ey - 16}`}
        stroke={col} strokeWidth={w} fill="none" strokeLinecap="round" strokeLinejoin="round" />}
    </g>
  );
};

/* ---------- 侧躺小人（床里）：awake 睁着眼 ---------- */
const LyingPerson = ({x, y, s = 1, awake = false, blanket = P.blanket}: {
  x: number; y: number; s?: number; awake?: boolean; blanket?: string;
}) => {
  const f = useCurrentFrame();
  const breathe = Math.sin(f * 0.09) * 4;
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      {/* 头 */}
      <circle cx={0} cy={0} r={42} fill={P.skin} />
      <path d="M -42 -6 A 42 42 0 0 1 40 -22 L 40 -34 Q 0 -52 -42 -34 Z" fill={P.ink} />
      {awake ? (
        <g>
          <ellipse cx={-16} cy={-4} rx={7.5} ry={9} fill="#fff" />
          <ellipse cx={16} cy={-4} rx={7.5} ry={9} fill="#fff" />
          <circle cx={-15} cy={-3} r={3.4} fill={P.ink} />
          <circle cx={17} cy={-3} r={3.4} fill={P.ink} />
          <path d="M -25 -20 Q -16 -26 -7 -21 M 7 -21 Q 16 -26 25 -20" stroke={P.ink} strokeWidth={4} fill="none" strokeLinecap="round" />
        </g>
      ) : (
        <g>
          <path d="M -23 -3 Q -16 3 -9 -3 M 9 -3 Q 16 3 23 -3" stroke={P.ink} strokeWidth={4.5} fill="none" strokeLinecap="round" />
        </g>
      )}
      {/* 被子裹住的身体（随呼吸起伏） */}
      <g transform={`translate(0,${-breathe * 0.4})`}>
        <path d={`M 34 6 L 560 6 Q 596 10 596 44 L 596 ${78 + breathe} L 30 ${78 + breathe} Q 26 40 34 6 Z`}
          fill={blanket} />
        <path d={`M 34 6 L 560 6 Q 596 10 596 44 L 596 58 L 30 58 Q 26 40 34 6 Z`} fill={P.blanketD} opacity={0.55} />
        <path d="M 120 6 L 120 78 M 300 6 L 300 78 M 480 6 L 480 78" stroke={P.blanketD} strokeWidth={5} opacity={0.4} />
      </g>
    </g>
  );
};

/* ---------- 床 ---------- */
const Bed = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-70} y={-160} width={26} height={170} rx={10} fill={P.woodD} />
    <rect x={-70} y={-170} width={26} height={26} rx={10} fill={P.wood} />
    <rect x={-70} y={-6} width={720} height={34} rx={14} fill={P.wood} />
    <rect x={-40} y={28} width={24} height={52} rx={8} fill={P.woodD} />
    <rect x={600} y={28} width={24} height={52} rx={8} fill={P.woodD} />
    <rect x={-60} y={-46} width={700} height={46} rx={18} fill={P.sheet} />
    <ellipse cx={-2} cy={-40} rx={72} ry={30} fill={P.pillow} />
    <ellipse cx={-2} cy={-32} rx={72} ry={16} fill={P.sheet} />
  </g>
);

/* ---------- 床边黑影（“感知存在”） ---------- */
const ShadowFigure = ({x, y, h = 420, k = 1, lean = 0}: {x: number; y: number; h?: number; k?: number; lean?: number}) => {
  const kk = clamp01(k);
  if (kk <= 0.01) return null;
  return (
    <g transform={`translate(${x},${y}) rotate(${lean})`} opacity={kk * 0.96}>
      <path d={`M 0 ${-h} Q ${-h * 0.26} ${-h} ${-h * 0.26} ${-h * 0.82}
        Q ${-h * 0.30} ${-h * 0.66} ${-h * 0.22} ${-h * 0.6}
        L ${-h * 0.2} 0 Q 0 ${h * 0.06} ${h * 0.2} 0
        L ${h * 0.22} ${-h * 0.6} Q ${h * 0.3} ${-h * 0.66} ${h * 0.26} ${-h * 0.82}
        Q ${h * 0.26} ${-h} 0 ${-h} Z`} fill={P.ghost} />
      <ellipse cx={-h * 0.075} cy={-h * 0.83} rx={h * 0.028} ry={h * 0.02} fill={P.cream} opacity={0.85} />
      <ellipse cx={h * 0.075} cy={-h * 0.83} rx={h * 0.028} ry={h * 0.02} fill={P.cream} opacity={0.85} />
    </g>
  );
};

/* ---------- 猫（做梦扑咬） ---------- */
const Cat = ({x, y, s = 1, hop = 0, flip = 1}: {x: number; y: number; s?: number; hop?: number; flip?: number}) => {
  const f = useCurrentFrame();
  const hy = hop > 0 ? Math.abs(Math.sin(f * 0.3)) * 26 * hop : 0;
  return (
    <g transform={`translate(${x},${y - hy}) scale(${flip * s},${s})`}>
      <path d="M 62 -30 Q 108 -56 116 -18 Q 118 -4 104 0" stroke={P.cat} strokeWidth={13} fill="none" strokeLinecap="round" />
      <ellipse rx={64} ry={38} fill={P.cat} />
      <ellipse cx={-10} cy={-14} rx={52} ry={26} fill={P.catHi} />
      <circle cx={-56} cy={-34} r={30} fill={P.cat} />
      <path d="M -78 -56 L -84 -84 L -58 -64 Z M -38 -58 L -30 -86 L -16 -60 Z" fill={P.catD} />
      <path d="M -66 -34 Q -60 -28 -54 -34" stroke={P.cream} strokeWidth={4} fill="none" strokeLinecap="round" />
      <path d="M -96 -30 L -76 -32 M -96 -22 L -78 -26" stroke={P.cream} strokeWidth={3.5} strokeLinecap="round" opacity={0.8} />
      <rect x={-40} y={18} width={14} height={26} rx={7} fill={P.catD} transform={`rotate(${Math.sin(f * 0.3) * 8} -33 18)`} />
      <rect x={18} y={18} width={14} height={26} rx={7} fill={P.catD} />
    </g>
  );
};

/* ---------- 火柴小人（患病率统计） ---------- */
const StickFig = ({x, y, s = 1, col = P.metal, k = 1, exclaim = false}: {
  x: number; y: number; s?: number; col?: string; k?: number; exclaim?: boolean;
}) => {
  const kk = Math.max(0.001, clamp01(k));
  return (
    <g transform={`translate(${x},${y}) scale(${s * kk})`} opacity={kk > 0.02 ? 1 : 0}>
      <circle cy={-52} r={13} fill={col} />
      <line x1={0} y1={-38} x2={0} y2={-4} stroke={col} strokeWidth={9} strokeLinecap="round" />
      <line x1={-20} y1={-30} x2={20} y2={-30} stroke={col} strokeWidth={8} strokeLinecap="round" />
      <line x1={0} y1={-4} x2={-15} y2={26} stroke={col} strokeWidth={8} strokeLinecap="round" />
      <line x1={0} y1={-4} x2={15} y2={26} stroke={col} strokeWidth={8} strokeLinecap="round" />
      {exclaim && (
        <g transform="translate(0,-86)">
          <rect x={-3.5} y={-14} width={7} height={18} rx={3.5} fill={P.coral} />
          <circle cy={12} r={4.5} fill={P.coral} />
        </g>
      )}
    </g>
  );
};

/* ---------- 眼睛图标 ---------- */
const EyeIcon = ({x, y, s = 1, col = P.mint}: {x: number; y: number; s?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <path d="M -34 0 Q 0 -30 34 0 Q 0 30 -34 0 Z" fill="none" stroke={col} strokeWidth={6} />
    <circle r={11} fill={col} />
    <circle cx={3.5} cy={-3.5} r={3.5} fill={P.card} />
  </g>
);

/* ---------- 肺图标 ---------- */
const LungsIcon = ({x, y, s = 1, col = P.mint}: {x: number; y: number; s?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} fill={col}>
    <rect x={-4} y={-34} width={8} height={24} rx={4} />
    <path d="M -6 -14 Q -34 -12 -38 12 Q -40 26 -28 26 L -8 26 Q -4 26 -4 14 Z" />
    <path d="M 6 -14 Q 34 -12 38 12 Q 40 26 28 26 L 8 26 Q 4 26 4 14 Z" />
  </g>
);

/* ---------- 脑侧影 ---------- */
const BrainBlob = ({x, y, r = 150, k = 1}: {x: number; y: number; r?: number; k?: number}) => {
  const kk = Math.max(0.001, clamp01(k));
  return (
    <g transform={`translate(${x},${y}) scale(${kk})`} opacity={kk > 0.02 ? 1 : 0}>
      <path d={`M ${-r} 0 Q ${-r} ${-r * 0.95} ${-r * 0.2} ${-r * 0.98}
        Q ${r * 0.55} ${-r * 1.05} ${r * 0.9} ${-r * 0.42}
        Q ${r * 1.12} ${-r * 0.1} ${r * 0.82} ${r * 0.32}
        Q ${r * 0.6} ${r * 0.66} ${r * 0.1} ${r * 0.62}
        Q ${-r * 0.4} ${r * 0.6} ${-r * 0.72} ${r * 0.34}
        Q ${-r * 1.05} ${r * 0.2} ${-r} 0 Z`} fill={P.brain} />
      {[[-0.5, -0.5], [0.1, -0.62], [0.55, -0.3], [-0.35, 0.05], [0.25, 0.12], [-0.05, -0.25]].map((q, i) => (
        <path key={i}
          d={`M ${q[0] * r - 26} ${q[1] * r} q 14 -16 30 -6 q 12 8 26 -4`}
          stroke={P.brainD} strokeWidth={6} fill="none" strokeLinecap="round" />
      ))}
    </g>
  );
};

/* ---------- 勾选/叉 ---------- */
const CheckDot = ({x, y, r = 22, k = 1, col = P.mint}: {x: number; y: number; r?: number; k?: number; col?: string}) => {
  const kk = Math.max(0.001, clamp01(k));
  return (
    <g transform={`translate(${x},${y}) scale(${kk})`} opacity={kk > 0.02 ? 1 : 0}>
      <circle r={r} fill={col} />
      <path d={`M ${-r * 0.45} 0 L ${-r * 0.1} ${r * 0.38} L ${r * 0.5} ${-r * 0.36}`}
        stroke={P.ink} strokeWidth={6} fill="none" strokeLinecap="round" strokeLinejoin="round" />
    </g>
  );
};

/* ---------- 波带：一条可摆动的折线 ---------- */
const wavePts = (x0: number, x1: number, y: number, amp: number, freq: number, phase: number) => {
  const n = 36;
  let d = '';
  for (let i = 0; i <= n; i++) {
    const px = x0 + (x1 - x0) * (i / n);
    const py = y + Math.sin(i * freq + phase) * amp * (0.6 + 0.4 * Math.sin(i * 0.7 + phase));
    d += `${i ? 'L' : 'M'} ${px.toFixed(1)} ${py.toFixed(1)} `;
  }
  return d;
};

/* ============ 标题卡 ============ */
const TitleCard = ({data}: {data: any}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 13, mass: 0.7}});
  const pop = spring({frame: f - 6, fps, config: {damping: 10, stiffness: 140}});
  const ghostK = clamp01((f - 26) / 20);
  const o = interpolate(f, [0, 8], [0, 1]);
  return (
    <AbsoluteFill>
      <Backdrop top={P.bg} bottom={P.bg} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <circle cx={960} cy={348} r={212} fill={P.cardLite} />
        <g transform={`translate(960,392) scale(${clamp01(pop)})`}>
          {/* 床 + 睁眼的人 + 石板 + 黑影 */}
          <g transform="scale(0.36) translate(-290,-60)">
            <Bed x={0} y={0} s={1} />
            <LyingPerson x={0} y={-66} s={1} awake />
            <rect x={180} y={-64} width={190} height={56} rx={14} fill={P.metalD} transform="rotate(-3 275 -36)" />
            <rect x={196} y={-56} width={60} height={12} rx={6} fill={P.metal} opacity={0.8} transform="rotate(-3 275 -36)" />
          </g>
          <g transform="translate(96,-40) scale(0.4)" opacity={ghostK}>
            <ShadowFigure x={0} y={0} h={300} k={1} lean={-6} />
          </g>
          <g transform="translate(-150,-138) scale(0.62)">
            <Moon x={0} y={0} scale={1} />
          </g>
        </g>
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 268}}>
          <div style={{color: P.stem, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 26}}>
            冷知识 · 第 {data.episode} 期
          </div>
          <div style={{color: C.text, fontSize: 80, fontWeight: 900, fontFamily: FONT_HEAD, letterSpacing: 5}}>
            {data.title}
          </div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ============ S0 清醒地动不了：卧室 · 睁眼 · 石板 · 黑影 → 三重真实 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1); // ≈271f “这不是灵异事件…”

  const roomK = spring({frame: f - 6, fps, config: {damping: 13, stiffness: 120}});
  const eyeK = clamp01((f - 30) / 14);
  const slabK = spring({frame: f - 78, fps, config: {damping: 9, stiffness: 200, mass: 0.9}});
  const slabS = 1 - (1 - clamp01(slabK)) * 0.9;
  const ghostK = clamp01((f - 195) / 34);
  const ghostX = interpolate(f, [195, 330], [1560, 1330], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const stampK = clamp01((f - b1 - 10) / 16);
  const facts = [0, 1, 2].map((i) => clamp01(spring({frame: f - b1 - 60 - i * 22, fps, config: {damping: 12, stiffness: 140}})));

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={0} name={sec.name} accent={P.moon} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 窗 + 月 + 星 */}
        <g opacity={clamp01(roomK)}>
          <rect x={140} y={170} width={330} height={330} rx={26} fill={P.bgDeep} stroke={P.wood} strokeWidth={12} />
          <line x1={305} y1={182} x2={305} y2={488} stroke={P.wood} strokeWidth={10} />
          <line x1={152} y1={335} x2={458} y2={335} stroke={P.wood} strokeWidth={10} />
          <g transform="translate(0,0)">
            <Moon x={230} y={255} scale={0.8} />
          </g>
          {[0, 1, 2, 3].map((i) => (
            <circle key={i} cx={330 + rand(i, 1) * 110} cy={215 + rand(i, 2) * 110} r={3.4}
              fill="#fff" opacity={0.5 + Math.sin(f / 8 + i) * 0.3} />
          ))}
        </g>

        {/* 床 + 睁眼的人 */}
        <g transform={`translate(300,742)`} opacity={clamp01(roomK)}>
          <Bed x={0} y={0} s={1} />
          <g transform="translate(0,-66)">
            <LyingPerson x={0} y={0} s={1} awake={eyeK > 0.5} />
            {/* 石板压胸 */}
            <g transform={`translate(268,${-44}) rotate(-3) scale(${slabS})`} opacity={clamp01(slabK)}>
              <rect x={-96} y={-30} width={192} height={60} rx={14} fill={P.metalD} />
              <rect x={-78} y={-20} width={62} height={12} rx={6} fill={P.metal} opacity={0.8} />
              <rect x={-10} y={-20} width={90} height={12} rx={6} fill={P.metal} opacity={0.5} />
            </g>
            {clamp01(slabK) > 0.6 && (
              <g transform="translate(268,-96)">
                <text x={0} y={-14} textAnchor="middle" fill={P.cream} fontSize={26} fontFamily={FONT_HEAD} fontWeight={700}>
                  胸口像压石板
                </text>
              </g>
            )}
          </g>
        </g>

        {/* 清醒标记：头顶的“！” */}
        {eyeK > 0.4 && (
          <g transform={`translate(300,586)`} opacity={0.7 + Math.sin(f / 6) * 0.3}>
            <rect x={-4} y={-30} width={8} height={22} rx={4} fill={P.stem} />
            <circle cy={4} r={5} fill={P.stem} />
          </g>
        )}

        {/* 黑影靠近床边 */}
        <g transform="translate(0,822)">
          <ShadowFigure x={ghostX} y={0} h={430} k={ghostK} lean={-5} />
        </g>
        {ghostK > 0.5 && (
          <text x={1600} y={560} textAnchor="middle" fill={P.cream} fontSize={26}
            fontFamily={FONT_HEAD} opacity={ghostK * 0.9}>
            有东西在靠近…
          </text>
        )}

        {/* 揭示：三项全是真的 */}
        <g transform={`translate(0,0)`} opacity={stampK}>
          <rect x={1130} y={120} width={620} height={240} rx={32} fill={P.card} stroke={P.stem} strokeWidth={4} />
          <text x={1440} y={178} textAnchor="middle" fill={P.stem} fontSize={34} fontFamily={FONT_HEAD} fontWeight={800}>
            医学承认：睡眠瘫痪
          </text>
          {['瘫 痪', '压迫感', '恐 惧'].map((t, i) => (
            <g key={i} transform={`translate(${1245 + i * 200},268)`} opacity={facts[i]}>
              <CheckDot x={0} y={0} r={24} k={1} col={P.mint} />
              <text x={0} y={66} textAnchor="middle" fill={P.cream} fontSize={30} fontFamily={FONT_HEAD} fontWeight={700}>
                {t}
              </text>
              <text x={0} y={-40} textAnchor="middle" fill={P.mint} fontSize={22} fontFamily={FONT_HEAD}>
                真的
              </text>
            </g>
          ))}
        </g>
      </svg>
      <Chip text="意识清醒·身体掉线" x={140} y={560} at={60} accent={P.stem} />
      <Stamp text="不是灵异事件" x={1180} y={60} at={b1 + 8} color={P.coral} size={50} rot={-4} />
      <Chip text="三重全是真的" x={950} y={430} at={b1 + 140} accent={P.mint} />
    </FadeIn>
  );
};

/* ============ S1 肌肉早就上了锁：脑干信号 → 全身锁 → 拆锁的猫 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1); // ≈327f “这是防梦游的安全锁…”

  const headK = spring({frame: f - 8, fps, config: {damping: 13, stiffness: 120}});
  const boltK = (i: number) => clamp01((f - 55 - i * 16) / 26);
  const lockK = (i: number) => clamp01(spring({frame: f - 78 - i * 16, fps, config: {damping: 11, stiffness: 160}}));
  const spareK = clamp01((f - 250) / 20); // 只剩眼球呼吸肌
  const cardK = spring({frame: f - b1 - 4, fps, config: {damping: 12, stiffness: 130}});

  const body = {x: 180, y: 740};
  const locks = [
    {x: body.x + 250, y: body.y - 18},
    {x: body.x + 470, y: body.y - 6},
    {x: body.x + 660, y: body.y + 12},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={1} name={sec.name} accent={P.stem} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* ---- 左：头 + 脑 + 脑干 + 下行抑制 ---- */}
        <g opacity={clamp01(headK)}>
          {/* 侧脸轮廓 */}
          <path d="M 270 470 Q 210 470 200 380 Q 190 280 280 250 Q 390 215 450 290 Q 500 350 470 430
            L 450 470 Q 430 500 380 505 L 330 505 Z" fill={P.skin} opacity={0.95} />
          <BrainBlob x={330} y={330} r={112} k={1} />
          {/* 脑干 */}
          <g transform="translate(272,436)">
            <path d="M 0 -14 Q -18 0 -8 26 L -4 70 Q -2 84 10 84 L 18 84 Q 28 84 28 70 L 26 20 Q 24 -6 0 -14 Z"
              fill={P.stem} />
            <circle cx={8} cy={30} r={9} fill={P.coral} opacity={0.8 + Math.sin(f / 5) * 0.2} />
          </g>
          <text x={180} y={560} fill={P.cream} fontSize={28} fontFamily={FONT_HEAD} fontWeight={700} opacity={0.95}>
            脑干·REM期放抑制信号
          </text>
        </g>

        {/* 脊髓虚线 + 信号闪电 */}
        <line x1={282} y1={520} x2={body.x + 90} y2={body.y - 30} stroke={P.stem} strokeWidth={5}
          strokeDasharray="10 12" opacity={0.7} />
        {locks.map((l, i) => (
          <Bolt key={i} x1={body.x + 92} y1={body.y - 32} x2={l.x} y2={l.y - 44} k={boltK(i)} n={2} w={6} />
        ))}

        {/* 躺着的身体 */}
        <g transform={`translate(${body.x},${body.y})`} opacity={clamp01(headK)}>
          <path d="M 0 30 Q -14 -18 30 -34 L 700 -8 Q 760 -4 762 40 Q 762 74 720 78 L 60 78 Q 8 74 0 30 Z"
            fill={P.skin} opacity={0.28} />
          <line x1={30} y1={30} x2={730} y2={44} stroke={P.cream} strokeWidth={5} strokeDasharray="4 14" opacity={0.5} />
        </g>
        {locks.map((l, i) => (
          <Padlock key={i} x={l.x} y={l.y} s={1.05} k={lockK(i)} />
        ))}

        {/* 幸免部位：眼球 + 呼吸肌 */}
        <g opacity={spareK}>
          <rect x={120} y={136} width={560} height={110} rx={24} fill={P.card} stroke={P.mint} strokeWidth={4} />
          <EyeIcon x={195} y={191} s={0.95} />
          <LungsIcon x={300} y={191} s={0.95} />
          <text x={360} y={182} fill={P.cream} fontSize={30} fontFamily={FONT_HEAD} fontWeight={700}>
            只剩它俩能动
          </text>
          <text x={360} y={224} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            眼球 + 呼吸肌（防梦游锁）
          </text>
        </g>

        {/* ---- 右：拆锁实验 · 猫 ---- */}
        <g transform={`translate(${(1 - clamp01(cardK)) * 60},0)`} opacity={clamp01(cardK)}>
          <rect x={1120} y={170} width={690} height={620} rx={40} fill={P.card} stroke={P.coral} strokeWidth={4} />
          <text x={1465} y={230} textAnchor="middle" fill={P.cream} fontSize={33} fontFamily={FONT_HEAD} fontWeight={800}>
            拆掉锁的猫（茹韦实验）
          </text>
          {/* 梦泡：追老鼠 */}
          <g transform="translate(1600,330)" opacity={clamp01((f - b1 - 50) / 18)}>
            <circle r={86} fill={P.cardLite} stroke={P.wave} strokeWidth={3} strokeDasharray="8 10" />
            <ellipse cx={-6} cy={6} rx={34} ry={20} fill={P.metal} />
            <circle cx={24} cy={-2} r={13} fill={P.metal} />
            <circle cx={30} cy={-14} r={5} fill={P.metal} />
            <circle cx={18} cy={-14} r={5} fill={P.metal} />
            <path d="M -38 8 q -26 4 -30 20" stroke={P.metal} strokeWidth={5} fill="none" strokeLinecap="round" />
            <circle cx={29} cy={-3} r={2.6} fill={P.ink} />
          </g>
          <text x={1600} y={452} textAnchor="middle" fill={P.wave} fontSize={25} fontFamily={FONT_HEAD}
            opacity={clamp01((f - b1 - 60) / 18)}>
            梦里在追老鼠
          </text>
          <path d="M 1470 430 q 40 -30 96 -56" stroke={P.wave} strokeWidth={4} strokeDasharray="2 12"
            strokeLinecap="round" opacity={0.7} />
          {/* 猫：睡着但真扑咬 */}
          <Cat x={1350} y={610} s={1.35} hop={f > b1 + 70 ? 1 : 0} />
          <text x={1465} y={742} textAnchor="middle" fill={P.coral} fontSize={28} fontFamily={FONT_HEAD} fontWeight={700}>
            脑干损毁 → 梦直接演成动作
          </text>
        </g>
      </svg>
      <Chip text="REM肌张力抑制" x={140} y={620} at={50} accent={P.stem} />
      <Stamp text="锁=安全装置" x={720} y={140} at={b1 + 8} color={P.stem} size={48} rot={-4} />
      <Chip text="每晚被锁好几次·你不知情" x={150} y={300} at={b1 + 120} accent={P.wave} />
    </FadeIn>
  );
};

/* ============ S2 醒了大半锁没开：脑电/肌电错位带 + 13~14人里1个 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1); // ≈301f “这不算稀奇…”

  const panelK = spring({frame: f - 8, fps, config: {damping: 13, stiffness: 120}});
  const gapK = clamp01((f - 120) / 24);
  const figK = (i: number) => clamp01(spring({frame: f - b1 - 8 - i * 5, fps, config: {damping: 12, stiffness: 170}}));
  const hiK = clamp01((f - b1 - 90) / 18);
  const tagsK = clamp01((f - b1 - 130) / 20);

  const ph = f * 0.16;
  const GX0 = 620, GX1 = 800; // 错位窗口

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={2} name={sec.name} accent={P.wave} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* ---- 左：双波形面板 ---- */}
        <g opacity={clamp01(panelK)}>
          <rect x={110} y={160} width={960} height={600} rx={40} fill={P.card} stroke={P.wave} strokeWidth={4} />
          <text x={590} y={222} textAnchor="middle" fill={P.cream} fontSize={33} fontFamily={FONT_HEAD} fontWeight={800}>
            两边的钟没踩在同一点上
          </text>

          {/* 错位窗口高亮 */}
          <rect x={GX0} y={262} width={GX1 - GX0} height={400} rx={18} fill={P.coral} opacity={0.16 * gapK} />
          <rect x={GX0} y={262} width={GX1 - GX0} height={400} rx={18} fill="none" stroke={P.coral}
            strokeWidth={4} strokeDasharray="12 12" opacity={gapK * 0.9} />
          <text x={(GX0 + GX1) / 2} y={706} textAnchor="middle" fill={P.coral} fontSize={27}
            fontFamily={FONT_HEAD} fontWeight={800} opacity={gapK}>
            怪区间·几秒~2分钟
          </text>

          {/* 脑电：清醒高频 */}
          <text x={170} y={330} fill={P.wave} fontSize={28} fontFamily={FONT_HEAD} fontWeight={700}>脑电</text>
          <text x={170} y={366} fill={P.muted} fontSize={23} fontFamily={FONT_HEAD}>已经清醒</text>
          <path d={wavePts(310, 1010, 340, 46, 1.35, ph)} stroke={P.wave} strokeWidth={6}
            fill="none" strokeLinecap="round" opacity={clamp01((f - 30) / 20)} />

          {/* 肌电：REM 平线 */}
          <text x={170} y={510} fill={P.waveD} fontSize={28} fontFamily={FONT_HEAD} fontWeight={700}>肌电</text>
          <text x={170} y={546} fill={P.muted} fontSize={23} fontFamily={FONT_HEAD}>还在抑制</text>
          <path d={wavePts(310, 1010, 520, 9, 0.5, ph * 0.6)} stroke={P.waveD} strokeWidth={6}
            fill="none" strokeLinecap="round" opacity={clamp01((f - 50) / 20)} />
          {/* 锁标记 */}
          <g transform="translate(300,520)">
            <Padlock x={0} y={0} s={0.8} k={clamp01((f - 60) / 14)} />
          </g>
          <g transform="translate(980,330)">
            <text x={0} y={0} textAnchor="middle" fill={P.wave} fontSize={24} fontFamily={FONT_HEAD} opacity={0.9}>
              意识上线
            </text>
          </g>
        </g>

        {/* ---- 右：13~14 人里 1 个 ---- */}
        <g opacity={clamp01(panelK)}>
          <rect x={1140} y={160} width={670} height={600} rx={40} fill={P.card} stroke={P.stem} strokeWidth={4} />
          <text x={1475} y={222} textAnchor="middle" fill={P.cream} fontSize={32} fontFamily={FONT_HEAD} fontWeight={800}>
            一生至少撞上一次 ≈ 7.6%
          </text>
          {[0, 1].map((r) =>
            [0, 1, 2, 3, 4, 5, 6].map((c) => {
              const i = r * 7 + c;
              const hi = i === 9;
              const col = hi && hiK > 0.5 ? P.coral : P.metal;
              return (
                <g key={i}>
                  <StickFig x={1250 + c * 74} y={r === 0 ? 380 : 560} s={0.92}
                    col={col} k={figK(i)} exclaim={hi && hiK > 0.5} />
                  {hi && hiK > 0 && (
                    <circle cx={1250 + c * 74} cy={r === 0 ? 408 : 588} r={44 * hiK} fill="none"
                      stroke={P.coral} strokeWidth={5} strokeDasharray="10 10" />
                  )}
                </g>
              );
            })
          )}
          <text x={1475} y={660} textAnchor="middle" fill={P.stem} fontSize={29} fontFamily={FONT_HEAD} fontWeight={800}
            opacity={hiK}>
            十三四个人里有一个
          </text>
          <g opacity={tagsK}>
            {['熬夜', '压力大', '作息乱'].map((t, i) => (
              <g key={i} transform={`translate(${1255 + i * 150},712)`}>
                <rect x={-58} y={-26} width={116 + (i === 0 ? 0 : 16)} height={48} rx={24}
                  fill={P.cardLite} stroke={P.coral} strokeWidth={3} />
                <text x={i === 0 ? 0 : 8} y={8} textAnchor="middle" fill={P.cream} fontSize={25} fontFamily={FONT_HEAD}>
                  {t}
                </text>
              </g>
            ))}
            <text x={1475} y={748} textAnchor="middle" fill={P.muted} fontSize={23} fontFamily={FONT_HEAD}>
              REM 期不稳 → 更容易撞上
            </text>
          </g>
        </g>
      </svg>
      <Chip text="清醒着被锁死" x={140} y={810} at={120} accent={P.coral} />
      <Stamp text="时序错乱而已" x={1180} y={92} at={b1 + 6} color={P.stem} size={48} rot={-4} />
    </FadeIn>
  );
};

/* ============ S3 恐惧和石板都有出处：杏仁核点火 → 配影子；膈肌苦撑 → 坐上去的东西 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1); // ≈330f “而呼吸肌被抑制…”

  const brainK = spring({frame: f - 8, fps, config: {damping: 13, stiffness: 120}});
  const amygPulse = 0.65 + Math.abs(Math.sin(f * 0.22)) * 0.35;
  const ghostK = clamp01((f - 190) / 34);
  const cardK = spring({frame: f - b1 - 4, fps, config: {damping: 12, stiffness: 130}});
  const diaY = Math.sin(f * 0.24) * 9; // 膈肌独自起伏
  const pressK = clamp01((f - b1 - 60) / 24);

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={3} name={sec.name} accent={P.coral} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* ---- 左：脑内 · 杏仁核点火 ---- */}
        <g opacity={clamp01(brainK)}>
          <BrainBlob x={430} y={430} r={215} k={1} />
          {/* 前额叶：离线 */}
          <g opacity={0.55}>
            <path d="M 268 268 Q 226 300 232 352 Q 268 336 306 344 Q 318 296 268 268 Z"
              fill={P.metalD} stroke={P.metal} strokeWidth={4} strokeDasharray="8 8" />
          </g>
          <text x={206} y={240} fill={C.text} fontSize={24} fontFamily={FONT_HEAD} fontWeight={700}>
            理性脑·没上线
          </text>
          {/* 杏仁核 */}
          <g transform="translate(478,474)">
            <ellipse rx={52} ry={34} fill={P.coral} opacity={amygPulse} transform="rotate(-18)" />
            {[0, 1, 2, 3, 4].map((i) => {
              const a = -90 + i * 36 - 72;
              const r1 = 62, r2 = 62 + 26 * amygPulse;
              return (
                <line key={i} x1={Math.cos(a * Math.PI / 180) * r1} y1={Math.sin(a * Math.PI / 180) * r1}
                  x2={Math.cos(a * Math.PI / 180) * r2} y2={Math.sin(a * Math.PI / 180) * r2}
                  stroke={P.stem} strokeWidth={5} strokeLinecap="round" opacity={amygPulse} />
              );
            })}
            <text x={0} y={150} textAnchor="middle" fill={P.coral} fontSize={26} fontFamily={FONT_HEAD} fontWeight={800}>
              杏仁核·烟雾报警器
            </text>
          </g>
          {/* 箭头：恐惧→配图 */}
          <path d="M 560 470 Q 660 430 720 380" stroke={P.coral} strokeWidth={7} fill="none"
            strokeDasharray="14 12" strokeLinecap="round" opacity={ghostK} />
          <path d="M 720 380 l -26 -6 m 26 6 l -8 24" stroke={P.coral} strokeWidth={7} fill="none"
            strokeLinecap="round" opacity={ghostK} />
        </g>
        {/* 大脑自动配的影子 */}
        <g transform="translate(0,760)">
          <ShadowFigure x={836} y={0} h={340} k={ghostK} lean={-4} />
        </g>
        {ghostK > 0.4 && (
          <text x={836} y={330} textAnchor="middle" fill={P.coral} fontSize={26} fontFamily={FONT_HEAD}
            fontWeight={700} opacity={ghostK}>
            自动配图：感知存在
          </text>
        )}

        {/* ---- 右：胸腔剖面 · 只剩膈肌 ---- */}
        <g transform={`translate(${(1 - clamp01(cardK)) * 60},0)`} opacity={clamp01(cardK)}>
          <rect x={1080} y={170} width={730} height={620} rx={40} fill={P.card} stroke={P.waveD} strokeWidth={4} />
          <text x={1445} y={230} textAnchor="middle" fill={P.cream} fontSize={32} fontFamily={FONT_HEAD} fontWeight={800}>
            胸口压感也有物理来源
          </text>
          {/* 胸腔 */}
          <path d="M 1180 640 Q 1160 420 1280 360 L 1610 360 Q 1730 420 1710 640 Z" fill={P.cardLite}
            stroke={P.skinD} strokeWidth={6} />
          {/* 被抑制的肋间肌 */}
          {[0, 1, 2].map((i) => (
            <g key={i} opacity={0.5}>
              <line x1={1200 + i * 0} y1={410 + i * 60} x2={1260} y2={424 + i * 60} stroke={P.metal} strokeWidth={8} strokeLinecap="round" />
              <line x1={1690} y1={410 + i * 60} x2={1630} y2={424 + i * 60} stroke={P.metal} strokeWidth={8} strokeLinecap="round" />
            </g>
          ))}
          {/* 肺 */}
          <path d="M 1330 400 Q 1290 440 1296 520 Q 1300 560 1340 556 L 1420 556 Q 1428 520 1426 460 Q 1424 410 1390 400 Z"
            fill={P.waveD} opacity={0.6} />
          <path d="M 1560 400 Q 1600 440 1594 520 Q 1590 560 1550 556 L 1470 556 Q 1462 520 1464 460 Q 1466 410 1500 400 Z"
            fill={P.waveD} opacity={0.6} />
          {/* 膈肌：独自撑起 */}
          <path d={`M 1240 ${598 + diaY} Q 1445 ${540 + diaY * 1.6} 1650 ${598 + diaY}`}
            stroke={P.stem} strokeWidth={14} fill="none" strokeLinecap="round" />
          <text x={1445} y={676} textAnchor="middle" fill={P.stem} fontSize={26} fontFamily={FONT_HEAD} fontWeight={700}>
            只剩膈肌在撑·呼吸变浅
          </text>
          {/* 坐上去的小鬼 incubus */}
          <g transform={`translate(1445,${352 - (1 - pressK) * 60})`} opacity={pressK}>
            <ellipse rx={92} ry={58} fill={P.ghost} />
            <path d="M -56 -38 L -78 -78 L -34 -52 Z M 56 -38 L 78 -78 L 34 -52 Z" fill={P.ghost} />
            <ellipse cx={-24} cy={-16} rx={9} ry={6} fill={P.cream} opacity={0.9} />
            <ellipse cx={24} cy={-16} rx={9} ry={6} fill={P.cream} opacity={0.9} />
            <path d="M -18 12 Q 0 24 18 12" stroke={P.coral} strokeWidth={5} fill="none" strokeLinecap="round" />
            <path d="M -88 18 q -22 10 -18 30 M 88 18 q 22 10 18 30" stroke={P.ghost} strokeWidth={14} fill="none" strokeLinecap="round" />
            {/* 下压箭头 */}
            <g stroke={P.coral} strokeWidth={7} strokeLinecap="round" fill={P.coral}>
              <line x1={-128} y1={-30} x2={-128} y2={10} />
              <path d="M -140 -2 L -128 16 L -116 -2 Z" stroke="none" />
              <line x1={128} y1={-30} x2={128} y2={10} />
              <path d="M 116 -2 L 128 16 L 140 -2 Z" stroke="none" />
            </g>
          </g>
          <text x={1445} y={300} textAnchor="middle" fill={P.muted} fontSize={24} fontFamily={FONT_HEAD}
            opacity={pressK}>
            incubus 本义 = 坐上去的东西
          </text>
        </g>
      </svg>
      <Chip text="恐惧是系统自带的" x={140} y={140} at={120} accent={P.coral} />
      <Stamp text="不是想象·有出处" x={700} y={110} at={b1 + 60} color={P.stem} size={48} rot={-4} />
      <Chip text="同一个 bug·全球本地化命名" x={1120} y={830} at={b1 + 130} accent={P.wave} />
    </FadeIn>
  );
};

/* ============ S4 怎么解开这道锁：末端先解锁 → 预防清单 → 就医红线 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1); // ≈164f “试着动指尖转眼珠…”
  const b2 = beat(2); // ≈401f “平时睡够少仰卧…”

  const roomK = spring({frame: f - 6, fps, config: {damping: 13, stiffness: 120}});
  const calmK = clamp01((f - 40) / 40);
  const fear = 1 - calmK * 0.55; // 恐怖度 100%→45%
  const steps = [0, 1, 2].map((i) => clamp01(spring({frame: f - b1 - 12 - i * 26, fps, config: {damping: 11, stiffness: 160}})));
  const unlockAll = clamp01((f - b1 - 130) / 30);
  const listK = spring({frame: f - b2 - 4, fps, config: {damping: 12, stiffness: 130}});
  const warnK = spring({frame: f - b2 - 60, fps, config: {damping: 12, stiffness: 130}});

  // 解锁点（相对床的绝对坐标）
  const pts = [
    {x: 302, y: 606, label: '① 转眼珠'},
    {x: 620, y: 700, label: '② 动指尖'},
    {x: 872, y: 726, label: '③ 动脚趾'},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={4} name={sec.name} accent={P.mint} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 恐怖度表 */}
        <g opacity={clamp01(roomK)}>
          <rect x={130} y={320} width={86} height={380} rx={30} fill={P.card} stroke={P.coral} strokeWidth={4} />
          <rect x={146} y={336 + (348 * (1 - fear))} width={54} height={348 * fear} rx={20}
            fill={fear > 0.7 ? P.coral : P.stem} opacity={0.92} />
          <text x={173} y={292} textAnchor="middle" fill={P.cream} fontSize={26} fontFamily={FONT_HEAD} fontWeight={800}>
            恐怖度
          </text>
          <text x={173} y={760} textAnchor="middle" fill={P.mint} fontSize={25} fontFamily={FONT_HEAD} fontWeight={700}
            opacity={calmK}>
            知道真相
          </text>
          <text x={173} y={796} textAnchor="middle" fill={P.mint} fontSize={28} fontFamily={FONT_HEAD} fontWeight={800}
            opacity={calmK}>
            −50%
          </text>
        </g>

        {/* 床 + 人 + 末端解锁点 */}
        <g transform="translate(300,742)" opacity={clamp01(roomK)}>
          <Bed x={0} y={0} s={1} />
          <g transform="translate(0,-66)">
            <LyingPerson x={0} y={0} s={1} awake />
            {/* 露出的手和脚 */}
            <circle cx={322} cy={62} r={22} fill={P.skin} />
            <ellipse cx={575} cy={88} rx={26} ry={20} fill={P.skin} />
          </g>
        </g>
        {/* 手指出被窝 */}
        <g transform="translate(622,738)" opacity={clamp01(roomK)}>
          <rect x={-40} y={-16} width={80} height={30} rx={15} fill={P.skin} transform="rotate(8)" />
          <rect x={22} y={-20} width={34} height={18} rx={9} fill={P.skinD} transform="rotate(8)" />
        </g>

        {pts.map((p, i) => (
          <g key={i} opacity={steps[i]}>
            <circle cx={p.x} cy={p.y} r={34 + Math.sin(f * 0.2 + i) * 4} fill="none"
              stroke={P.mint} strokeWidth={5} strokeDasharray="8 9" />
            <text x={p.x} y={p.y - 48} textAnchor="middle" fill={P.mint} fontSize={27}
              fontFamily={FONT_HEAD} fontWeight={800}>{p.label}</text>
          </g>
        ))}
        {/* 大开锁 */}
        <g transform="translate(960,420)">
          <Padlock x={0} y={0} s={1.6} k={clamp01((f - b1 - 100) / 16)} open={unlockAll} col={P.mint} />
          {unlockAll > 0.5 && (
            <text x={0} y={76} textAnchor="middle" fill={P.mint} fontSize={28} fontFamily={FONT_HEAD} fontWeight={800}>
              几十秒自己解开
            </text>
          )}
        </g>
        {steps[0] > 0.5 && <Ripple x={pts[0].x} y={pts[0].y} startFrame={b1 + 12} color={P.mint} />}
        {steps[1] > 0.5 && <Ripple x={pts[1].x} y={pts[1].y} startFrame={b1 + 38} color={P.mint} />}
        {steps[2] > 0.5 && <Ripple x={pts[2].x} y={pts[2].y} startFrame={b1 + 64} color={P.mint} />}

        {/* ---- 右：预防清单 ---- */}
        <g transform={`translate(${(1 - clamp01(listK)) * 60},0)`} opacity={clamp01(listK)}>
          <rect x={1130} y={160} width={680} height={330} rx={36} fill={P.card} stroke={P.mint} strokeWidth={4} />
          <text x={1470} y={218} textAnchor="middle" fill={P.cream} fontSize={32} fontFamily={FONT_HEAD} fontWeight={800}>
            预防三件套
          </text>
          {['睡够不熬夜', '少仰卧·换侧卧', '压力大更要稳作息'].map((t, i) => (
            <g key={i} opacity={clamp01((f - b2 - 20 - i * 18) / 16)}>
              <CheckDot x={1200} y={282 + i * 64} r={22} k={1} />
              <text x={1246} y={292 + i * 64} fill={P.cream} fontSize={30} fontFamily={FONT_HEAD}>{t}</text>
            </g>
          ))}
        </g>
        {/* 就医红线 */}
        <g transform={`translate(${(1 - clamp01(warnK)) * 60},0)`} opacity={clamp01(warnK)}>
          <rect x={1130} y={530} width={680} height={240} rx={36} fill={P.card} stroke={P.coral} strokeWidth={5} />
          <g transform="translate(1216,650)">
            <rect x={-30} y={-30} width={60} height={60} rx={14} fill={P.coral} />
            <rect x={-8} y={-20} width={16} height={40} rx={4} fill={P.cream} />
            <rect x={-20} y={-8} width={40} height={16} rx={4} fill={P.cream} />
          </g>
          <text x={1280} y={616} fill={P.cream} fontSize={30} fontFamily={FONT_HEAD} fontWeight={800}>
            每周都犯 + 白天嗜睡猝倒
          </text>
          <text x={1280} y={664} fill={P.coral} fontSize={28} fontFamily={FONT_HEAD} fontWeight={700}>
            → 睡眠门诊查发作性睡病
          </text>
          <text x={1280} y={716} fill={P.muted} fontSize={24} fontFamily={FONT_HEAD}>
            偶发=正常现象，频繁才算症状
          </text>
        </g>
      </svg>
      <Chip text="先别慌·这是睡眠瘫痪" x={250} y={140} at={30} accent={P.stem} />
      <Stamp text="从末端开始解锁" x={700} y={180} at={b1 + 6} color={P.mint} size={50} rot={-4} />
      <Chip text="REM 不稳就少喂它素材" x={1140} y={810} at={b2 + 20} accent={P.wave} />
    </FadeIn>
  );
};

/* ============ 结尾卡 ============ */
const EndCard = () => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 14}});
  const pop = spring({frame: f - 5, fps, config: {damping: 10, stiffness: 140}});
  return (
    <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
      <Backdrop top={P.bgDeep} bottom={P.bgDeep} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={20} from={0} dur={24} />
        <Moon x={1620} y={180} scale={1.1} />
        <g transform="translate(960,880) scale(0.6)">
          <g transform={`scale(${clamp01(pop)})`}>
            <Bed x={-340} y={0} s={1} />
            <g transform="translate(-340,-66)">
              <LyingPerson x={0} y={0} s={1} awake={false} />
            </g>
            {/* 打开的安全锁当装饰 */}
            <Padlock x={80} y={-110} s={1.3} k={1} open={1} col={P.mint} />
          </g>
        </g>
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -260}}>
        <div style={{fontSize: 62, fontWeight: 800, color: C.text}}>压你的不是鬼，是防梦游的安全锁</div>
        <div style={{fontSize: 38, color: C.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E224 = ({data = tl}: {data?: any}) => {
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
