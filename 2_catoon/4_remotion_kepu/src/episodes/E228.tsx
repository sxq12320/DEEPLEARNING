import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01, rand} from '../lib/style';
import {Backdrop, Stamp, SceneTag, Chip, Subtitles, ProgressBar, FadeIn, Moon, Stars, Sun} from '../lib/atoms';
import tl from '../timeline_228.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：深紫夜 × 晨曦橙 × 皮质醇金 ============ */
const P = {
  bg: '#231c46',
  bgDeep: '#181236',
  bgLite: '#2c2456',
  card: '#3b3370',
  panel: '#1c163e',
  paper: '#f6eeda',
  dawn: '#ff9e64',     // 晨曦橙 · 主强调
  dawnD: '#dd7a42',
  gold: '#ffd166',     // 皮质醇金
  goldD: '#e2ac44',
  sky: '#6fd8ff',      // 清醒青蓝
  skyD: '#3fa9d9',
  vio: '#a08cf5',      // 腺苷紫
  vioD: '#7c68d8',
  teal: '#5ce8c8',
  coral: '#ff6b6b',
  coralD: '#d14f4f',
  cream: '#f8f0dd',
  ink: '#1d1838',
  muted: '#a5a0d4',
};

/* ---------- Zzz ---------- */
const Zzz = ({x, y, k = 1, col = P.gold}: {x: number; y: number; k?: number; col?: string}) => {
  const f = useCurrentFrame();
  return (
    <g transform={`translate(${x},${y})`} opacity={k}>
      {[0, 1, 2].map((i) => {
        const t = (f * 0.9 + i * 20) % 60;
        const p = t / 60;
        return (
          <text key={i} x={i * 22 + Math.sin(p * 3) * 9} y={-p * 66}
            fontSize={22 + i * 12} fill={col} fontFamily={FONT_HEAD}
            fontWeight={700} opacity={0.95 * (1 - p)}>z</text>
        );
      })}
    </g>
  );
};

/* ---------- 床头数字闹钟（本期图标） ---------- */
const Alarm = ({x, y, s = 1, on = 1, t = '6:55'}: {
  x: number; y: number; s?: number; on?: number; t?: string;
}) => {
  const f = useCurrentFrame();
  const wob = on > 0 ? Math.sin(f * 0.85) * 5 * on : 0;
  const arcO = on > 0 ? 0.3 + 0.6 * Math.abs(Math.sin(f * 0.55)) : 0;
  return (
    <g transform={`translate(${x},${y}) rotate(${wob}) scale(${s})`}>
      {on > 0 && [-1, 1].map((sd) => (
        <g key={sd} opacity={arcO}>
          <path d={`M ${sd * 78} -30 Q ${sd * 108} -44 ${sd * 114} -76`} stroke={P.dawn} strokeWidth={5} fill="none" strokeLinecap="round" />
          <path d={`M ${sd * 88} -8 Q ${sd * 122} -18 ${sd * 128} -48`} stroke={P.dawn} strokeWidth={4} fill="none" strokeLinecap="round" opacity={0.6} />
        </g>
      ))}
      {/* 双铃 + 锤 */}
      <path d="M -72 -44 A 26 26 0 0 1 -20 -44 Z" fill={P.dawn} />
      <path d="M 20 -44 A 26 26 0 0 1 72 -44 Z" fill={P.dawn} />
      <line x1={0} y1={-42} x2={0} y2={-62} stroke={P.dawnD} strokeWidth={6} strokeLinecap="round" />
      <circle cy={-66} r={8} fill={P.dawn} />
      {/* 机身 */}
      <rect x={-70} y={-46} width={140} height={108} rx={22} fill={P.paper} stroke={P.coral} strokeWidth={6} />
      <rect x={-56} y={-30} width={112} height={64} rx={12} fill={P.ink} />
      <text x={0} y={15} textAnchor="middle" fontSize={40} fill={P.dawn}
        fontFamily={FONT_HEAD} fontWeight={900}>{t}</text>
      <line x1={-40} y1={62} x2={-50} y2={78} stroke={P.coral} strokeWidth={7} strokeLinecap="round" />
      <line x1={40} y1={62} x2={50} y2={78} stroke={P.coral} strokeWidth={7} strokeLinecap="round" />
    </g>
  );
};

/* ---------- 侧视床上小人 ---------- */
const Sleeper = ({x, y, s = 1, face = 'sleep'}: {
  x: number; y: number; s?: number; face?: 'sleep' | 'wake' | 'groggy';
}) => {
  const f = useCurrentFrame();
  const dy = face === 'groggy' ? Math.abs(Math.sin(f / 8)) * -6 : 0;
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      <rect x={-200} y={-14} width={400} height={26} rx={12} fill={P.card} />
      <rect x={-196} y={12} width={18} height={44} rx={8} fill={P.card} />
      <rect x={178} y={12} width={18} height={44} rx={8} fill={P.card} />
      <rect x={-190} y={-104} width={22} height={90} rx={9} fill={P.card} />
      <rect x={-162} y={-58} width={106} height={40} rx={18} fill={P.cream} />
      <path d="M -62 -14 Q 40 -86 188 -22 L 194 -14 Z" fill={P.vio} stroke={P.vioD} strokeWidth={3} />
      <g transform={`translate(-102,${-64 + dy})`}>
        <circle r={28} fill={C.skin} />
        <path d="M -27 -10 Q 0 -38 27 -10 L 27 -2 Q 0 -22 -27 -2 Z" fill="#42304e" />
        {face === 'sleep' && (
          <g stroke={P.ink} strokeWidth={3.2} strokeLinecap="round" fill="none">
            <path d="M -14 -2 q 6 7 12 0" />
            <path d="M 6 -2 q 6 7 12 0" />
            <path d="M -6 14 q 6 5 12 0" />
          </g>
        )}
        {face === 'wake' && (
          <g>
            <circle cx={-9} cy={-2} r={4.4} fill={P.ink} />
            <circle cx={11} cy={-2} r={4.4} fill={P.ink} />
            <path d="M -8 14 q 9 9 18 0" stroke={P.ink} strokeWidth={3.2} fill="none" strokeLinecap="round" />
          </g>
        )}
        {face === 'groggy' && (
          <g>
            <path d="M -15 -6 a 6 6 0 1 1 6 6 a 3.4 3.4 0 1 0 -3.4 -3.4" stroke={P.ink} strokeWidth={2.6} fill="none" />
            <path d="M 5 -6 a 6 6 0 1 1 6 6 a 3.4 3.4 0 1 0 -3.4 -3.4" stroke={P.ink} strokeWidth={2.6} fill="none" />
            <ellipse cx={2} cy={16} rx={6.5} ry={8.5} fill={P.ink} />
          </g>
        )}
      </g>
    </g>
  );
};

/* ---------- 眼睛 ---------- */
const Eye = ({x, y, s = 1, open = 1}: {x: number; y: number; s?: number; open?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <ellipse rx={42} ry={24 * open + 3} fill={P.cream} />
    <ellipse rx={42} ry={24 * open + 3} fill="none" stroke={P.ink} strokeWidth={3.5} />
    {open > 0.3 && <circle r={13} fill={P.dawn} stroke={P.dawnD} strokeWidth={3.5} />}
    {open > 0.3 && <circle cx={4} cy={-4} r={4} fill="#fff" opacity={0.9} />}
  </g>
);

/* ---------- 激素试管 ---------- */
const Tube = ({x, y, s = 1, k = 1, col = P.gold}: {x: number; y: number; s?: number; k?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-14} y={-58} width={28} height={10} rx={5} fill={P.cream} />
    <path d="M -14 -48 L -14 34 A 14 14 0 0 0 14 34 L 14 -48 Z" fill="none" stroke={P.cream} strokeWidth={4.5} />
    <path d={`M -11 ${34 - 60 * k - 13} L -11 30 A 11 11 0 0 0 11 30 L 11 ${34 - 60 * k - 13} Z`} fill={col} opacity={0.9} />
  </g>
);

/* ---------- 小便签 ---------- */
const Note = ({x, y, rot = -4, s = 1, text, sub}: {
  x: number; y: number; rot?: number; s?: number; text: string; sub?: string;
}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`}>
    <rect x={-130} y={-70} width={260} height={140} rx={10} fill={P.paper} />
    <rect x={-130} y={-70} width={260} height={26} rx={10} fill={P.dawn} opacity={0.9} />
    <text x={0} y={sub ? 8 : 22} textAnchor="middle" fontSize={30} fill={P.ink}
      fontFamily={FONT_HEAD} fontWeight={800}>{text}</text>
    {sub && <text x={0} y={48} textAnchor="middle" fontSize={22} fill={P.ink}
      fontFamily={FONT_HEAD} opacity={0.75}>{sub}</text>}
  </g>
);

/* ---------- 小激素曲线（S0 实验卡用） ---------- */
const HormChart = ({x, y, w, h, riseU, riseK, col}: {
  x: number; y: number; w: number; h: number; riseU: number; riseK: number; col: string;
}) => {
  const N = 40;
  const pts: string[] = [];
  for (let i = 0; i <= N; i++) {
    const u = i / N;
    const wig = (rand(i, 5) - 0.5) * 0.05;
    const lvl = u < riseU ? 0.16 + wig
      : 0.16 + 0.78 * clamp01((u - riseU) / 0.22) + wig * (1 - clamp01((u - riseU) / 0.22));
    pts.push(`${(x + u * w).toFixed(1)},${(y + h - lvl * h).toFixed(1)}`);
  }
  const n = Math.max(2, Math.floor(N * clamp01(riseK)));
  return <polyline points={pts.slice(0, n + 1).join(' ')} fill="none" stroke={col}
    strokeWidth={5} strokeLinecap="round" strokeLinejoin="round" />;
};

/* ---------- S1 皮质醇曲线 ---------- */
const cort = (h: number) => 0.16 + 0.84 * Math.exp(-((h - 8.5) ** 2) / (2 * 3.1 * 3.1));
const cortPre = (h: number) => 0.16 + 0.84 * Math.exp(-((h - 7.0) ** 2) / (2 * 2.55 * 2.55));

const CortCurve = ({x0, yB, yT, w, k, fn, col, dash, lw = 6}: {
  x0: number; yB: number; yT: number; w: number; k: number;
  fn: (h: number) => number; col: string; dash?: string; lw?: number;
}) => {
  const pts: string[] = [];
  const hMax = 24 * clamp01(k);
  for (let h = 0; h <= hMax; h += 0.25) {
    pts.push(`${(x0 + (h / 24) * w).toFixed(1)},${(yB - fn(h) * (yB - yT)).toFixed(1)}`);
  }
  if (pts.length < 2) return null;
  return <polyline points={pts.join(' ')} fill="none" stroke={col} strokeWidth={lw}
    strokeLinecap="round" strokeLinejoin="round" strokeDasharray={dash} />;
};

/* ---------- 齿轮 ---------- */
const Gear = ({x, y, r = 30, col = P.gold, speed = 1}: {x: number; y: number; r?: number; col?: string; speed?: number}) => {
  const f = useCurrentFrame();
  return (
    <g transform={`translate(${x},${y}) rotate(${f * speed})`}>
      {Array.from({length: 8}).map((_, i) => (
        <rect key={i} x={-r * 0.16} y={-r * 1.32} width={r * 0.32} height={r * 0.42} rx={r * 0.1}
          fill={col} transform={`rotate(${i * 45})`} />
      ))}
      <circle r={r} fill={col} />
      <circle r={r * 0.42} fill={P.panel} />
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
  const pB = Math.max(0.001, spring({frame: f - 16, fps, config: {damping: 10, stiffness: 150}}));
  return (
    <AbsoluteFill>
      <Backdrop top={P.bg} bottom={P.bg} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={20} from={0} dur={24} />
        <Moon x={250} y={160} scale={0.75} />
        {/* 中央大闹钟 */}
        <g transform={`translate(960,400) scale(${pA}) translate(-960,-400)`} opacity={clamp01(pA)}>
          <Alarm x={960} y={360} s={1.7} on={0} t="6:55" />
        </g>
        {/* 两个时间签 */}
        <g transform={`translate(700,270) scale(${pB})`} opacity={clamp01(pB)}>
          <rect x={-110} y={-34} width={220} height={68} rx={16} fill={P.panel} stroke={P.gold} strokeWidth={4} />
          <text x={0} y={14} textAnchor="middle" fontSize={34} fill={P.gold}
            fontFamily={FONT_HEAD} fontWeight={800}>6:55 自醒</text>
        </g>
        <g transform={`translate(1240,270) scale(${pB})`} opacity={clamp01(pB)}>
          <rect x={-110} y={-34} width={220} height={68} rx={16} fill={P.panel} stroke={P.muted} strokeWidth={4} />
          <text x={0} y={14} textAnchor="middle" fontSize={34} fill={P.muted}
            fontFamily={FONT_HEAD} fontWeight={800}>7:00 闹钟</text>
        </g>
        <Zzz x={540} y={360} k={0.9} />
        <Eye x={1420} y={330} s={0.9} />
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 430}}>
          <div style={{color: P.dawn, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 24}}>
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

/* ============ S0 睡前存的起床指令：1999 对照实验 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const popL = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const popR = Math.max(0.001, spring({frame: f - 40, fps, config: {damping: 11, stiffness: 120}}));
  const riseK = clamp01((f - 195) / 55);          // A组激素开爬 ≈ 6.5s
  const flatK = clamp01((f - 300) / 40);          // B组安静 ≈ 10s
  const noteK = clamp01((f - b1) / 26);           // cue1：指令便签飞入
  const progK = clamp01((f - b1 - 40) / 22);      // → 激素程序卡
  const nx = interpolate(noteK, [0, 1], [1620, 575]);
  const ny = interpolate(noteK, [0, 1], [130, 348]);

  const panel = (x0: number, title: string, tcol: string, k: number, kids: React.ReactNode) => (
    <g transform={`translate(${x0},150) scale(${k}) translate(${-x0},-150)`} opacity={clamp01(k * 2)}>
      <rect x={x0} y={150} width={740} height={600} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
      <text x={x0 + 36} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>睡前被告知</text>
      <text x={x0 + 36} y={256} fill={tcol} fontSize={38} fontFamily={FONT_HEAD} fontWeight={900}>{title}</text>
      {kids}
    </g>
  );

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={0} name={sec.name} accent={P.dawn} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={12} from={0} dur={26} />
        {/* A组 */}
        {panel(140, '「明早 6:00 叫你」', P.dawn, popL, (
          <g>
            <Sleeper x={505} y={415} s={0.82} face="sleep" />
            {noteK > 0.4 && <Eye x={392} y={330} s={0.55} open={0.3} />}
            <rect x={190} y={490} width={640} height={212} rx={14} fill={P.bgDeep} />
            <text x={214} y={530} fill={P.muted} fontSize={23} fontFamily={FONT_HEAD}>皮质醇水平</text>
            <line x1={230} y1={660} x2={800} y2={660} stroke={P.muted} strokeWidth={3} />
            <line x1={230} y1={660} x2={230} y2={556} stroke={P.muted} strokeWidth={3} />
            <HormChart x={250} y={560} w={520} h={88} riseU={0.62} riseK={Math.max(flatK * 0.4, riseK)} col={P.gold} />
            {riseK > 0.05 && (
              <g>
                <line x1={250 + 0.62 * 520} y1={556} x2={250 + 0.62 * 520} y2={660}
                  stroke={P.gold} strokeWidth={3.5} strokeDasharray="7 7" />
                <rect x={250 + 0.62 * 520 - 66} y={512} width={132} height={40} rx={10}
                  fill={P.gold} opacity={clamp01(riseK * 2)} />
                <text x={250 + 0.62 * 520} y={540} textAnchor="middle" fontSize={24}
                  fill={P.ink} fontFamily={FONT_HEAD} fontWeight={800}
                  opacity={clamp01(riseK * 2)}>4:30 开爬</text>
              </g>
            )}
            <text x={250} y={690} fill={P.muted} fontSize={21} fontFamily={FONT_HEAD}>入睡</text>
            <text x={770} y={690} textAnchor="end" fill={P.muted} fontSize={21} fontFamily={FONT_HEAD}>6:00</text>
            <text x={510} y={736} textAnchor="middle" fill={P.gold} fontSize={26}
              fontFamily={FONT_HEAD} fontWeight={700}>激素自己先把身体叫醒</text>
          </g>
        ))}
        {/* B组 */}
        {panel(1040, '「明早 9:00 叫你」', P.muted, popR, (
          <g>
            <Sleeper x={1405} y={415} s={0.82} face="sleep" />
            <Zzz x={1510} y={320} k={0.85} col={P.muted} />
            <rect x={1090} y={490} width={640} height={212} rx={14} fill={P.bgDeep} />
            <text x={1114} y={530} fill={P.muted} fontSize={23} fontFamily={FONT_HEAD}>皮质醇水平</text>
            <line x1={1130} y1={660} x2={1700} y2={660} stroke={P.muted} strokeWidth={3} />
            <line x1={1130} y1={660} x2={1130} y2={556} stroke={P.muted} strokeWidth={3} />
            <HormChart x={1150} y={560} w={520} h={88} riseU={2} riseK={flatK} col={P.muted} />
            <text x={1150} y={690} fill={P.muted} fontSize={21} fontFamily={FONT_HEAD}>入睡</text>
            <text x={1670} y={690} textAnchor="end" fill={P.muted} fontSize={21} fontFamily={FONT_HEAD}>9:00</text>
            <text x={1410} y={736} textAnchor="middle" fill={P.muted} fontSize={26}
              fontFamily={FONT_HEAD}>同一时段 · 安静如常</text>
          </g>
        ))}
        {/* VS 徽标 */}
        <g transform="translate(960,440)">
          <circle r={36} fill={P.coral} />
          <text y={12} textAnchor="middle" fontSize={32} fill={P.cream}
            fontFamily={FONT_HEAD} fontWeight={900}>VS</text>
        </g>
        {/* 便签飞入 A组枕头 → 程序卡（cue1） */}
        {noteK > 0 && progK < 0.6 && <Note x={nx} y={ny} s={0.9} text="明早 6:00 要起" />}
        {progK > 0 && (
          <g opacity={progK} transform={`translate(0,${(1 - progK) * 60})`}>
            <Note x={640} y={780} s={0.45} rot={-6} text="6:00 要起" />
            <path d="M 712 780 L 850 780" stroke={P.gold} strokeWidth={6} strokeLinecap="round" />
            <polygon points="862,780 840,770 840,790" fill={P.gold} />
            <rect x={880} y={748} width={640} height={66} rx={16} fill={P.panel} stroke={P.gold} strokeWidth={4} />
            <Gear x={922} y={781} r={20} speed={2} />
            <text x={958} y={790} fill={P.cream} fontSize={28} fontFamily={FONT_HEAD} fontWeight={700}>
              翻译成：凌晨提前启动的激素程序</text>
          </g>
        )}
      </svg>
      <Chip text="1999 · 实验室对照" x={1380} y={70} at={beat(0) + 30} accent={P.dawn} />
      <Chip text="指令没消失 · 被翻译了" x={560} y={80} at={b1 + 55} accent={P.gold} />
      <Stamp text="身体先给自己打了闹钟" x={640} y={858} at={b1 + 75} color={P.dawnD} size={40} rot={-3} />
    </FadeIn>
  );
};

/* ============ S1 皮质醇 = 天然兴奋剂：24h 曲线 + 睡前订单 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const drawK = clamp01((f - 20) / 120);
  const orderK = clamp01((f - b1 - 15) / 55);     // 新曲线 ≈ cue1
  const noteP = Math.max(0.001, spring({frame: f - b1, fps, config: {damping: 11, stiffness: 120}}));
  const rowK = clamp01((f - b1 - 80) / 24);

  const X0 = 250, X1 = 1210, YB = 660, YT = 250, W = X1 - X0;
  const hx = (h: number) => X0 + (h / 24) * W;

  const tag = (atF: number, kids: React.ReactNode) => (
    <g opacity={clamp01((f - atF) / 14)} transform={`translate(0,${(1 - clamp01((f - atF) / 14)) * 24})`}>
      {kids}
    </g>
  );

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={1} name={sec.name} accent={P.gold} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 主图面板 */}
        <rect x={140} y={150} width={1130} height={620} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <text x={176} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>皮质醇 · 一天的自然曲线</text>
        {/* 夜间区 */}
        <rect x={X0} y={YT - 10} width={hx(6.5) - X0} height={YB - YT + 20} fill={P.vio} opacity={0.1} />
        <Moon x={290} y={345} scale={0.3} />
        {/* 轴 */}
        <line x1={X0} y1={YB} x2={X1} y2={YB} stroke={P.muted} strokeWidth={3.5} />
        <line x1={X0} y1={YB} x2={X0} y2={YT - 14} stroke={P.muted} strokeWidth={3.5} />
        {[0, 6, 12, 18, 24].map((h) => (
          <g key={h}>
            <line x1={hx(h)} y1={YB} x2={hx(h)} y2={YB + 10} stroke={P.muted} strokeWidth={3} />
            <text x={hx(h)} y={YB + 40} textAnchor="middle" fill={P.muted} fontSize={23}
              fontFamily={FONT_HEAD}>{h === 24 ? '24点' : `${h}点`}</text>
          </g>
        ))}
        {/* 日常曲线 */}
        <CortCurve x0={X0} yB={YB} yT={YT} w={W} k={drawK} fn={cort} col={P.sky} />
        {/* 开爬标记 ≈4点 */}
        {tag(110, (
          <g>
            <line x1={hx(4)} y1={YB} x2={hx(4)} y2={YT + 60} stroke={P.sky} strokeWidth={3} strokeDasharray="7 8" opacity={0.8} />
            <text x={hx(4)} y={YT + 46} textAnchor="middle" fill={P.sky} fontSize={24}
              fontFamily={FONT_HEAD} fontWeight={700}>凌晨3–4点 · 开爬</text>
          </g>
        ))}
        {/* 峰顶旗 ≈8.5 */}
        {tag(200, (
          <g>
            <line x1={hx(8.5)} y1={YB - cort(8.5) * (YB - YT)} x2={hx(8.5)} y2={YT - 80}
              stroke={P.sky} strokeWidth={3.5} />
            <path d={`M ${hx(8.5)} ${YT - 80} L ${hx(8.5) + 56} ${YT - 68} L ${hx(8.5)} ${YT - 56} Z`} fill={P.sky} />
            <text x={hx(8.5) + 68} y={YT - 60} fill={P.sky} fontSize={25}
              fontFamily={FONT_HEAD} fontWeight={700}>8–9点 峰顶</text>
          </g>
        ))}
        {/* 干的活 ≈ cue0 尾 */}
        {tag(255, (
          <g>
            {['升血糖', '提心率', '拉警觉'].map((t, i) => (
              <g key={t} transform={`translate(${X0 + 60 + i * 180},${YB - 34})`}>
                <rect width={140} height={48} rx={24} fill={P.card} stroke={P.skyD} strokeWidth={3} />
                <text x={70} y={33} textAnchor="middle" fill={P.cream} fontSize={24}
                  fontFamily={FONT_HEAD} fontWeight={700}>{t}</text>
              </g>
            ))}
            <text x={X0 + 600} y={YB - 10} fill={P.muted} fontSize={24} fontFamily={FONT_HEAD}>
              = 身体的天然兴奋剂</text>
          </g>
        ))}
        {/* 订单曲线（cue1）：峰值提前到 7:00 */}
        {orderK > 0 && (
          <g opacity={orderK}>
            <CortCurve x0={X0} yB={YB} yT={YT} w={W} k={orderK} fn={cortPre} col={P.gold} dash="14 10" lw={6} />
            {orderK > 0.8 && (
              <g>
                <line x1={hx(7)} y1={YB} x2={hx(7)} y2={YT - 14} stroke={P.gold} strokeWidth={3.5} strokeDasharray="7 8" />
                <text x={hx(7) + 18} y={YT - 24} fill={P.gold} fontSize={25}
                  fontFamily={FONT_HEAD} fontWeight={800}>新峰值 ≈ 7:00</text>
              </g>
            )}
          </g>
        )}
        {/* 右列：订单便签 + 两个时刻 */}
        <g transform={`translate(1600,300) scale(${noteP})`} opacity={clamp01(noteP)}>
          <Note x={0} y={0} rot={-3} s={1.15} text="睡前订单" sub="明早 7:00 要到峰值" />
          <line x1={-60} y1={80} x2={-200} y2={170} stroke={P.gold} strokeWidth={4} strokeDasharray="8 8" />
        </g>
        {/* 右列早期占位：试管 + 名称（订单弹出后隐藏） */}
        <g opacity={clamp01(1 - noteP)}>
          <Tube x={1560} y={480} s={1.5} k={0.75} />
          <text x={1560} y={560} textAnchor="middle" fill={P.gold} fontSize={30}
            fontFamily={FONT_HEAD} fontWeight={800}>皮质醇</text>
          <text x={1560} y={600} textAnchor="middle" fill={P.muted} fontSize={23}
            fontFamily={FONT_HEAD}>肾上腺分泌</text>
          <text x={1560} y={700} textAnchor="middle" fill={P.muted} fontSize={24}
            fontFamily={FONT_HEAD}>睡前一下单 →</text>
        </g>
        <g opacity={rowK} transform={`translate(0,${(1 - rowK) * 50})`}>
          <rect x={1340} y={480} width={440} height={96} rx={16} fill={P.panel} stroke={P.gold} strokeWidth={4} />
          <Eye x={1410} y={528} s={0.8} />
          <text x={1468} y={540} fill={P.gold} fontSize={31} fontFamily={FONT_HEAD} fontWeight={800}>6:55 身体先睁眼</text>
          <rect x={1340} y={600} width={440} height={96} rx={16} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <Alarm x={1410} y={638} s={0.5} on={0} t="" />
          <text x={1468} y={652} fill={P.muted} fontSize={31} fontFamily={FONT_HEAD} fontWeight={700}>7:00 闹钟只确认</text>
          <text x={1340} y={750} fill={P.cream} fontSize={26} fontFamily={FONT_HEAD}>
            不是醒在闹钟之前 ——</text>
          <text x={1340} y={790} fill={P.gold} fontSize={28} fontFamily={FONT_HEAD} fontWeight={800}>
            是醒在自己订的时间上</text>
        </g>
      </svg>
      <Chip text="皮质醇 ≠ 只有压力" x={560} y={80} at={beat(0) + 30} accent={P.gold} />
      <Stamp text="睡前下单 · 身体照办" x={980} y={88} at={b1 + 95} color={P.goldD} size={40} rot={3} />
    </FadeIn>
  );
};

/* ============ S2 醒来是主动程序：脑干总闸 + 别掐程序 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const at = (s: number) => Math.round(s * fps);
  const bootK = clamp01((f - 30) / 140);
  const snzAt = at(10.8);
  const snzK = clamp01((f - snzAt) / 20);
  const press = snzK > 0 ? 1 - Math.abs(Math.sin((f - snzAt) / 6)) * 0.08 : 1;

  const pkt = (x0: number, y0: number, x1: number, y1: number, sp: number, ph: number) => {
    const t = ((f * sp + ph) % 60) / 60;
    return {x: x0 + (x1 - x0) * t, y: y0 + (y1 - y0) * t, o: t > 0.06 && t < 0.94 ? 1 : 0};
  };
  const pClock = pkt(690, 505, 636, 566, 1.4, 0);
  const pTube = pkt(690, 598, 636, 596, 1.4, 30);

  const NodeBox = ({x, y, w, h, label, sub, col, fill = 0}: {
    x: number; y: number; w: number; h: number; label: string; sub?: string; col: string; fill?: number;
  }) => (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={16} fill={P.bgDeep} stroke={col} strokeWidth={4} />
      <rect x={x + 3} y={y + 3} width={(w - 6) * clamp01(fill)} height={h - 6} rx={13} fill={col} opacity={0.28} />
      <text x={x + w / 2} y={y + h / 2 + (sub ? -4 : 10)} textAnchor="middle" fontSize={27}
        fill={P.cream} fontFamily={FONT_HEAD} fontWeight={800}>{label}</text>
      {sub && <text x={x + w / 2} y={y + h / 2 + 30} textAnchor="middle" fontSize={20}
        fill={P.muted} fontFamily={FONT_HEAD}>{sub}</text>}
    </g>
  );

  return (
    <FadeIn>
      <Backdrop top={P.bgDeep} bottom={P.bgDeep} />
      <SceneTag index={2} name={sec.name} accent={P.sky} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：意识上线通路 */}
        <rect x={140} y={150} width={820} height={540} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <text x={176} y={204} fill={P.muted} fontSize={24} fontFamily={FONT_HEAD}>意识上线通路 · 不是断电重启</text>
        {/* 节点链 */}
        <NodeBox x={230} y={236} w={400} h={76} label="皮层 · 意识" col={P.sky} fill={bootK} />
        <NodeBox x={290} y={392} w={280} h={62} label="丘脑 · 中转" col={P.teal} fill={clamp01(bootK * 2)} />
        <NodeBox x={230} y={540} w={400} h={82} label="脑干 · 上行网状激活" sub="= 总闸 ARAS" col={P.dawn} />
        {/* 上行箭头 */}
        {[[440, 540, 440, 460], [440, 392, 440, 318]].map(([a, b2, c2, d2], i) => {
          const dots = [0, 1].map((k) => {
            const t = ((f * 1.2 + k * 30 + i * 15) % 60) / 60;
            return {x: a + (c2 - a) * t, y: b2 + (d2 - b2) * t};
          });
          return (
            <g key={i}>
              <line x1={a} y1={b2} x2={c2} y2={d2} stroke={P.dawn} strokeWidth={5}
                strokeDasharray="10 10" opacity={0.7} />
              {dots.map((d, k) => <circle key={k} cx={d.x} cy={d.y} r={8} fill={P.dawn} />)}
            </g>
          );
        })}
        <polygon points="440,306 428,326 452,326" fill={P.dawn} />
        <polygon points="440,448 428,468 452,468" fill={P.dawn} />
        {/* 两路输入 */}
        <g>
          <rect x={690} y={470} width={230} height={62} rx={31} fill={P.bgDeep} stroke={P.sky} strokeWidth={4} />
          <text x={805} y={509} textAnchor="middle" fontSize={24} fill={P.sky}
            fontFamily={FONT_HEAD} fontWeight={700}>生物钟：该醒了</text>
          <line x1={690} y1={501} x2={636} y2={560} stroke={P.sky} strokeWidth={3.5} strokeDasharray="6 8" />
          <rect x={690} y={562} width={230} height={62} rx={31} fill={P.bgDeep} stroke={P.gold} strokeWidth={4} />
          <text x={805} y={601} textAnchor="middle" fontSize={24} fill={P.gold}
            fontFamily={FONT_HEAD} fontWeight={700}>激素：准备好了</text>
          <line x1={690} y1={593} x2={636} y2={600} stroke={P.gold} strokeWidth={3.5} strokeDasharray="6 8" />
          {pClock.o > 0 && <circle cx={pClock.x} cy={pClock.y} r={9} fill={P.sky} />}
          {pTube.o > 0 && <circle cx={pTube.x} cy={pTube.y} r={9} fill={P.gold} />}
          <text x={805} y={668} textAnchor="middle" fill={P.muted} fontSize={22} fontFamily={FONT_HEAD}>
            两路信号 · 睡前已订好出发时间</text>
        </g>
        {/* 右：唤醒程序窗口 */}
        <rect x={1000} y={150} width={780} height={380} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <circle cx={1036} cy={186} r={7} fill={P.coral} />
        <circle cx={1062} cy={186} r={7} fill={P.gold} />
        <circle cx={1088} cy={186} r={7} fill={P.teal} />
        <text x={1120} y={196} fill={P.cream} fontSize={27} fontFamily={FONT_HEAD} fontWeight={800}>
          唤醒程序 · 执行中</text>
        <rect x={1036} y={230} width={700} height={34} rx={17} fill={P.card} />
        <rect x={1036} y={230} width={Math.max(24, 700 * bootK)} height={34} rx={17} fill={P.dawn} />
        <text x={1736} y={256} textAnchor="end" fill={P.dawn} fontSize={25}
          fontFamily={FONT_HEAD} fontWeight={800}>{Math.round(bootK * 100)}%</text>
        {[{t: '心率升', at: 0.3}, {t: '血糖升', at: 0.55}, {t: '皮层上线', at: 0.88}].map((it, i) => {
          const on = bootK > it.at;
          return (
            <g key={it.t} transform={`translate(${1060 + i * 236},330)`}>
              <circle r={17} fill={on ? P.teal : P.card} />
              {on && <MarkV x={0} y={0} s={0.9} col={P.panel} />}
              <text x={0} y={64} textAnchor="middle" fontSize={25} fill={on ? P.cream : P.muted}
                fontFamily={FONT_HEAD} fontWeight={700}>{it.t}</text>
            </g>
          );
        })}
        <text x={1036} y={486} fill={P.muted} fontSize={24} fontFamily={FONT_HEAD}>
          两路信号汇合 → 意识被主动拉上线</text>
        {/* 底条：再睡五分钟的代价（≈10.8s） */}
        {snzK > 0 && (
          <g opacity={snzK} transform={`translate(0,${(1 - snzK) * 70})`}>
            <rect x={140} y={726} width={1640} height={150} rx={20} fill={P.panel} stroke={P.coral} strokeWidth={4} />
            <g transform={`translate(400,800) scale(1,${press})`}>
              <rect x={-140} y={-50} width={280} height={100} rx={50} fill={P.coral} />
              <rect x={-140} y={-50} width={280} height={100} rx={50} fill="none" stroke={P.coralD} strokeWidth={5} />
              <text y={12} textAnchor="middle" fontSize={34} fill={P.cream}
                fontFamily={FONT_HEAD} fontWeight={900}>再睡 5 分钟</text>
            </g>
            <path d="M 570 800 L 690 800" stroke={P.coral} strokeWidth={6} strokeLinecap="round" />
            <polygon points="702,800 680,790 680,810" fill={P.coral} />
            <rect x={730} y={772} width={320} height={30} rx={15} fill={P.card} />
            <rect x={730} y={772} width={110} height={30} rx={15} fill={P.muted} />
            <MarkX x={1085} y={787} s={1.3} />
            <text x={730} y={752} fill={P.coral} fontSize={25} fontFamily={FONT_HEAD} fontWeight={700}>
              掐断正在运行的唤醒程序</text>
            <text x={1180} y={788} fill={P.coral} fontSize={29} fontFamily={FONT_HEAD} fontWeight={800}>
              再响时：已滑进下一圈深睡边缘</text>
            <text x={1180} y={836} fill={P.muted} fontSize={24} fontFamily={FONT_HEAD}>
              难度系数完全不同</text>
          </g>
        )}
      </svg>
      <Chip text="ARAS = 意识总闸" x={1380} y={70} at={at(2)} accent={P.sky} />
      <Stamp text="醒来是程序 · 不是重启" x={620} y={92} at={at(4)} color={P.skyD} size={38} rot={-3} />
      <Stamp text="别掐正在运行的程序" x={1430} y={600} at={snzAt + 30} color={P.coralD} size={40} rot={-5} />
    </FadeIn>
  );
};

/* ============ S3 两班岗：S压力 + C节律 + 指令；缺觉失灵 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);
  const drainK = clamp01(f / 150);
  const waveK = clamp01((f - 110) / 70);
  const convK = [0, 1, 2].map((i) => clamp01((f - b1 - i * 8) / 22));
  const failK = clamp01((f - b2) / 24);

  const cards = [
    {x: 140, col: P.vio, title: 'S 过程 · 睡眠压力', at: 20},
    {x: 700, col: P.sky, title: 'C 过程 · 昼夜节律', at: 110},
    {x: 1260, col: P.dawn, title: '睡前指令 → 激素提前', at: 200},
  ];
  const pop = cards.map((c) => Math.max(0.001, spring({frame: f - c.at, fps, config: {damping: 11, stiffness: 120}})));

  // C 过程正弦小波
  const wpts: string[] = [];
  for (let i = 0; i <= 48; i++) {
    const u = i / 48;
    wpts.push(`${(736 + u * 420).toFixed(1)},${(310 + Math.sin(u * Math.PI * 2 - Math.PI / 2) * 42).toFixed(1)}`);
  }
  const wDrawn = Math.max(2, Math.floor(48 * waveK));

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={3} name={sec.name} accent={P.vio} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 三班岗卡片 */}
        {cards.map((c, i) => (
          <g key={i} transform={`translate(${c.x},160) scale(${pop[i]}) translate(${-c.x},-160)`}
            opacity={clamp01(pop[i] * 2)}>
            <rect x={c.x} y={160} width={520} height={280} rx={18} fill={P.panel} stroke={c.col} strokeWidth={4} />
            <text x={c.x + 34} y={214} fill={c.col} fontSize={30} fontFamily={FONT_HEAD} fontWeight={800}>{c.title}</text>
            {i === 0 && (
              <g>
                <text x={174} y={264} fill={P.muted} fontSize={23} fontFamily={FONT_HEAD}>腺苷库存</text>
                <rect x={174} y={278} width={300} height={30} rx={15} fill={P.card} />
                <rect x={174} y={278} width={Math.max(20, 300 * (1 - drainK * 0.85))} height={30} rx={15} fill={P.vio} />
                <rect x={476} y={284} width={12} height={18} rx={5} fill={P.card} />
                {[0, 1, 2, 3].map((k) => (
                  <circle key={k} cx={540 + k * 26} cy={292 - k * 14} r={7}
                    fill={P.vio} opacity={0.9 - k * 0.18} />
                ))}
                <Moon x={570} y={360} scale={0.24} />
                <Sun x={630} y={356} r={18} />
                <text x={174} y={376} fill={P.cream} fontSize={24} fontFamily={FONT_HEAD}>
                  醒得越久越多 · 一夜快见底</text>
                <text x={174} y={416} fill={P.muted} fontSize={22} fontFamily={FONT_HEAD}>
                  债没还清 → 什么信号都压得住</text>
              </g>
            )}
            {i === 1 && (
              <g>
                <rect x={736} y={250} width={420} height={120} rx={10} fill={P.bgDeep} />
                <polyline points={wpts.slice(0, wDrawn).join(' ')} fill="none" stroke={P.sky}
                  strokeWidth={5} strokeLinecap="round" />
                {waveK > 0.9 && (
                  <g>
                    <circle cx={946} cy={268} r={9} fill={P.gold} />
                    <text x={946} y={246} textAnchor="middle" fill={P.gold} fontSize={21}
                      fontFamily={FONT_HEAD} fontWeight={700}>清醒信号</text>
                    <circle cx={1150} cy={352} r={7} fill={P.vio} />
                  </g>
                )}
                <text x={736} y={402} fill={P.cream} fontSize={24} fontFamily={FONT_HEAD}>
                  视交叉上核这座钟 · 准点派信号</text>
              </g>
            )}
            {i === 2 && (
              <g>
                <Note x={1400} y={300} s={0.72} text="6:00 要起" />
                <path d="M 1500 300 L 1570 300" stroke={P.dawn} strokeWidth={5} strokeLinecap="round" />
                <polygon points="1582,300 1560,290 1560,310" fill={P.dawn} />
                <Tube x={1630} y={300} s={0.9} k={0.7} />
                <text x={1296} y={402} fill={P.cream} fontSize={24} fontFamily={FONT_HEAD}>
                  翻译成激素程序 · 比闹钟早 30–60 分</text>
              </g>
            )}
          </g>
        ))}
        {/* 三力合一（cue1） */}
        {[0, 1, 2].map((i) => {
          const x0 = cards[i].x + 260;
          const p = convK[i];
          const mx = x0 + (960 - x0) * p, my = 440 + (566 - 440) * p;
          return (
            <g key={i} opacity={p}>
              <path d={`M ${x0} 440 Q ${(x0 + 960) / 2} ${500} 960 560`}
                fill="none" stroke={cards[i].col} strokeWidth={5} strokeDasharray="10 10" />
              <circle cx={mx} cy={my + (p > 0.98 ? -2 : 0)} r={10} fill={cards[i].col} />
            </g>
          );
        })}
        {convK[0] > 0.5 && (
          <g opacity={clamp01((convK[0] - 0.5) * 3)}>
            <rect x={770} y={556} width={380} height={68} rx={34} fill={P.paper} />
            <text x={960} y={601} textAnchor="middle" fontSize={30} fill={P.ink}
              fontFamily={FONT_HEAD} fontWeight={900}>三力合一 → 准点醒</text>
          </g>
        )}
        {/* 失灵现场（cue2） */}
        {failK > 0 && (
          <g opacity={failK} transform={`translate(${(1 - failK) * 160},0)`}>
            <rect x={140} y={660} width={1640} height={216} rx={20} fill={P.panel} stroke={P.coral} strokeWidth={4} />
            <text x={176} y={712} fill={P.coral} fontSize={30} fontFamily={FONT_HEAD} fontWeight={800}>
              缺觉时 · 失灵现场</text>
            {/* 腺苷大砝码压住信号 */}
            <g transform="translate(370,800)">
              <rect x={-110} y={-64} width={220} height={52} rx={12} fill={P.vioD} />
              <rect x={-110} y={-64} width={220} height={16} rx={8} fill={P.vio} />
              <rect x={-24} y={-84} width={48} height={22} rx={8} fill={P.vioD} />
              <text y={-30} textAnchor="middle" fontSize={26} fill={P.cream}
                fontFamily={FONT_HEAD} fontWeight={800}>腺苷债</text>
              {[0, 1, 2].map((k) => (
                <path key={k} d={`M ${-60 + k * 60} 18 q 8 10 16 0`} stroke={P.muted}
                  strokeWidth={4} fill="none" opacity={0.6} />
              ))}
              <text y={58} textAnchor="middle" fontSize={22} fill={P.muted}
                fontFamily={FONT_HEAD}>激素信号被压扁</text>
            </g>
            {/* 闹钟响三遍 人不起 */}
            <Alarm x={680} y={780} s={0.62} on={1} t="7:00" />
            <text x={790} y={770} fill={P.coral} fontSize={30} fontFamily={FONT_HEAD} fontWeight={800}>
              ×3 遍 · 没醒</text>
            <Sleeper x={1030} y={830} s={0.55} face="sleep" />
            {/* 失灵三条件 */}
            {['没睡够', '作息乱', '预期模糊'].map((t, i) => (
              <g key={t} transform={`translate(1330,${706 + i * 56})`}>
                <MarkX x={0} y={0} s={0.8} />
                <text x={30} y={9} fill={P.cream} fontSize={26} fontFamily={FONT_HEAD}>{t}</text>
              </g>
            ))}
            <text x={1330} y={866} fill={P.muted} fontSize={22} fontFamily={FONT_HEAD}>
              睡眠压力 ＞ 清醒意愿</text>
          </g>
        )}
      </svg>
      <Chip text="双过程模型 · 博尔贝利" x={1330} y={70} at={beat(0) + 30} accent={P.vio} />
      <Stamp text="缺觉时身体不讲理" x={250} y={600} at={b2 + 25} color={P.coralD} size={40} rot={-4} />
    </FadeIn>
  );
};

/* ============ S4 能练的是三个前提 + 别拧过头 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const ats = [90, 200, 310];
  const pop = ats.map((a) => Math.max(0.001, spring({frame: f - a, fps, config: {damping: 11, stiffness: 120}})));
  const arrowK = clamp01((f - 350) / 24);
  const dialK = clamp01((f - b1) / 22);
  const dialPop = Math.max(0.001, spring({frame: f - b1, fps, config: {damping: 11}}));
  const needle = interpolate(clamp01((f - b1 - 15) / 40), [0, 1], [-110, 40]);

  const cards = [
    {x: 170, col: P.dawn, title: '① 作息固定', cap: '皮质醇曲线才有基准'},
    {x: 725, col: P.sky, title: '② 睡前明确报时', cap: '预期越具体 激素越响应'},
    {x: 1280, col: P.teal, title: '③ 觉要睡够', cap: '债不还清 一切白搭'},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={4} name={sec.name} accent={P.dawn} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {cards.map((c, i) => (
          <g key={i} transform={`translate(${c.x},190) scale(${pop[i]}) translate(${-c.x},-190)`}
            opacity={clamp01(pop[i] * 2)}>
            <rect x={c.x} y={190} width={470} height={420} rx={20} fill={P.panel} stroke={c.col} strokeWidth={4} />
            <text x={c.x + 34} y={248} fill={c.col} fontSize={33} fontFamily={FONT_HEAD} fontWeight={800}>{c.title}</text>
            {i === 0 && (
              <g>
                {/* 日历 + 三个对齐的 7:00 */}
                <g transform="translate(405,380)">
                  <rect x={-95} y={-80} width={190} height={160} rx={14} fill={P.paper} />
                  <rect x={-95} y={-80} width={190} height={40} rx={14} fill={P.dawn} />
                  <rect x={-52} y={-96} width={14} height={30} rx={6} fill={P.cream} />
                  <rect x={38} y={-96} width={14} height={30} rx={6} fill={P.cream} />
                  {[0, 1, 2].map((k) => (
                    <g key={k}>
                      <circle cx={-52 + k * 52} cy={-6} r={6} fill={P.dawnD} />
                      <text x={-52 + k * 52} y={44} textAnchor="middle" fontSize={21}
                        fill={P.ink} fontFamily={FONT_HEAD} fontWeight={800}>7:00</text>
                    </g>
                  ))}
                </g>
                <line x1={300} y1={424} x2={510} y2={424} stroke={P.dawn} strokeWidth={3}
                  strokeDasharray="5 8" opacity={0.7} />
              </g>
            )}
            {i === 1 && (
              <g>
                <circle cx={960} cy={410} r={48} fill={C.skin} />
                <path d="M 916 398 Q 960 364 1004 398 L 1004 410 Q 960 382 916 410 Z" fill="#42304e" />
                <path d="M 946 406 q 7 7 13 0 M 968 406 q 7 7 13 0" stroke={P.ink} strokeWidth={3}
                  fill="none" strokeLinecap="round" />
                {/* 思考泡泡 */}
                <ellipse cx={1100} cy={330} rx={104} ry={58} fill={P.paper} />
                <circle cx={1015} cy={372} r={12} fill={P.paper} />
                <circle cx={997} cy={390} r={7} fill={P.paper} />
                <text x={1100} y={330} textAnchor="middle" fontSize={26} fill={P.ink}
                  fontFamily={FONT_HEAD} fontWeight={800}>明早 7:00</text>
                <text x={1100} y={360} textAnchor="middle" fontSize={22} fill={P.ink}
                  fontFamily={FONT_HEAD}>必起</text>
                <text x={960} y={512} textAnchor="middle" fontSize={22} fill={P.muted}
                  fontFamily={FONT_HEAD}>「早点起就行」→ 太模糊 ✗</text>
              </g>
            )}
            {i === 2 && (
              <g>
                <rect x={1410} y={360} width={260} height={70} rx={16} fill="none"
                  stroke={P.cream} strokeWidth={5} />
                <rect x={1670} y={382} width={16} height={26} rx={7} fill={P.cream} />
                <rect x={1418} y={368} width={244 * clamp01((f - ats[2] - 15) / 60)} height={54} rx={12} fill={P.teal} />
                <text x={1540} y={492} textAnchor="middle" fontSize={24} fill={P.cream}
                  fontFamily={FONT_HEAD} fontWeight={700}>腺苷债清零</text>
                <text x={1540} y={534} textAnchor="middle" fontSize={22} fill={P.muted}
                  fontFamily={FONT_HEAD}>不然激素叫不动</text>
              </g>
            )}
            <text x={c.x + 34} y={574} fill={P.muted} fontSize={24} fontFamily={FONT_HEAD}>{c.cap}</text>
          </g>
        ))}
        {/* 底部小结（cue0 尾） */}
        {arrowK > 0 && (
          <g opacity={arrowK} transform={`translate(0,${(1 - arrowK) * 40})`}>
            <path d="M 210 690 L 360 690" stroke={P.gold} strokeWidth={6} strokeLinecap="round" />
            <polygon points="372,690 350,680 350,700" fill={P.gold} />
            <text x={400} y={700} fill={P.cream} fontSize={29} fontFamily={FONT_HEAD} fontWeight={700}>
              原料靠谱 → 体内的闹钟才靠谱</text>
          </g>
        )}
        {/* 灵敏度旋钮警告（cue1） */}
        {dialK > 0 && (
          <g opacity={dialK} transform={`translate(1300,660) scale(${dialPop}) translate(-1300,-660)`}>
            <rect x={1300} y={660} width={480} height={216} rx={20} fill={P.panel} stroke={P.coral} strokeWidth={4} />
            <text x={1330} y={712} fill={P.coral} fontSize={28} fontFamily={FONT_HEAD} fontWeight={800}>
              灵敏度可调 · 别拧过头</text>
            <g transform="translate(1430,806)">
              <circle r={52} fill={P.card} stroke={P.cream} strokeWidth={4} />
              {[-110, -55, 0, 40].map((a, i) => (
                <line key={a} x1={0} y1={-44} x2={0} y2={-52} stroke={i === 3 ? P.coral : P.muted}
                  strokeWidth={4} transform={`rotate(${a})`} />
              ))}
              <line x1={0} y1={0} x2={0} y2={-38} stroke={P.coral} strokeWidth={6}
                strokeLinecap="round" transform={`rotate(${needle})`} />
              <circle r={7} fill={P.coral} />
            </g>
            <text x={1530} y={790} fill={P.cream} fontSize={25} fontFamily={FONT_HEAD} fontWeight={700}>
              过度警觉</text>
            <text x={1530} y={828} fill={P.muted} fontSize={23} fontFamily={FONT_HEAD}>
              → 睡不踏实 · 反成失眠</text>
          </g>
        )}
      </svg>
      <Chip text="人人都有的调度系统" x={1330} y={76} at={beat(0) + 30} accent={P.dawn} />
      <Stamp text="练的是前提 · 不是玄学" x={230} y={762} at={b1 + 55} color={P.dawnD} size={40} rot={3} />
    </FadeIn>
  );
};

/* ============ 结尾卡 ============ */
const EndCard = () => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 14}});
  const pop = Math.max(0.001, spring({frame: f - 5, fps, config: {damping: 10, stiffness: 140}}));
  const riseK = clamp01(f / 30);
  return (
    <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
      <Backdrop top={P.bgDeep} bottom={P.bgDeep} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={18} from={0} dur={22} />
        {/* 地平线日出 */}
        <line x1={240} y1={800} x2={1680} y2={800} stroke={P.card} strokeWidth={6} strokeLinecap="round" />
        <g transform={`translate(0,${(1 - riseK) * 90})`} opacity={riseK}>
          <circle cx={960} cy={800} r={130} fill={P.dawn} />
          <circle cx={960} cy={800} r={130} fill="none" stroke={P.gold} strokeWidth={5} />
        </g>
        {[0, 1, 2].map((i) => {
          const a = (-155 + i * 40) * Math.PI / 180;
          return <rect key={i} x={-7} y={-176} width={14} height={40} rx={7} fill={P.gold}
            transform={`translate(960,800) rotate(${(-155 + i * 40)}) translate(0,${-10 * riseK})`} opacity={riseK} />;
        })}
        <Alarm x={1360} y={756} s={0.85} on={1} t="7:00" />
        <g transform={`translate(560,782) scale(${pop})`} opacity={clamp01(pop)}>
          <Eye x={0} y={0} s={1.1} />
        </g>
        {/* 飞鸟 */}
        {[0, 1].map((i) => (
          <path key={i} d="M -22 0 Q -10 -14 0 0 Q 10 -14 22 0" stroke={P.muted} strokeWidth={4}
            fill="none" strokeLinecap="round"
            transform={`translate(${640 + i * 90 + Math.sin(f / 8 + i) * 10},${620 - i * 40})`} />
        ))}
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -260}}>
        <div style={{fontSize: 56, fontWeight: 800, color: C.text}}>身体比闹钟更早知道你要干嘛</div>
        <div style={{fontSize: 36, color: P.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E228 = ({data = tl}: {data?: any}) => {
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
