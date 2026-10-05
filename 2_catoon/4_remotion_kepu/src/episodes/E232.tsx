import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01, rand} from '../lib/style';
import {Backdrop, Stamp, SceneTag, Chip, Subtitles, ProgressBar, FadeIn} from '../lib/atoms';
import tl from '../timeline_232.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：酒店夜靛 × 警戒琥珀 × 睡眠紫（纯色平涂） ============ */
const P = {
  bg: '#181b3e',
  bgDeep: '#0f1129',
  bgLite: '#20265a',
  card: '#262c5e',
  cardD: '#1b2049',
  cream: '#f5eedd',
  ink: '#10142e',
  moon: '#ffd873',
  moonD: '#e0ab40',
  alert: '#ffb84d',
  alertD: '#dd8f2e',
  sleep: '#9b8cf0',
  sleepD: '#6f5fc9',
  sky: '#6fc4ff',
  skyD: '#3d82cc',
  cyan: '#66d9e8',
  cyanD: '#35a9c0',
  coral: '#ff7d6e',
  coralD: '#d35a4b',
  mint: '#86e2b0',
  mintD: '#4dab7d',
  grey: '#93a0c2',
  skin: '#f2c79b',
  skinD: '#d99f6b',
  hair: '#3d3252',
  pillow: '#eef1fb',
  mattress: '#dde2f5',
  blanket: '#5a6fb8',
  blanketD: '#3f4f8c',
  wood: '#8f6240',
  woodD: '#6a4729',
  gold: '#e8c15a',
  dolphin: '#6fa8d8',
  dolphinD: '#4a7fb5',
  bird: '#f2a65e',
  birdD: '#c97f3c',
  leaf: '#6fdc8c',
};

/* ---------- 月亮 ---------- */
const MoonIcon = ({x, y, s = 1, op = 1, hole = P.bg}: {x: number; y: number; s?: number; op?: number; hole?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
    <circle r={56} fill={P.moon} />
    <circle cx={26} cy={-16} r={46} fill={hole} />
    <circle cx={-18} cy={22} r={8} fill={P.moonD} />
    <circle cx={-34} cy={-6} r={5} fill={P.moonD} />
  </g>
);

/* ---------- 酒店床：高床头板 ---------- */
const HotelBed = ({x, y, s = 1, op = 1}: {x: number; y: number; s?: number; op?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
    <rect x={-196} y={-196} width={56} height={222} rx={16} fill={P.gold} />
    <rect x={-186} y={-184} width={36} height={198} rx={12} fill={P.woodD} />
    <rect x={-150} y={26} width={26} height={52} rx={8} fill={P.woodD} />
    <rect x={150} y={26} width={26} height={52} rx={8} fill={P.woodD} />
    <rect x={-168} y={-40} width={356} height={74} rx={18} fill={P.mattress} stroke={P.card} strokeWidth={5} />
    <rect x={-146} y={-68} width={104} height={42} rx={19} fill={P.pillow} stroke={P.card} strokeWidth={5} />
    <rect x={-30} y={-72} width={218} height={80} rx={18} fill={P.blanket} stroke={P.blanketD} strokeWidth={5} />
    <rect x={-30} y={-72} width={218} height={24} rx={12} fill={P.blanketD} opacity={0.55} />
  </g>
);

/* ---------- 旅人：深发；awake=睁眼失眠 / z=睡着 ---------- */
const Traveler = ({x, y, s = 1, op = 1, z = 0, awake = 0, rot = 0}: {
  x: number; y: number; s?: number; op?: number; z?: number; awake?: number; rot?: number;
}) => {
  const f = useCurrentFrame();
  return (
    <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`} opacity={op}>
      <circle r={46} fill={P.skin} stroke={P.skinD} strokeWidth={5} />
      {/* 深发盖 */}
      <path d="M -46 -8 A 46 46 0 0 1 46 -8 L 46 -16 A 46 46 0 0 0 -46 -16 Z" fill={P.hair} />
      <circle cx={-42} cy={-18} r={11} fill={P.hair} />
      <circle cx={42} cy={-18} r={11} fill={P.hair} />
      {awake > 0.5 ? (
        <g>
          <circle cx={-17} cy={8} r={5.5} fill={P.ink} />
          <circle cx={19} cy={8} r={5.5} fill={P.ink} />
          <path d="M -26 -4 Q -17 -9 -8 -4" stroke={P.ink} strokeWidth={3.5} fill="none" strokeLinecap="round" />
          <path d="M 10 -4 Q 19 -9 28 -4" stroke={P.ink} strokeWidth={3.5} fill="none" strokeLinecap="round" />
          <path d="M -4 34 Q 4 31 12 34" stroke={P.ink} strokeWidth={4} fill="none" strokeLinecap="round" />
        </g>
      ) : (
        <g stroke={P.ink} strokeWidth={4} fill="none" strokeLinecap="round">
          <path d="M -26 8 Q -17 16 -8 8" />
          <path d="M 10 8 Q 19 16 28 8" />
          <path d="M -2 30 Q 6 35 14 30" />
        </g>
      )}
      {z > 0 && [0, 1, 2].map((i) => {
        const t = (f / 30 + i * 0.9) % 3;
        return (
          <text key={i} x={56 + i * 40 + t * 22} y={-36 - t * 52} fill={P.sleep} fontSize={42 - i * 7}
            fontFamily={FONT_HEAD} fontWeight={800} opacity={z * (1 - t / 3)}>Z</text>
        );
      })}
    </g>
  );
};

/* ---------- 行李箱 ---------- */
const Suitcase = ({x, y, s = 1, op = 1}: {x: number; y: number; s?: number; op?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
    <line x1={-16} y1={-92} x2={-16} y2={-118} stroke={P.grey} strokeWidth={8} strokeLinecap="round" />
    <line x1={16} y1={-92} x2={16} y2={-118} stroke={P.grey} strokeWidth={8} strokeLinecap="round" />
    <rect x={-30} y={-132} width={60} height={18} rx={8} fill={P.grey} />
    <rect x={-66} y={-92} width={132} height={118} rx={18} fill={P.coral} stroke={P.coralD} strokeWidth={5} />
    <line x1={-22} y1={-92} x2={-22} y2={26} stroke={P.coralD} strokeWidth={5} opacity={0.6} />
    <line x1={22} y1={-92} x2={22} y2={26} stroke={P.coralD} strokeWidth={5} opacity={0.6} />
    <rect x={-40} y={-14} width={80} height={26} rx={8} fill={P.cream} opacity={0.9} />
    <circle cx={-36} cy={36} r={11} fill={P.ink} />
    <circle cx={36} cy={36} r={11} fill={P.ink} />
  </g>
);

/* ---------- 房卡 ---------- */
const Keycard = ({x, y, s = 1, rot = 0, op = 1}: {x: number; y: number; s?: number; rot?: number; op?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`} opacity={op}>
    <rect x={-52} y={-32} width={104} height={64} rx={10} fill={P.gold} stroke={P.moonD} strokeWidth={4} />
    <rect x={-52} y={-12} width={104} height={14} fill={P.moonD} opacity={0.7} />
    <circle cx={-30} cy={16} r={8} fill={P.cardD} />
    <rect x={-8} y={10} width={44} height={10} rx={5} fill={P.cardD} opacity={0.6} />
  </g>
);

/* ---------- 数字时钟 ---------- */
const DigiClock = ({x, y, s = 1, txt = '01:47', op = 1}: {x: number; y: number; s?: number; txt?: string; op?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
    <rect x={-92} y={-44} width={184} height={88} rx={14} fill={P.ink} stroke={P.card} strokeWidth={5} />
    <text x={0} y={16} textAnchor="middle" fontSize={46} fill={P.coral} fontFamily={FONT_HEAD} fontWeight={800}
      letterSpacing={4}>{txt}</text>
  </g>
);

/* ---------- 俯视裂脑：左右半球可独立着色 ---------- */
const SplitBrain = ({x, y, s = 1, op = 1, leftC = P.alert, leftD = P.alertD,
  rightC = P.sleep, rightD = P.sleepD, eye = 1, zz = 0}: {
  x: number; y: number; s?: number; op?: number; leftC?: string; leftD?: string;
  rightC?: string; rightD?: string; eye?: number; zz?: number;
}) => {
  const f = useCurrentFrame();
  return (
    <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
      <path d="M 0 -88 C -62 -96 -102 -58 -100 -6 C -98 44 -56 92 0 88 Z"
        fill={leftC} stroke={leftD} strokeWidth={6} strokeLinejoin="round" />
      <path d="M 0 -88 C 62 -96 102 -58 100 -6 C 98 44 56 92 0 88 Z"
        fill={rightC} stroke={rightD} strokeWidth={6} strokeLinejoin="round" />
      <line x1={0} y1={-86} x2={0} y2={86} stroke={P.bgDeep} strokeWidth={7} strokeLinecap="round" />
      <g stroke={leftD} strokeWidth={5} fill="none" strokeLinecap="round" opacity={0.85}>
        <path d="M -70 -48 Q -46 -64 -26 -44" />
        <path d="M -78 4 Q -50 -12 -30 8" />
        <path d="M -58 48 Q -38 62 -18 46" />
      </g>
      <g stroke={rightD} strokeWidth={5} fill="none" strokeLinecap="round" opacity={0.85}>
        <path d="M 26 -44 Q 46 -64 70 -48" />
        <path d="M 30 8 Q 50 -12 78 4" />
        <path d="M 18 46 Q 38 62 58 48" />
      </g>
      {eye > 0 && (
        <g opacity={eye}>
          <ellipse cx={-52} cy={-6} rx={17} ry={12} fill="#fff" />
          <circle cx={-52} cy={-6} r={6.5} fill={P.ink} />
          <circle cx={-54} cy={-9} r={2.2} fill="#fff" />
        </g>
      )}
      {zz > 0 && (
        <g opacity={zz}>
          <path d="M 34 -8 Q 52 4 70 -8" stroke={P.bgDeep} strokeWidth={5} fill="none" strokeLinecap="round" />
          {[0, 1].map((i) => {
            const t = (f / 34 + i * 1.1) % 2.6;
            return (
              <text key={i} x={46 + i * 30 + t * 16} y={-60 - t * 40} fill={P.sleep} fontSize={34 - i * 6}
                fontFamily={FONT_HEAD} fontWeight={800} opacity={1 - t / 2.6}>Z</text>
            );
          })}
        </g>
      )}
    </g>
  );
};

/* ---------- 警戒雷达弧 ---------- */
const Radar = ({x, y, s = 1, color = P.alert}: {x: number; y: number; s?: number; color?: string}) => {
  const f = useCurrentFrame();
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      {[0, 1, 2].map((i) => {
        const t = ((f * 0.9 + i * 22) % 66) / 66;
        return (
          <path key={i} d={`M ${-130 - t * 90} -60 A ${150 + t * 90} ${150 + t * 90} 0 0 0 ${-130 - t * 90} 60`}
            fill="none" stroke={color} strokeWidth={7} strokeLinecap="round" opacity={(1 - t) * 0.85} />
        );
      })}
    </g>
  );
};

/* ---------- 喇叭 + 声波 ---------- */
const Speaker = ({x, y, s = 1, on = 1}: {x: number; y: number; s?: number; on?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-52} y={-26} width={44} height={52} rx={10} fill={P.grey} />
    <path d="M -8 -26 L 40 -56 L 40 56 L -8 26 Z" fill={P.grey} />
    <g opacity={on} stroke={P.coral} strokeWidth={7} fill="none" strokeLinecap="round">
      <path d="M 56 -24 A 34 34 0 0 1 56 24" />
      <path d="M 74 -40 A 58 58 0 0 1 74 40" />
      <path d="M 92 -56 A 82 82 0 0 1 92 56" />
    </g>
  </g>
);

/* ---------- 波浪路径 ---------- */
const wave = (x0: number, y0: number, w: number, amp: number, period: number, jit = 0) => {
  const pts: string[] = [];
  const n = Math.max(8, Math.floor(w / 9));
  for (let i = 0; i <= n; i++) {
    const x = x0 + (w * i) / n;
    const y = y0 + Math.sin(((x - x0) / period) * Math.PI * 2) * amp + (jit ? Math.sin(i * 3.7) * jit : 0);
    pts.push(`${i === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`);
  }
  return pts.join(' ');
};

/* ---------- 海豚：睁眼值班 ---------- */
const Dolphin = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <path d="M -104 6 L -150 -30 L -134 8 L -152 42 L -106 22 Z" fill={P.dolphinD} />
    <path d="M -108 4 C -70 -48 20 -56 78 -26 C 104 -12 122 0 138 -4 C 148 -6 152 4 142 10
      C 120 24 92 22 62 30 C 8 46 -60 44 -96 24 C -114 14 -118 12 -108 4 Z"
      fill={P.dolphin} stroke={P.dolphinD} strokeWidth={5} strokeLinejoin="round" />
    <path d="M -6 -50 Q 6 -86 34 -72 Q 18 -56 20 -42 Z" fill={P.dolphinD} />
    <path d="M 30 26 Q 44 52 68 44 Q 52 26 50 18 Z" fill={P.dolphinD} />
    <circle cx={96} cy={-8} r={7} fill="#fff" />
    <circle cx={97} cy={-8} r={3.6} fill={P.ink} />
    <path d="M 112 8 Q 120 13 128 9" stroke={P.dolphinD} strokeWidth={4} fill="none" strokeLinecap="round" />
  </g>
);

/* ---------- 候鸟：枝头单半球值班 ---------- */
const Bird = ({x, y, s = 1}: {x: number; y: number; s?: number}) => {
  const f = useCurrentFrame();
  const bob = Math.sin(f / 16) * 4;
  return (
    <g transform={`translate(${x},${y + bob}) scale(${s})`}>
      <line x1={-140} y1={96} x2={140} y2={96} stroke={P.woodD} strokeWidth={12} strokeLinecap="round" />
      <line x1={90} y1={96} x2={128} y2={60} stroke={P.woodD} strokeWidth={9} strokeLinecap="round" />
      <ellipse cx={118} cy={52} rx={22} ry={10} fill={P.leaf} transform="rotate(-30 118 52)" />
      <line x1={-18} y1={60} x2={-18} y2={94} stroke={P.birdD} strokeWidth={7} strokeLinecap="round" />
      <line x1={14} y1={60} x2={14} y2={94} stroke={P.birdD} strokeWidth={7} strokeLinecap="round" />
      <ellipse rx={62} ry={48} fill={P.bird} stroke={P.birdD} strokeWidth={5} />
      <ellipse cx={-6} cy={16} rx={40} ry={26} fill="#f8cf9a" opacity={0.85} />
      <path d="M -58 -6 L -104 -26 L -96 2 L -108 24 L -60 22 Z" fill={P.birdD} />
      <ellipse cx={-8} cy={0} rx={34} ry={26} fill={P.birdD} opacity={0.45} />
      <circle cx={38} cy={-38} r={30} fill={P.bird} stroke={P.birdD} strokeWidth={5} />
      <path d="M 64 -44 L 96 -34 L 64 -26 Z" fill={P.gold} />
      <circle cx={44} cy={-44} r={6.5} fill="#fff" />
      <circle cx={45} cy={-44} r={3.4} fill={P.ink} />
      <path d="M 20 -60 Q 38 -72 56 -60" stroke={P.birdD} strokeWidth={5} fill="none" strokeLinecap="round" />
    </g>
  );
};

/* ---------- 枕头 ---------- */
const PillowIcon = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <path d="M -80 -34 Q 0 -52 80 -34 Q 96 0 80 34 Q 0 52 -80 34 Q -96 0 -80 -34 Z"
      fill={P.pillow} stroke={P.card} strokeWidth={5} />
    <path d="M -62 -18 Q 0 -32 62 -18" stroke={P.grey} strokeWidth={4} fill="none" opacity={0.6} strokeLinecap="round" />
    <circle cx={-76} cy={-30} r={7} fill={P.pillow} stroke={P.card} strokeWidth={4} />
    <circle cx={76} cy={-30} r={7} fill={P.pillow} stroke={P.card} strokeWidth={4} />
  </g>
);

/* ---------- 时钟：提前到 ---------- */
const ClockIcon = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <circle r={58} fill={P.bgLite} stroke={P.gold} strokeWidth={8} />
    {Array.from({length: 12}).map((_, i) => {
      const a = (i * 30 * Math.PI) / 180;
      return <circle key={i} cx={Math.sin(a) * 46} cy={-Math.cos(a) * 46} r={i % 3 === 0 ? 4 : 2.5} fill={P.grey} />;
    })}
    <line x1={0} y1={0} x2={0} y2={-34} stroke={P.cream} strokeWidth={7} strokeLinecap="round" />
    <line x1={0} y1={0} x2={-26} y2={10} stroke={P.coral} strokeWidth={7} strokeLinecap="round" />
    <circle r={7} fill={P.cream} />
  </g>
);

/* ---------- 连住：夜1× → 夜2√ ---------- */
const NightsIcon = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-88} y={-52} width={80} height={96} rx={14} fill={P.card} stroke={P.coral} strokeWidth={4} />
    <MoonIcon x={-48} y={-22} s={0.28} hole={P.card} />
    <text x={-48} y={32} textAnchor="middle" fontSize={26} fill={P.coral} fontFamily={FONT_HEAD} fontWeight={800}>第1晚</text>
    <g stroke={P.coral} strokeWidth={7} strokeLinecap="round">
      <line x1={-76} y1={-40} x2={-22} y2={14} />
      <line x1={-22} y1={-40} x2={-76} y2={14} />
    </g>
    <rect x={12} y={-52} width={80} height={96} rx={14} fill={P.card} stroke={P.mint} strokeWidth={4} />
    <MoonIcon x={52} y={-22} s={0.28} hole={P.card} />
    <text x={52} y={32} textAnchor="middle" fontSize={26} fill={P.mint} fontFamily={FONT_HEAD} fontWeight={800}>第2晚</text>
    <path d="M 34 -32 L 46 -18 L 74 -46" stroke={P.mint} strokeWidth={8} fill="none" strokeLinecap="round" strokeLinejoin="round" />
  </g>
);

/* ---------- 心：接受它 ---------- */
const HeartIcon = ({x, y, s = 1}: {x: number; y: number; s?: number}) => {
  const f = useCurrentFrame();
  const beat = 1 + Math.sin(f / 8) * 0.05;
  return (
    <g transform={`translate(${x},${y}) scale(${s * beat})`}>
      <path d="M 0 52 C -64 12 -78 -22 -56 -44 C -38 -62 -12 -54 0 -32 C 12 -54 38 -62 56 -44
        C 78 -22 64 12 0 52 Z" fill={P.coral} stroke={P.coralD} strokeWidth={5} strokeLinejoin="round" />
      <path d="M -34 -8 L -12 -8 L -2 -24 L 12 6 L 22 -8 L 40 -8" stroke="#fff" strokeWidth={6}
        fill="none" strokeLinecap="round" strokeLinejoin="round" />
    </g>
  );
};

/* ============ 标题卡 ============ */
const TitleCard = ({data}: {data: any}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 13, mass: 0.7}});
  const o = interpolate(f, [0, 8], [0, 1]);
  const p0 = Math.max(0.001, spring({frame: f - 4, fps, config: {damping: 10, stiffness: 150}}));
  const p1 = Math.max(0.001, spring({frame: f - 14, fps, config: {damping: 10, stiffness: 150}}));
  return (
    <AbsoluteFill>
      <Backdrop top={P.bg} bottom={P.bg} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <circle cx={960} cy={338} r={222} fill={P.cream} />
        <circle cx={960} cy={338} r={200} fill={P.bgLite} />
        <g transform={`translate(960,338) scale(${p0})`}>
          <g transform="translate(-960,-338)">
            <MoonIcon x={1092} y={226} s={0.5} hole={P.bgLite} />
            {[0, 1, 2].map((i) => (
              <circle key={i} cx={810 + rand(i, 51) * 240} cy={200 + rand(i, 52) * 70} r={3}
                fill="#fff" opacity={0.8} />
            ))}
            <HotelBed x={972} y={468} s={0.82} />
            <Traveler x={892} y={428} s={0.78} awake={1} />
            <Suitcase x={1098} y={430} s={0.5} />
          </g>
        </g>
        <g transform={`translate(700,566) rotate(-7) scale(${p1})`}>
          <rect x={-138} y={-34} width={276} height={64} rx={32} fill={P.card} stroke={P.gold} strokeWidth={4} />
          <text x={0} y={12} textAnchor="middle" fontSize={30} fill={P.gold} fontFamily={FONT_HEAD} fontWeight={800}>酒店床更软</text>
        </g>
        <g transform={`translate(1230,566) rotate(6) scale(${p1})`}>
          <rect x={-148} y={-34} width={296} height={64} rx={32} fill={P.card} stroke={P.coral} strokeWidth={4} />
          <text x={0} y={12} textAnchor="middle" fontSize={30} fill={P.coral} fontFamily={FONT_HEAD} fontWeight={800}>就是睡不着</text>
        </g>
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 330}}>
          <div style={{color: P.gold, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 26}}>
            冷知识 · 第 {data.episode} 期
          </div>
          <div style={{color: C.text, fontSize: 88, fontWeight: 900, fontFamily: FONT_HEAD, letterSpacing: 6}}>
            {data.title}
          </div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ============ S0 酒店夜 ×「半个哨兵」裂脑 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b0 = beat(0), b1 = beat(1);

  const brainK = Math.max(0.001, spring({frame: f - b1 - 6, fps, config: {damping: 12, stiffness: 140}}));
  const tagK = clamp01((f - b1 - 120) / 16);

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={0} name={sec.name} accent={P.gold} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：酒店房间 */}
        <ellipse cx={430} cy={880} rx={360} ry={28} fill={P.bgDeep} />
        <rect x={96} y={150} width={230} height={260} rx={14} fill={P.bgDeep} stroke={P.card} strokeWidth={6} />
        <MoonIcon x={172} y={232} s={0.5} hole={P.bgDeep} />
        {[0, 1, 2].map((i) => (
          <circle key={i} cx={130 + rand(i, 61) * 160} cy={300 + rand(i, 62) * 80} r={2.6} fill="#fff" opacity={0.7} />
        ))}
        <line x1={211} y1={150} x2={211} y2={410} stroke={P.card} strokeWidth={5} />
        <rect x={88} y={140} width={246} height={16} rx={8} fill={P.gold} opacity={0.5} />
        <HotelBed x={430} y={792} s={1.4} />
        <Traveler x={288} y={718} s={1.05} awake={1} />
        <text x={288} y={636} textAnchor="middle" fill={P.coral} fontSize={27} fontFamily={FONT_HEAD} fontWeight={700}>
          床更软 却睡不着</text>
        {/* 床头柜 + 房卡 + 闹钟 */}
        <rect x={666} y={726} width={116} height={94} rx={10} fill={P.wood} stroke={P.woodD} strokeWidth={5} />
        <rect x={686} y={760} width={76} height={14} rx={7} fill={P.woodD} />
        <DigiClock x={726} y={688} s={0.56} txt="01:47" />
        <Keycard x={790} y={656} s={0.62} rot={-14} />
        <Suitcase x={86} y={742} s={0.85} />
        <line x1={884} y1={170} x2={884} y2={860} stroke={P.card} strokeWidth={4} strokeDasharray="4 20" opacity={0.6} />
        {/* 右：裂脑哨兵 */}
        <g transform={`translate(1400,470) scale(${brainK})`} opacity={brainK}>
          <circle r={252} fill={P.card} />
          <circle r={252} fill="none" stroke={P.gold} strokeWidth={4} strokeDasharray="14 10" opacity={0.5} />
          <SplitBrain x={-30} y={-6} s={1.5} eye={1} zz={1} />
          <Radar x={-160} y={-10} s={0.9} />
        </g>
        <g opacity={tagK}>
          <rect x={1180} y={706} width={200} height={56} rx={28} fill={P.alert} />
          <text x={1280} y={744} textAnchor="middle" fontSize={28} fill={P.ink} fontFamily={FONT_HEAD} fontWeight={800}>
            左半球·放哨</text>
          <rect x={1420} y={706} width={200} height={56} rx={28} fill={P.sleep} />
          <text x={1520} y={744} textAnchor="middle" fontSize={28} fill={P.ink} fontFamily={FONT_HEAD} fontWeight={800}>
            右半球·睡觉</text>
        </g>
      </svg>
      <Chip text="不是认床 · 是认环境" x={100} y={130} at={b0 + 18} accent={P.gold} />
      <Chip text="几乎人人都中招" x={120} y={836} at={b1 + 40} accent={P.sleep} />
      <Stamp text="首夜效应" x={1560} y={116} at={b1 + 110} color={P.alertD} size={54} rot={5} />
    </FadeIn>
  );
};

/* ============ S1 睡眠实验室：电极受试者 + 首夜数据作废 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);

  const wv = clamp01((f - 14) / 90);
  const cardK = Math.max(0.001, spring({frame: f - b1 - 8, fps, config: {damping: 13, stiffness: 140}}));
  const rows = [
    {name: '入睡时间', v1: 0.92, v2: 0.4, unit: '更慢'},
    {name: '深睡占比', v1: 0.3, v2: 0.72, unit: '更少'},
    {name: '醒来次数', v1: 0.85, v2: 0.38, unit: '更多'},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={1} name={sec.name} accent={P.cyan} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 上：实验室场景 */}
        <rect x={110} y={150} width={860} height={340} rx={24} fill={P.card} />
        <text x={150} y={206} fill={P.cream} fontSize={31} fontFamily={FONT_HEAD} fontWeight={800}>
          睡眠实验室 · 满身电极也逃不掉</text>
        <ellipse cx={440} cy={436} rx={300} ry={20} fill={P.bgDeep} />
        <HotelBed x={440} y={390} s={1.15} />
        <Traveler x={322} y={330} s={0.92} awake={1} />
        {/* 电极点 + 导线 */}
        {[[-14, -30], [12, -36], [30, -22], [-32, -18]].map(([dx, dy], i) => (
          <g key={i}>
            <circle cx={322 + dx} cy={330 + dy} r={6} fill={P.cyan} stroke={P.cardD} strokeWidth={2} />
            <path d={`M ${322 + dx} ${330 + dy} Q ${560 + i * 20} ${240 + i * 16} 720 268`}
              stroke={P.cyan} strokeWidth={3.5} fill="none" opacity={0.65} />
          </g>
        ))}
        {/* 监护屏 */}
        <rect x={700} y={220} width={230} height={150} rx={14} fill={P.ink} stroke={P.card} strokeWidth={5} />
        <clipPath id="e232s1w"><rect x={712} y={236} width={206 * wv} height={86} /></clipPath>
        <g clipPath="url(#e232s1w)">
          <path d={wave(716, 292, 200, 20, 60, 7)} fill="none" stroke={P.cyan} strokeWidth={5} strokeLinecap="round" />
        </g>
        <rect x={700} y={236} width={86} height={26} rx={8} fill={P.coral} />
        <text x={743} y={255} textAnchor="middle" fontSize={20} fill={P.ink} fontFamily={FONT_HEAD} fontWeight={800}>第1晚</text>
        <text x={815} y={412} textAnchor="middle" fill={C.muted} fontSize={23} fontFamily={FONT_HEAD}>
          脑电全是警觉痕迹</text>
        {/* 下：首夜 vs 次夜对照 */}
        <g transform={`translate(0,${(1 - cardK) * 60})`} opacity={cardK}>
          <rect x={110} y={530} width={1700} height={330} rx={24} fill={P.card} />
          <text x={150} y={588} fill={P.cream} fontSize={31} fontFamily={FONT_HEAD} fontWeight={800}>
            同样的房间同样的仪器 · 数据差一整截</text>
          <rect x={1400} y={550} width={120} height={40} rx={20} fill={P.coral} />
          <text x={1460} y={578} textAnchor="middle" fontSize={24} fill={P.ink} fontFamily={FONT_HEAD} fontWeight={800}>第1晚</text>
          <rect x={1540} y={550} width={120} height={40} rx={20} fill={P.mint} />
          <text x={1600} y={578} textAnchor="middle" fontSize={24} fill={P.ink} fontFamily={FONT_HEAD} fontWeight={800}>第2晚</text>
          {rows.map((r, i) => {
            const y = 628 + i * 78;
            const k1 = clamp01((f - b1 - 30 - i * 22) / 30);
            const k2 = clamp01((f - b1 - 60 - i * 22) / 30);
            return (
              <g key={i}>
                <text x={170} y={y + 36} fill={P.cream} fontSize={27} fontFamily={FONT_HEAD} fontWeight={700}>
                  {r.name}</text>
                <rect x={330} y={y} width={900} height={30} rx={15} fill={P.bgDeep} />
                <rect x={330} y={y} width={900 * r.v1 * k1} height={30} rx={15} fill={P.coral} />
                <rect x={330} y={y + 38} width={900} height={30} rx={15} fill={P.bgDeep} />
                <rect x={330} y={y + 38} width={900 * r.v2 * k2} height={30} rx={15} fill={P.mint} />
                <text x={1252} y={y + 60} fill={C.muted} fontSize={24} fontFamily={FONT_HEAD}
                  opacity={Math.min(k1, k2)}>{r.unit}</text>
              </g>
            );
          })}
        </g>
        {/* 行规角标 */}
        <g opacity={clamp01((f - b2 - 6) / 16)}>
          <rect x={1010} y={152} width={790} height={336} rx={24} fill="none" stroke={P.coral}
            strokeWidth={5} strokeDasharray="16 12" />
          <text x={1405} y={300} textAnchor="middle" fill={P.coral} fontSize={40} fontFamily={FONT_HEAD}
            fontWeight={900} transform="rotate(-4 1405 300)">首夜数据作废</text>
          <text x={1405} y={360} textAnchor="middle" fill={P.cream} fontSize={28} fontFamily={FONT_HEAD}
            transform="rotate(-4 1405 360)">正式数据从第二晚算起</text>
          <text x={1405} y={428} textAnchor="middle" fill={P.alert} fontSize={30} fontFamily={FONT_HEAD}
            fontWeight={800} transform="rotate(-4 1405 428)">不是心理作用 · 是神经警觉程序</text>
        </g>
      </svg>
      <Chip text="绑满电极也睡不着" x={1500} y={56} at={20} accent={P.cyan} />
      <Stamp text="写在神经里" x={1120} y={150} at={b2 + 30} color={P.coralD} size={48} rot={4} />
    </FadeIn>
  );
};

/* ============ S2 2016 实验：左半球浅睡放哨 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);

  const wvL = clamp01((f - 20) / 90);
  const wvR = clamp01((f - 40) / 90);
  const ping = Math.max(0.001, spring({frame: f - 210, fps, config: {damping: 11, stiffness: 160}}));
  const respL = clamp01((f - 235) / 30);
  const respR = clamp01((f - 250) / 30);
  const resK = clamp01((f - b1 - 6) / 18);

  // 诱发电位尖峰（左侧大，右侧小）
  const spike = (x0: number, y0: number, h: number) =>
    `M ${x0} ${y0} L ${x0 + 18} ${y0 - h} L ${x0 + 36} ${y0}`;

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={2} name={sec.name} accent={P.alert} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <text x={960} y={150} textAnchor="middle" fill={P.cream} fontSize={34} fontFamily={FONT_HEAD} fontWeight={800}>
          2016《当代生物学》· 两半球的深睡不对称</text>
        {/* 左：裂脑 + 声音刺激 */}
        <rect x={100} y={190} width={800} height={560} rx={24} fill={P.card} />
        <SplitBrain x={520} y={430} s={2} eye={1} zz={1} />
        <Radar x={368} y={430} s={1.15} />
        <g transform={`translate(190,430) scale(${ping})`} opacity={ping}>
          <Speaker x={0} y={0} s={1} on={1} />
        </g>
        <text x={190} y={530} textAnchor="middle" fill={P.coral} fontSize={25} fontFamily={FONT_HEAD}
          fontWeight={800} opacity={ping}>异常声音</text>
        <text x={330} y={640} textAnchor="middle" fill={P.alert} fontSize={28} fontFamily={FONT_HEAD} fontWeight={800}>
          左半球·浅睡放哨</text>
        <text x={700} y={640} textAnchor="middle" fill={P.sleep} fontSize={28} fontFamily={FONT_HEAD} fontWeight={800}>
          右半球·正常深睡</text>
        {/* 右：两条慢波通道 */}
        <rect x={960} y={190} width={860} height={270} rx={24} fill={P.card} />
        <text x={1000} y={246} fill={P.alert} fontSize={29} fontFamily={FONT_HEAD} fontWeight={800}>
          左半球慢波 · 睡得浅</text>
        <rect x={1000} y={272} width={780} height={130} rx={14} fill={P.bgDeep} />
        <clipPath id="e232s2a"><rect x={1000} y={272} width={780 * wvL} height={130} /></clipPath>
        <g clipPath="url(#e232s2a)">
          <path d={wave(1010, 337, 760, 22, 120)} fill="none" stroke={P.alert} strokeWidth={7} strokeLinecap="round" />
        </g>
        <g opacity={respL}>
          <path d={spike(1460, 337, 64)} stroke={P.coral} strokeWidth={8} fill="none"
            strokeLinecap="round" strokeLinejoin="round" />
          <text x={1498} y={290} fill={P.coral} fontSize={26} fontFamily={FONT_HEAD} fontWeight={800}>反应大·易醒</text>
        </g>
        <rect x={960} y={480} width={860} height={270} rx={24} fill={P.card} />
        <text x={1000} y={536} fill={P.sleep} fontSize={29} fontFamily={FONT_HEAD} fontWeight={800}>
          右半球慢波 · 睡得沉</text>
        <rect x={1000} y={562} width={780} height={130} rx={14} fill={P.bgDeep} />
        <clipPath id="e232s2b"><rect x={1000} y={562} width={780 * wvR} height={130} /></clipPath>
        <g clipPath="url(#e232s2b)">
          <path d={wave(1010, 627, 760, 52, 150)} fill="none" stroke={P.sleep} strokeWidth={8} strokeLinecap="round" />
        </g>
        <g opacity={respR}>
          <path d={spike(1460, 627, 22)} stroke={P.grey} strokeWidth={6} fill="none"
            strokeLinecap="round" strokeLinejoin="round" />
          <text x={1498} y={600} fill={C.muted} fontSize={24} fontFamily={FONT_HEAD}>反应小</text>
        </g>
        {/* 底条结论 */}
        <g opacity={resK} transform={`translate(0,${(1 - resK) * 30})`}>
          <rect x={330} y={790} width={1260} height={88} rx={20} fill={P.bgDeep} stroke={P.alert} strokeWidth={4} />
          <text x={960} y={846} textAnchor="middle" fill={P.cream} fontSize={32} fontFamily={FONT_HEAD} fontWeight={800}>
            一边在把风 · 一边在睡觉 —— 哨兵的岗位被当场抓获</text>
        </g>
      </svg>
      <Chip text="第二晚两边就一样深" x={110} y={120} at={b1 - 100} accent={P.mint} />
      <Stamp text="哨兵在左半球" x={1480} y={110} at={30} color={P.alertD} size={46} rot={4} />
    </FadeIn>
  );
};

/* ============ S3 动物满配版：海豚候鸟单半球睡眠 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b0 = beat(0), b1 = beat(1);

  const cardA = Math.max(0.001, spring({frame: f - b0 - 10, fps, config: {damping: 12, stiffness: 140}}));
  const cardB = Math.max(0.001, spring({frame: f - b0 - 55, fps, config: {damping: 12, stiffness: 140}}));
  const humanK = clamp01((f - b1 - 8) / 20);

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={3} name={sec.name} accent={P.dolphin} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <text x={960} y={152} textAnchor="middle" fill={P.cream} fontSize={34} fontFamily={FONT_HEAD} fontWeight={800}>
          单半球慢波睡眠 · 动物界的满配技能</text>
        {/* 左：海豚 */}
        <g transform={`translate(${-(1 - cardA) * 60},0)`} opacity={cardA}>
          <rect x={100} y={190} width={840} height={400} rx={24} fill={P.card} />
          <text x={140} y={246} fill={P.dolphin} fontSize={31} fontFamily={FONT_HEAD} fontWeight={800}>
            海豚 · 游泳放哨两不误</text>
          {[0, 1, 2].map((i) => (
            <path key={i} d={`M ${160 + i * 240} 540 q 30 -18 60 0 q 30 18 60 0`}
              stroke={P.skyD} strokeWidth={6} fill="none" opacity={0.5} strokeLinecap="round" />
          ))}
          <Dolphin x={430} y={430} s={1.35} />
          {/* 值班指示 */}
          <g transform="translate(760,330)">
            <rect x={-108} y={-128} width={216} height={236} rx={20} fill={P.bgDeep} stroke={P.dolphinD} strokeWidth={4} />
            <SplitBrain x={-30} y={-50} s={0.62} eye={1} zz={1} />
            <text x={-30} y={56} textAnchor="middle" fontSize={24} fill={P.alert} fontFamily={FONT_HEAD} fontWeight={800}>
              一半醒着</text>
            <text x={-30} y={92} textAnchor="middle" fontSize={24} fill={P.sleep} fontFamily={FONT_HEAD} fontWeight={800}>
              一半深睡</text>
          </g>
          <text x={170} y={560} fill={C.muted} fontSize={25} fontFamily={FONT_HEAD}>
            睁着一只眼 · 控制游泳 · 两半轮流倒班</text>
        </g>
        {/* 右：候鸟 */}
        <g transform={`translate(${(1 - cardB) * 60},0)`} opacity={cardB}>
          <rect x={980} y={190} width={840} height={400} rx={24} fill={P.card} />
          <text x={1020} y={246} fill={P.bird} fontSize={31} fontFamily={FONT_HEAD} fontWeight={800}>
            候鸟 · 边睡边导航</text>
          {[0, 1].map((i) => (
            <path key={i} d={`M ${1040 + i * 330} 320 q 24 -20 48 0 q 24 20 48 0`}
              stroke={P.grey} strokeWidth={5} fill="none" opacity={0.4} strokeLinecap="round" />
          ))}
          <Bird x={1330} y={400} s={1.25} />
          <g transform="translate(1120,470)">
            <rect x={-104} y={-118} width={208} height={228} rx={20} fill={P.bgDeep} stroke={P.birdD} strokeWidth={4} />
            <SplitBrain x={-26} y={-42} s={0.6} eye={1} zz={1} />
            <text x={-26} y={62} textAnchor="middle" fontSize={24} fill={P.alert} fontFamily={FONT_HEAD} fontWeight={800}>
              飞行中照睡</text>
          </g>
        </g>
        {/* 下：人类缩水版 */}
        <g opacity={humanK} transform={`translate(0,${(1 - humanK) * 40})`}>
          <rect x={330} y={640} width={1260} height={200} rx={24} fill={P.card} stroke={P.sleepD} strokeWidth={4} />
          <Traveler x={440} y={740} s={0.85} z={1} />
          <g transform="translate(580,742) scale(0.62)">
            <path d="M 0 -88 C -62 -96 -102 -58 -100 -6 C -98 44 -56 92 0 88 Z"
              fill={P.alert} opacity={0.55} stroke={P.alertD} strokeWidth={6} />
            <path d="M 0 -88 C 62 -96 102 -58 100 -6 C 98 44 56 92 0 88 Z"
              fill={P.sleep} stroke={P.sleepD} strokeWidth={6} />
            <line x1={0} y1={-86} x2={0} y2={86} stroke={P.bgDeep} strokeWidth={7} />
          </g>
          <text x={720} y={716} fill={P.cream} fontSize={31} fontFamily={FONT_HEAD} fontWeight={800}>
            人类只剩缩水版 · 做不到半睡半醒</text>
          <text x={720} y={772} fill={P.sleep} fontSize={27} fontFamily={FONT_HEAD}>
            只能让一边睡得浅一点 · 同一套进化遗产</text>
        </g>
      </svg>
      <Chip text="海洋天空的保命硬件" x={110} y={120} at={b0 + 30} accent={P.dolphin} />
      <Stamp text="同源机制" x={1560} y={110} at={b1 + 60} color={P.sleepD} size={50} rot={4} />
    </FadeIn>
  );
};

/* ============ S4 哄哨兵下班：四张处方卡 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b0 = beat(0), b1 = beat(1), b2 = beat(2);

  const cards = [
    {name: '带上常用枕头', sub: '气味触觉=安全信号', accent: P.sleep, el: <PillowIcon x={0} y={-10} s={1.05} />},
    {name: '下午提前到', sub: '给身体适应窗口', accent: P.gold, el: <ClockIcon x={0} y={-14} s={0.95} />},
    {name: '连住同一家', sub: '次晚标记"已勘查"', accent: P.mint, el: <NightsIcon x={0} y={-10} s={0.95} />},
    {name: '接受它', sub: '知道机制=镇静剂', accent: P.coral, el: <HeartIcon x={0} y={-8} s={0.95} />},
  ];
  const bannerK = clamp01((f - b2 - 5) / 20);

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={4} name={sec.name} accent={P.mint} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <text x={960} y={152} textAnchor="middle" fill={P.cream} fontSize={34} fontFamily={FONT_HEAD} fontWeight={800}>
          思路就一个 · 让大脑觉得「这地方熟」</text>
        {cards.map((c, i) => {
          const g = Math.max(0.001, spring({frame: f - b0 - 16 - i * 30, fps, config: {damping: 12, stiffness: 140}}));
          const cx = 285 + i * 450;
          return (
            <g key={i} transform={`translate(${cx},390) scale(${g})`} opacity={g}>
              <rect x={-205} y={-185} width={410} height={360} rx={24} fill={P.card} />
              <rect x={-205} y={-185} width={410} height={62} rx={24} fill={P.cardD} />
              <text x={0} y={-142} textAnchor="middle" fill={c.accent} fontSize={30} fontFamily={FONT_HEAD} fontWeight={800}>
                第{i + 1}招</text>
              <g transform="translate(0,-32)">{c.el}</g>
              <text x={0} y={108} textAnchor="middle" fill={P.cream} fontSize={30} fontFamily={FONT_HEAD} fontWeight={800}>
                {c.name}</text>
              <text x={0} y={152} textAnchor="middle" fill={C.muted} fontSize={25} fontFamily={FONT_HEAD}>
                {c.sub}</text>
            </g>
          );
        })}
        {/* 底条 */}
        <g opacity={bannerK} transform={`translate(0,${(1 - bannerK) * 30})`}>
          <rect x={330} y={640} width={1260} height={120} rx={24} fill={P.bgDeep} stroke={P.coral} strokeWidth={4} />
          <HeartIcon x={420} y={702} s={0.6} />
          <text x={620} y={690} fill={P.cream} fontSize={33} fontFamily={FONT_HEAD} fontWeight={800}>
            越着急越睡不着 · 双重打击</text>
          <text x={620} y={738} fill={P.coral} fontSize={26} fontFamily={FONT_HEAD}>
            知道在经历什么 · 本身就是镇静剂</text>
        </g>
        <text x={960} y={838} textAnchor="middle" fill={C.muted} fontSize={26} fontFamily={FONT_HEAD} opacity={bannerK}>
          古老的哨兵不知道 · 你已不用在洞穴口听野兽了</text>
      </svg>
      <Chip text="焦虑=抬高警戒阈值" x={110} y={130} at={b1 + 10} accent={P.coral} />
      <Stamp text="哄哨兵下班" x={1550} y={108} at={b1 + 40} color={P.mintD} size={50} rot={4} />
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
        <ellipse cx={960} cy={1015} rx={1100} ry={120} fill={P.bg} />
        <MoonIcon x={640} y={620} s={0.8} hole={P.bgDeep} />
        {[0, 1, 2, 3].map((i) => (
          <circle key={i} cx={520 + rand(i, 71) * 300} cy={500 + rand(i, 72) * 140} r={3} fill="#fff" opacity={0.75} />
        ))}
        <HotelBed x={880} y={940} s={1.4} />
        <Traveler x={742} y={876} s={1} z={1} />
        <g transform={`translate(1420,860) scale(${pop})`}>
          <rect x={-170} y={-70} width={340} height={132} rx={20} fill={P.card} stroke={P.alert} strokeWidth={4} />
          <SplitBrain x={-92} y={-6} s={0.5} eye={1} zz={1} />
          <text x={28} y={-18} textAnchor="middle" fontSize={30} fill={P.alert} fontFamily={FONT_HEAD} fontWeight={800}>
            半个哨兵</text>
          <text x={28} y={30} textAnchor="middle" fontSize={26} fill={C.muted} fontFamily={FONT_HEAD}>
            还在替你值夜</text>
        </g>
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -300}}>
        <div style={{fontSize: 58, fontWeight: 800, color: C.text}}>不是你认床，是大脑在放哨</div>
        <div style={{fontSize: 38, color: C.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E232 = ({data = tl}: {data?: any}) => {
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
