import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01, rand} from '../lib/style';
import {Backdrop, Ripple, Stamp, SceneTag, Chip, Subtitles, ProgressBar, FadeIn} from '../lib/atoms';
import tl from '../timeline_226.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：深青夜 × 暖被琥珀 × 怪梦珊瑚 × 纸米白（纯色平涂） ============ */
const P = {
  bg: '#17393f',
  bgDeep: '#0f282d',
  bgLite: '#1f4a52',
  card: '#285763',
  cardLite: '#356b78',
  cream: '#f5eeda',
  ink: '#0f2329',
  pillow: '#f2ecdc',
  pillowD: '#d8cfb8',
  mattress: '#9fc4bd',
  mattressD: '#6f9a92',
  blanket: '#e0a54d',
  blanketD: '#b87f33',
  skin: '#f2c79b',
  skinD: '#d99f6b',
  hair: '#3a3049',
  coral: '#ff7a66',
  coralD: '#d65a48',
  amber: '#ffd166',
  amberD: '#e0a83e',
  mint: '#8fe3b0',
  mintD: '#4fa876',
  violet: '#b79cf5',
  violetD: '#8467cf',
  sky: '#7ecbf2',
  skyD: '#4d93c4',
  brain: '#f2a9bd',
  brainD: '#d17a94',
  paper: '#f6ecd8',
  paperSh: '#e2d0ae',
  ghost: '#0d1f24',
  ghostHi: '#2a535d',
  grey: '#93a7b3',
};

/* ---------- 月牙 ---------- */
const MoonIcon = ({x, y, s = 1, op = 1, hole = P.bg}: {x: number; y: number; s?: number; op?: number; hole?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
    <circle r={56} fill={P.amber} />
    <circle cx={26} cy={-16} r={46} fill={hole} />
    <circle cx={-18} cy={22} r={8} fill={P.amberD} />
    <circle cx={-34} cy={-6} r={5} fill={P.amberD} />
  </g>
);

/* ---------- 汗滴 ---------- */
const Sweat = ({x, y, on = 1}: {x: number; y: number; on?: number}) => {
  const f = useCurrentFrame();
  if (on <= 0) return null;
  return (
    <g opacity={on}>
      {[0, 1, 2].map((i) => {
        const t = ((f / 36 + i * 0.37) % 1);
        const dx = x + i * 34 - 20 + t * 10;
        const dy = y - t * 44;
        const o = (1 - t) * on;
        return (
          <path key={i} d={`M ${dx} ${dy} C ${dx - 9} ${dy + 12} ${dx - 7} ${dy + 22} ${dx} ${dy + 22}
            C ${dx + 7} ${dy + 22} ${dx + 9} ${dy + 12} ${dx} ${dy} Z`}
            fill={P.sky} opacity={o} />
        );
      })}
    </g>
  );
};

/* ---------- 俯视趴睡者（看到后背，脸埋枕头） ---------- */
const ProneTop = ({x, y, s = 1, op = 1}: {x: number; y: number; s?: number; op?: number}) => {
  const f = useCurrentFrame();
  const breathe = Math.sin(f / 22) * 3;
  return (
    <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
      {/* 床垫 */}
      <rect x={-200} y={-260} width={400} height={560} rx={30} fill={P.mattress} stroke={P.mattressD} strokeWidth={6} />
      {/* 枕头 */}
      <rect x={-150} y={-248} width={300} height={92} rx={34} fill={P.pillow} stroke={P.pillowD} strokeWidth={6} />
      {/* 头：只看到后脑勺 + 侧脸埋进枕头 */}
      <circle cx={0} cy={-196} r={62} fill={P.skin} stroke={P.skinD} strokeWidth={6} />
      <path d="M -62 -196 A 62 62 0 0 1 62 -196 Z" fill={P.hair} transform="translate(0,-8)" />
      <path d="M -30 -142 Q 0 -128 30 -142" stroke={P.skinD} strokeWidth={6} fill="none" strokeLinecap="round" />
      {/* 双臂弯在枕头两侧 */}
      <rect x={-128} y={-180} width={56} height={130} rx={26} fill={P.skin} stroke={P.skinD} strokeWidth={5}
        transform="rotate(18 -100 -150)" />
      <rect x={72} y={-180} width={56} height={130} rx={26} fill={P.skin} stroke={P.skinD} strokeWidth={5}
        transform="rotate(-18 100 -150)" />
      {/* 被子盖到肩 */}
      <g transform={`translate(0,${breathe})`}>
        <rect x={-150} y={-96} width={300} height={330} rx={30} fill={P.blanket} stroke={P.blanketD} strokeWidth={6} />
        <rect x={-150} y={-96} width={300} height={40} rx={20} fill={P.blanketD} opacity={0.6} />
        <rect x={-96} y={60} width={192} height={10} rx={5} fill={P.blanketD} opacity={0.45} />
        <rect x={-96} y={140} width={192} height={10} rx={5} fill={P.blanketD} opacity={0.45} />
      </g>
      {/* 脚丫露出 */}
      <ellipse cx={-52} cy={292} rx={26} ry={36} fill={P.skin} stroke={P.skinD} strokeWidth={5} />
      <ellipse cx={52} cy={292} rx={26} ry={36} fill={P.skin} stroke={P.skinD} strokeWidth={5} />
    </g>
  );
};

/* ---------- 梦境气泡（虚线边=不密封） ---------- */
const DreamBubble = ({x, y, r = 300, op = 1}: {x: number; y: number; r?: number; op?: number}) => (
  <g opacity={op}>
    <circle cx={x - 260} cy={y + 290} r={18} fill={P.paper} />
    <circle cx={x - 190} cy={y + 238} r={28} fill={P.paper} />
    <circle cx={x} cy={y} r={r} fill={P.paper} />
    <circle cx={x} cy={y} r={r} fill="none" stroke={P.coral} strokeWidth={7}
      strokeDasharray="26 20" strokeLinecap="round" />
  </g>
);

/* ---------- 小人剪影（奔跑/站立） ---------- */
const MiniPerson = ({x, y, s = 1, col = P.ink, run = 0, flip = false}: {
  x: number; y: number; s?: number; col?: string; run?: number; flip?: boolean;
}) => {
  const f = useCurrentFrame();
  const sw = Math.sin(f * 0.35) * 14 * run;
  return (
    <g transform={`translate(${x},${y}) scale(${flip ? -s : s},${s})`}>
      <circle cy={-52} r={17} fill={col} />
      <rect x={-13} y={-38} width={26} height={52} rx={13} fill={col} />
      <line x1={-10} y1={-26} x2={-26 - sw} y2={6} stroke={col} strokeWidth={11} strokeLinecap="round" />
      <line x1={10} y1={-26} x2={26 + sw} y2={6} stroke={col} strokeWidth={11} strokeLinecap="round" />
      <line x1={-7} y1={12} x2={-18 + sw} y2={48} stroke={col} strokeWidth={12} strokeLinecap="round" />
      <line x1={7} y1={12} x2={20 - sw} y2={48} stroke={col} strokeWidth={12} strokeLinecap="round" />
    </g>
  );
};

/* ---------- 追赶剪影：大黑影伸爪 ---------- */
const GhostBlob = ({x, y, s = 1, op = 1}: {x: number; y: number; s?: number; op?: number}) => {
  const f = useCurrentFrame();
  const reach = Math.sin(f / 12) * 8;
  return (
    <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
      <path d={`M -70 66 C -86 20 -74 -34 -38 -56 C -8 -76 34 -70 52 -40
        C ${86 + reach} -30 ${88 + reach} 6 56 26 C 76 44 70 62 44 66 Z`} fill={P.ghostHi} />
      <circle cx={-16} cy={-26} r={8} fill={P.cream} />
      <circle cx={22} cy={-26} r={8} fill={P.cream} />
      <path d={`M ${50 + reach} 6 L ${88 + reach} -4 M ${52 + reach} 18 L ${92 + reach} 22`}
        stroke={P.ghostHi} strokeWidth={12} strokeLinecap="round" />
    </g>
  );
};

/* ---------- 被追 ---------- */
const IconChase = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <GhostBlob x={52} y={-6} s={0.9} />
    <MiniPerson x={-58} y={-4} s={0.86} run={1} flip />
    <path d="M -84 52 L -60 52 M -76 64 L -44 64" stroke={P.grey} strokeWidth={6} strokeLinecap="round" />
  </g>
);

/* ---------- 被绑：绳子卷 ---------- */
const IconRope = ({x, y, s = 1}: {x: number; y: number; s?: number}) => {
  const f = useCurrentFrame();
  const w = Math.sin(f / 14) * 4;
  return (
    <g transform={`translate(${x},${y}) rotate(${w}) scale(${s})`}>
      <circle r={46} fill="none" stroke={P.amberD} strokeWidth={13} />
      <circle r={24} fill="none" stroke={P.amberD} strokeWidth={13} />
      <path d="M 34 32 Q 66 44 80 74" fill="none" stroke={P.amberD} strokeWidth={13} strokeLinecap="round" />
      <rect x={-14} y={-72} width={28} height={26} rx={8} fill={P.amberD} />
    </g>
  );
};

/* ---------- 被压：石板压小人 ---------- */
const IconPress = ({x, y, s = 1}: {x: number; y: number; s?: number}) => {
  const f = useCurrentFrame();
  const press = Math.sin(f / 10) * 5;
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      <MiniPerson x={0} y={26} s={0.8} />
      <rect x={-58} y={-72 + press} width={116} height={40} rx={12} fill={P.grey} stroke={P.ink} strokeWidth={5} />
      <path d={`M 0 ${-92 + press} L 0 ${-80 + press}`} stroke={P.coralD} strokeWidth={8} strokeLinecap="round" />
      <path d={`M -10 ${-96 + press} L 10 ${-96 + press} M 0 ${-106 + press} L 0 ${-96 + press}`}
        stroke={P.coralD} strokeWidth={7} strokeLinecap="round" />
    </g>
  );
};

/* ---------- 被困箱中人 ---------- */
const IconTrap = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-56} y={-64} width={112} height={128} rx={14} fill="none" stroke={P.ink} strokeWidth={9} />
    <line x1={-20} y1={-64} x2={-20} y2={64} stroke={P.ink} strokeWidth={7} />
    <line x1={20} y1={-64} x2={20} y2={64} stroke={P.ink} strokeWidth={7} />
    <circle cy={-16} r={14} fill={P.ink} />
    <rect x={-11} y={-4} width={22} height={44} rx={11} fill={P.ink} />
  </g>
);

/* ---------- 侧视趴睡剖面（S2 用）：zone 高亮 ---------- */
const ProneSide = ({x, y, s = 1, zone = 0}: {x: number; y: number; s?: number; zone?: number}) => {
  const f = useCurrentFrame();
  const breathe = Math.sin(f / 20) * 2.5;
  const pulse = 0.5 + 0.5 * Math.sin(f / 8);
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      {/* 床垫 */}
      <rect x={-300} y={64} width={600} height={76} rx={20} fill={P.mattress} stroke={P.mattressD} strokeWidth={6} />
      {/* 枕头 */}
      <rect x={-300} y={8} width={150} height={58} rx={26} fill={P.pillow} stroke={P.pillowD} strokeWidth={6} />
      {/* 躯干趴着：胸口压在床面 */}
      <g transform={`translate(0,${breathe})`}>
        <rect x={-170} y={-8} width={360} height={74} rx={36} fill={P.blanket} stroke={P.blanketD} strokeWidth={6} />
        <rect x={150} y={0} width={120} height={66} rx={30} fill={P.blanketD} />
        {/* 小腿翘一点 */}
        <rect x={228} y={-24} width={70} height={40} rx={18} fill={P.skin} stroke={P.skinD} strokeWidth={5}
          transform="rotate(-18 240 -10)" />
      </g>
      {/* 头扭向一侧，脸埋枕头 */}
      <g>
        <circle cx={-216} cy={2} r={56} fill={P.skin} stroke={P.skinD} strokeWidth={6} />
        <path d="M -272 -2 A 56 56 0 0 1 -160 -6 L -160 -30 A 60 60 0 0 0 -272 -34 Z" fill={P.hair} />
        {/* 挤压出的脸线 */}
        <path d="M -254 34 Q -240 44 -226 36" stroke={P.skinD} strokeWidth={5} fill="none" strokeLinecap="round" />
        <path d="M -214 40 Q -204 46 -196 40" stroke={P.skinD} strokeWidth={5} fill="none" strokeLinecap="round" />
      </g>
      {/* 手臂塞在胸下 */}
      <rect x={-150} y={18} width={130} height={34} rx={17} fill={P.skin} stroke={P.skinD} strokeWidth={5} />
      {/* zone1 呼吸：胸口压缩双向箭头 */}
      {zone === 1 && (
        <g opacity={0.6 + pulse * 0.4}>
          <ellipse cx={-20} cy={52} rx={120} ry={18} fill={P.coral} opacity={0.28} />
          <path d="M -60 12 L -60 52 M -60 52 L -70 40 M -60 52 L -50 40" stroke={P.coral} strokeWidth={7}
            strokeLinecap="round" fill="none" />
          <path d="M -60 12 L -70 24 M -60 12 L -50 24" stroke={P.coral} strokeWidth={7}
            strokeLinecap="round" fill="none" />
        </g>
      )}
      {/* zone2 触觉：脸部压点 */}
      {zone === 2 && (
        <g opacity={0.6 + pulse * 0.4}>
          {[0, 1, 2, 3, 4].map((i) => (
            <circle key={i} cx={-262 + i * 22} cy={56 + (i % 2) * 8} r={6} fill={P.coral} />
          ))}
        </g>
      )}
      {/* zone3 脖子扭转 90° */}
      {zone === 3 && (
        <g opacity={0.75 + pulse * 0.25}>
          <path d="M -158 -52 A 66 66 0 0 0 -196 14" fill="none" stroke={P.violet} strokeWidth={7}
            strokeDasharray="10 8" />
          <text x={-120} y={-64} fill={P.violet} fontSize={34} fontFamily={FONT_HEAD} fontWeight={800}>≈90°</text>
        </g>
      )}
    </g>
  );
};

/* ---------- 呼吸波形 ---------- */
const BreathWave = ({x, y, w = 260, amp = 10, col = P.sky, on = 1}: {
  x: number; y: number; w?: number; amp?: number; col?: string; on?: number;
}) => {
  const f = useCurrentFrame();
  if (on <= 0) return null;
  const pts: string[] = [];
  for (let i = 0; i <= 24; i++) {
    const px = (w * i) / 24;
    const py = Math.sin(i / 3.4 + f / 9) * amp;
    pts.push(`${i === 0 ? 'M' : 'L'} ${px.toFixed(1)} ${py.toFixed(1)}`);
  }
  return (
    <g transform={`translate(${x},${y})`} opacity={on}>
      <path d={pts.join(' ')} fill="none" stroke={col} strokeWidth={6} strokeLinecap="round" />
    </g>
  );
};

/* ---------- 水滴 / 雨云 / 闹铃 / 滑落的被子 / 抽筋腿 / 雪花 / 狗 ---------- */
const DropIcon = ({x, y, s = 1, op = 1}: {x: number; y: number; s?: number; op?: number}) => {
  const f = useCurrentFrame();
  const fall = (f * 3.2) % 90;
  return (
    <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
      <path d="M 0 -44 C -22 -12 -18 22 0 22 C 18 22 22 -12 0 -44 Z" fill={P.sky} />
      <path d={`M 0 ${-60 + fall} C -12 ${-44 + fall} -10 ${-30 + fall} 0 ${-30 + fall}
        C 10 ${-30 + fall} 12 ${-44 + fall} 0 ${-60 + fall} Z`} fill={P.sky} opacity={0.8} />
    </g>
  );
};

const RainCloud = ({x, y, s = 1, op = 1}: {x: number; y: number; s?: number; op?: number}) => {
  const f = useCurrentFrame();
  return (
    <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
      <ellipse cx={-24} cy={-18} rx={38} ry={26} fill={P.grey} />
      <ellipse cx={16} cy={-26} rx={42} ry={30} fill={P.grey} />
      <ellipse cx={4} cy={-6} rx={52} ry={24} fill={P.grey} />
      {[-30, 0, 30].map((dx, i) => {
        const t = (f * 3 + i * 24) % 46;
        return <line key={i} x1={dx} y1={20 + t} x2={dx - 6} y2={38 + t}
          stroke={P.sky} strokeWidth={6} strokeLinecap="round" opacity={1 - t / 46} />;
      })}
    </g>
  );
};

const BellIcon = ({x, y, s = 1, ring = 1}: {x: number; y: number; s?: number; ring?: number}) => {
  const f = useCurrentFrame();
  const j = Math.sin(f * 0.9) * 8 * ring;
  return (
    <g transform={`translate(${x},${y}) rotate(${j}) scale(${s})`}>
      <path d="M -40 18 A 40 44 0 0 1 40 18 L 48 34 L -48 34 Z" fill={P.amber} stroke={P.amberD} strokeWidth={5} />
      <circle cy={-30} r={9} fill={P.amberD} />
      <circle cy={40} r={9} fill={P.coralD} />
      {ring > 0 && (
        <g stroke={P.coral} strokeWidth={5} strokeLinecap="round" opacity={0.8}>
          <path d="M -56 -18 L -70 -30" /><path d="M 56 -18 L 70 -30" />
        </g>
      )}
    </g>
  );
};

const BlanketSlide = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-56} y={-30} width={112} height={60} rx={14} fill={P.blanket} stroke={P.blanketD} strokeWidth={5} />
    <path d="M 20 34 Q 34 52 24 68" fill="none" stroke={P.blanketD} strokeWidth={8} strokeLinecap="round" />
    <path d="M 44 22 Q 62 40 52 60" fill="none" stroke={P.blanketD} strokeWidth={6} strokeLinecap="round" />
    <path d="M -40 -44 L -52 -58 M -8 -48 L -8 -64 M 24 -44 L 36 -58" stroke={P.sky} strokeWidth={6}
      strokeLinecap="round" />
  </g>
);

const LegCramp = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <path d="M -52 -30 L 30 -30 L 44 30 L 12 30 L 4 -6 L -40 -2 Z" fill={P.skin} stroke={P.skinD}
      strokeWidth={5} strokeLinejoin="round" />
    <path d="M -6 -52 L 8 -34 L -8 -26 L 10 -8" fill="none" stroke={P.coral} strokeWidth={7}
      strokeLinecap="round" strokeLinejoin="round" />
    <circle cx={40} cy={30} r={12} fill={P.skin} stroke={P.skinD} strokeWidth={5} />
  </g>
);

const SnowIcon = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} stroke={P.sky} strokeWidth={7} strokeLinecap="round">
    <line x1={-38} y1={0} x2={38} y2={0} />
    <line x1={-19} y1={-33} x2={19} y2={33} />
    <line x1={-19} y1={33} x2={19} y2={-33} />
    <circle r={8} fill={P.sky} stroke="none" />
  </g>
);

const DogIcon = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <ellipse cx={0} cy={10} rx={52} ry={30} fill={P.ghostHi} />
    <circle cx={44} cy={-14} r={24} fill={P.ghostHi} />
    <path d="M 30 -32 L 26 -56 L 44 -40 Z" fill={P.ghostHi} />
    <path d="M 52 -34 L 60 -56 L 68 -38 Z" fill={P.ghostHi} />
    <circle cx={52} cy={-18} r={4.5} fill={P.cream} />
    <circle cx={66} cy={-8} r={6} fill={P.ink} />
    <path d="M -48 2 Q -70 -14 -60 -34" fill="none" stroke={P.ghostHi} strokeWidth={10} strokeLinecap="round" />
    <rect x={-34} y={30} width={14} height={30} rx={7} fill={P.ghostHi} />
    <rect x={14} y={30} width={14} height={30} rx={7} fill={P.ghostHi} />
    <path d="M 62 6 L 78 12 M 62 12 L 76 20" stroke={P.coral} strokeWidth={4} strokeLinecap="round" />
  </g>
);

/* ---------- 转译透镜 ---------- */
const TransLens = ({x, y, h = 760, on = 1}: {x: number; y: number; h?: number; on?: number}) => {
  const f = useCurrentFrame();
  if (on <= 0) return null;
  const wob = Math.sin(f / 11) * 8;
  return (
    <g transform={`translate(${x},${y})`} opacity={on}>
      <path d={`M -58 ${-h / 2} Q ${-58 + wob - 22} ${-h / 4} -58 0 Q ${-58 + wob + 22} ${h / 4} -58 ${h / 2}
        L 58 ${h / 2} Q ${58 + wob + 22} ${h / 4} 58 0 Q ${58 + wob - 22} ${-h / 4} 58 ${-h / 2} Z`}
        fill={P.cardLite} stroke={P.violet} strokeWidth={6} />
      <text x={0} y={-h / 2 - 26} textAnchor="middle" fill={P.violet} fontSize={34}
        fontFamily={FONT_HEAD} fontWeight={800}>转译</text>
      <text x={0} y={h / 2 + 46} textAnchor="middle" fill={P.violet} fontSize={27}
        fontFamily={FONT_HEAD}>经常扭得认不出原件</text>
    </g>
  );
};

/* ---------- 报告卡 ---------- */
const ReportCard = ({x, y, w = 780, h = 168, no, title, sub, icon, accent, at}: {
  x: number; y: number; w?: number; h?: number; no: string; title: string; sub: string;
  icon: React.ReactNode; accent: string; at: number;
}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const g = Math.max(0.001, spring({frame: f - at, fps, config: {damping: 12, stiffness: 140}}));
  return (
    <g transform={`translate(${x - (1 - g) * 70},${y})`} opacity={g}>
      <rect width={w} height={h} rx={22} fill={P.card} />
      <rect width={16} height={h} rx={8} fill={accent} />
      <circle cx={86} cy={h / 2} r={46} fill={P.bgLite} stroke={accent} strokeWidth={4} />
      <text x={86} y={h / 2 + 15} textAnchor="middle" fill={accent} fontSize={44}
        fontFamily={FONT_HEAD} fontWeight={900}>{no}</text>
      <g transform={`translate(170,${h / 2})`}>{icon}</g>
      <text x={272} y={h / 2 - 14} fill={P.cream} fontSize={36} fontFamily={FONT_HEAD} fontWeight={800}>{title}</text>
      <text x={272} y={h / 2 + 34} fill={P.grey} fontSize={27} fontFamily={FONT_HEAD}>{sub}</text>
    </g>
  );
};

/* ---------- 大脑小图标 ---------- */
const BrainDot = ({x, y, s = 1, op = 1}: {x: number; y: number; s?: number; op?: number}) => {
  const f = useCurrentFrame();
  const pulse = 0.85 + 0.15 * Math.sin(f / 9);
  return (
    <g transform={`translate(${x},${y}) scale(${s * pulse})`} opacity={op}>
      <path d="M -64 14 C -82 -14 -70 -48 -44 -58 C -34 -78 -2 -84 18 -70 C 44 -78 70 -60 70 -32
        C 88 -18 86 12 70 24 C 74 46 50 62 28 54 C 12 66 -24 62 -34 46 C -56 46 -72 34 -64 14 Z"
        fill={P.brain} stroke={P.brainD} strokeWidth={5} strokeLinejoin="round" />
      <g stroke={P.brainD} strokeWidth={4} fill="none" strokeLinecap="round">
        <path d="M -46 -26 Q -24 -42 -4 -26" />
        <path d="M 6 -48 Q 30 -58 48 -40" />
        <path d="M -30 12 Q -4 -2 22 12" />
      </g>
    </g>
  );
};

/* ============ 标题卡 ============ */
const TitleCard = ({data}: {data: any}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 13, mass: 0.7}});
  const o = interpolate(f, [0, 8], [0, 1]);
  const pop = Math.max(0.001, spring({frame: f - 4, fps, config: {damping: 10, stiffness: 150}}));
  const pop2 = Math.max(0.001, spring({frame: f - 14, fps, config: {damping: 10, stiffness: 150}}));
  return (
    <AbsoluteFill>
      <Backdrop top={P.bg} bottom={P.bg} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <circle cx={960} cy={330} r={232} fill={P.card} />
        <circle cx={960} cy={330} r={232} fill="none" stroke={P.coral} strokeWidth={7} strokeDasharray="24 18" />
        <g transform={`translate(960,352) scale(${pop})`}>
          <ProneTop x={0} y={18} s={0.62} />
        </g>
        <g transform={`translate(1224,${196 - 6 * Math.sin(f / 14)}) scale(${pop2})`}>
          <circle r={66} fill={P.paper} />
          <circle r={66} fill="none" stroke={P.coral} strokeWidth={6} strokeDasharray="14 12" />
          <GhostBlob x={6} y={10} s={0.5} />
        </g>
        <Sweat x={870} y={228} on={pop} />
        <MoonIcon x={300} y={180} s={0.8} />
        {[0, 1, 2, 3, 4, 5].map((i) => (
          <circle key={i} cx={130 + rand(i, 41) * 1660} cy={110 + rand(i, 42) * 500}
            r={2.4 + rand(i, 43) * 2} fill="#fff" opacity={0.5} />
        ))}
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 380}}>
          <div style={{color: P.coral, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 26}}>
            冷知识 · 第 {data.episode} 期
          </div>
          <div style={{color: C.text, fontSize: 92, fontWeight: 900, fontFamily: FONT_HEAD, letterSpacing: 6}}>
            {data.title}
          </div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ============ S0 一身汗+荒诞梦：俯视趴睡 + 虚线梦泡三剧目 + 信号渗入 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b0 = beat(0), b1 = beat(1), b2 = beat(2);

  const bubK = Math.max(0.001, spring({frame: f - b0 - 24, fps, config: {damping: 13, stiffness: 120}}));
  const icons = [
    {at: b0 + 60, node: <IconChase x={0} y={0} s={1} />, label: '被追', dx: -110, dy: -112},
    {at: b0 + 95, node: <IconRope x={0} y={0} s={0.95} />, label: '被绑', dx: 128, dy: -92},
    {at: b0 + 130, node: <IconPress x={0} y={0} s={1} />, label: '被压', dx: 10, dy: 108},
  ];
  const leakK = clamp01((f - b2 - 8) / 24);

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={0} name={sec.name} accent={P.coral} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <ellipse cx={430} cy={920} rx={330} ry={26} fill={P.bgDeep} />
        <ProneTop x={430} y={580} s={1.06} />
        <Sweat x={366} y={312} on={clamp01((f - b0 - 10) / 20)} />
        <text x={430} y={930} textAnchor="middle" fill={P.grey} fontSize={27} fontFamily={FONT_HEAD}>
          醒来一身汗 · 梦还特别荒诞</text>

        {/* 虚线梦泡：不是密封舱 */}
        <g transform={`translate(1300,420) scale(${bubK})`}>
          <g transform="translate(-1300,-420)">
            <DreamBubble x={1300} y={420} r={305} />
            {icons.map((it, i) => {
              const g = Math.max(0.001, spring({frame: f - it.at, fps, config: {damping: 11, stiffness: 150}}));
              return (
                <g key={i} transform={`translate(${1300 + it.dx},${420 + it.dy}) scale(${g})`} opacity={g}>
                  {it.node}
                  <text x={it.dx > 40 ? -10 : 0} y={it.dy > 0 ? 118 : -118} textAnchor="middle"
                    fill={P.ink} fontSize={30} fontFamily={FONT_HEAD} fontWeight={800}>{it.label}</text>
                </g>
              );
            })}
          </g>
        </g>

        {/* 信号渗入：闹钟在泡外，点线钻进虚线缝 */}
        <g opacity={leakK}>
          <BellIcon x={1690} y={792} s={0.9} ring={f > b2 + 8 ? 1 : 0} />
          <path d="M 1640 752 Q 1560 700 1500 640" fill="none" stroke={P.sky} strokeWidth={6}
            strokeDasharray="2 16" strokeLinecap="round" />
          {[0, 1, 2].map((i) => {
            const t = (f / 26 + i * 0.33) % 1;
            const cx = 1640 - t * 140, cy = 752 - t * 112 - Math.sin(t * Math.PI) * 30;
            return <circle key={i} cx={cx} cy={cy} r={7} fill={P.sky} opacity={1 - t * 0.7} />;
          })}
          <text x={1690} y={880} textAnchor="middle" fill={P.sky} fontSize={27} fontFamily={FONT_HEAD}>
            外面的动静在渗进去</text>
        </g>
        {f > b2 + 8 && <Ripple x={1500} y={640} startFrame={b2 + 8} color={P.sky} />}
      </svg>
      <Chip text="醒来一身汗 + 梦荒诞" x={100} y={150} at={b0 + 16} accent={P.coral} />
      <Stamp text="被追 · 被绑 · 被压" x={1290} y={104} at={b0 + 160} color={P.coralD} size={44} rot={-4} />
      <Chip text="长辈说法里有真东西" x={100} y={240} at={b1 + 14} accent={P.amber} />
      <Chip text="梦不是密封舱 · 一直偷听外界" x={1010} y={790} at={b2 + 40} accent={P.sky} />
    </FadeIn>
  );
};

/* ============ S1 外部刺激合并：左真实刺激 → 转译透镜 → 右梦境版本 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b0 = beat(0), b1 = beat(1), b2 = beat(2);

  const rows = [
    {y: 238, at: b0 + 14, left: <DropIcon x={0} y={-6} s={0.95} />, lt: '皮肤被滴水',
      right: <RainCloud x={0} y={-8} s={1} />, rt: '梦里下大雨', accent: P.sky},
    {y: 408, at: b0 + 85, left: <BellIcon x={0} y={0} s={0.85} ring={1} />, lt: '闹钟铃响前',
      right: <BellIcon x={0} y={0} s={0.85} ring={0} />, rt: '铃声编进剧情', accent: P.amber},
    {y: 578, at: b1 + 18, left: <BlanketSlide x={0} y={-4} s={0.95} />, lt: '被子掉了',
      right: <SnowIcon x={0} y={0} s={1} />, rt: '梦见冷', accent: P.sky},
    {y: 748, at: b1 + 68, left: <LegCramp x={0} y={0} s={0.9} />, lt: '腿抽筋',
      right: <DogIcon x={0} y={-8} s={0.9} />, rt: '变成被狗咬', accent: P.coral},
  ];
  const lensOn = clamp01((f - b0 - 8) / 20);

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={1} name={sec.name} accent={P.sky} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <text x={340} y={166} textAnchor="middle" fill={P.cream} fontSize={30} fontFamily={FONT_HEAD} fontWeight={700}>
          睡眠实验室的观察</text>
        <text x={1520} y={166} textAnchor="middle" fill={P.cream} fontSize={30} fontFamily={FONT_HEAD} fontWeight={700}>
          梦里收到的版本</text>
        <TransLens x={960} y={490} h={730} on={lensOn} />
        {rows.map((r, i) => {
          const g = Math.max(0.001, spring({frame: f - r.at, fps, config: {damping: 12, stiffness: 140}}));
          const g2 = Math.max(0.001, spring({frame: f - r.at - 22, fps, config: {damping: 12, stiffness: 140}}));
          const dash = clamp01((f - r.at) / 24);
          return (
            <g key={i}>
              {/* 左：真实刺激卡 */}
              <g transform={`translate(${-(1 - g) * 60},0)`} opacity={g}>
                <rect x={140} y={r.y - 62} width={400} height={124} rx={20} fill={P.card} />
                <g transform={`translate(220,${r.y})`}>{r.left}</g>
                <text x={452} y={r.y + 10} textAnchor="end" fill={P.cream} fontSize={31}
                  fontFamily={FONT_HEAD} fontWeight={700}>{r.lt}</text>
              </g>
              {/* 箭头穿过透镜 */}
              <line x1={560} y1={r.y} x2={560 + 760 * dash} y2={r.y} stroke={r.accent}
                strokeWidth={6} strokeDasharray="18 14" strokeLinecap="round" />
              {dash > 0.9 && <path d={`M 1330 ${r.y} L 1310 ${r.y - 11} L 1310 ${r.y + 11} Z`} fill={r.accent} />}
              {/* 右：梦境版本卡 */}
              <g transform={`translate(${(1 - g2) * 60},0)`} opacity={g2}>
                <rect x={1350} y={r.y - 62} width={400} height={124} rx={20} fill={P.paper} />
                <g transform={`translate(1436,${r.y})`}>{r.right}</g>
                <text x={1706} y={r.y + 10} textAnchor="end" fill={P.ink} fontSize={31}
                  fontFamily={FONT_HEAD} fontWeight={700}>{r.rt}</text>
              </g>
            </g>
          );
        })}
        <g opacity={clamp01((f - b2 - 6) / 20)}>
          <rect x={620} y={856} width={680} height={6} rx={3} fill={P.violet} opacity={0.5} />
          <text x={960} y={846} textAnchor="middle" fill={P.cream} fontSize={30} fontFamily={FONT_HEAD}
            fontWeight={700}>信号进来了 · 只是被翻译得面目全非</text>
        </g>
      </svg>
      <Stamp text="外部刺激合并" x={140} y={845} at={b2 + 30} color={P.violetD} size={50} rot={-4} />
      <Chip text="做梦时外界输入没断" x={1440} y={836} at={b2 + 60} accent={P.violet} />
    </FadeIn>
  );
};

/* ============ S2 趴睡三份异常报告：剖面 + 报告卡 + 递给大脑的清单 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const cue = sec.cues[0];
  const cueDur = cue.t1 - cue.t0;
  const segAt = (k: number) => Math.round((cue.t0 + cueDur * k - sec.start) * fps);
  const b0 = 6;
  const r1 = segAt(0.13), r2 = segAt(0.46), r3 = segAt(0.66), rEnd = segAt(0.9);

  const zone = f >= r3 ? 3 : f >= r2 ? 2 : f >= r1 ? 1 : 0;
  const listK = Math.max(0.001, spring({frame: f - rEnd, fps, config: {damping: 12, stiffness: 130}}));

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={2} name={sec.name} accent={P.amber} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <ellipse cx={470} cy={856} rx={370} ry={26} fill={P.bgDeep} />
        <ProneSide x={470} y={620} s={1.5} zone={zone} />
        {/* 呼吸波形对照 */}
        <g opacity={clamp01((f - r1) / 18)}>
          <text x={150} y={330} fill={P.grey} fontSize={26} fontFamily={FONT_HEAD}>做梦期本就浅的呼吸</text>
          <BreathWave x={150} y={370} w={300} amp={8} col={P.sky} />
          <text x={480} y={330} fill={P.coral} fontSize={26} fontFamily={FONT_HEAD}>趴睡后被憋得更浅</text>
          <BreathWave x={480} y={370} w={300} amp={3.5} col={P.coral} />
        </g>

        {/* 三张报告卡 */}
        <ReportCard x={1020} y={170} no="①" at={r1} accent={P.coral}
          title="呼吸变浅" sub="胸口压住 · 胸廓张不开 · 憋成喘不上气"
          icon={<BreathWave x={-28} y={0} w={120} amp={6} col={P.coral} />} />
        <ReportCard x={1020} y={390} no="②" at={r2} accent={P.sky}
          title="触觉受压" sub="脸·胸·腹 神经密集区持续受压"
          icon={
            <g>
              <circle cx={-20} cy={0} r={30} fill={P.skin} stroke={P.skinD} strokeWidth={4} />
              <rect x={-58} y={18} width={76} height={20} rx={10} fill={P.pillow} stroke={P.pillowD} strokeWidth={4} />
              {[0, 1, 2].map((i) => <circle key={i} cx={-44 + i * 22} cy={34} r={4.5} fill={P.coral} />)}
            </g>} />
        <ReportCard x={1020} y={610} no="③" at={r3} accent={P.violet}
          title="体位受限" sub="翻不了身 · 脖子扭一整夜 ≈90°"
          icon={
            <g>
              <circle cx={-38} cy={6} r={24} fill={P.skin} stroke={P.skinD} strokeWidth={4} />
              <rect x={-14} y={-6} width={66} height={26} rx={13} fill={P.blanket} stroke={P.blanketD} strokeWidth={4} />
              <path d="M 8 -34 A 40 40 0 0 0 -24 -14" fill="none" stroke={P.violet} strokeWidth={5}
                strokeDasharray="7 6" />
            </g>} />

        {/* 汇总清单 → 大脑（顶部横条，避开三行字幕区） */}
        <g transform={`translate(1020,${62 - (1 - listK) * 20})`} opacity={listK}>
          <rect width={780} height={64} rx={16} fill={P.paper} />
          {[0, 1, 2].map((i) => (
            <g key={i} opacity={clamp01((f - rEnd - 8 - i * 10) / 12)}>
              <circle cx={34 + i * 40} cy={32} r={13} fill={P.mint} />
              <path d={`M ${27 + i * 40} 32 L ${32 + i * 40} 37 L ${42 + i * 40} 25`}
                stroke={P.ink} strokeWidth={5} fill="none" strokeLinecap="round" />
            </g>
          ))}
          <text x={180} y={42} fill={P.ink} fontSize={29} fontFamily={FONT_HEAD} fontWeight={800}>
            递给大脑：『你被压制着』</text>
          <BrainDot x={730} y={32} s={0.55} op={listK} />
        </g>
      </svg>
      <Stamp text="三份异常报告" x={120} y={150} at={b0 + 30} color={P.amberD} size={50} rot={-4} />
      <Chip text="『动不了』正对上被绑剧本" x={100} y={245} at={r3 + 40} accent={P.violet} />
    </FadeIn>
  );
};

/* ============ S3 香港问卷：670份自述 + 趴睡组高出一截 + 相关≠因果 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b0 = beat(0), b1 = beat(1);

  const paperK = Math.max(0.001, spring({frame: f - b0 - 8, fps, config: {damping: 13, stiffness: 130}}));
  const cats = [
    {label: '窒息', prone: 0.78, other: 0.32},
    {label: '被捆绑', prone: 0.7, other: 0.26},
    {label: '动不了', prone: 0.74, other: 0.3},
    {label: '羞羞梦', prone: 0.62, other: 0.28},
  ];
  const magK = clamp01((f - b1 - 6) / 20);

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={3} name={sec.name} accent={P.mint} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：问卷纸 */}
        <g transform={`translate(300,180) rotate(-4) scale(${paperK})`}>
          <rect x={-30} y={-26} width={470} height={620} rx={18} fill={P.paperSh} />
          <rect width={440} height={600} rx={18} fill={P.paper} />
          <rect width={440} height={74} rx={18} fill={P.mint} />
          <text x={220} y={50} textAnchor="middle" fill={P.ink} fontSize={32} fontFamily={FONT_HEAD}
            fontWeight={800}>睡姿 × 梦境 问卷</text>
          {['你平时什么睡姿？', '梦到窒息/被绑/动不了？', '梦的内容还记得吗？'].map((q, i) => (
            <g key={i}>
              <rect x={30} y={106 + i * 120} width={380} height={96} rx={12} fill={P.paperSh} opacity={0.65} />
              <text x={52} y={142 + i * 120} fill={P.ink} fontSize={26} fontFamily={FONT_HEAD}>{q}</text>
              <rect x={52} y={156 + i * 120} width={26} height={26} rx={6} fill="none" stroke={P.ink} strokeWidth={4} />
              {i === 0 && (
                <path d="M 58 168 L 64 176 L 74 160" stroke={P.coralD} strokeWidth={6}
                  fill="none" strokeLinecap="round" />
              )}
              <text x={92} y={180 + i * 120} fill={P.grey} fontSize={23} fontFamily={FONT_HEAD}>凭印象填…</text>
            </g>
          ))}
          <g transform="translate(220,548)">
            <rect x={-130} y={-46} width={260} height={74} rx={14} fill={P.coral} />
            <text x={0} y={6} textAnchor="middle" fill="#fff" fontSize={42} fontFamily={FONT_HEAD} fontWeight={900}>
              670 份</text>
          </g>
          {/* 放大镜照出小字 */}
          <g opacity={magK}>
            <circle cx={330} cy={200} r={72} fill="rgba(126,203,242,0.18)" stroke={P.sky} strokeWidth={7} />
            <line x1={382} y1={252} x2={432} y2={302} stroke={P.sky} strokeWidth={14} strokeLinecap="round" />
            <text x={330} y={196} textAnchor="middle" fill={P.coralD} fontSize={26}
              fontFamily={FONT_HEAD} fontWeight={800}>全是自报</text>
            <text x={330} y={228} textAnchor="middle" fill={P.ink} fontSize={21} fontFamily={FONT_HEAD}>没监控睡姿</text>
          </g>
        </g>
        <text x={490} y={836} textAnchor="middle" fill={P.grey} fontSize={26} fontFamily={FONT_HEAD}>
          余启程 · 2012 《Dreaming》期刊</text>

        {/* 右：对照柱状图 */}
        <rect x={960} y={180} width={840} height={600} rx={24} fill={P.card} />
        <text x={1380} y={238} textAnchor="middle" fill={P.cream} fontSize={32} fontFamily={FONT_HEAD}
          fontWeight={800}>趴睡组 vs 其他睡姿 · 自报比例</text>
        {/* 图例 */}
        <rect x={1630} y={258} width={22} height={22} rx={5} fill={P.coral} />
        <text x={1660} y={276} fill={P.cream} fontSize={23} fontFamily={FONT_HEAD}>趴睡</text>
        <rect x={1730} y={258} width={22} height={22} rx={5} fill={P.grey} />
        <text x={1760} y={276} fill={P.cream} fontSize={23} fontFamily={FONT_HEAD}>其他</text>
        <line x1={1030} y1={655} x2={1740} y2={655} stroke={P.cream} strokeWidth={4} />
        {cats.map((c, i) => {
          const bx = 1090 + i * 168;
          const g = clamp01((f - b0 - 30 - i * 22) / 26);
          return (
            <g key={i}>
              <rect x={bx} y={655 - 315 * c.prone * g} width={52} height={315 * c.prone * g}
                rx={10} fill={P.coral} />
              <rect x={bx + 58} y={655 - 315 * c.other * g} width={52} height={315 * c.other * g}
                rx={10} fill={P.grey} />
              <text x={bx + 55} y={697} textAnchor="middle" fill={P.cream} fontSize={28}
                fontFamily={FONT_HEAD}>{c.label}</text>
            </g>
          );
        })}
        <g opacity={clamp01((f - b0 - 120) / 20)}>
          <path d="M 1090 350 L 1700 330" stroke={P.amber} strokeWidth={5} strokeDasharray="14 10" />
          <text x={1390} y={312} textAnchor="middle" fill={P.amber} fontSize={27} fontFamily={FONT_HEAD}
            fontWeight={700}>趴睡组显著更高</text>
        </g>
        <g opacity={magK}>
          <rect x={990} y={718} width={780} height={52} rx={14} fill={P.bgDeep} />
          <text x={1380} y={753} textAnchor="middle" fill={P.cream} fontSize={26} fontFamily={FONT_HEAD}>
            隔着好几层记忆与报告偏差</text>
        </g>
      </svg>
      <Chip text="窒息 / 被绑 / 动不了 比例更高" x={740} y={115} at={b0 + 110} accent={P.coral} />
      <Stamp text="相关 ≠ 因果" x={130} y={140} at={b1 + 40} color={P.coralD} size={52} rot={-5} />
      <Chip text="是线索 · 不是判决" x={1470} y={115} at={b1 + 80} accent={P.mint} />
    </FadeIn>
  );
};

/* ============ S4 编剧逻辑：翻译规则卡 + 高频剧目榜 + 三十秒排查 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b0 = beat(0), b1 = beat(1), b3 = beat(3);

  const rules = [
    {at: b0 + 20, from: '胸口闷', to: '有东西压着你', accent: P.coral},
    {at: b0 + 75, from: '动不了', to: '直接绑起来', accent: P.violet},
    {at: b0 + 130, from: '呼吸浅', to: '安排窒息', accent: P.sky},
  ];
  const ranks = [
    {label: '被追', node: <IconChase x={0} y={0} s={0.62} />},
    {label: '被压', node: <IconPress x={0} y={-4} s={0.62} />},
    {label: '被困', node: <IconTrap x={0} y={0} s={0.6} />},
  ];
  const checkK = Math.max(0.001, spring({frame: f - b3 - 6, fps, config: {damping: 11, stiffness: 140}}));

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={4} name={sec.name} accent={P.violet} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：编剧的翻译桌 */}
        <rect x={130} y={170} width={790} height={560} rx={24} fill={P.card} />
        <BrainDot x={240} y={248} s={0.8} />
        <text x={330} y={238} fill={P.cream} fontSize={34} fontFamily={FONT_HEAD} fontWeight={800}>
          大脑编剧部</text>
        <text x={330} y={278} fill={P.grey} fontSize={26} fontFamily={FONT_HEAD}>
          策略是解释 · 不是记录</text>
        {rules.map((r, i) => {
          const g = Math.max(0.001, spring({frame: f - r.at, fps, config: {damping: 12, stiffness: 150}}));
          const y = 330 + i * 112;
          return (
            <g key={i} transform={`translate(${-(1 - g) * 50},0)`} opacity={g}>
              <rect x={170} y={y} width={230} height={88} rx={16} fill={P.bgLite} />
              <text x={285} y={y + 56} textAnchor="middle" fill={P.cream} fontSize={30}
                fontFamily={FONT_HEAD} fontWeight={700}>{r.from}</text>
              <path d={`M 424 ${y + 44} L 486 ${y + 44}`} stroke={r.accent} strokeWidth={7}
                strokeLinecap="round" strokeDasharray="12 10" />
              <path d={`M 486 ${y + 44} L 468 ${y + 33} L 468 ${y + 55} Z`} fill={r.accent} />
              <rect x={506} y={y} width={380} height={88} rx={16} fill={r.accent} />
              <text x={696} y={y + 56} textAnchor="middle" fill="#fff" fontSize={31}
                fontFamily={FONT_HEAD} fontWeight={800}>{r.to}</text>
            </g>
          );
        })}
        <g opacity={clamp01((f - b0 - 180) / 20)}>
          <rect x={170} y={700} width={716} height={6} rx={3} fill={P.coral} opacity={0.5} />
          <text x={528} y={688} textAnchor="middle" fill={P.coral} fontSize={27} fontFamily={FONT_HEAD}
            fontWeight={700}>梦里没有『这不对劲』的自检 · 多离谱都照收</text>
        </g>

        {/* 右上：高频剧目榜 */}
        <g opacity={clamp01((f - b1) / 20)}>
          <rect x={990} y={170} width={800} height={330} rx={24} fill={P.paper} />
          <rect x={990} y={170} width={800} height={70} rx={24} fill={P.ink} />
          <text x={1390} y={216} textAnchor="middle" fill={P.cream} fontSize={31}
            fontFamily={FONT_HEAD} fontWeight={800}>全人类梦境高频榜 · 本来就常演</text>
          {ranks.map((r, i) => {
            const g = Math.max(0.001, spring({frame: f - b1 - 14 - i * 16, fps, config: {damping: 11, stiffness: 150}}));
            const x = 1170 + i * 240;
            return (
              <g key={i} transform={`translate(${x},326) scale(${g})`} opacity={g}>
                <circle cx={-74} cy={92} r={26} fill={P.amber} stroke={P.amberD} strokeWidth={4} />
                <text x={-74} y={103} textAnchor="middle" fill={P.ink} fontSize={28}
                  fontFamily={FONT_HEAD} fontWeight={900}>{i + 1}</text>
                {r.node}
                <text x={0} y={150} textAnchor="middle" fill={P.ink} fontSize={31}
                  fontFamily={FONT_HEAD} fontWeight={800}>{r.label}</text>
              </g>
            );
          })}
        </g>
        <g opacity={clamp01((f - b1 - 60) / 20)}>
          <rect x={990} y={528} width={800} height={64} rx={16} fill={P.cardLite} />
          <text x={1390} y={570} textAnchor="middle" fill={P.cream} fontSize={28} fontFamily={FONT_HEAD}>
            趴睡 = 给排演喂了最浓的一组信号</text>
        </g>

        {/* 右下：三十秒排查 */}
        <g transform={`translate(990,640)`} opacity={checkK}>
          <rect width={800} height={212} rx={24} fill={P.card} />
          <g transform="translate(120,106)">
            <circle r={62} fill={P.bgLite} stroke={P.mint} strokeWidth={6} />
            <text y={-8} textAnchor="middle" fill={P.mint} fontSize={38} fontFamily={FONT_HEAD} fontWeight={900}>30</text>
            <text y={30} textAnchor="middle" fill={P.mint} fontSize={26} fontFamily={FONT_HEAD} fontWeight={800}>秒</text>
          </g>
          <g transform="translate(262,106)">
            <rect x={-60} y={-18} width={150} height={40} rx={18} fill={P.blanket} stroke={P.blanketD} strokeWidth={4} />
            <circle cx={-72} cy={-8} r={26} fill={P.skin} stroke={P.skinD} strokeWidth={4} />
            <circle cx={30} cy={-40} r={46} fill="rgba(126,203,242,0.15)" stroke={P.sky} strokeWidth={6} />
            <line x1={62} y1={-8} x2={96} y2={26} stroke={P.sky} strokeWidth={11} strokeLinecap="round" />
          </g>
          <text x={420} y={92} fill={P.cream} fontSize={32} fontFamily={FONT_HEAD} fontWeight={800}>
            夜夜怪梦？先看一眼睡姿</text>
          <text x={420} y={140} fill={P.grey} fontSize={27} fontFamily={FONT_HEAD}>
            成本为零 · 先查最容易改的</text>
        </g>
      </svg>
      <Chip text="感觉在前 · 剧情在后" x={120} y={756} at={b0 + 200} accent={P.coral} />
      <Stamp text="解释 不是 记录" x={1480} y={102} at={b0 + 90} color={P.violetD} size={46} rot={4} />
      <Chip text="趴睡只是众多变量之一" x={1000} y={98} at={b1 + 100} accent={P.amber} />
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
        <ellipse cx={960} cy={1010} rx={1120} ry={110} fill={P.bg} />
        <g transform="translate(960,880) scale(0.9)">
          <ProneSide x={0} y={0} s={1.15} zone={0} />
        </g>
        <MoonIcon x={560} y={620} s={0.85} hole={P.bgDeep} />
        <g transform={`translate(1330,760) scale(${pop})`}>
          <circle r={70} fill={P.paper} />
          <circle r={70} fill="none" stroke={P.coral} strokeWidth={6} strokeDasharray="14 12" />
          <text x={0} y={12} textAnchor="middle" fill={P.ink} fontSize={44} fontFamily={FONT_HEAD}
            fontWeight={900}>?</text>
        </g>
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -300}}>
        <div style={{fontSize: 56, fontWeight: 800, color: C.text}}>有理由怀疑 · 没资格断定</div>
        <div style={{fontSize: 38, color: C.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E226 = ({data = tl}: {data?: any}) => {
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
