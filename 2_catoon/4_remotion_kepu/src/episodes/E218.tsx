import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01, rand} from '../lib/style';
import {Backdrop, Ripple, Stamp, SceneTag, Chip, Subtitles, ProgressBar, FadeIn} from '../lib/atoms';
import tl from '../timeline_218.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：泥炭棕底 × 古莲深褐 × 莲花粉 × 莲叶绿（纯色平涂） ============ */
const P = {
  bg: '#3a2b1d',       // 主底：泥炭棕
  bgDeep: '#2b1f13',
  bgLite: '#46351f',
  card: '#4d3a24',
  cream: '#f4ead2',
  ink: '#241708',
  seed: '#6b4326',     // 莲子壳
  seedD: '#452815',
  seedL: '#96683d',
  embryo: '#f0dcae',   // 胚乳
  embryoD: '#c9a86a',
  petal: '#f7a8bb',    // 莲花
  petalD: '#d9718a',
  petalL: '#ffd3dd',
  leaf: '#7fc98f',     // 莲叶绿
  leafD: '#4a9e63',
  peat: '#59402a',     // 泥炭层
  peatD: '#3d2b1a',
  peatL: '#7a5a3c',
  amber: '#ffc94d',
  amberD: '#c98f1f',
  water: '#57b6c6',    // 塘水
  waterD: '#367e8a',
  sky: '#8fd3ff',
  red: '#e0685a',
  grey: '#a89880',
  steel: '#b8c4cc',
  steelD: '#7e8b96',
};

/* ---------- 嫩芽：从莲子顶端钻出 ---------- */
const Sprout = ({x, y, s = 1, rot = 0, op = 1}: {x: number; y: number; s?: number; rot?: number; op?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`} opacity={op}>
    <path d="M 0 8 Q 6 -32 -4 -66" fill="none" stroke={P.leafD} strokeWidth={9} strokeLinecap="round" />
    <path d="M -4 -66 Q -46 -80 -58 -50 Q -26 -44 -4 -66 Z" fill={P.leaf} stroke={P.leafD} strokeWidth={4} strokeLinejoin="round" />
    <path d="M -4 -66 Q 32 -90 50 -62 Q 20 -48 -4 -66 Z" fill={P.leaf} stroke={P.leafD} strokeWidth={4} strokeLinejoin="round" />
  </g>
);

/* ---------- 莲子：深褐硬壳，可剖面 / 破口 / 发芽 / 表情 ---------- */
const LotusSeed = ({x, y, s = 1, rot = 0, op = 1, cut = 0, chipped = 0, germ = 0, face}: {
  x: number; y: number; s?: number; rot?: number; op?: number; cut?: number; chipped?: number; germ?: number;
  face?: 'sleep' | 'wake';
}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`} opacity={op}>
    {germ > 0 && <Sprout x={0} y={-64} s={germ} />}
    <ellipse rx={58} ry={76} fill={P.seed} stroke={P.seedD} strokeWidth={7} />
    <ellipse cx={-18} cy={-26} rx={20} ry={32} fill={P.seedL} opacity={0.55} />
    <ellipse cx={0} cy={-69} rx={9} ry={5} fill={P.seedD} />
    {cut > 0 && (
      <g opacity={cut}>
        <ellipse cx={3} cy={4} rx={43} ry={60} fill={P.cream} stroke={P.seedD} strokeWidth={5} />
        <ellipse cx={-10} cy={9} rx={19} ry={43} fill={P.embryo} stroke={P.embryoD} strokeWidth={4} />
        <ellipse cx={18} cy={9} rx={17} ry={43} fill={P.embryo} stroke={P.embryoD} strokeWidth={4} />
        <path d="M 4 -36 Q -12 -50 -4 -62 Q 6 -52 4 -36" fill={P.leaf} />
        <path d="M 7 -36 Q 24 -48 18 -62 Q 6 -52 7 -36" fill={P.leafD} />
        {face === 'sleep' && (
          <g stroke={P.ink} strokeWidth={4} fill="none" strokeLinecap="round">
            <path d="M -22 -4 Q -14 -12 -6 -4" />
            <path d="M 8 -4 Q 16 -12 24 -4" />
            <path d="M -4 24 Q 4 30 12 24" />
          </g>
        )}
        {face === 'wake' && (
          <g>
            <circle cx={-14} cy={-6} r={6} fill={P.ink} />
            <circle cx={17} cy={-6} r={6} fill={P.ink} />
            <path d="M -6 24 Q 4 32 14 22" stroke={P.ink} strokeWidth={4} fill="none" strokeLinecap="round" />
          </g>
        )}
      </g>
    )}
    {chipped > 0 && (
      <g opacity={chipped} transform="translate(20,-66) rotate(20)">
        <path d="M -16 -10 L 14 -14 L 18 8 L -10 14 Z" fill={P.cream} stroke={P.seedD} strokeWidth={4} />
        <path d="M -16 -10 L -30 -18 M -8 14 L -20 26" stroke={P.seedD} strokeWidth={4} strokeLinecap="round" />
      </g>
    )}
  </g>
);

/* ---------- 莲花 ---------- */
const PetalShape = ({ang, len, wid, col}: {ang: number; len: number; wid: number; col: string}) => (
  <g transform={`rotate(${ang})`}>
    <path d={`M 0 6 Q ${-wid} ${-len * 0.42} 0 ${-len} Q ${wid} ${-len * 0.42} 0 6 Z`}
      fill={col} stroke={P.petalD} strokeWidth={4} />
  </g>
);

const LotusFlower = ({x, y, s = 1, rot = 0, op = 1, bloom = 1}: {
  x: number; y: number; s?: number; rot?: number; op?: number; bloom?: number;
}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s * Math.max(0.001, bloom)})`} opacity={op}>
    {[-64, -32, 0, 32, 64].map((a) => <PetalShape key={'b' + a} ang={a} len={96} wid={30} col={P.petalD} />)}
    {[-48, -16, 16, 48].map((a) => <PetalShape key={'f' + a} ang={a} len={80} wid={30} col={P.petal} />)}
    <PetalShape ang={0} len={86} wid={32} col={P.petalL} />
    <path d="M -24 2 L -18 -26 Q 0 -36 18 -26 L 24 2 Q 0 12 -24 2 Z" fill={P.amber} stroke={P.amberD} strokeWidth={4} />
    <circle cx={-10} cy={-14} r={4} fill={P.amberD} />
    <circle cx={0} cy={-18} r={4} fill={P.amberD} />
    <circle cx={10} cy={-14} r={4} fill={P.amberD} />
  </g>
);

/* ---------- 荷叶（带缺口圆盘） ---------- */
const LotusPad = ({x, y, s = 1, rot = 0, op = 1}: {x: number; y: number; s?: number; rot?: number; op?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`} opacity={op}>
    <path d="M 10 0 L 66 -52 A 84 84 0 1 1 66 52 Z" fill={P.leaf} stroke={P.leafD} strokeWidth={6} strokeLinejoin="round" />
    <g stroke={P.leafD} strokeWidth={4} strokeLinecap="round" opacity={0.7}>
      {[55, 105, 155, 205, 255, 305].map((a) => {
        const r = a * Math.PI / 180;
        return <line key={a} x1={0} y1={0} x2={Math.cos(r) * 74} y2={Math.sin(r) * 74} />;
      })}
    </g>
    <circle r={9} fill={P.leafD} />
  </g>
);

/* ---------- 水滴 ---------- */
const WaterDrop = ({x, y, s = 1, rot = 0, op = 1}: {x: number; y: number; s?: number; rot?: number; op?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`} opacity={op}>
    <path d="M 0 -26 Q 20 4 0 26 Q -20 4 0 -26 Z" fill={P.water} stroke={P.waterD} strokeWidth={4} />
    <circle cx={-5} cy={4} r={5} fill="#bfecf3" />
  </g>
);

/* ---------- O₂ 分子胶囊 ---------- */
const O2 = ({x, y, s = 1, op = 1}: {x: number; y: number; s?: number; op?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
    <rect x={-32} y={-17} width={64} height={34} rx={17} fill={P.sky} />
    <text x={0} y={9} textAnchor="middle" fontSize={24} fill={P.ink} fontFamily={FONT_HEAD} fontWeight={800}>O₂</text>
  </g>
);

/* ---------- 细菌真菌（刺球） ---------- */
const GermBall = ({x, y, s = 1, rot = 0, op = 1}: {x: number; y: number; s?: number; rot?: number; op?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`} opacity={op}>
    {[0, 45, 90, 135, 180, 225, 270, 315].map((a) => {
      const r = a * Math.PI / 180;
      return <line key={a} x1={Math.cos(r) * 13} y1={Math.sin(r) * 13} x2={Math.cos(r) * 26} y2={Math.sin(r) * 26}
        stroke={P.red} strokeWidth={6} strokeLinecap="round" />;
    })}
    <circle r={15} fill={P.red} />
    <circle cx={-4} cy={-4} r={4} fill="#f2a091" />
  </g>
);

/* ---------- 铲子 ---------- */
const Shovel = ({x, y, s = 1, rot = 0, op = 1}: {x: number; y: number; s?: number; rot?: number; op?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`} opacity={op}>
    <rect x={-9} y={-270} width={18} height={210} rx={9} fill="#9a6a3c" />
    <rect x={-20} y={-292} width={40} height={28} rx={12} fill="#7a5226" />
    <path d="M -48 -60 L 48 -60 L 36 38 Q 0 66 -36 38 Z" fill={P.steel} stroke={P.steelD} strokeWidth={6} />
  </g>
);

/* ---------- 砂纸 ---------- */
const Sandpaper = ({x, y, s = 1, rot = 0, op = 1}: {x: number; y: number; s?: number; rot?: number; op?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`} opacity={op}>
    <rect x={-95} y={-32} width={190} height={64} rx={10} fill="#a8763e" stroke="#7a5226" strokeWidth={6} />
    {Array.from({length: 18}).map((_, i) => (
      <circle key={i} cx={-78 + (i % 6) * 31} cy={-16 + Math.floor(i / 6) * 16} r={3.4} fill="#6e4a20" opacity={0.8} />
    ))}
  </g>
);

/* ---------- 代谢率仪表盘（指针贴 0 抖动） ---------- */
const Gauge = ({x, y, s = 1, op = 1, k}: {x: number; y: number; s?: number; op?: number; k: number}) => {
  const th = (180 - k * 180) * Math.PI / 180;
  const pt = (deg: number, r: number) => [Math.cos(deg * Math.PI / 180) * r, -Math.sin(deg * Math.PI / 180) * r];
  const [nx, ny] = pt(180 - k * 180, 74);
  return (
    <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
      <path d="M -104 0 A 104 104 0 0 1 104 0" fill="none" stroke={P.card} strokeWidth={18} strokeLinecap="round" />
      <path d={`M ${pt(180, 104)[0]} ${pt(180, 104)[1]} A 104 104 0 0 1 ${pt(146, 104)[0]} ${pt(146, 104)[1]}`}
        fill="none" stroke={P.leaf} strokeWidth={18} strokeLinecap="round" />
      <path d={`M ${pt(26, 104)[0]} ${pt(26, 104)[1]} A 104 104 0 0 1 ${pt(0, 104)[0]} ${pt(0, 104)[1]}`}
        fill="none" stroke={P.red} strokeWidth={18} strokeLinecap="round" opacity={0.85} />
      {[150, 120, 90, 60, 30].map((d) => {
        const [x1, y1] = pt(d, 86);
        const [x2, y2] = pt(d, 98);
        return <line key={d} x1={x1} y1={y1} x2={x2} y2={y2} stroke={P.cream} strokeWidth={4} opacity={0.6} />;
      })}
      <line x1={0} y1={0} x2={nx} y2={ny} stroke={P.cream} strokeWidth={7} strokeLinecap="round" />
      <circle r={11} fill={P.cream} stroke={P.card} strokeWidth={4} />
      <text x={-118} y={10} textAnchor="middle" fontSize={24} fill={P.leaf} fontFamily={FONT_HEAD} fontWeight={800}>0</text>
      <text x={118} y={10} textAnchor="middle" fontSize={20} fill={P.red} fontFamily={FONT_HEAD} fontWeight={800}>MAX</text>
    </g>
  );
};

/* ---------- 待机电池 ---------- */
const Battery = ({x, y, s = 1, op = 1, k = 0.09}: {x: number; y: number; s?: number; op?: number; k?: number}) => {
  const f = useCurrentFrame();
  const blink = 0.55 + 0.45 * Math.sin(f / 8);
  return (
    <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
      <rect x={-60} y={-28} width={104} height={56} rx={10} fill="none" stroke={P.cream} strokeWidth={6} />
      <rect x={46} y={-12} width={14} height={24} rx={5} fill={P.cream} />
      <rect x={-52} y={-20} width={88 * k} height={40} rx={5} fill={P.leaf} opacity={blink} />
      <text x={74} y={10} fontSize={28} fill={P.cream} fontFamily={FONT_HEAD} fontWeight={800}>待机</text>
    </g>
  );
};

/* ---------- 蛋白维修组（圆球 + 小扳手） ---------- */
const RepairBot = ({x, y, s = 1, rot = 0, op = 1}: {x: number; y: number; s?: number; rot?: number; op?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`} opacity={op}>
    <line x1={0} y1={-21} x2={0} y2={-32} stroke="#5d84c9" strokeWidth={5} strokeLinecap="round" />
    <circle cx={0} cy={-37} r={5} fill={P.amber} />
    <g transform="translate(24,-18) rotate(40)">
      <line x1={0} y1={0} x2={0} y2={24} stroke={P.steelD} strokeWidth={6} strokeLinecap="round" />
      <path d="M -8 -6 A 10 10 0 1 1 8 -6" fill="none" stroke={P.steelD} strokeWidth={6} strokeLinecap="round" />
    </g>
    <circle r={21} fill={P.sky} stroke="#5d84c9" strokeWidth={5} />
    <circle cx={-7} cy={-3} r={4} fill={P.ink} />
    <circle cx={8} cy={-3} r={4} fill={P.ink} />
    <path d="M -8 8 Q 0 13 8 8" stroke={P.ink} strokeWidth={4} fill="none" strokeLinecap="round" />
  </g>
);

/* ---------- DNA 链（裂纹可修补） ---------- */
const DnaStrand = ({x, y, s = 1, op = 1, crack = 0, bgCol = P.bg}: {
  x: number; y: number; s?: number; op?: number; crack?: number; bgCol?: string;
}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
    <path d="M -170 0 Q -130 -30 -85 0 Q -40 30 0 0 Q 40 -30 85 0 Q 130 30 170 0" fill="none" stroke={P.amber} strokeWidth={9} strokeLinecap="round" />
    <path d="M -170 0 Q -130 30 -85 0 Q -40 -30 0 0 Q 40 30 85 0 Q 130 -30 170 0" fill="none" stroke={P.amberD} strokeWidth={9} strokeLinecap="round" />
    {[-127, -42, 42, 127].map((rx) => (
      <line key={rx} x1={rx} y1={-25} x2={rx} y2={25} stroke={P.amberD} strokeWidth={5} opacity={0.7} />
    ))}
    {crack > 0 && (
      <g opacity={crack}>
        <rect x={16} y={-42} width={52} height={84} fill={bgCol} />
        <path d="M 18 -34 L 30 -20 L 20 -6" fill="none" stroke={P.amber} strokeWidth={7} strokeLinecap="round" />
        <path d="M 66 34 L 56 20 L 68 8" fill="none" stroke={P.amberD} strokeWidth={7} strokeLinecap="round" />
      </g>
    )}
    {crack < 1 && (
      <g opacity={1 - crack}>
        <rect x={22} y={-30} width={40} height={60} rx={10} fill={P.cream} stroke={P.leaf} strokeWidth={5} />
        <line x1={30} y1={-14} x2={54} y2={-14} stroke={P.leaf} strokeWidth={4} />
        <line x1={30} y1={0} x2={54} y2={0} stroke={P.leaf} strokeWidth={4} />
        <line x1={30} y1={14} x2={54} y2={14} stroke={P.leaf} strokeWidth={4} />
      </g>
    )}
  </g>
);

/* ---------- 日历卡 ---------- */
const Calendar = ({x, y, s = 1, op = 1, label = ''}: {x: number; y: number; s?: number; op?: number; label?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
    <rect x={-78} y={-66} width={156} height={140} rx={14} fill={P.cream} stroke={P.card} strokeWidth={5} />
    <rect x={-78} y={-66} width={156} height={34} rx={14} fill={P.red} />
    <rect x={-78} y={-46} width={156} height={14} fill={P.red} />
    <line x1={-40} y1={-78} x2={-40} y2={-58} stroke={P.card} strokeWidth={7} strokeLinecap="round" />
    <line x1={40} y1={-78} x2={40} y2={-58} stroke={P.card} strokeWidth={7} strokeLinecap="round" />
    <text x={0} y={36} textAnchor="middle" fontSize={42} fontFamily={FONT_HEAD} fontWeight={800} fill={P.ink}>{label}</text>
  </g>
);

/* ---------- 石头 / 雪花 / 鸟（自然破壳三件套） ---------- */
const Rock = ({x, y, s = 1, rot = 0, op = 1}: {x: number; y: number; s?: number; rot?: number; op?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`} opacity={op}>
    <polygon points="-44,18 -30,-18 -2,-30 30,-14 46,10 20,26 -20,28" fill={P.grey} stroke="#7a6c58" strokeWidth={5} strokeLinejoin="round" />
    <line x1={-18} y1={-10} x2={2} y2={2} stroke="#7a6a52" strokeWidth={4} opacity={0.7} />
  </g>
);

const Snowflake = ({x, y, s = 1, rot = 0, op = 1}: {x: number; y: number; s?: number; rot?: number; op?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`} opacity={op} stroke={P.sky} strokeWidth={7} strokeLinecap="round">
    {[0, 60, 120].map((a) => (
      <g key={a} transform={`rotate(${a})`}>
        <line x1={-34} y1={0} x2={34} y2={0} />
        <line x1={26} y1={0} x2={34} y2={-9} />
        <line x1={26} y1={0} x2={34} y2={9} />
        <line x1={-26} y1={0} x2={-34} y2={-9} />
        <line x1={-26} y1={0} x2={-34} y2={9} />
      </g>
    ))}
  </g>
);

const Bird = ({x, y, s = 1, rot = 0, op = 1}: {x: number; y: number; s?: number; rot?: number; op?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`} opacity={op}>
    <path d="M -34 -6 L -54 -16 L -48 4 Z" fill="#9a8a70" />
    <ellipse rx={36} ry={24} fill="#9a8a70" />
    <circle cx={32} cy={-18} r={17} fill="#9a8a70" />
    <polygon points="46,-22 70,-15 46,-9" fill={P.amber} />
    <circle cx={36} cy={-22} r={4.5} fill={P.ink} />
    <path d="M -12 -4 Q 4 8 22 -2" stroke="#7a6a52" strokeWidth={5} fill="none" strokeLinecap="round" />
    <line x1={-8} y1={22} x2={-8} y2={34} stroke="#7a6a52" strokeWidth={5} strokeLinecap="round" />
    <line x1={8} y1={22} x2={8} y2={34} stroke="#7a6a52" strokeWidth={5} strokeLinecap="round" />
  </g>
);

/* ---------- 椰枣树 ---------- */
const Palm = ({x, y, s = 1, op = 1}: {x: number; y: number; s?: number; op?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
    <path d="M -8 44 Q -14 4 -4 -32" stroke="#8a5a33" strokeWidth={13} fill="none" strokeLinecap="round" />
    {[-64, -32, 0, 32, 64].map((a) => (
      <path key={a} d="M 0 -32 Q 30 -64 68 -46 Q 30 -30 0 -32 Z" fill={P.leafD}
        transform={`rotate(${a})`} />
    ))}
    {[-1, 1].map((sd) => (
      <g key={sd} transform={`translate(${sd * 16},-20)`}>
        <circle r={6} fill={P.amber} />
        <circle cx={sd * 9} cy={8} r={6} fill={P.amberD} />
        <circle cx={-sd * 5} cy={12} r={6} fill={P.amber} />
      </g>
    ))}
  </g>
);

/* ---------- 蝇子草（小白花） ---------- */
const Campion = ({x, y, s = 1, op = 1}: {x: number; y: number; s?: number; op?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
    <line x1={0} y1={44} x2={0} y2={-8} stroke={P.leafD} strokeWidth={7} strokeLinecap="round" />
    {[0, 72, 144, 216, 288].map((a) => (
      <ellipse key={a} cx={0} cy={-26} rx={9} ry={16} fill="#f2ede2" stroke="#c9bfa8" strokeWidth={3}
        transform={`rotate(${a})`} />
    ))}
    <circle cx={0} cy={-14} r={7} fill={P.amber} />
  </g>
);

/* ---------- 宋朝来信 ---------- */
const Envelope = ({x, y, s = 1, rot = 0, op = 1}: {x: number; y: number; s?: number; rot?: number; op?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`} opacity={op}>
    <rect x={-84} y={-54} width={168} height={108} rx={10} fill={P.cream} stroke={P.card} strokeWidth={5} />
    <path d="M -82 -50 L 0 16 L 82 -50" fill="none" stroke={P.card} strokeWidth={5} />
    <circle cx={56} cy={-30} r={17} fill={P.red} />
    <text x={56} y={-23} textAnchor="middle" fontSize={20} fill="#fff" fontFamily={FONT_HEAD} fontWeight={800}>宋</text>
  </g>
);

/* ============ 标题卡 ============ */
const TitleCard = ({data}: {data: any}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 13, mass: 0.7}});
  const o = interpolate(f, [0, 8], [0, 1]);
  const p0 = Math.max(0.001, spring({frame: f - 4, fps, config: {damping: 10, stiffness: 150}}));
  const p1 = Math.max(0.001, spring({frame: f - 12, fps, config: {damping: 10, stiffness: 150}}));
  const bob = Math.sin(f / 16) * 8;
  return (
    <AbsoluteFill>
      <Backdrop top={P.bg} bottom={P.bg} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <circle cx={960} cy={352} r={218} fill={P.cream} />
        <circle cx={960} cy={352} r={196} fill={P.bgLite} />
        <g transform={`translate(960,${340 + bob}) scale(${p0})`}>
          <LotusSeed x={0} y={0} s={1.35} cut={1} germ={1} />
        </g>
        <g transform={`translate(762,${436 - bob * 0.6}) scale(${p1})`}>
          <LotusPad x={0} y={0} s={0.78} rot={-12} />
        </g>
        <g transform={`translate(1162,${292 + bob * 0.8}) scale(${p1})`}>
          <LotusFlower x={0} y={0} s={0.7} />
        </g>
        <g transform={`translate(1150,${470 - bob * 0.5}) scale(${p1})`}>
          <WaterDrop x={0} y={0} s={0.7} rot={-14} />
        </g>
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 270}}>
          <div style={{color: P.leaf, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 26}}>
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

/* ============ S0 泥炭层出土：挖到古莲子 → 碳测定 → 磨壳发芽开花 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b0 = beat(0), b1 = beat(1), b2 = beat(2);

  const digT = b0 + 35;
  const shovelIn = clamp01((f - digT) / 18);
  const shovelOut = clamp01((f - (b1 - 55)) / 22);
  const digging = f > digT && f < b1 - 55;
  const shovelX = interpolate(shovelIn - shovelOut, [0, 1], [1520, 640]) + (digging ? Math.sin(f / 8) * 14 : 0);
  const shovelY = 700 + (digging ? Math.abs(Math.sin(f / 8)) * 34 : 0);
  const shovelRot = digging ? Math.sin(f / 8) * 10 : -8;

  const riseK = clamp01((f - (b1 - 50)) / 50);
  const heroX = interpolate(riseK, [0, 1], [660, 960]);
  const heroY = interpolate(riseK, [0, 1], [890, 470]);
  const heroS = interpolate(riseK, [0, 1], [0.75, 1.55]);

  const rubT = b1 + 70;
  const rubOn = f > rubT && f < rubT + 55;
  const spX = 1085 + (rubOn ? Math.sin(f / 3.5) * 26 : 0);
  const chipK = clamp01((f - rubT - 40) / 14);
  const germPop = Math.max(0.001, spring({frame: f - b2 - 8, fps, config: {damping: 8, stiffness: 170}}));
  const flowPop = Math.max(0.001, spring({frame: f - b2 - 58, fps, config: {damping: 10, stiffness: 130}}));
  const spOp = clamp01((f - rubT + 10) / 10) * (1 - clamp01((f - rubT - 70) / 14));

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={0} name={sec.name} accent={P.amber} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 荷塘水面 */}
        <rect x={0} y={680} width={1920} height={100} fill={P.waterD} />
        <rect x={0} y={680} width={1920} height={14} fill={P.water} />
        <LotusPad x={280} y={672} s={0.75} rot={-8} />
        <text x={64} y={742} fill="#bfe8ee" fontSize={28} fontFamily={FONT_HEAD}>荷塘水面</text>
        {/* 泥炭层 */}
        <rect x={0} y={780} width={1920} height={300} fill={P.peat} />
        <rect x={0} y={780} width={1920} height={10} fill={P.peatD} />
        {Array.from({length: 26}).map((_, i) => (
          <circle key={i} cx={rand(i, 81) * 1900 + 10} cy={rand(i, 82) * 270 + 800}
            r={2.5 + rand(i, 83) * 4} fill={i % 3 ? P.peatD : P.peatL} opacity={0.55} />
        ))}
        <text x={1240} y={858} fill={P.cream} fontSize={30} fontFamily={FONT_HEAD} opacity={0.9}>辽宁 · 普兰店泥炭层</text>
        {/* 挖掘坑 */}
        <rect x={430} y={795} width={430} height={172} rx={22} fill="#201407" />
        <rect x={430} y={795} width={430} height={172} rx={22} fill="none" stroke={P.peatL} strokeWidth={5} strokeDasharray="16 12" />
        <LotusSeed x={525} y={905} s={0.7} rot={-18} />
        <LotusSeed x={790} y={918} s={0.62} rot={22} />
        {/* 主角古莲子 */}
        <LotusSeed x={heroX} y={heroY} s={heroS} rot={riseK * Math.sin(f / 11) * 6} chipped={chipK} germ={germPop} />
        {/* 挖土碎屑 */}
        {[0, 1, 2, 3, 4].map((i) => {
          const at = digT + 14 + i * 17;
          const t = f - at;
          if (t < 0 || t > 32) return null;
          const p = t / 32;
          return <circle key={i} cx={540 + rand(i, 7) * 220} cy={806 - Math.sin(p * Math.PI) * 80}
            r={6 + rand(i, 9) * 6} fill={P.peatL} opacity={1 - p} />;
        })}
        {/* 铲子 */}
        <Shovel x={shovelX} y={shovelY} rot={shovelRot} s={1.05} op={shovelIn * (1 - shovelOut)} />
        {/* 实验室磨壳 */}
        {spOp > 0 && <Sandpaper x={spX} y={368} rot={-16} op={spOp} />}
        {rubOn && <Ripple x={1014} y={384} startFrame={rubT + 38} color={P.cream} />}
        {/* 发芽开花 */}
        <LotusFlower x={1300} y={600} s={0.95} bloom={flowPop} />
        <LotusPad x={1180} y={706} s={0.6} rot={10} op={flowPop} />
      </svg>
      <Chip text="1950s · 泥炭层出土" x={140} y={150} at={b0 + 20} accent={P.amber} />
      <Chip text="在泥里睡了一千多年" x={140} y={250} at={b0 + 90} accent={P.leaf} />
      <Stamp text="碳测定 ≈ 1300年" x={1330} y={160} at={b1 + 8} color={P.amberD} size={44} rot={4} />
      <Stamp text="居然开花了！" x={1350} y={255} at={b2 + 88} color={P.petalD} size={44} rot={-4} />
      <Chip text="莲子 · 发芽寿命天花板" x={140} y={780} at={b2 + 30} accent={P.petal} />
    </FadeIn>
  );
};

/* ============ S1 密封舱：剖面硬壳 + 水/氧/菌全部弹开 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b0 = beat(0), b1 = beat(1);

  const ringK = Math.max(0.001, spring({frame: f - b1 - 4, fps, config: {damping: 11, stiffness: 140}}));
  const ringPulse = 0.55 + 0.45 * Math.sin(f / 9);
  const shots: {at: number; y: number; type: 'o2' | 'drop' | 'germ'}[] = [
    {at: b0 + 30, y: 430, type: 'o2'},
    {at: b0 + 72, y: 560, type: 'drop'},
    {at: b0 + 112, y: 486, type: 'germ'},
    {at: b0 + 152, y: 636, type: 'o2'},
    {at: b0 + 196, y: 372, type: 'drop'},
    {at: b0 + 240, y: 600, type: 'germ'},
    {at: b0 + 284, y: 508, type: 'o2'},
    {at: b0 + 330, y: 452, type: 'germ'},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={1} name={sec.name} accent={P.leaf} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <ellipse cx={600} cy={856} rx={300} ry={32} fill={P.bgDeep} />
        {/* 密封边界环 */}
        <circle cx={600} cy={540} r={252 * ringK} fill="none" stroke={P.leaf} strokeWidth={6}
          strokeDasharray="16 13" opacity={0.25 + ringK * ringPulse * 0.6} />
        {/* 大剖面莲子 */}
        <LotusSeed x={600} y={540} s={2.4} cut={1} face="sleep" />
        {/* 标注引线 */}
        <g opacity={clamp01((f - b0 - 15) / 15)}>
          <path d="M 712 618 L 952 636" stroke={P.cream} strokeWidth={4} strokeDasharray="8 8" />
          <text x={962} y={646} fill={P.cream} fontSize={30} fontFamily={FONT_HEAD}>角质化硬壳 · 硬如陶瓷</text>
        </g>
        <g opacity={clamp01((f - b0 - 60) / 15)}>
          <path d="M 596 358 L 486 260" stroke={P.cream} strokeWidth={4} strokeDasharray="8 8" />
          <text x={120} y={252} fill={P.cream} fontSize={30} fontFamily={FONT_HEAD}>针尖大种孔 · 唯一换气口</text>
        </g>
        <g opacity={clamp01((f - b1 - 15) / 15)}>
          <path d="M 652 512 L 946 434" stroke={P.cream} strokeWidth={4} strokeDasharray="8 8" />
          <text x={956} y={444} fill={P.cream} fontSize={30} fontFamily={FONT_HEAD}>胚 · 干燥待机中</text>
          <text x={600} y={258} textAnchor="middle" fill={P.leaf} fontSize={30} fontFamily={FONT_HEAD} fontWeight={700}>密封舱边界</text>
        </g>
        {/* 弹开的水/氧/菌 */}
        {shots.map((o, i) => {
          const t = f - o.at;
          if (t < 0 || t > 95) return null;
          const hit = 46;
          let x = interpolate(t, [0, hit], [2030, 830], {extrapolateRight: 'clamp'});
          let y = o.y, op = 1, sq = 0;
          if (t >= hit) {
            const bt = t - hit;
            x = 830 + bt * 6.5;
            y = o.y + bt * 2.4;
            op = 1 - clamp01(bt / 30);
            sq = Math.sin(clamp01(bt / 8) * Math.PI) * 0.35;
          }
          return (
            <g key={i} transform={`translate(${x},${y}) scale(${1 + sq * 0.4},${1 - sq * 0.3})`} opacity={op}>
              {o.type === 'o2' && <O2 x={0} y={0} />}
              {o.type === 'drop' && <WaterDrop x={0} y={0} />}
              {o.type === 'germ' && <GermBall x={0} y={0} />}
              {t >= hit && t < hit + 12 && (
                <g stroke={P.cream} strokeWidth={5} strokeLinecap="round">
                  <line x1={-30} y1={-24} x2={-44} y2={-36} />
                  <line x1={-34} y1={0} x2={-50} y2={0} />
                  <line x1={-30} y1={24} x2={-44} y2={36} />
                </g>
              )}
            </g>
          );
        })}
      </svg>
      <Stamp text="随身真空包装" x={1230} y={160} at={b1 + 12} color={P.leafD} size={46} rot={-4} />
      <Chip text="细菌真菌 · 进不去" x={1440} y={780} at={b0 + 140} accent={P.red} />
      <Chip text="泡水几十年 · 里面照样干" x={140} y={780} at={b1 + 45} accent={P.water} />
    </FadeIn>
  );
};

/* ============ S2 待机不是关机：胚在睡 + 代谢贴零 + 夜班维修组补DNA ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b0 = beat(0), b1 = beat(1), b2 = beat(2);

  const wakeK = clamp01((f - b2 - 30) / 20);
  const zOn = 1 - clamp01((f - b2 - 8) / 18);
  const crack = 1 - clamp01((f - b1 - 70) / 35);
  const botK = clamp01((f - b1 + 10) / 20);
  const bob1 = Math.sin(f / 9) * 7;
  const bob2 = Math.cos(f / 8) * 7;

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={2} name={sec.name} accent={P.sky} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <ellipse cx={540} cy={856} rx={290} ry={30} fill={P.bgDeep} />
        {/* 睡着的胚 */}
        <LotusSeed x={540} y={560} s={2.5} cut={1} face={wakeK > 0.5 ? 'wake' : 'sleep'} />
        {zOn > 0 && [0, 1, 2].map((i) => {
          const t = ((f / 26 + i * 0.9) % 3);
          return (
            <text key={i} x={660 + i * 46 + t * 26} y={330 - t * 56} fill={C.muted}
              fontSize={42 - i * 6} fontFamily={FONT_HEAD} opacity={zOn * (1 - t / 3)}>Z</text>
          );
        })}
        {wakeK > 0 && (
          <g opacity={wakeK}>
            <WaterDrop x={700} y={330} s={0.9} rot={-10} />
            <path d="M 640 420 L 592 462" stroke={P.water} strokeWidth={6} strokeLinecap="round" strokeDasharray="10 9" />
          </g>
        )}
        {/* 代谢率仪表盘 */}
        <text x={1370} y={168} textAnchor="middle" fill={P.cream} fontSize={30} fontFamily={FONT_HEAD}>代谢率</text>
        <Gauge x={1370} y={330} s={1.1} k={0.045 + 0.02 * Math.sin(f / 9)} />
        <text x={1370} y={392} textAnchor="middle" fill={P.leaf} fontSize={30} fontFamily={FONT_HEAD} fontWeight={700}>≈ 0 · 但没归零</text>
        <Battery x={1660} y={305} s={1} />
        {/* DNA 夜班维修 */}
        <text x={1370} y={560} textAnchor="middle" fill={P.cream} fontSize={28} fontFamily={FONT_HEAD}>DNA 损伤 · 慢慢修</text>
        <DnaStrand x={1370} y={640} s={1.1} crack={crack} bgCol={P.bg} />
        <g opacity={botK}>
          <RepairBot x={1230} y={520 + bob1} s={0.95} rot={-6} />
          <RepairBot x={1520} y={726 + bob2} s={0.9} rot={8} />
        </g>
        <text x={1370} y={792} textAnchor="middle" fill={C.muted} fontSize={26} fontFamily={FONT_HEAD}>抗氧化酶 + 热休克蛋白 · 夜班维修组</text>
        {crack < 1 && f > b1 + 70 && (
          <text x={1370} y={490} textAnchor="middle" fill={P.leaf} fontSize={28} fontFamily={FONT_HEAD}
            fontWeight={800} opacity={1 - clamp01((f - b1 - 130) / 30)}>补好了 ✓</text>
        )}
      </svg>
      <Chip text="一千年损伤 · 在可修范围" x={140} y={150} at={b1 + 25} accent={P.amber} />
      <Stamp text="待机 ≠ 关机" x={140} y={255} at={b2 + 15} color={P.leafD} size={46} rot={4} />
      <Chip text="吸水 → 立刻重启" x={1520} y={150} at={b2 + 55} accent={P.sky} />
    </FadeIn>
  );
};

/* ============ S3 想发芽先磨壳：泡水几年纹丝不动 / 砂纸一磨就冒芽 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b0 = beat(0), b1 = beat(1), b2 = beat(2);

  const rubT = b1 + 15;
  const rubEnd = rubT + 80;
  const rubOn = f > rubT && f < rubEnd;
  const spX = 1580 + (rubOn ? Math.sin(f / 3.5) * 28 : 0);
  const spOp = clamp01((f - rubT + 8) / 10) * (1 - clamp01((f - rubEnd - 12) / 14));
  const chipK = clamp01((f - rubEnd + 5) / 12);
  const germPop = Math.max(0.001, spring({frame: f - rubEnd - 55, fps, config: {damping: 8, stiffness: 170}}));
  const calIdx = Math.min(3, Math.floor(Math.max(0, f - b0 - 10) / 34));
  const calLabel = ['泡 1 年', '泡 2 年', '泡 3 年', '泡 N 年'][calIdx];
  const iconK = (i: number) => Math.max(0.001, spring({frame: f - b2 - 10 - i * 12, fps, config: {damping: 10, stiffness: 150}}));

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={3} name={sec.name} accent={P.water} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <line x1={960} y1={150} x2={960} y2={845} stroke={P.cream} strokeWidth={3} strokeDasharray="4 18" opacity={0.3} />
        {/* 左：泡水烧杯 */}
        <ellipse cx={440} cy={858} rx={220} ry={26} fill={P.bgDeep} />
        <g transform="translate(440,590)">
          <path d="M -140 -160 L -140 110 Q -140 145 -105 145 L 105 145 Q 140 145 140 110 L 140 -160"
            fill="rgba(207,232,242,0.10)" stroke="#cfe4f0" strokeWidth={8} strokeLinecap="round" />
          <path d="M -128 -60 L 128 -60 L 128 108 Q 128 133 102 133 L -102 133 Q -128 133 -128 108 Z"
            fill={P.water} opacity={0.85} />
          <ellipse cx={0} cy={-60} rx={128} ry={10} fill="#7fd4e0" />
          <LotusSeed x={0} y={88} s={0.8} rot={-12} />
          <circle cx={0} cy={88} r={62} fill="none" stroke={P.cream} strokeWidth={4} strokeDasharray="9 9" opacity={0.5} />
        </g>
        <Calendar x={720} y={330} s={1} label={calLabel} />
        <text x={440} y={852} textAnchor="middle" fill={P.cream} fontSize={30} fontFamily={FONT_HEAD}>泡水好几年 · 纹丝不动</text>
        {/* 右：磨壳发芽 */}
        <ellipse cx={1420} cy={858} rx={220} ry={26} fill={P.bgDeep} />
        <LotusSeed x={1420} y={600} s={1.9} rot={6} chipped={chipK} germ={germPop} />
        {spOp > 0 && <Sandpaper x={spX} y={468} rot={-14} op={spOp} />}
        {rubOn && <Ripple x={1490} y={478} startFrame={rubEnd - 14} color={P.cream} />}
        {[0, 1, 2].map((i) => {
          const at = rubEnd + 8 + i * 10;
          const t = f - at;
          if (t < 0 || t > 30) return null;
          return <WaterDrop key={i} x={1470 + i * 16} y={interpolate(t, [0, 30], [330, 462], {extrapolateRight: 'clamp'})}
            s={0.6} op={1 - clamp01((t - 24) / 8)} />;
        })}
        <text x={1420} y={852} textAnchor="middle" fill={P.cream} fontSize={30} fontFamily={FONT_HEAD}>磨破进水 · 几天冒芽</text>
        {/* 自然破壳三件套 */}
        <g transform="translate(1120,300) scale(1.05)"><g transform={`scale(${iconK(0)})`}><Rock x={0} y={0} /></g></g>
        <g transform="translate(1420,300) scale(1.05)"><g transform={`scale(${iconK(1)})`}><Snowflake x={0} y={0} /></g></g>
        <g transform="translate(1720,296) scale(1.05)"><g transform={`scale(${iconK(2)})`}><Bird x={0} y={0} /></g></g>
        <g opacity={clamp01((f - b2 - 30) / 15)}>
          <text x={1120} y={366} textAnchor="middle" fill={C.muted} fontSize={26} fontFamily={FONT_HEAD}>摩擦</text>
          <text x={1420} y={366} textAnchor="middle" fill={C.muted} fontSize={26} fontFamily={FONT_HEAD}>冻融</text>
          <text x={1720} y={366} textAnchor="middle" fill={C.muted} fontSize={26} fontFamily={FONT_HEAD}>鸟兽啃咬</text>
        </g>
      </svg>
      <Chip text="泡水几年 → 0 动静" x={140} y={150} at={b0 + 30} accent={P.water} />
      <Stamp text="磨破 = 拿到钥匙" x={1430} y={150} at={rubEnd + 70} color={P.leafD} size={42} rot={5} />
      <Chip text="保险箱连自己都打不开" x={140} y={240} at={b2 + 10} accent={P.amber} />
    </FadeIn>
  );
};

/* ============ S4 长寿榜：莲子/椰枣自然发芽赛道 + 一朵宋朝的花 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b0 = beat(0), b1 = beat(1);

  const bars = [
    {x: 300, h: 175, col: P.leaf, name: '莲子', age: '≈1300年'},
    {x: 540, h: 235, col: P.amber, name: '椰枣', age: '≈2000年'},
  ];
  const g0 = (i: number) => Math.max(0.001, spring({frame: f - b0 - 20 - i * 16, fps, config: {damping: 11, stiffness: 140}}));
  const g3 = Math.max(0.001, spring({frame: f - b0 - 60, fps, config: {damping: 11, stiffness: 140}}));
  const tagK = clamp01((f - b0 - 120) / 18);
  const bloomK = Math.max(0.001, spring({frame: f - b1 - 10, fps, config: {damping: 10, stiffness: 130}}));
  const envK = Math.max(0.001, spring({frame: f - b1 - 55, fps, config: {damping: 10, stiffness: 140}}));

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={4} name={sec.name} accent={P.petal} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：发芽寿命柱状榜 */}
        <line x1={160} y1={820} x2={930} y2={820} stroke={P.card} strokeWidth={7} strokeLinecap="round" />
        <text x={400} y={430} textAnchor="middle" fill={P.cream} fontSize={30} fontFamily={FONT_HEAD}>自然发芽赛道 ★</text>
        {bars.map((b, i) => {
          const h = b.h * g0(i);
          return (
            <g key={i}>
              <rect x={b.x - 60} y={820 - h} width={120} height={h} rx={12} fill={b.col} stroke={i ? P.amberD : P.leafD} strokeWidth={5} />
              <text x={b.x} y={862} textAnchor="middle" fill={P.cream} fontSize={28} fontFamily={FONT_HEAD} fontWeight={700}>{b.age}</text>
              <text x={b.x} y={896} textAnchor="middle" fill={C.muted} fontSize={26} fontFamily={FONT_HEAD}>{b.name}</text>
            </g>
          );
        })}
        <LotusSeed x={300} y={820 - 175 * g0(0) - 52} s={0.52} rot={-8} />
        <Palm x={540} y={820 - 235 * g0(1) - 48} s={0.62} />
        {/* 蝇子草：另一条赛道（虚线空心柱） */}
        <rect x={720} y={820 - 315 * g3} width={120} height={315 * g3} rx={12}
          fill="rgba(168,152,128,0.10)" stroke={P.grey} strokeWidth={5} strokeDasharray="14 10" />
        <text x={780} y={862} textAnchor="middle" fill={P.grey} fontSize={28} fontFamily={FONT_HEAD} fontWeight={700}>3.2万年</text>
        <text x={780} y={896} textAnchor="middle" fill={C.muted} fontSize={26} fontFamily={FONT_HEAD}>蝇子草</text>
        <Campion x={780} y={820 - 315 * g3 - 42} s={0.55} />
        <g opacity={tagK}>
          <rect x={686} y={330} width={190} height={44} rx={12} fill="none" stroke={P.grey} strokeWidth={3} />
          <text x={781} y={361} textAnchor="middle" fill={P.grey} fontSize={24} fontFamily={FONT_HEAD}>组织培养赛道</text>
        </g>
        {/* 右：古莲开花 + 宋朝来信 */}
        <path d="M 1430 560 Q 1420 640 1430 716" fill="none" stroke={P.leafD} strokeWidth={9} strokeLinecap="round" opacity={bloomK} />
        <LotusPad x={1180} y={716} s={0.95} rot={-6} op={bloomK} />
        <LotusFlower x={1430} y={470} s={1.9} bloom={bloomK} />
        <g transform={`translate(1690,730) scale(${envK})`}>
          <Envelope x={0} y={0} rot={-10} />
        </g>
        <text x={1430} y={806} textAnchor="middle" fill={P.amber} fontSize={30} fontFamily={FONT_HEAD}
          fontWeight={700} opacity={envK}>宋朝 → 现代 · 直达</text>
      </svg>
      <Chip text="马萨达椰枣 · 2000岁" x={140} y={150} at={b0 + 25} accent={P.amber} />
      <Chip text="蝇子草 · 人工培养复活" x={140} y={240} at={b0 + 150} accent={P.grey} />
      <Stamp text="一朵宋朝寄来的花" x={1180} y={150} at={b1 + 70} color={P.petalD} size={44} rot={-4} />
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
        <LotusPad x={1180} y={950} s={1.25} rot={6} />
        <g transform={`translate(1240,800) scale(${pop})`}>
          <LotusFlower x={0} y={0} s={1.5} />
        </g>
        <g transform={`translate(690,905) rotate(-10) scale(${pop})`}>
          <LotusSeed x={0} y={0} s={1.05} germ={1} />
        </g>
        <WaterDrop x={880} y={700} s={0.8} op={0.7} />
        <WaterDrop x={1560} y={640} s={0.6} rot={10} op={0.6} />
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -250}}>
        <div style={{fontSize: 64, fontWeight: 800, color: C.text}}>它没赢过时间，是让时间扑了个空</div>
        <div style={{fontSize: 38, color: C.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E218 = ({data = tl}: {data?: any}) => {
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
