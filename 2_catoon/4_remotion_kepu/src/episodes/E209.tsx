import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01, rand} from '../lib/style';
import {Backdrop, Ripple, Stamp, SceneTag, Chip, Subtitles, ProgressBar, FadeIn} from '../lib/atoms';
import tl from '../timeline_209.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：果皮紫 × 果肉红 × 暖黄（纯色平涂） ============ */
const P = {
  bg: '#482049',      // 主底：紫李
  bgDeep: '#331437',  // 深一号
  bgLite: '#55264f',  // 亮一号
  bgFlesh: '#4c1d40', // 果肉场景底
  plate: '#f7ecd6',   // 奶油圆盘（衬剖面）
  card: '#5b2a54',    // 卡片底
  skin: '#8e4583',    // 果皮
  skinD: '#5e2a57',   // 果皮暗
  skinH: '#b068a0',   // 果皮亮
  flesh: '#ffc9a8',   // 果肉浅粉
  fleshD: '#e5587f',  // 果膛紫红
  fleshDD: '#bf3f66', // 果膛深边
  flower: '#ffd166',  // 内壁小花 / 脆粒
  seedD: '#e8a93f',
  wasp: '#2b1c22',    // 榕小蜂深色
  waspH: '#57414b',
  wing: '#f4e9f7',    // 翅膀
  amber: '#ffc94d',
  cream: '#f7ecd6',
  leaf: '#8fd18f',
  mint: '#7fe0c3',
  coral: '#ff7b6b',
  ink: '#3a1230',     // 奶油盘上的深字
};

/* ---------- 整果：水滴形，顶端果柄、底部果孔小眼 ---------- */
const figD = (W: number, H: number) =>
  `M 0 ${-H} C ${W * 0.5} ${-H * 0.86} ${W * 0.98} ${-H * 0.36} ${W} ${H * 0.08} ` +
  `C ${W} ${H * 0.6} ${W * 0.58} ${H} 0 ${H} C ${-W * 0.58} ${H} ${-W} ${H * 0.6} ${-W} ${H * 0.08} ` +
  `C ${-W * 0.98} ${-H * 0.36} ${-W * 0.5} ${-H * 0.86} 0 ${-H} Z`;

const FigWhole = ({x, y, s = 1, tilt = 0, pop = 1, tint = P.skin}: {
  x: number; y: number; s?: number; tilt?: number; pop?: number; tint?: string;
}) => {
  const W = 110, H = 140;
  const k = Math.max(0.001, pop);
  return (
    <g transform={`translate(${x},${y}) rotate(${tilt}) scale(${s * k})`} opacity={pop > 0 ? 1 : 0}>
      <rect x={-7} y={-H - 34} width={14} height={40} rx={6} fill={P.skinD} />
      <path d={figD(W, H)} fill={tint} />
      {/* 棱线 */}
      {[-0.45, 0, 0.45].map((k2, i) => (
        <path key={i} d={`M ${k2 * W * 0.6} ${-H * 0.66} Q ${k2 * W * 1.3} ${H * 0.1} ${k2 * W * 0.75} ${H * 0.78}`}
          stroke={P.skinD} strokeWidth={3.5} fill="none" opacity={0.35} />
      ))}
      <ellipse cx={-W * 0.36} cy={-H * 0.05} rx={W * 0.15} ry={H * 0.42} fill="#ffffff"
        opacity={0.15} transform="rotate(-10)" />
      {/* 果孔小眼 */}
      <ellipse cx={0} cy={H * 0.93} rx={W * 0.14} ry={H * 0.065} fill={P.skinD} />
      <ellipse cx={0} cy={H * 0.93} rx={W * 0.06} ry={H * 0.028} fill="#241122" />
    </g>
  );
};

/* ---------- 剖面：果皮环 + 果肉 + 内壁放射小花 + 底部果孔 ----------
   flowerK 0..1 小花从底部果孔侧绕圈依次点亮                          */
const FigCut = ({x, y, r, pop = 1, flowerK = 1, hi = 0}: {
  x: number; y: number; r: number; pop?: number; flowerK?: number; hi?: number;
}) => {
  const f = useCurrentFrame();
  const N = 42;
  const k = Math.max(0.001, pop);
  return (
    <g transform={`translate(${x},${y}) scale(${k})`} opacity={pop > 0 ? 1 : 0}>
      <circle r={r} fill={P.skin} />
      <circle r={r * 0.95} fill={P.skinD} opacity={0.55} />
      <circle r={r * 0.84} fill={P.flesh} />
      <circle r={r * 0.63} fill={P.fleshD} />
      {Array.from({length: N}).map((_, i) => {
        const a = Math.PI / 2 + (i / N) * Math.PI * 2;
        const g = clamp01(flowerK * 2 - (i / N) * 1.35);
        if (g <= 0) return null;
        const x1 = Math.cos(a) * r * 0.61, y1 = Math.sin(a) * r * 0.61;
        const x0 = Math.cos(a) * r * 0.30, y0 = Math.sin(a) * r * 0.30;
        const fx = Math.cos(a) * r * 0.335, fy = Math.sin(a) * r * 0.335;
        return (
          <g key={i}>
            <line x1={x1} y1={y1} x2={x0} y2={y0} stroke="#ffdca6" strokeWidth={Math.max(1.5, r * 0.011)} opacity={0.7 * g} />
            <g transform={`translate(${fx},${fy}) rotate(${(a * 180) / Math.PI + 90}) scale(${g})`}>
              <ellipse rx={r * 0.05} ry={r * 0.031} fill={P.flower} />
              <ellipse cy={-r * 0.01} rx={r * 0.02} ry={r * 0.013} fill={P.fleshDD} opacity={0.55} />
            </g>
          </g>
        );
      })}
      <circle r={r * 0.21} fill={P.fleshD} stroke="#ffdca6" strokeWidth={2.5} />
      {/* 果孔：底部楔形通道 */}
      <path d={`M ${-r * 0.10} ${r * 0.45} L ${-r * 0.175} ${r * 0.99} L ${r * 0.175} ${r * 0.99} L ${r * 0.10} ${r * 0.45} Z`}
        fill="#2a0f24" stroke={P.skinD} strokeWidth={4} />
      {[-1, 0, 1].map((i) => (
        <path key={i} d={`M ${i * r * 0.11 - r * 0.055} ${r * 0.985} q ${r * 0.055} ${r * 0.045} ${r * 0.11} 0`}
          stroke={P.skinD} strokeWidth={5} fill="none" strokeLinecap="round" />
      ))}
      {hi > 0 && (
        <ellipse cx={0} cy={r * 0.99} rx={r * 0.24} ry={r * 0.07} fill="none" stroke={P.amber}
          strokeWidth={6} opacity={hi * (0.55 + 0.45 * Math.sin(f / 6))} />
      )}
    </g>
  );
};

/* ---------- 榕小蜂：深色小蜂，默认朝右；wings 0..1 翅膀完整度 ---------- */
const Wasp = ({x, y, s = 1, rot = 0, wings = 1, crawl = 0}: {
  x: number; y: number; s?: number; rot?: number; wings?: number; crawl?: number;
}) => {
  const f = useCurrentFrame();
  const legW = Math.sin(f * 0.8) * 3 * crawl;
  const flap = Math.sin(f * 1.1) * 10 * (crawl > 0 ? 0.3 : 1);
  return (
    <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`}>
      {wings > 0.03 && (
        <g opacity={0.9}>
          <ellipse cx={-6} cy={-16} rx={26} ry={9} fill={P.wing} opacity={0.85}
            transform={`rotate(${-24 - flap}) scale(${wings})`} />
          <ellipse cx={-16} cy={-13} rx={22} ry={8} fill={P.wing} opacity={0.6}
            transform={`rotate(${-38 - flap}) scale(${wings})`} />
        </g>
      )}
      {[-1, 0, 1].map((i) => (
        <line key={i} x1={i * 14 - 4} y1={10} x2={i * 16 - 10 + legW * (i === 0 ? 1 : i)} y2={24}
          stroke={P.wasp} strokeWidth={4} strokeLinecap="round" />
      ))}
      <ellipse rx={30} ry={13} fill={P.wasp} />
      <ellipse cy={-3} rx={26} ry={9} fill={P.waspH} opacity={0.7} />
      {[-14, -4, 6].map((sx) => (
        <path key={sx} d={`M ${sx} -10 Q ${sx - 3} 0 ${sx} 10`} stroke={P.waspH} strokeWidth={2.5} fill="none" opacity={0.8} />
      ))}
      <circle cx={32} cy={-2} r={10} fill={P.wasp} />
      <path d="M 38 -10 Q 48 -22 56 -24" stroke={P.wasp} strokeWidth={3.5} fill="none" strokeLinecap="round" />
      <path d="M 34 -11 Q 40 -26 46 -30" stroke={P.wasp} strokeWidth={3.5} fill="none" strokeLinecap="round" />
      <circle cx={35} cy={-4} r={3.2} fill="#fff" />
    </g>
  );
};

/* ---------- 被挤脱的翅膀：飘落淡出 ---------- */
const LooseWing = ({x, y, at, dir = 1, seed = 0}: {x: number; y: number; at: number; dir?: number; seed?: number}) => {
  const f = useCurrentFrame();
  const t = f - at;
  if (t < 0 || t > 70) return null;
  const p = clamp01(t / 60);
  return (
    <g transform={`translate(${x + dir * t * 1.1 + Math.sin(t * 0.3 + seed) * 9},${y - t * 0.7}) rotate(${dir * 22 + Math.sin(t * 0.25 + seed) * 26})`}
      opacity={0.9 * (1 - p)}>
      <ellipse rx={22} ry={8} fill={P.wing} />
    </g>
  );
};

/* ---------- 粒子迸散（消化/溶解用） ---------- */
const Burst = ({x, y, at, n = 10, col = P.amber}: {x: number; y: number; at: number; n?: number; col?: string}) => {
  const f = useCurrentFrame();
  const t = f - at;
  if (t < 0) return null;
  const p = clamp01(t / 55);
  if (p >= 1) return null;
  return (
    <g>
      {Array.from({length: n}).map((_, i) => {
        const a = rand(i, 31) * Math.PI * 2;
        const d = p * (40 + rand(i, 32) * 80);
        return <circle key={i} cx={x + Math.cos(a) * d} cy={y + Math.sin(a) * d * 0.8 - p * 30}
          r={5 * (1 - p) + 2} fill={col} opacity={(1 - p) * 0.9} />;
      })}
    </g>
  );
};

/* ---------- 蛋白酶液滴：落下→溅开 ---------- */
const EnzymeDrop = ({x, y0, y1, at, dur = 42}: {x: number; y0: number; y1: number; at: number; dur?: number}) => {
  const f = useCurrentFrame();
  const t = f - at;
  if (t < 0 || t > dur + 14) return null;
  const p = clamp01(t / dur);
  const y = y0 + (y1 - y0) * p;
  return (
    <g>
      {t <= dur ? (
        <g transform={`translate(${x},${y})`}>
          <line x1={0} y1={-28} x2={0} y2={-9} stroke={P.amber} strokeWidth={5} strokeLinecap="round" opacity={0.5} />
          <circle r={9} fill={P.amber} />
          <circle cx={-3} cy={-3} r={3} fill="#fff" opacity={0.6} />
        </g>
      ) : (
        <circle cx={x} cy={y1} r={8 + (t - dur) * 1.6} fill="none" stroke={P.amber}
          strokeWidth={3} opacity={0.7 * (1 - (t - dur) / 14)} />
      )}
    </g>
  );
};

/* ---------- 风弧：吹过来被果皮弹开 ---------- */
const WindArc = ({d, at, dur = 34}: {d: string; at: number; dur?: number}) => {
  const f = useCurrentFrame();
  const t = f - at;
  if (t < 0) return null;
  const p = clamp01(t / dur);
  const out = clamp01((t - dur - 20) / 25);
  return (
    <path d={d} pathLength={100} stroke={P.skinH} strokeWidth={7} fill="none" strokeLinecap="round"
      strokeDasharray={100} strokeDashoffset={100 * (1 - p)} opacity={0.8 * (1 - out)} />
  );
};

/* ============ 标题卡 ============ */
const TitleCard = ({data}: {data: any}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 13, mass: 0.7}});
  const pop = spring({frame: f - 6, fps, config: {damping: 10, stiffness: 150}});
  const o = interpolate(f, [0, 8], [0, 1]);
  return (
    <AbsoluteFill>
      <Backdrop top={P.bg} bottom={P.bg} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <circle cx={960} cy={370} r={235} fill={P.plate} />
        <FigCut x={960} y={370} r={175} pop={pop} flowerK={clamp01((f - 10) / 40)} hi={1} />
        <Wasp x={875} y={566} s={0.45} rot={-38} wings={1} />
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 240}}>
          <div style={{color: P.amber, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 26}}>
            冷知识 · 第 {data.episode} 期
          </div>
          <div style={{color: C.text, fontSize: 96, fontWeight: 900, fontFamily: FONT_HEAD, letterSpacing: 6}}>
            {data.title}
          </div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ============ S0 传言：整果 + 剖面，蜂钻进果孔；住过蜂但被消化 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);

  const popFig = spring({frame: f - 8, fps, config: {damping: 11, stiffness: 140}});
  const popCut = spring({frame: f - 16, fps, config: {damping: 11, stiffness: 140}});

  // 蜂飞入：抛物线 → 果孔
  const tW = clamp01((f - 30) / 130);
  const bx = (1 - tW) * (1 - tW) * 140 + 2 * (1 - tW) * tW * 780 + tW * tW * 1310;
  const by = (1 - tW) * (1 - tW) * 250 + 2 * (1 - tW) * tW * 90 + tW * tW * 812;
  const dive = clamp01((f - 162) / 26);
  const waspO = tW > 0 ? 1 - dive : 0;
  const waspRot = -18 + tW * 100 + Math.sin(f * 0.6) * 6;

  // 果膛里的"住客"：住过蜂（出现）→ 被消化（消散）
  const g1 = spring({frame: f - beat(1) - 20, fps, config: {damping: 10, stiffness: 160}});
  const g2 = spring({frame: f - beat(1) - 38, fps, config: {damping: 10, stiffness: 160}});
  const melt = clamp01((f - beat(1) - 175) / 55);

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={0} name={sec.name} accent={P.amber} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <circle cx={1310} cy={530} r={380} fill={P.plate} />
        <FigWhole x={520} y={600} s={1.7} tilt={-8} pop={popFig} />
        <FigCut x={1310} y={530} r={300} pop={popCut} flowerK={clamp01((f - 30) / 70)} />
        <Ripple x={1310} y={838} startFrame={162} color={P.amber} />

        {/* 飞行拖尾 */}
        {Array.from({length: 7}).map((_, i) => {
          const tt = tW - 0.05 - i * 0.055;
          if (tt <= 0 || tW >= 1) return null;
          const px = (1 - tt) * (1 - tt) * 140 + 2 * (1 - tt) * tt * 780 + tt * tt * 1310;
          const py = (1 - tt) * (1 - tt) * 250 + 2 * (1 - tt) * tt * 90 + tt * tt * 812;
          return <circle key={i} cx={px} cy={py} r={6 - i * 0.7} fill={P.cream} opacity={0.35 - i * 0.04} />;
        })}
        {/* 飞进果孔的蜂 */}
        {waspO > 0 && (
          <g opacity={waspO}>
            <Wasp x={bx} y={by + dive * 42} s={0.8 * (1 - dive * 0.75)} rot={waspRot} wings={1} />
          </g>
        )}

        {/* 膛内住客：出现后又被"消化" */}
        <g opacity={clamp01(g1) * (1 - melt)}>
          <Wasp x={1248} y={472} s={0.5} rot={35} wings={0.4} />
        </g>
        <g opacity={clamp01(g2) * (1 - melt)}>
          <Wasp x={1372} y={608} s={0.42} rot={-150} wings={0.3} />
        </g>
        <Burst x={1248} y={472} at={beat(1) + 175} col={P.amber} n={10} />
        <Burst x={1372} y={608} at={beat(1) + 185} col={P.amber} n={10} />
      </svg>
      <Stamp text="果里藏着死蜂？！" x={150} y={235} at={beat(0) + 55} color={P.coral} size={52} rot={-7} />
      <Stamp text="确实住过蜂" x={1420} y={145} at={beat(1) + 16} color={P.amber} size={50} rot={5} />
      <Chip text="但早被消化干净" x={150} y={150} at={beat(1) + 190} accent={P.mint} />
    </FadeIn>
  );
};

/* ============ S1 藏花：剖面大特写，内壁几百朵小花 + 唯一果孔 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);

  const pop = spring({frame: f - 10, fps, config: {damping: 11, stiffness: 130}});
  const flowerK = clamp01((f - 28) / 95);
  const ringP = interpolate(f, [b1 + 6, b1 + 20, b1 + 55, b1 + 72], [0, 0.9, 0.9, 0],
    {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const ostK = clamp01((f - b2 - 14) / 16);
  const lab1 = clamp01((f - b1 - 18) / 10);
  const lab2 = clamp01((f - b1 - 62) / 10);
  const lab3 = clamp01((f - b2 - 34) / 10);

  const CX = 1290, CY = 520, R = 320;
  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={1} name={sec.name} accent={P.flower} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <circle cx={CX} cy={CY} r={385} fill={P.plate} />
        {/* 左：整果 + 放大定位虚线 */}
        <FigWhole x={360} y={645} s={1.05} tilt={-10} pop={spring({frame: f - 5, fps, config: {damping: 11, stiffness: 140}})} />
        <circle cx={360} cy={645} r={165} fill="none" stroke={P.amber} strokeWidth={5} strokeDasharray="12 10" opacity={0.9} />
        <line x1={520} y1={590} x2={905} y2={455} stroke={P.amber} strokeWidth={4} strokeDasharray="12 10" opacity={0.7} />
        {/* 右：剖面大特写 */}
        <FigCut x={CX} y={CY} r={R} pop={pop} flowerK={flowerK} hi={ostK} />
        {/* 果肉=花托的高亮环 */}
        <circle cx={CX} cy={CY} r={R * 0.735} fill="none" stroke={P.amber} strokeWidth={12} opacity={ringP} />
        {/* 引线标注 */}
        {lab1 > 0 && (
          <g opacity={lab1}>
            <line x1={885} y1={632} x2={1058} y2={603} stroke={P.amber} strokeWidth={3.5} strokeDasharray="7 6" />
            <text x={868} y={645} textAnchor="end" fill={P.flesh} fontSize={38} fontFamily={FONT_HEAD} fontWeight={700}>膨大的花托</text>
          </g>
        )}
        {lab2 > 0 && (
          <g opacity={lab2}>
            <line x1={1190} y1={188} x2={1252} y2={392} stroke={P.ink} strokeWidth={3.5} strokeDasharray="7 6" />
            <text x={1190} y={172} textAnchor="middle" fill={P.ink} fontSize={38} fontFamily={FONT_HEAD} fontWeight={700}>真正的小花</text>
          </g>
        )}
        {lab3 > 0 && (
          <g opacity={lab3}>
            <line x1={1450} y1={850} x2={1350} y2={838} stroke={P.amber} strokeWidth={3.5} strokeDasharray="7 6" />
            <text x={1465} y={862} fill={P.amber} fontSize={34} fontFamily={FONT_HEAD} fontWeight={700}>果孔·唯一入口</text>
          </g>
        )}
        {/* 风：吹不进去 */}
        <WindArc d="M 620 330 Q 840 295 950 355 Q 918 415 852 415" at={b2 + 8} />
        <WindArc d="M 585 520 Q 820 495 940 545 Q 908 605 838 595" at={b2 + 21} />
        <WindArc d="M 645 705 Q 855 695 955 735 Q 925 790 862 782" at={b2 + 34} />
      </svg>
      <Stamp text="几百朵花！" x={1420} y={105} at={beat(0) + 125} color={P.fleshDD} size={46} rot={4} />
      <Chip text="果肉其实是花托" x={140} y={170} at={b1 + 8} accent={P.mint} />
      <Chip text="脆粒=真·花" x={140} y={268} at={b1 + 55} accent={P.amber} />
      <Stamp text="风吹不进" x={610} y={215} at={b2 + 26} color={P.skinH} size={46} rot={-6} />
    </FadeIn>
  );
};

/* ============ S2 单程票：蜂挤进果孔（放大窗：挤断翅→爬进去授粉产卵→留下） ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b2 = beat(2);

  const popCut = spring({frame: f - 10, fps, config: {damping: 11, stiffness: 130}});

  // 外部飞来的蜂：落进底部果孔
  const tW = clamp01((f - 25) / 135);
  const bx = (1 - tW) * (1 - tW) * 170 + 2 * (1 - tW) * tW * 330 + tW * tW * 560;
  const by = (1 - tW) * (1 - tW) * 240 + 2 * (1 - tW) * tW * 700 + tW * tW * 828;
  const dive = clamp01((f - 163) / 26);
  const outO = tW > 0 ? 1 - dive : 0;

  // 放大窗
  const panelO = clamp01((f - 110) / 15);
  const phaseA = clamp01((f - 115) / 15) * (1 - clamp01((f - 432) / 20));
  const phaseB = clamp01((f - 432) / 20);

  // 隧道内硬挤（phaseA 内容）
  const wy = interpolate(f, [185, 330], [268, 655], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const sq = clamp01(1 - Math.abs(wy - 470) / 110);       // 最窄处挤压
  const inFade = 1 - clamp01((f - 335) / 15);
  const wingK = clamp01((300 - f) / 8);                    // 300f 翅膀脱落
  const channelX = 1430 + Math.sin(f * 0.3) * 5 * sq;

  // 膛内爬行（phaseB 内容）
  const wallTop = (wx: number) => 762 - 285 * Math.sqrt(Math.max(0.02, 1 - ((wx - 1430) / 410) ** 2));
  const cx2 = interpolate(f, [450, 565], [1185, 1655], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const cy2 = wallTop(cx2) - 34;
  const deadK = clamp01((f - 578) / 18);
  const crawl = f >= 450 && f < 575 ? 1 : 0;
  const flowersX = [1210, 1305, 1400, 1495, 1590, 1685];

  return (
    <FadeIn>
      <Backdrop top={P.bgDeep} bottom={P.bgDeep} />
      <SceneTag index={2} name={sec.name} accent={P.coral} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <defs>
          <clipPath id="s2clip"><rect x={1090} y={205} width={680} height={500} rx={42} /></clipPath>
        </defs>

        <FigCut x={560} y={530} r={300} pop={popCut} flowerK={clamp01((f - 20) / 60)} hi={clamp01((f - 140) / 20)} />
        <Ripple x={560} y={832} startFrame={160} color={P.amber} />
        {/* 果孔→放大窗 虚线 */}
        {panelO > 0 && (
          <path d="M 575 822 C 760 830 920 740 1078 610" stroke={P.amber} strokeWidth={4}
            strokeDasharray="10 12" fill="none" opacity={0.8 * panelO} />
        )}
        {/* 外部蜂拖尾 + 本体 */}
        {Array.from({length: 6}).map((_, i) => {
          const tt = tW - 0.05 - i * 0.06;
          if (tt <= 0 || tW >= 1) return null;
          const px = (1 - tt) * (1 - tt) * 170 + 2 * (1 - tt) * tt * 330 + tt * tt * 560;
          const py = (1 - tt) * (1 - tt) * 240 + 2 * (1 - tt) * tt * 700 + tt * tt * 828;
          return <circle key={i} cx={px} cy={py} r={5.5 - i * 0.7} fill={P.cream} opacity={0.35 - i * 0.045} />;
        })}
        {outO > 0 && (
          <g opacity={outO}>
            <Wasp x={bx} y={by + dive * 40} s={0.75 * (1 - dive * 0.75)} rot={-15 + tW * 85 + Math.sin(f * 0.6) * 5} />
          </g>
        )}

        {/* ============ 放大窗 ============ */}
        <g opacity={panelO}>
          <rect x={1090} y={205} width={680} height={500} rx={42} fill="#200e1f" stroke={P.skinH} strokeWidth={5} />
          <g clipPath="url(#s2clip)">
            {/* ---- A：隧道硬挤 ---- */}
            <g opacity={phaseA}>
              <rect x={1150} y={225} width={560} height={85} rx={20} fill={P.bgDeep} />
              <rect x={1150} y={320} width={225} height={330} rx={26} fill={P.flesh} />
              <rect x={1485} y={320} width={225} height={330} rx={26} fill={P.flesh} />
              {[355, 425, 495, 565].map((yy) => (
                <g key={yy}>
                  <circle cx={1378} cy={yy} r={13} fill={P.fleshD} />
                  <circle cx={1482} cy={yy} r={13} fill={P.fleshD} />
                </g>
              ))}
              {/* 最窄处鼓包 */}
              <ellipse cx={1382} cy={470} rx={36} ry={72} fill={P.flesh} />
              <ellipse cx={1478} cy={470} rx={36} ry={72} fill={P.flesh} />
              {/* 蜂：竖着挤进来 */}
              {inFade > 0 && (
                <g opacity={inFade}>
                  <Wasp x={channelX} y={wy} s={1.15} rot={90} wings={wingK} />
                  <g transform={`translate(${channelX},${wy}) scale(${1.15}) scale(1,${1 - sq * 0.45})`} opacity={0}>
                    <ellipse rx={30} ry={13} fill={P.wasp} />
                  </g>
                </g>
              )}
              <LooseWing x={1408} y={452} at={300} dir={-1} seed={2} />
              <LooseWing x={1452} y={468} at={302} dir={1} seed={7} />
              <Burst x={1430} y={672} at={336} col={P.amber} n={8} />
            </g>
            {/* ---- B：膛内爬壁授粉产卵 ---- */}
            <g opacity={phaseB}>
              <ellipse cx={1430} cy={762} rx={410} ry={285} fill={P.flesh} />
              {flowersX.map((fx, i) => {
                const fy = wallTop(fx) - 14;
                return (
                  <g key={i}>
                    <line x1={fx} y1={fy + 14} x2={fx} y2={fy} stroke="#ffdca6" strokeWidth={3.5} />
                    <ellipse cx={fx} cy={fy - 6} rx={11} ry={7} fill={P.flower} />
                    {[1, 3, 4].includes(i) && (
                      <ellipse cx={fx + 4} cy={fy - 10} rx={7} ry={5} fill={P.cream}
                        opacity={clamp01((f - (462 + ((fx - 1185) / 470) * 112)) / 6)} />
                    )}
                  </g>
                );
              })}
              {/* 爬行的无翅蜂 */}
              <g transform={`translate(${cx2},${cy2 + deadK * 8})`}>
                <Wasp x={0} y={0} s={0.95} rot={-6 + deadK * 160} wings={0.08} crawl={crawl} />
              </g>
              {/* 花粉闪点 */}
              {[0, 1, 2, 3].map((i) => {
                const at = 468 + i * 26;
                const tt = f - at;
                if (tt < 0 || tt > 40) return null;
                const kk = Math.sin(clamp01(tt / 40) * Math.PI);
                const px = cx2 - 30 - i * 24;
                return <circle key={i} cx={px} cy={cy2 - 26} r={6 * kk} fill={P.amber} opacity={0.85 * kk} />;
              })}
              {deadK > 0.4 && (
                <text x={1445} y={625} textAnchor="middle" fill={P.ink} fontSize={34}
                  fontFamily={FONT_HEAD} fontWeight={700} opacity={deadK}>干完活，留在果里</text>
              )}
            </g>
          </g>
          <text x={1140} y={262} fill={P.skinH} fontSize={30} fontFamily={FONT_HEAD} letterSpacing={4}>果孔 · 放大看</text>
        </g>
      </svg>
      <Chip text="榕小蜂 · 只有两毫米" x={140} y={150} at={40} accent={P.mint} />
      <Stamp text="硬挤进门！" x={240} y={820} at={170} color={P.coral} size={48} rot={-8} />
      <Stamp text="翅膀挤断" x={1170} y={745} at={315} color={P.fleshDD} size={44} rot={5} />
      <Chip text="边爬边授粉、边产卵" x={1090} y={132} at={b2 + 25} accent={P.amber} />
      <Stamp text="有去无回" x={1560} y={745} at={590} color={P.skinH} size={44} rot={4} />
    </FadeIn>
  );
};

/* ============ S3 后事：果膛内，蛋白酶液滴把死蜂溶成养分；脆粒=种子 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);

  const dissK = clamp01((f - 195) / 75);
  const ghostO = clamp01((f - 335) / 20) * 0.55;
  const CX = 960, CY = 545, R = 400;

  return (
    <FadeIn>
      <Backdrop top={P.bgFlesh} bottom={P.bgFlesh} />
      <SceneTag index={3} name={sec.name} accent={P.fleshD} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <circle cx={CX} cy={CY} r={R} fill={P.flesh} />
        <circle cx={CX} cy={CY} r={R * 0.82} fill={P.fleshD} />
        {/* 内壁小花（上半圈） */}
        {Array.from({length: 9}).map((_, i) => {
          const a = ((195 + i * 19) / 180) * Math.PI;
          const nx = CX + Math.cos(a) * R * 0.74, ny = CY + Math.sin(a) * R * 0.74;
          return (
            <g key={i} transform={`translate(${nx},${ny}) rotate(${(a * 180) / Math.PI + 90})`}>
              <ellipse rx={16} ry={9} fill={P.flower} />
            </g>
          );
        })}
        {/* 膛底 */}
        <ellipse cx={CX} cy={845} rx={295} ry={50} fill={P.fleshDD} opacity={0.55} />
        {/* 蛋白酶液滴 */}
        {Array.from({length: 9}).map((_, i) => (
          <EnzymeDrop key={i} x={680 + rand(i, 7) * 560} y0={250} y1={795 + rand(i, 8) * 30}
            at={b1 + 5 + i * 16} dur={42} />
        ))}
        {/* 死蜂：翅膀残缺，渐渐被溶掉 */}
        <g opacity={1 - dissK}>
          <Wasp x={830} y={805} s={1.05} rot={175} wings={0.15} />
        </g>
        <Burst x={830} y={805} at={205} col={P.amber} n={12} />
        <Burst x={830} y={805} at={228} col={P.waspH} n={8} />
        {/* 被溶后的"痕迹" */}
        <ellipse cx={830} cy={812} rx={52} ry={22} fill="none" stroke={P.cream}
          strokeWidth={3} strokeDasharray="9 9" opacity={ghostO} />
        {/* 脆粒=种子 */}
        {Array.from({length: 7}).map((_, i) => {
          const sx = 620 + i * 105 + rand(i, 9) * 20;
          const sy = 792 + rand(i, 10) * 42;
          const sp = spring({frame: f - b2 - 14 - i * 7, fps, config: {damping: 9, stiffness: 170}});
          return (
            <g key={i} transform={`translate(${sx},${sy}) rotate(${(rand(i, 11) - 0.5) * 50}) scale(${Math.max(0.001, sp)})`}>
              <ellipse rx={15} ry={9} fill={P.flower} />
              <ellipse rx={5.5} ry={3.5} fill={P.seedD} />
            </g>
          );
        })}
      </svg>
      <Chip text="无花果蛋白酶" x={140} y={150} at={b1 + 8} accent={P.amber} />
      <Chip text="吃到的只是痕迹" x={140} y={292} at={b2 + 20} accent={P.cream} />
      <Stamp text="几天溶成氨基酸" x={1330} y={135} at={b1 + 150} color={P.amber} size={44} rot={5} />
      <Stamp text="连渣都不剩" x={225} y={700} at={b1 + 265} color={P.coral} size={46} rot={-6} />
      <Stamp text="脆粒是种子！" x={1430} y={700} at={b2 + 65} color={P.skinH} size={44} rot={4} />
    </FadeIn>
  );
};

/* ============ S4 安心：鲜食款 0 蜂 vs 制干款靠蜂 + 六千万年契约 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);

  const cardL = spring({frame: f - 22, fps, config: {damping: 13, stiffness: 120}});
  const cardR = spring({frame: f - 250, fps, config: {damping: 13, stiffness: 120}});
  const checkK = spring({frame: f - 178, fps, config: {damping: 9, stiffness: 170}});
  const waspK = spring({frame: f - 288, fps, config: {damping: 10, stiffness: 160}});
  const barK = clamp01((f - b1 - 5) / 18);
  const figK = spring({frame: f - b1 - 16, fps, config: {damping: 10, stiffness: 160}});
  const wK = spring({frame: f - b1 - 26, fps, config: {damping: 10, stiffness: 160}});

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={4} name={sec.name} accent={P.mint} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左卡：超市鲜食款 */}
        <g transform={`translate(0,${(1 - clamp01(cardL)) * 60})`} opacity={clamp01(cardL)}>
          <rect x={250} y={215} width={460} height={515} rx={36} fill={P.card} stroke={P.skinH} strokeWidth={4} />
          <text x={480} y={295} textAnchor="middle" fill={P.cream} fontSize={42} fontFamily={FONT_HEAD} fontWeight={700}>超市鲜食款</text>
          <FigWhole x={480} y={505} s={1.25} />
          <text x={480} y={672} textAnchor="middle" fill={P.mint} fontSize={36} fontFamily={FONT_HEAD} fontWeight={700}>全程 0 蜂参与</text>
          <text x={480} y={712} textAnchor="middle" fill={P.cream} fontSize={26} fontFamily={FONT_HEAD} opacity={0.75}>培育出的单性结实</text>
          <g transform={`translate(635,385) scale(${Math.max(0.001, checkK)})`}>
            <circle r={55} fill={P.leaf} />
            <path d="M -24 0 L -8 16 L 26 -18" stroke="#fff" strokeWidth={10} fill="none" strokeLinecap="round" strokeLinejoin="round" />
          </g>
        </g>
        {/* 右卡：制干品种 */}
        <g transform={`translate(0,${(1 - clamp01(cardR)) * 60})`} opacity={clamp01(cardR)}>
          <rect x={1210} y={215} width={460} height={515} rx={36} fill={P.card} stroke={P.skinH} strokeWidth={4} />
          <text x={1440} y={295} textAnchor="middle" fill={P.cream} fontSize={42} fontFamily={FONT_HEAD} fontWeight={700}>制干 Smyrna 系</text>
          <FigWhole x={1440} y={505} s={1.25} tint="#a9714b" />
          <g transform={`translate(1440,378) scale(${Math.max(0.001, waspK)})`}>
            <Wasp x={0} y={0} s={0.62} rot={-16} />
          </g>
          <text x={1440} y={672} textAnchor="middle" fill={P.amber} fontSize={36} fontFamily={FONT_HEAD} fontWeight={700}>必须靠蜂传粉</text>
          <text x={1440} y={712} textAnchor="middle" fill={P.cream} fontSize={26} fontFamily={FONT_HEAD} opacity={0.75}>果园里挂果养蜂</text>
        </g>
        {/* 契约条 */}
        <g transform={`translate(0,${(1 - barK) * 50})`} opacity={barK}>
          <rect x={395} y={762} width={1130} height={112} rx={56} fill={P.plate} />
          <g transform={`translate(545,815) scale(${Math.max(0.001, figK)})`}>
            <g transform="scale(0.42)">
              <FigWhole x={0} y={0} s={1} />
            </g>
          </g>
          <g transform={`translate(1390,815) scale(${Math.max(0.001, wK)})`}>
            <Wasp x={0} y={0} s={0.9} rot={-8} />
          </g>
          <line x1={645} y1={845} x2={1335} y2={845} stroke={P.ink} strokeWidth={5} />
          <path d="M 672 833 L 648 845 L 672 857" stroke={P.ink} strokeWidth={5} fill="none" strokeLinecap="round" strokeLinejoin="round" />
          <path d="M 1308 833 L 1332 845 L 1308 857" stroke={P.ink} strokeWidth={5} fill="none" strokeLinecap="round" strokeLinejoin="round" />
          <text x={990} y={808} textAnchor="middle" fill={P.ink} fontSize={38} fontFamily={FONT_HEAD} fontWeight={700}>
            一种果配一种蜂 · 六千万年契约
          </text>
        </g>
      </svg>
      <Chip text="很多品种不授粉也结果" x={140} y={145} at={50} accent={P.mint} />
      <Stamp text="单性结实" x={800} y={170} at={195} color={P.leaf} size={46} rot={6} />
      <Stamp text="要蜂打工" x={870} y={430} at={298} color={P.amber} size={44} rot={-5} />
      <Stamp text="教科书级共生" x={1490} y={140} at={b1 + 55} color={P.coral} size={46} rot={4} />
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
        <ellipse cx={960} cy={1015} rx={1050} ry={120} fill="#241022" />
        <circle cx={1040} cy={915} r={180} fill={P.plate} />
        <FigCut x={1040} y={915} r={135} pop={pop} flowerK={1} />
        <FigWhole x={800} y={955} s={0.8} tilt={-10} pop={pop} />
        <Wasp x={1205} y={985} s={0.42} rot={-18} wings={0.6} />
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -200}}>
        <div style={{fontSize: 70, fontWeight: 800, color: C.text}}>清甜背后，是六千万年的合作</div>
        <div style={{fontSize: 38, color: C.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E209 = ({data = tl}: {data?: any}) => {
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
