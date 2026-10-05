import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01, rand} from '../lib/style';
import {Backdrop, Stamp, SceneTag, Chip, Subtitles, ProgressBar, FadeIn} from '../lib/atoms';
import tl from '../timeline_246.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：深夜健身房 · 蓝灰 × 皮带青 × 跑道橙 ============ */
const P = {
  bg: '#232b40',
  bgDeep: '#1a2133',
  bgLite: '#2c3654',
  panel: '#2a3450',
  card: '#3a4670',
  ink: '#121a30',
  cream: '#f5eeda',
  muted: '#a4aed2',
  orange: '#ffa14f',
  orangeD: '#e07c34',
  belt: '#5fe0b1',
  beltD: '#38b389',
  sky: '#6fc6f2',
  skyD: '#3f93c4',
  coral: '#ff6f61',
  coralD: '#d84f43',
  amber: '#ffc25c',
  heat: '#ff8d6e',
  road: '#4a5470',
};

const wrap = (v: number, m: number) => ((v % m) + m) % m;

/* ---------- 面板标题 ---------- */
const PanelTitle = ({x, y, text, col = P.muted}: {x: number; y: number; text: string; col?: string}) => (
  <text x={x} y={y} fill={col} fontSize={25} fontFamily={FONT_HEAD}>{text}</text>
);

/* ---------- 对勾 / 叉 ---------- */
const Check = ({x, y, s = 1, col = P.belt}: {x: number; y: number; s?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} stroke={col} strokeWidth={8}
    strokeLinecap="round" strokeLinejoin="round" fill="none">
    <path d="M -15 1 L -5 12 L 17 -12" />
  </g>
);
const MarkX = ({x, y, s = 1, col = P.coral}: {x: number; y: number; s?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} stroke={col} strokeWidth={7} strokeLinecap="round">
    <line x1={-12} y1={-12} x2={12} y2={12} /><line x1={12} y1={-12} x2={-12} y2={12} />
  </g>
);

/* ---------- 圆角标签牌 ---------- */
const TagPill = ({x, y, t, col, s = 1, w = 210, fs = 29}: {
  x: number; y: number; t: string; col: string; s?: number; w?: number; fs?: number;
}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-w / 2} y={-30} width={w} height={60} rx={16} fill={P.panel} stroke={col} strokeWidth={4} />
    <text x={0} y={11} textAnchor="middle" fontSize={fs} fill={col}
      fontFamily={FONT_HEAD} fontWeight={800}>{t}</text>
  </g>
);

/* ---------- 跑步小人（面向右，脚底 y=0，两腿反向摆动） ---------- */
const Runner = ({x, y, s = 1, jersey = P.orange, shorts = P.beltD, wig = 0, spd = 0.5, flip = 1, frozen}: {
  x: number; y: number; s?: number; jersey?: string; shorts?: string; wig?: number;
  spd?: number; flip?: number; frozen?: number;
}) => {
  const f = useCurrentFrame();
  const ph = frozen !== undefined ? frozen : f * spd + wig;
  const qF = Math.sin(ph);
  const qB = -qF;
  const bob = Math.abs(Math.cos(ph)) * 6;
  const leg = (q: number, back: boolean) => {
    const fx = 62 * q;
    const fy = -6 - 34 * Math.max(0, q);
    const kx = 28 * q + 10;
    const ky = -56 - 18 * Math.max(0, q);
    return (
      <g>
        <path d={`M 0 -116 L ${kx} ${ky} L ${fx} ${fy}`} fill="none"
          stroke={back ? '#1d2540' : P.ink} strokeWidth={24}
          strokeLinecap="round" strokeLinejoin="round" />
        <ellipse cx={fx + 8} cy={fy + 6} rx={22} ry={10} fill={P.cream}
          transform={`rotate(${-20 * Math.max(0, q)} ${fx + 8} ${fy + 6})`} />
      </g>
    );
  };
  return (
    <g transform={`translate(${x},${y}) scale(${flip * s},${s})`}>
      <ellipse cx={8} cy={8} rx={96} ry={11} fill={P.ink} opacity={0.3} />
      <g transform={`translate(0,${-bob})`}>
        {/* 后臂 */}
        <path d={`M -4 -182 L ${-38 + 14 * qF} -158 L ${-54 + 30 * qF} -124`}
          fill="none" stroke={C.skinDark} strokeWidth={17} strokeLinecap="round" strokeLinejoin="round" />
        {leg(qB, true)}
        {/* 躯干 + 短裤（前倾） */}
        <g transform="rotate(12 0 -160)">
          <rect x={-34} y={-222} width={72} height={104} rx={28} fill={jersey} />
          <rect x={-36} y={-136} width={78} height={30} rx={15} fill={shorts} />
        </g>
        {leg(qF, false)}
        {/* 前臂 */}
        <path d={`M 8 -186 L ${40 - 12 * qF} -160 L ${26 - 32 * qF} -126`}
          fill="none" stroke={C.skin} strokeWidth={17} strokeLinecap="round" strokeLinejoin="round" />
        <circle cx={26 - 32 * qF} cy={-124} r={11} fill={C.skin} />
        {/* 头 */}
        <g transform="translate(44,-246)">
          <circle r={30} fill={C.skin} />
          <path d="M -30 -4 Q -24 -40 8 -36 Q 34 -32 31 -2 Q 0 -24 -30 -4 Z" fill={P.ink} />
          <circle cx={13} cy={-2} r={4.4} fill={P.ink} />
        </g>
      </g>
    </g>
  );
};

/* ---------- 跑步机（面向右 = 控制台在右；tilt>0 前端抬起；children 画在台面坐标系里） ---------- */
const Treadmill = ({x, y, s = 1, tilt = 0, children}: {
  x: number; y: number; s?: number; tilt?: number; children?: React.ReactNode;
}) => {
  const f = useCurrentFrame();
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      <ellipse cx={6} cy={14} rx={262} ry={14} fill={P.ink} opacity={0.3} />
      <g transform={`rotate(${-tilt} -190 -4)`}>
        {/* 跑台边框 */}
        <rect x={-200} y={-58} width={424} height={54} rx={24} fill={P.card} stroke={P.muted} strokeWidth={4} />
        {/* 皮带槽 */}
        <rect x={-188} y={-52} width={392} height={22} rx={11} fill={P.ink} />
        {/* 皮带滚纹（向后跑 = 向左） */}
        {Array.from({length: 7}).map((_, i) => {
          const px = 168 - wrap(i * 58 + f * 3.2, 350);
          return <rect key={i} x={px} y={-47} width={24} height={12} rx={6} fill={P.belt} opacity={0.9} />;
        })}
        {/* 尾辊 */}
        <circle cx={-186} cy={-30} r={15} fill={P.card} stroke={P.muted} strokeWidth={3} />
        {/* 支脚 */}
        <rect x={-206} y={-12} width={36} height={16} rx={8} fill={P.card} />
        <rect x={184} y={-12} width={36} height={16} rx={8} fill={P.card} />
        {/* 前立柱 + 扶手 */}
        <path d="M 184 -54 L 216 -196" fill="none" stroke={P.card} strokeWidth={20} strokeLinecap="round" />
        <line x1={96} y1={-140} x2={206} y2={-172} stroke={P.card} strokeWidth={13} strokeLinecap="round" />
        {/* 控制台 */}
        <rect x={168} y={-238} width={132} height={56} rx={14} fill={P.panel} stroke={P.muted} strokeWidth={4} />
        <rect x={182} y={-227} width={64} height={17} rx={7} fill={P.belt} opacity={0.9} />
        <circle cx={274} cy={-218} r={8} fill={P.coral} />
        {children}
      </g>
    </g>
  );
};

/* ---------- 风 streak：从右向左（顶风） ---------- */
const Wind = ({x, y, w = 300, n = 5, gap = 34, on = 1, spd = 8, col = P.sky}: {
  x: number; y: number; w?: number; n?: number; gap?: number; on?: number; spd?: number; col?: string;
}) => {
  const f = useCurrentFrame();
  if (on <= 0) return null;
  return (
    <g opacity={on}>
      {Array.from({length: n}).map((_, i) => {
        const px = x + w - wrap(i * 97 + f * spd, w + 110);
        const py = y + i * gap;
        const fade = Math.min(clamp01((px - x) / 50), clamp01((x + w - px) / 50));
        return (
          <g key={i} opacity={0.8 * fade}>
            <line x1={px} y1={py} x2={px - 62} y2={py} stroke={col} strokeWidth={7} strokeLinecap="round" />
            <line x1={px - 54} y1={py} x2={px - 70} y2={py - 9} stroke={col} strokeWidth={7} strokeLinecap="round" />
            <line x1={px - 54} y1={py} x2={px - 70} y2={py + 9} stroke={col} strokeWidth={7} strokeLinecap="round" />
          </g>
        );
      })}
    </g>
  );
};

/* ---------- 静止空气：几乎不动的小点 ---------- */
const StillAir = ({x, y, w, n = 6, col = P.muted}: {x: number; y: number; w: number; n?: number; col?: string}) => {
  const f = useCurrentFrame();
  return (
    <g>
      {Array.from({length: n}).map((_, i) => (
        <circle key={i} cx={x + rand(i, 31) * w + Math.sin(f * 0.1 + i * 1.7) * 3}
          cy={y + rand(i, 47) * 120} r={4} fill={col} opacity={0.5} />
      ))}
    </g>
  );
};

/* ---------- 热气飘丝（dir=-1 向左飘走） ---------- */
const HeatWisp = ({x, y, on = 1, dir = -1, drift = 90, col = P.heat, n = 3, ph = 0}: {
  x: number; y: number; on?: number; dir?: 1 | -1; drift?: number; col?: string; n?: number; ph?: number;
}) => {
  const f = useCurrentFrame();
  if (on <= 0) return null;
  return (
    <g opacity={on}>
      {Array.from({length: n}).map((_, i) => {
        const t = wrap(f * 1.4 + i * 22 + ph, 72);
        const p = t / 72;
        const wx = x + dir * p * drift + Math.sin(p * 6) * 8;
        const wy = y - i * 8 - p * 14;
        return (
          <path key={i} d={`M ${wx} ${wy} q 10 -10 0 -20 q -10 -10 0 -20`} fill="none"
            stroke={col} strokeWidth={6} strokeLinecap="round" opacity={0.85 * (1 - p)} />
        );
      })}
    </g>
  );
};

/* ---------- 汗滴 ---------- */
const Sweat = ({x, y, s = 1}: {x: number; y: number; s?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <path d="M 0 -14 Q 11 2 11 9 A 11 11 0 0 1 -11 9 Q -11 2 0 -14 Z" fill={P.sky} />
  </g>
);

/* ---------- 曲线工具（chaikin 平滑 + 逐段画出） ---------- */
const chaikin = (pts: [number, number][], n = 2): [number, number][] => {
  let p = pts;
  for (let k = 0; k < n; k++) {
    const q: [number, number][] = [p[0]];
    for (let i = 0; i < p.length - 1; i++) {
      const [ax, ay] = p[i];
      const [bx, by] = p[i + 1];
      q.push([ax * 0.72 + bx * 0.28, ay * 0.72 + by * 0.28]);
      q.push([ax * 0.28 + bx * 0.72, ay * 0.28 + by * 0.72]);
    }
    q.push(p[p.length - 1]);
    p = q;
  }
  return p;
};
const Curve = ({pts, prog, col, lw = 6, dash}: {
  pts: [number, number][]; prog: number; col: string; lw?: number; dash?: string;
}) => {
  const sm = chaikin(pts);
  const uu = clamp01(prog) * (sm.length - 1);
  const seg = Math.floor(uu);
  const out = sm.slice(0, seg + 1);
  if (seg < sm.length - 1) {
    const [ax, ay] = sm[seg];
    const [bx, by] = sm[seg + 1];
    const t = uu - seg;
    out.push([ax + (bx - ax) * t, ay + (by - ay) * t]);
  }
  if (out.length < 2) return null;
  return (
    <polyline points={out.map((p) => p.join(',')).join(' ')} fill="none" stroke={col}
      strokeWidth={lw} strokeLinecap="round" strokeLinejoin="round" strokeDasharray={dash} />
  );
};

/* ============ 标题卡：跑步机 + 跑者 + 静止空气 ============ */
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
        {/* 顶部装饰小点 */}
        {Array.from({length: 9}).map((_, i) => (
          <circle key={i} cx={140 + i * 205} cy={84 + (i % 2) * 24} r={4} fill={P.cream} opacity={0.28} />
        ))}
        <g transform={`translate(960,590) scale(${pA}) translate(-960,-590)`} opacity={clamp01(pA)}>
          <Treadmill x={960} y={590} s={1.28}>
            <Runner x={-8} y={-56} s={1.02} spd={0.5} />
          </Treadmill>
          <StillAir x={620} y={300} w={640} n={7} col={P.sky} />
        </g>
        <g transform={`translate(545,335) scale(${pB})`} opacity={clamp01(pB)}>
          <TagPill x={0} y={0} t="5km 同配速" col={P.belt} w={240} />
        </g>
        <g transform={`translate(1380,335) scale(${pB})`} opacity={clamp01(pB)}>
          <TagPill x={0} y={0} t="却轻松一截" col={P.orange} w={230} />
        </g>
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 500}}>
          <div style={{color: P.belt, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 24}}>
            冷知识 · 第 {data.episode} 期
          </div>
          <div style={{color: C.text, fontSize: 72, fontWeight: 900, fontFamily: FONT_HEAD, letterSpacing: 4}}>
            {data.title}
          </div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ============ S0 省力不是错觉：同配速对照 + 四笔暗补账单 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const popL = Math.max(0.001, spring({frame: f - 8, fps, config: {damping: 11, stiffness: 120}}));
  const popR = Math.max(0.001, spring({frame: f - 24, fps, config: {damping: 11, stiffness: 120}}));
  const itemK = [0, 1, 2, 3].map((i) =>
    Math.max(0.001, spring({frame: f - b1 - 6 - i * 15, fps, config: {damping: 12, stiffness: 140}})));
  const noteK = clamp01((f - b1 - 75) / 24);

  const items = [
    {col: P.sky, label: '风阻 → 0', sub: '越快省得越多'},
    {col: P.belt, label: '皮带帮收腿', sub: '腘绳肌少出力'},
    {col: P.amber, label: '地形永远平', sub: '坡坑弯道清零'},
    {col: P.orange, label: '恒温无风', sub: '环境变量砍光'},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={0} name={sec.name} accent={P.belt} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 上左：跑步机 */}
        <g transform={`translate(535,315) scale(${popL}) translate(-535,-315)`} opacity={clamp01(popL * 2)}>
          <rect x={120} y={140} width={830} height={350} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <PanelTitle x={160} y={196} text="跑步机 · 同配速 5km" />
          <Treadmill x={560} y={452} s={0.66}>
            <Runner x={-8} y={-56} s={0.72} spd={0.5} />
          </Treadmill>
          <StillAir x={220} y={240} w={620} n={6} />
          <text x={170} y={462} fontSize={23} fill={P.belt} fontFamily={FONT_HEAD} fontWeight={800}>
            下来腿不沉
          </text>
        </g>
        {/* 上右：路跑 */}
        <g transform={`translate(1385,315) scale(${popR}) translate(-1385,-315)`} opacity={clamp01(popR * 2)}>
          <rect x={970} y={140} width={830} height={350} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <PanelTitle x={1010} y={196} text="路跑 · 同配速 5km" />
          <rect x={1010} y={424} width={750} height={24} rx={12} fill={P.road} />
          {[0, 1, 2, 3].map((i) => (
            <rect key={i} x={1050 + i * 180} y={432} width={70} height={8} rx={4} fill={P.muted} opacity={0.5} />
          ))}
          <Runner x={1340} y={424} s={0.88} spd={0.5} wig={2} jersey={P.sky} shorts={P.card} />
          <Wind x={1030} y={246} w={640} n={5} gap={34} spd={9} />
          <text x={1010} y={462} fontSize={23} fill={P.coral} fontFamily={FONT_HEAD} fontWeight={800}>
            顶风 · 更乏一点
          </text>
        </g>

        {/* 下：省力账单 */}
        <rect x={120} y={520} width={1680} height={360} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <PanelTitle x={160} y={576} text="省力账单 · 不是错觉，是好几笔叠出来的" col={P.cream} />
        {items.map((it, i) => {
          const cx = 360 + i * 400;
          const k = itemK[i];
          return (
            <g key={i} transform={`translate(${cx},700) scale(${k})`} opacity={clamp01(k * 2)}>
              <rect x={-180} y={-82} width={360} height={164} rx={20} fill={P.bgDeep} stroke={it.col} strokeWidth={4} />
              {/* 图标 */}
              {i === 0 && (
                <g transform="translate(-108,0)">
                  <line x1={-30} y1={-18} x2={18} y2={-18} stroke={it.col} strokeWidth={7} strokeLinecap="round" />
                  <line x1={-38} y1={2} x2={26} y2={2} stroke={it.col} strokeWidth={7} strokeLinecap="round" />
                  <line x1={-30} y1={22} x2={10} y2={22} stroke={it.col} strokeWidth={7} strokeLinecap="round" />
                  <line x1={-38} y1={-34} x2={38} y2={38} stroke={P.coral} strokeWidth={8} strokeLinecap="round" />
                </g>
              )}
              {i === 1 && (
                <g transform="translate(-108,8)">
                  <rect x={-44} y={-6} width={88} height={18} rx={9} fill={P.ink} />
                  {[-26, 0, 26].map((dx) => (
                    <rect key={dx} x={dx - 8} y={-2} width={16} height={9} rx={4.5} fill={it.col} />
                  ))}
                  <ellipse cx={-6} cy={-16} rx={30} ry={11} fill={P.cream} />
                </g>
              )}
              {i === 2 && (
                <g transform="translate(-108,4)">
                  <line x1={-46} y1={14} x2={46} y2={14} stroke={it.col} strokeWidth={8} strokeLinecap="round" />
                  <path d="M -40 -8 Q -28 -22 -16 -8 Q -6 4 4 -8 Q 16 -22 28 -8" fill="none"
                    stroke={P.muted} strokeWidth={5} strokeLinecap="round" opacity={0.6} />
                </g>
              )}
              {i === 3 && (
                <g transform="translate(-108,0)">
                  <rect x={-12} y={-34} width={24} height={52} rx={12} fill="none" stroke={it.col} strokeWidth={6} />
                  <rect x={-5} y={-6} width={10} height={22} rx={5} fill={it.col} />
                  <circle cy={24} r={14} fill={it.col} />
                </g>
              )}
              <text x={28} y={-14} textAnchor="middle" fontSize={26} fill={it.col}
                fontFamily={FONT_HEAD} fontWeight={900}>{it.label}</text>
              <text x={28} y={28} textAnchor="middle" fontSize={20} fill={P.muted}
                fontFamily={FONT_HEAD}>{it.sub}</text>
            </g>
          );
        })}
        {noteK > 0 && (
          <text x={960} y={852} textAnchor="middle" fontSize={24} fill={P.muted}
            fontFamily={FONT_HEAD} opacity={noteK}>
            单项都不算大 · 叠起来就是那一截轻松
          </text>
        )}
      </svg>
      <Chip text="同配速 · 两本账" x={1450} y={66} at={30} accent={P.orange} />
      <Stamp text="不是错觉" x={540} y={86} at={b1 + 30} color={P.beltD} size={42} rot={-3} />
    </FadeIn>
  );
};

/* ============ S1 风阻归零：顶风 vs 原地 + 风阻∝v² 曲线 ============ */
const DRAG_PTS: [number, number][] = [
  [300, 772], [510, 756], [720, 737], [930, 714], [1140, 688], [1350, 658], [1560, 625],
];

const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);
  const popL = Math.max(0.001, spring({frame: f - 8, fps, config: {damping: 11, stiffness: 120}}));
  const popR = Math.max(0.001, spring({frame: f - b2 - 10, fps, config: {damping: 11, stiffness: 120}}));
  const curveK = clamp01((f - b1 - 4) / 80);
  const mkA = clamp01((curveK - 0.25) * 4);
  const mkB = clamp01((curveK - 0.82) * 5);

  return (
    <FadeIn>
      <Backdrop top={P.bgDeep} bottom={P.bgDeep} />
      <SceneTag index={1} name={sec.name} accent={P.sky} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：路跑顶风 */}
        <g transform={`translate(535,355) scale(${popL}) translate(-535,-355)`} opacity={clamp01(popL * 2)}>
          <rect x={120} y={140} width={830} height={430} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <PanelTitle x={160} y={196} text="路跑 · 身体在以跑步速度撞风" />
          <rect x={160} y={492} width={750} height={24} rx={12} fill={P.road} />
          {[0, 1, 2, 3].map((i) => (
            <rect key={i} x={200 + i * 180} y={500} width={70} height={8} rx={4} fill={P.muted} opacity={0.5} />
          ))}
          <Runner x={500} y={490} s={1.05} spd={0.55} jersey={P.sky} shorts={P.card} />
          <Wind x={170} y={250} w={680} n={6} gap={34} spd={10} />
          {/* 风阻箭头压在胸口 */}
          <line x1={640} y1={292} x2={540} y2={292} stroke={P.coral} strokeWidth={10} strokeLinecap="round" />
          <polygon points="524,292 546,280 546,304" fill={P.coral} />
          <g>
            <rect x={648} y={252} width={240} height={58} rx={15} fill={P.bgDeep} stroke={P.coral} strokeWidth={4} />
            <text x={768} y={292} textAnchor="middle" fontSize={27} fill={P.coral}
              fontFamily={FONT_HEAD} fontWeight={900}>风阻 F ∝ v²</text>
          </g>
          <text x={160} y={550} fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>
            配速越快 · 这堵空气墙涨得越猛
          </text>
        </g>
        {/* 右：跑步机原地 */}
        <g transform={`translate(1385,355) scale(${popR}) translate(-1385,-355)`} opacity={clamp01(popR * 2)}>
          <rect x={970} y={140} width={830} height={430} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <PanelTitle x={1010} y={196} text="跑步机 · 皮带在脚下跑 · 人原地" />
          <Treadmill x={1390} y={470} s={0.8}>
            <Runner x={-8} y={-56} s={0.86} spd={0.55} wig={1} />
          </Treadmill>
          <StillAir x={1030} y={240} w={700} n={7} />
          <g transform="translate(1148,268)">
            <rect x={-150} y={-34} width={300} height={66} rx={17} fill={P.bgDeep} stroke={P.sky} strokeWidth={4} />
            <text x={0} y={12} textAnchor="middle" fontSize={36} fill={P.sky}
              fontFamily={FONT_HEAD} fontWeight={900}>风阻 ≈ 0</text>
          </g>
          <text x={1010} y={550} fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>
            空气相对静止 · 这笔支出直接归零
          </text>
        </g>

        {/* 下：风阻占能耗曲线 */}
        <rect x={120} y={600} width={1680} height={280} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <PanelTitle x={160} y={656} text="风阻占能耗比 · 随速度平方往上蹿" col={P.cream} />
        <line x1={300} y1={640} x2={300} y2={810} stroke={P.muted} strokeWidth={3.4} />
        <line x1={300} y1={810} x2={1660} y2={810} stroke={P.muted} strokeWidth={3.4} />
        <text x={272} y={652} textAnchor="end" fontSize={20} fill={P.muted} fontFamily={FONT_HEAD}>%</text>
        {[[510, '10'], [930, '14'], [1350, '18']].map(([tx, t]) => (
          <g key={t as string}>
            <line x1={tx as number} y1={810} x2={tx as number} y2={822} stroke={P.muted} strokeWidth={3} />
            <text x={tx as number} y={850} textAnchor="middle" fontSize={21} fill={P.muted}
              fontFamily={FONT_HEAD}>{t}</text>
          </g>
        ))}
        <text x={1660} y={850} textAnchor="end" fontSize={21} fill={P.muted} fontFamily={FONT_HEAD}>
          配速 km/h →
        </text>
        <Curve pts={DRAG_PTS} prog={curveK} col={P.sky} lw={7} />
        {mkA > 0 && (
          <g opacity={mkA}>
            <circle cx={510} cy={756} r={10} fill={P.sky} stroke={P.bgDeep} strokeWidth={4} />
            <rect x={540} y={714} width={230} height={50} rx={14} fill={P.bgDeep} stroke={P.sky} strokeWidth={3.6} />
            <text x={655} y={748} textAnchor="middle" fontSize={24} fill={P.sky}
              fontFamily={FONT_HEAD} fontWeight={800}>10km/h ≈ 2~3%</text>
          </g>
        )}
        {mkB > 0 && (
          <g opacity={mkB}>
            <circle cx={1350} cy={658} r={10} fill={P.coral} stroke={P.bgDeep} strokeWidth={4} />
            <rect x={1120} y={612} width={230} height={50} rx={14} fill={P.bgDeep} stroke={P.coral} strokeWidth={3.6} />
            <text x={1235} y={646} textAnchor="middle" fontSize={24} fill={P.coral}
              fontFamily={FONT_HEAD} fontWeight={800}>18km/h ≈ 8%</text>
          </g>
        )}
      </svg>
      <Chip text="风阻 ∝ 速度²" x={1500} y={66} at={20} accent={P.sky} />
      <Stamp text="这项支出归零" x={540} y={86} at={b2 + 30} color={P.skyD} size={40} rot={-3} />
    </FadeIn>
  );
};

/* ============ S2 皮带帮你收腿：步态两份工 + 皮带拖拽 + 做功对比 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const popA = Math.max(0.001, spring({frame: f - 8, fps, config: {damping: 11, stiffness: 120}}));
  const popB = Math.max(0.001, spring({frame: f - 30, fps, config: {damping: 11, stiffness: 120}}));
  const beltK = clamp01((f - b1 - 4) / 30);
  const barK = clamp01((f - b1 - 30) / 40);
  const dotsK = clamp01((f - b1 - 70) / 30);

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={2} name={sec.name} accent={P.belt} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：一步里的两份工 */}
        <rect x={120} y={140} width={1080} height={740} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <PanelTitle x={160} y={196} text="一步两份工 · 支撑蹬地 + 摆动收腿" />
        <line x1={660} y1={250} x2={660} y2={470} stroke={P.card} strokeWidth={3} strokeDasharray="6 10" />
        {/* 支撑期：后腿蹬地 */}
        <g transform={`translate(390,440) scale(${popA}) translate(-390,-440)`} opacity={clamp01(popA * 2)}>
          <line x1={230} y1={440} x2={560} y2={440} stroke={P.muted} strokeWidth={4} strokeLinecap="round" />
          <Runner x={390} y={440} s={0.92} frozen={4.26} jersey={P.orange} />
          {/* 蹬地箭头 */}
          <line x1={352} y1={414} x2={296} y2={444} stroke={P.amber} strokeWidth={9} strokeLinecap="round" />
          <polygon points="286,452 312,446 306,422" fill={P.amber} />
          <text x={300} y={476} fontSize={22} fill={P.amber} fontFamily={FONT_HEAD} fontWeight={800}>
            支撑期 · 蹬地推进
          </text>
        </g>
        {/* 摆动期：前腿收腿 + 肌肉高亮 */}
        <g transform={`translate(890,440) scale(${popB}) translate(-890,-440)`} opacity={clamp01(popB * 2)}>
          <line x1={730} y1={440} x2={1060} y2={440} stroke={P.muted} strokeWidth={4} strokeLinecap="round" />
          <Runner x={890} y={440} s={0.92} frozen={1.35} jersey={P.orange} />
          {/* 腘绳肌高亮（摆动腿大腿后侧） */}
          <line x1={896} y1={330} x2={920} y2={368} stroke={P.coral} strokeWidth={13} strokeLinecap="round" />
          {/* 髋屈肌高亮 */}
          <line x1={876} y1={322} x2={894} y2={348} stroke={P.amber} strokeWidth={11} strokeLinecap="round" />
          <line x1={930} y1={352} x2={1010} y2={330} stroke={P.coral} strokeWidth={3} strokeDasharray="4 6" />
          <text x={1020} y={324} fontSize={20} fill={P.coral} fontFamily={FONT_HEAD} fontWeight={800}>腘绳肌</text>
          <line x1={884} y1={316} x2={1000} y2={288} stroke={P.amber} strokeWidth={3} strokeDasharray="4 6" />
          <text x={1010} y={284} fontSize={20} fill={P.amber} fontFamily={FONT_HEAD} fontWeight={800}>髋屈肌</text>
          <text x={800} y={476} fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>
            摆动期 · 主动收腿 = 肌肉做功
          </text>
        </g>

        {/* 左下：皮带在后撤 */}
        {beltK > 0 && (
          <g opacity={beltK} transform={`translate(0,${(1 - beltK) * 26})`}>
            <rect x={160} y={530} width={1000} height={320} rx={18} fill={P.bgDeep} stroke={P.belt} strokeWidth={4} />
            <PanelTitle x={196} y={586} text="跑步机皮带在后撤 → 帮你把腿带回来" col={P.belt} />
            {/* 皮带特写 */}
            <rect x={196} y={650} width={720} height={44} rx={20} fill={P.card} />
            <rect x={206} y={656} width={700} height={18} rx={9} fill={P.ink} />
            {Array.from({length: 9}).map((_, i) => {
              const px = 884 - wrap(i * 76 + f * 3.4, 690);
              return <rect key={i} x={px} y={660} width={26} height={10} rx={5} fill={P.belt} />;
            })}
            {/* 脚被拖着走 */}
            <ellipse cx={430} cy={648} rx={42} ry={17} fill={P.cream} />
            <line x1={390} y1={726} x2={290} y2={726} stroke={P.belt} strokeWidth={9} strokeLinecap="round" />
            <polygon points="272,726 296,713 296,739" fill={P.belt} />
            <text x={290} y={772} fontSize={22} fill={P.belt} fontFamily={FONT_HEAD} fontWeight={800}>
              皮带把脚向后拖
            </text>
            <text x={290} y={812} fontSize={21} fill={P.muted} fontFamily={FONT_HEAD}>
              = 收腿这份工 · 它帮你干了一截
            </text>
            <text x={950} y={700} fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>
              步态周期和路跑有细微差
            </text>
            <text x={950} y={745} fontSize={23} fill={P.belt} fontFamily={FONT_HEAD} fontWeight={800}>
              腘绳肌激活度 ↓
            </text>
          </g>
        )}

        {/* 右：谁出的力 */}
        <rect x={1240} y={140} width={560} height={740} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <PanelTitle x={1280} y={196} text="收腿这份工 · 谁出的力" />
        {/* 路跑 bar */}
        <text x={1280} y={268} fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>路跑</text>
        <rect x={1280} y={286} width={460} height={52} rx={15} fill={P.coral} />
        <text x={1510} y={322} textAnchor="middle" fontSize={24} fill={P.ink}
          fontFamily={FONT_HEAD} fontWeight={900}>肌肉包圆 100%</text>
        {/* 跑步机 bar */}
        <text x={1280} y={404} fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>跑步机</text>
        <rect x={1280} y={422} width={460} height={52} rx={15} fill={P.bgDeep} stroke={P.card} strokeWidth={3} />
        <rect x={1280} y={422} width={460 * 0.62 * barK} height={52} rx={15} fill={P.belt} opacity={barK > 0 ? 1 : 0} />
        {barK > 0.9 && (
          <g>
            <rect x={1280 + 460 * 0.62} y={422} width={460 * 0.38} height={52} rx={15} fill={P.orange} />
            {[0, 1, 2, 3].map((i) => (
              <line key={i} x1={1584 + i * 38} y1={428} x2={1568 + i * 38} y2={470}
                stroke={P.orangeD} strokeWidth={6} strokeLinecap="round" />
            ))}
            <text x={1435} y={458} textAnchor="middle" fontSize={21} fill={P.ink}
              fontFamily={FONT_HEAD} fontWeight={900}>肌肉 ↓</text>
            <text x={1655} y={458} textAnchor="middle" fontSize={21} fill={P.ink}
              fontFamily={FONT_HEAD} fontWeight={900}>皮带帮</text>
          </g>
        )}
        {/* 腘绳肌激活度 */}
        <text x={1280} y={550} fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>腘绳肌激活度</text>
        <text x={1280} y={604} fontSize={21} fill={P.muted} fontFamily={FONT_HEAD}>路跑</text>
        {[0, 1, 2, 3].map((i) => (
          <circle key={i} cx={1380 + i * 46} cy={596} r={15} fill={P.coral} />
        ))}
        <text x={1280} y={658} fontSize={21} fill={P.muted} fontFamily={FONT_HEAD}>跑步机</text>
        {[0, 1].map((i) => (
          <circle key={i} cx={1380 + i * 46} cy={650} r={15} fill={P.belt} opacity={dotsK} />
        ))}
        {[2, 3].map((i) => (
          <circle key={i} cx={1380 + i * 46} cy={650} r={15} fill="none" stroke={P.muted}
            strokeWidth={3} opacity={dotsK * 0.6} />
        ))}
        {/* 小腿后侧示意 */}
        <g transform="translate(1470,745)" opacity={dotsK}>
          <path d="M -30 -70 L -16 -10 L -34 40" fill="none" stroke={P.cream} strokeWidth={20}
            strokeLinecap="round" strokeLinejoin="round" />
          <line x1={-30} y1={-64} x2={-26} y2={-6} stroke={P.coral} strokeWidth={9} strokeLinecap="round" />
          <ellipse cx={-30} cy={48} rx={24} ry={10} fill={P.cream} />
          <text x={40} y={-20} fontSize={20} fill={P.coral} fontFamily={FONT_HEAD} fontWeight={800}>腘绳肌</text>
          <text x={40} y={10} fontSize={18} fill={P.muted} fontFamily={FONT_HEAD}>大腿后侧</text>
        </g>
        <text x={1280} y={838} fontSize={20} fill={P.muted} fontFamily={FONT_HEAD}>
          量化结果不完全一致 · 但确实更省
        </text>
      </svg>
      <Chip text="传送带偷偷搭手" x={1430} y={66} at={30} accent={P.belt} />
      <Stamp text="悄悄分担" x={560} y={86} at={b1 + 40} color={P.beltD} size={42} rot={-3} />
    </FadeIn>
  );
};

/* ============ S3 平坦的另一面：变量砍光 vs 散热变差 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);
  const popL = Math.max(0.001, spring({frame: f - 8, fps, config: {damping: 11, stiffness: 120}}));
  const popR = Math.max(0.001, spring({frame: f - b1 - 4, fps, config: {damping: 11, stiffness: 120}}));
  const heatK = clamp01((f - b2 - 4) / 30);
  const icons = [
    {t: '小坡', col: P.amber},
    {t: '坑洼', col: P.muted},
    {t: '弯道', col: P.sky},
    {t: '红灯启停', col: P.coral},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bgDeep} bottom={P.bgDeep} />
      <SceneTag index={3} name={sec.name} accent={P.orange} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 上左：户外在微调 */}
        <g transform={`translate(535,330) scale(${popL}) translate(-535,-330)`} opacity={clamp01(popL * 2)}>
          <rect x={120} y={140} width={830} height={380} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <PanelTitle x={160} y={196} text="户外 · 路面从来不是平的" />
          {/* 障碍图标行 */}
          {icons.map((ic, i) => {
            const cx = 228 + i * 152;
            return (
              <g key={i} transform={`translate(${cx},268)`}>
                <rect x={-66} y={-44} width={132} height={92} rx={16} fill={P.bgDeep}
                  stroke={P.card} strokeWidth={3} />
                {i === 0 && <path d="M -30 16 L 0 -18 L 32 16 Z" fill={ic.col} />}
                {i === 1 && (
                  <path d="M -34 -4 L -12 -4 L -4 14 L 8 14 L 14 -4 L 34 -4" fill="none"
                    stroke={ic.col} strokeWidth={6} strokeLinecap="round" strokeLinejoin="round" />
                )}
                {i === 2 && (
                  <g>
                    <path d="M -28 18 Q -28 -14 4 -14 L 20 -14" fill="none" stroke={ic.col}
                      strokeWidth={6} strokeLinecap="round" />
                    <polygon points="34,-14 16,-23 16,-5" fill={ic.col} />
                  </g>
                )}
                {i === 3 && (
                  <g>
                    <rect x={-13} y={-30} width={26} height={52} rx={9} fill={P.card} />
                    <circle cy={-17} r={6} fill={P.coral} />
                    <circle cy={-4} r={6} fill={P.muted} opacity={0.5} />
                    <circle cy={9} r={6} fill={P.muted} opacity={0.5} />
                  </g>
                )}
                <text x={0} y={66} textAnchor="middle" fontSize={19} fill={P.muted}
                  fontFamily={FONT_HEAD}>{ic.t}</text>
              </g>
            );
          })}
          {/* 坑洼路 + 跑者 */}
          <path d="M 160 452 Q 200 428 240 452 Q 280 470 320 452 Q 360 434 400 452 Q 440 468 480 452 L 910 452"
            fill="none" stroke={P.road} strokeWidth={22} strokeLinecap="round" />
          <Runner x={848} y={446} s={0.72} spd={0.55} jersey={P.sky} shorts={P.card} />
          <text x={160} y={502} fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>
            每一步落地都变 · 脚踝小腿的稳定肌全程微调 = 全是能耗
          </text>
        </g>
        {/* 上右：机上砍光变量 */}
        <g transform={`translate(1385,330) scale(${popR}) translate(-1385,-330)`} opacity={clamp01(popR * 2)}>
          <rect x={970} y={140} width={830} height={380} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <PanelTitle x={1010} y={196} text="机上 · 变量砍光 · 定速" />
          <rect x={1010} y={428} width={750} height={24} rx={12} fill={P.road} />
          <Runner x={1190} y={430} s={0.8} spd={0.55} wig={2} />
          {/* 定速表 */}
          <g transform="translate(1640,310)">
            <path d="M -74 16 A 78 78 0 0 1 74 16" fill="none" stroke={P.card} strokeWidth={13} strokeLinecap="round" />
            <line x1={0} y1={16} x2={-34} y2={-40} stroke={P.belt} strokeWidth={8} strokeLinecap="round" />
            <circle cy={16} r={9} fill={P.belt} />
            <rect x={-58} y={34} width={116} height={44} rx={12} fill={P.bgDeep} stroke={P.belt} strokeWidth={3.6} />
            <text x={0} y={65} textAnchor="middle" fontSize={24} fill={P.belt}
              fontFamily={FONT_HEAD} fontWeight={900}>10.0</text>
            {/* 小锁 */}
            <g transform="translate(92,40)">
              <rect x={-14} y={-4} width={28} height={24} rx={7} fill={P.belt} />
              <path d="M -8 -4 v-7 a8 8 0 0 1 16 0 v7" fill="none" stroke={P.belt} strokeWidth={4.5} />
            </g>
          </g>
          <text x={1010} y={502} fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>
            皮带定速 · 匀速平坦恒温 · 想慢都慢不下来
          </text>
        </g>

        {/* 下：散热对照 */}
        <rect x={120} y={550} width={1680} height={330} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <PanelTitle x={160} y={606} text="但没风的代价 · 散热变差" col={P.cream} />
        <line x1={960} y1={630} x2={960} y2={856} stroke={P.card} strokeWidth={3} strokeDasharray="6 12" />
        {heatK > 0 && (
          <g opacity={heatK}>
            {/* 左：风带走热 */}
            <text x={170} y={656} fontSize={24} fill={P.sky}
              fontFamily={FONT_HEAD} fontWeight={800}>路跑 · 迎面风带走体热</text>
            <Runner x={620} y={816} s={0.85} spd={0.55} jersey={P.sky} shorts={P.card} />
            <Wind x={240} y={700} w={560} n={4} gap={30} spd={9} />
            <HeatWisp x={578} y={712} dir={-1} drift={130} n={4} />
            {/* 右：汗难蒸发 */}
            <text x={990} y={656} fontSize={24} fill={P.coral}
              fontFamily={FONT_HEAD} fontWeight={800}>机上 · 汗蒸发慢 · 更闷更喘</text>
            <Treadmill x={1450} y={810} s={0.6}>
              <Runner x={-8} y={-56} s={0.68} spd={0.5} wig={1} />
            </Treadmill>
            <HeatWisp x={1440} y={718} dir={-1} drift={26} n={3} ph={10} />
            <Sweat x={1380} y={640} s={0.9} />
            <Sweat x={1492} y={656} s={0.7} />
          </g>
        )}
      </svg>
      <Chip text="没风的代价" x={1490} y={66} at={b1 + 10} accent={P.coral} />
      <Stamp text="更喘不是幻觉" x={540} y={86} at={b2 + 40} color={P.coralD} size={40} rot={-3} />
    </FadeIn>
  );
};

/* ============ S4 百分之一法则：加坡度对齐能耗 + 分工处方 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);
  const inclK = clamp01((f - 30) / 70);
  const tilt = 5.5 * inclK;
  const popMain = Math.max(0.001, spring({frame: f - 8, fps, config: {damping: 11, stiffness: 120}}));
  const rowK = [0, 1, 2].map((i) => clamp01((f - b1 - 6 - i * 20) / 24));
  const cardK = [0, 1].map((i) =>
    Math.max(0.001, spring({frame: f - b2 - 4 - i * 18, fps, config: {damping: 12, stiffness: 130}})));
  const rows = [
    {ok: true, t: '能耗差距 · 补得上'},
    {ok: false, t: '风阻塑的步态 · 补不了'},
    {ok: false, t: '路面刺激稳定肌 · 补不了'},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={4} name={sec.name} accent={P.amber} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：坡度补偿 */}
        <g transform={`translate(590,480) scale(${popMain}) translate(-590,-480)`} opacity={clamp01(popMain * 2)}>
          <rect x={120} y={140} width={940} height={740} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <PanelTitle x={160} y={196} text="想模拟路跑 · 把坡度调到 1%" />
          {/* 地面参考线 */}
          <line x1={160} y1={600} x2={1020} y2={600} stroke={P.muted} strokeWidth={3} strokeDasharray="8 10" opacity={0.5} />
          <Treadmill x={560} y={600} s={1.05} tilt={tilt}>
            <Runner x={-8} y={-56} s={0.92} spd={0.5} />
          </Treadmill>
          {/* 坡度表 */}
          <g transform="translate(918,300)">
            <rect x={-96} y={-58} width={192} height={120} rx={18} fill={P.bgDeep} stroke={P.amber} strokeWidth={4} />
            <text x={0} y={-16} textAnchor="middle" fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>坡度</text>
            <text x={0} y={34} textAnchor="middle" fontSize={40} fill={P.amber}
              fontFamily={FONT_HEAD} fontWeight={900}>{(inclK * 1).toFixed(1)}%</text>
          </g>
          <text x={918} y={420} textAnchor="middle" fontSize={19} fill={P.muted} fontFamily={FONT_HEAD}>
            示意角度有夸张
          </text>
          {/* 能耗对齐条 */}
          <line x1={330} y1={810} x2={790} y2={810} stroke={P.muted} strokeWidth={3.4} />
          <rect x={400} y={810 - 128} width={72} height={128} rx={14} fill={P.coral} />
          <text x={436} y={842} textAnchor="middle" fontSize={21} fill={P.cream} fontFamily={FONT_HEAD}>路跑</text>
          <rect x={560} y={810 - (88 + 38 * inclK)} width={72} height={88 + 38 * inclK} rx={14} fill={P.belt} />
          <text x={596} y={842} textAnchor="middle" fontSize={21} fill={P.cream} fontFamily={FONT_HEAD}>跑步机</text>
          <line x1={386} y1={682} x2={672} y2={682} stroke={P.cream} strokeWidth={3} strokeDasharray="6 8" opacity={0.55} />
          {inclK > 0.85 && (
            <g>
              <rect x={688} y={656} width={150} height={52} rx={15} fill={P.bgDeep} stroke={P.belt} strokeWidth={4} />
              <text x={763} y={692} textAnchor="middle" fontSize={25} fill={P.belt}
                fontFamily={FONT_HEAD} fontWeight={900}>≈ 对齐</text>
            </g>
          )}
          <text x={160} y={868} fontSize={20} fill={P.muted} fontFamily={FONT_HEAD}>
            论文建议 0.5%~2% · 共识：至少加一点，别跑纯平
          </text>
        </g>

        {/* 右上：坡度补什么 */}
        <rect x={1100} y={140} width={700} height={340} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <PanelTitle x={1140} y={196} text="坡度补得上什么 · 补不上什么" />
        {rows.map((r, i) => (
          <g key={i} opacity={rowK[i]} transform={`translate(0,${(1 - rowK[i]) * 18})`}>
            <circle cx={1176} cy={262 + i * 74} r={26} fill={P.bgDeep}
              stroke={r.ok ? P.belt : P.coral} strokeWidth={4} />
            {r.ok ? <Check x={1176} y={262 + i * 74} s={0.9} /> : <MarkX x={1176} y={262 + i * 74} s={0.85} />}
            <text x={1226} y={272 + i * 74} fontSize={25} fill={r.ok ? P.belt : P.muted}
              fontFamily={FONT_HEAD} fontWeight={r.ok ? 900 : 400}>{r.t}</text>
          </g>
        ))}

        {/* 右下：分工处方 */}
        <rect x={1100} y={510} width={700} height={370} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
        <PanelTitle x={1140} y={566} text="分工处方 · 两个目标两种设置" />
        <g transform={`translate(1450,668) scale(${cardK[0]})`} opacity={clamp01(cardK[0] * 2)}>
          <rect x={-320} y={-58} width={640} height={116} rx={18} fill={P.bgDeep} stroke={P.belt} strokeWidth={4} />
          {/* 小跑步机图标 */}
          <g transform="translate(-252,16)">
            <rect x={-44} y={-10} width={88} height={16} rx={8} fill={P.card} />
            <rect x={-40} y={-7} width={80} height={8} rx={4} fill={P.belt} opacity={0.8} />
            <line x1={30} y1={-10} x2={40} y2={-34} stroke={P.card} strokeWidth={7} strokeLinecap="round" />
          </g>
          <text x={-160} y={-10} fontSize={25} fill={P.cream} fontFamily={FONT_HEAD} fontWeight={800}>
            恢复跑 · 低强度有氧
          </text>
          <text x={-160} y={32} fontSize={23} fill={P.belt} fontFamily={FONT_HEAD} fontWeight={900}>
            → 留在机上
          </text>
        </g>
        <g transform={`translate(1450,806) scale(${cardK[1]})`} opacity={clamp01(cardK[1] * 2)}>
          <rect x={-320} y={-58} width={640} height={116} rx={18} fill={P.bgDeep} stroke={P.orange} strokeWidth={4} />
          {/* 小路面图标 */}
          <g transform="translate(-252,12)">
            <line x1={-44} y1={0} x2={44} y2={0} stroke={P.road} strokeWidth={12} strokeLinecap="round" />
            <line x1={-30} y1={0} x2={-14} y2={0} stroke={P.cream} strokeWidth={4} strokeLinecap="round" />
            <line x1={2} y1={0} x2={18} y2={0} stroke={P.cream} strokeWidth={4} strokeLinecap="round" />
          </g>
          <text x={-160} y={-10} fontSize={25} fill={P.cream} fontFamily={FONT_HEAD} fontWeight={800}>
            专项强度 · 比赛模拟
          </text>
          <text x={-160} y={32} fontSize={23} fill={P.orange} fontFamily={FONT_HEAD} fontWeight={900}>
            → 出门上路
          </text>
        </g>
      </svg>
      <Chip text="至少别跑纯平" x={1470} y={66} at={20} accent={P.amber} />
      <Stamp text="1% 法则" x={560} y={86} at={b1 + 40} color={P.orangeD} size={44} rot={-3} />
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
        <g transform={`translate(960,800) scale(${pop})`} opacity={clamp01(pop)}>
          <Treadmill x={0} y={0} s={1.05} tilt={4}>
            <Runner x={-8} y={-56} s={0.95} spd={0.5} />
          </Treadmill>
          <TagPill x={-400} y={-190} t="机上练有氧" col={P.belt} w={230} />
          <TagPill x={450} y={-195} t="坡度 +1%" col={P.amber} w={210} />
        </g>
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -330}}>
        <div style={{fontSize: 52, fontWeight: 800, color: C.text}}>同一个动作 · 两本账</div>
        <div style={{fontSize: 36, color: P.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E246 = ({data = tl}: {data?: any}) => {
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
