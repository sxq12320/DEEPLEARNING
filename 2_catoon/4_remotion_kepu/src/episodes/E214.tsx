import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01} from '../lib/style';
import {Backdrop, Ripple, Stamp, SceneTag, Chip, Subtitles, ProgressBar, FadeIn, Sun, Moon, Rain} from '../lib/atoms';
import tl from '../timeline_214.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：苔绿底 × 木棕鳞 × 琥珀干 × 天蓝湿 × 薄荷仿生 ============ */
const P = {
  bg: '#26332b',      // 主底：深苔绿
  bgDeep: '#1a241e',
  bgLite: '#2f4034',
  card: '#33473a',
  cream: '#f4ead6',
  ink: '#241708',
  wood: '#a06a35',    // 鳞片木色
  woodD: '#6f4520',   // 鳞片暗边
  woodL: '#d2a06a',   // 外层浅色（吸水猛）
  axis: '#54341a',    // 中轴
  seed: '#e8b96a',    // 松子
  wing: '#f5dfa8',    // 种翅
  dry: '#f2b23e',     // 干燥=琥珀
  wet: '#5cc8ff',     // 潮湿=天蓝
  resin: '#e8a13c',   // 树脂封条
  fire: '#ff7a45',
  fireD: '#d94f2b',
  mint: '#7fe0c3',    // 仿生件
  sq: '#c07a45',      // 松鼠
  sqD: '#8f5527',
  night: '#131a15',
};

/* ---------- 单片鳞：杏仁形，本地坐标，尖朝下 ---------- */
const SCALE_D = 'M 0 -34 Q 20 -18 20 8 Q 20 30 0 36 Q -20 30 -20 8 Q -20 -18 0 -34 Z';

/* ---------- 带翅松子：身子+斜翅 ---------- */
const Seed = ({x, y, s = 1, rot = 0, op = 1}: {x: number; y: number; s?: number; rot?: number; op?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`} opacity={op}>
    <ellipse cx={16} cy={-14} rx={27} ry={7.5} fill={P.wing} stroke={P.woodD} strokeWidth={2.4}
      transform={`rotate(-30 16 -14)`} />
    <ellipse cx={0} cy={0} rx={9} ry={13} fill={P.seed} stroke={P.woodD} strokeWidth={2.4} />
    <circle cx={0} cy={-10} r={4} fill={P.woodD} />
  </g>
);

/* ---------- 松果：中轴 + 6 排鳞片；openK 0=紧闭 1=张开；挂柄在上 ---------- */
const Pinecone = ({x, y, s = 1, rot = 0, openK = 1, seedShow = 0, resinK = 0}: {
  x: number; y: number; s?: number; rot?: number; openK?: number; seedShow?: number; resinK?: number;
}) => {
  const rows = [
    {y: -112, hw: 30}, {y: -74, hw: 58}, {y: -36, hw: 80},
    {y: 2, hw: 86}, {y: 40, hw: 70}, {y: 78, hw: 44},
  ];
  return (
    <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`}>
      <rect x={-9} y={-168} width={18} height={46} rx={9} fill={P.axis} />
      <ellipse cx={0} cy={-10} rx={76} ry={140} fill={P.woodD} opacity={0.5} />
      {/* 缝里露出的种子 */}
      {seedShow > 0 && rows.slice(1, 5).map((r, i) => (
        <g key={i} opacity={seedShow}>
          <Seed x={-r.hw * 0.55} y={r.y + 4} s={0.5} rot={-38} />
          <Seed x={r.hw * 0.55} y={r.y + 4} s={0.5} rot={38} />
        </g>
      ))}
      {/* 鳞片三列 */}
      {rows.map((r, ri) => (
        <g key={ri}>
          {[-1, 0, 1].map((side) => {
            const sx = side * r.hw * (0.5 + openK * 0.38);
            const srot = side * (10 + openK * 36);
            const sc = 1 - ri * 0.035;
            return (
              <g key={side} transform={`translate(${sx},${r.y}) rotate(${srot}) scale(${sc})`}>
                <path d={SCALE_D} fill={P.wood} stroke={P.woodD} strokeWidth={4} />
                <path d="M 0 -30 Q 13 -16 13 6" stroke={P.woodL} strokeWidth={5}
                  fill="none" opacity={0.65} strokeLinecap="round" />
              </g>
            );
          })}
        </g>
      ))}
      {/* 树脂封层 */}
      {resinK > 0 && (
        <g opacity={resinK}>
          {[[-46, -60], [40, -66], [-58, 6], [52, 10], [-20, 62], [24, 58]].map(([rx, ry], i) => (
            <circle key={i} cx={rx} cy={ry} r={15 + (i % 3) * 4} fill={P.resin}
              transform={`scale(${resinK})`} opacity={0.9} />
          ))}
          <path d="M -30 80 Q -28 112 -30 130" stroke={P.resin} strokeWidth={9 * resinK}
            fill="none" strokeLinecap="round" opacity={0.85} />
        </g>
      )}
    </g>
  );
};

/* ---------- 双层材料条：外层(上,浅)吸水猛 / 内层(下,深)吸水少；m 0干→1湿 ---------- */
const Bilayer = ({x, y, m}: {x: number; y: number; m: number}) => {
  const len = 340;
  const bend = -0.2 + m * 0.9;            // 干：梢头上翘；湿：向下弯（朝内合拢）
  const outerT = 22 + m * 18;             // 湿时外层变厚
  const innerT = 22;
  const cy = bend * len * 1.15;
  const ty = bend * len * 0.82;
  const q = (t: number, a: number, c: number, b: number) => (1 - t) * (1 - t) * a + 2 * (1 - t) * t * c + t * t * b;
  const qd = (t: number, a: number, c: number, b: number) => 2 * (1 - t) * (c - a) + 2 * t * (b - c);
  const ticks = (off: number, w: number, col: string) =>
    [0.18, 0.34, 0.5, 0.66, 0.82].map((t, i) => {
      const px = q(t, 0, len / 2, len);
      const py = q(t, off, cy + off, ty + off);
      const ang = Math.atan2(qd(t, off, cy + off, ty + off), qd(t, 0, len / 2, len)) * 180 / Math.PI;
      return <line key={i} x1={px} y1={py - w * 0.38} x2={px} y2={py + w * 0.38}
        stroke={col} strokeWidth={4} opacity={0.5} transform={`rotate(${ang} ${px} ${py})`} />;
    });
  return (
    <g transform={`translate(${x},${y})`}>
      {/* 基座（连着果轴） */}
      <rect x={-40} y={-64} width={28} height={128} rx={12} fill={P.axis} />
      <path d={`M 0 ${-outerT / 2} Q ${len / 2} ${cy - outerT / 2} ${len} ${ty - outerT / 2}`}
        stroke={P.woodL} strokeWidth={outerT} fill="none" strokeLinecap="round" />
      {ticks(-outerT / 2, outerT, P.ink)}
      <path d={`M 0 ${innerT / 2} Q ${len / 2} ${cy + innerT / 2} ${len} ${ty + innerT / 2}`}
        stroke={P.woodD} strokeWidth={innerT} fill="none" strokeLinecap="round" />
      {ticks(innerT / 2, innerT, '#3d2410')}
      <path d={`M 0 0 Q ${len / 2} ${cy} ${len} ${ty}`} stroke={P.ink} strokeWidth={3}
        fill="none" opacity={0.45} strokeDasharray="8 8" />
      {/* 层标签：图例式固定在左上，干湿两态都不压条 */}
      <rect x={-14} y={-176} width={24} height={24} rx={6} fill={P.woodL} />
      <text x={20} y={-156} fill={P.woodL} fontSize={27}
        fontFamily={FONT_HEAD} fontWeight={700}>外层 · 吸水猛</text>
      <rect x={-14} y={-130} width={24} height={24} rx={6} fill={P.woodD} />
      <text x={20} y={-110} fill="#c89b6b" fontSize={27}
        fontFamily={FONT_HEAD} fontWeight={700}>内层 · 吸水少</text>
      {/* 弯折方向箭头 */}
      <g opacity={0.9}>
        <path d={`M ${len + 30} ${ty * 0.5 - 40} Q ${len + 62} ${ty * 0.5} ${len + 30} ${ty * 0.5 + 40}`}
          stroke={P.cream} strokeWidth={6} fill="none" strokeLinecap="round" />
        <polygon points={`${len + 22},${ty * 0.5 + 34} ${len + 44},${ty * 0.5 + 40} ${len + 30},${ty * 0.5 + 56}`}
          fill={P.cream} />
      </g>
    </g>
  );
};

/* ---------- 风弧 ---------- */
const WindArc = ({d, at, dur = 30, col = P.cream, w = 7}: {d: string; at: number; dur?: number; col?: string; w?: number}) => {
  const f = useCurrentFrame();
  const t = f - at;
  if (t < 0) return null;
  const p = clamp01(t / dur);
  const out = clamp01((t - dur - 16) / 20);
  return (
    <path d={d} pathLength={100} stroke={col} strokeWidth={w} fill="none" strokeLinecap="round"
      strokeDasharray={100} strokeDashoffset={100 * (1 - p)} opacity={0.75 * (1 - out)} />
  );
};

/* ---------- 松鼠 ---------- */
const Squirrel = ({x, y, s = 1, reach = 0}: {x: number; y: number; s?: number; reach?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    {/* 大尾巴 */}
    <path d="M -30 -30 Q -95 -80 -70 -140 Q -55 -172 -20 -160 Q -60 -130 -38 -60 Q -34 -44 -30 -30 Z" fill={P.sqD} />
    {/* 身体 */}
    <ellipse cx={0} cy={-34} rx={32} ry={42} fill={P.sq} />
    <ellipse cx={8} cy={-18} rx={20} ry={26} fill="#e0a071" />
    {/* 后腿 */}
    <ellipse cx={-6} cy={6} rx={24} ry={12} fill={P.sqD} />
    {/* 头 */}
    <circle cx={26} cy={-76} r={21} fill={P.sq} />
    <polygon points="16,-92 20,-112 32,-94" fill={P.sq} />
    <circle cx={34} cy={-80} r={4.5} fill={P.ink} />
    <circle cx={44} cy={-66} r={5} fill={P.sqD} />
    {/* 前爪：够向松果 */}
    <g transform={`translate(20,-52) rotate(${-30 - reach * 34})`}>
      <rect x={0} y={-7} width={52} height={14} rx={7} fill={P.sq} />
      <circle cx={52} cy={0} r={9} fill={P.sq} />
    </g>
  </g>
);

/* ---------- 喷壶 ---------- */
const SprayBottle = ({x, y, rot = 0, op = 1}: {x: number; y: number; rot?: number; op?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot})`} opacity={op}>
    <rect x={-34} y={0} width={68} height={110} rx={16} fill={P.wet} />
    <rect x={-34} y={0} width={68} height={26} rx={13} fill="#8fdcff" opacity={0.7} />
    <rect x={-12} y={-42} width={24} height={46} rx={8} fill="#3b82a8" />
    <rect x={-12} y={-58} width={56} height={22} rx={10} fill="#3b82a8" />
    <rect x={40} y={-54} width={22} height={14} rx={6} fill="#3b82a8" />
    <path d="M -12 -36 L -34 -14 L -22 -4 L -4 -28 Z" fill="#2e6a8a" />
  </g>
);

/* ---------- 吹风机 ---------- */
const HairDryer = ({x, y, rot = 0, op = 1}: {x: number; y: number; rot?: number; op?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot})`} opacity={op}>
    <rect x={-14} y={6} width={26} height={64} rx={12} fill={P.fireD} transform="rotate(-18 0 40)" />
    <ellipse cx={0} cy={0} rx={52} ry={40} fill={P.fire} />
    <ellipse cx={-8} cy={0} rx={30} ry={26} fill="#ff9a6b" />
    <rect x={40} y={-16} width={52} height={32} rx={14} fill={P.fireD} />
    <rect x={86} y={-10} width={18} height={20} rx={8} fill={P.ink} />
  </g>
);

/* ---------- 火焰 ---------- */
const Flame = ({x, y, s = 1, h = 1, seed = 0}: {x: number; y: number; s?: number; h?: number; seed?: number}) => {
  const f = useCurrentFrame();
  const w = Math.sin(f / 4.2 + seed * 2.6) * 14;
  return (
    <g transform={`translate(${x},${y}) scale(${s})`} opacity={h}>
      <path d={`M 0 0 Q ${-34 + w} -64 ${-12 + w * 0.6} -116 Q ${-4 + w * 0.4} -150 0 -168 Q ${16 + w * 0.3} -132 ${14 + w * 0.5} -96 Q ${34 + w * 0.8} -52 0 0 Z`} fill={P.fire} />
      <path d={`M 0 0 Q ${-14 + w * 0.4} -38 ${-2 + w * 0.3} -78 Q ${6} -58 ${14 + w * 0.4} -34 Q ${12} -12 0 0 Z`} fill="#ffd23e" />
    </g>
  );
};

/* ---------- 仿生种载：锚在左端的条带，k 0平直 → 1卷成螺 ---------- */
const Carrier = ({x, y, k, rot = 0, s = 1}: {x: number; y: number; k: number; rot?: number; s?: number}) => {
  const segs = 9, segLen = 34;
  const curl = k * (Math.PI * 2.05) / segs;
  const pts: {x: number; y: number; a: number}[] = [];
  let px = 0, py = 0, ang = 0;
  for (let i = 0; i <= segs; i++) {
    pts.push({x: px, y: py, a: ang});
    px += Math.cos(ang) * segLen;
    py += Math.sin(ang) * segLen;
    ang += curl;
  }
  return (
    <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`}>
      {pts.slice(0, segs).map((p, i) => (
        <rect key={i} x={-4} y={-9} width={segLen + 6} height={18} rx={8}
          fill={i % 2 ? P.mint : '#9fe9d2'} stroke="#4aa88d" strokeWidth={2.5}
          transform={`translate(${p.x},${p.y}) rotate(${p.a * 180 / Math.PI})`} />
      ))}
      {/* 绑在条带上的种子 */}
      <Seed x={pts[2].x + Math.cos(pts[2].a + 1.2) * 22} y={pts[2].y + Math.sin(pts[2].a + 1.2) * 22}
        s={0.6} rot={pts[2].a * 180 / Math.PI} />
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
  const p1 = Math.max(0.001, spring({frame: f - 12, fps, config: {damping: 10, stiffness: 150}}));
  const bob = Math.sin(f / 16) * 8;
  return (
    <AbsoluteFill>
      <Backdrop top={P.bg} bottom={P.bg} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <circle cx={960} cy={348} r={215} fill={P.cream} />
        <g transform={`translate(960,${352 + bob}) scale(${p0})`}>
          <Pinecone x={0} y={0} s={1.05} rot={-6} openK={0.75} />
        </g>
        <g transform={`translate(742,${300 - bob}) scale(${p1})`}><Seed x={0} y={0} s={0.9} rot={-24} /></g>
        <g transform={`translate(1180,${316 + bob * 0.7}) scale(${p1})`}><Seed x={0} y={0} s={0.8} rot={150} /></g>
        {/* 一滴水 + 一缕热风小标 */}
        <g transform={`translate(790,430) scale(${p1})`}>
          <path d="M 0 -20 Q 14 4 0 18 Q -14 4 0 -20 Z" fill={P.wet} />
        </g>
        <g transform={`translate(1140,438) scale(${p1})`} stroke={P.dry} strokeWidth={6} fill="none" strokeLinecap="round">
          <path d="M -16 8 Q 0 -8 16 8" />
          <path d="M -12 20 Q 0 8 12 20" />
        </g>
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 260}}>
          <div style={{color: P.dry, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 26}}>
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

/* ============ S0 双层机关：左松果 + 右放大剖面，湿弯干翘 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);

  const m = clamp01((f - (b1 + 16)) / 70) * (1 - clamp01((f - (b1 + 205)) / 80)); // 湿度 0→1→0
  const openK = 1 - m;
  const conePop = spring({frame: f - 8, fps, config: {damping: 11, stiffness: 120}});
  const dropZone = f > b1 + 10 && f < b1 + 100;

  const CX = 1330, CY = 470, R = 292;

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={0} name={sec.name} accent={P.dry} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <defs>
          <clipPath id="s0zoom"><circle cx={CX} cy={CY} r={R - 10} /></clipPath>
        </defs>
        {/* 左：整果（干开湿合） */}
        <ellipse cx={430} cy={840} rx={240} ry={34} fill={P.night} />
        <g transform={`translate(430,520) scale(${1.5 * Math.max(0.001, conePop)})`}>
          <Pinecone x={0} y={0} openK={openK} seedShow={openK * 0.9} />
        </g>
        {/* 放大定位圈 + 引线 */}
        <circle cx={536} cy={372} r={56} fill="none" stroke={P.mint} strokeWidth={5}
          strokeDasharray="11 9" />
        <path d="M 590 380 C 740 380 860 400 1035 420" stroke={P.mint} strokeWidth={4}
          strokeDasharray="10 11" fill="none" opacity={0.8} />
        {/* 右：放大窗 */}
        <circle cx={CX} cy={CY} r={R} fill="#1c2b22" stroke={P.wood} strokeWidth={7} />
        <text x={CX} y={CY - R - 24} textAnchor="middle" fill={P.mint} fontSize={30}
          fontFamily={FONT_HEAD} letterSpacing={4}>鳞片横切 · 放大看</text>
        <g clipPath="url(#s0zoom)">
          <Bilayer x={CX - 200} y={CY + 10} m={m} />
          {/* 湿：水滴落在外层上 */}
          {dropZone && [0, 1, 2, 3, 4].map((i) => {
            const at = b1 + 14 + i * 16;
            const t = f - at;
            if (t < 0) return null;
            const dy = t * 11;
            const hit = dy > 150;
            const dx = CX - 130 + i * 78;
            if (hit && t > 165) return null;
            return hit
              ? <ellipse key={i} cx={dx} cy={CY - 88} rx={16 + (t - 150) * 1.4} ry={5}
                  fill="none" stroke={P.wet} strokeWidth={3} opacity={0.8 - (t - 150) * 0.06} />
              : <path key={i} d="M 0 -13 Q 9 3 0 12 Q -9 3 0 -13 Z" fill={P.wet}
                  transform={`translate(${dx},${CY - 240 + dy})`} />;
          })}
          {/* 干：小太阳把水汽烘走 */}
          {m < 0.95 && f > b1 + 190 && <Sun x={CX + 160} y={CY - 185} r={44} opacity={1 - m} />}
        </g>
        {/* 湿度小仪表 */}
        <g transform={`translate(180,140)`}>
          <rect x={0} y={0} width={240} height={30} rx={15} fill="none" stroke={P.cream} strokeWidth={4} opacity={0.8} />
          <rect x={5} y={5} width={230 * m} height={20} rx={10} fill={P.wet} />
          <text x={0} y={-14} fill={P.cream} fontSize={26} fontFamily={FONT_HEAD}>空气湿度</text>
        </g>
        <Ripple x={536} y={372} startFrame={beat(0) + 30} color={P.mint} />
      </svg>
      <Chip text="两层脾气不同" x={140} y={230} at={beat(0) + 20} accent={P.woodL} />
      <Chip text="不用电的湿度执行器" x={140} y={790} at={b1 + 110} accent={P.mint} />
      <Stamp text="潮湿 → 合拢" x={1240} y={830} at={b1 + 60} color={P.wet} size={46} rot={4} />
      <Stamp text="干燥 → 张开" x={1240} y={830} at={b1 + 240} color={P.dry} size={46} rot={-4} />
      <Chip text="双金属片同款原理" x={1350} y={140} at={b1 + 285} accent={P.dry} />
    </FadeIn>
  );
};

/* ============ S1 挑日子：干燥放飞带翅松子，雨一打就坠毁 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);

  const conePop = spring({frame: f - 8, fps, config: {damping: 11, stiffness: 120}});
  const rainK = clamp01((f - (b2 + 18)) / 30);
  const cloudX = interpolate(f, [b2 + 8, b2 + 45], [2350, 1490], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});

  // 种子飞行路径（Q 贝塞尔），前4颗干爽飞出，第5颗雨中坠毁
  const seeds = [
    {x0: 470, y0: 360, cx: 950, cy: 160, x1: 1560, y1: 300, at: b1 + 8},
    {x0: 450, y0: 430, cx: 1000, cy: 300, x1: 1690, y1: 480, at: b1 + 30},
    {x0: 460, y0: 500, cx: 940, cy: 380, x1: 1600, y1: 640, at: b1 + 52},
    {x0: 480, y0: 330, cx: 1100, cy: 90, x1: 1760, y1: 190, at: b1 + 76},
  ];
  const badAt = b2 + 55;                                    // 雨天那颗
  const crashT = badAt + 75;                                // 被雨打湿点
  const qq = (t: number, a: number, c: number, b: number) => (1 - t) * (1 - t) * a + 2 * (1 - t) * t * c + t * t * b;

  const seedFly = (sd: typeof seeds[0], i: number) => {
    const t = f - sd.at;
    if (t < 0) return null;
    const p = clamp01(t / 95);
    if (p >= 1) return null;
    const sx = qq(p, sd.x0, sd.cx, sd.x1);
    const sy = qq(p, sd.y0, sd.cy, sd.y1);
    return <Seed key={i} x={sx} y={sy} s={0.85} rot={t * 7 + i * 40} />;
  };

  // 雨种：先飞后坠
  const badT = f - badAt;
  let badSeed = null;
  if (badT > 0) {
    const p = clamp01(badT / 95);
    const flyX = qq(Math.min(p, 0.62), 460, 900, 1300);
    const flyY = qq(Math.min(p, 0.62), 380, 230, 360);
    if (f < crashT) {
      badSeed = <Seed x={flyX} y={flyY} s={0.85} rot={badT * 7} />;
    } else {
      const dt = f - crashT;
      const fy = Math.min(845, flyY + dt * dt * 0.5);
      badSeed = (
        <g>
          <Seed x={flyX + Math.sin(dt / 5) * 14} y={fy} s={0.85}
            rot={60 + dt * 2.4} op={fy >= 845 ? 0.85 : 1} />
        </g>
      );
    }
  }

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={1} name={sec.name} accent={P.wet} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <ellipse cx={400} cy={880} rx={230} ry={30} fill={P.night} />
        <g transform={`translate(400,540) scale(${1.55 * Math.max(0.001, conePop)})`}>
          <Pinecone x={0} y={0} openK={1 - rainK * 0.92} seedShow={1 - rainK} />
        </g>
        {/* 风道虚线 */}
        <g opacity={1 - rainK}>
          <WindArc d="M 520 330 Q 1000 170 1500 260" at={b1 + 8} dur={36} col={P.dry} w={6} />
          <WindArc d="M 540 470 Q 1050 380 1640 500" at={b1 + 22} dur={36} col={P.dry} w={6} />
          <WindArc d="M 530 560 Q 1000 520 1550 660" at={b1 + 40} dur={36} col={P.dry} w={6} />
        </g>
        {seeds.map((sd, i) => seedFly(sd, i))}
        {/* 距离刻度旗 */}
        {f > b1 + 90 && f < b2 + 20 && (
          <g opacity={clamp01((f - b1 - 90) / 10) * (1 - rainK)}>
            <line x1={1620} y1={700} x2={1620} y2={810} stroke={P.cream} strokeWidth={5} />
            <polygon points="1620,700 1690,714 1620,728" fill={P.dry} />
            <text x={1618} y={852} textAnchor="middle" fill={P.cream} fontSize={26}
              fontFamily={FONT_HEAD}>几十米外</text>
          </g>
        )}
        {/* 雨云入场 */}
        <g transform={`translate(${cloudX},190)`} opacity={clamp01((f - b2 - 8) / 20)}>
          <ellipse cx={0} cy={0} rx={95} ry={44} fill="#5a6b78" />
          <ellipse cx={-52} cy={14} rx={60} ry={34} fill="#5a6b78" />
          <ellipse cx={54} cy={16} rx={64} ry={36} fill="#5a6b78" />
          <ellipse cx={0} cy={-14} rx={70} ry={34} fill="#72869a" />
        </g>
        <Rain on={rainK} n={50} />
        {badSeed}
        {/* 坠毁点水花 */}
        {f > crashT + 55 && f < crashT + 95 && (
          <g transform="translate(905,850)" opacity={1 - (f - crashT - 55) / 40}>
            <path d="M -24 0 Q -12 -18 0 0 Q 12 -18 24 0" stroke={P.wet} strokeWidth={5} fill="none" />
          </g>
        )}
      </svg>
      <Stamp text="只在干燥天开张" x={1320} y={130} at={b1 + 6} color={P.dry} size={46} rot={4} />
      <Chip text="小翅 = 微型滑翔伞" x={140} y={140} at={b1 + 40} accent={P.wing} />
      <Chip text="风一吹飞出几十米" x={700} y={130} at={b1 + 95} accent={P.dry} />
      <Stamp text="雨天放飞 = 坠毁" x={140} y={196} at={b2 + 70} color={P.wet} size={48} rot={-6} />
      <Chip text="雨天坚决收摊" x={1450} y={700} at={b2 + 120} accent={P.wet} />
    </FadeIn>
  );
};

/* ============ S2 上锁：夜里闭合防松鼠；白天被风吹落滚着撒种 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);

  const det = b2 + 20;                       // 脱落时刻
  const fallK = f > det ? clamp01((f - det) / 42) : 0;
  const rollK = f > det + 44 ? clamp01((f - det - 44) / 150) : 0;
  const coneY = 385 + fallK * fallK * 400;   // 加速下落
  const coneX = 640 + rollK * 700;
  const coneRot = rollK * 300;
  const openK = rollK * 0.65;                // 滚的时候半开撒种
  const paw = Math.max(0, Math.sin((f - b1) / 11)) * clamp01((f - b1 - 20) / 15) * (1 - fallK);
  const lockK = clamp01((f - (b1 + 165)) / 16) * (1 - fallK);

  return (
    <FadeIn>
      <Backdrop top={P.bgDeep} bottom={P.bgDeep} />
      <SceneTag index={2} name={sec.name} accent={P.mint} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Moon x={1620} y={160} scale={0.9} />
        {/* 树枝 */}
        <path d="M 40 262 Q 380 218 760 236" stroke={P.axis} strokeWidth={30} fill="none" strokeLinecap="round" />
        <path d="M 40 262 Q 380 218 760 236" stroke={P.wood} strokeWidth={14} fill="none" strokeLinecap="round" opacity={0.5} />
        {/* 松鼠在枝上够松果 */}
        <g transform={`translate(0,${fallK * -10})`} opacity={1 - rollK}>
          <Squirrel x={515} y={252} s={1.05} reach={paw} />
          {/* 够不着的气泡 */}
          {paw > 0.6 && (
            <g transform="translate(668,112)">
              <circle r={30} fill={P.cream} opacity={0.95} />
              <text x={0} y={9} textAnchor="middle" fill={P.ink} fontSize={26} fontFamily={FONT_HEAD}>?</text>
            </g>
          )}
        </g>
        {/* 松果：挂着→坠落→滚动 */}
        <g transform={`translate(${coneX},${coneY}) rotate(${coneRot})`}>
          <Pinecone x={0} y={0} s={1.25} openK={openK} seedShow={openK * 0.7} />
        </g>
        {/* 夜班锁 */}
        {lockK > 0 && (
          <g transform={`translate(${coneX},${coneY + 10}) scale(${Math.max(0.001, spring({frame: f - (b1 + 165), fps, config: {damping: 9, stiffness: 170}}))})`} opacity={lockK}>
            <rect x={-34} y={-10} width={68} height={56} rx={14} fill={P.dry} stroke={P.ink} strokeWidth={4} />
            <path d="M -20 -10 L -20 -30 Q -20 -52 0 -52 Q 20 -52 20 -30 L 20 -10"
              stroke={P.dry} strokeWidth={10} fill="none" />
            <circle cx={0} cy={14} r={8} fill={P.ink} />
            <rect x={-4} y={16} width={8} height={16} rx={4} fill={P.ink} />
          </g>
        )}
        {/* 风弧推它下枝 */}
        <WindArc d="M 300 360 Q 480 330 610 380" at={det - 22} col={P.cream} w={7} />
        <WindArc d="M 320 470 Q 500 440 615 470" at={det - 8} col={P.cream} w={7} />
        {/* 地面 */}
        <rect x={0} y={860} width={1920} height={220} fill={P.night} />
        <rect x={0} y={860} width={1920} height={10} fill="#2c2418" />
        {/* 落地弹跳涟漪 */}
        {fallK >= 1 && <Ripple x={640} y={848} startFrame={det + 42} color={P.cream} />}
        {/* 滚过处落下种子 */}
        {[[780, 830], [1030, 845], [1260, 825]].map(([sx, sy], i) => {
          const at = det + 60 + i * 36;
          const sp = Math.max(0.001, spring({frame: f - at, fps, config: {damping: 10, stiffness: 160}}));
          return f > at ? <Seed key={i} x={sx} y={sy} s={0.62 * sp} rot={i * 55 - 30} /> : null;
        })}
        {/* 滚动轨迹虚线 */}
        {rollK > 0 && (
          <path d={`M 640 852 L ${620 + rollK * 720} 852`} stroke={P.cream} strokeWidth={4}
            strokeDasharray="3 16" opacity={0.5} />
        )}
      </svg>
      <Chip text="鳞片合紧 = 防盗门" x={140} y={140} at={b1 + 25} accent={P.wood} />
      <Stamp text="夜班锁" x={1390} y={300} at={b1 + 170} color={P.dry} size={48} rot={5} />
      <Chip text="松鼠·鸟·虫都下不了嘴" x={140} y={700} at={b1 + 120} accent={P.mint} />
      <Stamp text="落地继续撒种" x={1280} y={640} at={det + 90} color={P.mint} size={46} rot={-5} />
      <Chip text="第二段扩散 · 零能耗" x={140} y={790} at={det + 140} accent={P.dry} />
    </FadeIn>
  );
};

/* ============ S3 死后表演：喷壶合、烘机开、角度当湿度计 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);

  const wetT = b1 + 45, dryT = b1 + 165;
  const openK = 1 - clamp01((f - wetT) / 60) + clamp01((f - dryT) / 60);
  const oK = clamp01(openK);
  const sprayO = clamp01((f - (wetT - 25)) / 14) * (1 - clamp01((f - (wetT + 95)) / 18));
  const dryO = clamp01((f - (dryT - 25)) / 14) * (1 - clamp01((f - (dryT + 95)) / 18));
  const dialK = clamp01((f - (b2 + 10)) / 25);
  const needle = -62 + oK * 124;              // 湿→左 干→右

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={3} name={sec.name} accent={P.wet} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 桌面 */}
        <rect x={0} y={806} width={1920} height={274} fill={P.night} />
        <rect x={0} y={806} width={1920} height={12} fill={P.woodD} />
        {/* 松果标本 */}
        <ellipse cx={620} cy={828} rx={210} ry={26} fill="#0d120f" />
        <g transform="translate(620,600) scale(1.5)">
          <Pinecone x={0} y={0} openK={oK} seedShow={oK * 0.6} />
        </g>
        {/* “已挂掉”吊牌 */}
        <g opacity={clamp01((f - beat(0) - 30) / 12)}>
          <line x1={620} y1={368} x2={672} y2={330} stroke={P.cream} strokeWidth={3} />
          <g transform="translate(700,322) rotate(14)">
            <rect x={-58} y={-30} width={118} height={56} rx={10} fill={P.cream} />
            <circle cx={-40} cy={0} r={6} fill={P.bg} />
            <text x={12} y={10} textAnchor="middle" fill={P.ink} fontSize={30}
              fontFamily={FONT_HEAD} fontWeight={700}>已离世</text>
          </g>
        </g>
        {/* 喷壶 + 水线 */}
        <g transform={`translate(300,470) rotate(24)`} opacity={sprayO}>
          <SprayBottle x={0} y={0} />
        </g>
        {sprayO > 0.4 && [0, 1, 2].map((i) => (
          <path key={i} d={`M 360 ${455 + i * 26} Q 470 ${440 + i * 30} 560 ${470 + i * 36}`}
            stroke={P.wet} strokeWidth={5} fill="none" strokeDasharray="4 14" strokeLinecap="round"
            strokeDashoffset={-((f * 2.2) % 36)} opacity={sprayO} />
        ))}
        {/* 吹风机 + 热风 */}
        <g transform={`translate(960,470) rotate(160)`} opacity={dryO}>
          <HairDryer x={0} y={0} />
        </g>
        {dryO > 0.4 && [0, 1, 2].map((i) => (
          <path key={i} d={`M 870 ${445 + i * 30} Q 770 ${430 + i * 32} 690 ${470 + i * 38}`}
            stroke={P.fire} strokeWidth={6} fill="none" strokeDasharray="16 20" strokeLinecap="round"
            strokeDashoffset={(f * 2.6) % 36} opacity={dryO * 0.85} />
        ))}
        {/* 湿度刻度盘 */}
        <g transform="translate(1420,560)" opacity={dialK}>
          <path d="M -160 60 A 175 175 0 0 1 160 60" fill="none" stroke={P.card} strokeWidth={46} strokeLinecap="round" />
          <path d="M -160 60 A 175 175 0 0 1 -55 -140" fill="none" stroke={P.wet} strokeWidth={46} opacity={0.8} />
          <path d="M 55 -140 A 175 175 0 0 1 160 60" fill="none" stroke={P.dry} strokeWidth={46} opacity={0.8} />
          {[-62, -31, 0, 31, 62].map((a) => (
            <line key={a} x1={Math.sin(a * Math.PI / 180) * 136} y1={-Math.cos(a * Math.PI / 180) * 136 + 60}
              x2={Math.sin(a * Math.PI / 180) * 118} y2={-Math.cos(a * Math.PI / 180) * 118 + 60}
              stroke={P.cream} strokeWidth={4} />
          ))}
          <text x={-150} y={110} fill={P.wet} fontSize={30} fontFamily={FONT_HEAD} fontWeight={700} textAnchor="middle">湿</text>
          <text x={150} y={110} fill={P.dry} fontSize={30} fontFamily={FONT_HEAD} fontWeight={700} textAnchor="middle">干</text>
          {/* 针 = 一片小鳞 */}
          <g transform={`rotate(${needle} 0 60)`}>
            <line x1={0} y1={60} x2={0} y2={-96} stroke={P.cream} strokeWidth={8} strokeLinecap="round" />
            <path d="M 0 -124 L 12 -96 L -12 -96 Z" fill={P.cream} />
          </g>
          <circle cx={0} cy={60} r={16} fill={P.cream} stroke={P.ink} strokeWidth={4} />
          <text x={0} y={172} textAnchor="middle" fill={P.mint} fontSize={28}
            fontFamily={FONT_HEAD}>鳞片角度 = 湿度</text>
        </g>
        <path d="M 830 560 C 1000 560 1100 560 1230 560" stroke={P.mint} strokeWidth={4}
          strokeDasharray="10 11" fill="none" opacity={dialK * 0.8} />
      </svg>
      <Stamp text="死了照样干活" x={140} y={150} at={beat(0) + 15} color={P.wood} size={48} rot={-5} />
      <Chip text="不要命 · 不要电" x={1450} y={140} at={beat(0) + 60} accent={P.mint} />
      <Chip text="喷水 → 合拢" x={140} y={660} at={wetT + 20} accent={P.wet} />
      <Chip text="烘干 → 张开" x={1450} y={250} at={dryT + 20} accent={P.fire} />
      <Stamp text="能无限循环" x={120} y={330} at={b2 - 20} color={P.dry} size={44} rot={-5} />
    </FadeIn>
  );
};

/* ============ S4 火媒与仿生：左火烧爆开，右钻土载体 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);

  const fireK = clamp01((f - (b1 - 240)) / 40) * (1 - clamp01((f - (b1 + 40)) / 30));
  const meltK = clamp01((f - (b1 - 160)) / 55);              // 树脂熔化
  const burst = b1 - 95;                                     // 爆开时刻
  const openK = f > burst ? Math.min(1, spring({frame: f - burst, fps, config: {damping: 8, stiffness: 200}}) * 1.15) : 0;
  const sproutK = clamp01((f - (b1 - 10)) / 25);
  const bioK = clamp01((f - (b1 + 55)) / 30);                // 右侧亮起
  const curlK = clamp01((f - (b1 + 120)) / 80);              // 遇湿卷曲
  const drillK = clamp01((f - (b1 + 215)) / 60);             // 钻入土
  const dropT = b1 + 95;

  // 爆开的种子
  const burstDirs = [[-60, -30], [-20, -60], [20, -64], [60, -30], [-70, 20], [70, 18], [-34, 40], [36, 44]];
  const carY = 500 + drillK * 330;

  return (
    <FadeIn>
      <Backdrop top={P.bgDeep} bottom={P.bgDeep} />
      <SceneTag index={4} name={sec.name} accent={P.fire} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 分界虚线 */}
        <line x1={955} y1={140} x2={955} y2={830} stroke={P.cream} strokeWidth={3}
          strokeDasharray="4 18" opacity={0.35} />
        {/* 左：焦土地面 + 余烬 */}
        <rect x={0} y={858} width={960} height={222} fill="#1c130d" />
        <rect x={0} y={858} width={960} height={8} fill="#332417" />
        {[90, 200, 330, 470, 590, 700, 830].map((ex, i) => (
          <circle key={i} cx={ex} cy={872 + (i % 2) * 14} r={5 + (i % 3) * 2}
            fill={i % 2 ? P.fire : '#ffb03a'}
            opacity={0.4 + 0.55 * Math.abs(Math.sin(f / 9 + i * 1.7))} />
        ))}
        {/* 枝 + 封死的松果 */}
        <rect x={60} y={300} width={500} height={26} rx={13} fill={P.axis} transform="rotate(-6 60 300)" />
        <g transform="translate(430,468)">
          <Pinecone x={0} y={0} s={1.3} openK={openK} resinK={1 - meltK} />
        </g>
        {/* 树脂熔滴 */}
        {meltK > 0 && meltK < 1 && [0, 1, 2].map((i) => {
          const t = f - (b1 - 160) - i * 14;
          if (t < 0) return null;
          return <circle key={i} cx={395 + i * 36} cy={560 + Math.min(280, t * t * 0.55)}
            r={9 - i} fill={P.resin} opacity={clamp01(1 - t / 110)} />;
        })}
        {/* 火焰 */}
        {[140, 280, 420, 560, 700, 820].map((fx, i) => (
          <Flame key={i} x={fx} y={868} s={0.9 + (i % 3) * 0.28} seed={i}
            h={fireK * (0.75 + 0.25 * Math.sin(f / 6 + i * 2))} />
        ))}
        {/* 爆开的种子：飞出→落焦土 */}
        {burstDirs.map(([dx, dy], i) => {
          const t = f - burst - i * 3;
          if (t < 0) return null;
          const p = clamp01(t / 40);
          const sx = 430 + dx * 3.4 * p;
          const sy = 468 + dy * 3.4 * p + Math.max(0, (t - 40) * 4.2);
          const fy = Math.min(848, sy);
          return <Seed key={i} x={sx} y={fy} s={0.6} rot={t * 8 + i * 40} />;
        })}
        {/* 焦土里冒出的第一棵苗 */}
        <g transform={`translate(700,858) scale(${Math.max(0.001, spring({frame: f - (b1 - 10), fps, config: {damping: 9, stiffness: 160}}))})`} opacity={sproutK}>
          <path d="M 0 0 Q -4 -40 0 -62" stroke={P.mint} strokeWidth={7} fill="none" strokeLinecap="round" />
          <path d="M 0 -40 Q -34 -48 -38 -74 Q -12 -68 0 -46" fill={P.mint} />
          <path d="M 0 -52 Q 30 -58 36 -84 Q 10 -78 0 -58" fill="#9fe9d2" />
        </g>
        {/* 温度读数 */}
        {fireK > 0.3 && (
          <g transform="translate(160,640)" opacity={fireK}>
            <rect x={0} y={0} width={190} height={54} rx={14} fill={P.card} stroke={P.fire} strokeWidth={4} />
            <text x={95} y={38} textAnchor="middle" fill={P.fire} fontSize={30}
              fontFamily={FONT_HEAD} fontWeight={700}>≈50°C 解锁</text>
          </g>
        )}

        {/* 右：仿生钻土载体 */}
        <g opacity={0.35 + bioK * 0.65}>
          <rect x={960} y={858} width={960} height={222} fill="#241b10" />
          <ellipse cx={1400} cy={862} rx={330} ry={66} fill="#3a2c1c" />
          <ellipse cx={1400} cy={850} rx={330} ry={52} fill="#4a3824" />
        </g>
        {/* 水滴先落下 */}
        {f > dropT && f < b1 + 150 && (
          <path d="M 0 -16 Q 11 3 0 15 Q -11 3 0 -16 Z" fill={P.wet}
            transform={`translate(1400,${300 + (f - dropT) * 9})`} opacity={1 - clamp01((f - dropT - 32) / 8)} />
        )}
        {/* 载体：平直→卷曲→钻入土（土盖下半） */}
        <g transform={`translate(${1280 - drillK * 60},${carY}) rotate(${drillK * 620})`}>
          <Carrier x={0} y={0} k={curlK} s={1.05} />
        </g>
        {/* 土丘前景盖住钻入部分 */}
        <ellipse cx={1400} cy={876} rx={330} ry={60} fill="#3a2c1c" />
        <ellipse cx={1400} cy={868} rx={300} ry={44} fill="#4a3824" />
        {/* 钻完冒苗 */}
        <g transform={`translate(1420,852) scale(${Math.max(0.001, spring({frame: f - (b1 + 285), fps, config: {damping: 9, stiffness: 160}}))})`}
          opacity={clamp01((f - (b1 + 280)) / 20)}>
          <path d="M 0 0 Q -3 -30 0 -48" stroke={P.mint} strokeWidth={6} fill="none" strokeLinecap="round" />
          <path d="M 0 -30 Q -26 -38 -30 -58 Q -8 -52 0 -36" fill={P.mint} />
        </g>
        <text x={1400} y={180} textAnchor="middle" fill={P.mint} fontSize={30}
          fontFamily={FONT_HEAD} letterSpacing={4} opacity={bioK}>仿生种子载体</text>
      </svg>
      <Stamp text="树脂封死 · 火烧才开" x={130} y={140} at={beat(0) + 15} color={P.fire} size={46} rot={-5} />
      <Chip text="大火替它清场" x={130} y={250} at={b1 - 130} accent={P.resin} />
      <Chip text="灾后第一时间撒种" x={120} y={740} at={b1 - 20} accent={P.mint} />
      <Chip text="遇湿自动卷曲" x={1500} y={250} at={b1 + 90} accent={P.wet} />
      <Stamp text="自己往土里钻" x={1480} y={680} at={b1 + 225} color={P.mint} size={46} rot={5} />
      <Chip text="零能源 · 原型是松果" x={1080} y={140} at={b1 + 300} accent={P.dry} />
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
        <ellipse cx={960} cy={1010} rx={1080} ry={120} fill={P.night} />
        <g transform={`translate(700,915) rotate(96) scale(${pop})`}><Pinecone x={0} y={0} s={0.9} openK={0.85} /></g>
        <g transform={`translate(960,935) rotate(-84) scale(${pop})`}><Pinecone x={0} y={0} s={0.8} openK={0.15} /></g>
        <g transform={`translate(1220,915) rotate(70) scale(${pop})`}><Pinecone x={0} y={0} s={0.9} openK={0.6} /></g>
        <Seed x={540 + Math.sin(f / 14) * 30} y={700} s={0.8} rot={-20 + Math.sin(f / 11) * 30} />
        <Seed x={1380 + Math.sin(f / 12) * 30} y={640} s={0.75} rot={140 + Math.sin(f / 10) * 30} />
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -220}}>
        <div style={{fontSize: 70, fontWeight: 800, color: C.text}}>松果：不用电的湿度机器人</div>
        <div style={{fontSize: 38, color: C.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E214 = ({data = tl}: {data?: any}) => {
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
