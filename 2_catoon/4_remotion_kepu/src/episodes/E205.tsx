import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01, rand} from '../lib/style';
import {Backdrop, Ripple, Stamp, SceneTag, Chip, Subtitles, ProgressBar, Finger, Moon, Stars, Rain, FadeIn} from '../lib/atoms';
import tl from '../timeline_205.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 紫夜森林色板 ============ */
const P = {
  sky0: '#2b1d5e',
  sky1: '#0d0a22',
  soil0: '#1d1338',
  soil1: '#0b0718',
  grass: '#3a2c78',
  grassHi: '#584a9e',
  cap: '#9d6bff',
  capDark: '#6b3fd4',
  capRed: '#e05560',
  capPale: '#efe9ff',
  gill: '#e9d9ff',
  stem: '#f3eeff',
  stemDark: '#c2ade9',
  glow: '#55f0d4',
  glowDim: '#2bb8a3',
  violet: '#a78bfa',
};

/* ---------- 参数化蘑菇：origin 在着地点，向上长 ----------
   grow: 0..1 出土进度（可带 spring 过冲做果冻感）
   open: 伞盖打开度，默认跟随 grow                                   */
const Shroom = ({x, y, s = 1, grow = 1, open, cap = P.cap, spots = true, lean = 0, sway = false}: {
  x: number; y: number; s?: number; grow?: number; open?: number;
  cap?: string; spots?: boolean; lean?: number; sway?: boolean;
}) => {
  const f = useCurrentFrame();
  const g = Math.max(0.001, grow);
  const op = open === undefined ? clamp01(g * 1.5 - 0.35) : clamp01(open);
  const rx = 30 + op * 52;
  const rh = 26 + op * 30;
  const idle = sway ? Math.sin(f * 0.05 + x * 0.13) * 1.8 * clamp01(g) : 0;
  const capD = `M ${-rx} 0 C ${-rx} ${-rh * 1.3} ${-rx * 0.5} ${-rh * 1.62} 0 ${-rh * 1.62}
    C ${rx * 0.5} ${-rh * 1.62} ${rx} ${-rh * 1.3} ${rx} 0
    Q ${rx * 0.55} 10 0 10 Q ${-rx * 0.55} 10 ${-rx} 0 Z`;
  const spotPos = [[-0.42, -0.78, 8], [0.06, -1.05, 9], [0.48, -0.7, 7], [-0.08, -0.62, 5.5]];
  return (
    <g transform={`translate(${x},${y}) scale(${s * (0.35 + 0.65 * Math.min(g, 1))},${s * Math.min(Math.max(g, 0.02), 1.15)}) rotate(${lean + idle})`}
      opacity={grow <= 0 ? 0 : 1}>
      {/* 菌柄 */}
      <path d="M -14 0 C -17 -34 -12 -62 -9 -88 L 9 -88 C 12 -62 17 -34 14 0 Z" fill={P.stem} />
      <path d="M 2 0 C 4 -34 3 -62 2 -88 L 9 -88 C 12 -62 17 -34 14 0 Z" fill={P.stemDark} opacity={0.55} />
      {/* 菌褶（伞下扇面） */}
      <g transform={`translate(0,-88) scale(1,${0.35 + op * 0.65})`} opacity={0.35 + op * 0.65}>
        <ellipse cx={0} cy={7} rx={rx * 0.82} ry={9 + op * 6} fill={P.gill} />
        {[-0.66, -0.38, -0.12, 0.12, 0.38, 0.66].map((k, i) => (
          <line key={i} x1={k * rx * 0.3} y1={0} x2={k * rx * 0.82} y2={9 + op * 6}
            stroke={P.capDark} strokeWidth={2.5} opacity={0.55} />
        ))}
      </g>
      {/* 伞盖 */}
      <g transform="translate(0,-88)">
        <path d={capD} fill={cap} stroke={P.capDark} strokeWidth={4} strokeLinejoin="round" />
        <path d={`M ${-rx * 0.7} ${-rh * 0.9} Q 0 ${-rh * 1.75} ${rx * 0.7} ${-rh * 0.9} Q 0 ${-rh * 1.15} ${-rx * 0.7} ${-rh * 0.9} Z`}
          fill="#ffffff" opacity={0.18} />
        {spots && spotPos.map(([sx, sy, sr], i) => (
          <circle key={i} cx={sx * rx} cy={sy * rh} r={sr} fill="#e6dbff" opacity={0.25 + op * 0.6} />
        ))}
      </g>
    </g>
  );
};

/* ---------- 一条菌丝：发光双层描边 + 末端节点，grow 控制生长 ---------- */
const hyphaPath = (mx: number, baseY: number, seed: number, side: number, depth = 3) => {
  let d = `M ${mx} ${baseY}`;
  let cx = mx;
  let cy = baseY;
  for (let k = 0; k < depth; k++) {
    const nx = cx + side * (45 + rand(seed, k) * 140) + (rand(seed, k + 11) - 0.5) * 110;
    const ny = cy + 80 + rand(seed, k + 3) * 75;
    const mxp = cx + (nx - cx) * 0.5 + (rand(seed, k + 5) - 0.5) * 80;
    const myp = cy + (ny - cy) * 0.5;
    d += ` Q ${Math.round(mxp)} ${Math.round(myp)} ${Math.round(nx)} ${Math.round(ny)}`;
    cx = nx;
    cy = ny;
  }
  return {d, ex: cx, ey: cy};
};

const Hypha = ({d, ex, ey, grow, color = P.glow, w = 5}: {
  d: string; ex: number; ey: number; grow: number; color?: string; w?: number;
}) => {
  const f = useCurrentFrame();
  const g = clamp01(grow);
  if (g <= 0) return null;
  const dash = (1 - g) * 100;
  return (
    <g>
      <path d={d} fill="none" stroke={color} strokeWidth={w * 3.4} strokeLinecap="round"
        opacity={0.10 * g} pathLength={100} strokeDasharray={100} strokeDashoffset={dash} />
      <path d={d} fill="none" stroke={color} strokeWidth={w} strokeLinecap="round"
        opacity={0.85} pathLength={100} strokeDasharray={100} strokeDashoffset={dash} />
      {g > 0.9 && (
        <circle cx={ex} cy={ey} r={6 + Math.sin(f / 7 + ex) * 2.2} fill={color} opacity={0.9}>
        </circle>
      )}
    </g>
  );
};

/* ---------- 土里漂浮的养分微粒 ---------- */
const SoilDust = ({n = 16, top = 560}: {n?: number; top?: number}) => {
  const f = useCurrentFrame();
  return (
    <g>
      {Array.from({length: n}).map((_, i) => (
        <circle key={i}
          cx={rand(i, 21) * 1800 + 60 + Math.sin(f / 30 + i) * 12}
          cy={top + 30 + rand(i, 22) * (1060 - top) + Math.cos(f / 26 + i * 2) * 10}
          r={2.5 + rand(i, 23) * 3} fill={C.amber} opacity={0.25 + rand(i, 24) * 0.3} />
      ))}
    </g>
  );
};

/* ---------- 伞下飘孢子 ---------- */
const Spores = ({x, y, on, n = 12}: {x: number; y: number; on: number; n?: number}) => {
  const f = useCurrentFrame();
  if (on <= 0) return null;
  return (
    <g opacity={on}>
      {Array.from({length: n}).map((_, i) => {
        const t = ((f * 0.9 + i * 23) % 140) / 140;
        const sx = x + (rand(i, 31) - 0.5) * 60 + Math.sin(t * 6 + i) * 34 + t * (rand(i, 32) - 0.3) * 130;
        const sy = y + t * 190 + Math.sin(f / 9 + i) * 8;
        return <circle key={i} cx={sx} cy={sy} r={3.5 - t * 2} fill={P.gill} opacity={0.85 * (1 - t)} />;
      })}
    </g>
  );
};

/* ---------- 夜草地地面（S0 用） ---------- */
const Meadow = ({y = 950}: {y?: number}) => (
  <g>
    <ellipse cx={960} cy={y + 130} rx={1500} ry={210} fill="#20164a" />
    <ellipse cx={960} cy={y + 60} rx={1500} ry={150} fill="#2a1e5c" />
    {Array.from({length: 30}).map((_, i) => {
      const gx = rand(i, 41) * 1900 + 10;
      const gy = y - 18 + rand(i, 42) * 60;
      return <path key={i} d={`M ${gx} ${gy} l 6 -26 l 6 26 Z`} fill={P.grassHi} opacity={0.8} />;
    })}
  </g>
);

/* ============ 标题卡 ============ */
const TitleCard = ({data}: {data: any}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 13, mass: 0.7}});
  const pop = spring({frame: f - 6, fps, config: {damping: 9, stiffness: 160}});
  const o = interpolate(f, [0, 8], [0, 1]);
  return (
    <AbsoluteFill>
      <Backdrop top={P.sky0} bottom={P.sky1} glow="#55f0d414" />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={20} from={0} dur={30} />
        {/* 三朵小蘑菇弹出卖萌 */}
        <Shroom x={880} y={400} s={1.15} grow={pop} lean={-5} />
        <Shroom x={975} y={410} s={0.8} grow={Math.max(0, pop - 0.15)} lean={7} cap={P.glow} spots={false} />
        <Shroom x={1040} y={395} s={0.6} grow={Math.max(0, pop - 0.3)} lean={12} />
        <ellipse cx={960} cy={415} rx={200} ry={22} fill="#2a1e5c" />
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 170}}>
          <div style={{color: P.glow, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 26}}>
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

/* ============ S0 开场：雨后夜草地，蘑菇一个个弹出来 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);

  const rainK = interpolate(f, [95, 150], [1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const pops = [300, 520, 760, 1000, 1240, 1480, 1680].map((x, i) => ({
    x, y: 950 - (i % 3) * 26 + rand(i, 51) * 20,
    s: 0.75 + rand(i, 52) * 0.7,
    at: 24 + i * 17 + rand(i, 53) * 10,
    cap: i === 3 ? P.glow : P.cap,
    spots: i !== 3,
    lean: (rand(i, 54) - 0.5) * 14,
  }));

  return (
    <FadeIn>
      <Backdrop top={P.sky0} bottom={P.sky1} glow="#55f0d412" />
      <SceneTag index={0} name={sec.name} accent={P.glow} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={22} from={0} dur={40} />
        <Moon x={1560} y={180} scale={1.1} opacity={0.95} />
        <Rain on={rainK} />
        <Meadow />
        {pops.map((m, i) => {
          const p = spring({frame: f - m.at, fps, config: {damping: 9, stiffness: 170, mass: 0.7}});
          return (
            <g key={i}>
              <Ripple x={m.x} y={m.y} startFrame={m.at} color={P.violet} />
              <Shroom x={m.x} y={m.y} s={m.s} grow={p} cap={m.cap} spots={m.spots} lean={m.lean} sway />
            </g>
          );
        })}
        {/* 最大那朵伞下飘孢子 */}
        <Spores x={1000} y={870} on={interpolate(f, [pops[4].at + 60, pops[4].at + 90], [0, 0.9], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})} />
      </svg>
      <Stamp text="一夜之间？" x={170} y={250} at={beat(0) + 100} color={P.violet} size={58} rot={-6} />
      <Stamp text="不是新来的！" x={1330} y={220} at={beat(1) + 8} color={P.glowDim} size={62} rot={5} />
      <Chip text="它连植物都不是" x={180} y={150} at={beat(3) + 4} accent={P.glow} />
    </FadeIn>
  );
};

/* ============ S1 本体：地下剖面，发光菌丝网铺开（视觉核心） ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);

  const GY = 560; // 地表线
  const shrooms = [640, 960, 1280];
  // 每朵蘑菇下面 5 条菌丝
  const strands = shrooms.flatMap((mx, mi) =>
    [-1.4, -0.7, 0, 0.7, 1.4].map((side, si) => {
      const {d, ex, ey} = hyphaPath(mx, GY + 10, mi * 10 + si, side === 0 ? (mi % 2 ? 1 : -1) * 0.4 : side);
      return {d, ex, ey, delay: mi * 8 + si * 5};
    })
  );
  const growK = clamp01((f - 55) / 150); // 菌丝总生长
  const hub = clamp01((f - 150) / 40);   // 中央母体节点
  const mapK = clamp01((f - beat(1)) / 30); // 俄勒冈 10km² 轮廓

  return (
    <FadeIn>
      <Backdrop top={P.sky0} bottom={P.sky1} glow="#55f0d412" />
      <SceneTag index={1} name={sec.name} accent={P.violet} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <defs>
          <linearGradient id="soilG" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={P.soil0} />
            <stop offset="100%" stopColor={P.soil1} />
          </linearGradient>
          <clipPath id="soilClip"><rect x={0} y={GY} width={1920} height={1080 - GY} /></clipPath>
        </defs>
        {/* 地表：夜空 + 草皮 */}
        <Stars n={16} from={0} dur={30} />
        <rect x={0} y={GY - 14} width={1920} height={18} fill={P.grass} />
        {Array.from({length: 26}).map((_, i) => {
          const gx = 30 + i * 74 + rand(i, 61) * 30;
          return <path key={i} d={`M ${gx} ${GY - 12} l 5 -22 l 5 22 Z`} fill={P.grassHi} opacity={0.75} />;
        })}
        {/* 土层 */}
        <rect x={0} y={GY} width={1920} height={1080 - GY} fill="url(#soilG)" />
        <g clipPath="url(#soilClip)">
          <SoilDust top={GY} />
          {/* 菌丝网 */}
          {strands.map((st, i) => (
            <Hypha key={i} d={st.d} ex={st.ex} ey={st.ey}
              grow={growK * 1.6 - st.delay / 40} w={4.5} />
          ))}
          {/* 蘑菇根部往下的主根 */}
          {shrooms.map((mx, i) => (
            <Hypha key={`r${i}`} d={`M ${mx} ${GY} L ${mx + (i - 1) * 30} ${GY + 70}`}
              ex={mx + (i - 1) * 30} ey={GY + 70} grow={growK * 3} w={7} />
          ))}
          {/* 母体节点 */}
          {hub > 0 && (
            <g transform={`translate(960,${GY + 250})`} opacity={hub}>
              <circle r={46 + Math.sin(f / 8) * 6} fill={P.glow} opacity={0.16} />
              <circle r={20 + Math.sin(f / 8) * 3} fill={P.glow} />
              <text x={110} y={12} fill={P.glow} fontSize={32} fontFamily={FONT_HEAD}
                letterSpacing={4}>同一个本体</text>
            </g>
          )}
          {/* 俄勒冈尺度：巨大虚线轮廓 */}
          {mapK > 0 && (
            <g opacity={mapK}>
              <rect x={170} y={GY + 40} width={1580} height={470} rx={80}
                fill="none" stroke={P.violet} strokeWidth={5} strokeDasharray="18 14" />
              <text x={960} y={GY + 92} textAnchor="middle" fill={P.violet} fontSize={44}
                fontFamily={FONT_HEAD} letterSpacing={8} opacity={0.9}>1 株 ≈ 10 km²</text>
            </g>
          )}
        </g>
        {/* 地表上的蘑菇：只是"果" */}
        {shrooms.map((mx, i) => {
          const p = spring({frame: f - 12 - i * 9, fps, config: {damping: 10, stiffness: 160}});
          return <Shroom key={i} x={mx} y={GY} s={0.85 + i * 0.12} grow={p} lean={(i - 1) * 6} sway />;
        })}
      </svg>
      <Chip text="蘑菇 = 果 · 本体在地下" x={180} y={160} at={beat(0) + 30} accent={P.violet} />
      <Stamp text="俄勒冈 · 上千年" x={1260} y={170} at={beat(1) + 8} color={P.glowDim} size={50} rot={4} />
    </FadeIn>
  );
};

/* ============ S2 开工令：细胞吸水膨大，蘑菇几小时顶出土 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);

  const rainK = interpolate(f, [0, 24], [0, 1], {extrapolateRight: 'clamp'});
  const rain2 = interpolate(f, [beat(2) - 10, beat(2) + 16], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});

  // 蘑菇：先是小鼓包，beat(1) 后吸水猛长
  const bud = 0.16;
  const popS = spring({frame: f - beat(1) - 12, fps, config: {damping: 10, stiffness: 110, mass: 0.9}});
  const grow = bud + (1 - bud) * clamp01(popS);

  // 细胞剖面
  const CX = 1420, CY = 540, R = 235;
  const inflate = interpolate(f, [beat(1) + 18, beat(1) + 85], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});

  return (
    <FadeIn>
      <Backdrop top="#241a52" bottom={P.sky1} glow="#55f0d410" />
      <SceneTag index={2} name={sec.name} accent={P.glow} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <defs>
          <clipPath id="tissueClip"><circle cx={CX} cy={CY} r={R - 16} /></clipPath>
        </defs>
        <Rain on={rainK * (0.5 + rain2 * 0.5)} n={rain2 > 0.5 ? 70 : 46} />
        {/* 地面 */}
        <rect x={0} y={760} width={1920} height={320} fill={P.soil1} />
        <rect x={0} y={748} width={1920} height={16} fill={P.grass} />
        {/* 主角蘑菇：破土 */}
        <ellipse cx={560} cy={768} rx={150} ry={26} fill="#241a52" />
        <Shroom x={560} y={764} s={1.9} grow={grow} cap={P.cap} sway />
        <Ripple x={560} y={760} startFrame={beat(1) + 12} color={P.glow} />
        {/* 放大虚线 */}
        <circle cx={560} cy={640} r={60} fill="none" stroke={P.glow} strokeWidth={3.5} strokeDasharray="9 9" opacity={0.8} />
        <line x1={620} y1={620} x2={CX - R * 0.8} y2={CY - R * 0.6} stroke={P.glow} strokeWidth={3.5}
          strokeDasharray="9 9" opacity={0.6} />
        {/* 组织剖面：细胞吸水膨大 */}
        <circle cx={CX} cy={CY} r={R} fill="#140c2e" stroke={P.violet} strokeWidth={6} />
        <circle cx={CX} cy={CY} r={R + 16} fill="none" stroke={P.violet} strokeWidth={3} opacity={0.3} />
        <g clipPath="url(#tissueClip)">
          {Array.from({length: 7}).map((_, i) => {
            const a = (i / 7) * Math.PI * 2 - Math.PI / 2;
            const rr = i === 0 ? 0 : 118;
            const cx = CX + Math.cos(a) * rr;
            const cy = CY + Math.sin(a) * rr;
            const sc = 0.5 + 0.5 * clamp01(inflate * 1.3 - i * 0.05);
            return (
              <g key={i} transform={`translate(${cx},${cy}) scale(${sc})`}>
                <ellipse rx={72} ry={60} fill={P.violet} />
                <ellipse cy={14} rx={58} ry={34} fill={P.capDark} opacity={0.45} />
                <circle cx={-12} cy={-12} r={11} fill="#dcd0ff" opacity={0.85} />
              </g>
            );
          })}
          {/* 灌进来的水滴 */}
          {inflate > 0 && inflate < 1 && Array.from({length: 12}).map((_, i) => {
            const drop = (f * 5 + i * 57) % 320;
            return <circle key={i} cx={CX - 160 + rand(i, 71) * 320} cy={CY - R + drop}
              r={8} fill={P.glow} opacity={0.8} />;
          })}
        </g>
        <text x={CX} y={CY - R - 30} textAnchor="middle" fill={P.violet} fontSize={44}
          fontFamily={FONT_HEAD} fontWeight={700}>菌丝细胞</text>
      </svg>
      <Chip text="九成是水" x={180} y={170} at={beat(1) - 6} accent={P.glow} />
      <Chip text="吸水膨大，不是分裂" x={180} y={262} at={beat(1) + 50} accent={P.violet} />
      <Stamp text="开工令！" x={1350} y={860} at={beat(2) + 6} color={P.glowDim} size={58} rot={-6} />
    </FadeIn>
  );
};

/* ============ S3 仙女环：蘑菇排成圈 + 地下菌丝边缘发光 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);

  const CXR = 960, CYR = 580;
  const R = interpolate(f, [beat(1), beat(1) + 45], [140, 330], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const ringO = clamp01((f - beat(1) + 10) / 25);
  const nextK = clamp01((f - beat(2)) / 40); // 明年的圈
  const N = 9;

  return (
    <FadeIn>
      <Backdrop top="#231a50" bottom={P.sky1} glow="#a78bfa14" />
      <SceneTag index={3} name={sec.name} accent={P.violet} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 俯视草地 */}
        <ellipse cx={CXR} cy={CYR + 60} rx={1050} ry={520} fill="#241a52" />
        {Array.from({length: 40}).map((_, i) => (
          <circle key={i} cx={rand(i, 81) * 1920} cy={140 + rand(i, 82) * 880}
            r={3 + rand(i, 83) * 4} fill={P.grassHi} opacity={0.3} />
        ))}
        {/* 中心死区：营养被吃完 */}
        <ellipse cx={CXR} cy={CYR} rx={R * 0.55} ry={R * 0.55 * 0.62} fill="#160e33" opacity={0.75} />
        {/* 活跃菌丝边缘：发光环（与蘑菇同一椭圆透视） */}
        {ringO > 0 && (
          <g opacity={ringO}>
            <ellipse cx={CXR} cy={CYR} rx={R} ry={R * 0.62} fill="none" stroke={P.glow} strokeWidth={26} opacity={0.10} />
            <ellipse cx={CXR} cy={CYR} rx={R} ry={R * 0.62} fill="none" stroke={P.glow}
              strokeWidth={6 + Math.sin(f / 8) * 1.6} opacity={0.85} />
          </g>
        )}
        {/* 蘑菇沿圈挨个弹出 */}
        {Array.from({length: N}).map((_, i) => {
          const a = -Math.PI / 2 + (i / N) * Math.PI * 2;
          const mx = CXR + Math.cos(a) * 330;
          const my = CYR + Math.sin(a) * 330 * 0.62; // 椭圆透视
          const p = spring({frame: f - beat(1) - 14 - i * 7, fps, config: {damping: 9, stiffness: 165}});
          return <Shroom key={i} x={mx} y={my} s={0.62 + (i % 3) * 0.08} grow={p}
            lean={(rand(i, 91) - 0.5) * 10} />;
        })}
        {/* 明年：更外一圈虚线 */}
        {nextK > 0 && (
          <g opacity={nextK}>
            <ellipse cx={CXR} cy={CYR} rx={432} ry={432 * 0.62} fill="none" stroke={P.violet}
              strokeWidth={5} strokeDasharray="16 14" />
            <path d={`M ${CXR + 352} ${CYR} L ${CXR + 416} ${CYR}`} stroke={P.violet} strokeWidth={6} />
            <path d={`M ${CXR + 416} ${CYR} l -18 -12 M ${CXR + 416} ${CYR} l -18 12`}
              stroke={P.violet} strokeWidth={6} fill="none" />
          </g>
        )}
      </svg>
      <Stamp text="仙女环" x={210} y={210} at={beat(0) + 60} color={P.violet} size={60} rot={-6} />
      <Chip text="活的菌丝只在最外圈" x={1270} y={180} at={beat(1) + 80} accent={P.glow} />
      <Stamp text="每年往外挪一圈" x={1230} y={870} at={beat(2) + 14} color={P.capDark} size={48} rot={4} />
    </FadeIn>
  );
};

/* ============ S4 收尾：别乱吃 + 冰山一角 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);

  const noK = clamp01((f - beat(1) - 20) / 12);          // 禁止符号
  const fx = interpolate(f, [beat(1) - 14, beat(1) + 16], [1750, 1290], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const fback = interpolate(f, [beat(1) + 30, beat(1) + 70], [0, 200], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const phase2 = clamp01((f - beat(2)) / 40);           // 冰山一角段
  const GY = 560;

  const strands = [700, 1000].flatMap((mx, mi) =>
    [-1, -0.4, 0.4, 1].map((side, si) => {
      const {d, ex, ey} = hyphaPath(mx, GY + 8, 40 + mi * 7 + si, side, 3);
      return {d, ex, ey, delay: mi * 6 + si * 4};
    })
  );

  return (
    <FadeIn>
      <Backdrop top={P.sky0} bottom={P.sky1} glow="#55f0d412" />
      <SceneTag index={4} name={sec.name} accent={P.glow} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={14} from={0} dur={30} />
        {/* 后半段：剖面地层压上来 */}
        <g opacity={phase2}>
          <rect x={0} y={GY} width={1920} height={1080 - GY} fill={P.soil1} />
          <rect x={0} y={GY - 12} width={1920} height={14} fill={P.grass} />
          <SoilDust top={GY} />
          {strands.map((st, i) => (
            <Hypha key={i} d={st.d} ex={st.ex} ey={st.ey}
              grow={phase2 * 2.2 - st.delay / 30} w={5} />
          ))}
          {/* 巨大的本体光团 */}
          <ellipse cx={850} cy={960} rx={520} ry={130} fill={P.glow} opacity={0.07 + Math.sin(f / 10) * 0.02} />
        </g>
        {/* 草地（前半段地面） */}
        <g opacity={1 - phase2}>
          <Meadow y={800} />
        </g>
        {/* 两朵蘑菇：艳丽红伞 + 朴素白伞，位置跨越前后两段 */}
        <Shroom x={700} y={780 - phase2 * (780 - GY)} s={1.25} grow={1} cap={P.capRed} sway />
        <Shroom x={1000} y={800 - phase2 * (800 - GY)} s={1.05} grow={1} cap={P.capPale} spots={false} sway />
        {/* 禁止符号：弹两个 */}
        {noK > 0 && [700, 1000].map((mx, i) => {
          const my = (i === 0 ? 640 : 665) - phase2 * 180;
          const s2 = spring({frame: f - beat(1) - 20 - i * 8, fps, config: {damping: 10, stiffness: 180}});
          return (
            <g key={i} transform={`translate(${mx},${my}) scale(${s2})`} opacity={noK * (1 - phase2 * 0.4)}>
              <circle r={64} fill="none" stroke={C.coral} strokeWidth={13} />
              <line x1={-44} y1={-44} x2={44} y2={44} stroke={C.coral} strokeWidth={13} strokeLinecap="round" />
            </g>
          );
        })}
        {/* 手伸过来又缩回去 */}
        <g opacity={1 - phase2}>
          <Finger x={fx + fback} y={690 + fback * 0.4} rot={-30} scale={1.05} />
        </g>
        {/* 冰山一角的标注线 */}
        {phase2 > 0.4 && (
          <g opacity={(phase2 - 0.4) / 0.6}>
            <text x={1360} y={GY - 40} fill={P.gill} fontSize={34} fontFamily={FONT_HEAD} letterSpacing={4}>你看到的</text>
            <text x={1360} y={GY + 330} fill={P.glow} fontSize={34} fontFamily={FONT_HEAD} letterSpacing={4}>真正的本体</text>
          </g>
        )}
      </svg>
      <Stamp text="一朵都别吃" x={1320} y={200} at={beat(1) + 26} color={C.coral} size={56} rot={6} />
      <Stamp text="冰山一角" x={200} y={210} at={beat(2) + 34} color={P.glowDim} size={54} rot={-5} />
    </FadeIn>
  );
};

/* ============ 结尾卡 ============ */
const EndCard = () => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 14}});
  const pop = spring({frame: f - 4, fps, config: {damping: 10, stiffness: 150}});
  return (
    <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
      <Backdrop top={P.sky0} bottom={P.sky1} glow="#55f0d416" />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <ellipse cx={960} cy={1010} rx={900} ry={130} fill="#20164a" />
        <Shroom x={880} y={990} s={0.9} grow={pop} lean={-6} />
        <Shroom x={985} y={995} s={0.7} grow={Math.max(0, pop - 0.15)} lean={8} cap={P.glow} spots={false} />
        <Shroom x={1060} y={988} s={0.55} grow={Math.max(0, pop - 0.3)} lean={14} />
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -160}}>
        <div style={{fontSize: 74, fontWeight: 800, color: C.text}}>下次雨后，低头看看脚下</div>
        <div style={{fontSize: 38, color: C.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E205 = ({data = tl}: {data?: any}) => {
  const {fps} = useVideoConfig();
  return (
    <AbsoluteFill style={{background: P.sky1}}>
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
