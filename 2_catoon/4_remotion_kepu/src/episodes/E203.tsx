import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01, rand} from '../lib/style';
import {Backdrop, Ripple, Stamp, SceneTag, Chip, Subtitles, ProgressBar, Sun, Moon, Stars, FadeIn} from '../lib/atoms';
import tl from '../timeline_203.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};
type TL = {title: string; episode: string; titleCardSec: number; endCardSec: number; totalSec: number; sections: Sec[]};

/* ---------- 沙漠色系 ---------- */
const CACTUS = '#55b07c';   // 茎绿
const CACTUS_D = '#35795a'; // 棱沟
const SPINE = '#f6e5bd';    // 刺（干枯的叶子）
const PAD = '#e8c98d';      // 刺座垫
const EUPH = '#7ec85f';     // 大戟冒牌货（更鲜绿）
const DUSK_T = '#3a2a5e';
const DUSK_B = '#e0714f';
const DUNE1 = '#4a2a55';
const DUNE2 = '#36204a';

/* ============ 柱状仙人掌（参数化：茎 + 棱 + 刺座 + 放射刺） ============ */
const Cactus = ({x, y, scale = 1, w = 190, h = 460, arms = false, seed = 0,
  tint = CACTUS, dark = CACTUS_D, pad = PAD, spine = SPINE, sway = true,
  areoleEvery = 66, spineLen = 30}: {
  x: number; y: number; scale?: number; w?: number; h?: number; arms?: boolean; seed?: number;
  tint?: string; dark?: string; pad?: string; spine?: string; sway?: boolean;
  areoleEvery?: number; spineLen?: number;
}) => {
  const f = useCurrentFrame();
  const swayA = sway ? Math.sin(f * 0.04 + seed * 2.1) * 1.4 : 0;
  const cols = [-0.32, 0, 0.32];
  const rows = Math.max(2, Math.floor((h - 60) / areoleEvery));
  return (
    <g transform={`translate(${x},${y}) scale(${scale}) rotate(${swayA})`}>
      <defs>
        <clipPath id={`cb${seed}`}>
          <rect x={-w / 2} y={-h} width={w} height={h} rx={w / 2} />
        </clipPath>
      </defs>
      {/* 侧臂：先伸出去再拐头向上 */}
      {arms && (<>
        <path d={`M ${-w / 2 + 8} ${-h * 0.45} Q ${-w * 0.92} ${-h * 0.46} ${-w * 0.92} ${-h * 0.74}`}
          stroke={tint} strokeWidth={w * 0.4} fill="none" strokeLinecap="round" />
        <path d={`M ${w / 2 - 8} ${-h * 0.56} Q ${w * 0.88} ${-h * 0.57} ${w * 0.88} ${-h * 0.82}`}
          stroke={tint} strokeWidth={w * 0.38} fill="none" strokeLinecap="round" />
        <path d={`M ${-w / 2 + 8} ${-h * 0.45} Q ${-w * 0.92} ${-h * 0.46} ${-w * 0.92} ${-h * 0.74}`}
          stroke={dark} strokeWidth={w * 0.4} fill="none" strokeLinecap="round" opacity={0.18}
          transform="translate(10,0)" />
      </>)}
      {/* 茎身 */}
      <rect x={-w / 2} y={-h} width={w} height={h} rx={w / 2} fill={tint} />
      <g clipPath={`url(#cb${seed})`}>
        {Array.from({length: 6}).map((_, i) => (
          <line key={i} x1={-w / 2 + (i + 0.5) * (w / 6)} y1={-h} x2={-w / 2 + (i + 0.5) * (w / 6)} y2={0}
            stroke={dark} strokeWidth={w * 0.055} opacity={0.42} />
        ))}
        <rect x={w * 0.12} y={-h} width={w * 0.38} height={h} fill={dark} opacity={0.16} />
        <rect x={-w / 2} y={-h} width={w * 0.16} height={h} fill="#ffffff" opacity={0.10} />
      </g>
      {/* 刺座 + 放射刺 */}
      {cols.map((u, ci) =>
        Array.from({length: rows}).map((_, r) => {
          const ay = -34 - r * areoleEvery - (ci % 2) * areoleEvery * 0.5;
          if (ay < -h + 30) return null;
          const fan = -90 + u * 165;
          return (
            <g key={`${ci}-${r}`} transform={`translate(${u * w},${ay})`}>
              {[(-24), 0, 24].map((k) => {
                const a = (fan + k) * Math.PI / 180;
                const L = spineLen * (0.8 + rand(ci * 31 + r * 7 + k + 40, seed) * 0.45);
                return <line key={k} x1={0} y1={0} x2={Math.cos(a) * L} y2={Math.sin(a) * L}
                  stroke={spine} strokeWidth={3.6} strokeLinecap="round" />;
              })}
              <ellipse rx={8.5} ry={10.5} fill={pad} />
              <ellipse cx={-2} cy={-2} rx={4} ry={5} fill="#fff" opacity={0.35} />
            </g>
          );
        })
      )}
    </g>
  );
};

/* ============ 球状仙人掌 ============ */
const BallCactus = ({x, y, r = 92, scale = 1, tint = CACTUS, dark = CACTUS_D, seed = 5}: {
  x: number; y: number; r?: number; scale?: number; tint?: string; dark?: string; seed?: number;
}) => {
  const spots: {px: number; py: number}[] = [];
  for (let i = 0; i < 12; i++) {
    const a = rand(i, seed) * Math.PI * 2;
    const rr = r * (0.3 + rand(i, seed + 3) * 0.58);
    spots.push({px: Math.cos(a) * rr, py: Math.sin(a) * rr});
  }
  return (
    <g transform={`translate(${x},${y}) scale(${scale})`}>
      <defs>
        <clipPath id={`bb${seed}`}><circle r={r} /></clipPath>
      </defs>
      <circle r={r} fill={tint} />
      <g clipPath={`url(#bb${seed})`}>
        {[-2, -1, 0, 1, 2].map((i) => (
          <line key={i} x1={i * r * 0.36} y1={-r} x2={i * r * 0.36} y2={r}
            stroke={dark} strokeWidth={r * 0.08} opacity={0.4} />
        ))}
        <circle cx={r * 0.3} cy={r * 0.25} r={r} fill={dark} opacity={0.15} />
      </g>
      {spots.map((s, i) => {
        const a = Math.atan2(s.py, s.px);
        return (
          <g key={i} transform={`translate(${s.px},${s.py})`}>
            {[(-22), 0, 22].map((k) => {
              const aa = a + k * Math.PI / 180;
              const L = 22 + rand(i + k + 30, seed) * 14;
              return <line key={k} x1={0} y1={0} x2={Math.cos(aa) * L} y2={Math.sin(aa) * L}
                stroke={SPINE} strokeWidth={3} strokeLinecap="round" />;
            })}
            <circle r={6.5} fill={PAD} />
          </g>
        );
      })}
      <circle cx={-r * 0.32} cy={-r * 0.4} r={r * 0.2} fill="#ffffff" opacity={0.14} />
    </g>
  );
};

/* ============ 大戟冒牌货：茎上直接长刺，没有刺座 ============ */
const Euphorbia = ({x, y, scale = 1, latex = 0}: {x: number; y: number; scale?: number; latex?: number}) => {
  const f = useCurrentFrame();
  const branches = [
    {x1: 0, y1: -30, x2: 0, y2: -300, w: 74},
    {x1: -8, y1: -190, x2: -110, y2: -330, w: 44},
    {x1: 8, y1: -160, x2: 100, y2: -290, w: 44},
    {x1: -110, y1: -330, x2: -140, y2: -420, w: 34},
    {x1: 100, y1: -290, x2: 128, y2: -380, w: 34},
  ];
  // 沿每条枝摆成对的皮刺（直接从茎皮冒出）
  const thorns: {x: number; y: number; a: number}[] = [];
  branches.forEach((b, bi) => {
    const n = Math.max(2, Math.floor(Math.hypot(b.x2 - b.x1, b.y2 - b.y1) / 60));
    for (let i = 1; i <= n; i++) {
      const t = i / (n + 1);
      const px = b.x1 + (b.x2 - b.x1) * t;
      const py = b.y1 + (b.y2 - b.y1) * t;
      const dir = Math.atan2(b.y2 - b.y1, b.x2 - b.x1);
      [-1, 1].forEach((s) => thorns.push({x: px, y: py, a: dir + s * Math.PI / 2 + (rand(bi * 13 + i * 5 + s, 9) - 0.5) * 0.3}));
    }
  });
  const drip = clamp01(latex);
  return (
    <g transform={`translate(${x},${y}) scale(${scale})`}>
      {branches.map((b, i) => (
        <line key={i} x1={b.x1} y1={b.y1} x2={b.x2} y2={b.y2}
          stroke={EUPH} strokeWidth={b.w} strokeLinecap="round" />
      ))}
      {branches.map((b, i) => (
        <line key={`d${i}`} x1={b.x1 + 8} y1={b.y1} x2={b.x2 + 8} y2={b.y2}
          stroke="#4d8f3f" strokeWidth={b.w * 0.35} strokeLinecap="round" opacity={0.5} />
      ))}
      {thorns.map((t, i) => (
        <g key={i} transform={`translate(${t.x},${t.y}) rotate(${t.a * 180 / Math.PI - 90})`}>
          <path d="M0 0 L -7 -22 L 7 -22 Z" fill="#3e5a33" />
        </g>
      ))}
      {/* 断口 + 毒奶 */}
      <circle cx={0} cy={-300} r={20} fill="#d9ecd0" />
      {drip > 0 && [0, 1].map((i) => {
        const t = ((f * 1 + i * 40) % 90) / 90;
        return <circle key={i} cx={i ? 6 : -4} cy={-296 + t * t * 300}
          r={9 - t * 3} fill="#f4f7ee" opacity={drip * (1 - t * 0.7)} />;
      })}
    </g>
  );
};

/* ============ 花盆 ============ */
const Pot = ({x, y, w = 250, h = 150}: {x: number; y: number; w?: number; h?: number}) => (
  <g transform={`translate(${x},${y})`}>
    <path d={`M ${-w / 2} 0 L ${w / 2} 0 L ${w / 2 - 30} ${h} L ${-w / 2 + 30} ${h} Z`} fill="#c96f4a" />
    <path d={`M ${-w / 2} 0 L ${w / 2} 0 L ${w / 2} 26 L ${-w / 2} 26 Z`} fill="#a95738" opacity={0.7} />
    <rect x={-w / 2 - 14} y={-36} width={w + 28} height={40} rx={12} fill="#d9825f" />
    <rect x={-w / 2 - 14} y={-36} width={w + 28} height={12} rx={6} fill="#e99a72" />
  </g>
);

/* ============ 沙丘地面 ============ */
const Dunes = () => (
  <g>
    <ellipse cx={430} cy={1150} rx={950} ry={270} fill={DUNE1} />
    <ellipse cx={1450} cy={1210} rx={1050} ry={300} fill={DUNE2} />
    <ellipse cx={960} cy={1240} rx={800} ry={220} fill="#2c1a3e" />
  </g>
);

/* ============ 标题卡 ============ */
const TitleCard = ({data}: {data: TL}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 13, mass: 0.7}});
  const o = interpolate(f, [0, 8], [0, 1]);
  return (
    <AbsoluteFill>
      <Backdrop top={DUSK_T} bottom={DUSK_B} glow="#ffc94d14" />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Sun x={1560} y={760} r={70} opacity={0.9} />
        <Dunes />
        {/* 小图标：刺座上爆出一把小刺 + 后面一片淡淡的叶子残影 */}
        <g transform="translate(960,300)">
          <ellipse cx={-6} cy={40} rx={46} ry={64} fill={C.leaf} opacity={0.28} />
          {[(-66), (-38), (-12), 14, 40].map((k, i) => {
            const a = (-92 + k) * Math.PI / 180;
            const pop = spring({frame: f - 10 - i * 3, fps, config: {damping: 12}});
            return <line key={k} x1={0} y1={0} x2={Math.cos(a) * 92 * pop} y2={Math.sin(a) * 92 * pop}
              stroke={SPINE} strokeWidth={7} strokeLinecap="round" />;
          })}
          <ellipse rx={22} ry={26} fill={PAD} />
        </g>
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 160}}>
          <div style={{color: C.amber, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 26}}>
            冷知识 · 第 {data.episode} 期
          </div>
          <div style={{color: C.text, fontSize: 100, fontWeight: 900, fontFamily: FONT_HEAD, letterSpacing: 6}}>
            {data.title}
          </div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ============ S0 开场：黄昏沙漠里的仙人掌，刺的特写 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round(((sec.cues[i]?.t0 ?? sec.start + 2 + i * 3) - sec.start) * fps);

  const sunY = interpolate(f, [0, 200], [560, 720], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  // 放大镜里的"叶子→刺"一闪而过
  const morph = clamp01(interpolate(f, [beat(1) + 8, beat(1) + 48], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}));

  return (
    <FadeIn>
      <Backdrop top={DUSK_T} bottom={DUSK_B} glow="#ffc94d12" />
      <SceneTag index={0} name={sec.name} accent={C.amber} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Sun x={380} y={sunY} r={85} />
        <Stars n={16} from={180} dur={60} />
        <Dunes />
        {/* 主角：巨人柱 */}
        <Cactus x={1260} y={975} scale={1.15} w={200} h={520} arms seed={1} spineLen={34} />
        <BallCactus x={520} y={930} r={95} seed={5} />
        <Cactus x={250} y={1010} scale={0.62} w={150} h={380} seed={2} />
        {/* 刺簇特写放大圈 */}
        <circle cx={1195} cy={430} r={86} fill="none" stroke={C.amber} strokeWidth={4}
          strokeDasharray="12 10" opacity={0.95} />
        {f > beat(1) && (
          <g>
            <circle cx={620} cy={330} r={120} fill="#14203f" stroke={C.amber} strokeWidth={6} />
            <line x1={700} y1={390} x2={1130} y2={450} stroke={C.amber} strokeWidth={4} strokeDasharray="10 10" opacity={0.7} />
            {/* 圈里：叶子横着缩成一根刺 */}
            <g transform="translate(500,330)">
              <path d={`M0 0 C ${62 * (1 - morph) + 6} ${-52 * (1 - morph) - 4} ${170 * (1 - morph) + 30} ${-52 * (1 - morph) - 4} 240 0
                C ${170 * (1 - morph) + 30} ${52 * (1 - morph) + 4} ${62 * (1 - morph) + 6} ${52 * (1 - morph) + 4} 0 0 Z`}
                fill={C.leaf} opacity={1 - morph * 0.85} />
              <path d="M0 -7 L 245 0 L 0 7 Z" fill={SPINE} opacity={morph} />
            </g>
          </g>
        )}
      </svg>
      <Stamp text="这是叶子！" x={360} y={140} at={beat(1) + 14} color={C.coral} size={58} rot={-7} />
      <Stamp text="为啥改成刺？" x={1180} y={150} at={beat(2) + 4} color="#6b4a9e" size={52} rot={5} />
    </FadeIn>
  );
};

/* ============ S1 刺座：放大镜看小圆垫，刺一根根爆出来 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round(((sec.cues[i]?.t0 ?? sec.start + 2 + i * 3) - sec.start) * fps);

  const CX = 1300, CY = 520, R = 290;
  const popStart = beat(1) + 6;
  const flowerAt = beat(2) + 4;

  return (
    <FadeIn>
      <Backdrop top="#241d4e" bottom={C.bgDeep} glow="#63e2c612" />
      <SceneTag index={1} name={sec.name} accent={C.cell} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <defs>
          <clipPath id="areoleClip"><circle cx={CX} cy={CY} r={R - 14} /></clipPath>
        </defs>

        {/* 左：迷你仙人掌 + 虚线圈定位刺座 */}
        <ellipse cx={560} cy={968} rx={200} ry={30} fill={DUNE2} />
        <Cactus x={560} y={950} scale={0.62} w={190} h={460} seed={3} />
        <circle cx={560 + 0.32 * 190 * 0.62} cy={950 - 226 * 0.62} r={62} fill="none"
          stroke={C.amber} strokeWidth={4} strokeDasharray="10 10" />
        <line x1={560 + 0.32 * 190 * 0.62 + 56} y1={950 - 226 * 0.62 - 20} x2={CX - R * 0.8} y2={CY + R * 0.3}
          stroke={C.amber} strokeWidth={4} strokeDasharray="10 10" opacity={0.7} />

        {/* 右：放大镜下的刺座 */}
        <circle cx={CX} cy={CY} r={R} fill="#14203f" stroke={C.cell} strokeWidth={6} />
        <circle cx={CX} cy={CY} r={R + 14} fill="none" stroke={C.cell} strokeWidth={3} opacity={0.3} />
        <g clipPath="url(#areoleClip)">
          {/* 茎皮底色 */}
          <circle cx={CX} cy={CY} r={R - 14} fill="#2a4a3a" />
          {/* 大垫子 */}
          <ellipse cx={CX} cy={CY} rx={120} ry={95} fill={PAD} />
          <ellipse cx={CX - 20} cy={CY - 18} rx={80} ry={55} fill="#f2dda8" opacity={0.8} />
          {/* 垫子上的绒毛点 */}
          {Array.from({length: 16}).map((_, i) => (
            <circle key={i} cx={CX + (rand(i, 21) - 0.5) * 200} cy={CY + (rand(i, 22) - 0.5) * 150}
              r={5 + rand(i, 23) * 4} fill="#c9a86a" opacity={0.6} />
          ))}
          {/* 刺一根根爆出 */}
          {[(-168), (-138), (-108), (-78), (-48), (-18), 12, 42].map((deg, i) => {
            const a = deg * Math.PI / 180;
            const pop = spring({frame: f - popStart - i * 7, fps, config: {damping: 10, stiffness: 200}});
            const L = (170 + rand(i, 31) * 70) * pop;
            return (
              <line key={i} x1={CX + Math.cos(a) * 60} y1={CY + Math.sin(a) * 46}
                x2={CX + Math.cos(a) * (60 + L)} y2={CY + Math.sin(a) * (46 + L)}
                stroke={SPINE} strokeWidth={10} strokeLinecap="round" opacity={pop > 0 ? 1 : 0} />
            );
          })}
          {/* 花和果也从这里开 */}
          {f > flowerAt && (
            <g transform={`translate(${CX + 100},${CY - 110})`}>
              {[0, 72, 144, 216, 288].map((deg, i) => {
                const ps = spring({frame: f - flowerAt - i * 2, fps, config: {damping: 11}});
                return <ellipse key={deg} cx={Math.cos(deg * Math.PI / 180) * 30 * ps}
                  cy={Math.sin(deg * Math.PI / 180) * 30 * ps} rx={24 * ps} ry={14 * ps}
                  fill={C.petal} transform={`rotate(${deg})`} />;
              })}
              <circle r={14 * clamp01((f - flowerAt) / 10)} fill={C.amber} />
            </g>
          )}
        </g>
        <text x={CX} y={CY - R - 36} textAnchor="middle" fill={C.cell}
          fontSize={54} fontFamily={FONT_HEAD} fontWeight={700}>刺座 areole</text>
        <Ripple x={CX} y={CY} startFrame={popStart - 4} color={C.cell} />
      </svg>
      <Stamp text="独家身份证" x={220} y={170} at={beat(1) + 40} color={C.coral} size={52} rot={-6} />
      <Chip text="花·果·新茎都从这开" x={150} y={720} at={flowerAt + 8} accent={C.cell} />
    </FadeIn>
  );
};

/* ============ S2 改造：一片绿叶收缩变形成刺 + 四个岗位 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round(((sec.cues[i]?.t0 ?? sec.start + 2 + i * 3) - sec.start) * fps);

  // 叶子 → 刺 的连续 morph（控制点插值）
  const p = clamp01(interpolate(f, [30, 150], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}));
  const LEN = 240 + p * 40;
  const HH = 56 * (1 - p) + 5;
  const leafPath = `M0 0 C ${LEN * 0.26} ${-HH} ${LEN * 0.7} ${-HH} ${LEN} 0 C ${LEN * 0.7} ${HH} ${LEN * 0.26} ${HH} 0 0 Z`;

  const jobs = [
    {icon: '省水', color: C.water, y: 210},
    {icon: '防身', color: C.coral, y: 350},
    {icon: '遮阴', color: C.amber, y: 490},
    {icon: '导水', color: C.sky, y: 630},
  ];

  return (
    <FadeIn>
      <Backdrop top="#2c1f52" bottom="#8a4a44" glow="#ffc94d10" />
      <SceneTag index={2} name={sec.name} accent={C.amber} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Dunes />
        {/* 左侧主舞台：一小段茎 + 正在变形的叶子 */}
        <ellipse cx={560} cy={880} rx={300} ry={40} fill={DUNE2} />
        <rect x={330} y={560} width={70} height={320} rx={35} fill={CACTUS} />
        <rect x={356} y={560} width={22} height={320} fill={CACTUS_D} opacity={0.4} />
        <ellipse cx={386} cy={640} rx={16} ry={20} fill={PAD} />
        {/* 变形主体：锚在刺座上向右伸出 */}
        <g transform="translate(390,640)">
          <path d={leafPath} fill={C.leaf} />
          <path d={leafPath} fill={SPINE} opacity={p} />
          <line x1={6} y1={0} x2={LEN - 14} y2={0} stroke={C.leafDark} strokeWidth={6}
            opacity={0.5 * (1 - p)} strokeLinecap="round" />
          {/* 蒸发的水汽 */}
          {p > 0.08 && p < 0.95 && Array.from({length: 7}).map((_, i) => {
            const t = ((f * 1.6 + i * 17) % 60) / 60;
            return <circle key={i} cx={60 + rand(i, 41) * (LEN - 90)} cy={-20 - t * 120}
              r={7 - t * 4} fill={C.water} opacity={(1 - t) * 0.7 * Math.sin(p * Math.PI)} />;
          })}
        </g>

        {/* 右侧四个岗位图标牌 */}
        {jobs.map((j, i) => {
          const at = beat(1) + 10 + i * 22;
          const ss = spring({frame: f - at, fps, config: {damping: 12}});
          const o = interpolate(f, [at, at + 5], [0, 1], {extrapolateLeft: 'clamp'});
          if (o <= 0) return null;
          return (
            <g key={i} transform={`translate(1280,${j.y}) scale(${ss})`} opacity={o}>
              <rect x={-70} y={-52} width={340} height={104} rx={26} fill="rgba(20,24,50,0.8)"
                stroke={j.color} strokeWidth={4} />
              <circle cx={-16} cy={0} r={34} fill={j.color} opacity={0.25} />
              <circle cx={-16} cy={0} r={34} fill="none" stroke={j.color} strokeWidth={3} />
              <text x={52} y={14} fill={C.text} fontSize={40} fontFamily={FONT_HEAD} fontWeight={700}>{j.icon}</text>
              {/* 小图标 */}
              {i === 0 && <path d="M-16 -18 C -2 -2 -2 12 -16 18 C -30 12 -30 -2 -16 -18 Z" fill={C.water} />}
              {i === 1 && <path d="M-16 -20 L 2 -13 L 2 4 C 2 15 -16 21 -16 21 C -16 21 -34 15 -34 4 L -34 -13 Z" fill={C.coral} />}
              {i === 2 && (<g>
                <path d={`M-38 2 A 22 22 0 0 1 6 2 Z`} fill={C.amber} />
                <line x1={-16} y1={2} x2={-16} y2={22} stroke={C.amber} strokeWidth={4} strokeLinecap="round" />
              </g>)}
              {i === 3 && (<g>
                <path d="M-24 -16 C -14 -6 -14 4 -24 9 C -34 4 -34 -6 -24 -16 Z" fill={C.sky} />
                <line x1={-8} y1={-14} x2={-8} y2={16} stroke={C.sky} strokeWidth={4} strokeLinecap="round" />
                <path d="M-14 8 L -8 17 L -2 8" stroke={C.sky} strokeWidth={4} fill="none" strokeLinecap="round" />
              </g>)}
            </g>
          );
        })}
      </svg>
      <Stamp text="大叶子 = 蒸发器" x={280} y={200} at={beat(0) + 20} color={C.coral} size={48} rot={-5} />
      <Stamp text="一根针，四个岗！" x={1050} y={780} at={beat(2) + 4} color={C.amber} size={54} rot={-4} />
    </FadeIn>
  );
};

/* ============ S3 夜班：夜里气孔开门吸 CO2，存成苹果酸 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round(((sec.cues[i]?.t0 ?? sec.start + 2 + i * 3) - sec.start) * fps);

  // 日→夜切换发生在 cue1 前（"白天关死，夜里开门"）
  const night = interpolate(f, [beat(1) - 50, beat(1) + 20], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const open = interpolate(f, [beat(1) + 14, beat(1) + 40], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const gap = 6 + open * 52;
  const appleAt = beat(1) + 150;

  const CX = 560, CY = 540, R = 260;
  return (
    <FadeIn>
      <Backdrop top="#2c2154" bottom="#7a4a52" glow="#ffc94d10" />
      <div style={{position: 'absolute', inset: 0, opacity: night}}>
        <Backdrop top={C.night0} bottom={C.night1} glow="#5cc8ff10" />
      </div>
      <SceneTag index={3} name={sec.name} accent={C.sky} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={24} from={beat(1) - 50} dur={50} />
        <Sun x={1700} y={interpolate(night, [0, 1], [230, 560])} r={60} opacity={1 - night} />
        <Moon x={1560} y={interpolate(night, [0, 1], [120, 210])} opacity={night} />
        <Dunes />
        <Cactus x={1450} y={990} scale={1.05} w={200} h={500} arms seed={4} spineLen={30} />
        <BallCactus x={1700} y={960} r={70} seed={8} />

        {/* 左：茎皮放大圈里的气孔 */}
        <circle cx={CX} cy={CY} r={R} fill="#16304a" stroke={C.sky} strokeWidth={6} />
        <circle cx={CX} cy={CY} r={R - 14} fill="#2a4a3a" />
        {/* 气孔：两瓣保卫细胞，夜里张开 */}
        <g transform={`translate(${CX},${CY})`}>
          <ellipse cx={0} cy={0} rx={open * 36} ry={open * 60} fill="#0d2036" />
          <ellipse cx={-gap / 2 - 56} cy={0} rx={58} ry={88} fill={C.cell}
            transform={`rotate(${-8 - open * 10} ${-gap / 2 - 56} 0)`} />
          <ellipse cx={gap / 2 + 56} cy={0} rx={58} ry={88} fill={C.cell}
            transform={`rotate(${8 + open * 10} ${gap / 2 + 56} 0)`} />
          <ellipse cx={-gap / 2 - 56} cy={18} rx={44} ry={56} fill={C.cellDark} opacity={0.5}
            transform={`rotate(${-8 - open * 10} ${-gap / 2 - 56} 0)`} />
          <ellipse cx={gap / 2 + 56} cy={18} rx={44} ry={56} fill={C.cellDark} opacity={0.5}
            transform={`rotate(${8 + open * 10} ${gap / 2 + 56} 0)`} />
        </g>
        {/* CO2 圆点飘进气孔 */}
        {open > 0.4 && Array.from({length: 6}).map((_, i) => {
          const t = ((f * 1.1 + i * 26) % 110) / 110;
          const sx = CX - R - 140 + t * (R + 140);
          const sy = CY + Math.sin(i * 2.1) * 150 * (1 - t);
          return (
            <g key={i} opacity={t < 0.9 ? 1 : 0}>
              <circle cx={sx} cy={sy} r={17} fill={C.sky} opacity={0.9} />
              <text x={sx} y={sy + 6} textAnchor="middle" fill="#0d2036" fontSize={17} fontWeight={800}>CO₂</text>
            </g>
          );
        })}
        <text x={CX + 200} y={CY - R - 30} textAnchor="middle" fill={C.sky}
          fontSize={48} fontFamily={FONT_HEAD} fontWeight={700}>气孔 · 夜班窗口</text>

        {/* 苹果酸仓库 */}
        {f > appleAt && (
          <g transform={`translate(${CX + 60},${CY + R + 60})`}>
            {Array.from({length: 4}).map((_, i) => {
              const ps = spring({frame: f - appleAt - i * 6, fps, config: {damping: 12}});
              return (
                <g key={i} transform={`translate(${i * 74},0) scale(${ps})`}>
                  <circle r={26} fill={C.coral} />
                  <path d={`M0 -24 Q 10 -38 24 -34`} stroke={C.stemDark} strokeWidth={6} fill="none" strokeLinecap="round" />
                </g>
              );
            })}
          </g>
        )}
        <Ripple x={CX} y={CY} startFrame={beat(1) + 10} color={C.sky} />
      </svg>
      <Stamp text="白天关死" x={240} y={160} at={beat(1) - 46} color={C.coral} size={46} rot={-6} />
      <Stamp text="夜里开门吸碳" x={240} y={250} at={beat(1) + 30} color={C.sky} size={46} rot={-3} />
      <Chip text="存成苹果酸，白天慢慢用" x={330} y={920} at={appleAt + 10} accent={C.coral} />
      <Stamp text="代价：长得慢" x={1350} y={140} at={beat(2) + 6} color="#6b4a9e" size={50} rot={5} />
    </FadeIn>
  );
};

/* ============ S4 认亲：花市真仙人掌 vs 大戟冒牌货 + 怕涝 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round(((sec.cues[i]?.t0 ?? sec.start + 2 + i * 3) - sec.start) * fps);

  const okAt = 80;      // 真货打勾
  const badAt = 190;    // 冒牌打叉
  const waterAt = beat(1) + 10; // "就怕涝"
  const waterK = interpolate(f, [waterAt, waterAt + 20, waterAt + 150, waterAt + 170],
    [0, 1, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});

  return (
    <FadeIn>
      <Backdrop top="#1e1a3e" bottom="#4a2b52" glow="#ffc94d10" />
      <SceneTag index={4} name={sec.name} accent={C.amber} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 货架台面 */}
        <rect x={0} y={905} width={1920} height={175} fill="#3a2244" />
        <rect x={0} y={905} width={1920} height={10} fill="#5a3560" />

        {/* 左：真仙人掌 */}
        <Pot x={560} y={905} />
        <Cactus x={560} y={872} scale={0.8} w={170} h={420} seed={6} sway={false} />
        {/* 脉冲圈盯住刺座 */}
        {f > 40 && f < okAt + 60 && (
          <circle cx={560 + 0.32 * 170 * 0.8} cy={872 - 200 * 0.8} r={44 + Math.sin(f / 6) * 6}
            fill="none" stroke={C.cell} strokeWidth={5} />
        )}
        {/* 打勾 */}
        {f > okAt && (
          <g transform={`translate(710,470) scale(${spring({frame: f - okAt, fps, config: {damping: 11}})})`}>
            <circle r={52} fill={C.leaf} />
            <path d="M-24 2 L -6 22 L 28 -18" stroke="#fff" strokeWidth={12} fill="none" strokeLinecap="round" />
          </g>
        )}

        {/* 右：大戟冒牌货 */}
        <Pot x={1380} y={905} />
        <Euphorbia x={1380} y={872} scale={0.78} latex={f > badAt ? 1 : 0} />
        {f > badAt && (
          <g transform={`translate(1590,300) scale(${spring({frame: f - badAt, fps, config: {damping: 11}})})`}>
            <circle r={52} fill={C.coral} />
            <path d="M-20 -20 L 20 20 M 20 -20 L -20 20" stroke="#fff" strokeWidth={12} strokeLinecap="round" />
          </g>
        )}

        {/* 火龙果：正经亲戚的小牌子 */}
        {f > 260 && (
          <g transform={`translate(960,410) scale(${spring({frame: f - 260, fps, config: {damping: 12}})})`}>
            <ellipse rx={62} ry={44} fill="#ff5f9e" />
            {[(-40), 0, 40].map((dx) => (
              <path key={dx} d={`M${dx} -30 Q ${dx + 16} -48 ${dx + 6} -60 L ${dx - 2} -40 Z`} fill={C.leaf} />
            ))}
            <ellipse rx={62} ry={44} fill="none" stroke="#ff8fbe" strokeWidth={4} />
          </g>
        )}

        {/* 浇水壶 + 水滴 */}
        {waterK > 0 && (
          <g transform={`translate(${560 + interpolate(waterK, [0, 1], [480, 140])},${340}) rotate(${-18 * waterK})`} opacity={waterK}>
            <rect x={-70} y={-46} width={140} height={92} rx={20} fill="#5a8fd6" />
            <path d="M-70 -10 L -150 -50 L -140 -66 L -60 -28 Z" fill="#5a8fd6" />
            <rect x={60} y={-58} width={24} height={34} rx={10} fill="#5a8fd6" />
            <circle cx={-148} cy={-58} r={10} fill="#4a7bc0" />
          </g>
        )}
        {waterK > 0.6 && Array.from({length: 8}).map((_, i) => {
          const t = ((f * 2 + i * 9) % 50) / 50;
          return <circle key={i} cx={500 + rand(i, 51) * 60} cy={330 + t * t * 400}
            r={8 - t * 3} fill={C.water} opacity={1 - t * 0.8} />;
        })}
        {/* 涝害：仙人掌蔫掉 */}
        {f > waterAt + 100 && (
          <g transform="translate(560,872) scale(0.8)" opacity={clamp01((f - waterAt - 100) / 30)}>
            <ellipse cx={0} cy={-60} rx={130} ry={26} fill="#7a4a35" opacity={0.7} />
          </g>
        )}
      </svg>
      <Stamp text="有刺座 = 真货" x={180} y={150} at={okAt + 8} color={C.cell} size={48} rot={-5} />
      <Stamp text="没刺座 = 冒牌" x={1330} y={150} at={badAt + 8} color={C.coral} size={48} rot={5} />
      <Chip text="火龙果也是仙人掌科" x={840} y={480} at={272} accent="#ff5f9e" />
      <Stamp text="不怕旱，就怕涝！" x={760} y={140} at={waterAt + 24} color={C.water} size={52} rot={-4} />
    </FadeIn>
  );
};

/* ============ 结尾卡 ============ */
const EndCard = () => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 14}});
  return (
    <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
      <Backdrop top={DUSK_T} bottom={DUSK_B} glow="#ffc94d12" />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Dunes />
        <Cactus x={960} y={1090} scale={0.85} w={180} h={430} arms seed={7} />
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -160}}>
        <div style={{fontSize: 74, fontWeight: 800, color: C.text}}>有刺座的，才是仙人掌</div>
        <div style={{fontSize: 38, color: C.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E203 = ({data = tl}: {data?: any}) => {
  const {fps} = useVideoConfig();
  const d = data as TL;
  return (
    <AbsoluteFill style={{background: C.bgDeep}}>
      <Sequence durationInFrames={Math.round(d.titleCardSec * fps)}>
        <TitleCard data={d} />
      </Sequence>

      {d.sections.map((sec, i) => {
        const Scene = SCENES[i];
        return (
          <Sequence key={i} from={Math.round(sec.start * fps)} durationInFrames={Math.round(sec.dur * fps)}>
            <Scene sec={sec} />
            <Audio src={staticFile(sec.audio)} />
          </Sequence>
        );
      })}

      <Sequence from={Math.round((d.totalSec - d.endCardSec) * fps)} durationInFrames={Math.round(d.endCardSec * fps)}>
        <EndCard />
      </Sequence>

      <Subtitles data={d} />
      <ProgressBar />
    </AbsoluteFill>
  );
};
