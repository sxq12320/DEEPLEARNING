import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01, rand} from '../lib/style';
import {Backdrop, Ripple, Stamp, SceneTag, Chip, Subtitles, ProgressBar, Bug, FadeIn} from '../lib/atoms';
import tl from '../timeline_206.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 洋葱厨房色板：紫白洋葱 + 暖棕砧板 + 青色泪滴 + 硫黄气体 ============ */
const O = {
  bg: '#3a2a56',       // 深紫茄色底（平涂）
  bgAlt: '#33244c',
  bgDeep2: '#2b1e42',
  board: '#c98a54',    // 砧板暖棕
  boardDark: '#a56a3a',
  skin: '#9b59c9',     // 洋葱紫皮
  skinDeep: '#7a3fa8',
  flesh: '#f6effc',    // 洋葱肉（淡紫白）
  fleshLine: '#c9a8e8',
  gas: '#ffd166',      // 挥发硫化物
  gasDeep: '#f0a840',
  tear: '#5cc8ff',     // 泪滴
  tearDeep: '#2e9fd8',
  enzyme: '#ff8fa3',   // 蒜酶粉
  amino: '#ffe08a',    // 蒜氨酸黄
  steel: '#e4e9f6',
  steelDark: '#9aa3c0',
  handle: '#4a3358',
  leaf: '#7fd08c',
  leafDark: '#4ea368',
  panel: '#241a3e',
  eyeWhite: '#fdfaff',
  iris: '#8a5fc9',
};

/* ---------- 三次贝塞尔取点 ---------- */
const cubic = (p0: number[], p1: number[], p2: number[], p3: number[], t: number) => {
  const u = 1 - t;
  return [
    u * u * u * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t * t * t * p3[0],
    u * u * u * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t * t * t * p3[1],
  ];
};

/* ---------- 半颗洋葱（切口朝上，同心圈） ---------- */
const OnionHalf = ({x, y, s = 1, wob = 0}: {x: number; y: number; s?: number; wob?: number}) => {
  const f = useCurrentFrame();
  const w = Math.sin(f / 16) * wob;
  return (
    <g transform={`translate(${x},${y}) rotate(${w}) scale(${s})`}>
      <ellipse rx={136} ry={104} fill={O.skin} />
      <ellipse cy={-4} rx={122} ry={90} fill={O.flesh} />
      {[0.78, 0.56, 0.34].map((k, i) => (
        <ellipse key={i} cy={-4} rx={122 * k} ry={90 * k} fill="none" stroke={O.fleshLine} strokeWidth={6} />
      ))}
      <ellipse cy={-4} rx={122 * 0.13} ry={90 * 0.13} fill={O.fleshLine} opacity={0.7} />
      {/* 高光 */}
      <path d="M -90 -50 Q -40 -78 30 -72" stroke="#fff" strokeWidth={9} fill="none" strokeLinecap="round" opacity={0.55} />
    </g>
  );
};

/* ---------- 整颗洋葱（球茎，origin 在中心） ---------- */
const OnionBulb = ({x, y, s = 1, rot = 0, hiRoot = 0}: {x: number; y: number; s?: number; rot?: number; hiRoot?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`}>
    {/* 芽 */}
    <path d="M -3 -86 L -12 -124 L -3 -114 Z" fill={O.leafDark} />
    <path d="M 2 -86 L 14 -120 L 5 -110 Z" fill={O.leaf} />
    {/* 球茎 */}
    <path d="M 0 -90 C 54 -84 88 -44 88 8 C 88 62 48 90 0 90 C -48 90 -88 62 -88 8 C -88 -44 -54 -84 0 -90 Z"
      fill={O.flesh} stroke={O.skin} strokeWidth={7} />
    {/* 纵向条纹 */}
    {[[-58, 70], [-26, 84], [26, 84], [58, 70]].map(([ex, ey], i) => (
      <path key={i} d={`M 0 -84 Q ${ex * 0.7} 0 ${ex} ${ey}`} fill="none" stroke={O.fleshLine} strokeWidth={5} opacity={0.85} />
    ))}
    {/* 根盘 */}
    <ellipse cy={90} rx={30} ry={12} fill={O.boardDark} />
    {[0, 1, 2].map((i) => (
      <line key={i} x1={-14 + i * 14} y1={98} x2={-18 + i * 16} y2={112} stroke={O.boardDark} strokeWidth={4} strokeLinecap="round" />
    ))}
    {hiRoot > 0 && (
      <g opacity={hiRoot}>
        <circle cy={94} r={46} fill="none" stroke={C.coral} strokeWidth={6} strokeDasharray="12 10" />
        <text x={0} y={164} textAnchor="middle" fill={C.coral} fontSize={30} fontFamily={FONT_HEAD} fontWeight={700}>弹药最浓</text>
      </g>
    )}
  </g>
);

/* ---------- 菜刀（刀尖朝下） ---------- */
const Knife = ({x, y, rot = 0, s = 1, dull = false}: {x: number; y: number; rot?: number; s?: number; dull?: boolean}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`}>
    {/* 柄 */}
    <rect x={-19} y={-228} width={38} height={70} rx={14} fill={O.handle} />
    <circle cy={-206} r={4.5} fill={O.steelDark} />
    <circle cy={-178} r={4.5} fill={O.steelDark} />
    {/* 护手 */}
    <rect x={-24} y={-162} width={48} height={14} rx={6} fill={O.steelDark} />
    {/* 刀身 */}
    <path d={dull
      ? 'M -16 -148 L 16 -148 L 16 -34 L 10 -22 L 14 -12 L 2 0 L -8 -6 L -16 -12 Z'
      : 'M -16 -148 L 16 -148 L 16 -30 L 0 4 L -16 -8 Z'}
      fill={dull ? O.steelDark : O.steel} />
    {!dull && <line x1={-6} y1={-140} x2={-6} y2={-22} stroke="#fff" strokeWidth={5} strokeLinecap="round" opacity={0.7} />}
  </g>
);

/* ---------- 泪滴 ---------- */
const TearDrop = ({x, y, s = 1, o = 1}: {x: number; y: number; s?: number; o?: number}) => (
  <path transform={`translate(${x},${y}) scale(${s})`} opacity={o}
    d="M 0 -15 C 9 -2 12 6 0 13 C -12 6 -9 -2 0 -15 Z" fill={O.tear} />
);

/* ---------- 杏眼：cry 0..1 流泪强度 ---------- */
const Eye = ({x, y, s = 1, cry = 0, iris = O.iris}: {x: number; y: number; s?: number; cry?: number; iris?: string}) => {
  const f = useCurrentFrame();
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      <path d="M -132 0 Q 0 -88 132 0 Q 0 88 -132 0 Z" fill={O.eyeWhite} />
      <circle r={48} fill={iris} />
      <circle r={24} fill="#241738" />
      <circle cx={15} cy={-15} r={9} fill="#fff" />
      {/* 泪光：下睑积泪 */}
      {cry > 0 && <path d="M -110 18 Q 0 80 110 18 Q 0 62 -110 18 Z" fill={O.tear} opacity={0.35 + cry * 0.4} />}
      {/* 掉泪 */}
      {cry > 0.05 && Array.from({length: 5}).map((_, i) => {
        const sp = 3.6 + rand(i, 3) * 2.4;
        const t = ((f * sp + i * 47) % 150) / 150;
        const dx = [-70, -24, 30, 78, 0][i];
        return (
          <TearDrop key={i} x={dx + Math.sin(t * 6 + i) * 5} y={62 + t * 200} s={1 + rand(i, 4) * 0.7}
            o={cry * (t < 0.85 ? 1 : (1 - t) / 0.15)} />
        );
      })}
    </g>
  );
};

/* ---------- 气体弹道：虚线轨迹 + 循环飞行的硫黄粒子 ---------- */
const GasTrail = ({p0, p1, p2, p3, n = 9, delay = 0, dur = 100, showPath = true, on = 1, r = 9}: {
  p0: number[]; p1: number[]; p2: number[]; p3: number[];
  n?: number; delay?: number; dur?: number; showPath?: boolean; on?: number; r?: number;
}) => {
  const f = useCurrentFrame();
  const d = `M ${p0[0]} ${p0[1]} C ${p1[0]} ${p1[1]} ${p2[0]} ${p2[1]} ${p3[0]} ${p3[1]}`;
  const pathO = interpolate(f, [delay, delay + 18], [0, 0.55], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  return (
    <g opacity={on}>
      {showPath && <path d={d} fill="none" stroke={O.gas} strokeWidth={5} strokeDasharray="4 18"
        strokeLinecap="round" opacity={pathO} />}
      {f > delay && Array.from({length: n}).map((_, i) => {
        const t = (((f - delay) + i * (dur / n)) % dur) / dur;
        const [gx, gy] = cubic(p0, p1, p2, p3, t);
        const o = Math.sin(Math.PI * clamp01(t)) * 0.95;
        return <circle key={i} cx={gx} cy={gy} r={r * (0.7 + rand(i, 9) * 0.5)} fill={i % 3 === 2 ? O.gasDeep : O.gas} opacity={o} />;
      })}
    </g>
  );
};

/* ---------- 蒜酶（吃豆人形） ---------- */
const Pac = ({x, y, rot = 0, s = 1, color = O.enzyme}: {x: number; y: number; rot?: number; s?: number; color?: string}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`}>
    <path d="M 0 0 L 30 -17 A 34 34 0 1 0 30 17 Z" fill={color} />
    <circle cx={8} cy={-14} r={4.5} fill="#fff" opacity={0.85} />
  </g>
);

/* ---------- 催泪因子分子：丙基骨架 - CH=S=O ---------- */
const LFMolecule = ({x, y, s = 1, o = 1, label = false}: {x: number; y: number; s?: number; o?: number; label?: boolean}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={o}>
    <polyline points="-150,10 -110,-14 -70,10 -30,-14" fill="none" stroke={C.muted} strokeWidth={7} strokeLinecap="round" strokeLinejoin="round" />
    {/* 双键 S=O */}
    <line x1={8} y1={-22} x2={56} y2={-22} stroke={C.text} strokeWidth={6} strokeLinecap="round" />
    <line x1={8} y1={-6} x2={56} y2={-6} stroke={C.text} strokeWidth={6} strokeLinecap="round" />
    <circle cx={-10} cy={-14} r={24} fill={O.gas} />
    <text x={-10} y={-4} textAnchor="middle" fontSize={28} fontWeight={800} fill="#5c3a10" fontFamily={FONT_HEAD}>S</text>
    <circle cx={82} cy={-14} r={22} fill={O.enzyme} />
    <text x={82} y={-5} textAnchor="middle" fontSize={26} fontWeight={800} fill="#5c1030" fontFamily={FONT_HEAD}>O</text>
    {label && (
      <text x={-30} y={70} textAnchor="middle" fill={O.gas} fontSize={34} fontFamily={FONT_HEAD} fontWeight={700}>
        催泪因子 · 丙硫醛硫氧化物
      </text>
    )}
  </g>
);

/* ============ 标题卡 ============ */
const TitleCard = ({data}: {data: any}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 13, mass: 0.7}});
  const pop = spring({frame: f - 5, fps, config: {damping: 10, stiffness: 160}});
  const o = interpolate(f, [0, 8], [0, 1]);
  return (
    <AbsoluteFill>
      <Backdrop top={O.bg} bottom={O.bg} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <g transform={`translate(960,330) scale(${Math.max(0.001, pop)})`}>
          <OnionBulb x={0} y={0} s={1.05} />
          <TearDrop x={104} y={-52} s={1.8} o={0.95} />
          <TearDrop x={126} y={-18} s={1.15} o={0.8} />
        </g>
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 190}}>
          <div style={{color: O.tear, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 26}}>
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

/* ============ S0 开场：刀落洋葱，气体直奔眼睛 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);

  const chop = 62;                                    // 刀落下帧
  const ky = interpolate(f, [chop - 22, chop], [-260, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const bounce = f > chop ? Math.sin((f - chop) / 5) * Math.max(0, 10 - (f - chop)) * 0.8 : 0;
  const gasOn = clamp01((f - (chop + 26)) / 20);
  const cryK = clamp01((f - (beat(1) + 160)) / 60);

  // 切口飞溅汁液粒子
  const splash = f - chop;
  return (
    <FadeIn>
      <Backdrop top={O.bg} bottom={O.bg} />
      <SceneTag index={0} name={sec.name} accent={O.skin} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 砧板 */}
        <rect x={0} y={760} width={1920} height={120} fill={O.board} />
        <rect x={0} y={760} width={1920} height={14} fill={O.boardDark} opacity={0.7} />
        <rect x={0} y={874} width={1920} height={10} fill={O.boardDark} />

        {/* 半颗洋葱 + 切下的一片 */}
        <OnionHalf x={640} y={688} s={1.5} />
        <ellipse cx={810} cy={742} rx={58} ry={16} fill={O.flesh} stroke={O.fleshLine} strokeWidth={5}
          transform={`rotate(${f > chop ? 12 : 0} 810 742)`} />

        {/* 飞溅汁液 */}
        {splash > 0 && splash < 40 && Array.from({length: 12}).map((_, i) => {
          const a = -Math.PI * (0.15 + rand(i, 14) * 0.7);
          const v = 14 + rand(i, 15) * 14;
          const px = 700 + Math.cos(a) * v * splash * (rand(i, 16) > 0.5 ? 1 : -1) * 0.5;
          const py = 640 + Math.sin(a) * v * splash + splash * splash * 0.55;
          return <circle key={i} cx={px} cy={py} r={5 + rand(i, 17) * 5}
            fill={i % 3 ? O.flesh : O.gas} opacity={clamp01(1 - splash / 38)} />;
        })}
        <Ripple x={700} y={655} startFrame={chop} color={O.gas} />

        {/* 刀 */}
        <Knife x={706} y={ky + 560 + bounce} rot={-8} s={1.25} />

        {/* 气体弹道：洋葱 → 眼睛 */}
        <GasTrail p0={[700, 600]} p1={[880, 300]} p2={[1180, 210]} p3={[1390, 360]}
          delay={chop + 26} dur={110} n={10} on={gasOn} />
        <GasTrail p0={[640, 560]} p1={[760, 200]} p2={[1050, 140]} p3={[1420, 330]}
          delay={chop + 44} dur={130} n={7} on={gasOn} showPath={false} r={7} />

        {/* 右眼：被命中后流泪 */}
        <Eye x={1520} y={400} s={1.7} cry={cryK} />
        <text x={1130} y={660} textAnchor="middle" fill={O.tear} fontSize={34} fontFamily={FONT_HEAD}
          opacity={cryK} fontWeight={700}>泪腺全力放水</text>
      </svg>
      <Stamp text="不是气味！" x={200} y={220} at={beat(0) + 55} color={C.coral} size={60} rot={-7} />
      <Stamp text="闻不出来的催泪弹" x={150} y={330} at={beat(1) + 30} color={O.skinDeep} size={50} rot={-4} />
      <Chip text="化学防御系统" x={1420} y={170} at={beat(2) + 8} accent={O.gas} />
    </FadeIn>
  );
};

/* ============ S1 武器库：弹药与扳机分开放，刀落膜破 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);

  const CX = 1230, CY = 500, R = 300;
  const cut = beat(1) + 12;                             // 刀劈开细胞
  const cutK = clamp01((f - cut) / 10);                  // 膜破进度
  const mixK = clamp01((f - cut - 12) / 40);             // 酶扩散
  const gasK = clamp01((f - cut - 46) / 30);             // 气体外逸
  const knifeX = interpolate(f, [cut - 10, cut + 14], [CX - R - 340, CX + R + 80], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});

  // 细胞质里的蒜氨酸弹药（黄点）
  const amino = Array.from({length: 15}).map((_, i) => ({
    x: CX + (rand(i, 21) - 0.5) * 2 * (R - 60),
    y: CY + (rand(i, 22) - 0.5) * 2 * (R - 60),
    r: 10 + rand(i, 23) * 6,
  })).filter(p => Math.hypot(p.x - (CX - 80), p.y - (CY + 90)) > 150 && Math.hypot(p.x - CX, p.y - CY) < R - 60);

  // 液泡里的蒜酶
  const VX = CX - 80, VY = CY + 90, VR = 118;
  const enzymeHome = [0, 1, 2, 3, 4].map((i) => {
    const a = (i / 5) * Math.PI * 2 + 0.6;
    return {hx: VX + Math.cos(a) * 52, hy: VY + Math.sin(a) * 52,
      tx: CX + Math.cos(a * 1.7 + 1) * (R - 90), ty: CY + Math.sin(a * 1.7 + 1) * (R - 100),
      rot: i * 72};
  });

  return (
    <FadeIn>
      <Backdrop top={O.bgAlt} bottom={O.bgAlt} />
      <SceneTag index={1} name={sec.name} accent={O.enzyme} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <defs>
          <clipPath id="c206"><circle cx={CX} cy={CY} r={R - 16} /></clipPath>
        </defs>

        {/* 左：迷你洋葱 + 放大虚线 */}
        <OnionBulb x={500} y={600} s={1.5} />
        <rect x={330} y={770} width={340} height={90} rx={16} fill={O.board} opacity={0.9} />
        <circle cx={545} cy={560} r={70} fill="none" stroke={O.gas} strokeWidth={4} strokeDasharray="10 10" />
        <line x1={610} y1={530} x2={CX - R * 0.78} y2={CY - R * 0.55} stroke={O.gas} strokeWidth={4}
          strokeDasharray="10 10" opacity={0.7} />

        {/* 大细胞 */}
        <circle cx={CX} cy={CY} r={R} fill="#46336e" stroke={O.fleshLine} strokeWidth={7} />
        <circle cx={CX} cy={CY} r={R + 16} fill="none" stroke={O.fleshLine} strokeWidth={3} opacity={0.3} />
        <g clipPath="url(#c206)">
          {/* 细胞质纹理 */}
          {Array.from({length: 10}).map((_, i) => (
            <circle key={i} cx={CX + (rand(i, 31) - 0.5) * 2 * R} cy={CY + (rand(i, 32) - 0.5) * 2 * R}
              r={26 + rand(i, 33) * 30} fill="#3c2a5e" opacity={0.6} />
          ))}
          {/* 蒜氨酸弹药 */}
          {amino.map((p, i) => (
            <g key={i} transform={`translate(${p.x + Math.sin(f / 14 + i) * 5},${p.y + Math.cos(f / 12 + i) * 5})`}>
              <circle r={p.r} fill={O.amino} />
              <circle r={p.r * 0.4} fill="#fff" opacity={0.6} />
            </g>
          ))}
          {/* 液泡 */}
          <circle cx={VX} cy={VY} r={VR} fill="#2f2050"
            stroke={O.enzyme} strokeWidth={6} strokeDasharray={cutK > 0 ? 'none' : 'none'} opacity={1 - cutK * 0.9} />
          <circle cx={VX} cy={VY} r={VR} fill="none" stroke={O.enzyme} strokeWidth={5} opacity={1 - cutK} />
          {/* 蒜酶：膜破后扩散到全细胞 */}
          {enzymeHome.map((e, i) => {
            const ex = e.hx + (e.tx - e.hx) * mixK;
            const ey = e.hy + (e.ty - e.hy) * mixK;
            return <Pac key={i} x={ex} y={ey} rot={e.rot + mixK * 140} s={1} />;
          })}
          {/* 合成出的刺激气体向上飘 */}
          {gasK > 0 && Array.from({length: 10}).map((_, i) => {
            const t = ((f * 2.4 + i * 37) % 200) / 200;
            return <circle key={i} cx={CX - 140 + rand(i, 41) * 300 + Math.sin(f / 9 + i) * 14}
              cy={CY - 60 - t * (R + 160)} r={8 + rand(i, 42) * 5}
              fill={O.gas} opacity={gasK * Math.sin(Math.PI * clamp01(t))} />;
          })}
        </g>
        {/* 切口白闪 */}
        {f > cut - 2 && f < cut + 14 && (
          <line x1={CX - R} y1={CY - R * 0.4} x2={CX + R} y2={CY + R * 0.4} stroke="#fff"
            strokeWidth={14} strokeLinecap="round" opacity={clamp01(1 - (f - cut) / 12)} />
        )}
        <Ripple x={CX} y={CY} startFrame={cut} color={O.enzyme} />
        {/* 劈下来的刀 */}
        {f > cut - 12 && f < cut + 26 && (
          <g opacity={clamp01(1 - (f - cut - 16) / 8)}>
            <Knife x={knifeX} y={CY - 60} rot={52} s={0.9} />
          </g>
        )}

        {/* 标注 */}
        <line x1={CX - 250} y1={CY - 200} x2={CX - 130} y2={CY - 130} stroke={O.amino} strokeWidth={4} />
        <text x={CX - 262} y={CY - 214} textAnchor="end" fill={O.amino} fontSize={34}
          fontFamily={FONT_HEAD} fontWeight={700}>细胞质 · 蒜氨酸</text>
        <line x1={VX - 120} y1={VY + 130} x2={VX - 60} y2={VY + 80} stroke={O.enzyme} strokeWidth={4} />
        <text x={VX - 150} y={VY + 175} textAnchor="middle" fill={O.enzyme} fontSize={34}
          fontFamily={FONT_HEAD} fontWeight={700} opacity={1 - cutK}>液泡 · 蒜酶</text>
        <text x={CX} y={CY + R + 54} textAnchor="middle" fill={C.muted} fontSize={30}
          fontFamily={FONT_HEAD}>洋葱细胞（剖面）</text>
      </svg>
      <Chip text="弹药和扳机分开放" x={170} y={170} at={beat(0) + 30} accent={O.amino} />
      <Stamp text="膜破了！" x={1460} y={170} at={cut + 8} color={C.coral} size={56} rot={6} />
      <Stamp text="芥菜同款思路" x={200} y={800} at={beat(2) + 6} color={O.leafDark} size={44} rot={-5} />
    </FadeIn>
  );
};

/* ============ S2 产线：催泪因子合成酶把中间产物改造成催泪因子 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);

  const MX = 800;                    // 机器中心
  const beltY = 660;
  const cycle = 95;                  // 加工节拍
  const stampStart = 46;
  const ph = ((f - stampStart) % cycle + cycle) % cycle / cycle;
  const armY = ph < 0.16 ? interpolate(ph, [0, 0.16], [0, 58]) : ph < 0.5 ? interpolate(ph, [0.16, 0.5], [58, 0]) : 0;
  const flash = ph > 0.14 && ph < 0.3 ? 1 - (ph - 0.14) / 0.16 : 0;

  // 传送带上的中间产物（灰紫团）
  const items = Array.from({length: 6}).map((_, i) => 240 + ((f * 2.1 + i * 300) % 1500));
  // 成品：冲压时刻生成，向上飘
  const made = Math.max(0, Math.floor((f - stampStart) / cycle));

  return (
    <FadeIn>
      <Backdrop top={O.bg} bottom={O.bg} />
      <SceneTag index={2} name={sec.name} accent={O.gas} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 传送带 */}
        <rect x={180} y={beltY} width={1560} height={70} rx={35} fill={O.panel} stroke={O.fleshLine} strokeWidth={4} />
        {Array.from({length: 16}).map((_, i) => {
          const bx = 210 + ((f * 2.1 + i * 100) % 1500);
          return <line key={i} x1={bx} y1={beltY + 14} x2={bx} y2={beltY + 56} stroke={O.fleshLine}
            strokeWidth={4} opacity={0.35} />;
        })}

        {/* 中间产物团：到机器下方消失 */}
        {items.map((ix, i) => {
          if (ix < 240 || ix > 1720) return null;
          const inMachine = Math.abs(ix - MX) < 70;
          return (
            <g key={i} transform={`translate(${ix},${beltY + 2})`} opacity={inMachine ? 0 : 1}>
              <circle cy={-20} r={22} fill={O.steelDark} />
              <circle cx={-18} cy={-12} r={15} fill={O.steelDark} />
              <circle cx={17} cy={-12} r={15} fill={O.steelDark} />
            </g>
          );
        })}

        {/* 机器：催泪因子合成酶 */}
        <g transform={`translate(${MX},${beltY - 210})`}>
          <rect x={-130} y={0} width={260} height={150} rx={26} fill={O.enzyme} />
          <rect x={-130} y={0} width={260} height={46} rx={20} fill="#ffb3c1" />
          {/* 工程师表情 */}
          <circle cx={-46} cy={80} r={13} fill="#5c1030" />
          <circle cx={46} cy={80} r={13} fill="#5c1030" />
          <path d="M -22 108 Q 0 122 22 108" stroke="#5c1030" strokeWidth={7} fill="none" strokeLinecap="round" />
          {/* 冲压臂 */}
          <g transform={`translate(0,${150 + armY})`}>
            <rect x={-15} y={0} width={30} height={52} fill={O.handle} />
            <rect x={-44} y={48} width={88} height={26} rx={10} fill={O.steel} />
          </g>
          {flash > 0 && <circle cy={360} r={70} fill={O.gas} opacity={flash * 0.7} />}
          <text x={0} y={-22} textAnchor="middle" fill={O.enzyme} fontSize={36}
            fontFamily={FONT_HEAD} fontWeight={700}>催泪因子合成酶</text>
        </g>
        {/* 进料箭头 */}
        <path d={`M ${MX - 330} ${beltY - 60} L ${MX - 200} ${beltY - 60}`} stroke={C.muted} strokeWidth={7} />
        <path d={`M ${MX - 200} ${beltY - 60} l -24 -14 M ${MX - 200} ${beltY - 60} l -24 14`}
          stroke={C.muted} strokeWidth={7} fill="none" strokeLinecap="round" />
        <text x={MX - 265} y={beltY - 92} textAnchor="middle" fill={C.muted} fontSize={30}
          fontFamily={FONT_HEAD}>中间产物</text>

        {/* 成品分子：飘起挥发 */}
        {Array.from({length: Math.min(made, 7)}).map((_, i) => {
          const age = f - (stampStart + i * cycle) - 16;
          if (age <= 0) return null;
          const yy = beltY - 120 - age * 3.1;
          const xx = MX + Math.sin(age / 10 + i) * 34;
          return <circle key={i} cx={xx} cy={yy} r={13} fill={O.gas}
            opacity={clamp01(age / 12) * clamp01((yy - 90) / 120)} />;
        })}

        {/* 分子示意图（右上） */}
        <g opacity={clamp01((f - beat(1) - 24) / 20)}>
          <rect x={1280} y={280} width={520} height={300} rx={30} fill={O.panel} stroke={O.fleshLine} strokeWidth={4} />
          <LFMolecule x={1540} y={420} s={1} label />
        </g>
      </svg>
      <Chip text="专属工程师" x={180} y={170} at={beat(0) + 12} accent={O.enzyme} />
      <Chip text="沸点低 · 几秒就挥发" x={180} y={262} at={beat(1) + 40} accent={O.gas} />
      <Stamp text="2002 年才摸清" x={1330} y={170} at={beat(2) + 8} color={O.skinDeep} size={52} rot={5} />
    </FadeIn>
  );
};

/* ============ S3 中弹：气体溶进泪液，三叉神经报警，泪腺放水 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);

  const EX = 700, EY = 540;
  const alarm = beat(1) + 8;                       // 警报起点
  const alarmK = clamp01((f - alarm) / 40);        // 信号抵达泪腺
  const pour = clamp01((f - alarm - 42) / 30);     // 放水
  const cryK = clamp01(0.25 + pour * 0.75 + clamp01((f - beat(2) - 20) / 40) * 0.4);
  const GX = 1500, GY = 310;                       // 泪腺位置

  return (
    <FadeIn>
      <Backdrop top={O.bgDeep2} bottom={O.bgDeep2} />
      <SceneTag index={3} name={sec.name} accent={O.tear} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 气体飘到角膜 */}
        <GasTrail p0={[120, 330]} p1={[330, 240]} p2={[430, 420]} p3={[512, 470]}
          delay={20} dur={90} n={8} on={1 - pour * 0.4} />
        {/* 命中点酸刺激火花 */}
        {f > 55 && Array.from({length: 4}).map((_, i) => {
          const t = ((f * 3 + i * 21) % 60) / 60;
          const sx = 520 + i * 46;
          const sy = 428 + Math.sin(i * 2) * 20;
          return (
            <g key={i} transform={`translate(${sx},${sy}) scale(${1 - t * 0.5})`} opacity={1 - t}>
              <path d="M 0 -16 L 5 -4 L 16 -3 L 6 4 L 9 15 L 0 8 L -9 15 L -6 4 L -16 -3 L -5 -4 Z" fill={C.coral} />
            </g>
          );
        })}

        {/* 大眼睛（放大版） */}
        <g transform={`translate(${EX},${EY}) scale(2.1)`}>
          <path d="M -132 0 Q 0 -88 132 0 Q 0 88 -132 0 Z" fill={O.eyeWhite} />
          {/* 泪液层高光（角膜外那层膜） */}
          <path d="M -124 -8 Q 0 -80 124 -8" fill="none" stroke={O.tear} strokeWidth={9} opacity={0.75} />
          <circle r={48} fill={O.iris} />
          <circle r={24} fill="#241738" />
          <circle cx={15} cy={-15} r={9} fill="#fff" />
          {/* 三叉神经末梢：角膜上小触角 */}
          {[-0.9, -0.45, 0, 0.45, 0.9].map((k, i) => {
            const nx = k * 110;
            const ny = -Math.sqrt(Math.max(0, 1 - (nx / 132) ** 2)) * 78;
            const hot = f > alarm - 8 ? 0.9 : 0.35;
            return (
              <g key={i} transform={`translate(${nx},${ny - 8})`}>
                <line x1={0} y1={0} x2={0} y2={-14} stroke={C.coral} strokeWidth={3.5} opacity={hot} />
                <circle cy={-18} r={4.5} fill={C.coral} opacity={hot} />
              </g>
            );
          })}
          {/* 流泪：泪腺放水后大滴掉 */}
          {pour > 0 && Array.from({length: 6}).map((_, i) => {
            const sp = 2.8 + rand(i, 61) * 2.2;
            const t = ((f * sp + i * 39) % 130) / 130;
            const dx = [-80, -35, 12, 60, 95, -5][i];
            return (
              <TearDrop key={i} x={dx} y={58 + t * 112} s={1.1 + rand(i, 62) * 0.8}
                o={pour * (t < 0.8 ? 1 : (1 - t) / 0.2)} />
            );
          })}
        </g>
        <line x1={330} y1={632} x2={462} y2={566} stroke={C.muted} strokeWidth={4} />
        <text x={290} y={668} textAnchor="middle" fill={C.muted} fontSize={30} fontFamily={FONT_HEAD}>
          角膜 · 最敏感的神经阵列
        </text>

        {/* 警报闪电：沿虚线冲到泪腺 */}
        <path d={`M ${EX + 250} ${EY - 180} C 1050 160 1250 180 ${GX - 130} ${GY + 30}`}
          fill="none" stroke={C.coral} strokeWidth={4} strokeDasharray="8 14" opacity={0.5} />
        {alarmK > 0 && alarmK < 1 && (() => {
          const [bx, by] = cubic([EX + 250, EY - 180], [1050, 160], [1250, 180], [GX - 130, GY + 30], alarmK);
          return (
            <g transform={`translate(${bx},${by})`}>
              <path d="M 0 -26 L 10 -6 L 26 -4 L 12 8 L 16 26 L 0 14 L -16 26 L -12 8 L -26 -4 L -10 -6 Z"
                fill={C.coral} />
            </g>
          );
        })()}

        {/* 泪腺：小水库 */}
        <g transform={`translate(${GX},${GY})`}>
          <path d="M -95 -20 C -95 -80 -40 -105 10 -95 C 60 -85 95 -55 95 -10 C 95 40 55 80 0 80 C -60 80 -95 40 -95 -20 Z"
            fill={O.tearDeep} stroke={O.tear} strokeWidth={6} />
          <path d="M -60 -30 C -50 -58 -20 -70 5 -62" stroke="#bfe9ff" strokeWidth={9} fill="none"
            strokeLinecap="round" opacity={0.8} />
          {/* 放水脉动 */}
          {pour > 0 && <circle r={105 + Math.sin(f / 6) * 7} fill="none" stroke={O.tear}
            strokeWidth={5} opacity={0.4 * pour} />}
          <text x={0} y={150} textAnchor="middle" fill={O.tear} fontSize={34}
            fontFamily={FONT_HEAD} fontWeight={700}>泪腺</text>
          {/* 输泪管 */}
          <path d={`M -80 30 C -160 90 -320 160 -420 ${EY - GY + 90}`} fill="none"
            stroke={O.tear} strokeWidth={6} strokeDasharray="14 10" opacity={0.8} />
          {/* 放水的水滴沿管飞 */}
          {pour > 0 && Array.from({length: 5}).map((_, i) => {
            const t = ((f * 4 + i * 26) % 90) / 90;
            const [qx, qy] = cubic([-80, 30], [-160, 90], [-320, 160], [-420, EY - GY + 90], t);
            return <circle key={i} cx={qx} cy={qy} r={9} fill={O.tear} opacity={Math.sin(Math.PI * t)} />;
          })}
        </g>
      </svg>
      <Stamp text="警报！" x={1120} y={200} at={alarm + 4} color={C.coral} size={54} rot={6} />
      <Chip text="溶进泪液层" x={170} y={170} at={beat(0) + 20} accent={O.tear} />
      <Stamp text="消防栓在干活" x={1380} y={640} at={beat(2) + 10} color={O.tearDeep} size={52} rot={5} />
      <Chip text="低级神经回路 · 意志力管不着" x={1020} y={800} at={beat(2) + 90} accent={C.coral} />
    </FadeIn>
  );
};

/* ============ S4 拆招：冷藏 / 快刀 / 眼镜 / 根部最后切 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);

  const popAt = [beat(1) + 14, beat(1) + 100, beat(1) + 185, beat(1) + 268];
  const dim = clamp01((f - beat(2) - 10) / 35);        // 结尾段压暗
  const bugX = interpolate(f, [beat(2) + 30, beat(2) + 160], [200, 1700], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const panels = [300, 755, 1210, 1665];

  return (
    <FadeIn>
      <Backdrop top={O.bgAlt} bottom={O.bgAlt} />
      <SceneTag index={4} name={sec.name} accent={O.leaf} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <g opacity={1 - dim * 0.66}>
          {/* P1 冷藏 */}
          {(() => {
            const s = spring({frame: f - popAt[0], fps, config: {damping: 11, stiffness: 170}});
            return (
              <g transform={`translate(${panels[0]},520) scale(${Math.max(0.001, s)})`}>
                <rect x={-200} y={-270} width={400} height={480} rx={30} fill={O.panel} stroke={O.tearDeep} strokeWidth={5} />
                {/* 冰箱 */}
                <rect x={-90} y={-200} width={180} height={250} rx={18} fill="#cfe0f2" />
                <line x1={-90} y1={-120} x2={90} y2={-120} stroke={O.steelDark} strokeWidth={6} />
                <rect x={-76} y={-108} width={12} height={40} rx={6} fill={O.steelDark} />
                <OnionBulb x={6} y={-30} s={0.62} />
                {/* 雪花 */}
                {[-150, 150].map((sx, i) => (
                  <g key={i} transform={`translate(${sx},-140)`} stroke={O.tear} strokeWidth={5} strokeLinecap="round">
                    {[0, 60, 120].map((a) => (
                      <line key={a} x1={-Math.cos(a * Math.PI / 180) * 26} y1={-Math.sin(a * Math.PI / 180) * 26}
                        x2={Math.cos(a * Math.PI / 180) * 26} y2={Math.sin(a * Math.PI / 180) * 26} />
                    ))}
                  </g>
                ))}
                <text x={0} y={140} textAnchor="middle" fill={C.text} fontSize={40} fontFamily={FONT_HEAD} fontWeight={800}>冷藏半小时</text>
                <text x={0} y={188} textAnchor="middle" fill={C.muted} fontSize={28} fontFamily={FONT_HEAD}>酶怠工 · 产量腰斩</text>
              </g>
            );
          })()}
          {/* P2 快刀 vs 钝刀 */}
          {(() => {
            const s = spring({frame: f - popAt[1], fps, config: {damping: 11, stiffness: 170}});
            return (
              <g transform={`translate(${panels[1]},520) scale(${Math.max(0.001, s)})`}>
                <rect x={-200} y={-270} width={400} height={480} rx={30} fill={O.panel} stroke={O.leaf} strokeWidth={5} />
                {/* 左快右钝 */}
                <line x1={0} y1={-200} x2={0} y2={60} stroke={C.muted} strokeWidth={4} strokeDasharray="8 8" />
                <Knife x={-95} y={-60} rot={12} s={0.72} />
                <Knife x={95} y={-60} rot={-12} s={0.72} dull />
                {/* 快：完整细胞 / 钝：碎渣 */}
                {[-150, -110, -70].map((cx, i) => <circle key={i} cx={cx} cy={45} r={17} fill={O.amino} />)}
                {[70, 105, 140].map((cx, i) => (
                  <g key={i} transform={`translate(${cx + 8},48)`}>
                    <path d="M -12 -8 L 12 8 M -12 8 L 12 -8" stroke={C.coral} strokeWidth={6} strokeLinecap="round" />
                  </g>
                ))}
                <text x={-100} y={100} textAnchor="middle" fill={O.leaf} fontSize={30} fontFamily={FONT_HEAD} fontWeight={700}>快刀</text>
                <text x={100} y={100} textAnchor="middle" fill={C.coral} fontSize={30} fontFamily={FONT_HEAD} fontWeight={700}>钝刀</text>
                <text x={0} y={140} textAnchor="middle" fill={C.text} fontSize={40} fontFamily={FONT_HEAD} fontWeight={800}>刀磨快一点</text>
                <text x={0} y={188} textAnchor="middle" fill={C.muted} fontSize={28} fontFamily={FONT_HEAD}>压碎的细胞少</text>
              </g>
            );
          })()}
          {/* P3 眼镜 */}
          {(() => {
            const s = spring({frame: f - popAt[2], fps, config: {damping: 11, stiffness: 170}});
            return (
              <g transform={`translate(${panels[2]},520) scale(${Math.max(0.001, s)})`}>
                <rect x={-200} y={-270} width={400} height={480} rx={30} fill={O.panel} stroke={O.skin} strokeWidth={5} />
                {/* 护目镜 */}
                <line x1={-160} y1={-80} x2={-118} y2={-60} stroke={O.tear} strokeWidth={14} />
                <line x1={160} y1={-80} x2={118} y2={-60} stroke={O.tear} strokeWidth={14} />
                <circle cx={-62} cy={-60} r={58} fill="rgba(92,200,255,0.18)" stroke={O.tear} strokeWidth={10} />
                <circle cx={62} cy={-60} r={58} fill="rgba(92,200,255,0.18)" stroke={O.tear} strokeWidth={10} />
                <path d="M -4 -60 Q 0 -72 4 -60" stroke={O.tear} strokeWidth={10} fill="none" />
                {/* 镜片后的眼睛 */}
                <path d="M -86 -60 Q -62 -82 -38 -60 Q -62 -44 -86 -60 Z" fill={O.eyeWhite} />
                <path d="M 38 -60 Q 62 -82 86 -60 Q 62 -44 38 -60 Z" fill={O.eyeWhite} />
                {/* 气体被挡在外面 */}
                {[0, 1, 2].map((i) => (
                  <circle key={i} cx={-60 + i * 60} cy={-160 + Math.sin(f / 10 + i * 2) * 8} r={11} fill={O.gas} opacity={0.9} />
                ))}
                <path d="M -90 -128 L -70 -112 M -40 -128 L -24 -112 M 30 -128 L 50 -112" stroke={O.gas}
                  strokeWidth={5} strokeLinecap="round" opacity={0.7} />
                <text x={0} y={140} textAnchor="middle" fill={C.text} fontSize={40} fontFamily={FONT_HEAD} fontWeight={800}>戴副眼镜</text>
                <text x={0} y={188} textAnchor="middle" fill={C.muted} fontSize={28} fontFamily={FONT_HEAD}>物理隔离最省事</text>
              </g>
            );
          })()}
          {/* P4 根部最后切 */}
          {(() => {
            const s = spring({frame: f - popAt[3], fps, config: {damping: 11, stiffness: 170}});
            return (
              <g transform={`translate(${panels[3]},520) scale(${Math.max(0.001, s)})`}>
                <rect x={-200} y={-270} width={400} height={480} rx={30} fill={O.panel} stroke={O.amino} strokeWidth={5} />
                <OnionBulb x={-30} y={-80} s={1.15} hiRoot={1} />
                {/* 先切上半的刀 */}
                <g transform={`translate(95,-160)`}>
                  <Knife x={0} y={0} rot={90} s={0.55} />
                  <text x={-30} y={-58} fill={O.leaf} fontSize={28} fontFamily={FONT_HEAD} fontWeight={700}>先切</text>
                </g>
                <text x={0} y={140} textAnchor="middle" fill={C.text} fontSize={40} fontFamily={FONT_HEAD} fontWeight={800}>根部留到最后</text>
                <text x={0} y={188} textAnchor="middle" fill={C.muted} fontSize={28} fontFamily={FONT_HEAD}>那里弹药最浓</text>
              </g>
            );
          })()}
        </g>

        {/* 结尾段：虫子才是目标 */}
        {dim > 0 && (
          <g opacity={dim}>
            <rect x={560} y={300} width={800} height={330} rx={34} fill={O.panel} stroke={C.coral} strokeWidth={6} />
            <Bug x={Math.min(bugX, 760)} y={470} scale={1.5} walk={bugX < 760 ? 1 : 0} />
            <OnionBulb x={1150} y={440} s={0.9} />
            {/* 洋葱举盾 */}
            <g transform="translate(1150,440)">
              <path d="M -70 -10 Q 0 -50 70 -10 L 70 40 Q 0 90 -70 40 Z" fill="none"
                stroke={O.gas} strokeWidth={7} transform="translate(0,-10) scale(1.15)" />
            </g>
            <text x={960} y={580} textAnchor="middle" fill={C.text} fontSize={44}
              fontFamily={FONT_HEAD} fontWeight={800}>这套防御本来是给虫子准备的</text>
          </g>
        )}
      </svg>
      <Chip text="少哭四件套" x={170} y={150} at={popAt[0] - 6} accent={O.leaf} />
      <Stamp text="你只是路过" x={1400} y={160} at={beat(2) + 60} color={C.coral} size={54} rot={5} />
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
      <Backdrop top={O.bg} bottom={O.bg} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <ellipse cx={960} cy={990} rx={900} ry={120} fill={O.board} opacity={0.5} />
        <g transform={`translate(960,930) scale(${Math.max(0.001, pop)})`}>
          <OnionBulb x={0} y={0} s={0.95} />
          <TearDrop x={92} y={-40} s={1.5} o={0.9} />
        </g>
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -220}}>
        <div style={{fontSize: 74, fontWeight: 800, color: C.text}}>你的眼泪，是它的防身术</div>
        <div style={{fontSize: 38, color: C.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E206 = ({data = tl}: {data?: any}) => {
  const {fps} = useVideoConfig();
  return (
    <AbsoluteFill style={{background: O.bg}}>
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
