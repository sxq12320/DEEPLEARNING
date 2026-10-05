import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01, rand} from '../lib/style';
import {Backdrop, Ripple, Stamp, SceneTag, Chip, Subtitles, ProgressBar, Sun, FadeIn} from '../lib/atoms';
import tl from '../timeline_208.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 土棕+芽绿+警示红色板 ============ */
const P = {
  bg: '#1a1108',
  bg2: '#130c05',
  board: '#4a3520',
  boardD: '#332413',
  skin: '#e0a75e',
  skinD: '#a06a28',
  skinHi: '#f2c384',
  flesh: '#f3ddab',
  fleshD: '#d8b878',
  sprout: '#a4d65c',
  sproutD: '#6fa03c',
  green: '#79a74b',
  toxin: '#ff5d4d',
  toxinD: '#c23a30',
  soil: '#2e1d0d',
  soilHi: '#402a14',
  good: '#8fd16a',
  pot: '#3a3f52',
  potRim: '#565e78',
};

/* ---------- 土豆轮廓（局部坐标，原点在中心） ---------- */
const POT_D = 'M -180 -8 C -192 -98 -98 -152 0 -142 C 98 -154 188 -92 182 6 C 190 82 112 148 4 142 C -96 152 -176 92 -180 -8 Z';
const EYES = [
  {x: -78, y: -52, r: -15},
  {x: 66, y: -64, r: 8},
  {x: 118, y: 22, r: 20},
  {x: -34, y: 58, r: -10},
];
/* 毒素小点：预生成在土豆轮廓内 */
const TOXIN_DOTS = Array.from({length: 30}).map((_, i) => ({
  dx: (rand(i, 3) - 0.5) * 330,
  dy: (rand(i, 4) - 0.5) * 240,
  th: rand(i, 5),
  r: 7 + rand(i, 6) * 8,
})).filter((d) => (d.dx * d.dx) / (165 * 165) + (d.dy * d.dy) / (118 * 118) < 0.82);

/* ---------- 一根芽：弯茎 + 三片小叶，k=0..1 弹出 ---------- */
const Sprout = ({x, y, k, lean = 0, s = 1}: {x: number; y: number; k: number; lean?: number; s?: number}) => {
  const kk = Math.max(0, Math.min(k, 1.15));
  if (kk <= 0.01) return null;
  return (
    <g transform={`translate(${x},${y}) rotate(${lean}) scale(${s * kk},${s * Math.min(kk, 1)})`}>
      <path d="M 0 6 C -10 -26 8 -50 2 -76" fill="none" stroke={P.sproutD} strokeWidth={10} strokeLinecap="round" />
      <g transform="translate(2,-76)">
        <ellipse cx={-15} cy={-4} rx={21} ry={9} fill={P.sprout} transform="rotate(-34)" />
        <ellipse cx={16} cy={-5} rx={21} ry={9} fill={P.sproutD} transform="rotate(32)" />
        <ellipse cx={1} cy={-15} rx={9} ry={15} fill={P.sprout} />
      </g>
    </g>
  );
};

/* ---------- 参数化土豆 ----------
   green: 0..1 见光发绿晕染   wrinkle: 皱缩   sprout: 芽弹出度
   sproutS: 芽长度倍率        toxin: 内部毒点弥漫度           */
const Potato = ({x, y, s = 1, rot = 0, green = 0, wrinkle = 0, sprout = 0, sproutS = 1, toxin = 0, cid}: {
  x: number; y: number; s?: number; rot?: number; green?: number; wrinkle?: number;
  sprout?: number; sproutS?: number; toxin?: number; cid: string;
}) => {
  const g = clamp01(green);
  const tx = clamp01(toxin);
  const wr = clamp01(wrinkle);
  return (
    <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s},${s * (1 - wr * 0.12)})`}>
      <defs><clipPath id={`pc${cid}`}><path d={POT_D} /></clipPath></defs>
      <path d={POT_D} fill={P.skin} />
      <g clipPath={`url(#pc${cid})`}>
        <ellipse cx={0} cy={115} rx={215} ry={85} fill={P.skinD} opacity={0.4} />
        <ellipse cx={-60} cy={-85} rx={120} ry={48} fill={P.skinHi} opacity={0.5} />
        {/* 表皮麻麻点 */}
        {Array.from({length: 10}).map((_, i) => (
          <circle key={i} cx={(rand(i, 8) - 0.5) * 320} cy={(rand(i, 9) - 0.5) * 220}
            r={3 + rand(i, 10) * 4} fill={P.skinD} opacity={0.5} />
        ))}
        {/* 变绿：受光一侧晕开 */}
        {g > 0 && <circle cx={-60} cy={-75} r={g * 340} fill={P.green} opacity={0.9} />}
        {/* 皱皮纹 */}
        {wr > 0 && [-70, -12, 52].map((yy, i) => (
          <path key={i} d={`M -150 ${yy} Q -40 ${yy + 20 * wr} 60 ${yy + 4} T 170 ${yy + 14}`}
            fill="none" stroke={P.skinD} strokeWidth={7} opacity={0.6 * wr} strokeLinecap="round" />
        ))}
        {/* 毒素点：从芽眼附近先起，逐渐弥漫整个块茎 */}
        {tx > 0 && TOXIN_DOTS.map((d, i) => {
          const a = clamp01(tx * 1.7 - d.th * 1.0);
          return <circle key={i} cx={d.dx} cy={d.dy} r={Math.max(0, d.r * a)} fill={P.toxin} opacity={0.8 * a} />;
        })}
      </g>
      <path d={POT_D} fill="none" stroke={P.skinD} strokeWidth={7} opacity={0.85} />
      {/* 芽眼 */}
      {EYES.map((e, i) => (
        <ellipse key={i} cx={e.x} cy={e.y} rx={15} ry={8} fill={P.skinD}
          transform={`rotate(${e.r} ${e.x} ${e.y})`} opacity={0.9} />
      ))}
      {/* 芽 */}
      <Sprout x={EYES[1].x} y={EYES[1].y} k={sprout} lean={-8} s={sproutS} />
      <Sprout x={EYES[2].x} y={EYES[2].y} k={Math.max(0, sprout - 0.12)} lean={18} s={sproutS * 0.85} />
      {sprout > 0.55 && <Sprout x={EYES[0].x} y={EYES[0].y} k={sprout - 0.5} lean={-28} s={sproutS * 0.6} />}
    </g>
  );
};

/* ---------- 菜刀：柄左刃右 ---------- */
const Knife = ({x, y, rot = 150, s = 1}: {x: number; y: number; rot?: number; s?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`}>
    <rect x={-150} y={-15} width={112} height={32} rx={15} fill="#6b4a2c" />
    <circle cx={-128} cy={1} r={6} fill="#4a3016" />
    <path d="M -38 -16 L 128 -9 L 152 6 L -38 17 Z" fill="#dde6ee" />
    <path d="M 96 3 L 152 6 L 128 -9 L 88 -6 Z" fill="#aeb9c8" opacity={0.8} />
  </g>
);

/* ---------- 判定小徽章：✓ / ✗ ---------- */
const Badge = ({x, y, ok, k}: {x: number; y: number; ok: boolean; k: number}) => {
  const kk = Math.max(0, Math.min(k, 1.15));
  if (kk <= 0.01) return null;
  return (
    <g transform={`translate(${x},${y}) scale(${kk})`}>
      <circle r={48} fill={ok ? P.good : P.toxin} />
      {ok ? (
        <path d="M -22 2 L -6 20 L 26 -18" fill="none" stroke="#fff" strokeWidth={11} strokeLinecap="round" strokeLinejoin="round" />
      ) : (
        <path d="M -18 -18 L 18 18 M 18 -18 L -18 18" stroke="#fff" strokeWidth={11} strokeLinecap="round" />
      )}
    </g>
  );
};

/* ============ 标题卡 ============ */
const TitleCard = ({data}: {data: any}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 13, mass: 0.7}});
  const pop = spring({frame: f - 8, fps, config: {damping: 9, stiffness: 150, mass: 0.8}});
  const o = interpolate(f, [0, 8], [0, 1]);
  return (
    <AbsoluteFill>
      <Backdrop top={P.bg} bottom={P.bg} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <ellipse cx={960} cy={555} rx={230} ry={26} fill={P.boardD} />
        <g transform={`translate(960,395) scale(${0.4 + 0.6 * Math.min(pop, 1.1)})`}>
          <Potato x={0} y={0} s={1.05} rot={-6} sprout={clamp01((pop - 0.4) * 2)} cid="t0" />
        </g>
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 170}}>
          <div style={{color: P.sprout, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 26}}>
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

/* ============ S0 开场：抠芽下锅？芽一启动毒已扩散 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);

  const pop = spring({frame: f - 8, fps, config: {damping: 10, stiffness: 140}});
  const sproutK = clamp01(spring({frame: f - 48, fps, config: {damping: 9, stiffness: 150}}) * 1.2);
  const sprout2 = clamp01(spring({frame: f - 74, fps, config: {damping: 9, stiffness: 150}}) * 1.2);

  // 菜刀挖芽眼
  const kx = interpolate(f, [95, 122], [1250, 895], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const ky = interpolate(f, [95, 122], [280, 555], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const dip = interpolate(f, [122, 132, 142], [0, 42, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const kOut = interpolate(f, [165, 205], [0, 330], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const wedgeT = clamp01((f - 138) / 26);            // 挖掉的芽眼小块飞出去
  const wedgeO = f > 200 ? 0 : 1;

  const toxin = interpolate(f, [beat(1), beat(1) + 85], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={0} name={sec.name} accent={P.skin} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 砧板 */}
        <rect x={200} y={790} width={1520} height={90} rx={24} fill={P.board} />
        <rect x={200} y={866} width={1520} height={16} rx={8} fill={P.boardD} />
        {/* 土豆本体：挖过的芽眼留个坑 */}
        <g transform={`translate(660,690) scale(${0.4 + 0.6 * Math.min(pop, 1.1)})`}>
          <Potato x={0} y={0} s={1.35} rot={-5} sprout={Math.max(sproutK, sprout2 * 0.8)} toxin={toxin} cid="s0" />
          {/* 挖掉的锥坑 */}
          {f > 138 && <ellipse cx={89} cy={-86} rx={26} ry={16} fill={P.boardD} opacity={0.95} transform="rotate(-18 89 -86)" />}
        </g>
        {/* 飞出去的芽眼小块 */}
        {wedgeT > 0 && wedgeT < 1 && (
          <g transform={`translate(${748 + wedgeT * 210},${600 - Math.sin(wedgeT * Math.PI) * 130 + wedgeT * 120}) rotate(${wedgeT * 300})`} opacity={wedgeO}>
            <path d="M 0 -16 L 26 14 L -22 12 Z" fill={P.flesh} stroke={P.skinD} strokeWidth={4} />
          </g>
        )}
        <Ripple x={748} y={604} startFrame={122} color={P.skin} />
        <Knife x={kx + kOut} y={ky + dip + kOut * 0.55} rot={152} s={1.15} />
      </svg>
      <Stamp text="抠掉照样炒？" x={1240} y={250} at={132} color={P.skinD} size={54} rot={-6} />
      <Stamp text="毒已扩散" x={1330} y={390} at={beat(1) + 34} color={P.toxin} size={62} rot={5} />
      <Chip text="挖芽 = 只拔警报器" x={1180} y={150} at={beat(1) + 75} accent={P.toxin} />
    </FadeIn>
  );
};

/* ============ S1 毒从哪来：茄科亲戚 + 剖面热力扩散 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);

  const heat = interpolate(f, [beat(1), beat(1) + 55], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const spread = interpolate(f, [beat(1) + 40, beat(1) + 160], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const gauge = interpolate(f, [beat(1) + 25, beat(1) + 100], [0.08, 0.92], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});

  const CX = 980, CY = 680, RX = 315, RY = 200;
  const EX = 1090, EY = 580; // 剖面上毒源芽眼
  const fam = [
    {name: '土豆', x: 430},
    {name: '番茄', x: 700},
    {name: '茄子', x: 970},
    {name: '颠茄', x: 1240},
  ];
  return (
    <FadeIn>
      <Backdrop top={P.bg2} bottom={P.bg2} />
      <SceneTag index={1} name={sec.name} accent={P.toxin} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <defs><clipPath id="fleshClip"><ellipse cx={CX} cy={CY} rx={RX - 18} ry={RY - 18} /></clipPath></defs>

        {/* 茄科一家 */}
        {fam.map((m, i) => {
          const p = spring({frame: f - 14 - i * 12, fps, config: {damping: 10, stiffness: 160}});
          const ss = Math.max(0, Math.min(p, 1.1));
          return (
            <g key={i} transform={`translate(${m.x},280) scale(${ss})`} opacity={ss > 0 ? 1 : 0}>
              {i === 0 && <Potato x={0} y={0} s={0.42} cid={`fam${i}`} />}
              {i === 1 && (
                <g>
                  <circle r={66} fill="#e8564a" />
                  <ellipse cx={-20} cy={-28} rx={26} ry={14} fill="#ff8a75" opacity={0.6} />
                  <path d="M 0 -66 l -24 -20 M 0 -66 l 24 -20 M 0 -66 l 0 -26" stroke={P.sproutD} strokeWidth={8} strokeLinecap="round" />
                </g>
              )}
              {i === 2 && (
                <g>
                  <ellipse rx={46} ry={72} fill="#7a4fd0" />
                  <ellipse cx={-14} cy={-20} rx={16} ry={34} fill="#9b7ede" opacity={0.55} />
                  <path d="M -18 -68 Q 0 -88 18 -68 Q 0 -58 -18 -68 Z" fill={P.sproutD} />
                  <line x1={0} y1={-82} x2={0} y2={-98} stroke={P.sproutD} strokeWidth={8} strokeLinecap="round" />
                </g>
              )}
              {i === 3 && (
                <g>
                  <circle cx={-28} cy={10} r={26} fill="#4a2a72" />
                  <circle cx={26} cy={16} r={22} fill="#3a1f5c" />
                  <ellipse cx={6} cy={-30} rx={34} ry={14} fill={P.sproutD} transform="rotate(-20)" />
                  <ellipse cx={-30} cy={-20} rx={28} ry={12} fill={P.sprout} transform="rotate(30)" />
                </g>
              )}
              <text x={0} y={125} textAnchor="middle" fill={C.muted} fontSize={32} fontFamily={FONT_HEAD} letterSpacing={4}>{m.name}</text>
            </g>
          );
        })}
        <text x={170} y={195} fill={C.muted} fontSize={34} fontFamily={FONT_HEAD} letterSpacing={6}>茄科一家子</text>
        <text x={170} y={245} fill={P.toxin} fontSize={30} fontFamily={FONT_HEAD} letterSpacing={4}>都玩毒</text>

        {/* 剖面大土豆 */}
        <ellipse cx={CX} cy={CY} rx={RX + 16} ry={RY + 16} fill={P.skinD} />
        <ellipse cx={CX} cy={CY} rx={RX} ry={RY} fill={P.flesh} />
        <ellipse cx={CX} cy={CY + RY * 0.42} rx={RX * 0.7} ry={RY * 0.4} fill={P.fleshD} opacity={0.4} />
        <g clipPath="url(#fleshClip)">
          {/* 芽眼热力环 */}
          {[55, 105, 160, 220].map((r, i) => (
            <circle key={i} cx={EX} cy={EY} r={r * (0.4 + heat * 0.8)} fill={P.toxin}
              opacity={heat * (0.34 - i * 0.06)} />
          ))}
          {/* 弥漫到整块 */}
          {spread > 0 && TOXIN_DOTS.map((d, i) => {
            const a = clamp01(spread * 1.7 - d.th * 1.0);
            return <circle key={i} cx={CX + d.dx * 1.55} cy={CY + d.dy * 1.35}
              r={Math.max(0, d.r * a)} fill={P.toxin} opacity={0.6 * a} />;
          })}
        </g>
        <circle cx={EX} cy={EY} r={16} fill={P.toxinD} opacity={heat * 0.9} />
        <ellipse cx={EX} cy={EY} rx={22} ry={11} fill={P.skinD} transform={`rotate(15 ${EX} ${EY})`} />
        <Sprout x={CX + 150} y={CY - RY - 4} k={clamp01((f - 40) / 40)} s={0.8} lean={6} />
        <Ripple x={EX} y={EY} startFrame={beat(1)} color={P.toxin} />

        {/* 浓度条：冲过安全线 */}
        <rect x={200} y={838} width={430} height={30} rx={15} fill="#00000055" stroke={C.muted} strokeWidth={3} />
        <rect x={200} y={838} width={430 * gauge} height={30} rx={15} fill={gauge > 0.3 ? P.toxin : P.good} />
        <line x1={200 + 430 * 0.3} y1={824} x2={200 + 430 * 0.3} y2={884} stroke="#fff" strokeWidth={5} />
        <text x={200} y={812} fill={C.muted} fontSize={30} fontFamily={FONT_HEAD} letterSpacing={4}>块茎毒素浓度</text>
        <text x={329} y={906} textAnchor="middle" fill="#fff" fontSize={26} fontFamily={FONT_HEAD}>安全线</text>
      </svg>
      <Stamp text="芽眼周 5–10 倍" x={1300} y={430} at={beat(1) + 50} color={P.toxin} size={52} rot={5} />
      <Chip text="龙葵素 solanine" x={1400} y={180} at={beat(0) + 40} accent={P.toxin} />
    </FadeIn>
  );
};

/* ============ S2 变绿报警：见光发绿 = 毒素同涨 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);

  const beam = interpolate(f, [10, 50], [0, 1], {extrapolateRight: 'clamp'});
  const greenK = interpolate(f, [beat(0) + 45, beat(1) + 90], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const tongue = spring({frame: f - beat(2) - 12, fps, config: {damping: 10, stiffness: 160}});

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={2} name={sec.name} accent={P.green} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 阳光斜照 */}
        <g opacity={beam}>
          <polygon points="330,190 560,880 1180,880" fill={C.amber} opacity={0.10} />
          <polygon points="330,190 720,880 1010,880" fill={C.amber} opacity={0.10} />
          <Sun x={330} y={190} r={72} />
        </g>
        {/* 台面上的土豆：受光侧变绿 + 毒点起 */}
        <ellipse cx={870} cy={872} rx={330} ry={34} fill={P.boardD} />
        <Potato x={860} y={700} s={1.5} rot={4} green={greenK} toxin={greenK * 0.9}
          sprout={0.3} sproutS={0.7} cid="s2" />
        {/* 绿色标注 */}
        {greenK > 0.5 && (
          <g opacity={clamp01((greenK - 0.5) * 2)}>
            <line x1={700} y1={560} x2={560} y2={430} stroke={P.green} strokeWidth={5} strokeDasharray="10 8" />
            <text x={330} y={415} fill={P.green} fontSize={38} fontFamily={FONT_HEAD} letterSpacing={4}>见光变绿</text>
          </g>
        )}
        {/* 舌头检测仪 */}
        {tongue > 0.02 && (
          <g transform={`translate(1500,660) scale(${Math.min(tongue, 1.15)})`}>
            <ellipse rx={120} ry={52} fill="#ff9db0" />
            <ellipse cy={-14} rx={108} ry={30} fill="#ffb9c6" />
            <line x1={0} y1={-38} x2={0} y2={40} stroke="#e57f95" strokeWidth={6} />
            {/* 麻：颤抖波浪线 */}
            <path d="M -160 -70 q 12 -14 24 0 q 12 14 24 0" fill="none" stroke={P.toxin} strokeWidth={6} strokeLinecap="round" />
            <path d="M 90 -76 q 12 -14 24 0 q 12 14 24 0" fill="none" stroke={P.toxin} strokeWidth={6} strokeLinecap="round" />
          </g>
        )}
      </svg>
      <Chip text="见光 = 启动防御" x={1330} y={170} at={beat(1) + 15} accent={P.green} />
      <Stamp text="苦！" x={1370} y={400} at={beat(2) + 16} color={P.toxin} size={80} rot={-8} />
      <Chip text="发苦发麻就吐掉" x={1330} y={510} at={beat(2) + 42} accent={P.toxin} />
    </FadeIn>
  );
};

/* ============ S3 能不能救：分档判断 + 煮不掉 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);

  const okBadge = spring({frame: f - 208, fps, config: {damping: 10, stiffness: 170}});
  const noBadge = spring({frame: f - 300, fps, config: {damping: 10, stiffness: 170}});
  const fly = clamp01((f - 318) / 34);   // 坏土豆飞进垃圾桶
  const binX = interpolate(fly, [0, 1], [1440, 1690]);
  const binY = interpolate(fly, [0, 0.55, 1], [565, 480, 735]);
  const binS = interpolate(fly, [0, 1], [1, 0.42]);

  const phase2 = clamp01((f - beat(1)) / 30);
  const potPop = spring({frame: f - beat(1) - 6, fps, config: {damping: 11, stiffness: 140}});

  return (
    <FadeIn>
      <Backdrop top={P.bg2} bottom={P.bg2} />
      <SceneTag index={3} name={sec.name} accent={C.amber} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 中缝虚线 */}
        <line x1={960} y1={170} x2={960} y2={850} stroke={C.muted} strokeWidth={4} strokeDasharray="14 14" opacity={0.4} />

        {/* —— 左：刚冒尖，能救 —— */}
        <g opacity={1 - phase2 * 0.6}>
          <Potato x={480} y={565} s={1.05} rot={-4} sprout={0.25} sproutS={0.7} cid="s3a" />
          {/* 挖掉的小坑 */}
          {f > 130 && <ellipse cx={550} cy={500} rx={22} ry={13} fill={P.boardD} transform="rotate(-14 550 500)" />}
          <Knife x={interpolate(f, [80, 110], [880, 640], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})
            + interpolate(f, [165, 210], [0, 180], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})}
            y={interpolate(f, [80, 110, 122, 134], [330, 460, 500, 462], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})
            + interpolate(f, [165, 210], [0, 120], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})}
            rot={155} s={0.95} />
          <Badge x={710} y={415} ok k={okBadge} />
        </g>
        {/* —— 右：芽成枝软皱发绿，整颗扔 —— */}
        <g opacity={1 - phase2 * 0.6}>
          <g transform={`translate(${binX - 1440},${binY - 565})`}>
            <Potato x={1440} y={565} s={1.08 * binS} rot={8 + fly * 50} sprout={0.95} sproutS={1.9}
              wrinkle={1} green={0.8} cid="s3b" />
          </g>
          {fly < 0.9 && <Badge x={1700} y={420} ok={false} k={noBadge * (1 - fly)} />}
          {/* 垃圾桶 */}
          <g transform="translate(1690,810)">
            <path d="M -74 -86 L 74 -86 L 54 62 L -54 62 Z" fill="#4a5168" />
            <rect x={-88} y={-104} width={176} height={22} rx={10} fill="#5d6680" />
            <rect x={-16} y={-126} width={32} height={24} rx={8} fill="#5d6680" />
            <line x1={-30} y1={-60} x2={-22} y2={40} stroke="#353b4e" strokeWidth={8} />
            <line x1={10} y1={-60} x2={16} y2={40} stroke="#353b4e" strokeWidth={8} />
          </g>
        </g>

        {/* —— 阶段二：煮不掉 —— */}
        {phase2 > 0 && <rect x={0} y={0} width={1920} height={1080} fill="#0a0703" opacity={phase2 * 0.45} />}
        {potPop > 0.02 && (
          <g transform={`translate(960,660) scale(${Math.min(potPop, 1.15)})`}>
            {/* 火苗 */}
            {[-160, -80, 0, 80, 160].map((dx, i) => {
              const h = 46 + Math.sin(f * 0.55 + i * 1.7) * 14;
              return (
                <path key={i} d={`M ${dx - 26} 178 Q ${dx} ${178 - h - 22} ${dx + 26} 178 Z`}
                  fill={i % 2 ? '#ff9a3d' : C.amber} opacity={0.9} />
              );
            })}
            {/* 锅体 */}
            <path d="M -250 -40 Q -250 175 0 175 Q 250 175 250 -40 Z" fill={P.pot} />
            <ellipse cx={0} cy={-40} rx={250} ry={46} fill={P.potRim} />
            <ellipse cx={0} cy={-40} rx={218} ry={34} fill="#2c5f7d" />
            {/* 水里翻滚的毒点：煮不烂 */}
            {TOXIN_DOTS.slice(0, 12).map((d, i) => (
              <circle key={i} cx={d.dx * 1.05} cy={-52 + Math.sin(f / 6 + i * 1.3) * 9 + (i % 3) * 7}
                r={d.r * 0.75} fill={P.toxin} opacity={0.85} />
            ))}
            {/* 蒸汽 */}
            {[0, 1, 2].map((i) => {
              const t = (f * 1.4 + i * 40) % 130;
              return (
                <path key={i} d={`M ${-90 + i * 80} -90 q -18 -30 0 -55 q 18 -25 0 -50`}
                  fill="none" stroke="#cfd6e4" strokeWidth={8} strokeLinecap="round"
                  opacity={0.5 * (1 - t / 130)} transform={`translate(0,${-t * 1.1})`} />
              );
            })}
          </g>
        )}
      </svg>
      <Chip text="刚冒尖·皮没绿·还硬实" x={150} y={225} at={30} accent={P.good} />
      <Chip text="芽成枝·软皱·发绿" x={1180} y={225} at={60} accent={P.toxin} />
      <Stamp text="整颗扔" x={1330} y={320} at={312} color={P.toxin} size={58} rot={6} />
      <Stamp text="蒸煮炖炒都拆不掉" x={600} y={240} at={beat(1) + 42} color={P.toxinD} size={54} rot={-5} />
      <Chip text="180℃油炸才降解一点" x={1240} y={150} at={beat(1) + 75} accent={C.amber} />
    </FadeIn>
  );
};

/* ============ S4 正确姿势：避光储存 + 种回土里 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);

  const intoBag = interpolate(f, [70, 118], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const therm = interpolate(f, [40, 110], [0.15, 0.6], {extrapolateRight: 'clamp'});
  const fridgeX = spring({frame: f - 150, fps, config: {damping: 10, stiffness: 170}});
  const onionX = spring({frame: f - 185, fps, config: {damping: 10, stiffness: 170}});

  const phase2 = clamp01((f - beat(1)) / 30);
  const soilP = spring({frame: f - beat(1) - 4, fps, config: {damping: 12, stiffness: 120}});
  const shoot = spring({frame: f - beat(1) - 50, fps, config: {damping: 10, stiffness: 130}});

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={4} name={sec.name} accent={P.sprout} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* —— 左：阴凉角落的储存角 —— */}
        <g opacity={1 - phase2 * 0.35}>
          {/* 置物架 */}
          <rect x={130} y={840} width={780} height={24} fill={P.board} />
          <rect x={130} y={864} width={24} height={200} fill={P.boardD} />
          <rect x={886} y={864} width={24} height={200} fill={P.boardD} />
          {/* 先画土豆，再画袋子罩住下半 */}
          <g transform="translate(430,560)">
            <Potato x={0} y={-60 + intoBag * 70} s={0.72} rot={-8} cid="s4a" />
          </g>
          {/* 黑色不透光袋 */}
          <g transform="translate(430,700)">
            <path d="M -160 -140 Q -170 40 -140 130 L 140 130 Q 170 40 160 -140 Q 80 -178 0 -178 Q -80 -178 -160 -140 Z"
              fill="#1d2338" stroke="#333c5c" strokeWidth={6} />
            <ellipse cx={0} cy={-160} rx={118} ry={24} fill="#0e1224" stroke="#333c5c" strokeWidth={5} />
            <text x={0} y={20} textAnchor="middle" fill="#5a6488" fontSize={34} fontFamily={FONT_HEAD} letterSpacing={6}>不透光袋</text>
          </g>
          {/* 温度计 ≈10℃ */}
          <g transform="translate(760,430)">
            <rect x={-17} y={-110} width={34} height={170} rx={17} fill="#e8ecf4" />
            <circle cy={72} r={32} fill="#e8ecf4" />
            <rect x={-7} y={-110 + 170 * (1 - therm)} width={14} height={170 * therm} fill={P.toxin} />
            <circle cy={72} r={20} fill={P.toxin} />
            <text x={52} y={-20} fill={C.text} fontSize={42} fontFamily={FONT_HEAD}>≈10℃</text>
          </g>
          {/* 别放冰箱 / 别混放 */}
          {fridgeX > 0.02 && (
            <g transform={`translate(1080,330) scale(${Math.min(fridgeX, 1.15)})`}>
              <rect x={-55} y={-85} width={110} height={170} rx={14} fill="#c9d4e4" />
              <line x1={-55} y1={-12} x2={55} y2={-12} stroke="#8fa0b8" strokeWidth={6} />
              <rect x={28} y={-66} width={12} height={34} rx={6} fill="#8fa0b8" />
              <Badge x={58} y={62} ok={false} k={1} />
            </g>
          )}
          {onionX > 0.02 && (
            <g transform={`translate(1340,390) scale(${Math.min(onionX, 1.15)})`}>
              <ellipse rx={62} ry={54} fill="#d9a05b" />
              <path d="M 0 -54 q -8 -30 -2 -44 M -14 -50 q -14 -22 -10 -36 M 14 -50 q 14 -22 10 -36"
                stroke={P.sproutD} strokeWidth={7} fill="none" strokeLinecap="round" />
              <path d="M -58 10 Q 0 40 58 10" stroke="#b57e3e" strokeWidth={6} fill="none" />
              <Badge x={58} y={62} ok={false} k={1} />
            </g>
          )}
        </g>

        {/* —— 右下：埋回土里长一窝 —— */}
        {phase2 > 0 && (
          <g transform={`translate(1480,800) scale(${Math.min(soilP, 1.1)})`} opacity={phase2}>
            <ellipse cx={0} cy={0} rx={430} ry={140} fill={P.soil} />
            <ellipse cx={0} cy={-16} rx={400} ry={100} fill={P.soilHi} opacity={0.55} />
            {/* 带芽切块半埋 */}
            <g transform="translate(-30,-58)">
              <Potato x={0} y={0} s={0.5} rot={6} cid="s4b" />
              <Sprout x={-10} y={-60} k={shoot} s={1.5} lean={-4} />
            </g>
            {/* 土前唇遮住切块下半 */}
            <path d="M -430 -10 Q -200 60 0 62 Q 200 60 430 -10 L 430 140 L -430 140 Z" fill={P.soil} />
            {/* 地下新土豆弹出来 */}
            {[0, 1, 2].map((i) => {
              const bp = spring({frame: f - beat(1) - 95 - i * 16, fps, config: {damping: 9, stiffness: 170}});
              const pos = [[-215, 15], [145, 35], [-35, 52]][i];
              return (
                <g key={i} transform={`translate(${pos[0]},${pos[1]}) scale(${Math.max(0, Math.min(bp, 1.15)) * 0.46})`}>
                  <path d={POT_D} fill={P.skin} stroke={P.skinD} strokeWidth={8} />
                </g>
              );
            })}
            {shoot > 0.6 && (
              <text x={210} y={-95} fill={P.flesh} fontSize={40} fontFamily={FONT_HEAD}
                letterSpacing={4} opacity={clamp01((shoot - 0.6) * 2)}>一窝新土豆</text>
            )}
          </g>
        )}
      </svg>
      <Chip text="避光" x={250} y={215} at={42} accent={P.sprout} />
      <Chip text="阴凉" x={440} y={215} at={58} accent={P.sprout} />
      <Chip text="通风" x={630} y={215} at={74} accent={P.sprout} />
      <Chip text="发芽的种回去最体面" x={1190} y={200} at={beat(1) + 70} accent={P.sprout} />
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
      <Backdrop top={P.bg} bottom={P.bg} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <ellipse cx={960} cy={1030} rx={880} ry={120} fill={P.soil} />
        <g transform={`translate(960,985) scale(${Math.max(0, Math.min(pop, 1.1))})`}>
          <Potato x={0} y={0} s={0.85} rot={-6} sprout={0.9} cid="end" />
        </g>
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -170}}>
        <div style={{fontSize: 72, fontWeight: 800, color: C.text}}>发芽的土豆，就别心疼这两块</div>
        <div style={{fontSize: 38, color: C.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E208 = ({data = tl}: {data?: any}) => {
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
