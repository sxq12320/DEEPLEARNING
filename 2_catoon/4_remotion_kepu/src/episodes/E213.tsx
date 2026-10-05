import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01, rand} from '../lib/style';
import {Backdrop, Ripple, Stamp, SceneTag, Chip, Subtitles, ProgressBar, Sun, Rain, FadeIn} from '../lib/atoms';
import tl from '../timeline_213.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：翠绿夹壳 × 内膛粉红 × 虫深色（纯色平涂） ============ */
const V = {
  bg: '#173b2b',        // 沼泽深绿
  bg2: '#0f3340',       // 数两下·深青
  bg3: '#21455c',       // 雨天蓝
  bg4: '#3c2a1e',       // 胃袋暖棕
  skin: '#54b567',      // 夹壳绿
  skinDark: '#2f8144',  // 夹壳暗绿
  face: '#ff9eb8',      // 内膛粉
  faceDeep: '#ef6f95',  // 内膛深粉
  hair: '#93264c',      // 触发毛绛红
  tooth: '#f2f8cd',     // 齿牙黄白
  bug: '#33243f',       // 虫深紫
  bugHi: '#5d4683',
  wing: '#d9ecf2',
  shell: '#9d93a8',     // 虫壳灰
  ca: '#ffc94d',        // 钙离子琥珀
  drop: '#6ecbff',      // 雨滴
  dryLeaf: '#b8925c',   // 枯叶
  leaf: '#6fc57a',
  leafDark: '#3d8a4f',
  cream: '#fff3d6',
  panel: '#0d291f',     // HUD面板
  dead: '#33302b',      // 枯死夹
  mud: '#2e2117',
};

/* ---------- 单片夹瓣几何：局部坐标从铰链(0,0)沿 +X 伸展，宽 W ---------- */
const lobeW = (L: number, W: number, t: number) =>
  Math.max(0.01, (W / 2) * Math.sin(Math.PI * Math.pow(t, 0.72)));

const lobePath = (L: number, W: number) => {
  const n = 18;
  const top: string[] = [];
  const bot: string[] = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n, x = t * L, w = lobeW(L, W, t);
    top.push(`${x.toFixed(1)},${(-w).toFixed(1)}`);
    bot.unshift(`${x.toFixed(1)},${w.toFixed(1)}`);
  }
  return `M ${top.join(' L ')} L ${bot.join(' L ')} Z`;
};

/* 内膛（上半透镜形） */
const facePath = (L: number, W: number) => {
  const n = 18;
  const top: string[] = [];
  const mid: string[] = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n, x = t * L, w = lobeW(L, W, t);
    top.push(`${x.toFixed(1)},${(-w + 3).toFixed(1)}`);
    mid.unshift(`${x.toFixed(1)},2`);
  }
  return `M ${top.join(' L ')} L ${mid.join(' L ')} Z`;
};

/* ---------- 捕蝇草：side=±1 两片瓣，open 0闭→1开，铰链在 origin ---------- */
const HAIR_TS = [0.42, 0.56, 0.7, 0.83];
const Trap = ({x, y, s = 1, open = 1, rot = 0, hairFlash = -1, hairTick = -1,
  sealed = 0, dead = false, ghost = 0, legs = 0, stem = true}: {
  x: number; y: number; s?: number; open?: number; rot?: number;
  hairFlash?: number; hairTick?: number; sealed?: number; dead?: boolean;
  ghost?: number; legs?: number; stem?: boolean;
}) => {
  const f = useCurrentFrame();
  const L = 300, W = 196;
  const a = 50 + (1 - open) * 36;                     // 瓣倾角：开≈50° 闭≈86°
  const skin = dead ? V.dead : V.skin;
  const skinD = dead ? '#211f1b' : V.skinDark;
  const face = dead ? '#4a443c' : V.face;
  const path = lobePath(L, W);
  const fpath = facePath(L, W);
  const wig = open > 0.92 ? Math.sin(f * 0.06) * 1.2 : 0;   // 开着时微微呼吸

  const lobe = (side: 1 | -1) => (
    <g key={side} transform={side === 1 ? `rotate(${-a + wig})` : `rotate(${a - wig}) scale(-1,1)`}>
      {/* 瓣体：绿底 + 粉色内膛 */}
      <path d={path} fill={skin} />
      <path d={fpath} fill={face} />
      {/* 外缘齿：沿上缘一排睫毛刺，闭合时交错 */}
      {Array.from({length: 8}).map((_, i) => {
        const t = 0.3 + (i / 7) * 0.66;
        const px = t * L, w = lobeW(L, W, t);
        const len = 26 + 16 * Math.sin(Math.PI * t);
        const lean = 0.45;
        return (
          <path key={i}
            d={`M ${px - 5} ${-w} L ${px + len * lean} ${-w - len} L ${px + 7} ${-w} Z`}
            fill={dead ? skinD : V.tooth} opacity={0.96} />
        );
      })}
      {/* 触发毛：内膛上一排绛红针 */}
      {HAIR_TS.map((t, i) => {
        const px = t * L, w = lobeW(L, W, t);
        const hy = -w + 14;
        const bend = hairTick === i ? 10 : 0;
        return (
          <g key={i} transform={`translate(${px},${hy})`}>
            <circle r={7.5} fill={hairFlash === i ? V.ca : V.hair}
              stroke={hairFlash === i ? '#fff0c0' : V.faceDeep} strokeWidth={2.5} />
            <path d={`M 0 0 Q ${4 + bend} ${-18} ${10 + bend * 1.6} ${-34}`}
              fill="none" stroke={hairFlash === i ? V.ca : V.hair}
              strokeWidth={5.5} strokeLinecap="round" />
          </g>
        );
      })}
      {/* 蜜腺点 */}
      {[0.34, 0.62].map((t, i) => {
        const px = t * L, w = lobeW(L, W, t);
        return <circle key={i} cx={px + 10} cy={-w + 30} r={4.5} fill={V.faceDeep} opacity={0.8} />;
      })}
      {/* 中脊线 */}
      <line x1={10} y1={0} x2={L - 8} y2={0} stroke={skinD} strokeWidth={4} opacity={0.4} />
    </g>
  );

  return (
    <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`}>
      {stem && (
        <g>
          {/* 叶柄 */}
          <path d="M -14 12 C -18 70 -30 110 -44 150 L 20 150 C 8 110 6 70 12 12 Z" fill={skinD} />
          <path d="M -2 20 C -4 70 -8 105 -12 140" fill="none" stroke={skin} strokeWidth={6} opacity={0.6} />
        </g>
      )}
      {/* 铰链 */}
      <ellipse cx={0} cy={6} rx={34} ry={26} fill={skinD} />
      {lobe(1)}
      {lobe(-1)}
      {/* 夹住的虫腿从缝里伸出来挣扎 */}
      {legs > 0 && open < 0.4 && (
        <g stroke={V.bug} strokeWidth={5} strokeLinecap="round">
          {[-14, 0, 14].map((dx, i) => (
            <line key={i} x1={dx} y1={-8} x2={dx + Math.sin(f * 0.9 + i * 2) * 9 * legs}
              y2={-30 - Math.abs(Math.sin(f * 0.7 + i)) * 8} />
          ))}
        </g>
      )}
      {/* 密封线 */}
      {sealed > 0 && (
        <line x1={0} y1={-L * Math.cos((90 - a) * Math.PI / 180) - 4} x2={0} y2={-30}
          stroke={V.faceDeep} strokeWidth={6} opacity={sealed * 0.8} strokeLinecap="round" />
      )}
      {/* 剖面幽灵化 */}
      {ghost > 0 && <rect x={-L - 10} y={-L - 80} width={2 * L + 20} height={L + 170} fill={V.bg4} opacity={0} />}
    </g>
  );
};

/* ---------- 小黑蝇：双飞翅扇动 ---------- */
const Fly = ({x, y, s = 1, rot = 0, flip = 1, squirm = 1, ghost = false}: {
  x: number; y: number; s?: number; rot?: number; flip?: number; squirm?: number; ghost?: boolean;
}) => {
  const f = useCurrentFrame();
  const flap = Math.sin(f * 1.9) * 20 * squirm;
  const body = ghost ? V.shell : V.bug;
  const hi = ghost ? '#7e7488' : V.bugHi;
  return (
    <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s * flip},${s})`} opacity={ghost ? 0.75 : 1}>
      {/* 翅 */}
      <ellipse cx={-6} cy={-24} rx={26} ry={11} fill={V.wing} opacity={ghost ? 0.35 : 0.75}
        transform={`rotate(${-24 - flap})`} />
      <ellipse cx={-6} cy={-22} rx={22} ry={9} fill={V.wing} opacity={ghost ? 0.25 : 0.5}
        transform={`rotate(${-6 + flap})`} />
      {/* 腿 */}
      {[-16, -2, 12].map((lx, i) => (
        <line key={i} x1={lx} y1={12} x2={lx - 6 + Math.sin(f * 0.8 + i * 2) * 4 * squirm} y2={30}
          stroke={body} strokeWidth={4.5} strokeLinecap="round" />
      ))}
      {/* 身 + 头 */}
      <ellipse rx={30} ry={21} fill={body} />
      <ellipse cx={-6} cy={-6} rx={22} ry={12} fill={hi} opacity={0.85} />
      <circle cx={30} cy={-4} r={13} fill={body} />
      <circle cx={34} cy={-8} r={5.5} fill={ghost ? '#5a5266' : '#ff7d6e'} />
      {!ghost && <circle cx={35.5} cy={-9.5} r={2} fill="#fff" opacity={0.9} />}
      {/* 触角 */}
      <line x1={38} y1={-14} x2={48} y2={-24} stroke={body} strokeWidth={3.5} strokeLinecap="round" />
    </g>
  );
};

/* ---------- 枯叶（干扰物） ---------- */
const DryLeaf = ({x, y, rot = 0, s = 1, o = 1}: {x: number; y: number; rot?: number; s?: number; o?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`} opacity={o}>
    <path d="M 0 -44 C 34 -30 44 10 0 46 C -44 10 -34 -30 0 -44 Z" fill={V.dryLeaf} />
    <line x1={0} y1={-40} x2={0} y2={60} stroke="#8a6a3c" strokeWidth={5} strokeLinecap="round" />
    <line x1={0} y1={-6} x2={-20} y2={-22} stroke="#8a6a3c" strokeWidth={3.5} />
    <line x1={0} y1={8} x2={18} y2={-4} stroke="#8a6a3c" strokeWidth={3.5} />
  </g>
);

/* ---------- 计数 HUD：触碰次数 + 20s 窗口条 ---------- */
const CountPanel = ({x, y, n, timer = 1, dim = 0, done = false}: {
  x: number; y: number; n: number; timer?: number; dim?: number; done?: boolean;
}) => (
  <g opacity={1 - dim}>
    <rect x={x} y={y} width={330} height={150} rx={20} fill={V.panel} stroke={V.ca} strokeWidth={4} />
    <text x={x + 26} y={y + 46} fill={V.cream} fontSize={30} fontFamily={FONT_HEAD} letterSpacing={4}>
      触碰计数
    </text>
    <text x={x + 250} y={y + 96} textAnchor="middle" fill={done ? '#7dffa8' : V.ca}
      fontSize={76} fontFamily={FONT_HEAD} fontWeight={900}>{n}</text>
    <text x={x + 306} y={y + 92} fill={C.muted} fontSize={26} fontFamily={FONT_HEAD}>/2</text>
    <rect x={x + 24} y={y + 116} width={282} height={14} rx={7} fill="#123726" />
    <rect x={x + 24} y={y + 116} width={282 * clamp01(timer)} height={14} rx={7} fill={V.ca} />
    <text x={x + 24} y={y + 112} fill={C.muted} fontSize={18} fontFamily={FONT_HEAD}>≈20 秒窗口</text>
  </g>
);

/* ---------- 钙离子迸发 ---------- */
const CaBurst = ({x, y, at, n = 9}: {x: number; y: number; at: number; n?: number}) => {
  const f = useCurrentFrame();
  const t = f - at;
  if (t < 0 || t > 34) return null;
  return (
    <g>
      {Array.from({length: n}).map((_, i) => {
        const ang = rand(i, 3) * Math.PI * 2;
        const d = 10 + rand(i, 4) * 40 + t * (1.6 + rand(i, 5));
        return (
          <circle key={i} cx={x + Math.cos(ang) * d} cy={y + Math.sin(ang) * d * 0.8}
            r={5 + rand(i, 6) * 4} fill={V.ca} opacity={clamp01(1 - t / 30)} />
        );
      })}
    </g>
  );
};

/* ---------- 电火花 ---------- */
const Spark = ({x, y, s = 1, o = 1}: {x: number; y: number; s?: number; o?: number}) => (
  <path transform={`translate(${x},${y}) scale(${s})`} opacity={o}
    d="M 6 -34 L -14 2 L -2 4 L -8 34 L 14 -6 L 2 -8 Z"
    fill={V.ca} stroke="#fff0c0" strokeWidth={2.5} strokeLinejoin="round" />
);

/* ============ 标题卡 ============ */
const TitleCard = ({data}: {data: any}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 13, mass: 0.7}});
  const o = interpolate(f, [0, 8], [0, 1]);
  const pop = spring({frame: f - 5, fps, config: {damping: 11, stiffness: 150}});
  return (
    <AbsoluteFill>
      <Backdrop top={V.bg} bottom={V.bg} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 小草叶衬景 */}
        {[180, 1740].map((gx, i) => (
          <g key={i} transform={`translate(${gx},880)`}>
            <path d="M 0 0 C -18 -60 -12 -120 8 -160 C 22 -110 16 -50 0 0 Z" fill={V.leafDark} />
            <path d="M 20 0 C 30 -50 46 -92 74 -118 C 66 -64 46 -22 20 0 Z" fill={V.leaf} />
          </g>
        ))}
        {/* 大夹子张开 + 苍蝇绕着飞 */}
        <g transform={`translate(960,430) scale(${Math.max(0.001, pop * 0.82)})`}>
          <Trap x={0} y={0} open={1} stem={false} />
        </g>
        <Fly x={700 + Math.sin(f * 0.08) * 26} y={330 + Math.cos(f * 0.11) * 18} s={0.9} />
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 330}}>
          <div style={{color: V.face, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 26}}>
            冷知识 · 第 {data.episode} 期
          </div>
          <div style={{color: C.text, fontSize: 92, fontWeight: 900, fontFamily: FONT_HEAD, letterSpacing: 5}}>
            {data.title}
          </div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ============ S0 会数数：落叶碰一下不夹，计数器归零 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);

  const popK = spring({frame: f - 6, fps, config: {damping: 12, stiffness: 130}});
  // 落叶飘入：贝塞尔轨迹 f24→f140 落到右瓣触发毛上
  const leafK = clamp01((f - 24) / 116);
  const leafPts = [[300, -60], [560, 140], [860, 320], [1064, 462]];
  const cubic = (pts: number[][], t: number) => {
    const u = 1 - t;
    return [
      u * u * u * pts[0][0] + 3 * u * u * t * pts[1][0] + 3 * u * t * t * pts[2][0] + t * t * t * pts[3][0],
      u * u * u * pts[0][1] + 3 * u * u * t * pts[1][1] + 3 * u * t * t * pts[2][1] + t * t * t * pts[3][1],
    ];
  };
  const [lx, ly] = cubic(leafPts, leafK);
  const touchF = 140;                                     // 叶子碰到毛
  const leafSlide = clamp01((f - touchF - 130) / 60);     // 之后滑走
  const hairHit = f >= touchF ? 1 : -1;
  const cnt = f < touchF ? 0 : (f < touchF + 150 ? 1 : 0); // 计数 1 然后清零
  const timer = f < touchF ? 0 : clamp01(1 - (f - touchF) / 150);
  const flyK = clamp01((f - 400) / 90);                   // 苍蝇后段入场
  const flyX = interpolate(flyK, [0, 1], [1980, 1190]) + Math.sin(f * 0.09) * 20;
  const flyY = interpolate(flyK, [0, 1], [350, 468]) + Math.sin(f * 0.13) * 26 * (1 - flyK);

  return (
    <FadeIn>
      <Backdrop top={V.bg} bottom={V.bg} />
      <SceneTag index={0} name={sec.name} accent={V.face} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 沼泽小草 + 水面反光条 */}
        <rect x={0} y={838} width={1920} height={242} fill={V.mud} />
        <ellipse cx={960} cy={848} rx={430} ry={26} fill={V.skinDark} opacity={0.55} />
        {[120, 380, 1560, 1800].map((gx, i) => (
          <g key={i} transform={`translate(${gx},846) scale(${i % 2 ? -1 : 1})`}>
            <path d="M 0 0 C -14 -52 -8 -104 10 -138 C 20 -90 14 -40 0 0 Z" fill={V.leafDark} />
            <path d="M 22 0 C 34 -42 50 -76 76 -100 C 66 -52 46 -18 22 0 Z" fill={V.leaf} />
          </g>
        ))}

        {/* 主角捕蝇草 */}
        <g transform={`translate(960,682) scale(${Math.max(0.001, popK)})`}>
          <Trap x={0} y={0} open={1} hairFlash={hairHit === 1 ? 1 : -1} />
        </g>

        {/* 落叶飘进来碰毛 */}
        {leafK > 0 && leafSlide < 1 && (
          <g>
            <DryLeaf x={lx + leafSlide * 130} y={ly + leafSlide * 90 - leafSlide * 10}
              rot={-30 + leafK * 80 + Math.sin(f * 0.1) * 10} s={0.9} o={1 - leafSlide * 0.9} />
          </g>
        )}
        {f >= touchF && f < touchF + 40 && <Ripple x={1064} y={440} startFrame={touchF} color={V.ca} />}
        {f >= touchF && f < touchF + 40 && <CaBurst x={1064} y={448} at={touchF} n={7} />}

        {/* 计数 HUD */}
        {f >= touchF + 8 && (
          <CountPanel x={1500} y={170} n={cnt} timer={timer} dim={f > touchF + 170 ? clamp01((f - touchF - 170) / 30) : 0} />
        )}

        {/* 苍蝇压轴入场，落在夹口边 */}
        {flyK > 0 && <Fly x={flyX} y={flyY} s={1} flip={-1} squirm={flyK >= 1 ? 0.25 : 1} />}

        {/* 引导虚线：指向触发毛 */}
        {f > beat(1) + 30 && f < beat(1) + 130 && (
          <g opacity={clamp01((f - beat(1) - 30) / 20)}>
            <line x1={1410} y1={640} x2={1150} y2={470} stroke={V.cream} strokeWidth={4}
              strokeDasharray="8 12" />
            <text x={1470} y={660} fill={V.cream} fontSize={32} fontFamily={FONT_HEAD} fontWeight={700}>
              触发毛 ×6
            </text>
          </g>
        )}
      </svg>
      <Stamp text="虫还是落叶？" x={150} y={170} at={beat(0) + 24} color={C.coral} size={54} rot={-5} />
      <Stamp text="碰一次＝干扰信号" x={1270} y={330} at={touchF + 70} color={V.faceDeep} size={46} rot={4} />
      <Stamp text="它会数数！" x={170} y={330} at={beat(1) + 8} color={V.skinDark} size={58} rot={-4} />
      <Chip text="20秒内两次才动手" x={700} y={160} at={beat(1) + 90} accent={V.ca} />
    </FadeIn>
  );
};

/* ============ S1 机关毛：放大镜看触发毛＝机械开关 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);

  const magK = spring({frame: f - beat(1) + 20, fps, config: {damping: 12, stiffness: 140}});
  const probeF = beat(1) + 60;                       // 探针下压
  const probeK = clamp01((f - probeF) / 26);
  const sparkK = clamp01((f - probeF - 22) / 14);
  const pathK = clamp01((f - beat(1) - 150) / 60);   // 必经之路虚线
  const CX = 1360, CY = 470, R = 300;

  const hairXs = [1180, 1240, 1300, 1360, 1420, 1480, 1540];
  const barTop = 500, barBot = 610;
  const bendI = 3;                                    // 被压的那根

  return (
    <FadeIn>
      <Backdrop top={V.bg} bottom={V.bg} />
      <SceneTag index={1} name={sec.name} accent={V.face} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <defs>
          <clipPath id="mg213"><circle cx={CX} cy={CY} r={R - 16} /></clipPath>
        </defs>
        <rect x={0} y={838} width={1920} height={242} fill={V.mud} />
        {[140, 1740].map((gx, i) => (
          <g key={i} transform={`translate(${gx},846) scale(${i % 2 ? -1 : 1})`}>
            <path d="M 0 0 C -14 -52 -8 -104 10 -138 C 20 -90 14 -40 0 0 Z" fill={V.leafDark} />
          </g>
        ))}

        {/* 左：整夹 + 虚线圈指向内膛 */}
        <Trap x={520} y={680} s={0.78} open={1} hairFlash={f > probeF ? bendI - 1 : -1} />
        <circle cx={598} cy={476} r={72} fill="none" stroke={V.face} strokeWidth={4.5}
          strokeDasharray="10 10" opacity={0.9} />
        <line x1={656} y1={430} x2={CX - R * 0.72} y2={CY - R * 0.42} stroke={V.face}
          strokeWidth={4} strokeDasharray="10 10" opacity={0.6} />

        {/* 右：放大镜 → 内膛横截面 */}
        <circle cx={CX} cy={CY} r={R} fill="#123527" stroke={V.face} strokeWidth={7} />
        <circle cx={CX} cy={CY} r={R + 14} fill="none" stroke={V.face} strokeWidth={3} opacity={0.35} />
        <g clipPath="url(#mg213)" opacity={clamp01(magK)}>
          {/* 内膛粉面（截面：上粉下绿两层） */}
          <rect x={CX - 300} y={barTop} width={600} height={110} fill={V.face} />
          <rect x={CX - 300} y={barTop + 110} width={600} height={140} fill={V.skin} />
          <rect x={CX - 300} y={barTop - 40} width={600} height={50} fill="#123527" opacity={0} />
          {/* 猎物必经之路：虚线弧穿过毛尖 */}
          {pathK > 0 && (
            <path d={`M ${CX - 270} ${barTop - 130} Q ${CX} ${barTop - 30} ${CX + 270} ${barTop - 140}`}
              fill="none" stroke={V.cream} strokeWidth={5} strokeDasharray="14 14" opacity={pathK * 0.9} />
          )}
          {/* 脚印 */}
          {pathK > 0.4 && [0, 1, 2, 3].map((i) => (
            <ellipse key={i} cx={CX - 200 + i * 130} cy={barTop - 118 + (i % 2) * 10}
              rx={10} ry={6} fill={V.cream} opacity={0.55} transform={`rotate(${i % 2 ? 14 : -10})`} />
          ))}
          {/* 6根触发毛 + 基部感受细胞 */}
          {hairXs.map((hx, i) => {
            const bend = i === bendI ? probeK * 20 : 0;
            const hit = i === bendI && sparkK > 0;
            const pop = spring({frame: f - 20 - i * 5, fps, config: {damping: 11, stiffness: 170}});
            return (
              <g key={i} transform={`translate(${hx},${barTop}) scale(1,${Math.max(0.001, pop)})`}>
                {/* 毛：带关节弯折 */}
                <path d={`M 0 0 Q ${6 + bend} -60 ${14 + bend * 1.8} -120`}
                  fill="none" stroke={hit ? V.ca : V.hair} strokeWidth={9} strokeLinecap="round" />
                <circle cx={0} cy={0} r={10} fill={hit ? V.ca : V.hair} />
                {/* 基部感受细胞 */}
                <circle cx={0} cy={38} r={16} fill={hit ? V.ca : '#7a1f3e'}
                  stroke={V.faceDeep} strokeWidth={3} />
                {hit && <Spark x={14 + bend * 1.8} y={-128} s={0.9} o={sparkK} />}
                {hit && <Spark x={0} y={40} s={0.7} o={sparkK * 0.9} />}
              </g>
            );
          })}
          {/* 信号小点沿细胞链跑 */}
          {sparkK > 0 && sparkK < 1 && (
            <circle cx={CX - 150 + sparkK * 300} cy={barTop + 48} r={8} fill={V.ca} />
          )}
        </g>
        <text x={CX} y={CY - R - 26} textAnchor="middle" fill={V.face} fontSize={40}
          fontFamily={FONT_HEAD} fontWeight={700}>内膛放大 20×</text>
        <text x={CX} y={CY + R + 46} textAnchor="middle" fill={C.muted} fontSize={30}
          fontFamily={FONT_HEAD}>弯到阈值 → 电信号</text>
      </svg>
      <Stamp text="六七根机械开关" x={150} y={180} at={beat(0) + 24} color={V.faceDeep} size={50} rot={-5} />
      <Chip text="基部连着感受细胞" x={120} y={330} at={probeF + 30} accent={V.ca} />
      <Stamp text="全是哨兵位" x={220} y={660} at={beat(1) + 190} color={V.skinDark} size={48} rot={-4} />
    </FadeIn>
  );
};

/* ============ S2 数两下：第一次上膛→第二次啪！→超时清零 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);

  const HX = 830, HY = 690;                            // 铰链位置
  const t1F = beat(0) + 40;                            // 第一次碰毛
  const t2F = beat(0) + 330;                           // 第二次碰毛 → 夹
  const snapK = spring({frame: f - t2F, fps, config: {damping: 12, stiffness: 300, mass: 0.5}});
  const open = f < t2F ? 1 : Math.max(0, 1 - snapK);
  const shake = f > t2F && f < t2F + 18 ? Math.sin((f - t2F) * 1.7) * (18 - (f - t2F)) * 0.5 : 0;
  const flyIn = clamp01((f - 16) / 50);
  const flyX = interpolate(flyIn, [0, 1], [1750, HX + 108]) ;
  const flyY = interpolate(flyIn, [0, 1], [300, HY - 210]);
  const flyWob = f > t1F - 20 && f < t1F + 30 ? Math.sin(f * 1.4) * 6 : 0;
  const timer = f < t1F ? 0 : (f < t2F ? 1 - (f - t1F) / (t2F - t1F) : 0);
  const n = f < t1F ? 0 : (f < t2F ? 1 : 2);
  const resetK = clamp01((f - beat(1)) / 30);          // cue1: 清零演示
  const ghostK = clamp01((f - beat(1) - 30) / 30);     // "如果没第二下" 幽灵面板

  return (
    <FadeIn>
      <Backdrop top={V.bg2} bottom={V.bg2} />
      <SceneTag index={2} name={sec.name} accent={V.ca} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <g transform={`translate(${shake},${Math.abs(shake) * 0.4})`}>
          <rect x={0} y={846} width={1920} height={234} fill="#0a242e" />
          <ellipse cx={HX} cy={852} rx={360} ry={22} fill="#123b3f" opacity={0.7} />

          {/* 捕蝇草：被第二次触碰触发夹合 */}
          <Trap x={HX} y={HY} open={open} hairFlash={f >= t1F && f < t1F + 50 ? 2 : (f >= t2F && f < t2F + 30 ? 1 : -1)}
            legs={f > t2F ? 1 : 0} />

          {/* 苍蝇爬进夹口 */}
          {flyIn > 0 && open > 0.15 && (
            <Fly x={flyX} y={flyY + flyWob} s={0.94} flip={-1} squirm={open > 0.9 ? 0.5 : 0} />
          )}
          {open <= 0.15 && (
            <Fly x={HX + 40} y={HY - 200} s={0.85} flip={-1} ghost squirm={0} />
          )}

          {/* 触碰涟漪 + 钙离子 */}
          {f >= t1F && f < t1F + 40 && <Ripple x={HX + 158} y={HY - 250} startFrame={t1F} color={V.ca} />}
          {f >= t1F && f < t1F + 34 && <CaBurst x={HX + 130} y={HY - 230} at={t1F} n={9} />}
          {f >= t2F && f < t2F + 40 && <Ripple x={HX + 90} y={HY - 300} startFrame={t2F} color="#fff0c0" />}
          {f >= t2F && f < t2F + 34 && <CaBurst x={HX + 80} y={HY - 280} at={t2F} n={12} />}

          {/* 啪！冲击放射线 */}
          {f > t2F + 2 && f < t2F + 22 && (
            <g opacity={1 - (f - t2F) / 22} stroke="#fff3c0" strokeWidth={7} strokeLinecap="round">
              {Array.from({length: 9}).map((_, i) => {
                const ang = (-160 + i * 24) * Math.PI / 180;
                const r0 = 210, r1 = 300 + (f - t2F) * 8;
                return <line key={i} x1={HX + Math.cos(ang) * r0} y1={HY - 130 + Math.sin(ang) * r0 * 0.7}
                  x2={HX + Math.cos(ang) * r1} y2={HY - 130 + Math.sin(ang) * r1 * 0.7} />;
              })}
            </g>
          )}
        </g>

        {/* HUD 计数 */}
        {f >= t1F + 4 && <CountPanel x={1480} y={160} n={n} timer={timer} done={n >= 2} />}

        {/* 上膛小标签 */}
        {f >= t1F + 10 && f < t2F - 10 && (
          <Chip text="第一下：只上膛" x={170} y={200} at={t1F + 10} accent={V.ca} />
        )}

        {/* cue1：超时清零·幽灵面板 */}
        {ghostK > 0 && (
          <g opacity={ghostK}>
            <rect x={1280} y={560} width={560} height={250} rx={22} fill="#0a2530"
              stroke={C.muted} strokeWidth={4} strokeDasharray="12 10" />
            <text x={1560} y={620} textAnchor="middle" fill={C.muted} fontSize={30}
              fontFamily={FONT_HEAD}>假如没有第二下</text>
            <g transform="translate(1420,690)">
              <rect width={280} height={18} rx={9} fill="#123726" />
              <rect width={280 * clamp01(1 - (f - beat(1) - 40) / 120)} height={18} rx={9} fill={V.ca} />
              <text x={140} y={60} textAnchor="middle" fill={V.ca} fontSize={34}
                fontFamily={FONT_HEAD} fontWeight={700}>信号衰减 → 清零</text>
            </g>
            <text x={1700} y={705} textAnchor="middle" fill={C.coral} fontSize={56}
              fontFamily={FONT_HEAD} fontWeight={900}>不夹</text>
          </g>
        )}
      </svg>
      <Stamp text="啪！" x={HX + 250} y={HY - 420} at={t2F + 4} color={C.coral} size={84} rot={-8} />
      <Chip text="第二次：跨阈值" x={170} y={330} at={t2F + 16} accent="#7dffa8" />
      <Stamp text="自带遗忘机制" x={170} y={620} at={beat(1) + 60} color={V.faceDeep} size={48} rot={-4} />
    </FadeIn>
  );
};

/* ============ S3 假警报：雨滴碰一下不夹=省钱；爬虫两下才夹=验证码 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);

  const HX = 760, HY = 690;
  const dropF = beat(2) + 40;                          // 雨滴砸中毛
  const rainOn = clamp01((f - beat(2)) / 24) * (1 - clamp01((f - beat(3)) / 30));
  const blip = f >= dropF && f < dropF + 140;          // 计数 1 然后衰减
  const timerB = blip ? 1 - (f - dropF) / 140 : 0;
  const flyK = clamp01((f - beat(3) - 40) / 60);       // 爬虫入场
  const hop1 = beat(3) + 110;                          // 第一碰
  const hop2 = beat(3) + 168;                          // 第二碰
  const snapK = spring({frame: f - hop2, fps, config: {damping: 12, stiffness: 300, mass: 0.5}});
  const open = f < hop2 ? 1 : Math.max(0, 1 - snapK);
  const shake = f > hop2 && f < hop2 + 18 ? Math.sin((f - hop2) * 1.7) * (18 - (f - hop2)) * 0.45 : 0;
  const n = f < hop1 ? (blip ? 1 : 0) : (f < hop2 ? 1 : 2);
  const ghostK = clamp01((f - beat(2) + 20) / 30) * (1 - clamp01((f - beat(3) - 10) / 40));

  return (
    <FadeIn>
      <Backdrop top={V.bg3} bottom={V.bg3} />
      <SceneTag index={3} name={sec.name} accent={V.drop} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Rain on={rainOn} n={54} />
        <g transform={`translate(${shake},0)`}>
          <rect x={0} y={846} width={1920} height={234} fill="#16303f" />
          <ellipse cx={HX} cy={852} rx={330} ry={22} fill="#0e2733" opacity={0.8} />

          <Trap x={HX} y={HY} open={open} hairFlash={blip ? 1 : (f >= hop1 && f < hop1 + 40 ? 2 : -1)}
            legs={f > hop2 ? 1 : 0} />

          {/* 一颗高亮雨滴砸中触发毛 */}
          {f > dropF - 30 && f < dropF + 26 && (
            <g>
              <line x1={HX + 196} y1={HY - 560 + (f - (dropF - 30)) * 12}
                x2={HX + 176} y2={HY - 500 + (f - (dropF - 30)) * 12}
                stroke={V.drop} strokeWidth={8} strokeLinecap="round" />
            </g>
          )}
          {f >= dropF && f < dropF + 40 && <Ripple x={HX + 180} y={HY - 330} startFrame={dropF} color={V.drop} />}

          {/* 爬虫两下 */}
          {flyK > 0 && open > 0.15 && (
            <Fly
              x={f < hop1 ? interpolate(clamp01((f - beat(3) - 40) / 70), [0, 1], [1760, HX + 210])
                : f < hop2 ? HX + 210 - (f - hop1) * 0.9 : HX + 120}
              y={f < hop1 ? 420 + Math.sin(f * 0.12) * 16 : HY - 245}
              s={0.94} flip={-1} squirm={0.6} />
          )}
          {open <= 0.15 && <Fly x={HX + 30} y={HY - 200} s={0.85} flip={-1} ghost />}
          {f >= hop1 && f < hop1 + 34 && <Ripple x={HX + 190} y={HY - 300} startFrame={hop1} color={V.ca} />}
          {f >= hop2 && f < hop2 + 30 && <Ripple x={HX + 120} y={HY - 320} startFrame={hop2} color="#fff0c0" />}
        </g>

        {/* 太阳+能量条：空夹白搭一天光合作用 */}
        <Sun x={330} y={210} r={62} />
        <g opacity={ghostK}>
          <rect x={150} y={330} width={360} height={150} rx={18} fill="#14303d"
            stroke={V.ca} strokeWidth={4} />
          <text x={330} y={372} textAnchor="middle" fill={V.cream} fontSize={26} fontFamily={FONT_HEAD}>
            空夹一次 ＝ 白晒一天
          </text>
          <rect x={176} y={396} width={308} height={20} rx={10} fill="#0d2229" />
          <rect x={176} y={396} width={308 * clamp01(1 - (f - beat(2)) / 160)} height={20} rx={10}
            fill={V.ca} />
          <text x={330} y={456} textAnchor="middle" fill={C.coral} fontSize={26}
            fontFamily={FONT_HEAD} fontWeight={700}>膨压开合很耗能</text>
        </g>

        {blip && <CountPanel x={1480} y={170} n={1} timer={timerB} />}
        {f >= hop1 - 10 && <CountPanel x={1480} y={170} n={n} timer={f < hop2 ? 1 - (f - hop1) / (hop2 - hop1) : 0} done={n >= 2} />}
      </svg>
      <Stamp text="省钱！" x={620} y={170} at={beat(1) + 10} color={V.ca} size={64} rot={-5} />
      <Stamp text="雨滴：只敲一下" x={1270} y={600} at={dropF + 30} color={V.drop} size={44} rot={4} />
      <Stamp text="两下＝活猎物签名" x={200} y={560} at={hop2 + 12} color={V.skinDark} size={50} rot={-4} />
      <Chip text="报警器验证码·带时效" x={1180} y={330} at={hop2 + 70} accent={V.ca} />
    </FadeIn>
  );
};

/* ============ S4 慢炖：五次确认锁死→分泌消化液→只剩空壳→耗材 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);

  const HX = 700, HY = 680;
  const lockF = beat(0) + 150;                          // 攒够五次→锁死
  const openStart = 0.22;
  const lockK = spring({frame: f - lockF, fps, config: {damping: 13, stiffness: 280, mass: 0.5}});
  const open = f < lockF ? openStart : Math.max(0, openStart * (1 - lockK));
  const reopenF = beat(1) + 90;                         // 吃完重新张开
  const reopenK = clamp01((f - reopenF) / 90);
  const open2 = reopenK > 0 ? reopenK * 0.8 : open;
  const nStruggle = Math.min(5, 3 + Math.floor(clamp01((f - beat(0) - 30) / 36) * 3));
  const digestK = clamp01((f - lockF - 30) / 40);       // 消化液滴
  const dayN = Math.floor(5 + clamp01((f - beat(1) - 10) / 120) * 7);
  const shellK = clamp01((f - reopenF - 50) / 40);

  return (
    <FadeIn>
      <Backdrop top={V.bg4} bottom={V.bg4} />
      <SceneTag index={4} name={sec.name} accent={V.face} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <rect x={0} y={846} width={1920} height={234} fill={V.mud} />

        {/* 主夹：留缝→锁死→张开 */}
        <Trap x={HX} y={HY} open={open2} sealed={digestK} legs={f > lockF - 60 && f < reopenF ? 1 : 0} />
        {/* 缝里挣扎的虫影 */}
        {open2 < 0.3 && digestK < 1 && f < reopenF && (
          <g transform={`translate(${HX},${HY - 210})`} opacity={0.9}>
            <ellipse rx={26} ry={14} fill={V.bug}
              transform={`rotate(${Math.sin(f * 0.8) * 8})`} />
          </g>
        )}
        {/* 锁死后剖面化：内部胃袋 */}
        {digestK > 0 && open2 < 0.12 && (
          <g opacity={digestK}>
            <ellipse cx={HX} cy={HY - 150} rx={110} ry={130} fill="#54282a" />
            {/* 消化液滴 */}
            {Array.from({length: 6}).map((_, i) => (
              <circle key={i} cx={HX - 70 + i * 28} cy={HY - 240 + ((f * 2.2 + i * 53) % 130)}
                r={7 + rand(i, 9) * 4} fill={V.ca} opacity={0.85} />
            ))}
            {/* 虫被消化中：越来越小 */}
            <g transform={`translate(${HX},${HY - 140})`}>
              <Fly x={0} y={0} s={0.8 * (1 - shellK * 0.5)} ghost={shellK > 0.5} squirm={0.5} />
            </g>
            <line x1={HX + 106} y1={HY - 160} x2={HX + 180} y2={HY - 190} stroke={V.face}
              strokeWidth={3.5} strokeDasharray="6 8" opacity={0.8} />
            <text x={HX + 196} y={HY - 184} fill={V.face} fontSize={26}
              fontFamily={FONT_HEAD}>胃袋剖面</text>
          </g>
        )}

        {/* 重新张开只剩空壳 */}
        {shellK > 0 && (
          <g opacity={shellK}>
            <Fly x={HX - 20 - shellK * 150} y={HY - 280 + shellK * 60} s={0.7} ghost rot={-20 - shellK * 35} />
            <text x={HX - 300} y={HY - 330} fill={V.cream} fontSize={28} fontFamily={FONT_HEAD}>
              只剩几丁质空壳
            </text>
          </g>
        )}

        {/* 挣扎计数 → 五次锁死 */}
        {f > beat(0) + 20 && f < lockF + 70 && (
          <g>
            <rect x={1180} y={170} width={330} height={140} rx={20} fill="#2a1c14" stroke={V.face} strokeWidth={4} />
            <text x={1206} y={218} fill={V.cream} fontSize={28} fontFamily={FONT_HEAD} letterSpacing={3}>
              挣扎计数
            </text>
            <text x={1450} y={288} textAnchor="middle" fill={nStruggle >= 5 ? '#7dffa8' : V.face}
              fontSize={70} fontFamily={FONT_HEAD} fontWeight={900}>×{nStruggle}</text>
            <text x={1206} y={288} fill={C.muted} fontSize={24} fontFamily={FONT_HEAD}>≈5 次才锁死</text>
          </g>
        )}

        {/* 天数计数牌 */}
        {f > beat(1) + 5 && (
          <g>
            <rect x={1330} y={420} width={300} height={130} rx={20} fill="#2a1c14" stroke={V.ca} strokeWidth={4} />
            <text x={1480} y={472} textAnchor="middle" fill={V.cream} fontSize={26} fontFamily={FONT_HEAD}>
              慢炖中
            </text>
            <text x={1480} y={534} textAnchor="middle" fill={V.ca} fontSize={54}
              fontFamily={FONT_HEAD} fontWeight={900}>第 {dayN} 天</text>
            <text x={1480} y={586} textAnchor="middle" fill={C.muted} fontSize={22}
              fontFamily={FONT_HEAD}>5–12 天</text>
          </g>
        )}

        {/* 一排小夹：耗材寿命 3~4 次 */}
        {f > beat(1) + 100 && (
          <g>
            {[0, 1, 2, 3].map((i) => {
              const dx = 1240 + i * 150;
              const dead = i === 3;
              return (
                <g key={i} transform={`translate(${dx},${i === 0 ? 700 : 710})`}>
                  <Trap x={0} y={0} s={0.26} open={dead ? 0.15 : 1} dead={dead} stem={false} />
                  <text x={0} y={110} textAnchor="middle" fontSize={22} fontFamily={FONT_HEAD}
                    fill={dead ? V.shell : V.leaf}>{dead ? '报废' : `第${i + 1}夹`}</text>
                </g>
              );
            })}
            <text x={1460} y={870} textAnchor="middle" fill={V.cream} fontSize={26}
              fontFamily={FONT_HEAD}>每个夹子一生只能开工 3~4 次</text>
          </g>
        )}
      </svg>
      <Stamp text="五次确认·锁死！" x={150} y={180} at={lockF + 10} color={V.faceDeep} size={52} rot={-5} />
      <Chip text="分泌消化液" x={170} y={330} at={lockF + 60} accent={V.ca} />
      <Stamp text="限量耗材·别老逗它" x={180} y={560} at={beat(1) + 190} color={C.coral} size={50} rot={-4} />
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
      <Backdrop top={V.bg} bottom={V.bg} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <rect x={0} y={880} width={1920} height={200} fill={V.mud} />
        {[120, 1780].map((gx, i) => (
          <g key={i} transform={`translate(${gx},886) scale(${i % 2 ? -1 : 1})`}>
            <path d="M 0 0 C -14 -52 -8 -104 10 -138 C 20 -90 14 -40 0 0 Z" fill={V.leafDark} />
          </g>
        ))}
        <g transform={`translate(960,800) scale(${Math.max(0.001, pop * 0.66)})`}>
          <Trap x={0} y={0} open={0.9 + Math.sin(f * 0.03) * 0.08} stem={false} />
        </g>
        <Fly x={960 + Math.sin(f * 0.07) * 270} y={520 + Math.cos(f * 0.11) * 34} s={0.8}
          flip={Math.cos(f * 0.07) > 0 ? -1 : 1} />
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -320}}>
        <div style={{fontSize: 68, fontWeight: 800, color: C.text}}>两根毛 · 二十秒 · 五次触碰</div>
        <div style={{fontSize: 38, color: C.muted, marginTop: 22, letterSpacing: 6}}>植物的算法 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E213 = ({data = tl}: {data?: any}) => {
  const {fps} = useVideoConfig();
  return (
    <AbsoluteFill style={{background: V.bg}}>
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
