import {AbsoluteFill, Audio, Sequence, interpolate, interpolateColors, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01, rand} from '../lib/style';
import {Backdrop, Ripple, Stamp, SceneTag, Chip, Subtitles, ProgressBar, Sun, FadeIn} from '../lib/atoms';
import tl from '../timeline_207.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：香蕉明黄 + 叶绿 + 棕点（平涂） ============ */
const B = {
  bg: '#0f4a3a',        // 主底：叶绿
  bgLab: '#0d4256',     // 机制场景：深青
  bgSoil: '#33251a',    // 草本场景：暖棕土
  bgRoom: '#40300f',    // 催熟库：暖棕
  fruit: '#ffd23f',     // 熟蕉黄
  fruitHi: '#ffe680',
  fruitDark: '#e0a41f',
  green: '#a9c94e',     // 青蕉
  greenDark: '#7ba336',
  stem: '#8a6a3f',      // 果柄
  rachis: '#b28a4a',    // 花轴
  bract: '#7d4a78',     // 香蕉蕾（紫）
  tip: '#5a3a1a',       // 梢头褐
  spot: '#5f3d1e',      // 芝麻斑
  leaf: '#4fbf74',
  leafDark: '#2e7d4d',
  sheath: '#9ccf6a',    // 叶鞘
  sheathHi: '#c6e896',
  soil: '#4a3420',
  soilDark: '#2e2013',
  bark: '#7a5230',
  barkDark: '#4a3018',
  ring: '#5e3d22',
  cream: '#fff3d6',
  gas: '#9fd8ff',
  hood: '#171726',      // 遮光罩
  panel: '#0a2e26',
};

/* ---------- 香蕉脊椎：等曲率圆弧，heading 从 th0 线性转到 th1（度，y 向下顺时针） ---------- */
const spine = (len: number, th0: number, th1: number) => {
  const n = 24;
  const ds = len / n;
  const pts: {x: number; y: number; h: number}[] = [];
  let x = 0, y = 0;
  for (let i = 0; i <= n; i++) {
    pts.push({x, y, h: (th0 + (th1 - th0) * (i / n)) * (Math.PI / 180)});
    const hm = (th0 + (th1 - th0) * ((i + 0.5) / n)) * (Math.PI / 180);
    x += ds * Math.cos(hm);
    y += ds * Math.sin(hm);
  }
  return pts;
};

/* ---------- 单根香蕉：origin=果柄着生点，bend 0=直下 1=上翘月牙 ---------- */
const Banana = ({x, y, rot = 0, len = 170, wMax = 34, th0 = 60, bend = 1,
  fill = B.fruit, dark = B.fruitDark, spots = 0, seed = 0, sway = 0, ghost = false}: {
  x: number; y: number; rot?: number; len?: number; wMax?: number; th0?: number;
  bend?: number; fill?: string; dark?: string; spots?: number; seed?: number;
  sway?: number; ghost?: boolean;
}) => {
  const f = useCurrentFrame();
  const b = clamp01(bend);
  const side = th0 >= 90 ? 1 : -1;                    // 左串顺时针翘 / 右串逆时针翘
  const endYoung = th0 + side * 10;                   // 刚坐果：基本直
  const endRipe = side > 0 ? 262 : -82;               // 熟透：梢头朝上、微微内扣
  const th1 = endYoung + (endRipe - endYoung) * b;
  const pts = spine(len, th0, th1);
  const n = pts.length;
  const wAt = (t: number) => (wMax / 2) * Math.max(0.14, Math.sin(Math.PI * (0.09 + 0.82 * t)));
  const at = (i: number, k: number) => {
    const p = pts[i];
    const nx = -Math.sin(p.h), ny = Math.cos(p.h);
    const w = wAt(i / (n - 1));
    return `${(p.x + nx * w * k).toFixed(1)},${(p.y + ny * w * k).toFixed(1)}`;
  };
  const outline = `M ${pts.map((_, i) => at(i, 1)).join(' L ')} L ${pts.map((_, i) => at(i, -1)).reverse().join(' L ')} Z`;
  const ridge = (k: number) => `M ${pts.map((_, i) => at(i, k)).join(' L ')}`;
  const tip = pts[n - 1];
  const h0 = pts[0].h;
  const sw = sway ? Math.sin(f * 0.045 + seed * 2.1) * sway : 0;

  if (ghost) {
    return (
      <g transform={`translate(${x},${y}) rotate(${rot})`}>
        <path d={outline} fill="none" stroke={B.cream} strokeWidth={4}
          strokeDasharray="12 12" opacity={0.75} />
        <circle cx={tip.x} cy={tip.y} r={5} fill={B.cream} opacity={0.75} />
      </g>
    );
  }
  return (
    <g transform={`translate(${x},${y}) rotate(${rot + sw})`}>
      {/* 果柄（反向伸出，接到花轴上） */}
      <line x1={-Math.cos(h0) * wMax * 0.95} y1={-Math.sin(h0) * wMax * 0.95}
        x2={Math.cos(h0) * 3} y2={Math.sin(h0) * 3}
        stroke={B.stem} strokeWidth={wMax * 0.3} strokeLinecap="round" />
      <path d={outline} fill={fill} />
      {/* 内侧（凹面）阴影 + 两条棱线 */}
      <path d={ridge(-0.55 * side)} fill="none" stroke={dark} strokeWidth={wMax * 0.22}
        opacity={0.35} strokeLinecap="round" />
      <path d={ridge(0.5)} fill="none" stroke={dark} strokeWidth={2.8} opacity={0.5} />
      <path d={ridge(-0.5)} fill="none" stroke={dark} strokeWidth={2.8} opacity={0.5} />
      {/* 凸面高光 */}
      <path d={ridge(0.7 * side)} fill="none" stroke={B.fruitHi} strokeWidth={3.2}
        opacity={0.55} strokeLinecap="round" />
      {/* 芝麻斑 */}
      {spots > 0 && Array.from({length: 11}).map((_, i) => {
        const t = 0.2 + (i / 10) * 0.72;
        const idx = Math.round(t * (n - 1));
        const p = pts[idx];
        const k = (rand(i, seed + 5) - 0.5) * 1.15;
        const nx = -Math.sin(p.h), ny = Math.cos(p.h);
        const w = wAt(t);
        return (
          <circle key={i} cx={p.x + nx * w * k} cy={p.y + ny * w * k}
            r={2.6 + rand(i, seed + 6) * 3.2} fill={B.spot}
            opacity={spots * (0.5 + rand(i, seed + 7) * 0.45)} />
        );
      })}
      {/* 梢头（宿存花端） */}
      <circle cx={tip.x} cy={tip.y} r={wAt(1) * 0.62} fill={B.tip} />
    </g>
  );
};

/* ---------- 一串倒挂香蕉：origin=皇冠（连着假茎的果柄顶） ---------- */
const Bunch = ({x, y, bend, rows = 4, scale = 1, fill = B.fruit, dark = B.fruitDark,
  spots = 0, sway = 0, lenMul = 1}: {
  x: number; y: number; bend: number; rows?: number; scale?: number; fill?: string;
  dark?: string; spots?: number; sway?: number; lenMul?: number;
}) => {
  const f = useCurrentFrame();
  const gap = 72;
  const sw = sway ? Math.sin(f * 0.04 + x * 0.01) * sway : 0;
  // 每把果：外侧两根（略小、色深）+ 中两根 + 前排中央一根
  const hand = [128, 52, 104, 76, 88];
  return (
    <g transform={`translate(${x},${y}) scale(${scale})`}>
      <g transform={`rotate(${sw},0,-140)`}>
        {/* 果柄主杆（向上接假茎） */}
        <rect x={-15} y={-155} width={30} height={170} rx={13} fill={B.stem} />
        <ellipse cx={0} cy={2} rx={26} ry={14} fill={B.stem} />
        {/* 花轴 */}
        <rect x={-9} y={-4} width={18} height={rows * gap + 66} rx={9} fill={B.rachis} />
        {Array.from({length: rows}).map((_, i) => {
          const ny = 34 + i * gap;
          return (
            <g key={i}>
              <ellipse cx={0} cy={ny} rx={17} ry={13} fill={B.rachis} />
              {hand.map((th0, j) => {
                const center = j === 4;
                const bx = Math.cos(th0 * Math.PI / 180) * 20;
                const bj = clamp01(bend * 1.35 - j * 0.05 - i * 0.06);
                return (
                  <Banana key={j} x={bx} y={ny + (center ? 10 : j % 2 * 7)}
                    th0={th0} bend={bj}
                    len={(center ? 196 : 176 - (j % 2) * 16) * lenMul}
                    wMax={center ? 36 : 32}
                    fill={center ? fill : (j % 2 ? dark : fill)}
                    dark={dark} spots={spots} seed={i * 7 + j} sway={0} />
                );
              })}
            </g>
          );
        })}
        {/* 末端苞片：香蕉蕾 */}
        <g transform={`translate(0,${rows * gap + 70})`}>
          <path d="M 0 -6 C 40 4 46 58 12 104 C -2 124 -20 122 -24 106 C -6 72 -30 24 0 -6 Z"
            fill={B.bract} />
          <path d="M 4 8 C 22 22 26 56 10 88" fill="none" stroke="#5d3558" strokeWidth={5}
            strokeLinecap="round" opacity={0.7} />
        </g>
      </g>
    </g>
  );
};

/* ---------- 香蕉叶：长桨形 + 中肋 ---------- */
const BananaLeaf = ({x, y, rot = 0, len = 330, w = 95, flip = 1, tint = B.leaf}: {
  x: number; y: number; rot?: number; len?: number; w?: number; flip?: number; tint?: string;
}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${flip},1)`}>
    <path d={`M 0 0 Q ${len * 0.5} ${-w} ${len} ${-w * 0.1} Q ${len * 0.55} ${w * 0.62} 0 ${w * 0.3} Z`}
      fill={tint} />
    <line x1={10} y1={w * 0.1} x2={len * 0.95} y2={-w * 0.09} stroke={B.leafDark}
      strokeWidth={4.5} strokeLinecap="round" opacity={0.75} />
    {[0.32, 0.55, 0.78].map((t, i) => (
      <line key={i} x1={len * t} y1={-w * 0.62 + t * 6} x2={len * t + 16} y2={-w * 0.2}
        stroke={B.leafDark} strokeWidth={3} opacity={0.4} />
    ))}
  </g>
);

/* ---------- 香蕉"树"：假茎 + 叶冠 + 地下球茎 ---------- */
const BananaPlant = ({x, y, h = 560, squash = 0}: {x: number; y: number; h?: number; squash?: number}) => {
  const f = useCurrentFrame();
  const wob = squash > 0 ? Math.sin((f) * 0.9) * squash * 5 : 0;
  const w = 92;
  return (
    <g transform={`translate(${x},${y}) rotate(${wob})`}>
      {/* 叶冠 */}
      <BananaLeaf x={-8} y={-h + 8} rot={-168} len={300} w={82} tint={B.leafDark} />
      <BananaLeaf x={6} y={-h} rot={-12} len={330} w={92} />
      <BananaLeaf x={0} y={-h - 14} rot={-52} len={310} w={88} tint="#63cf83" />
      <BananaLeaf x={-4} y={-h - 4} rot={-128} len={300} w={84} tint="#63cf83" />
      {/* 假茎 */}
      <path d={`M ${-w / 2} 0 C ${-w / 2 - 8} ${-h * 0.3} ${-w * 0.38} ${-h * 0.72} ${-w * 0.3} ${-h}
        L ${w * 0.3} ${-h} C ${w * 0.38} ${-h * 0.72} ${w / 2 + 8} ${-h * 0.3} ${w / 2} 0 Z`}
        fill={B.sheath} />
      {[0.16, 0.34, 0.52, 0.7, 0.87].map((t, i) => {
        const ww = (w * (1 - t * 0.36)) / 2 + 4;
        return <path key={i} d={`M ${-ww} ${-h * t} Q 0 ${-h * t + 20} ${ww} ${-h * t}`}
          fill="none" stroke={B.leafDark} strokeWidth={5} opacity={0.55} />;
      })}
      <line x1={-w * 0.14} y1={-14} x2={-w * 0.2} y2={-h + 18} stroke="#ffffff"
        strokeWidth={7} opacity={0.18} strokeLinecap="round" />
    </g>
  );
};

/* ---------- 雨滴滴滑走的小球 ---------- */
const SlideDrop = ({pts, k, on}: {pts: {x: number; y: number}[]; k: number; on: number}) => {
  if (on <= 0) return null;
  const i = Math.min(pts.length - 1, Math.floor(k * (pts.length - 1)));
  const p = pts[i];
  return <circle cx={p.x} cy={p.y} r={10} fill={C.sky} opacity={on * 0.9} />;
};

/* ============ 标题卡 ============ */
const TitleCard = ({data}: {data: any}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 13, mass: 0.7}});
  const o = interpolate(f, [0, 8], [0, 1]);
  const flip = interpolate(f, [12, 66], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  return (
    <AbsoluteFill>
      <Backdrop top={B.bg} bottom={B.bg} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 一根小香蕉当场表演掉头 */}
        <g transform={`translate(850,310) scale(${Math.max(0.001, clamp01(o))})`}>
          <Banana x={0} y={0} th0={62} bend={flip} len={240} wMax={48} />
          {flip < 0.5 && <Banana x={0} y={0} th0={62} bend={1} len={240} wMax={48} ghost />}
        </g>
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 190}}>
          <div style={{color: B.fruit, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 26}}>
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

/* ============ S0 开场：整串倒挂，小果直溜溜朝下 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);

  const popK = spring({frame: f - 8, fps, config: {damping: 12, stiffness: 140}});
  const ghostK = clamp01((f - beat(2) - 10) / 30);          // cue2: 演示掉头
  const qK = clamp01((f - beat(1) - 6) / 18);               // "被晒弯"问号
  const rot = interpolate(f, [beat(2), beat(2) + 90], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});

  return (
    <FadeIn>
      <Backdrop top={B.bg} bottom={B.bg} />
      <SceneTag index={0} name={sec.name} accent={B.fruit} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 大叶子衬景 */}
        <BananaLeaf x={120} y={180} rot={18} len={420} w={110} tint="#0c3d2f" />
        <BananaLeaf x={1800} y={160} rot={160} len={420} w={110} tint="#0c3d2f" />
        <Sun x={1560} y={210} r={80} />

        {/* 一整串：倒挂 + 直果 */}
        <g transform={`translate(760,300) scale(${1.18 * Math.max(0.001, popK)})`}>
          <Bunch x={0} y={0} bend={0.04 + rot * 0.2} rows={4} fill={B.green} dark={B.greenDark} />
        </g>

        {/* 掉头预告：虚线鬼魂 + 弧形箭头 */}
        {ghostK > 0 && (
          <g opacity={ghostK} transform="translate(760,300) scale(1.18)">
            <Banana x={20} y={104} th0={52} bend={1} len={176} wMax={32} ghost />
          </g>
        )}
        {ghostK > 0 && (
          <g opacity={ghostK}>
            <path d="M 1150 640 Q 1250 540 1190 430" fill="none" stroke={B.fruit}
              strokeWidth={7} strokeDasharray="4 18" strokeLinecap="round" />
            <path d="M 1190 430 l -22 6 M 1190 430 l 6 22" stroke={B.fruit} strokeWidth={7}
              fill="none" strokeLinecap="round" />
          </g>
        )}

        {/* 太阳旁边的"？" */}
        {qK > 0 && (
          <text x={1690} y={260} fill={B.cream} fontSize={110} fontFamily={FONT_HEAD}
            fontWeight={900} opacity={qK} transform={`rotate(${Math.sin(f / 10) * 6} 1690 260)`}>?</text>
        )}
      </svg>
      <Stamp text="被太阳晒弯的？" x={1230} y={330} at={beat(1) + 8} color={C.coral} size={54} rot={-6} />
      <Stamp text="刚坐果是直的！" x={170} y={250} at={beat(2) + 6} color={B.leafDark} size={52} rot={-5} />
      <Chip text="几周后集体掉头" x={1330} y={560} at={beat(2) + 80} accent={B.fruit} />
    </FadeIn>
  );
};

/* ============ S1 掉头：负向地性——平衡石沉底，生长素下行，下侧长得快 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);

  const bend = interpolate(f, [30, 560], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const magK = spring({frame: f - beat(1) + 18, fps, config: {damping: 12, stiffness: 140}});
  const arrowK = clamp01((f - beat(1) - 10) / 24);      // 生长素下行箭头
  const growK = clamp01((f - beat(2) + 10) / 60);       // 下侧细胞拉长

  const CX = 1430, CY = 510, R = 248;
  // 细胞网格：上排矮、下排被拉长
  const cellW = 118, gap = 14;
  const topH = 86, botH = 86 + 62 * growK;
  const gridW = cellW * 3 + gap * 2;
  const gx0 = CX - gridW / 2, gy0 = CY - (topH + botH + gap) / 2;

  return (
    <FadeIn>
      <Backdrop top={B.bgLab} bottom={B.bgLab} />
      <SceneTag index={1} name={sec.name} accent={B.fruit} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <defs>
          <clipPath id="mg207"><circle cx={CX} cy={CY} r={R - 14} /></clipPath>
        </defs>

        {/* 左：整串做慢动作掉头 */}
        <g transform="translate(620,310) scale(1.12)">
          <Bunch x={0} y={0} bend={bend} rows={3} fill={B.green} dark={B.greenDark} />
        </g>
        {/* 放大虚线圈 → 剖面 */}
        <circle cx={668} cy={560} r={58} fill="none" stroke={B.fruit} strokeWidth={4}
          strokeDasharray="10 10" opacity={0.9} />
        <line x1={720} y1={540} x2={CX - R * 0.75} y2={CY - R * 0.5} stroke={B.fruit}
          strokeWidth={4} strokeDasharray="10 10" opacity={0.6} />

        {/* 重力计：左缘大箭头 */}
        <g opacity={arrowK}>
          <line x1={230} y1={360} x2={230} y2={560} stroke={C.sky} strokeWidth={9} strokeLinecap="round" />
          <path d="M 230 584 l -18 -30 M 230 584 l 18 -30" stroke={C.sky} strokeWidth={9}
            fill="none" strokeLinecap="round" />
          <text x={230} y={330} textAnchor="middle" fill={C.sky} fontSize={34}
            fontFamily={FONT_HEAD} fontWeight={700}>重力</text>
        </g>

        {/* 右：果实剖面放大镜 */}
        <circle cx={CX} cy={CY} r={R} fill="#082834" stroke={B.fruit} strokeWidth={6} />
        <circle cx={CX} cy={CY} r={R + 14} fill="none" stroke={B.fruit} strokeWidth={3} opacity={0.3} />
        <g clipPath="url(#mg207)" opacity={clamp01(magK)}>
          {/* 两排细胞 */}
          {[0, 1].map((row) =>
            [0, 1, 2].map((col) => {
              const cx = gx0 + col * (cellW + gap) + cellW / 2;
              const cy = row === 0 ? gy0 + topH / 2 : gy0 + topH + gap + botH / 2;
              const hh = row === 0 ? topH : botH;
              const aux = row === 0 ? 2 : 1 + Math.round(4 * growK);
              return (
                <g key={`${row}${col}`}>
                  <rect x={cx - cellW / 2} y={cy - hh / 2} width={cellW} height={hh} rx={22}
                    fill={row === 0 ? '#12525e' : '#175f66'}
                    stroke="#3fae8c" strokeWidth={4} />
                  {/* 平衡石：沉在细胞底部 */}
                  {[0, 1, 2].map((d) => (
                    <circle key={d} cx={cx - 26 + d * 26} cy={cy + hh / 2 - 16} r={7}
                      fill="#0a1c26" stroke="#7ecbf2" strokeWidth={2.5} />
                  ))}
                  {/* 生长素：靠下侧攒得更多 */}
                  {Array.from({length: aux}).map((_, d) => (
                    <circle key={`a${d}`} cx={cx - cellW / 2 + 18 + d * 16}
                      cy={cy + hh / 2 - 34} r={6} fill={C.amber}
                      opacity={0.4 + growK * 0.6} />
                  ))}
                </g>
              );
            })
          )}
          {/* 生长素往下运的小箭头 */}
          {arrowK > 0 && [0, 1, 2].map((col) => {
            const cx = gx0 + col * (cellW + gap) + cellW / 2;
            const ay = gy0 + topH + gap / 2;
            return (
              <path key={col} d={`M ${cx} ${ay - 14} l -10 -14 M ${cx} ${ay - 14} l 10 -14 M ${cx} ${ay - 40} L ${cx} ${ay - 14}`}
                stroke={C.amber} strokeWidth={6} fill="none" strokeLinecap="round" opacity={arrowK} />
            );
          })}
        </g>
        <text x={CX} y={CY - R - 28} textAnchor="middle" fill={B.fruit} fontSize={44}
          fontFamily={FONT_HEAD} fontWeight={700}>果实剖面 · 细胞</text>
        <text x={gx0 - 24} y={gy0 + topH / 2 + 10} textAnchor="end" fill={C.muted}
          fontSize={30} fontFamily={FONT_HEAD} opacity={clamp01(magK)}>上侧</text>
        <text x={gx0 - 24} y={gy0 + topH + gap + botH / 2 + 10} textAnchor="end" fill={C.amber}
          fontSize={30} fontFamily={FONT_HEAD} fontWeight={700}
          opacity={clamp01(magK)}>下侧·长得快</text>
      </svg>
      <Chip text="平衡石＝内置重力计" x={170} y={170} at={beat(1) + 6} accent={C.sky} />
      <Chip text="生长素往下侧运" x={1330} y={830} at={beat(1) + 70} accent={C.amber} />
      <Stamp text="负向地性" x={150} y={830} at={beat(2) + 16} color={B.fruitDark} size={56} rot={-4} />
    </FadeIn>
  );
};

/* ============ S2 帮凶：遮光照样弯 + 收成梳子打包 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);

  const bend = interpolate(f, [16, 150], [0.25, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const hoodK = interpolate(f, [52, 84], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const phaseB = clamp01((f - beat(1)) / 26);            // 对比段进场
  const phaseA = 1 - phaseB;

  // 梳子打包的雨滴滑轨：贴着最外侧那根弯蕉的凸面滚下去
  const slidePts = spine(176, 52, -82).map((p) => ({x: p.x + 12, y: p.y + 27}));
  const dropK = ((f * 1.4) % 90) / 90;

  return (
    <FadeIn>
      <Backdrop top="#123f33" bottom="#123f33" />
      <SceneTag index={2} name={sec.name} accent={B.fruit} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* ===== 上半场：遮光实验 ===== */}
        <g opacity={phaseA}>
          <Sun x={1560} y={200} r={76} />
          {/* 太阳→果穗的细虚线：只是助攻 */}
          <line x1={1490} y1={250} x2={1060} y2={430} stroke={C.amber} strokeWidth={5}
            strokeDasharray="6 16" opacity={0.55} />
          <g transform="translate(700,320) scale(1.12)">
            <Bunch x={0} y={0} bend={bend} rows={4} fill={B.green} dark={B.greenDark} />
          </g>
          {/* 遮光罩：透光黑棚，果在里面照样翘 */}
          <g opacity={hoodK}>
            <path d="M 440 860 L 440 330 Q 440 240 560 240 L 840 240 Q 960 240 960 330 L 960 860 Z"
              fill={B.hood} opacity={0.68} />
            <path d="M 440 860 L 440 330 Q 440 240 560 240 L 840 240 Q 960 240 960 330 L 960 860"
              fill="none" stroke={C.muted} strokeWidth={5} strokeDasharray="14 12" />
            <rect x={560} y={268} width={280} height={58} rx={14} fill={B.hood}
              stroke={C.muted} strokeWidth={4} strokeDasharray="10 8" />
            <text x={700} y={308} textAnchor="middle" fill={C.muted} fontSize={34}
              fontFamily={FONT_HEAD} letterSpacing={6}>遮光罩 · 弱光</text>
          </g>
        </g>

        {/* ===== 下半场：直筷子 vs 弯梳子 ===== */}
        {phaseB > 0 && (
          <g opacity={phaseB}>
            {/* 左：全是直的 */}
            <g transform="translate(520,620)">
              {[52, 70, 88, 106, 124].map((th0, j) => (
                <Banana key={j} x={Math.cos(th0 * Math.PI / 180) * 20} y={(j % 2) * 8}
                  th0={th0} bend={0.02} len={210} wMax={34}
                  fill={B.green} dark={B.greenDark} seed={j} />
              ))}
              {/* 占地虚线圈：摊得很大 */}
              <ellipse cx={0} cy={180} rx={330} ry={200} fill="none" stroke={C.coral}
                strokeWidth={5} strokeDasharray="14 12" opacity={0.85} />
              <g transform="translate(-258,-40)">
                <circle r={44} fill="none" stroke={C.coral} strokeWidth={10} />
                <line x1={-30} y1={-30} x2={30} y2={30} stroke={C.coral} strokeWidth={10}
                  strokeLinecap="round" />
                <line x1={-30} y1={30} x2={30} y2={-30} stroke={C.coral} strokeWidth={10}
                  strokeLinecap="round" />
              </g>
              <text x={0} y={-30} textAnchor="middle" fill={C.coral} fontSize={40}
                fontFamily={FONT_HEAD} fontWeight={700}>全直：一把筷子戳出去</text>
            </g>
            {/* 右：收成梳子 */}
            <g transform="translate(1400,470) scale(0.92)">
              <Bunch x={0} y={0} bend={1} rows={3} fill={B.fruit} dark={B.fruitDark} />
              <ellipse cx={0} cy={200} rx={215} ry={225} fill="none" stroke={B.leaf}
                strokeWidth={5} strokeDasharray="14 12" opacity={0.9} />
              <g transform="translate(-196,30)">
                <circle r={44} fill="none" stroke={B.leaf} strokeWidth={10} />
                <path d="M -24 0 l 16 18 l 32 -36" stroke={B.leaf} strokeWidth={10}
                  fill="none" strokeLinecap="round" strokeLinejoin="round" />
              </g>
              {/* 雨水顺弧滑走：贴在弯蕉凸面上滚动 */}
              <SlideDrop pts={slidePts} k={dropK} on={phaseB} />
            </g>
            <text x={1400} y={296} textAnchor="middle" fill={B.leaf} fontSize={40}
              fontFamily={FONT_HEAD} fontWeight={700}>全弯：收成一把梳子</text>
            <text x={1670} y={600} textAnchor="middle" fill={C.sky} fontSize={30}
              fontFamily={FONT_HEAD} opacity={phaseB}>雨水顺弧流走</text>
          </g>
        )}
      </svg>
      {/* 上半场字幕贴片随遮光实验一起退场 */}
      <div style={{opacity: phaseA}}>
        <Stamp text="遮住也照弯！" x={1080} y={170} at={100} color={B.fruitDark}
          size={52} rot={4} />
        <Stamp text="主谋是重力" x={1310} y={330} at={beat(0) + 150} color={C.coral} size={50} rot={-5} />
        <Chip text="光只是修整齐" x={150} y={180} at={beat(0) + 200} accent={C.amber} />
      </div>
      <div style={{opacity: phaseB}}>
        <Stamp text="弯 = 打包方案" x={740} y={140} at={beat(1) + 30} color={B.leafDark} size={52} rot={-3} />
      </div>
    </FadeIn>
  );
};

/* ============ S3 是草：假茎=叶鞘卷的 vs 真树年轮 + 一脚一个坑 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);

  const magK = spring({frame: f - beat(1) + 14, fps, config: {damping: 12, stiffness: 140}});
  const kickF = beat(1) + 120;                            // "一脚能踹出个坑"
  const bootX = interpolate(f, [kickF - 20, kickF], [-260, 80], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const squash = clamp01(1 - Math.abs(f - kickF - 8) / 26);
  const cormK = clamp01((f - beat(2)) / 30);              // 地下球茎亮起
  const suckerK = spring({frame: f - beat(2) - 30, fps, config: {damping: 10, stiffness: 150}});

  const GY = 830;
  const CX = 1300, CY = 470, R = 235;

  return (
    <FadeIn>
      <Backdrop top={B.bgSoil} bottom={B.bgSoil} />
      <SceneTag index={3} name={sec.name} accent={B.sheath} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <defs>
          <clipPath id="she207"><circle cx={CX} cy={CY} r={R - 14} /></clipPath>
        </defs>
        {/* 土层 */}
        <rect x={0} y={GY} width={1920} height={250} fill={B.soil} />
        <rect x={0} y={GY - 12} width={1920} height={16} fill={B.soilDark} />

        {/* 香蕉"树" */}
        <BananaPlant x={540} y={GY + 6} h={545} squash={squash} />
        {/* 地下球茎 + 接班新芽 */}
        <g transform={`translate(540,${GY + 74})`} opacity={0.5 + cormK * 0.5}>
          <ellipse rx={92} ry={52} fill={B.stem} />
          {[0.7, 0.45, 0.22].map((k, i) => (
            <ellipse key={i} rx={92 * k} ry={52 * k} fill="none" stroke={B.tip}
              strokeWidth={4} opacity={0.6} />
          ))}
          {[-52, 0, 52].map((dx, i) => (
            <line key={i} x1={dx} y1={44} x2={dx + (i - 1) * 10} y2={78} stroke={B.stem}
              strokeWidth={5} strokeLinecap="round" />
          ))}
        </g>
        {cormK > 0 && (
          <g transform={`translate(540,${GY + 74})`} opacity={cormK}>
            <circle r={118} fill="none" stroke={B.fruit} strokeWidth={5} strokeDasharray="12 10" />
            <text x={0} y={-140} textAnchor="middle" fill={B.fruit} fontSize={32}
              fontFamily={FONT_HEAD} fontWeight={700}>真茎在地下＝球茎</text>
          </g>
        )}
        {/* 新芽接班 */}
        {[372, 706].map((sx, i) => (
          <g key={i} transform={`translate(${sx},${GY + 4}) scale(${Math.max(0.001, suckerK * (i ? 0.8 : 1))})`}>
            <path d="M 0 0 C -14 -30 -10 -62 4 -84 C 16 -60 14 -28 0 0 Z" fill={B.leaf} />
            <path d="M 2 -6 C 20 -34 40 -46 62 -48 C 46 -26 24 -10 2 -6 Z" fill={B.leafDark} />
          </g>
        ))}

        {/* 踹一脚：靴子飞入 + 坑印 */}
        {f > kickF - 26 && f < kickF + 46 && (
          <g transform={`translate(${368 + bootX},${GY - 120}) rotate(-18)`}>
            <rect x={-110} y={-34} width={150} height={52} rx={18} fill="#6b4a2e" />
            <rect x={-110} y={-104} width={64} height={86} rx={18} fill="#7d5736" />
            <rect x={26} y={-28} width={26} height={40} rx={8} fill="#523620" />
          </g>
        )}
        {squash > 0 && (
          <g transform={`translate(${486},${GY - 118})`} opacity={squash}>
            <path d="M 0 -26 Q -26 0 0 26 Q 14 0 0 -26 Z" fill={B.soilDark} />
            <Ripple x={0} y={0} startFrame={kickF} color={B.fruit} />
          </g>
        )}
        {/* 放大虚线 → 假茎剖面 */}
        <circle cx={540} cy={GY - 300} r={62} fill="none" stroke={B.sheath} strokeWidth={4}
          strokeDasharray="10 10" />
        <line x1={598} y1={GY - 320} x2={CX - R * 0.75} y2={CY + R * 0.55} stroke={B.sheath}
          strokeWidth={4} strokeDasharray="10 10" opacity={0.6} />

        {/* 假茎横截面：一层层叶鞘卷 */}
        <circle cx={CX} cy={CY} r={R} fill="#123326" stroke={B.sheath} strokeWidth={6} />
        <circle cx={CX} cy={CY} r={R + 14} fill="none" stroke={B.sheath} strokeWidth={3} opacity={0.3} />
        <g clipPath="url(#she207)" opacity={clamp01(magK)}>
          <g transform={`translate(${CX},${CY})`}>
            {[186, 152, 118, 84, 50].map((r, i) => (
              <circle key={i} r={r} fill="none"
                stroke={i % 2 ? B.sheath : B.sheathHi} strokeWidth={30}
                pathLength={360} strokeDasharray="292 68"
                transform={`rotate(${-50 + i * 27})`} opacity={0.95} />
            ))}
            <circle r={20} fill={B.sheathHi} />
          </g>
        </g>
        <text x={CX} y={CY + R + 48} textAnchor="middle" fill={B.sheath} fontSize={36}
          fontFamily={FONT_HEAD} fontWeight={700}>假茎剖面＝叶鞘一层卷一层</text>

        {/* 真树年轮对照 */}
        <g transform="translate(1690,330)" opacity={clamp01(magK)}>
          <circle r={98} fill={B.bark} stroke={B.barkDark} strokeWidth={9} />
          {[0.76, 0.56, 0.37, 0.18].map((k, i) => (
            <circle key={i} r={98 * k} fill="none" stroke={B.ring} strokeWidth={4.5} />
          ))}
          <circle r={5} fill={B.ring} />
          <text x={0} y={152} textAnchor="middle" fill={B.bark} fontSize={30}
            fontFamily={FONT_HEAD} fontWeight={700}>真树：年轮＋木质部</text>
        </g>
      </svg>
      <Stamp text="它是草！" x={790} y={150} at={beat(0) + 10} color={B.leafDark} size={64} rot={-5} />
      <Chip text="没有木质部·一脚一个坑" x={160} y={300} at={kickF + 8} accent={B.sheath} />
      <Stamp text="用完就换" x={200} y={640} at={beat(2) + 60} color={B.bract} size={48} rot={-6} />
    </FadeIn>
  );
};

/* ============ S4 定型：催熟库熏乙烯，只长斑不变弯；挑蕉看斑点 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);

  const ripen = interpolate(f, [36, 270], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const fruitFill = interpolateColors(ripen, [0, 0.55, 1], [B.green, '#d8c53e', B.fruit]);
  const spotK = clamp01((ripen - 0.62) / 0.3);
  const pickK = clamp01((f - beat(1)) / 30);              // 挑蕉段

  return (
    <FadeIn>
      <Backdrop top={B.bgRoom} bottom={B.bgRoom} />
      <SceneTag index={4} name={sec.name} accent={B.fruit} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 催熟库大箱子 */}
        <g opacity={1 - pickK * 0.55}>
          <rect x={600} y={180} width={880} height={590} rx={26} fill="#4d3a17"
            stroke="#8a6a35" strokeWidth={7} />
          <rect x={600} y={180} width={880} height={82} rx={26} fill="#3c2c10" />
          <rect x={600} y={236} width={880} height={26} fill="#3c2c10" />
          <text x={648} y={236} fill={B.fruit} fontSize={40} fontFamily={FONT_HEAD}
            fontWeight={700} letterSpacing={6}>催熟库 · 20℃</text>
          <rect x={1250} y={196} width={196} height={52} rx={12} fill={B.gas} opacity={0.92} />
          <text x={1348} y={232} textAnchor="middle" fill="#12303f" fontSize={30}
            fontFamily={FONT_HEAD} fontWeight={700}>C₂H₄ 乙烯</text>
          {/* 两层货架，每层三把 */}
          {[430, 630].map((sy, r) => (
            <g key={r}>
              <rect x={636} y={sy + 128} width={810} height={16} rx={8} fill="#8a6a35" />
              {[700, 920, 1140, 1340].map((bx, i) => (
                <g key={i} transform={`translate(${bx},${sy}) scale(0.42)`}>
                  <Bunch x={0} y={0} bend={1} rows={2} fill={fruitFill}
                    dark={ripen > 0.55 ? B.fruitDark : B.greenDark} spots={spotK * 0.5} />
                </g>
              ))}
            </g>
          ))}
          {/* 乙烯分子飘 */}
          {Array.from({length: 9}).map((_, i) => (
            <circle key={i}
              cx={660 + rand(i, 61) * 760 + Math.sin(f / 22 + i) * 18}
              cy={300 + rand(i, 62) * 420 + Math.cos(f / 19 + i * 2) * 14}
              r={7 + rand(i, 63) * 5} fill={B.gas} opacity={0.4 + rand(i, 64) * 0.35} />
          ))}
          {/* 门缝 */}
          <line x1={1480} y1={190} x2={1480} y2={760} stroke="#2b1f0a" strokeWidth={6}
            strokeDasharray="20 14" />
        </g>

        {/* 挑蕉指南：三张卡片横排（避开字幕区） */}
        {pickK > 0 && (
          <g opacity={pickK}>
            {([380, 960, 1540] as const).map((cx, i) => (
              <g key={i}>
                <rect x={cx - 200} y={742} width={400} height={172} rx={24} fill="#2e2210"
                  stroke="#8a6a35" strokeWidth={4.5} />
                <g transform={`translate(${cx - 118},806) rotate(${i === 0 ? -18 : -12})`}>
                  <Banana x={0} y={0} th0={62} bend={1} len={i === 1 ? 122 : 102}
                    wMax={i === 1 ? 28 : 25}
                    fill={i === 0 ? B.green : i === 2 ? '#d8a92e' : B.fruit}
                    dark={i === 0 ? B.greenDark : i === 2 ? '#b07f1e' : B.fruitDark}
                    spots={i === 0 ? 0 : i === 1 ? 0.55 : 1} />
                </g>
                <text x={cx + 34} y={892} textAnchor="middle" fontSize={31} fontFamily={FONT_HEAD}
                  fontWeight={i === 1 ? 800 : 400}
                  fill={i === 1 ? B.fruit : C.muted}>
                  {i === 0 ? '还青着' : i === 1 ? '芝麻斑·巅峰' : '熟过头'}
                </text>
                {/* ✓ / ✗ 角标 */}
                <g transform={`translate(${cx + 156},788)`}>
                  <circle r={32} fill="none"
                    stroke={i === 1 ? B.leaf : i === 2 ? C.coral : C.muted} strokeWidth={8} />
                  {i === 1
                    ? <path d="M -17 0 l 12 13 l 24 -27" stroke={B.leaf} strokeWidth={8}
                        fill="none" strokeLinecap="round" strokeLinejoin="round" />
                    : i === 2
                      ? <g stroke={C.coral} strokeWidth={8} strokeLinecap="round">
                          <line x1={-18} y1={-18} x2={18} y2={18} />
                          <line x1={-18} y1={18} x2={18} y2={-18} />
                        </g>
                      : <line x1={-16} y1={0} x2={16} y2={0} stroke={C.muted} strokeWidth={8}
                          strokeLinecap="round" />}
                </g>
              </g>
            ))}
          </g>
        )}
      </svg>
      <Stamp text="只长斑，不再弯" x={150} y={170} at={beat(0) + 190} color={B.fruitDark} size={52} rot={-5} />
      <div style={{opacity: 1 - pickK}}>
        <Chip text="青蕉进库·熏一两天" x={1330} y={830} at={beat(0) + 60} accent={B.gas} />
      </div>
      <Stamp text="挑蕉别看弯度！" x={190} y={330} at={beat(1) + 12} color={C.coral} size={54} rot={-5} />
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
      <Backdrop top={B.bg} bottom={B.bg} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <g transform={`translate(960,880) scale(${Math.max(0.001, pop)}) rotate(-8)`}>
          <Banana x={0} y={0} th0={62} bend={1} len={300} wMax={58} spots={0.55} sway={2} />
        </g>
        <BananaLeaf x={120} y={1010} rot={-12} len={380} w={100} tint="#0c3d2f" />
        <BananaLeaf x={1780} y={1000} rot={192} len={380} w={100} tint="#0c3d2f" />
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -170}}>
        <div style={{fontSize: 72, fontWeight: 800, color: C.text}}>弯是长出来的，不是晒出来的</div>
        <div style={{fontSize: 38, color: C.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E207 = ({data = tl}: {data?: any}) => {
  const {fps} = useVideoConfig();
  return (
    <AbsoluteFill style={{background: B.bg}}>
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
