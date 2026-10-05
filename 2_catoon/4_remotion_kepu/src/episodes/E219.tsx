import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01, rand} from '../lib/style';
import {Backdrop, Ripple, Stamp, SceneTag, Chip, Subtitles, ProgressBar, Sun, FadeIn} from '../lib/atoms';
import tl from '../timeline_219.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：砖墙暖灰 × 藤蔓绿 × 吸盘紫（纯色平涂） ============ */
const V = {
  bg0: '#2c2734',        // S0 深紫灰
  bg1: '#24353a',        // S1 深青·剖面实验台
  bg2: '#382a20',        // S2 深暖棕·力学
  bg3: '#1e2738',        // S3 深靛·向暗
  bg4: '#2e3024',        // S4 深橄榄灰
  wall: '#a98f78',       // 砖墙暖灰
  wallHi: '#c0a88d',
  wallDark: '#82684f',
  mortar: '#d9c8b3',     // 砖缝
  tile: '#93b7c6',       // 瓷砖好墙
  tileLine: '#7ba2b3',
  peel: '#e2cfa4',       // 老粉刷层
  crack: '#5c4630',      // 裂缝
  vine: '#5fb96a',       // 藤蔓绿
  vineDark: '#3a8a4c',
  leaf: '#7bcf82',
  leafDark: '#49a05a',
  pad: '#b78ae0',        // 吸盘紫
  padDeep: '#8f5cc0',
  padRim: '#d9b9f4',
  glue: '#ffd166',       // 多糖粘液琥珀
  glueDark: '#e8ae3f',
  cream: '#f6ecd8',
  panel: '#1d1a26',
  shadow: '#141a28',     // 暗墙块
  shadowBrick: '#232c42',
};

/* ---------- 砖墙（平涂砖块 + 砖缝底） ---------- */
const BrickWall = ({x, y, w, h, brick = V.wall, brickHi = V.wallHi, mortar = V.mortar, dark = false}: {
  x: number; y: number; w: number; h: number; brick?: string; brickHi?: string; mortar?: string; dark?: boolean;
}) => {
  const bw = 156, bh = 78, gap = 10;
  const rows = Math.ceil(h / (bh + gap));
  const cols = Math.ceil(w / (bw + gap)) + 1;
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} fill={mortar} />
      {Array.from({length: rows}).map((_, r) => {
        const off = r % 2 ? (bw + gap) / 2 : 0;
        return Array.from({length: cols}).map((_, c) => {
          const bx = x + c * (bw + gap) - off;
          const by = y + r * (bh + gap);
          if (bx + bw < x || bx > x + w) return null;
          return <rect key={`${r}-${c}`} x={bx} y={by} width={bw} height={bh} rx={8}
            fill={dark ? (r % 3 === 1 ? brick : V.shadowBrick) : (r % 3 === 1 ? brick : brickHi)} />;
        });
      })}
    </g>
  );
};

/* ---------- 爬山虎叶：葡萄科锯齿圆叶 ---------- */
const Leaf = ({x, y, s = 1, rot = 0, c = V.leaf, stemLen = 54}: {
  x: number; y: number; s?: number; rot?: number; c?: string; stemLen?: number;
}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`}>
    <line x1={0} y1={0} x2={0} y2={-stemLen} stroke={V.vineDark} strokeWidth={7} strokeLinecap="round" />
    <g transform={`translate(0,${-stemLen - 58})`}>
      <path d="M 0 6
        C -10 -4 -30 -8 -40 -26
        C -60 -18 -68 -42 -56 -60
        C -70 -70 -62 -98 -40 -96
        C -38 -120 -12 -124 0 -110
        C 12 -124 38 -120 40 -96
        C 62 -98 70 -70 56 -60
        C 68 -42 60 -18 40 -26
        C 30 -8 10 -4 0 6 Z" fill={c} />
      <line x1={0} y1={0} x2={0} y2={-104} stroke={V.leafDark} strokeWidth={4.5} />
      <line x1={0} y1={-38} x2={-26} y2={-58} stroke={V.leafDark} strokeWidth={3.5} />
      <line x1={0} y1={-38} x2={26} y2={-58} stroke={V.leafDark} strokeWidth={3.5} />
      <line x1={0} y1={-66} x2={-20} y2={-84} stroke={V.leafDark} strokeWidth={3.5} />
      <line x1={0} y1={-66} x2={20} y2={-84} stroke={V.leafDark} strokeWidth={3.5} />
    </g>
  </g>
);

/* ---------- 贴墙吸盘（侧面）：口朝 -X·flip 贴墙，碗身鼓向反向 ---------- */
const PadOnWall = ({x, y, s = 1, o = 1, hi = false, flip = 1}: {
  x: number; y: number; s?: number; o?: number; hi?: boolean; flip?: number;
}) => (
  <g transform={`translate(${x},${y}) scale(${s * flip},${s})`} opacity={o}>
    <ellipse cx={0} cy={0} rx={8} ry={32} fill={hi ? V.padRim : V.padDeep} />
    <path d="M 0 -32 C 38 -32 56 -17 56 0 C 56 17 38 32 0 32 Z" fill={hi ? V.padRim : V.pad} />
    <path d="M 10 -19 C 32 -19 45 -9 45 0 C 45 9 32 19 10 19" fill="none"
      stroke={hi ? '#fff' : V.padRim} strokeWidth={5} opacity={0.75} />
  </g>
);

/* ---------- 正面吸盘（俯视图小圆垫） ---------- */
const PadDisc = ({x, y, r = 26, o = 1}: {x: number; y: number; r?: number; o?: number}) => (
  <g transform={`translate(${x},${y})`} opacity={o}>
    <circle r={r} fill={V.pad} />
    <circle r={r * 0.62} fill={V.padDeep} />
    <circle r={r * 0.3} fill={V.padRim} opacity={0.85} />
    <circle cx={-r * 0.3} cy={-r * 0.32} r={r * 0.16} fill="#fff" opacity={0.55} />
  </g>
);

/* ---------- 弹簧卷须：A→B 的螺旋线，coils=圈数 ---------- */
const coilPath = (x0: number, y0: number, x1: number, y1: number, coils: number, r: number) => {
  const dx = x1 - x0, dy = y1 - y0;
  const len = Math.max(1, Math.hypot(dx, dy));
  const nx = -dy / len, ny = dx / len;
  const n = Math.max(8, Math.round(coils * 18));
  let d = `M ${x0.toFixed(1)} ${y0.toFixed(1)}`;
  for (let i = 1; i <= n; i++) {
    const t = i / n;
    const s = Math.sin(t * coils * Math.PI * 2) * r * Math.sin(Math.PI * Math.min(1, t * 6)) * Math.sin(Math.PI * Math.min(1, (1 - t) * 6));
    d += ` L ${(x0 + dx * t + nx * s).toFixed(1)} ${(y0 + dy * t + ny * s).toFixed(1)}`;
  }
  return d;
};

/* ---------- 藤蔓：主茎 + 叶 ---------- */
const Vine = ({d, w = 16, leaves = []}: {d: string; w?: number; leaves?: {x: number; y: number; s: number; rot: number}[]}) => (
  <g>
    <path d={d} fill="none" stroke={V.vineDark} strokeWidth={w + 8} strokeLinecap="round" />
    <path d={d} fill="none" stroke={V.vine} strokeWidth={w} strokeLinecap="round" />
    {leaves.map((l, i) => <Leaf key={i} x={l.x} y={l.y} s={l.s} rot={l.rot} />)}
  </g>
);

/* ---------- 扇形细枝：从 (x0,y0) 分 n 枝贴向墙面 x=wallX；dir=-1 墙在左 / +1 墙在右 ---------- */
const TwigSpray = ({x0, y0, wallX, ys, prog, padS = 1, dir = -1}: {
  x0: number; y0: number; wallX: number; ys: number[]; prog: number[]; padS?: number; dir?: number;
}) => (
  <g>
    {ys.map((wy, i) => {
      const k = clamp01(prog[i] ?? 0);
      if (k <= 0) return null;
      const tx = x0 + (wallX - 60 * dir - x0) * k;
      const ty = y0 + (wy - y0) * k + Math.sin(k * Math.PI) * -26;
      const mx = (x0 + tx) / 2, my = (y0 + ty) / 2 - 30 * Math.sin(k * Math.PI);
      return (
        <g key={i}>
          <path d={`M ${x0} ${y0} Q ${mx} ${my} ${tx} ${ty}`} fill="none"
            stroke={V.vine} strokeWidth={8} strokeLinecap="round" />
          {k > 0.55 && <PadOnWall x={wallX - 4 * dir} y={wy} s={padS * clamp01((k - 0.55) / 0.45)} flip={-dir} />}
        </g>
      );
    })}
  </g>
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
      <Backdrop top={V.bg0} bottom={V.bg0} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 右侧竖墙条 */}
        <BrickWall x={1560} y={-20} w={400} h={1120} />
        {/* 主角：一段藤蔓带着卷须+吸盘贴墙 */}
        <g transform={`translate(960,430) scale(${Math.max(0.001, pop)})`}>
          <path d="M -330 240 C -250 140 -300 20 -190 -60 C -110 -120 -130 -190 -60 -240"
            fill="none" stroke={V.vineDark} strokeWidth={22} strokeLinecap="round" />
          <path d="M -330 240 C -250 140 -300 20 -190 -60 C -110 -120 -130 -190 -60 -240"
            fill="none" stroke={V.vine} strokeWidth={14} strokeLinecap="round" />
          <Leaf x={-288} y={150} s={0.9} rot={-24} />
          <Leaf x={-180} y={-40} s={0.85} rot={18} />
          <Leaf x={-96} y={-196} s={0.8} rot={-14} />
          {/* 顶端卷须向右散开贴墙 */}
          <TwigSpray x0={-60} y0={-240} wallX={596} ys={[-296, -236, -176]} prog={[1, 1, 1]} padS={1.15} dir={1} />
        </g>
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 330}}>
          <div style={{color: V.pad, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 26}}>
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

/* ============ S0 一排小圆垫：卷须分细枝→吸盘贴墙→挑材质 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);

  const reachK = clamp01((f - 14) / 70);                        // 卷须伸向墙
  const twigAt = 60;                                            // 细枝开始散开
  const twigK = [0, 1, 2, 3, 4, 5].map(i => clamp01((f - twigAt - i * 12) / 40));
  const cardK = clamp01((f - beat(1) - 16) / 30);               // 材质识别卡
  const card2K = clamp01((f - beat(1) - 130) / 30);
  const WALLX = 470;
  const jx = 640, jy = 470;                                     // 卷须出发点
  const ex = jx + (WALLX + 60 - jx) * reachK;                   // 卷须末端(细枝散点)
  const ey = jy - 40 * reachK;

  return (
    <FadeIn>
      <Backdrop top={V.bg0} bottom={V.bg0} />
      <SceneTag index={0} name={sec.name} accent={V.pad} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左侧整面砖墙 */}
        <BrickWall x={0} y={0} w={470} h={1080} />

        {/* 藤蔓沿墙爬 */}
        <Vine d="M 760 1000 C 710 880 780 780 730 680 C 690 600 700 520 640 470"
          leaves={[{x: 726, y: 860, s: 0.95, rot: -30}, {x: 700, y: 640, s: 0.85, rot: 26}]} />

        {/* 卷须从节点伸向墙 */}
        <path d={`M ${jx} ${jy} Q ${(jx + ex) / 2 + 20} ${(jy + ey) / 2 - 44} ${ex} ${ey}`}
          fill="none" stroke={V.vine} strokeWidth={11} strokeLinecap="round" />
        {/* 末端分出 6 根细枝 + 吸盘 */}
        <TwigSpray x0={ex} y0={ey} wallX={WALLX} ys={[296, 356, 420, 486, 548, 606]}
          prog={twigK} padS={0.92} />

        {/* 触墙涟漪 */}
        {twigK[2] > 0.9 && f < twigAt + 120 && (
          <Ripple x={WALLX + 6} y={420} startFrame={twigAt + 62} color={V.pad} />
        )}

        {/* 吸盘特写放大镜（顶端那枚） */}
        {twigK[0] > 0.9 && (
          <g opacity={clamp01((f - twigAt - 40) / 24)}>
            <circle cx={WALLX + 40} cy={296} r={66} fill="none" stroke={V.cream}
              strokeWidth={4.5} strokeDasharray="10 10" />
            <line x1={WALLX + 90} y1={266} x2={WALLX + 210} y2={196} stroke={V.cream}
              strokeWidth={4} strokeDasharray="10 10" opacity={0.7} />
            <text x={WALLX + 226} y={210} fill={V.cream} fontSize={30} fontFamily={FONT_HEAD} fontWeight={700}>
              Ø 几毫米
            </text>
          </g>
        )}

        {/* 材质识别双卡 */}
        {cardK > 0 && (
          <g opacity={cardK}>
            <rect x={1180} y={196} width={620} height={240} rx={24} fill={V.panel}
              stroke={V.leaf} strokeWidth={5} />
            <BrickWall x={1220} y={236} w={200} h={160} />
            <PadOnWall x={1428} y={316} s={1.1} hi />
            <path d="M 1600 300 L 1636 336 L 1696 268" fill="none" stroke="#7dffa8"
              strokeWidth={14} strokeLinecap="round" strokeLinejoin="round" />
            <text x={1500} y={392} fill={V.cream} fontSize={30} fontFamily={FONT_HEAD} fontWeight={700}>
              砖石树皮 → 开吸
            </text>
          </g>
        )}
        {card2K > 0 && (
          <g opacity={card2K}>
            <rect x={1180} y={472} width={620} height={240} rx={24} fill={V.panel}
              stroke={C.muted} strokeWidth={5} />
            {/* 松软海绵块 */}
            <rect x={1230} y={512} width={180} height={150} rx={26} fill={V.peel} />
            {[0, 1, 2, 3, 4, 5].map(i => (
              <circle key={i} cx={1260 + (i % 3) * 52} cy={544 + Math.floor(i / 3) * 64}
                r={12} fill="#c9b182" />
            ))}
            {/* 吸盘虚影：不长 */}
            <g opacity={0.35}>
              <PadOnWall x={1428} y={586} s={1.1} />
            </g>
            <g stroke={C.coral} strokeWidth={13} strokeLinecap="round">
              <line x1={1608} y1={556} x2={1680} y2={628} />
              <line x1={1680} y1={556} x2={1608} y2={628} />
            </g>
            <text x={1500} y={672} fill={V.cream} fontSize={30} fontFamily={FONT_HEAD} fontWeight={700}>
              松软潮湿 → 不长
            </text>
          </g>
        )}
      </svg>
      <Stamp text="卷须 = 攀爬器官" x={620} y={160} at={beat(0) + 26} color={V.vineDark} size={48} rot={-4} />
      <Chip text="一根分 5~10 根细枝" x={1060} y={110} at={twigAt + 66} accent={V.pad} />
      <Stamp text="顶端顶着小吸盘" x={152} y={760} at={twigAt + 100} color={V.padDeep} size={50} rot={4} />
      <Stamp text="自带材料识别！" x={860} y={780} at={beat(1) + 40} color={C.coral} size={52} rot={-4} />
    </FadeIn>
  );
};

/* ============ S1 不是吸的是粘的：真空≈0，胶+互锁剖面 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);

  const PX = 1260, PY = 640;                                   // 剖面吸盘贴墙点
  const pressK = spring({frame: f - beat(1) - 10, fps, config: {damping: 14, stiffness: 120}});
  const press = Math.max(0, 1 - pressK) * -60;                 // 下压位移
  const glueK = clamp01((f - beat(1) - 60) / 50);              // 胶液渗出
  const lockK = clamp01((f - beat(1) - 170) / 46);             // 互锁齿伸入
  const vacNo = clamp01((f - beat(0) - 80) / 24);              // 真空打叉

  const pits = [880, 1090, 1430, 1640];                        // 墙面微孔 x

  return (
    <FadeIn>
      <Backdrop top={V.bg1} bottom={V.bg1} />
      <SceneTag index={1} name={sec.name} accent={V.glue} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：浴室挂钩误解卡 */}
        <rect x={120} y={180} width={480} height={340} rx={24} fill={V.panel}
          stroke={C.muted} strokeWidth={4} />
        <rect x={170} y={230} width={22} height={190} rx={6} fill={V.tile} />
        {/* 挂钩吸盘 */}
        <path d="M 192 262 C 240 262 268 296 268 330 C 268 364 240 398 192 398 Z" fill="#cfd8e6" />
        <ellipse cx={192} cy={330} rx={9} ry={68} fill="#aab6c8" />
        <path d="M 268 330 C 320 330 320 372 296 392" fill="none" stroke="#cfd8e6"
          strokeWidth={12} strokeLinecap="round" />
        <circle cx={292} cy={396} r={12} fill="none" stroke="#cfd8e6" strokeWidth={10} />
        <text x={330} y={284} fill={V.cream} fontSize={32} fontFamily={FONT_HEAD} fontWeight={700}>
          真空吸力？
        </text>
        {vacNo > 0 && (
          <g opacity={vacNo}>
            <g stroke={C.coral} strokeWidth={14} strokeLinecap="round">
              <line x1={330} y1={300} x2={420} y2={390} />
              <line x1={420} y1={300} x2={330} y2={390} />
            </g>
            <text x={330} y={476} fill={C.coral} fontSize={44} fontFamily={FONT_HEAD} fontWeight={900}>
              ≈ 0%
            </text>
          </g>
        )}
        <text x={120} y={590} fill={C.muted} fontSize={28} fontFamily={FONT_HEAD}>
          不是浴室挂钩那套
        </text>

        {/* 主剖面：墙面 + 吸盘碗 */}
        <g>
          {/* 墙体：带 4 个微孔 */}
          <rect x={760} y={PY} width={1000} height={250} fill={V.wall} />
          <rect x={760} y={PY} width={1000} height={16} fill={V.wallDark} />
          {pits.map(px => (
            <path key={px} d={`M ${px - 34} ${PY + 16} Q ${px} ${PY + 92} ${px + 34} ${PY + 16} Z`}
              fill={V.wallDark} />
          ))}
          {/* 砖纹底线 */}
          <line x1={760} y1={PY + 180} x2={1760} y2={PY + 180} stroke={V.mortar} strokeWidth={8} opacity={0.5} />

          {/* 胶液层：碗底与墙顶之间的薄带 + 灌进孔的胶柱 */}
          {glueK > 0 && (
            <g opacity={glueK}>
              <ellipse cx={PX} cy={PY - 6} rx={40 + 172 * glueK} ry={10 + 8 * glueK} fill={V.glue} />
              {pits.map((px, i) => {
                const k = clamp01(glueK * 1.6 - i * 0.15);
                return k > 0 ? (
                  <path key={i} d={`M ${px - 30 * k} ${PY + 16} Q ${px} ${PY + 16 + 66 * k} ${px + 30 * k} ${PY + 16} Z`}
                    fill={V.glue} />
                ) : null;
              })}
              {/* 两边溢出的胶珠 */}
              <circle cx={PX - 220} cy={PY - 10} r={13 * glueK} fill={V.glueDark} />
              <circle cx={PX + 220} cy={PY - 10} r={13 * glueK} fill={V.glueDark} />
            </g>
          )}

          {/* 吸盘碗（带卷边 + 内腔 + 互锁齿） */}
          <g transform={`translate(${PX},${PY + press})`}>
            <path d="M -222 0 C -232 -96 -160 -182 -50 -200 C -18 -206 18 -206 50 -200
              C 160 -182 232 -96 222 0 Z" fill={V.pad} />
            <path d="M -192 -2 C -186 -76 -130 -146 -48 -163 C -16 -169 16 -169 48 -163
              C 130 -146 186 -76 192 -2 Z" fill={V.padRim} opacity={0.5} />
            {/* 碗口缘加厚 */}
            <rect x={-212} y={-14} width={424} height={14} rx={7} fill={V.pad} />
            <ellipse cx={-206} cy={-4} rx={19} ry={12} fill={V.padDeep} />
            <ellipse cx={206} cy={-4} rx={19} ry={12} fill={V.padDeep} />
            {/* 碗缘微齿 */}
            {[-186, -158, -130, 130, 158, 186].map((tx, i) => (
              <path key={i} d={`M ${tx} -2 l 9 -16 l 9 16 Z`} fill={V.padDeep} />
            ))}
            {/* 互锁指：伸进墙孔 */}
            {pits.map(px => {
              const k = lockK;
              return (
                <rect key={px} x={px - PX - 13} y={-4} width={26} height={70 * k}
                  rx={11} fill={V.padDeep} />
              );
            })}
            {/* 须柄 */}
            <path d="M 0 -200 C 6 -240 -10 -272 -30 -300" fill="none" stroke={V.vine}
              strokeWidth={16} strokeLinecap="round" />
          </g>

          {/* 引线标注 */}
          {glueK > 0.6 && (
            <g opacity={clamp01((f - beat(1) - 110) / 20)}>
              <line x1={PX + 150} y1={PY - 70} x2={PX + 330} y2={PY - 180}
                stroke={V.glue} strokeWidth={4} strokeDasharray="8 9" />
              <text x={PX + 344} y={PY - 172} fill={V.glue} fontSize={32} fontFamily={FONT_HEAD} fontWeight={700}>
                多糖粘液·填满微缝
              </text>
            </g>
          )}
          {lockK > 0.6 && (
            <g opacity={clamp01((f - beat(1) - 215) / 20)}>
              <line x1={PX - 150} y1={PY + 60} x2={PX - 330} y2={PY - 140}
                stroke={V.padRim} strokeWidth={4} strokeDasharray="8 9" />
              <text x={PX - 640} y={PY - 132} fill={V.padRim} fontSize={32} fontFamily={FONT_HEAD} fontWeight={700}>
                边缘突起伸进小孔 → 机械锁
              </text>
            </g>
          )}

          {/* 放大角标 */}
          <rect x={772} y={196} width={200} height={64} rx={32} fill="none" stroke={V.cream} strokeWidth={4} />
          <text x={872} y={240} textAnchor="middle" fill={V.cream} fontSize={30}
            fontFamily={FONT_HEAD}>剖面 ×30</text>
        </g>
      </svg>
      <Stamp text="不是吸的，是粘的" x={140} y={100} at={beat(0) + 30} color={C.coral} size={50} rot={-4} />
      <Stamp text="现场调胶·越放越牢" x={640} y={80} at={beat(1) + 120} color={V.glueDark} size={48} rot={3} />
      <Chip text="胶 + 倒钩 = 活贴纸" x={1420} y={780} at={beat(1) + 240} accent={V.glue} />
    </FadeIn>
  );
};

/* ============ S2 越拽越紧：拉→卷须弹簧收紧→藤蔓贴墙→抗拉力↑ ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);

  const WALLX = 380;
  const pullF = beat(0) + 110;                                 // 开始拽
  const pullK = spring({frame: f - pullF, fps, config: {damping: 13, stiffness: 110}});
  const gap = 330 - 185 * clamp01(pullK);                      // 藤蔓离墙距离 330→145
  const stemX = WALLX + gap;
  const coils = 1.4 + 3.0 * clamp01(pullK);                    // 卷须圈数
  const anchors = [300, 480, 660];                             // 墙上三个吸盘锚点
  const arrowK = clamp01((f - pullF - 10) / 30);               // 拉力箭头
  const gaugeK = clamp01((f - beat(1) - 10) / 40);             // 抗拉力计
  const windK = clamp01((f - beat(2) - 10) / 30) * (1 - clamp01((f - beat(2) - 260) / 40));
  const jy = 500;

  return (
    <FadeIn>
      <Backdrop top={V.bg2} bottom={V.bg2} />
      <SceneTag index={2} name={sec.name} accent={V.pad} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左侧竖墙 */}
        <BrickWall x={0} y={0} w={WALLX} h={1080} />

        {/* 墙上三个吸盘锚 */}
        {anchors.map((ay, i) => (
          <PadOnWall key={i} x={WALLX + 4} y={ay} s={1.05} hi={pullK > 0.5} />
        ))}

        {/* 藤蔓组：离墙 gap，被拉后贴近 */}
        <g transform={`translate(${stemX - 560},0)`}>
          <Vine d="M 560 980 C 540 830 590 700 560 560 C 545 480 555 380 565 260"
            leaves={[{x: 585, y: 800, s: 0.9, rot: 34}, {x: 540, y: 620, s: 0.8, rot: -28}, {x: 575, y: 380, s: 0.75, rot: 20}]} />
        </g>

        {/* 三根弹簧卷须：墙上锚点 → 藤蔓节点，斜向扇形拉开 */}
        {anchors.map((ay, i) => {
          const vx = stemX + 6;
          const vy = jy + (i - 1) * 95;
          return (
            <path key={i} d={coilPath(WALLX + 56, ay, vx, vy, coils, 20)}
              fill="none" stroke={i === 1 ? V.glue : V.vine} strokeWidth={i === 1 ? 11 : 9}
              strokeLinecap="round" />
          );
        })}

        {/* 拉力箭头：向右拽 */}
        {arrowK > 0 && (
          <g opacity={arrowK}>
            <line x1={stemX + 40} y1={jy + 10} x2={stemX + 40 + 300 * arrowK} y2={jy + 10}
              stroke={C.coral} strokeWidth={18} strokeLinecap="round" />
            <path d={`M ${stemX + 40 + 300 * arrowK} ${jy - 26} L ${stemX + 100 + 300 * arrowK} ${jy + 10} L ${stemX + 40 + 300 * arrowK} ${jy + 46} Z`}
              fill={C.coral} />
            <text x={stemX + 170 + 300 * arrowK} y={jy + 26} fill={C.coral} fontSize={40}
              fontFamily={FONT_HEAD} fontWeight={900}>拽！</text>
          </g>
        )}

        {/* 抗拉力仪表 */}
        {gaugeK > 0 && (
          <g opacity={gaugeK}>
            <rect x={1420} y={190} width={400} height={330} rx={24} fill={V.panel}
              stroke={V.glue} strokeWidth={5} />
            <text x={1620} y={246} textAnchor="middle" fill={V.cream} fontSize={32}
              fontFamily={FONT_HEAD} letterSpacing={4}>抗拉力</text>
            <rect x={1470} y={286} width={70} height={190} rx={12} fill="#241d14" />
            <rect x={1470} y={286 + 190 * (1 - clamp01(gaugeK * 0.55))} width={70}
              height={190 * clamp01(gaugeK * 0.55)} rx={12} fill={V.glue} />
            <rect x={1600} y={286} width={70} height={190} rx={12} fill="#241d14" />
            <rect x={1600} y={286 + 190 * (1 - clamp01(gaugeK * 0.8))} width={70}
              height={190 * clamp01(gaugeK * 0.8)} rx={12} fill={V.glue} />
            <rect x={1730} y={286} width={70} height={190} rx={12} fill="#241d14" />
            <rect x={1730} y={286 + 190 * (1 - gaugeK)} width={70}
              height={190 * gaugeK} rx={12} fill={C.coral} />
            <text x={1505} y={512} textAnchor="middle" fill={C.muted} fontSize={22} fontFamily={FONT_HEAD}>刚粘</text>
            <text x={1635} y={512} textAnchor="middle" fill={C.muted} fontSize={22} fontFamily={FONT_HEAD}>拉过</text>
            <text x={1765} y={512} textAnchor="middle" fill={C.muted} fontSize={22} fontFamily={FONT_HEAD}>老须</text>
          </g>
        )}

        {/* 大风来袭：曲线风 + 叶动 */}
        {windK > 0 && (
          <g opacity={windK} stroke="#8fd0ff" strokeWidth={9} strokeLinecap="round" fill="none">
            {[0, 1, 2].map(i => (
              <path key={i}
                d={`M ${-40} ${300 + i * 140} Q ${300 + ((f * 14 + i * 90) % 200)} ${270 + i * 140} ${620} ${300 + i * 140}`}
                strokeDasharray="130 80" strokeDashoffset={-(f * 16 + i * 70)} opacity={0.85} />
            ))}
            <text x={120} y={210} fill="#8fd0ff" fontSize={36} fontFamily={FONT_HEAD}
              fontWeight={900} stroke="none">大风 →</text>
          </g>
        )}

        {/* 距离标注 */}
        {pullK > 0.6 && (
          <g opacity={clamp01((f - pullF - 80) / 24)}>
            <line x1={WALLX + 14} y1={810} x2={stemX - 20} y2={810} stroke={V.cream}
              strokeWidth={4} strokeDasharray="8 8" />
            <text x={WALLX + 40} y={852} fill={V.cream} fontSize={28} fontFamily={FONT_HEAD}>
              越拉越贴墙
            </text>
          </g>
        )}
      </svg>
      <Stamp text="拉力 → 加固信号" x={720} y={150} at={pullF + 20} color={C.coral} size={52} rot={-4} />
      <Chip text="卷须像弹簧收紧" x={1180} y={620} at={pullF + 80} accent={V.glue} />
      <Stamp text="越拽扒得越紧！" x={1150} y={750} at={beat(1) + 50} color={V.padDeep} size={54} rot={4} />
      <Stamp text="风越大贴越死" x={160} y={640} at={beat(2) + 40} color={V.vineDark} size={48} rot={-4} />
    </FadeIn>
  );
};

/* ============ S3 往哪儿爬：背光生长→暗=墙→触到才固定 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);

  const growK = clamp01((f - 20) / 200);                       // 主须向暗生长
  const touchF = 20 + 200;                                     // 触墙时刻
  const fixK = clamp01((f - beat(1) - 30) / 40);               // 固定：吸盘亮起
  const exploreK = clamp01((f - beat(1) - 90) / 120);          // 没碰到的继续探索
  const DARKX = 1180;

  // 主须路径（向左上暗墙弯曲）
  const mainD = "M 250 900 C 460 830 640 780 820 700 C 950 645 1060 590 1176 545";
  // 探索须（亮区打转）
  const explD = "M 620 800 C 700 720 780 700 830 640 C 870 592 850 540 810 540 C 770 540 770 600 815 596";

  return (
    <FadeIn>
      <Backdrop top={V.bg3} bottom={V.bg3} />
      <SceneTag index={3} name={sec.name} accent={'#8fd0ff'} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 右：暗色墙块（最暗的立体结构） */}
        <g>
          <BrickWall x={DARKX} y={0} w={740} h={1080} brick={V.shadowBrick} mortar={V.shadow} dark />
          <rect x={DARKX - 12} y={0} width={12} height={1080} fill={V.shadow} />
          {fixK > 0.4 && (
            <text x={1540} y={120} textAnchor="middle" fill={C.muted} fontSize={30}
              fontFamily={FONT_HEAD} opacity={0.9}>暗 = 墙面·支撑</text>
          )}
        </g>

        {/* 左上太阳：亮区 */}
        <Sun x={270} y={190} r={66} />
        <g stroke="#ffe9a8" strokeWidth={5} opacity={0.4} strokeLinecap="round">
          <line x1={330} y1={250} x2={430} y2={330} />
          <line x1={350} y1={200} x2={480} y2={240} />
        </g>
        <text x={290} y={310} fill="#ffe9a8" fontSize={28} fontFamily={FONT_HEAD} opacity={0.85}>向阳亮区</text>

        {/* 地面 */}
        <rect x={0} y={920} width={DARKX} height={160} fill="#181f30" />

        {/* 藤蔓主干 */}
        <Vine d="M 240 950 C 250 880 260 830 290 780"
          leaves={[{x: 280, y: 830, s: 0.8, rot: -24}]} />

        {/* 主卷须：背光生长伸向暗墙 */}
        <path d={mainD} fill="none" stroke={V.vine} strokeWidth={11} strokeLinecap="round"
          pathLength={100} strokeDasharray={`${growK * 100} 200`} />

        {/* 触墙涟漪 + 固定吸盘群 */}
        {growK >= 1 && fixK < 1 && f < touchF + 60 && (
          <Ripple x={1178} y={543} startFrame={touchF} color={V.pad} />
        )}
        {fixK > 0 && (
          <g opacity={fixK}>
            <TwigSpray x0={1176} y0={545} wallX={DARKX} ys={[470, 528, 590]}
              prog={[fixK, clamp01(fixK * 1.3 - 0.2), clamp01(fixK * 1.3 - 0.4)]} padS={0.95} dir={1} />
          </g>
        )}

        {/* 向暗箭头提示 */}
        {growK > 0.15 && growK < 0.95 && (
          <g opacity={0.9}>
            <path d={`M ${960} ${430} Q 1080 400 1150 470`} fill="none" stroke={V.cream}
              strokeWidth={5} strokeDasharray="12 12" markerEnd="none" />
            <path d="M 1140 442 L 1170 480 L 1126 474 Z" fill={V.cream} />
            <text x={960} y={390} fill={V.cream} fontSize={30} fontFamily={FONT_HEAD} fontWeight={700}>
              背着光往暗处长
            </text>
          </g>
        )}

        {/* 探索须：没摸到东西的继续打转伸长 */}
        {exploreK > 0 && (
          <g>
            <path d={explD} fill="none" stroke={V.vine} strokeWidth={9} strokeLinecap="round"
              pathLength={100} strokeDasharray={`${exploreK * 100} 200`} opacity={0.95} />
            {exploreK > 0.85 && (
              <g opacity={clamp01((exploreK - 0.85) / 0.15)}>
                <circle cx={815} cy={596} r={14} fill={V.padDeep} />
                <text x={740} y={500} fill={C.muted} fontSize={27} fontFamily={FONT_HEAD}>
                  没摸到 → 继续探
                </text>
              </g>
            )}
          </g>
        )}

        {/* 固定完成标签 */}
        {fixK > 0.7 && (
          <g opacity={clamp01((fixK - 0.7) / 0.3)}>
            <text x={1420} y={660} fill={V.padRim} fontSize={32} fontFamily={FONT_HEAD} fontWeight={700}>
              碰到 → 卷曲 + 长吸盘
            </text>
          </g>
        )}
      </svg>
      <Stamp text="不是瞎爬！" x={160} y={340} at={beat(0) + 24} color={C.coral} size={54} rot={-5} />
      <Chip text="向暗性 = 免费导航" x={560} y={170} at={beat(0) + 110} accent="#8fd0ff" />
      <Stamp text="摸到才固定" x={1380} y={740} at={fixK > 0 ? touchF + 60 : beat(1) + 30} color={V.padDeep} size={50} rot={4} />
    </FadeIn>
  );
};

/* ============ S4 力气大但挑墙：百倍拉力 → 好墙✓ 坏墙⚠ ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);

  const numK = clamp01((f - beat(0) - 30) / 80);               // 计数 0→300
  const pullN = Math.round(numK * 300);
  const gridK = clamp01((f - beat(0) - 150) / 50);             // 吸盘阵列
  const wallK = clamp01((f - beat(1) - 20) / 34);              // 好墙板
  const badK = clamp01((f - beat(1) - 140) / 34);              // 坏墙板

  return (
    <FadeIn>
      <Backdrop top={V.bg4} bottom={V.bg4} />
      <SceneTag index={4} name={sec.name} accent={V.pad} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：数据卡 */}
        <rect x={120} y={170} width={610} height={330} rx={26} fill={V.panel}
          stroke={V.pad} strokeWidth={5} />
        <PadDisc x={240} y={330} r={66} />
        <text x={340} y={300} fill={V.cream} fontSize={32} fontFamily={FONT_HEAD}>
          单个吸盘能挂
        </text>
        <text x={340} y={400} fill={V.pad} fontSize={92} fontFamily={FONT_HEAD} fontWeight={900}>
          ×{pullN}
        </text>
        <text x={340} y={458} fill={C.muted} fontSize={28} fontFamily={FONT_HEAD}>
          自身体积倍重的拉力
        </text>

        {/* 左下：吸盘阵列 = 一墙几万个 */}
        {gridK > 0 && (
          <g opacity={gridK}>
            <rect x={120} y={540} width={610} height={290} rx={26} fill={V.panel}
              stroke={V.vineDark} strokeWidth={5} />
            {Array.from({length: 9}).map((_, r) =>
              Array.from({length: 12}).map((_, c) => {
                const k = clamp01(gridK * 2.4 - (r * 12 + c) * 0.016);
                return (
                  <circle key={`${r}-${c}`} cx={172 + c * 44} cy={596 + r * 24}
                    r={9 * k} fill={r === 4 && c === 5 ? V.padRim : V.pad} />
                );
              })
            )}
            <text x={425} y={806} textAnchor="middle" fill={V.cream} fontSize={28}
              fontFamily={FONT_HEAD}>一墙几万个 · 共同受力</text>
          </g>
        )}

        {/* 右上：好墙板（瓷砖） */}
        {wallK > 0 && (
          <g opacity={wallK}>
            <rect x={800} y={170} width={1000} height={300} rx={26} fill={V.panel}
              stroke={V.tile} strokeWidth={5} />
            <g transform="translate(840,210)">
              <rect width={400} height={220} rx={14} fill={V.tile} />
              {[1, 2, 3].map(i => (
                <line key={i} x1={i * 100} y1={0} x2={i * 100} y2={220} stroke={V.tileLine} strokeWidth={6} />
              ))}
              <line x1={0} y1={110} x2={400} y2={110} stroke={V.tileLine} strokeWidth={6} />
              {/* 吸盘贴瓷砖右缘（口朝右） */}
              <PadOnWall x={402} y={110} s={0.9} hi flip={-1} />
            </g>
            <path d="M 1330 290 L 1390 350 L 1500 236" fill="none" stroke="#7dffa8"
              strokeWidth={18} strokeLinecap="round" strokeLinejoin="round" />
            <text x={1560} y={290} fill={V.cream} fontSize={36} fontFamily={FONT_HEAD} fontWeight={700}>
              瓷砖·水泥·石材
            </text>
            <text x={1560} y={352} fill={C.muted} fontSize={30} fontFamily={FONT_HEAD}>
              表面粘附 · 基本不伤
            </text>
          </g>
        )}

        {/* 右下：坏墙板（老粉刷+裂缝） */}
        {badK > 0 && (
          <g opacity={badK}>
            <rect x={800} y={520} width={1000} height={300} rx={26} fill={V.panel}
              stroke={C.coral} strokeWidth={5} />
            <g transform="translate(840,560)">
              <rect width={400} height={220} rx={14} fill={V.peel} />
              {/* 剥落块 */}
              <path d="M 60 60 L 130 30 L 150 90 L 90 120 Z" fill="#c9b182" />
              <path d="M 250 150 L 320 130 L 340 190 L 270 200 Z" fill="#c9b182" />
              {/* 裂缝 */}
              <path d="M 180 220 L 200 160 L 190 120 L 220 70 L 210 20"
                fill="none" stroke={V.crack} strokeWidth={8} strokeLinecap="round" />
              {/* 卷须钻进缝 */}
              <path d="M 440 100 C 340 96 260 88 212 70"
                fill="none" stroke={V.vine} strokeWidth={9} strokeLinecap="round" />
              <ellipse cx={206} cy={68} rx={16} ry={10} fill={V.padDeep} transform="rotate(-24 206 68)" />
            </g>
            {/* 警示三角 */}
            <path d="M 1350 620 L 1410 730 L 1290 730 Z" fill={C.coral} />
            <text x={1350} y={712} textAnchor="middle" fill="#fff" fontSize={52}
              fontFamily={FONT_HEAD} fontWeight={900}>!</text>
            <text x={1480} y={660} fill={V.cream} fontSize={36} fontFamily={FONT_HEAD} fontWeight={700}>
              老粉刷·木框·裂缝墙
            </text>
            <text x={1480} y={722} fill={C.muted} fontSize={30} fontFamily={FONT_HEAD}>
              硬拽连皮撕下 · 钻缝帮倒忙
            </text>
          </g>
        )}
      </svg>
      <Stamp text="几毫米 · 几百倍" x={180} y={105} at={beat(0) + 40} color={V.padDeep} size={50} rot={-4} />
      <Stamp text="好墙扒得漂亮" x={880} y={90} at={beat(1) + 40} color={V.tileLine} size={48} rot={3} />
      <Stamp text="坏墙它帮倒忙" x={1240} y={790} at={beat(1) + 170} color={C.coral} size={48} rot={-3} />
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
      <Backdrop top={V.bg0} bottom={V.bg0} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 底部砖墙 + 爬满的藤蔓 */}
        <BrickWall x={0} y={860} w={1920} h={220} />
        <g transform={`translate(960,830) scale(${Math.max(0.001, pop)})`}>
          <path d="M -420 60 C -360 -30 -420 -110 -330 -170 C -260 -216 -290 -290 -210 -330"
            fill="none" stroke={V.vineDark} strokeWidth={18} strokeLinecap="round" />
          <path d="M -420 60 C -360 -30 -420 -110 -330 -170 C -260 -216 -290 -290 -210 -330"
            fill="none" stroke={V.vine} strokeWidth={11} strokeLinecap="round" />
          <Leaf x={-380} y={-20} s={0.75} rot={-26} />
          <Leaf x={-310} y={-180} s={0.7} rot={22} />
          <Leaf x={-225} y={-290} s={0.62} rot={-16} />
          {/* 墙面上密布的吸盘（贴砖正面） */}
          {[0, 1, 2, 3, 4, 5].map(i => (
            <PadDisc key={i} x={-160 + i * 96} y={118 + (i % 2) * 46} r={22} />
          ))}
        </g>
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -340}}>
        <div style={{fontSize: 66, fontWeight: 800, color: C.text}}>胶水加倒钩 · 越拉越牢</div>
        <div style={{fontSize: 38, color: C.muted, marginTop: 22, letterSpacing: 6}}>柔性的胜利 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E219 = ({data = tl}: {data?: any}) => {
  const {fps} = useVideoConfig();
  return (
    <AbsoluteFill style={{background: V.bg0}}>
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
