import {AbsoluteFill, Audio, Sequence, Easing, interpolate, interpolateColors, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01, rand} from '../lib/style';
import {Backdrop, Ripple, Stamp, SceneTag, Chip, Subtitles, ProgressBar, Stars, FadeIn} from '../lib/atoms';
import tl from '../timeline_204.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};
type TL = {title: string; episode: string; titleCardSec: number; endCardSec: number; totalSec: number; sections: Sec[]};

/* ============ 本期色板：竹青系 ============ */
const B = {
  bgTop: '#0d2f27', bgBot: '#061820',
  culm: '#9fd689', culmDark: '#4d9159', culmDeep: '#2c5f3e',
  leaf: '#7fc97a', leafDark: '#4a8f55',
  aqua: '#55e6c0', aquaDim: '#2b9c84',
  soil0: '#3c2c1d', soil1: '#221812', turf: '#2e5c3a',
  shell: '#c98d54', shellDark: '#8a5a33', shootTip: '#b7e58d',
  flower: '#e8c46a', dead: '#8d9a92', deadDark: '#55645c',
  pandaW: '#f4f6fa', pandaB: '#20242e',
  ring: '#8a6a45', bark: '#6b4a2e',
};

/* ============ 竹叶：梭形叶片 ============ */
const LeafBlade = ({x = 0, y = 0, rot = 0, len = 95, w = 15, k = 1, tint = B.leaf}: {
  x?: number; y?: number; rot?: number; len?: number; w?: number; k?: number; tint?: string;
}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${k})`}>
    <path d={`M0 0 Q ${len * 0.5} ${-w * 1.15} ${len} 0 Q ${len * 0.5} ${w * 0.8} 0 0 Z`} fill={tint} />
    <line x1={8} y1={0} x2={len * 0.88} y2={0} stroke={B.culmDeep} strokeWidth={1.6} opacity={0.45} />
  </g>
);

/* ============ 竹秆：带节空心秆 + 顶部叶簇 ============ */
const BambooCulm = ({x, y, h = 420, w = 30, segs = 6, grow = 1, lean = 0, leafK = 1,
  tint = B.culm, dark = B.culmDark, sway = 0, seed = 0, dots = false, dotOn = 0}: {
  x: number; y: number; h?: number; w?: number; segs?: number; grow?: number; lean?: number;
  leafK?: number; tint?: string; dark?: string; sway?: number; seed?: number;
  dots?: boolean; dotOn?: number;
}) => {
  const f = useCurrentFrame();
  const H = h * clamp01(grow);
  if (H < 4) return null;
  const segH = H / segs;
  const swy = sway ? Math.sin(f * 0.045 + seed * 2.3) * sway : 0;
  const topW = w * (1 - 0.32);
  return (
    <g transform={`translate(${x},${y}) rotate(${lean + swy})`}>
      {Array.from({length: segs}).map((_, i) => {
        const wi = w * (1 - (i / segs) * 0.32);      // 微微上细
        const yb = -i * segH;                        // 本节底
        return (
          <g key={i}>
            <rect x={-wi / 2} y={yb - segH + 2} width={wi} height={segH - 2} rx={wi * 0.16} fill={tint} />
            <rect x={-wi / 2 + wi * 0.16} y={yb - segH + 6} width={wi * 0.14} height={segH - 10} rx={wi * 0.07}
              fill="#ffffff" opacity={0.14} />
            {/* 节环 */}
            <rect x={-wi * 0.62} y={yb - 4.5} width={wi * 1.24} height={7.5} rx={3.5} fill={dark} />
            <rect x={-wi * 0.62} y={yb - 4.5} width={wi * 1.24} height={3} rx={1.5} fill="#ffffff" opacity={0.18} />
            {/* 节间拉长时的细胞亮点 */}
            {dots && (
              <circle cx={0} cy={yb - segH / 2} r={5}
                fill={B.aqua} opacity={dotOn * (0.45 + 0.45 * Math.sin(f * 0.4 + i * 1.3))} />
            )}
          </g>
        );
      })}
      {/* 顶端：空心截面 */}
      <ellipse cx={0} cy={-H} rx={topW * 0.5} ry={topW * 0.22} fill={dark} />
      <ellipse cx={0} cy={-H - 1.5} rx={topW * 0.34} ry={topW * 0.14} fill={B.bgBot} />
      {/* 顶部叶簇 */}
      <g transform={`translate(0,${-H})`}>
        <LeafBlade rot={-32} k={leafK} len={w * 3.4} />
        <LeafBlade rot={-78} k={leafK} len={w * 2.9} tint={B.leafDark} />
        <LeafBlade rot={-140} k={leafK} len={w * 3.1} />
      </g>
      <g transform={`translate(0,${-H * 0.82})`}>
        <LeafBlade rot={-18} k={leafK * 0.8} len={w * 2.4} tint={B.leafDark} />
        <LeafBlade rot={-160} k={leafK * 0.8} len={w * 2.4} />
      </g>
    </g>
  );
};

/* ============ 竹笋：锥形 + 壳片 ============ */
const BambooShoot = ({x, y, h = 170, pop = 1, tint = B.shell}: {
  x: number; y: number; h?: number; pop?: number; tint?: string;
}) => {
  const p = clamp01(pop);
  if (p <= 0.02) return null;
  const lift = (1 - p) * h * 0.85;                  // 没出土的藏进土里
  const w = h * 0.42;
  const squeeze = 0.6 + p * 0.4;                     // 出土过程稍微拔长
  return (
    <g transform={`translate(${x},${y + lift}) scale(1,${squeeze})`} opacity={p < 0.12 ? p * 8 : 1}>
      <path d={`M ${-w / 2} 0 Q ${-w * 0.34} ${-h * 0.55} 0 ${-h} Q ${w * 0.34} ${-h * 0.55} ${w / 2} 0 Z`} fill={tint} />
      {/* 壳片弧纹 */}
      {[0.3, 0.52, 0.72].map((t, i) => (
        <path key={i} d={`M ${-w * 0.46 * (1 - t)} ${-h * t} Q 0 ${-h * t + h * 0.1} ${w * 0.46 * (1 - t)} ${-h * t}`}
          stroke={B.shellDark} strokeWidth={5} fill="none" strokeLinecap="round" opacity={0.75} />
      ))}
      {/* 笋尖嫩芽 */}
      <ellipse cx={0} cy={-h - 3} rx={7} ry={13} fill={B.shootTip} />
    </g>
  );
};

/* ============ 出土尘土小爆点 ============ */
const DirtPuff = ({x, y, startFrame}: {x: number; y: number; startFrame: number}) => {
  const f = useCurrentFrame();
  const t = f - startFrame;
  if (t < 0 || t > 22) return null;
  return (
    <g>
      {[0, 1, 2, 3].map((i) => {
        const p = t / 22;
        const dx = (i - 1.5) * 26 * p;
        return <circle key={i} cx={x + dx} cy={y - Math.sin(p * Math.PI) * (26 + i * 8)}
          r={7 - p * 4} fill={B.soil0} opacity={0.9 * (1 - p)} />;
      })}
    </g>
  );
};

/* ============ 标题卡 ============ */
const TitleCard = ({data}: {data: TL}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 13, mass: 0.7}});
  const o = interpolate(f, [0, 8], [0, 1]);
  const grow = clamp01(f / 26);
  return (
    <AbsoluteFill>
      <Backdrop top={B.bgTop} bottom={B.bgBot} glow="#55e6c014" />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <g transform="translate(960,420) scale(0.55)">
          <BambooCulm x={-150} y={0} h={300} w={30} segs={5} grow={grow} sway={1.2} leafK={grow} />
          <BambooCulm x={-40} y={0} h={380} w={34} segs={6} grow={clamp01(grow * 1.15)} sway={1.6} seed={2}
            tint="#8ecf7d" leafK={grow} />
          <BambooShoot x={90} y={4} h={130} pop={clamp01((f - 12) / 18)} />
          <BambooShoot x={180} y={6} h={95} pop={clamp01((f - 22) / 18)} tint={B.shellDark} />
        </g>
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 180}}>
          <div style={{color: B.aqua, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 26}}>
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

/* ============ S0 开场：一夜冒一片笋 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round(((sec.cues[i]?.t0 ?? 0) - sec.start) * fps);
  const GY = 830;

  // 7 根笋错落弹出
  const shoots = [0, 1, 2, 3, 4, 5, 6].map((i) => {
    const x = 300 + i * 210 + rand(i, 5) * 60;
    const h = 120 + rand(i, 6) * 130;
    const at = 18 + i * 14 + Math.round(rand(i, 7) * 10);
    const p = spring({frame: f - at, fps, config: {damping: 12, mass: 0.7, stiffness: 170}});
    return {x, h, at, p, tint: i % 2 ? B.shell : '#b87f49'};
  });

  const glow = interpolate(f, [beat(2), beat(2) + 40], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  return (
    <FadeIn>
      <Backdrop top="#123830" bottom={B.bgBot} glow="#55e6c012" />
      <SceneTag index={0} name={sec.name} accent={B.aqua} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={16} from={0} dur={30} />
        {/* 地下微光（cue2 提示往土里看） */}
        <ellipse cx={960} cy={GY + 190} rx={1000} ry={190} fill={B.aqua} opacity={glow * 0.1} />
        {glow > 0 && Array.from({length: 9}).map((_, i) => (
          <circle key={i} cx={320 + i * 165} cy={GY + 150 + Math.sin(f * 0.08 + i * 2) * 24}
            r={7} fill={B.aqua} opacity={glow * (0.35 + 0.3 * Math.sin(f * 0.2 + i * 1.9))} />
        ))}
        {/* 地面 */}
        <rect x={0} y={GY} width={1920} height={250} fill={B.soil1} />
        <rect x={0} y={GY - 16} width={1920} height={20} rx={8} fill={B.turf} />
        {/* 老竹剪影两根 */}
        <BambooCulm x={140} y={GY - 10} h={560} w={40} segs={8} tint="#1d4a3a" dark="#12332a" leafK={0.9} sway={0.8} />
        <BambooCulm x={1790} y={GY - 10} h={500} w={36} segs={7} tint="#1d4a3a" dark="#12332a" leafK={0.9} sway={0.8} seed={3} />
        {/* 笋弹出 */}
        {shoots.map((s, i) => (
          <g key={i}>
            <BambooShoot x={s.x} y={GY + 4} h={s.h} pop={s.p} tint={s.tint} />
            <DirtPuff x={s.x} y={GY + 6} startFrame={s.at} />
          </g>
        ))}
      </svg>
      <Stamp text="一夜一片！" x={1480} y={230} at={36} color={B.aquaDim} size={58} rot={6} />
      <Stamp text="撒的种子？" x={200} y={250} at={beat(1)} color="#3d6a8f" size={52} rot={-5} />
      <Chip text="不是种子，往土里看" x={690} y={160} at={beat(2) + 4} accent={B.aqua} />
    </FadeIn>
  );
};

/* ============ S1 联网：地下剖面·竹鞭网络 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round(((sec.cues[i]?.t0 ?? 0) - sec.start) * fps);
  const GY = 560;

  // 主鞭爬行进度
  const crawl = clamp01(interpolate(f, [beat(1), beat(1) + 150], [0, 1], {
    extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.out(Easing.cubic),
  }));
  const branch = clamp01(interpolate(f, [beat(1) + 80, beat(1) + 150], [0, 1], {
    extrapolateLeft: 'clamp', extrapolateRight: 'clamp',
  }));

  const whip = 'M 330 745 C 470 705 570 790 700 750 S 900 705 1010 765 S 1230 815 1340 755 S 1560 705 1680 765';
  const whip2 = 'M 1010 765 C 1090 830 1180 850 1290 880';

  // 鞭上节节点（手摆对齐 path 走向），hit=鞭爬到该点
  const nodes = [
    {x: 480, y: 722, t: 0.13},
    {x: 700, y: 750, t: 0.30},
    {x: 1010, y: 765, t: 0.55},
    {x: 1340, y: 755, t: 0.80},
    {x: 1620, y: 745, t: 0.96},
  ];

  const lit = clamp01(interpolate(f, [beat(2) + 10, beat(2) + 60], [0, 1], {
    extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}));

  return (
    <FadeIn>
      <Backdrop top="#0e2a30" bottom={B.bgBot} glow="#55e6c010" />
      <SceneTag index={1} name={sec.name} accent={B.aqua} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <defs>
          <linearGradient id="soilG" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={B.soil0} />
            <stop offset="100%" stopColor={B.soil1} />
          </linearGradient>
        </defs>
        <Stars n={14} from={0} dur={30} />

        {/* ---- 地上 ---- */}
        <BambooCulm x={330} y={GY - 6} h={400} w={34} segs={6} tint="#2a6349" dark="#1a4234" leafK={1} sway={1} />
        {nodes.slice(1, 5).map((n, i) => {
          const born = crawl > n.t;
          const up = born ? spring({frame: f - (beat(1) + Math.round(n.t * 150)), fps,
            config: {damping: 13, mass: 0.7}}) : 0;
          const toCulm = clamp01(interpolate(f, [beat(1) + 170 + i * 18, beat(1) + 230 + i * 18], [0, 1], {
            extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}));
          return (
            <g key={i}>
              {/* 笋出土后拔成细竹 */}
              <BambooCulm x={n.x} y={GY - 4} h={150 + i * 26} w={17} segs={4}
                grow={clamp01(up * 0.35 + toCulm)} leafK={toCulm} tint="#3c7d55" dark="#25573c" />
              <BambooShoot x={n.x} y={GY + 2} h={105 - i * 8} pop={up * (1 - toCulm * 0.4)} />
              <DirtPuff x={n.x} y={GY + 4} startFrame={beat(1) + Math.round(n.t * 150)} />
              <Ripple x={n.x} y={GY} startFrame={beat(1) + Math.round(n.t * 150)} color={B.aqua} />
            </g>
          );
        })}

        {/* ---- 地面 ---- */}
        <rect x={0} y={GY} width={1920} height={520} fill="url(#soilG)" />
        <rect x={0} y={GY - 14} width={1920} height={18} rx={8} fill={B.turf} />
        {/* 土壤颗粒 */}
        {Array.from({length: 40}).map((_, i) => (
          <circle key={i} cx={rand(i, 21) * 1900 + 10} cy={GY + 40 + rand(i, 22) * 440}
            r={rand(i, 23) * 5 + 2} fill="#4a3826" opacity={0.5} />
        ))}

        {/* ---- 地下竹鞭网络 ---- */}
        {/* 岔鞭 */}
        <path d={whip2} fill="none" stroke={B.aquaDim} strokeWidth={22} strokeLinecap="round"
          pathLength={1} strokeDasharray={1} strokeDashoffset={1 - branch} opacity={0.4 + lit * 0.3} />
        <path d={whip2} fill="none" stroke={B.aqua} strokeWidth={7} strokeLinecap="round"
          pathLength={1} strokeDasharray={1} strokeDashoffset={1 - branch} />
        {/* 主鞭：外发光 + 亮芯 */}
        <path d={whip} fill="none" stroke={B.aqua} strokeWidth={26} strokeLinecap="round"
          pathLength={1} strokeDasharray={1} strokeDashoffset={1 - crawl} opacity={0.16 + lit * 0.2} />
        <path d={whip} fill="none" stroke={B.aquaDim} strokeWidth={17} strokeLinecap="round"
          pathLength={1} strokeDasharray={1} strokeDashoffset={1 - crawl} />
        <path d={whip} fill="none" stroke="#c8fff0" strokeWidth={6} strokeLinecap="round"
          pathLength={1} strokeDasharray={1} strokeDashoffset={1 - crawl} opacity={0.85} />
        {/* 爬行前端的光头 */}
        {crawl > 0 && crawl < 1 && (
          <circle cx={330 + (1680 - 330) * crawl} cy={745 - Math.sin(crawl * 9) * 26}
            r={16} fill={B.aqua} opacity={0.9}>
          </circle>
        )}
        {/* 节节点：鞭爬到就点亮 */}
        {nodes.map((n, i) => {
          const on = crawl >= n.t ? 1 : 0;
          const pulse = on ? 0.7 + 0.3 * Math.sin(f * 0.25 + i * 1.6) : 0;
          return (
            <g key={i} opacity={on}>
              <circle cx={n.x} cy={n.y} r={20} fill={B.aqua} opacity={0.2 * pulse} />
              <circle cx={n.x} cy={n.y} r={10} fill={B.aqua} opacity={pulse} />
              <circle cx={n.x} cy={n.y} r={4.5} fill="#e6fff7" />
              {/* 连接笋的细通道 */}
              <line x1={n.x} y1={n.y - 8} x2={n.x} y2={GY} stroke={B.aqua} strokeWidth={4}
                strokeDasharray="4 10" opacity={on * 0.6} />
            </g>
          );
        })}
        {/* 母竹基部根团 */}
        <ellipse cx={330} cy={GY + 30} rx={60} ry={26} fill={B.culmDeep} opacity={0.7} />
        {/* 全网连通闪光 */}
        {lit > 0 && (
          <ellipse cx={990} cy={760} rx={760} ry={130} fill={B.aqua} opacity={lit * 0.07} />
        )}
        <text x={330} y={GY + 140} fill={B.aqua} fontSize={40} fontFamily={FONT_HEAD}
          fontWeight={700} opacity={interpolate(f, [40, 60], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})}>
          竹鞭 · 地下茎
        </text>
      </svg>
      <Chip text="一根鞭，串起整片林" x={1420} y={800} at={beat(1) + 60} accent={B.aqua} />
      <Stamp text="同一株植物！" x={1330} y={180} at={beat(2) + 10} color={B.aquaDim} size={56} rot={5} />
    </FadeIn>
  );
};

/* ============ S2 冲刺：预制细胞柱·一天一米 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round(((sec.cues[i]?.t0 ?? 0) - sec.start) * fps);
  const GY = 960;

  const T0 = beat(1) + 10;                       // 拔高启动帧
  const grow = interpolate(f, [T0, T0 + 44], [0.13, 1], {
    extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.out(Easing.cubic),
  });
  const rush = clamp01(grow * 3) * clamp01((1 - grow) * 6);  // 长高窗口内=1
  const topY = GY - 560 * grow;

  // 标尺在 cue2 出现
  const rulerK = interpolate(f, [beat(2), beat(2) + 16], [0, 1], {
    extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});

  return (
    <FadeIn>
      <Backdrop top="#103028" bottom={B.bgBot} glow="#9fd68914" />
      <SceneTag index={2} name={sec.name} accent={B.culm} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 速度线：只在蹿升时出现 */}
        {rush > 0.02 && Array.from({length: 18}).map((_, i) => {
          const x = 180 + rand(i, 31) * 1560;
          const y = 1150 - ((f * 42 + i * 137) % 1250);
          return <line key={i} x1={x} y1={y} x2={x} y2={y + 90}
            stroke={B.aqua} strokeWidth={4} strokeLinecap="round" opacity={rush * 0.4} />;
        })}

        {/* 地面 */}
        <rect x={0} y={GY} width={1920} height={120} fill={B.soil1} />
        <rect x={0} y={GY - 14} width={1920} height={18} rx={8} fill={B.turf} />

        {/* 左侧：矮笋对比（一动不动） */}
        <BambooShoot x={430} y={GY + 2} h={150} pop={1} tint={B.shellDark} />
        <text x={505} y={GY - 40} textAnchor="start" fill={C.muted} fontSize={30}
          fontFamily={FONT_HEAD}>昨天的它</text>

        {/* 中央：预制秆同步拉长 */}
        <BambooCulm x={900} y={GY - 2} h={560} w={42} segs={9} grow={grow}
          leafK={clamp01((grow - 0.7) * 3)} dots dotOn={rush * 0.9 + 0.25} sway={grow >= 1 ? 0.9 : 0} />
        {/* 底部笋壳残片 */}
        {grow < 0.5 && <BambooShoot x={900} y={GY + 4} h={120} pop={1 - grow} />}

        {/* 顶部冲刺光点 */}
        {rush > 0.05 && (
          <g transform={`translate(900,${topY})`}>
            <circle r={26} fill={B.aqua} opacity={rush * 0.35} />
            <circle r={10} fill="#eafff8" opacity={rush} />
          </g>
        )}

        {/* 高度标尺 */}
        <g opacity={rulerK}>
          <line x1={1420} y1={GY} x2={1420} y2={GY - 560} stroke={C.text} strokeWidth={4} opacity={0.6} />
          {[0, 0.5, 1].map((t, i) => (
            <g key={i}>
              <line x1={1405} y1={GY - 560 * t} x2={1435} y2={GY - 560 * t} stroke={C.text} strokeWidth={4} />
              <text x={1460} y={GY - 560 * t + 10} fill={C.text} fontSize={32}
                fontFamily={FONT_HEAD}>{t}m</text>
            </g>
          ))}
          {/* 跟着竹顶的箭头 */}
          <g transform={`translate(1380,${topY})`}>
            <path d="M 0 -12 L -34 0 L 0 12 Z" fill={B.flower} />
          </g>
        </g>
      </svg>
      <Chip text="节间细胞·同时拉长" x={220} y={170} at={beat(1)} accent={B.aqua} />
      <Stamp text="一天 ≈ 一米！" x={1260} y={210} at={beat(2) + 4} color={B.flower} size={56} rot={-5} />
      <Stamp text="预制好的充气柱" x={150} y={680} at={beat(0) + 20} color="#3d6a8f" size={42} rot={-6} />
    </FadeIn>
  );
};

/* ============ S3 身世：是草不是树 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round(((sec.cues[i]?.t0 ?? 0) - sec.start) * fps);
  const GY = 980;

  const ringsIn = clamp01((f - beat(1)) / 60);     // 年轮长出来
  const chopT = beat(2) + 6;                       // 砍竹演示
  const chopK = interpolate(f, [chopT, chopT + 12], [0, 1], {
    extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.in(Easing.quad)});
  const fallK = interpolate(f, [chopT + 10, chopT + 30], [0, 1], {
    extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const resprout = spring({frame: f - chopT - 34, fps, config: {damping: 12, stiffness: 160}});

  return (
    <FadeIn>
      <Backdrop top="#11302b" bottom={B.bgBot} glow="#e8c46a10" />
      <SceneTag index={3} name={sec.name} accent={B.flower} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 中缝虚线 */}
        <line x1={960} y1={170} x2={960} y2={1020} stroke={C.muted} strokeWidth={3}
          strokeDasharray="6 18" opacity={0.35} />

        {/* ---- 左：树桩年轮 ---- */}
        <g transform="translate(540,600)">
          <circle r={230} fill={B.bark} />
          <circle r={212} fill="#d9b98a" />
          {[0, 1, 2, 3, 4, 5].map((i) => {
            const rr = 34 + i * 30;
            const k = clamp01(ringsIn * 6 - i);
            return <circle key={i} r={rr * (0.6 + k * 0.4)} fill="none"
              stroke={B.ring} strokeWidth={7} opacity={k * 0.85} />;
          })}
          <circle r={12} fill={B.ring} opacity={0.8} />
          {/* 树皮裂纹 */}
          {[0, 1, 2, 3].map((i) => (
            <line key={i} x1={Math.cos(i * 1.9) * 216} y1={Math.sin(i * 1.9) * 216}
              x2={Math.cos(i * 1.9) * 230} y2={Math.sin(i * 1.9) * 230}
              stroke="#4a3018" strokeWidth={5} />
          ))}
        </g>
        <text x={540} y={920} textAnchor="middle" fill={B.flower} fontSize={42}
          fontFamily={FONT_HEAD} fontWeight={700}>树：一年一圈年轮</text>
        <text x={540} y={170} textAnchor="middle" fill={C.muted} fontSize={34}
          fontFamily={FONT_HEAD}>TREE</text>

        {/* ---- 右：竹秆横截面 ---- */}
        <g transform="translate(1370,600)">
          <circle r={230} fill={B.culmDark} />
          <circle r={214} fill={B.culm} />
          <circle r={150} fill={B.bgBot} />
          {/* 维管束一圈 */}
          {Array.from({length: 16}).map((_, i) => {
            const a = (i / 16) * Math.PI * 2;
            return <circle key={i} cx={Math.cos(a) * 186} cy={Math.sin(a) * 186}
              r={9} fill={B.culmDeep} stroke={B.aqua} strokeWidth={2} opacity={0.9} />;
          })}
          <text y={8} textAnchor="middle" fill={B.aqua} fontSize={44}
            fontFamily={FONT_HEAD} fontWeight={700} opacity={0.9}>空</text>
        </g>
        <text x={1370} y={920} textAnchor="middle" fill={B.aqua} fontSize={42}
          fontFamily={FONT_HEAD} fontWeight={700}>竹：中空，没有年轮</text>
        <text x={1370} y={170} textAnchor="middle" fill={C.muted} fontSize={34}
          fontFamily={FONT_HEAD}>GRASS</text>

        {/* ---- 右下：砍不死演示 ---- */}
        <g>
          {/* 竹子（下半截 + 会倒的上半截，绕截断点倒） */}
          <g transform={`translate(1370,${GY})`}>
            <g transform={`translate(0,-80) rotate(${fallK * 78})`} opacity={1 - fallK * 0.85}>
              <BambooCulm x={0} y={0} h={140} w={22} segs={3} leafK={0.6} tint={B.culm} dark={B.culmDark} />
            </g>
            <BambooCulm x={0} y={0} h={80} w={22} segs={1} leafK={0} tint={B.culm} dark={B.culmDark} />
            {/* 新笋从基部弹回 */}
            <BambooShoot x={42} y={4} h={110} pop={resprout} />
          </g>
          {/* 砍刀：要砍的瞬间才入场 */}
          <g transform={`translate(${1370 + 130 - chopK * 118},${GY - 260 + chopK * 150}) rotate(${40 - chopK * 34})`}
            opacity={interpolate(f, [chopT - 8, chopT], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})}>
            <rect x={-10} y={-130} width={20} height={80} rx={8} fill="#7a5230" />
            <path d="M -26 -50 L 26 -50 L 34 20 Q 0 34 -34 20 Z" fill="#cfd8e6" />
            <path d="M -34 20 Q 0 34 34 20 L 30 10 Q 0 22 -30 10 Z" fill="#8fa0b8" />
          </g>
          {chopK > 0.5 && chopK < 1 && (
            <Ripple x={1370} y={GY - 140} startFrame={chopT + 10} color={B.flower} />
          )}
          {/* 地下鞭安然无恙 */}
          <rect x={0} y={GY + 4} width={1920} height={100} fill={B.soil1} />
          <line x1={1180} y1={GY + 50} x2={1560} y2={GY + 50} stroke={B.aqua} strokeWidth={10}
            strokeLinecap="round" opacity={0.7} strokeDasharray="2 20" />
          <text x={1370} y={GY + 90} textAnchor="middle" fill={B.aqua} fontSize={24}
            fontFamily={FONT_HEAD} opacity={0.8}>地下的鞭：毫发无损</text>
        </g>

        {/* 顶部禾本科小穗 */}
        <g transform="translate(960,120)" opacity={interpolate(f, [16, 40], [0, 1],
          {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})}>
          <path d="M 0 60 Q -14 20 0 -30" stroke={B.culm} strokeWidth={6} fill="none" />
          {[0, 1, 2, 3, 4].map((i) => (
            <ellipse key={i} cx={-10 + (i % 2) * 20} cy={-24 + i * 16} rx={9} ry={15}
              fill={B.flower} transform={`rotate(${(i % 2 ? 24 : -24)} ${-10 + (i % 2) * 20} ${-24 + i * 16})`} />
          ))}
        </g>
      </svg>
      <Stamp text="它是草！" x={810} y={240} at={beat(0) + 8} color={B.culmDark} size={60} rot={-5} />
      <Chip text="水稻小麦的亲戚" x={760} y={330} at={beat(0) + 30} accent={B.flower} />
      <Chip text="出土多粗，一辈子多粗" x={1140} y={210} at={beat(1) + 40} accent={B.aqua} />
      <Stamp text="砍不死" x={1500} y={700} at={chopT + 36} color={C.coral} size={48} rot={7} />
    </FadeIn>
  );
};

/* ============ S4 结局：同步开花·熊猫挨饿 ============ */
const Panda = ({x, y, scale = 1}: {x: number; y: number; scale?: number}) => (
  <g transform={`translate(${x},${y}) scale(${scale})`}>
    <ellipse cx={0} cy={-46} rx={95} ry={82} fill={B.pandaW} />
    <circle cx={-58} cy={-142} r={24} fill={B.pandaB} />
    <circle cx={58} cy={-142} r={24} fill={B.pandaB} />
    <circle cx={0} cy={-108} r={68} fill={B.pandaW} />
    <ellipse cx={-27} cy={-116} rx={15} ry={20} fill={B.pandaB} transform="rotate(-18 -27 -116)" />
    <ellipse cx={27} cy={-116} rx={15} ry={20} fill={B.pandaB} transform="rotate(18 27 -116)" />
    <circle cx={-25} cy={-116} r={4.5} fill="#fff" />
    <circle cx={25} cy={-116} r={4.5} fill="#fff" />
    <ellipse cx={0} cy={-94} rx={9} ry={6} fill={B.pandaB} />
    {/* 嘴角下撇 = 难过 */}
    <path d="M -13 -78 Q 0 -88 13 -78" stroke={B.pandaB} strokeWidth={3.5} fill="none" strokeLinecap="round" />
    <ellipse cx={-54} cy={-44} rx={20} ry={36} fill={B.pandaB} transform="rotate(16 -54 -44)" />
    <ellipse cx={54} cy={-44} rx={20} ry={36} fill={B.pandaB} transform="rotate(-16 54 -44)" />
    <ellipse cx={-48} cy={16} rx={32} ry={19} fill={B.pandaB} />
    <ellipse cx={48} cy={16} rx={32} ry={19} fill={B.pandaB} />
  </g>
);

const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round(((sec.cues[i]?.t0 ?? 0) - sec.start) * fps);
  const GY = 950;

  // 六竿竹：开花 → 灰败 → 部分倒伏
  const culms = [240, 520, 800, 1080, 1360, 1640].map((x, i) => ({
    x, h: 380 + rand(i, 41) * 160, w: 26 + rand(i, 42) * 12, seed: i,
    bloomAt: 100 + i * 12,
    fallAt: i === 1 ? 400 : i === 4 ? 430 : -1,
  }));

  const bloom = (i: number) => clamp01((f - culms[i].bloomAt) / 26);
  const grayK = interpolate(f, [beat(1), beat(1) + 85], [0, 1], {
    extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const pandaIn = spring({frame: f - beat(1) - 50, fps, config: {damping: 13}});

  return (
    <FadeIn>
      <Backdrop top="#122a26" bottom={B.bgBot} glow="#e8c46a0e" />
      <SceneTag index={4} name={sec.name} accent={B.flower} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <rect x={0} y={GY} width={1920} height={130} fill={B.soil1} />
        <rect x={0} y={GY - 14} width={1920} height={18} rx={8} fill={B.turf} opacity={1 - grayK * 0.7} />

        {culms.map((c, i) => {
          const culmTint = interpolateColors(grayK, [0, 1], [B.culm, B.dead]);
          const darkTint = interpolateColors(grayK, [0, 1], [B.culmDark, B.deadDark]);
          const fall = c.fallAt > 0 ? interpolate(f, [c.fallAt, c.fallAt + 55], [0, 1],
            {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.in(Easing.quad)}) : 0;
          return (
            <g key={i} transform={`translate(${c.x},${GY}) rotate(${fall * (i === 1 ? -74 : 68)})`}
              opacity={1 - fall * 0.35}>
              <BambooCulm x={0} y={0} h={c.h} w={c.w} segs={6} tint={culmTint} dark={darkTint}
                leafK={1 - grayK * 0.8} sway={grayK > 0.8 ? 0 : 0.9} seed={c.seed} />
              {/* 花穗：顶部一簇 */}
              {bloom(i) > 0 && (
                <g transform={`translate(0,${-c.h + 14})`}>
                  {Array.from({length: 6}).map((_, j) => {
                    const a = -Math.PI / 2 + (j - 2.5) * 0.5;
                    return <ellipse key={j} cx={Math.cos(a) * 30 * bloom(i)} cy={Math.sin(a) * 34 * bloom(i)}
                      rx={7 * bloom(i)} ry={16 * bloom(i)} fill={B.flower}
                      transform={`rotate(${(a * 180) / Math.PI + 90} ${Math.cos(a) * 30 * bloom(i)} ${Math.sin(a) * 34 * bloom(i)})`}
                      opacity={1 - grayK * 0.6} />;
                  })}
                </g>
              )}
            </g>
          );
        })}

        {/* 枯叶飘落 */}
        {grayK > 0.15 && Array.from({length: 14}).map((_, i) => {
          const t = (f * 1.4 + i * 43) % 300;
          const x = 200 + rand(i, 51) * 1500 + Math.sin(f * 0.06 + i) * 30;
          const y = 180 + t * 2.6;
          if (y > GY - 10) return null;
          return <LeafBlade key={i} x={x} y={y} rot={f * 1.8 + i * 40} len={34} w={7}
            tint={interpolateColors(grayK, [0, 1], [B.leaf, '#9a8d6a'])} />;
        })}

        {/* 熊猫：开花后才滑进来挨饿 */}
        <g transform={`translate(${1560 + (1 - pandaIn) * 300},${GY - 6})`}>
          <Panda x={0} y={0} scale={1.05} />
          {/* 气泡：饿 */}
          <g transform="translate(-90,-230)" opacity={clamp01(pandaIn * 1.4 - 0.4)}>
            <rect x={-66} y={-34} width={132} height={62} rx={20} fill="#ffffff" opacity={0.92} />
            <path d="M 30 24 L 52 52 L 8 28 Z" fill="#ffffff" opacity={0.92} />
            <text x={0} y={9} textAnchor="middle" fill={B.pandaB} fontSize={36}
              fontFamily={FONT_HEAD} fontWeight={700}>咕——</text>
          </g>
        </g>
      </svg>
      <Stamp text="约好一起开花" x={1260} y={140} at={110} color={B.flower} size={50} rot={4} />
      <Chip text="开花 = 整片枯死" x={210} y={280} at={beat(1) + 20} accent={B.dead} />
      <Stamp text="一株假装成森林的植物" x={560} y={140} at={beat(2) + 6} color={B.aquaDim} size={48} rot={-3} />
    </FadeIn>
  );
};

/* ============ 结尾卡 ============ */
const EndCard = () => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 14}});
  const grow = clamp01(f / 30);
  return (
    <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
      <Backdrop top={B.bgTop} bottom={B.bgBot} glow="#55e6c014" />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <BambooCulm x={960} y={1080} h={420} w={38} segs={7} grow={grow} sway={1.1} leafK={grow} />
        <BambooShoot x={860} y={1084} h={110} pop={clamp01((f - 10) / 20)} />
        <BambooShoot x={1066} y={1082} h={90} pop={clamp01((f - 18) / 20)} tint={B.shellDark} />
        {/* 地下鞭收尾呼应 */}
        <path d="M 620 1030 Q 960 990 1300 1030" fill="none" stroke={B.aqua} strokeWidth={8}
          strokeLinecap="round" opacity={0.5} strokeDasharray="2 18" />
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -260}}>
        <div style={{fontSize: 66, fontWeight: 800, color: C.text}}>它是一片假装成森林的单株植物</div>
        <div style={{fontSize: 38, color: C.muted, marginTop: 26, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E204 = ({data = tl}: {data?: any}) => {
  const {fps} = useVideoConfig();
  return (
    <AbsoluteFill style={{background: B.bgBot}}>
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

      <Sequence from={Math.round((data.totalSec - data.endCardSec) * fps)}
        durationInFrames={Math.round(data.endCardSec * fps)}>
        <EndCard />
      </Sequence>

      <Subtitles data={data} />
      <ProgressBar />
    </AbsoluteFill>
  );
};
