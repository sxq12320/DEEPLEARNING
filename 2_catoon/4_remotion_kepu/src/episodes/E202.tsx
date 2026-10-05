import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01} from '../lib/style';
import {Backdrop, Ripple, Stamp, SceneTag, Chip, Subtitles, ProgressBar, Sun, Moon, Stars, FadeIn} from '../lib/atoms';
import tl from '../timeline_202.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};
type TL = {title: string; episode: string; titleCardSec: number; endCardSec: number; totalSec: number; sections: Sec[]};

/* ============ 向日葵：原点在根部土面，花盘绕茎顶转 tilt 度（+朝右/西，-朝左/东） ============ */
const STEM_H = 300;
const Sunflower = ({x, y, scale = 1, tilt = 0, sway = true, opacity = 1, petals = 16, headR = 84}: {
  x: number; y: number; scale?: number; tilt?: number; sway?: boolean; opacity?: number;
  petals?: number; headR?: number;
}) => {
  const f = useCurrentFrame();
  const idle = sway ? Math.sin(f * 0.045) * 1.4 : 0;
  const squash = 1 - Math.abs(Math.sin((tilt * Math.PI) / 180)) * 0.26; // 转到侧面时沿朝向压扁
  return (
    <g transform={`translate(${x},${y}) scale(${scale}) rotate(${idle})`} opacity={opacity}>
      {/* 土堆 */}
      <ellipse cx={0} cy={10} rx={150} ry={26} fill="#232a52" />
      <ellipse cx={-18} cy={4} rx={88} ry={13} fill="#2d3566" />
      {/* 茎 */}
      <path d={`M 0 0 Q 20 ${-STEM_H / 2} 4 ${-STEM_H}`} fill="none" stroke={C.stem} strokeWidth={15} strokeLinecap="round" />
      <path d={`M 0 0 Q 20 ${-STEM_H / 2} 4 ${-STEM_H}`} fill="none" stroke={C.stemDark} strokeWidth={6}
        strokeLinecap="round" transform="translate(-3,0)" opacity={0.5} />
      {/* 两片叶 */}
      <g transform="translate(10,-112) rotate(-16)">
        <ellipse cx={60} cy={0} rx={58} ry={23} fill={C.leafDark} />
        <line x1={8} y1={0} x2={112} y2={0} stroke="#2f8f63" strokeWidth={3} />
      </g>
      <g transform="translate(2,-196) rotate(198)">
        <ellipse cx={54} cy={0} rx={50} ry={20} fill={C.leaf} />
        <line x1={6} y1={0} x2={98} y2={0} stroke={C.leafDark} strokeWidth={3} />
      </g>
      {/* 花盘组：绕茎顶旋转 */}
      <g transform={`translate(4,${-STEM_H}) rotate(${tilt})`}>
        <g transform={`scale(1,${squash})`}>
          {Array.from({length: petals}).map((_, i) => (
            <ellipse key={i} cx={0} cy={-(headR + 22)} rx={14} ry={36}
              fill={i % 2 ? C.amber : '#ffb03a'} transform={`rotate(${(i * 360) / petals})`} />
          ))}
          <circle r={headR} fill="#8a5327" />
          <circle r={headR} fill="none" stroke="#5e3a1e" strokeWidth={5} />
          <circle r={headR * 0.74} fill="#6b3d1c" />
          {[0, 1, 2].map((ring) =>
            Array.from({length: 8 + ring * 5}).map((_, i) => {
              const a = (i / (8 + ring * 5)) * Math.PI * 2 + ring * 0.9;
              const rr = 16 + ring * 22;
              return <circle key={`${ring}-${i}`} cx={Math.cos(a) * rr} cy={Math.sin(a) * rr}
                r={4.5} fill="#3d2410" opacity={0.85} />;
            })
          )}
          <circle cx={-headR * 0.36} cy={-headR * 0.38} r={headR * 0.2} fill="#a06a38" opacity={0.9} />
        </g>
      </g>
    </g>
  );
};

/* ============ 蜜蜂 ============ */
const Bee = ({x, y, scale = 1, rot = 0}: {x: number; y: number; scale?: number; rot?: number}) => {
  const f = useCurrentFrame();
  const flap = Math.sin(f * 2.6) * 26;
  return (
    <g transform={`translate(${x},${y}) rotate(${rot}) scale(${scale})`}>
      <ellipse cx={-9} cy={-17} rx={15} ry={9} fill="#dff1ff" opacity={0.85}
        transform={`rotate(${-30 - flap} -9 -14)`} />
      <ellipse cx={9} cy={-19} rx={13} ry={8} fill="#eaf6ff" opacity={0.7}
        transform={`rotate(${20 + flap} 9 -15)`} />
      <ellipse rx={26} ry={18} fill={C.amber} />
      <ellipse cx={-11} rx={5} ry={14.5} fill="#4a3419" />
      <ellipse cx={2} rx={5.5} ry={16.5} fill="#4a3419" />
      <ellipse cx={14} rx={4.5} ry={11} fill="#4a3419" />
      <circle cx={24} cy={-2} r={11} fill="#4a3419" />
      <circle cx={27.5} cy={-5} r={3.2} fill="#fff" />
      <line x1={-25} y1={2} x2={-35} y2={7} stroke="#4a3419" strokeWidth={4} strokeLinecap="round" />
    </g>
  );
};

/* ============ 小时钟（生物钟） ============ */
const Clock = ({x, y, r = 70, spin = 1, accent = C.amber, opacity = 1}: {
  x: number; y: number; r?: number; spin?: number; accent?: string; opacity?: number;
}) => {
  const f = useCurrentFrame();
  return (
    <g transform={`translate(${x},${y})`} opacity={opacity}>
      <circle r={r} fill="#10142e" stroke={accent} strokeWidth={6} />
      {Array.from({length: 12}).map((_, i) => {
        const a = (i * 30 * Math.PI) / 180;
        return <line key={i} x1={Math.cos(a) * r * 0.82} y1={Math.sin(a) * r * 0.82}
          x2={Math.cos(a) * r * 0.92} y2={Math.sin(a) * r * 0.92}
          stroke={C.muted} strokeWidth={3} opacity={0.8} />;
      })}
      <g transform={`rotate(${f * 0.6 * spin})`}>
        <line x1={0} y1={0} x2={0} y2={-r * 0.5} stroke="#fff" strokeWidth={7} strokeLinecap="round" />
      </g>
      <g transform={`rotate(${f * 4 * spin})`}>
        <line x1={0} y1={0} x2={0} y2={-r * 0.72} stroke={accent} strokeWidth={4} strokeLinecap="round" />
      </g>
      <circle r={6} fill={accent} />
    </g>
  );
};

/* 绕花盘中心的朝向圆弧（tilt 角 a0→a1），返回 path d */
const tiltArc = (cx: number, cy: number, R: number, a0: number, a1: number) => {
  const pts: string[] = [];
  const n = 20;
  for (let i = 0; i <= n; i++) {
    const t = a0 + ((a1 - a0) * i) / n;
    const rad = (t * Math.PI) / 180;
    pts.push(`${i === 0 ? 'M' : 'L'} ${(cx + Math.sin(rad) * R).toFixed(1)} ${(cy - Math.cos(rad) * R).toFixed(1)}`);
  }
  return pts.join(' ');
};

/* ============ 标题卡 ============ */
const TitleCard = ({data}: {data: TL}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 13, mass: 0.7}});
  const o = interpolate(f, [0, 8], [0, 1]);
  // 小太阳沿弧线走，花盘图标跟着转
  const p = Math.sin(f * 0.02) * 0.5 + 0.5;
  const dotA = ((-62 + p * 124) * Math.PI) / 180;
  const dotX = 960 + Math.sin(dotA) * 190;
  const dotY = 390 - Math.cos(dotA) * 190;
  const iconTilt = Math.max(-55, Math.min(55, ((dotX - 960) / 620) * 55));
  return (
    <AbsoluteFill>
      <Backdrop top="#2b3a75" bottom={C.bgDeep} glow="#ffc94d16" />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <path d={tiltArc(960, 390, 190, -62, 62)} fill="none" stroke={C.amber}
          strokeWidth={4} strokeDasharray="4 18" strokeLinecap="round" opacity={0.4} />
        <circle cx={dotX} cy={dotY} r={22} fill={C.amber} opacity={0.95} />
        <circle cx={dotX} cy={dotY} r={34} fill={C.amber} opacity={0.2} />
        {/* 图标花盘 */}
        <g transform={`translate(960,390) rotate(${iconTilt})`}>
          {Array.from({length: 16}).map((_, i) => (
            <ellipse key={i} cx={0} cy={-136} rx={19} ry={46} fill={i % 2 ? C.amber : '#ffb03a'}
              transform={`rotate(${i * 22.5})`} />
          ))}
          <circle r={114} fill="#8a5327" />
          <circle r={114} fill="none" stroke="#5e3a1e" strokeWidth={7} />
          <circle r={85} fill="#6b3d1c" />
          {Array.from({length: 22}).map((_, i) => {
            const a = i * 2.4;
            const rr = 12 + Math.sqrt(i) * 19;
            return <circle key={i} cx={Math.cos(a) * rr} cy={Math.sin(a) * rr} r={7} fill="#3d2410" />;
          })}
        </g>
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 300}}>
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

/* ============ S0 开场：太阳走弧线，花盘从东转到西 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number, fb = 5) => Math.round(((sec.cues[i] ? sec.cues[i].t0 : sec.start + fb) - sec.start) * fps);

  const p = interpolate(f, [15, 250], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const sunX = 360 + p * 1200;
  const sink = interpolate(f, [beat(1), beat(1) + 60], [0, 150], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const sunY = 780 - Math.sin(p * Math.PI) * 470 + sink;
  const dusk = interpolate(f, [beat(1), beat(1) + 80], [0, 0.4], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const tilt = Math.max(-70, Math.min(70, ((sunX - 960) / 600) * 70));

  // 弧线虚线轨迹
  let arc = 'M 360 780';
  for (let i = 1; i <= 24; i++) {
    const q = i / 24;
    arc += ` L ${360 + q * 1200} ${780 - Math.sin(q * Math.PI) * 470}`;
  }

  return (
    <FadeIn>
      <Backdrop top="#2f4a94" bottom="#151a40" glow="#ffc94d14" />
      <SceneTag index={0} name={sec.name} accent={C.amber} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <defs>
          <linearGradient id="duskSky" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={C.dusk0} />
            <stop offset="100%" stopColor={C.dusk1} />
          </linearGradient>
        </defs>
        <rect width={1920} height={1080} fill="url(#duskSky)" opacity={dusk * 0.85} />
        <path d={arc} fill="none" stroke={C.amber} strokeWidth={4} strokeDasharray="3 20"
          strokeLinecap="round" opacity={0.35} />
        <Sun x={sunX} y={sunY} opacity={1 - dusk * 0.6} />
        {/* 时刻小标签 */}
        {[
          {t: '早晨', x: 330, y: 850, at: 20},
          {t: '正午', x: 920, y: 260, at: 110},
          {t: '傍晚', x: 1520, y: 850, at: 230},
        ].map((m) => (
          <text key={m.t} x={m.x} y={m.y} fill={C.muted} fontSize={30} fontFamily={FONT_HEAD}
            opacity={clamp01((f - m.at) / 15)}>{m.t}</text>
        ))}
        {/* 地面 */}
        <ellipse cx={960} cy={1030} rx={1200} ry={140} fill="#10142e" />
        {/* 远景两株 + 主角一株，跟着同一个太阳转 */}
        <Sunflower x={510} y={958} scale={0.6} tilt={tilt} opacity={0.55} petals={14} headR={80} />
        <Sunflower x={1440} y={964} scale={0.7} tilt={tilt} opacity={0.62} petals={14} headR={80} />
        <Sunflower x={960} y={978} scale={1.2} tilt={tilt} />
        {/* 朝向箭头：花盘 → 太阳 */}
        <g transform={`translate(965,618) rotate(${tilt})`} opacity={clamp01((f - 40) / 15) * (1 - dusk)}>
          <line x1={0} y1={-135} x2={0} y2={-185} stroke={C.amber} strokeWidth={7} strokeLinecap="round" />
          <path d="M -14 -176 L 0 -206 L 14 -176 Z" fill={C.amber} />
        </g>
      </svg>
      <Chip text="白天 · 从东转到西" x={150} y={180} at={30} accent={C.amber} />
      <Stamp text="落山以后呢？" x={1250} y={330} at={beat(1) + 10} color={C.coral} size={58} rot={-6} />
      <Stamp text="夜里怎么转回去？" x={170} y={640} at={beat(2) + 8} color="#3d4a8f" size={46} rot={4} />
    </FadeIn>
  );
};

/* ============ S1 夜返：日落月升，花盘一整夜慢慢回摆朝东 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number, fb = 5) => Math.round(((sec.cues[i] ? sec.cues[i].t0 : sec.start + fb) - sec.start) * fps);

  const night = interpolate(f, [30, 150], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const sunY = interpolate(f, [10, 90], [700, 1130], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const moonY = interpolate(f, [90, 220], [560, 210], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const moonO = interpolate(f, [80, 140], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});

  const HX = 960, HY = 978, SC = 1.2;
  const headY = HY - STEM_H * SC; // ≈ 618
  // 主回摆：70°（朝西）→ -64°（朝东），一整夜匀速
  const tilt = interpolate(f, [150, 470], [70, -64], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});

  return (
    <FadeIn>
      <Backdrop top={C.dusk0} bottom={C.dusk1} glow="#ffffff10" />
      <SceneTag index={1} name={sec.name} accent={C.amber} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <defs>
          <linearGradient id="nightSky" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={C.night0} />
            <stop offset="100%" stopColor={C.night1} />
          </linearGradient>
        </defs>
        <rect width={1920} height={1080} fill="url(#nightSky)" opacity={night} />
        <Stars n={26} from={55} dur={60} />
        <Sun x={1530} y={sunY} opacity={clamp01(1 - night * 1.4)} />
        <Moon x={430} y={moonY} opacity={moonO} scale={1.1} />
        <ellipse cx={960} cy={1030} rx={1200} ry={140} fill={C.night0} opacity={0.9} />
        {/* 回摆虚线轨道 + 箭头（朝东方向） */}
        <path d={tiltArc(HX, headY, 168, 78, -72)} fill="none" stroke={C.amber} strokeWidth={4}
          strokeDasharray="3 16" strokeLinecap="round" opacity={0.5} />
        {(() => {
          const a = (-72 * Math.PI) / 180;
          const ax = HX + Math.sin(a) * 168, ay = headY - Math.cos(a) * 168;
          return <path d={`M ${ax - 20} ${ay + 8} L ${ax} ${ay - 10} L ${ax - 4} ${ay + 22} Z`} fill={C.amber} opacity={0.8} />;
        })()}
        {/* 残影：一晚上不同时间的朝向 */}
        {[48, 22, -8, -38].map((g, i) => (
          <Sunflower key={i} x={HX} y={HY} scale={SC} tilt={g} sway={false} opacity={0.1} petals={12} />
        ))}
        <Sunflower x={HX} y={HY} scale={SC} tilt={tilt} sway={false} />
        {/* 慢速进度条 */}
        <g transform={`translate(${HX + 220},${headY + 40})`} opacity={clamp01((f - beat(2)) / 15)}>
          <rect x={0} y={0} width={240} height={14} rx={7} fill="#ffffff18" />
          <rect x={0} y={0} width={240 * clamp01((f - beat(2)) / 150)} height={14} rx={7} fill={C.amber} />
          <text x={0} y={-14} fill={C.muted} fontSize={26} fontFamily={FONT_HEAD}>一整夜 · 匀速慢挪</text>
        </g>
        {/* 角落小时钟：时间在流 */}
        <Clock x={1560} y={820} r={72} spin={1.4} opacity={clamp01((f - 100) / 20)} />
      </svg>
      <Stamp text="不是猛回头" x={220} y={250} at={beat(2) + 6} color={C.coral} size={54} rot={-6} />
      <Chip text="赶在日出前站好位" x={200} y={700} at={beat(1) + 10} accent={C.amber} />
    </FadeIn>
  );
};

/* ============ S2 机关：茎两侧差速生长 + 生物钟 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number, fb = 5) => Math.round(((sec.cues[i] ? sec.cues[i].t0 : sec.start + fb) - sec.start) * fps);

  // 弯曲方向：白天东侧长 → 推向西(+)；夜里西侧长 → 推回东(-)
  const bend = interpolate(f, [70, 200, 255, 385], [0, 1, 1, -1], {
    extrapolateLeft: 'clamp', extrapolateRight: 'clamp',
  });
  const tilt = bend * 48;

  const CX = 1330, CY = 545, W = 400, H = 540;
  const colX = {east: CX - 100, west: CX + 100}; // 东在左（推向右/西）
  const eastGrow = interpolate(f, [80, 195], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const westGrow = interpolate(f, [265, 380], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const rows = [400, 505, 610, 715];

  const cellCol = (side: 'east' | 'west', grow: number, accent: string) =>
    rows.map((ry, i) => {
      const gi = clamp01(grow * 1.7 - i * 0.16);
      const hh = 92 * (1 + gi * 0.5);
      return (
        <g key={i} transform={`translate(${colX[side]},${ry})`}>
          <rect x={-60} y={-hh / 2} width={120} height={hh} rx={24}
            fill={gi > 0.05 ? accent : '#8f6a35'} opacity={0.65 + gi * 0.35} />
          <circle r={13} cy={-hh * 0.18} fill="#5e3a1e" opacity={0.7} />
          {/* 向上生长箭头 */}
          {gi > 0.3 && (
            <g opacity={gi}>
              <line x1={0} y1={-hh / 2 - 6} x2={0} y2={-hh / 2 - 44 * gi} stroke="#fff" strokeWidth={6} strokeLinecap="round" />
              <path d={`M -11 ${-hh / 2 - 40 * gi} L 0 ${-hh / 2 - 62 * gi} L 11 ${-hh / 2 - 40 * gi} Z`} fill="#fff" />
            </g>
          )}
        </g>
      );
    });

  return (
    <FadeIn>
      <Backdrop top="#1c2350" bottom={C.bgDeep} glow="#ffc94d12" />
      <SceneTag index={2} name={sec.name} accent={C.amber} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：迷你向日葵，朝向被茎两侧差速推着走 */}
        <Sunflower x={430} y={945} scale={0.78} tilt={tilt} sway={false} />
        <circle cx={438} cy={945 - 150 * 0.78} r={56} fill="none" stroke={C.amber}
          strokeWidth={4} strokeDasharray="10 10" opacity={0.9} />
        <line x1={494} y1={828} x2={CX - W / 2 - 8} y2={CY - 40} stroke={C.amber}
          strokeWidth={4} strokeDasharray="10 10" opacity={0.7} />

        {/* 右：茎的纵切面放大 */}
        <rect x={CX - W / 2} y={CY - H / 2} width={W} height={H} rx={60}
          fill="#101a3e" stroke={C.stem} strokeWidth={6} />
        <rect x={CX - W / 2 - 14} y={CY - H / 2 - 14} width={W + 28} height={H + 28} rx={72}
          fill="none" stroke={C.stem} strokeWidth={3} opacity={0.3} />
        <text x={colX.east} y={CY - H / 2 - 28} textAnchor="middle" fontSize={40} fontFamily={FONT_HEAD}
          fill={eastGrow > 0.05 && eastGrow < 1 ? C.amber : C.muted}
          fontWeight={700}>东侧细胞</text>
        <text x={colX.west} y={CY - H / 2 - 28} textAnchor="middle" fontSize={40} fontFamily={FONT_HEAD}
          fill={westGrow > 0.05 && westGrow < 1 ? '#5cc8ff' : C.muted}
          fontWeight={700}>西侧细胞</text>
        {cellCol('east', eastGrow, '#f0b245')}
        {cellCol('west', westGrow, '#7fb4e8')}
        {/* 中央分界线 */}
        <line x1={CX} y1={CY - H / 2 + 90} x2={CX} y2={CY + H / 2 - 30} stroke="#ffffff"
          strokeWidth={2} strokeDasharray="4 12" opacity={0.2} />
        <text x={CX} y={CY + H / 2 + 58} textAnchor="middle" fill={C.muted}
          fontSize={34} fontFamily={FONT_HEAD}>茎两侧 · 差速伸长</text>

        {/* 生物钟时钟 */}
        <g opacity={clamp01((f - beat(1)) / 12)}>
          <Clock x={730} y={330} r={92} spin={1} />
          <text x={730} y={452} textAnchor="middle" fill={C.amber} fontSize={32}
            fontFamily={FONT_HEAD} fontWeight={700}>体内的生物钟</text>
        </g>
        <Ripple x={CX} y={CY} startFrame={255} color="#7fb4e8" />
      </svg>
      <Stamp text="白天：东侧狂长推向西" x={150} y={170} at={95} color={C.amber} size={44} rot={-4} />
      <Stamp text="夜里：换西侧推回东" x={150} y={290} at={270} color="#4a80c8" size={44} rot={3} />
      <Stamp text="不是眼睛追光，是钟表跑点" x={880} y={140} at={beat(1) + 10} color={C.coral} size={46} rot={3} />
    </FadeIn>
  );
};

/* ============ S3 朝东：成熟花盘钉死朝东，升温抢蜜蜂 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number, fb = 5) => Math.round(((sec.cues[i] ? sec.cues[i].t0 : sec.start + fb) - sec.start) * fps);

  const sunY = interpolate(f, [15, 150], [870, 640], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const dawn = interpolate(f, [10, 120], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const HX = 960, HY = 990, SC = 1.3;
  const headY = HY - STEM_H * SC; // ≈ 600

  // 蜜蜂 A：从太阳方向飞来 → 在花盘左侧悬停
  const beeIn = 268;
  const bA = clamp01((f - beeIn) / 62);
  const beeAx = interpolate(bA, [0, 1], [240, 838]) + Math.sin(f * 0.09) * 12 * bA;
  const beeAy = interpolate(bA, [0, 1], [420, 578]) + Math.sin(f * 0.13) * 34 * bA * (1 - bA)
    + Math.cos(f * 0.11) * 9 * bA;
  // 蜜蜂 B：绕花盘转圈
  const beeIn2 = 330;
  const ang = (f - beeIn2) * 0.055;
  const beeBx = HX + Math.cos(ang + 2.6) * 185;
  const beeBy = headY + Math.sin(ang + 2.6) * 118;

  // 对照组卡片
  const cardAt = beat(2) + 150;
  const cardS = spring({frame: f - cardAt, fps, config: {damping: 14, mass: 0.7}});

  return (
    <FadeIn>
      <Backdrop top="#2b3a75" bottom="#151a40" glow="#ffc94d10" />
      <SceneTag index={3} name={sec.name} accent={C.amber} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <defs>
          <linearGradient id="dawnSky" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#7a4a7e" />
            <stop offset="100%" stopColor={C.dusk1} />
          </linearGradient>
          <radialGradient id="sunGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#ffc94d55" />
            <stop offset="100%" stopColor="#ffc94d00" />
          </radialGradient>
        </defs>
        <rect width={1920} height={1080} fill="url(#dawnSky)" opacity={dawn * 0.75} />
        <circle cx={330} cy={sunY} r={330} fill="url(#sunGlow)" />
        <Sun x={330} y={sunY} r={80} />
        <ellipse cx={960} cy={1030} rx={1200} ry={140} fill="#10142e" opacity={0.9} />

        {/* 一排成熟花盘：全部钉死朝东 */}
        <Sunflower x={520} y={968} scale={0.68} tilt={-58} opacity={0.8} sway={false} petals={14} />
        <Sunflower x={1400} y={972} scale={0.78} tilt={-65} opacity={0.85} sway={false} petals={14} />
        <Sunflower x={HX} y={HY} scale={SC} tilt={-62} sway={false} />

        {/* 禁转标志：不再追了 */}
        <g transform="translate(1178,430)" opacity={clamp01((f - 90) / 15)}>
          <circle r={52} fill="none" stroke={C.coral} strokeWidth={7} />
          <path d="M -28 -10 A 30 30 0 0 1 22 -22" fill="none" stroke={C.muted} strokeWidth={7} strokeLinecap="round" />
          <path d="M 28 -32 L 24 -12 L 8 -26 Z" fill={C.muted} />
          <line x1={-36} y1={36} x2={36} y2={-36} stroke={C.coral} strokeWidth={8} strokeLinecap="round" />
        </g>

        {/* 热浪：朝东的面最早升温 */}
        {[0, 1, 2].map((i) => {
          const t = (((f - 200 + i * 12) % 46) + 46) % 46;
          const o = f > 200 ? clamp01(t / 10) * clamp01(1 - (t - 30) / 16) : 0;
          return (
            <path key={i} d="M 0 0 q 14 -20 0 -40 q -14 -20 0 -40"
              transform={`translate(${830 + i * 34},${560 - i * 10 - t * 1.6})`}
              fill="none" stroke={C.amber} strokeWidth={6} strokeLinecap="round" opacity={o * 0.85} />
          );
        })}

        {f > beeIn && <Bee x={beeAx} y={beeAy} rot={Math.sin(f * 0.1) * 10} />}
        {f > beeIn2 && <Bee x={beeBx} y={beeBy} scale={0.85} rot={Math.cos(ang) * 18} />}
        <Ripple x={830} y={580} startFrame={beeIn + 60} color={C.amber} />

        {/* 对照组卡片：被拧到西边 */}
        {f > cardAt - 2 && (
          <g transform={`translate(1560,265) scale(${Math.max(0.001, cardS)}) translate(-1560,-265)`}
            opacity={clamp01((f - cardAt) / 10)}>
            <rect x={1240} y={110} width={640} height={310} rx={28}
              fill="rgba(14,17,38,0.92)" stroke="#4a55a8" strokeWidth={3} />
            <text x={1280} y={168} fill={C.muted} fontSize={30} fontFamily={FONT_HEAD}>对照组：被人拧到西边</text>
            <Sunflower x={1620} y={400} scale={0.45} tilt={62} sway={false} opacity={0.85} petals={12} headR={80} />
            <text x={1280} y={250} fill={C.coral} fontSize={36} fontFamily={FONT_HEAD} fontWeight={700}>
              蜜蜂来访 ≈ 0 次
            </text>
            <text x={1280} y={300} fill={C.muted} fontSize={26} fontFamily={FONT_HEAD}>一上午门可罗雀</text>
          </g>
        )}
      </svg>
      <Stamp text="钉死朝东" x={1250} y={560} at={beat(0) + 40} color={C.coral} size={52} rot={6} />
      <Chip text="清晨最早晒到太阳" x={180} y={240} at={beat(1) + 30} accent={C.amber} />
      <Stamp text="暖和的着陆场最招蜂" x={170} y={360} at={beat(1) + 55} color={C.amber} size={42} rot={-5} />
    </FadeIn>
  );
};

/* ============ S4 收尾：花盘斐波那契螺线 + 时间表写进茎里 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number, fb = 5) => Math.round(((sec.cues[i] ? sec.cues[i].t0 : sec.start + fb) - sec.start) * fps);

  const CXS = 960, CYS = 545, R = 315;
  const GOLD = 2.39996; // 137.5°
  const seedsN = Math.min(140, Math.floor((f - 20) / 0.85));
  const spiralO = interpolate(f, [185, 215], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const punch = beat(1);
  const dim = interpolate(f, [punch, punch + 25], [1, 0.25], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});

  const spiralPath = (dir: 1 | -1, rot: number) => {
    let d = '';
    for (let t = 0.6; t <= 15.6; t += 0.3) {
      const r = 16 + t * 17.8;
      const a = dir * t + rot;
      d += `${d === '' ? 'M' : 'L'} ${(CXS + Math.cos(a) * r).toFixed(1)} ${(CYS + Math.sin(a) * r).toFixed(1)}`;
    }
    return d;
  };

  return (
    <FadeIn>
      <Backdrop top="#232a52" bottom={C.bgDeep} glow="#ffc94d12" />
      <SceneTag index={4} name={sec.name} accent={C.amber} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 大特写花盘：花瓣在后缓慢旋转 */}
        <g transform={`translate(${CXS},${CYS})`}>
          <g transform={`rotate(${f * 0.18})`}>
            {Array.from({length: 20}).map((_, i) => (
              <ellipse key={i} cx={0} cy={-(R + 68)} rx={30} ry={82}
                fill={i % 2 ? C.amber : '#ffb03a'} opacity={0.95}
                transform={`rotate(${i * 18})`} />
            ))}
          </g>
        </g>
        <circle cx={CXS} cy={CYS} r={R + 8} fill="#8a5327" />
        <circle cx={CXS} cy={CYS} r={R + 8} fill="none" stroke="#5e3a1e" strokeWidth={8} />
        <g opacity={dim}>
          <circle cx={CXS} cy={CYS} r={R - 20} fill="#6b3d1c" />
          {/* 籽按黄金角逐颗长出 */}
          {Array.from({length: Math.max(0, seedsN)}).map((_, i) => {
            const a = i * GOLD;
            const r = 23.5 * Math.sqrt(i);
            return (
              <ellipse key={i} cx={CXS + Math.cos(a) * r} cy={CYS + Math.sin(a) * r}
                rx={8.5} ry={13} fill={i % 3 ? '#4a2a12' : '#5e3a1e'}
                transform={`rotate(${(a * 180) / Math.PI + 90} ${CXS + Math.cos(a) * r} ${CYS + Math.sin(a) * r})`} />
            );
          })}
          {/* 两组反向螺旋 */}
          {[0, 1, 2, 3].map((k) => (
            <path key={`a${k}`} d={spiralPath(1, (k * Math.PI) / 2)} fill="none"
              stroke={C.amber} strokeWidth={5} opacity={spiralO * 0.9} strokeLinecap="round" />
          ))}
          {[0, 1, 2, 3].map((k) => (
            <path key={`b${k}`} d={spiralPath(-1, (k * Math.PI) / 2 + 0.5)} fill="none"
              stroke="#5cc8ff" strokeWidth={5} opacity={spiralO * 0.85} strokeLinecap="round" />
          ))}
        </g>
        {/* 茎伸出画面 */}
        <path d={`M ${CXS} ${CYS + R} L ${CXS} 1100`} stroke={C.stem} strokeWidth={26} />
        {/* 彩蛋揭晓后：花盘中心浮现时钟 = 时间表 */}
        <g opacity={clamp01((f - punch) / 15)}>
          <circle cx={CXS} cy={CYS} r={86} fill="#10142e" opacity={0.92} />
          <Clock x={CXS} y={CYS} r={72} spin={1} />
        </g>
      </svg>
      <Chip text="34 条 · 顺时针" x={150} y={180} at={200} accent={C.amber} />
      <Chip text="55 条 · 逆时针" x={150} y={285} at={215} accent="#5cc8ff" />
      <Chip text="黄金角 137.5°" x={1420} y={180} at={230} accent={C.amber} />
      <Stamp text="太阳的时间表写进了茎里" x={570} y={120} at={punch + 8} color={C.amber} size={54} rot={-3} />
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
      <Backdrop top={C.night1} bottom={C.bgDeep} glow="#ffc94d12" />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={18} from={0} dur={30} />
        <Moon x={1520} y={220} opacity={0.9} />
        <Sunflower x={960} y={1105} scale={0.95} tilt={-38} sway={false} />
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -160}}>
        <div style={{fontSize: 70, fontWeight: 800, color: C.text}}>它不是追太阳，是按表打卡</div>
        <div style={{fontSize: 38, color: C.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E202 = ({data = tl}: {data?: any}) => {
  const {fps} = useVideoConfig();
  return (
    <AbsoluteFill style={{background: C.bgDeep}}>
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
