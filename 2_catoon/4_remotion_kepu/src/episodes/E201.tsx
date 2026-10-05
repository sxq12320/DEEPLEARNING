import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01} from '../lib/style';
import {Backdrop, Ripple, Stamp, SceneTag, Chip, Subtitles, ProgressBar, Finger, Sun, Moon, Stars, Bug, Rain, FadeIn} from '../lib/atoms';
import {MimosaPlant, stemPoint, stemPath} from '../lib/Mimosa';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};
type TL = {title: string; episode: string; titleCardSec: number; endCardSec: number; totalSec: number; sections: Sec[]};

/* ============ 标题卡 ============ */
const TitleCard = ({data}: {data: TL}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 13, mass: 0.7}});
  const o = interpolate(f, [0, 8], [0, 1]);
  return (
    <AbsoluteFill>
      <Backdrop top={C.bg1} bottom={C.bgDeep} glow="#58cc7e18" />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 小叶子图标，开合一次卖萌 */}
        <g transform="translate(960,330)">
          <g transform={`rotate(${-24 - Math.max(0, Math.sin(f / 14)) * 26})`}>
            <ellipse cx={55} cy={0} rx={58} ry={22} fill={C.leaf} />
          </g>
          <g transform={`rotate(${24 + Math.max(0, Math.sin(f / 14)) * 26})`}>
            <ellipse cx={55} cy={0} rx={58} ry={22} fill={C.leafDark} />
          </g>
          <circle r={16} fill={C.stemDark} />
        </g>
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 140}}>
          <div style={{color: C.leaf, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 26}}>
            冷知识 · 第 {data.episode} 期
          </div>
          <div style={{color: C.text, fontSize: 104, fontWeight: 900, fontFamily: FONT_HEAD, letterSpacing: 6}}>
            {data.title}
          </div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ============ S0 开场：手指碰，叶子唰地合上 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();

  // 手指入场 → 触碰
  const touchF = 82;
  const fx = interpolate(f, [40, touchF], [1700, 1180], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const fy = interpolate(f, [40, touchF], [760, 580], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const back = interpolate(f, [touchF + 20, touchF + 60], [0, 90], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});

  // 折叠波：触后依次倒
  const wave = interpolate(f, [touchF + 4, touchF + 60], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const folds = [0, 1, 2, 3].map((i) => clamp01(wave * 1.7 - i * 0.22));
  const droop = clamp01(wave * 1.3 - 0.5);

  const cue2 = Math.round((sec.cues[1].t0 - sec.start) * fps);
  return (
    <FadeIn>
      <Backdrop top="#1c2350" bottom={C.bgDeep} glow="#5cc8ff12" />
      <SceneTag index={0} name={sec.name} accent={C.leaf} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <MimosaPlant x={900} y={960} scale={1.25} folds={folds} droop={droop} />
        <Ripple x={1180} y={560} startFrame={touchF} />
        <Finger x={fx + back} y={fy + back * 0.5} rot={-32} scale={1.15} />
      </svg>
      <Stamp text="唰！" x={1240} y={430} at={touchF + 2} color={C.coral} size={64} rot={-10} />
      <Stamp text="没有神经，也能动？" x={170} y={240} at={cue2} color="#3d4a8f" size={52} rot={-4} />
    </FadeIn>
  );
};

/* ============ S1 机关：放大叶枕横截面，细胞放水 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[i].t0 - sec.start) * fps);

  const drain = interpolate(f, [beat(2) + 4, beat(2) + 30], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const cellScale = 1 - drain * 0.48;
  const leafFold = clamp01(drain * 1.2);

  const CX = 1250, CY = 530, R = 300;
  return (
    <FadeIn>
      <Backdrop top="#1a2148" bottom={C.bgDeep} glow="#63e2c612" />
      <SceneTag index={1} name={sec.name} accent={C.cell} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <defs>
          <clipPath id="cellClip"><circle cx={CX} cy={CY} r={R - 14} /></clipPath>
        </defs>

        {/* 左：迷你植株 + 虚线圈定位叶枕 */}
        <MimosaPlant x={560} y={930} scale={0.72} folds={[0, 0, leafFold, 0]} droop={leafFold * 0.4} />
        <circle cx={560 - 3 * 0.72} cy={930 - 237 * 0.72} r={66} fill="none"
          stroke={C.amber} strokeWidth={4} strokeDasharray="10 10" opacity={0.9} />
        <line x1={626} y1={760} x2={CX - R * 0.72} y2={CY - R * 0.55}
          stroke={C.amber} strokeWidth={4} strokeDasharray="10 10" opacity={0.7} />

        {/* 右：叶枕横截面大圆 */}
        <circle cx={CX} cy={CY} r={R} fill="#0e1b3d" stroke={C.cell} strokeWidth={6} />
        <circle cx={CX} cy={CY} r={R + 14} fill="none" stroke={C.cell} strokeWidth={3} opacity={0.3} />
        <g clipPath="url(#cellClip)">
          {/* 中央髓部 */}
          <circle cx={CX} cy={CY} r={86} fill="#16305a" />
          {/* 一圈水分饱满的细胞 */}
          {Array.from({length: 8}).map((_, i) => {
            const a = (i / 8) * Math.PI * 2 - Math.PI / 2;
            const rr = 185;
            const cx = CX + Math.cos(a) * rr;
            const cy = CY + Math.sin(a) * rr;
            const pulse = 1 + Math.sin(f / 12 + i) * 0.04 * (1 - drain);
            return (
              <g key={i} transform={`translate(${cx},${cy}) scale(${cellScale * pulse}) rotate(${(a * 180) / Math.PI + 90})`}>
                <ellipse rx={88} ry={70} fill={C.cell} />
                <ellipse cy={18} rx={72} ry={42} fill={C.cellDark} opacity={0.5} />
                <circle cx={-14} cy={-12} r={14} fill="#bdf5e5" opacity={0.8} />
              </g>
            );
          })}
          {/* 放水：水滴向下流出 */}
          {drain > 0 && Array.from({length: 14}).map((_, i) => {
            const a = (i / 14) * Math.PI * 2;
            const sx = CX + Math.cos(a) * 185;
            const sy = CY + Math.sin(a) * 185;
            const drop = clamp01(drain * 1.4 - (i % 4) * 0.12);
            return (
              <circle key={i} cx={sx + (CX - sx) * drop * 0.4} cy={sy + drop * (CY + R - 60 - sy)}
                r={9} fill={C.water} opacity={drop > 0 && drop < 1 ? 0.95 : 0} />
            );
          })}
        </g>
        <text x={CX} y={CY - R - 40} textAnchor="middle" fill={C.cell}
          fontSize={54} fontFamily={FONT_HEAD} fontWeight={700}>叶枕 pulvinus</text>
        <Ripple x={CX} y={CY} startFrame={beat(2)} color={C.water} />
      </svg>
      <Chip text="吸饱水的细胞" x={320} y={180} at={beat(1) + 6} accent={C.cell} />
      <Stamp text="气球撒气！" x={1350} y={200} at={beat(2) + 8} color={C.coral} size={54} rot={6} />
    </FadeIn>
  );
};

/* ============ S2 信号：电脉冲顺茎跑，叶子一波波倒 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[i].t0 - sec.start) * fps);

  const start = beat(1) + 8;
  const T = 130; // 脉冲走完全程帧数
  const p = clamp01((f - start) / T);          // 0→1 从上往下
  const pulseT = 0.96 - p * (0.96 - 0.2);      // 茎参数当前位置

  const leafT = [0.32, 0.55, 0.78, 0.95];
  const folds = leafT.map((t) => clamp01((t - pulseT) * 7));
  const pos = stemPoint(pulseT);
  const PX = 960, PY = 955, SC = 1.15;

  return (
    <FadeIn>
      <Backdrop top="#141a3d" bottom={C.bgDeep} glow="#ffc94d10" />
      <SceneTag index={2} name={sec.name} accent={C.amber} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 信号走线虚影 */}
        <g transform={`translate(${PX},${PY}) scale(${SC})`}>
          <path d={stemPath} fill="none" stroke={C.amber} strokeWidth={5}
            strokeDasharray="2 22" strokeLinecap="round" opacity={0.35} />
        </g>
        <MimosaPlant x={PX} y={PY} scale={SC} folds={folds} droop={p * 0.5} />
        {/* 每个节点经过时的小闪光 */}
        {leafT.map((t, i) => {
          const node = stemPoint(t);
          const hit = clamp01((t - pulseT) * 7);
          const ringP = clamp01(hit * 2.5);
          return (
            <g key={i} transform={`translate(${PX + node.x * SC},${PY + node.y * SC})`}>
              <circle r={14} fill={C.amber} opacity={0.25 + hit * 0.75} />
              {hit > 0 && hit < 1 && (
                <circle r={14 + ringP * 46} fill="none" stroke={C.amber}
                  strokeWidth={4 * (1 - ringP)} opacity={0.8 * (1 - ringP)} />
              )}
            </g>
          );
        })}
        {/* 脉冲本体：亮核 + 光晕 + 拖尾 */}
        {p > 0 && p < 1 && (
          <g transform={`translate(${PX + pos.x * SC},${PY + pos.y * SC})`}>
            {Array.from({length: 6}).map((_, i) => {
              const tp = Math.min(0.96, pulseT + i * 0.012);
              const tp2 = stemPoint(tp);
              return <circle key={i} cx={(tp2.x - pos.x)} cy={(tp2.y - pos.y)}
                r={12 - i * 1.6} fill={C.amber} opacity={0.5 - i * 0.08} />;
            })}
            <circle r={46} fill={C.amber} opacity={0.22} />
            <circle r={17} fill="#ffe9b0" />
          </g>
        )}
        <Ripple x={PX + stemPoint(0.95).x * SC} y={PY + stemPoint(0.95).y * SC}
          startFrame={start - 6} />
      </svg>
      <Stamp text="电信号" x={1240} y={330} at={start + 10} color={C.amber} size={52} rot={-6} />
      <Chip text="一波一波倒，不是一起倒" x={240} y={280} at={beat(2) + 4} accent={C.amber} />
    </FadeIn>
  );
};

/* ============ S3 晚上：日落月升，睡觉模式 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[i].t0 - sec.start) * fps);

  const night = interpolate(f, [10, 110], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const sunY = interpolate(f, [0, 120], [300, 900], {extrapolateRight: 'clamp'});
  const sleep = interpolate(f, [70, 210], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const folds = [sleep, sleep * 0.95, sleep, sleep * 0.9].map(clamp01);

  return (
    <FadeIn>
      {/* 黄昏底 + 夜色渐入 */}
      <Backdrop top={C.dusk0} bottom={C.dusk1} glow="#ffffff10" />
      <div style={{position: 'absolute', inset: 0, opacity: night}}>
        <Backdrop top={C.night0} bottom={C.night1} glow="#5cc8ff10" />
      </div>
      <SceneTag index={3} name={sec.name} accent={C.sky} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={26} from={50} dur={60} />
        <Sun x={430} y={sunY} opacity={1 - night * 0.9} />
        <Moon x={1520} y={230 - night * 60} opacity={night} />
        {/* 地面剪影 */}
        <ellipse cx={960} cy={1010} rx={1150} ry={150} fill={C.night0} opacity={0.85} />
        <MimosaPlant x={960} y={975} scale={1.15} folds={folds} droop={sleep * 0.35} sway={false} />
        {/* Zzz 飘上去 */}
        {[0, 1, 2].map((i) => {
          const t = f - 170 - i * 26;
          const o = t > 0 ? clamp01(t / 20) * clamp01(1 - (t - 60) / 30) : 0;
          return (
            <text key={i} x={1180 + i * 60} y={620 - Math.max(0, t) * 2.2 - i * 40}
              fill="#9fb4ff" fontSize={44 + i * 16} fontFamily={FONT_HEAD} opacity={o}>Z</text>
          );
        })}
      </svg>
      <Stamp text="不是怕，是睡觉" x={210} y={270} at={beat(1) + 4} color="#4a55a8" size={52} rot={-5} />
    </FadeIn>
  );
};

/* ============ S4 图啥：虫来塌叶吓退，雨来收叶挡 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[i].t0 - sec.start) * fps);

  // 时间轴（局部帧）
  const bugIn = 60, bugStep = 150, bugOff = 175, rainOn = 215, rainOff = 305, reopen = 320;

  const bugX = interpolate(f, [bugIn, bugStep], [240, 1120], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const bugJump = f > bugStep ? interpolate(f, [bugStep, bugStep + 40], [0, 1], {extrapolateRight: 'clamp'}) : 0;
  const bugY = f <= bugStep ? 940 : 940 - Math.sin(bugJump * Math.PI) * 160;
  const bugXX = bugX + bugJump * 300;
  const bugRot = bugJump * 220;
  const bugO = bugJump >= 1 ? 0 : 1;

  const foldA = clamp01(interpolate(f, [bugStep, bugStep + 14], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}));
  const rainK = interpolate(f, [rainOn, rainOn + 15, rainOff, rainOff + 15], [0, 1, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const foldB = clamp01(interpolate(f, [rainOn, rainOn + 16], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}));
  const openK = clamp01(interpolate(f, [reopen, reopen + 70], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}));
  const masterFold = Math.max(foldA, foldB) * (1 - openK);
  const folds = [masterFold, masterFold * 0.9, masterFold, masterFold * 0.85].map(clamp01);

  return (
    <FadeIn>
      <Backdrop top="#1c2350" bottom={C.bgDeep} glow="#58cc7e12" />
      <SceneTag index={4} name={sec.name} accent={C.leaf} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Rain on={rainK} />
        <MimosaPlant x={1050} y={960} scale={1.3} folds={folds} droop={masterFold * 0.4} />
        {/* 雨滴砸在收起叶子上的弹开 */}
        {rainK > 0.5 && [0, 1, 2].map((i) => {
          const t = (f + i * 7) % 18;
          return <circle key={i} cx={1000 + i * 55} cy={700 - t * 4} r={8 - t * 0.4}
            fill="none" stroke={C.sky} strokeWidth={3} opacity={0.7 * (1 - t / 18)} />;
        })}
        {bugO > 0 && <Bug x={bugXX} y={bugY} rot={bugRot} walk={f > bugStep ? 0 : 1} scale={1.05} />}
        <Ripple x={1120} y={880} startFrame={bugStep} color={C.coral} />
      </svg>
      <Stamp text="下嘴失败" x={1250} y={760} at={bugStep + 14} color={C.coral} size={44} rot={8} />
      <Stamp text="少挨几下" x={1330} y={280} at={rainOn + 24} color={C.sky} size={44} rot={-6} />
      <Chip text="防虫" x={250} y={260} at={beat(3) - 40 > 0 ? bugOff + 10 : 0} accent={C.leaf} />
      <Chip text="挡雨" x={430} y={260} at={reopen - 20} accent={C.sky} />
      <Stamp text="开合很费能量" x={1450} y={150} at={beat(3) + 6} color={C.amber} size={46} rot={4} />
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
      <Backdrop top={C.bg1} bottom={C.bgDeep} glow="#58cc7e16" />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <MimosaPlant x={960} y={1080} scale={0.9} folds={[0, 0, 0, 0]} />
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -120}}>
        <div style={{fontSize: 74, fontWeight: 800, color: C.text}}>下次路过，轻一点碰</div>
        <div style={{fontSize: 38, color: C.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E201 = ({data}: {data: TL}) => {
  const {fps} = useVideoConfig();
  return (
    <AbsoluteFill style={{background: C.bgDeep}}>
      <Sequence durationInFrames={Math.round(data.titleCardSec * fps)}>
        <TitleCard data={data} />
      </Sequence>

      {data.sections.map((sec, i) => {
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
