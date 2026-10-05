import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01, rand} from '../lib/style';
import {Backdrop, Stamp, SceneTag, Chip, Subtitles, ProgressBar, FadeIn, Moon, Stars} from '../lib/atoms';
import tl from '../timeline_235.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：深青墨绿 × 神经黄 × 信号青 × 珊瑚红 ============ */
const P = {
  bg: '#15313b',
  bgDeep: '#0d2129',
  bgLite: '#1c404c',
  panel: '#112a32',
  card: '#2b4b55',
  ink: '#102330',
  cream: '#f3ecd8',
  muted: '#9dc3cd',
  nerve: '#ffd166',
  nerveD: '#dfa63c',
  sig: '#5fe0ff',
  coral: '#ff7a6e',
  coralD: '#d8564c',
  teal: '#57d9b0',
  bone: '#ece4d0',
  blood: '#ff8d7a',
  bloodD: '#d95f50',
  hair: '#39304e',
};

/* ---------- 小叉 / 小勾 ---------- */
const MarkX = ({x, y, s = 1, col = P.coral}: {x: number; y: number; s?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} stroke={col} strokeWidth={7} strokeLinecap="round">
    <line x1={-13} y1={-13} x2={13} y2={13} /><line x1={13} y1={-13} x2={-13} y2={13} />
  </g>
);
const MarkV = ({x, y, s = 1, col = P.teal}: {x: number; y: number; s?: number; col?: string}) => (
  <path d="M -13 0 L -3 11 L 15 -11" transform={`translate(${x},${y}) scale(${s})`}
    stroke={col} strokeWidth={7} fill="none" strokeLinecap="round" strokeLinejoin="round" />
);

/* ---------- 飘z ---------- */
const Zzz = ({x, y, k = 1, col = P.sig}: {x: number; y: number; k?: number; col?: string}) => {
  const f = useCurrentFrame();
  return (
    <g transform={`translate(${x},${y})`} opacity={k}>
      {[0, 1, 2].map((i) => {
        const t = (f * 0.9 + i * 20) % 60;
        const p = t / 60;
        return (
          <text key={i} x={i * 24 + Math.sin(p * 3) * 9} y={-p * 70}
            fontSize={24 + i * 13} fill={col} fontFamily={FONT_HEAD}
            fontWeight={700} opacity={0.95 * (1 - p)}>z</text>
        );
      })}
    </g>
  );
};

/* ---------- 神经被压处的火花 ---------- */
const Spark = ({x, y, s = 1, col = P.nerve, spin = 1}: {x: number; y: number; s?: number; col?: string; spin?: number}) => {
  const f = useCurrentFrame();
  const pulse = 0.75 + Math.abs(Math.sin(f * 0.35)) * 0.45;
  return (
    <g transform={`translate(${x},${y}) scale(${s * pulse}) rotate(${f * spin})`} stroke={col}
      strokeWidth={5} strokeLinecap="round">
      {[0, 1, 2, 3, 4, 5].map((i) => {
        const a = i * 60 * Math.PI / 180;
        return <line key={i} x1={Math.cos(a) * 12} y1={Math.sin(a) * 12}
          x2={Math.cos(a) * 30} y2={Math.sin(a) * 30} />;
      })}
      <circle r={6} fill={col} stroke="none" />
    </g>
  );
};

/* ---------- 圆角标签牌 ---------- */
const TagPill = ({x, y, t, col, s = 1, w = 200, fs = 30}: {
  x: number; y: number; t: string; col: string; s?: number; w?: number; fs?: number;
}) => (
  <g transform={`translate(${x},${y}) scale(${s})`}>
    <rect x={-w / 2} y={-30} width={w} height={60} rx={16} fill={P.panel} stroke={col} strokeWidth={4} />
    <text x={0} y={11} textAnchor="middle" fontSize={fs} fill={col}
      fontFamily={FONT_HEAD} fontWeight={800}>{t}</text>
  </g>
);

/* ---------- 床上小人：side 侧睡压臂 / prone 趴睡垫臂 / free 手臂放好 ---------- */
const Sleeper = ({x, y, s = 1, mode = 'side', zzz = true}: {
  x: number; y: number; s?: number; mode?: 'side' | 'prone' | 'free'; zzz?: boolean;
}) => {
  const f = useCurrentFrame();
  const br = Math.sin(f * 0.14) * 3;
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      {/* 床 */}
      <rect x={-260} y={-10} width={520} height={30} rx={15} fill={P.card} />
      <rect x={-244} y={20} width={22} height={56} rx={9} fill={P.card} />
      <rect x={222} y={20} width={22} height={56} rx={9} fill={P.card} />
      {/* 枕头（趴睡时枕头让位给手臂） */}
      {mode !== 'prone' && <rect x={-226} y={-64} width={150} height={56} rx={24} fill={P.cream} />}
      {/* 被压住的手臂：沿床面伸直、垫在头/枕下 */}
      {mode === 'side' && (
        <g>
          <rect x={-196} y={-36} width={252} height={30} rx={15} fill={C.skin} />
          <circle cx={66} cy={-21} r={18} fill={C.skin} />
          <path d="M 58 -30 l 7 8 M 70 -32 l 7 8 M 82 -30 l 6 8" stroke={C.skinDark}
            strokeWidth={3.5} strokeLinecap="round" />
        </g>
      )}
      {mode === 'prone' && (
        <g>
          <rect x={-196} y={-30} width={170} height={28} rx={14} fill={C.skin} />
          <rect x={-196} y={-30} width={60} height={28} rx={14} fill={C.skinDark} opacity={0.5} />
        </g>
      )}
      {mode === 'free' && (
        <g>
          <rect x={-40} y={-30} width={210} height={28} rx={14} fill={C.skin} transform="rotate(-6 40 -16)" />
          <circle cx={176} cy={-42} r={17} fill={C.skin} />
        </g>
      )}
      {/* 被子下的身体 */}
      <path d={`M -58 -8 Q 24 ${-104 + br} 196 -12 L 244 -8 L 244 8 L -58 8 Z`} fill={P.teal} opacity={0.92} />
      <path d={`M -58 -8 Q 24 ${-104 + br} 196 -12`} fill="none" stroke={P.cream}
        strokeWidth={5} strokeLinecap="round" opacity={0.55} />
      {/* 头 */}
      <g transform={`translate(-152,${mode === 'prone' ? -52 : -84})`}>
        <circle r={33} fill={C.skin} />
        <path d="M -32 -8 Q 0 -46 32 -8 L 32 2 Q 0 -22 -32 2 Z" fill={P.hair} />
        <path d="M -14 4 q 6 7 12 0 M 6 4 q 6 7 12 0" stroke={P.ink} strokeWidth={3.2}
          fill="none" strokeLinecap="round" />
      </g>
      {zzz && <Zzz x={-92} y={mode === 'prone' ? -86 : -128} k={0.85} />}
    </g>
  );
};

/* ---------- 信号小点：沿神经跑动，pressed 时在断点前排不上 ---------- */
const SignalDots = ({pressed, color = P.sig}: {pressed: number; color?: string}) => {
  const f = useCurrentFrame();
  return (
    <g>
      {[0, 1, 2, 3].map((i) => {
        const t = ((f * 0.02 + i * 0.25) % 1);
        // pressed: 点跑到 62% 处就停下消失
        const lim = pressed > 0.5 ? 0.6 : 1;
        if (t > lim) return null;
        const x = -230 + t * 460;
        const y = Math.sin(t * 9) * 8;
        const fade = pressed > 0.5 ? clamp01((lim - t) * 8) : 1;
        return <circle key={i} cx={x} cy={y} r={9} fill={color} opacity={fade} />;
      })}
    </g>
  );
};

/* ============ 标题卡 ============ */
const TitleCard = ({data}: {data: any}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 13, mass: 0.7}});
  const o = interpolate(f, [0, 8], [0, 1]);
  const pA = Math.max(0.001, spring({frame: f - 4, fps, config: {damping: 10, stiffness: 150}}));
  const pB = Math.max(0.001, spring({frame: f - 16, fps, config: {damping: 10, stiffness: 150}}));
  return (
    <AbsoluteFill>
      <Backdrop top={P.bg} bottom={P.bg} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={22} from={0} dur={24} />
        <Moon x={250} y={150} scale={0.7} />
        {/* 中央：趴着压住手臂睡觉 + 神经火花 */}
        <g transform={`translate(960,330) scale(${pA}) translate(-960,-330)`} opacity={clamp01(pA)}>
          <Sleeper x={960} y={330} s={1.15} mode="side" zzz={false} />
          {/* 手臂上的神经线 + 压迫火花 */}
          <g transform="translate(960,330) scale(1.15)">
            <path d="M -196 -21 L 60 -21" stroke={P.nerve} strokeWidth={7} strokeLinecap="round" opacity={0.95} />
            <Spark x={-60} y={-22} s={0.8} />
            <text x={40} y={-58} fontSize={22} fill={P.nerve} fontFamily={FONT_HEAD}
              fontWeight={800} opacity={0.95}>神经被压</text>
          </g>
          <Zzz x={1070} y={186} k={0.9} />
          {/* 小问号 */}
          <g transform="translate(1250,180)">
            <circle r={34} fill={P.coral} />
            <text y={12} textAnchor="middle" fontSize={36} fill={P.cream}
              fontFamily={FONT_HEAD} fontWeight={900}>?</text>
          </g>
        </g>
        {/* 两侧小标签 */}
        <g transform={`translate(560,560) scale(${pB})`} opacity={clamp01(pB)}>
          <TagPill x={0} y={0} t="血没过来?" col={P.blood} w={240} />
          <MarkX x={132} y={-8} s={0.9} />
        </g>
        <g transform={`translate(1360,560) scale(${pB})`} opacity={clamp01(pB)}>
          <TagPill x={0} y={0} t="神经被压住" col={P.nerve} w={260} />
          <MarkV x={152} y={-6} s={0.9} />
        </g>
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 440}}>
          <div style={{color: P.sig, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 24}}>
            冷知识 · 第 {data.episode} 期
          </div>
          <div style={{color: C.text, fontSize: 78, fontWeight: 900, fontFamily: FONT_HEAD, letterSpacing: 4}}>
            {data.title}
          </div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/* ============ S0 手麻不是血的锅：流言 vs 真相 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const popL = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const popR = Math.max(0.001, spring({frame: f - 40, fps, config: {damping: 11, stiffness: 120}}));
  const verdictK = clamp01((f - b1 - 5) / 22);
  const conK = clamp01((f - b1 - 30) / 26);
  const wob = Math.sin(f * 0.5) * 3;

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={0} name={sec.name} accent={P.nerve} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={10} from={0} dur={26} />
        {/* 左：流行解释 —— 血管被压 */}
        <g transform={`translate(120,160) scale(${popL}) translate(-120,-160)`} opacity={clamp01(popL * 2)}>
          <rect x={120} y={160} width={740} height={430} rx={20} fill={P.panel}
            stroke={verdictK > 0.3 ? P.coral : P.card} strokeWidth={4} />
          <text x={156} y={214} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>长辈的流行版本</text>
          {/* 血管横管 */}
          <rect x={190} y={330} width={600} height={52} rx={26} fill={P.bloodD} />
          <rect x={190} y={330} width={600} height={22} rx={11} fill={P.blood} opacity={0.7} />
          {/* 被掐紧处 */}
          <rect x={452} y={306} width={76} height={100} rx={18} fill={P.card}
            transform={`rotate(6 490 356) translate(0 ${wob})`} />
          <rect x={452} y={306} width={76} height={34} rx={14} fill={P.cream} opacity={0.5}
            transform={`rotate(6 490 356) translate(0 ${wob})`} />
          <rect x={474} y={296} width={32} height={20} rx={8} fill={P.cream} />
          {/* 血滴：左边排队右边断流 */}
          {[0, 1, 2].map((i) => {
            const t = (f * 0.03 + i * 0.33) % 1;
            if (t > 0.55) return null;
            return <circle key={i} cx={220 + t * 460} cy={356} r={9} fill={P.cream} opacity={0.85} />;
          })}
          <text x={490} y={478} textAnchor="middle" fill={P.cream} fontSize={34}
            fontFamily={FONT_HEAD} fontWeight={800}>血没过来</text>
          {verdictK <= 0.3 && (
            <g transform="translate(735,255)">
              <circle r={30} fill={P.nerve} />
              <text y={11} textAnchor="middle" fontSize={32} fill={P.ink}
                fontFamily={FONT_HEAD} fontWeight={900}>?</text>
            </g>
          )}
          {/* 判决：错 */}
          {verdictK > 0 && (
            <g opacity={verdictK}>
              <MarkX x={735} y={255} s={1.6} />
              <rect x={330} y={510} width={320} height={56} rx={16} fill={P.coral} />
              <text x={490} y={549} textAnchor="middle" fontSize={28} fill={P.cream}
                fontFamily={FONT_HEAD} fontWeight={900}>只占个帮凶</text>
            </g>
          )}
        </g>
        {/* 右：真相 —— 神经被压住 */}
        <g transform={`translate(1060,160) scale(${popR}) translate(-1060,-160)`} opacity={clamp01(popR * 2)}>
          <rect x={1060} y={160} width={740} height={430} rx={20} fill={P.panel}
            stroke={verdictK > 0.3 ? P.teal : P.card} strokeWidth={4} />
          <text x={1096} y={214} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>真正的元凶</text>
          {/* 上压块（手臂/枕头） */}
          <g transform={`translate(1430,${268 + Math.sin(f * 0.3) * 4})`}>
            <rect x={-80} y={-56} width={160} height={60} rx={16} fill={P.card} />
            <rect x={-80} y={-56} width={160} height={18} rx={9} fill={P.muted} opacity={0.4} />
            <text x={0} y={-18} textAnchor="middle" fontSize={22} fill={P.muted}
              fontFamily={FONT_HEAD} fontWeight={700}>身体的重量</text>
          </g>
          {/* 下垫块（骨头） */}
          <rect x={1170} y={392} width={520} height={36} rx={16} fill={P.bone} opacity={0.9} />
          <text x={1690} y={420} textAnchor="end" fontSize={22} fill={P.bone}
            fontFamily={FONT_HEAD} fontWeight={700}>硬骨头</text>
          {/* 神经管：两头粗、压点扁 */}
          <path d="M 1150 352 L 1380 352 Q 1430 352 1430 366 Q 1430 380 1380 380 L 1150 380 Z" fill={P.nerve} />
          <path d="M 1480 352 L 1710 352 L 1710 380 L 1480 380 Q 1440 380 1440 366 Q 1440 352 1480 352 Z" fill={P.nerve} />
          <rect x={1424} y={358} width={22} height={16} rx={8} fill={P.nerveD} />
          {/* 信号点左段跑动、压点前消失 */}
          <g transform="translate(1400,366)">
            <SignalDots pressed={verdictK} />
          </g>
          <Spark x={1435} y={330} s={0.9} />
          <text x={1430} y={478} textAnchor="middle" fill={P.nerve} fontSize={34}
            fontFamily={FONT_HEAD} fontWeight={800}>神经被压 · 信号断线</text>
          {/* 判决：对 */}
          {verdictK > 0 && (
            <g opacity={verdictK}>
              <MarkV x={1740} y={255} s={1.5} />
              <rect x={1270} y={510} width={340} height={56} rx={16} fill={P.teal} />
              <text x={1440} y={549} textAnchor="middle" fontSize={28} fill={P.ink}
                fontFamily={FONT_HEAD} fontWeight={900}>麻木的真身</text>
            </g>
          )}
        </g>
        {/* 中央 VS */}
        <g transform="translate(960,375)">
          <circle r={40} fill={P.coral} />
          <text y={12} textAnchor="middle" fontSize={30} fill={P.cream}
            fontFamily={FONT_HEAD} fontWeight={900}>VS</text>
        </g>
        {/* 底部结论（cue1 后段） */}
        {conK > 0 && (
          <g opacity={conK} transform={`translate(0,${(1 - conK) * 60})`}>
            <rect x={120} y={640} width={1680} height={200} rx={20} fill={P.panel} stroke={P.nerve} strokeWidth={4} />
            <text x={160} y={700} fill={P.nerve} fontSize={30} fontFamily={FONT_HEAD} fontWeight={900}>
              缺血只是帮凶</text>
            <text x={160} y={756} fill={P.cream} fontSize={29} fontFamily={FONT_HEAD} fontWeight={700}>
              神经对压迫和缺氧远比肌肉敏感 —— 压久了第一个罢工的是它</text>
            {/* 敏感性小仪表 */}
            <g transform="translate(1500,720)">
              <text x={-30} y={-34} textAnchor="end" fill={P.muted} fontSize={22} fontFamily={FONT_HEAD}>敏感度</text>
              <rect x={-20} y={-52} width={220} height={26} rx={13} fill={P.card} />
              <rect x={-20} y={-52} width={220 * clamp01((f - b1 - 40) / 30)} height={26} rx={13} fill={P.nerve} />
              <text x={-20} y={8} fill={P.nerve} fontSize={22} fontFamily={FONT_HEAD} fontWeight={800}>神经 拉满</text>
              <rect x={-20} y={30} width={220} height={26} rx={13} fill={P.card} />
              <rect x={-20} y={30} width={70} height={26} rx={13} fill={P.bloodD} />
              <text x={-20} y={90} fill={P.muted} fontSize={22} fontFamily={FONT_HEAD} fontWeight={700}>肌肉 能扛</text>
            </g>
          </g>
        )}
      </svg>
      <Chip text="又麻又木像装了别人的手" x={1240} y={72} at={beat(0) + 25} accent={P.nerve} />
      <Stamp text="麻 = 神经断线的电噪" x={640} y={86} at={b1 + 45} color={P.coralD} size={38} rot={-3} />
    </FadeIn>
  );
};

/* ============ S1 哪条神经先倒霉：手臂神经地图 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const mapPop = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const posePop = [0, 1].map((i) => Math.max(0.001, spring({frame: f - 55 - i * 25, fps, config: {damping: 11, stiffness: 130}})));
  const pinchK = clamp01((f - b1) / 20);
  const nerveK = clamp01((f - b1 - 15) / 26);

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={1} name={sec.name} accent={P.sig} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：手臂神经地图 */}
        <g transform={`translate(110,150) scale(${mapPop}) translate(-110,-150)`} opacity={clamp01(mapPop * 2)}>
          <rect x={110} y={150} width={860} height={700} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={146} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            手臂剖面 · 两条走浅路贴骨头的神经</text>
          {/* 手臂外轮廓（上臂到手掌竖直示意） */}
          <rect x={430} y={250} width={160} height={420} rx={70} fill={C.skin} />
          {/* 手掌 + 手指 */}
          <circle cx={510} cy={712} r={66} fill={C.skin} />
          {[0, 1, 2].map((i) => (
            <rect key={i} x={462 + i * 34} y={752} width={24} height={52} rx={12} fill={C.skin} />
          ))}
          {/* 肱骨 */}
          <rect x={488} y={290} width={44} height={330} rx={20} fill={P.bone} />
          <circle cx={510} cy={296} r={26} fill={P.bone} />
          <circle cx={510} cy={616} r={26} fill={P.bone} />
          <text x={552} y={330} fill={P.ink} fontSize={20} fontFamily={FONT_HEAD} fontWeight={800}>骨</text>
          {/* 桡神经（外侧黄线，贴着骨走） */}
          {nerveK > 0 && (
            <g opacity={nerveK}>
              <path d="M 452 262 C 430 340 436 430 444 500 C 450 560 444 620 452 690"
                fill="none" stroke={P.nerve} strokeWidth={9} strokeLinecap="round"
                strokeDasharray="600" strokeDashoffset={600 * (1 - nerveK)} />
              <circle cx={452} cy={262} r={11} fill={P.nerve} />
            </g>
          )}
          {/* 尺神经（内侧青线，过肘内侧凹槽） */}
          {nerveK > 0 && (
            <g opacity={nerveK}>
              <path d="M 560 268 C 584 350 588 440 586 520 C 585 560 596 580 592 610 C 588 648 580 668 576 694"
                fill="none" stroke={P.sig} strokeWidth={9} strokeLinecap="round"
                strokeDasharray="640" strokeDashoffset={640 * (1 - nerveK)} />
              <circle cx={560} cy={268} r={11} fill={P.sig} />
            </g>
          )}
          {/* 凹槽提示：肘内侧 */}
          <path d="M 592 560 q 34 10 30 44" fill="none" stroke={P.bone} strokeWidth={5}
            strokeLinecap="round" opacity={0.8} />
          {/* 压迫高发点（cue1 起搏） */}
          {pinchK > 0 && (
            <g opacity={pinchK}>
              <circle cx={505} cy={248} r={26} fill="none" stroke={P.coral} strokeWidth={5}
                strokeDasharray="8 7" transform={`rotate(${f * 1.4} 505 248)`} />
              <Spark x={505} y={248} s={0.62} col={P.coral} />
              <circle cx={592} cy={586} r={26} fill="none" stroke={P.coral} strokeWidth={5}
                strokeDasharray="8 7" transform={`rotate(${-f * 1.4} 592 586)`} />
              <Spark x={592} y={586} s={0.62} col={P.coral} />
            </g>
          )}
          {/* 引线 + 标注 */}
          <line x1={452} y1={330} x2={300} y2={330} stroke={P.nerve} strokeWidth={3} strokeDasharray="6 6" />
          <rect x={140} y={296} width={172} height={92} rx={14} fill={P.bgDeep} stroke={P.nerve} strokeWidth={3.5} />
          <text x={226} y={332} textAnchor="middle" fill={P.nerve} fontSize={26}
            fontFamily={FONT_HEAD} fontWeight={900}>桡神经</text>
          <text x={226} y={366} textAnchor="middle" fill={P.muted} fontSize={20}
            fontFamily={FONT_HEAD}>外侧·贴骨浅路</text>
          <line x1={588} y1={470} x2={730} y2={470} stroke={P.sig} strokeWidth={3} strokeDasharray="6 6" />
          <rect x={724} y={436} width={172} height={92} rx={14} fill={P.bgDeep} stroke={P.sig} strokeWidth={3.5} />
          <text x={810} y={472} textAnchor="middle" fill={P.sig} fontSize={26}
            fontFamily={FONT_HEAD} fontWeight={900}>尺神经</text>
          <text x={810} y={506} textAnchor="middle" fill={P.muted} fontSize={20}
            fontFamily={FONT_HEAD}>内侧·肘凹槽</text>
          {/* 腋窝凹槽小注 */}
          {pinchK > 0 && (
            <g opacity={pinchK}>
              <line x1={505} y1={248} x2={330} y2={226} stroke={P.coral} strokeWidth={3} strokeDasharray="6 6" />
              <text x={318} y={232} textAnchor="end" fill={P.coral} fontSize={22}
                fontFamily={FONT_HEAD} fontWeight={800}>腋窝凹槽</text>
              <line x1={592} y1={586} x2={724} y2={610} stroke={P.coral} strokeWidth={3} strokeDasharray="6 6" />
              <text x={736} y={640} fill={P.coral} fontSize={22}
                fontFamily={FONT_HEAD} fontWeight={800}>肘内侧凹槽</text>
            </g>
          )}
          <text x={540} y={818} textAnchor="middle" fill={P.muted} fontSize={22}
            fontFamily={FONT_HEAD}>凹槽里被骨头和床板两头一夹 · 一压一个准</text>
        </g>
        {/* 右：两个作案姿势 */}
        {[0, 1].map((i) => (
          <g key={i} transform={`translate(1010,${150 + i * 360}) scale(${posePop[i]}) translate(-1010,${-(150 + i * 360)})`}
            opacity={clamp01(posePop[i] * 2)}>
            <rect x={1010} y={150 + i * 360} width={770} height={330} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
            <text x={1046} y={206 + i * 360} fill={P.muted} fontSize={24} fontFamily={FONT_HEAD}>
              {i === 0 ? '睡姿一 · 侧睡' : '睡姿二 · 趴睡'}</text>
            {i === 0 ? (
              <Sleeper x={1260} y={392 + i * 360} s={0.85} mode="side" zzz={false} />
            ) : (
              <Sleeper x={1260} y={392 + i * 360} s={0.85} mode="prone" zzz={false} />
            )}
            {/* 受压臂高亮 */}
            {i === 0 && pinchK > 0 && (
              <g opacity={pinchK} transform={`translate(1260,${392 + i * 360}) scale(0.85)`}>
                <path d="M -196 -21 L 60 -21" stroke={P.nerve} strokeWidth={7} strokeLinecap="round" />
                <Spark x={-70} y={-22} s={0.6} />
              </g>
            )}
            {i === 1 && pinchK > 0 && (
              <g opacity={pinchK} transform={`translate(1260,${392 + i * 360}) scale(0.85)`}>
                <path d="M -196 -16 L -30 -16" stroke={P.nerve} strokeWidth={7} strokeLinecap="round" />
                <Spark x={-120} y={-18} s={0.6} />
              </g>
            )}
            <text x={1470} y={306 + i * 360} fill={P.cream} fontSize={30}
              fontFamily={FONT_HEAD} fontWeight={800}>
              {i === 0 ? '整条胳膊外侧' : '前臂垫在头下'}
            </text>
            <text x={1470} y={352 + i * 360} fill={P.muted} fontSize={24} fontFamily={FONT_HEAD}>
              {i === 0 ? '体重全压在桡神经上' : '尺神经在肘弯里顶着'}
            </text>
            <TagPill x={1470} y={424 + i * 360}
              t={i === 0 ? '桡神经 报警' : '尺神经 报警'} col={i === 0 ? P.nerve : P.sig} w={230} fs={26} />
          </g>
        ))}
      </svg>
      <Chip text="麻的位置 = 哪条神经的辖区" x={1170} y={72} at={beat(0) + 25} accent={P.sig} />
      <Stamp text="贴着骨头走浅路" x={640} y={86} at={b1 + 40} color={P.coralD} size={38} rot={-3} />
    </FadeIn>
  );
};

/* ============ S2 麻是乱码不是静音：信号监视器 + 复工顺序 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const monPop = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const glitchK = clamp01((f - 80) / 30);
  const seqK = clamp01((f - b1) / 22);
  const seqPop = [0, 1, 2].map((i) => Math.max(0.001, spring({frame: f - b1 - 10 - i * 14, fps, config: {damping: 12, stiffness: 140}})));

  // 正常波形 vs 乱码波形
  const clean: string[] = [];
  const noise: string[] = [];
  for (let i = 0; i <= 60; i++) {
    const u = i / 60;
    const x = 200 + u * 660;
    clean.push(`${x.toFixed(1)},${(360 - Math.sin(u * Math.PI * 4) * 46).toFixed(1)}`);
    const j = (rand(i, 17) - 0.5) * 120;
    noise.push(`${x.toFixed(1)},${(360 + j).toFixed(1)}`);
  }
  const cn = Math.max(2, Math.floor(60 * clamp01(f / 50)));
  const nn = Math.max(2, Math.floor(60 * glitchK));

  const stations = [
    {t: '先麻', col: P.sig, icon: 'numb', cap: '粗感觉纤维先回'},
    {t: '再刺', col: P.nerve, icon: 'pin', cap: '细痛觉纤维慢半拍'},
    {t: '才正常', col: P.teal, icon: 'ok', cap: '按序复工 · 不是加重'},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={2} name={sec.name} accent={P.teal} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 信号监视器主面板 */}
        <g transform={`translate(120,150) scale(${monPop}) translate(-120,-150)`} opacity={clamp01(monPop * 2)}>
          <rect x={120} y={150} width={1680} height={430} rx={20} fill={P.panel} stroke={P.card} strokeWidth={4} />
          <text x={156} y={206} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            神经信号监视器 · 学名「感觉异常」</text>
          {/* 左半屏：正常 */}
          <rect x={170} y={240} width={720} height={240} rx={16} fill={P.bgDeep} />
          <line x1={200} y1={360} x2={860} y2={360} stroke={P.card} strokeWidth={3} strokeDasharray="8 8" />
          <polyline points={clean.slice(0, cn + 1).join(' ')} fill="none" stroke={P.teal}
            strokeWidth={6} strokeLinecap="round" strokeLinejoin="round" />
          <TagPill x={310} y={292} t="通畅 · 信号整齐" col={P.teal} w={270} fs={24} s={0.9} />
          {/* 大脑接收端 */}
          <g transform="translate(830,300)">
            <circle r={30} fill={P.card} />
            <path d="M -18 -4 a 9 9 0 0 1 18 -2 M -2 -14 a 9 9 0 0 1 18 4 M -16 10 a 10 10 0 0 0 16 6"
              stroke={P.cream} strokeWidth={3.5} fill="none" strokeLinecap="round" />
          </g>
          {/* 右半屏：受压乱码 */}
          <rect x={1030} y={240} width={720} height={240} rx={16} fill={P.bgDeep}
            stroke={glitchK > 0.4 ? P.coral : 'transparent'} strokeWidth={3} />
          <line x1={1060} y1={360} x2={1720} y2={360} stroke={P.card} strokeWidth={3} strokeDasharray="8 8" />
          <g transform="translate(860,0)">
            <polyline points={noise.slice(0, nn + 1).join(' ')} fill="none" stroke={P.coral}
              strokeWidth={5} strokeLinecap="round" strokeLinejoin="round"
              opacity={0.6 + Math.abs(Math.sin(f * 0.4)) * 0.4} />
          </g>
          {glitchK > 0.4 && (
            <g>
              <TagPill x={1170} y={292} t="受压 · 乱放电" col={P.coral} w={240} fs={24} s={0.9} />
              <Spark x={1420} y={360} s={0.7} col={P.coral} />
              {/* 乱码符号 */}
              {['!', '#', '%', '?'].map((t, i) => (
                <text key={i} x={1150 + i * 130} y={452 - (i % 2) * 8} textAnchor="middle"
                  fontSize={30} fill={P.coral} fontFamily={FONT_HEAD} fontWeight={900}
                  opacity={0.5 + Math.abs(Math.sin(f * 0.5 + i)) * 0.5}>{t}</text>
              ))}
            </g>
          )}
          {/* 中央分隔：压迫点 */}
          <g transform="translate(960,360)">
            <rect x={-14} y={-120} width={28} height={240} rx={12} fill={P.card} />
            <Spark x={0} y={-140} s={0.8} />
            <text y={170} textAnchor="middle" fontSize={24} fill={P.cream}
              fontFamily={FONT_HEAD} fontWeight={800}>压住 →</text>
          </g>
          <text x={960} y={556} textAnchor="middle" fill={P.cream} fontSize={30}
            fontFamily={FONT_HEAD} fontWeight={800}>
            不是「没有信号」 是「信号错了」 —— 那根刺刺的感觉就是神经在喊乱码</text>
        </g>
        {/* 复工顺序（cue1） */}
        {seqK > 0 && (
          <g opacity={seqK} transform={`translate(0,${(1 - seqK) * 60})`}>
            <rect x={120} y={640} width={1680} height={220} rx={20} fill={P.panel} stroke={P.teal} strokeWidth={4} />
            <text x={156} y={700} fill={P.teal} fontSize={30} fontFamily={FONT_HEAD} fontWeight={900}>
              松开后的复工直播 · 有固定顺序</text>
            {stations.map((st, i) => (
              <g key={i} transform={`translate(${400 + i * 520},756) scale(${seqPop[i]}) translate(${-(400 + i * 520)},-756)`}
                opacity={clamp01(seqPop[i] * 2)}>
                <rect x={400 + i * 520 - 210} y={706} width={420} height={120} rx={16}
                  fill={P.bgDeep} stroke={st.col} strokeWidth={4} />
                {/* 图标 */}
                <g transform={`translate(${400 + i * 520 - 160},766)`}>
                  {st.icon === 'numb' && (
                    <g>
                      <circle r={26} fill={C.skin} />
                      {[0, 1, 2, 3, 4].map((k) => (
                        <circle key={k} cx={Math.cos(k * 1.26 + f * 0.05) * 40}
                          cy={Math.sin(k * 1.26 + f * 0.05) * 40} r={4.5}
                          fill={P.sig} opacity={0.5 + Math.abs(Math.sin(f * 0.4 + k)) * 0.5} />
                      ))}
                    </g>
                  )}
                  {st.icon === 'pin' && (
                    <g stroke={P.nerve} strokeWidth={5} strokeLinecap="round">
                      {[0, 1, 2].map((k) => {
                        const a = (-60 + k * 60) * Math.PI / 180;
                        return <line key={k} x1={Math.cos(a) * 14} y1={Math.sin(a) * 14}
                          x2={Math.cos(a) * 38} y2={Math.sin(a) * 38} />;
                      })}
                      <circle r={10} fill={P.nerve} stroke="none" />
                    </g>
                  )}
                  {st.icon === 'ok' && <MarkV x={0} y={0} s={2.4} col={P.teal} />}
                </g>
                <text x={400 + i * 520 - 110} y={752} fill={st.col} fontSize={32}
                  fontFamily={FONT_HEAD} fontWeight={900}>{st.t}</text>
                <text x={400 + i * 520 - 110} y={794} fill={P.muted} fontSize={21}
                  fontFamily={FONT_HEAD}>{st.cap}</text>
                {i < 2 && (
                  <g stroke={P.muted} strokeWidth={5} strokeLinecap="round">
                    <line x1={400 + i * 520 + 226} y1={766} x2={400 + i * 520 + 280} y2={766} />
                    <polygon points={`${400 + i * 520 + 292},766 ${400 + i * 520 + 274},756 ${400 + i * 520 + 274},776`}
                      fill={P.muted} stroke="none" />
                  </g>
                )}
              </g>
            ))}
          </g>
        )}
      </svg>
      <Chip text="刺刺的 = 乱码信号" x={1280} y={72} at={beat(0) + 30} accent={P.teal} />
      <Stamp text="感觉异常 · 不是没感觉" x={640} y={98} at={55} color={P.coralD} size={38} rot={3} />
    </FadeIn>
  );
};

/* ============ S3 压几分钟 vs 压一整夜：星期六夜麻痹 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1), b2 = beat(2);
  const popL = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const popR = Math.max(0.001, spring({frame: f - b1 - 5, fps, config: {damping: 11, stiffness: 120}}));
  const dropK = clamp01((f - b1 - 55) / 26);
  const lineK = clamp01((f - b2) / 22);
  const wristRot = interpolate(dropK, [0, 1], [0, 62]);

  return (
    <FadeIn>
      <Backdrop top={P.bgDeep} bottom={P.bgDeep} />
      <SceneTag index={3} name={sec.name} accent={P.coral} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={8} from={0} dur={26} />
        {/* 左：压几分钟 */}
        <g transform={`translate(120,155) scale(${popL}) translate(-120,-155)`} opacity={clamp01(popL * 2)}>
          <rect x={120} y={155} width={760} height={470} rx={20} fill={P.panel} stroke={P.teal} strokeWidth={4} />
          <text x={156} y={210} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>压几分钟 · 暂时断线</text>
          {/* 小时钟 */}
          <g transform="translate(310,330)">
            <circle r={74} fill={P.bgDeep} stroke={P.teal} strokeWidth={7} />
            {[0, 1, 2, 3].map((i) => (
              <line key={i} x1={0} y1={-64} x2={0} y2={-52} stroke={P.muted} strokeWidth={5}
                transform={`rotate(${i * 90})`} strokeLinecap="round" />
            ))}
            <line x1={0} y1={0} x2={0} y2={-42} stroke={P.cream} strokeWidth={7} strokeLinecap="round" />
            <line x1={0} y1={0} x2={Math.cos(f * 0.12) * 34} y2={Math.sin(f * 0.12) * 34}
              stroke={P.teal} strokeWidth={6} strokeLinecap="round" />
            <text y={104} textAnchor="middle" fontSize={24} fill={P.teal}
              fontFamily={FONT_HEAD} fontWeight={800}>几分钟</text>
          </g>
          {/* 神经弹回来 */}
          <g transform="translate(600,330)">
            <path d="M -110 0 C -60 -34 -30 -34 0 0 C 30 34 60 34 110 0"
              fill="none" stroke={P.nerve} strokeWidth={11} strokeLinecap="round"
              transform={`scale(1 ${0.7 + Math.abs(Math.sin(f * 0.25)) * 0.5})`} />
            <SignalDots pressed={0} />
            <text y={104} textAnchor="middle" fontSize={24} fill={P.nerve}
              fontFamily={FONT_HEAD} fontWeight={800}>一松就弹回来</text>
          </g>
          <MarkV x={795} y={250} s={1.4} />
          <rect x={200} y={500} width={600} height={66} rx={18} fill={P.teal} />
          <text x={500} y={545} textAnchor="middle" fontSize={30} fill={P.ink}
            fontFamily={FONT_HEAD} fontWeight={900}>传导阻滞 · 缓一缓就好</text>
        </g>
        {/* 右：压一整夜 → 星期六夜麻痹 */}
        <g transform={`translate(1040,155) scale(${popR}) translate(-1040,-155)`} opacity={clamp01(popR * 2)}>
          <rect x={1040} y={155} width={760} height={470} rx={20} fill={P.panel}
            stroke={popR > 0.5 ? P.coral : P.card} strokeWidth={4} />
          <text x={1076} y={210} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            压一整夜 · 真受伤</text>
          <Moon x={1740} y={225} scale={0.35} opacity={0.9} />
          {/* 酒瓶 */}
          <g transform="translate(1150,330) rotate(-14)">
            <rect x={-16} y={-86} width={32} height={40} rx={8} fill={P.card} />
            <path d="M -24 -46 L 24 -46 L 30 44 A 12 10 0 0 1 -30 44 Z" fill={P.coralD} opacity={0.9} />
            <rect x={-24} y={-6} width={48} height={22} rx={6} fill={P.cream} opacity={0.85} />
            <text x={0} y={96} textAnchor="middle" fontSize={22} fill={P.muted}
              fontFamily={FONT_HEAD} fontWeight={700}>喝醉昏睡</text>
          </g>
          {/* 垂腕手臂 */}
          <g transform="translate(1430,296)">
            {/* 上臂 */}
            <rect x={-170} y={-26} width={240} height={52} rx={26} fill={C.skin} />
            {/* 前臂下垂 */}
            <g transform={`translate(70,0) rotate(${wristRot})`}>
              <rect x={0} y={-22} width={150} height={44} rx={22} fill={C.skin} />
              <circle cx={168} cy={0} r={26} fill={C.skin} />
              {dropK > 0.5 && [0, 1, 2].map((i) => (
                <line key={i} x1={60 + i * 30} y1={34} x2={60 + i * 30} y2={52}
                  stroke={P.coral} strokeWidth={4} strokeLinecap="round" opacity={dropK} />
              ))}
            </g>
            {/* 腋窝压迫点 */}
            <circle cx={-140} cy={0} r={20} fill="none" stroke={P.coral} strokeWidth={5}
              strokeDasharray="7 6" transform={`rotate(${f} -140 0)`} />
            <text x={-20} y={104} textAnchor="middle" fontSize={24} fill={P.coral}
              fontFamily={FONT_HEAD} fontWeight={800}>腋窝压着桡神经睡到天亮</text>
          </g>
          {dropK > 0 && (
            <g opacity={dropK}>
              <rect x={1120} y={452} width={300} height={56} rx={16} fill={P.coral} />
              <text x={1270} y={490} textAnchor="middle" fontSize={26} fill={P.cream}
                fontFamily={FONT_HEAD} fontWeight={900}>垂腕 · 抬不起来</text>
            </g>
          )}
          {dropK > 0.4 && (
            <g opacity={clamp01((dropK - 0.4) * 2)}>
              <rect x={1480} y={452} width={280} height={56} rx={16} fill={P.bgDeep} stroke={P.coral} strokeWidth={3.5} />
              <text x={1620} y={490} textAnchor="middle" fontSize={24} fill={P.coral}
                fontFamily={FONT_HEAD} fontWeight={800}>几周到几个月长回来</text>
            </g>
          )}
          <TagPill x={1440} y={560} t="星期六夜麻痹" col={P.coral} w={300} fs={30} />
        </g>
        {/* 中央对比 */}
        <g transform="translate(960,390)">
          <circle r={40} fill={P.coral} />
          <text y={12} textAnchor="middle" fontSize={30} fill={P.cream}
            fontFamily={FONT_HEAD} fontWeight={900}>VS</text>
        </g>
        {/* 底部分界线（cue2） */}
        {lineK > 0 && (
          <g opacity={lineK} transform={`translate(0,${(1 - lineK) * 60})`}>
            <rect x={120} y={668} width={1680} height={190} rx={20} fill={P.panel} stroke={P.coral} strokeWidth={4} />
            <text x={160} y={730} fill={P.coral} fontSize={30} fontFamily={FONT_HEAD} fontWeight={900}>
              生理性和病理性的分界线</text>
            <text x={160} y={788} fill={P.cream} fontSize={29} fontFamily={FONT_HEAD} fontWeight={700}>
              不在麻得多狠 · 在压了多久 + 有没有抬不起来的运动丧失</text>
            <g transform="translate(1620,764)">
              <rect x={-150} y={-34} width={300} height={68} rx={16} fill={P.bgDeep} stroke={P.nerve} strokeWidth={3.5} />
              <text x={0} y={-6} textAnchor="middle" fontSize={22} fill={P.muted} fontFamily={FONT_HEAD}>醒来麻超半小时</text>
              <text x={0} y={24} textAnchor="middle" fontSize={24} fill={P.nerve}
                fontFamily={FONT_HEAD} fontWeight={900}>别拖 · 去看医生</text>
            </g>
          </g>
        )}
      </svg>
      <Chip text="压多久才是关键" x={1330} y={72} at={beat(0) + 30} accent={P.coral} />
      <Stamp text="几秒就好 ≠ 一直不好" x={660} y={86} at={b2 + 40} color={P.coralD} size={38} rot={-3} />
    </FadeIn>
  );
};

/* ============ S4 睡着后没人帮你翻身：保护下线 + 自测三招 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b1 = beat(1);
  const popL = Math.max(0.001, spring({frame: f - 10, fps, config: {damping: 11, stiffness: 120}}));
  const popR = Math.max(0.001, spring({frame: f - 45, fps, config: {damping: 11, stiffness: 120}}));
  const listK = clamp01((f - b1) / 22);
  const listPop = [0, 1, 2].map((i) => Math.max(0.001, spring({frame: f - b1 - 10 - i * 12, fps, config: {damping: 12, stiffness: 140}})));
  const ringA = f * 1.2;

  const checks = [
    {q: '麻在哪?', a: '整臂手背 → 压的\n只麻拇食中指 → 腕管', col: P.sig},
    {q: '夜夜都麻?', a: '偶尔压久 → 姿势锅\n每晚准时 → 要查', col: P.nerve},
    {q: '多久能缓?', a: '几分钟就好 → 放心\n超半小时+无力 → 就医', col: P.coral},
  ];

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={4} name={sec.name} accent={P.nerve} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 左：醒着 · 保护环路在岗 */}
        <g transform={`translate(120,155) scale(${popL}) translate(-120,-155)`} opacity={clamp01(popL * 2)}>
          <rect x={120} y={155} width={800} height={410} rx={20} fill={P.panel} stroke={P.teal} strokeWidth={4} />
          <text x={156} y={210} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            醒着 · 保安在岗</text>
          {/* 反馈环路 */}
          <g transform="translate(520,370)">
            <circle r={108} fill="none" stroke={P.card} strokeWidth={5} strokeDasharray="10 10" />
            <g transform={`rotate(${ringA})`}>
              <polygon points="0,-118 14,-96 -14,-96" fill={P.teal} />
            </g>
            {/* 三个节点 */}
            <g transform="translate(0,-108)">
              <circle r={30} fill={P.bgDeep} stroke={P.coral} strokeWidth={4} />
              <Spark x={0} y={0} s={0.45} col={P.coral} spin={-1} />
              <text y={-46} textAnchor="middle" fontSize={20} fill={P.coral}
                fontFamily={FONT_HEAD} fontWeight={800}>压到</text>
            </g>
            <g transform="translate(94,54)">
              <circle r={30} fill={P.bgDeep} stroke={P.sig} strokeWidth={4} />
              <path d="M -12 -2 a 7 7 0 0 1 12 -2 M -2 -10 a 7 7 0 0 1 13 4 M -11 8 a 8 8 0 0 0 11 4"
                stroke={P.sig} strokeWidth={3} fill="none" strokeLinecap="round" />
              <text x={16} y={48} textAnchor="middle" fontSize={20} fill={P.sig}
                fontFamily={FONT_HEAD} fontWeight={800}>大脑接警</text>
            </g>
            <g transform="translate(-94,54)">
              <circle r={30} fill={P.bgDeep} stroke={P.teal} strokeWidth={4} />
              <path d="M -10 4 q 10 -14 20 0" stroke={P.teal} strokeWidth={4} fill="none" strokeLinecap="round" />
              <text x={-8} y={48} textAnchor="middle" fontSize={20} fill={P.teal}
                fontFamily={FONT_HEAD} fontWeight={800}>挪一挪</text>
            </g>
            <text y={14} textAnchor="middle" fontSize={24} fill={P.cream}
              fontFamily={FONT_HEAD} fontWeight={800}>本能换姿势</text>
          </g>
          <text x={160} y={528} fill={P.teal} fontSize={26} fontFamily={FONT_HEAD} fontWeight={700}>
            压几分钟就自动调整 · 神经有人管</text>
        </g>
        {/* 右：睡着 · 保安下班 */}
        <g transform={`translate(1000,155) scale(${popR}) translate(-1000,-155)`} opacity={clamp01(popR * 2)}>
          <rect x={1000} y={155} width={800} height={410} rx={20} fill={P.panel} stroke={P.coral} strokeWidth={4} />
          <text x={1036} y={210} fill={P.muted} fontSize={25} fontFamily={FONT_HEAD}>
            睡着 · 保安下班了</text>
          {/* 断掉的环路 */}
          <g transform="translate(1250,370)" opacity={0.9}>
            <circle r={96} fill="none" stroke={P.card} strokeWidth={5} strokeDasharray="10 10" opacity={0.6} />
            <MarkX x={0} y={0} s={2.2} />
            <text y={140} textAnchor="middle" fontSize={23} fill={P.coral}
              fontFamily={FONT_HEAD} fontWeight={800}>反馈环路断线</text>
          </g>
          {/* 睡姿不变的小人 */}
          <Sleeper x={1560} y={440} s={0.72} mode="side" zzz={true} />
          {/* 酒 + 安眠药小标 */}
          <g transform="translate(1560,232)">
            <rect x={-190} y={-34} width={180} height={56} rx={14} fill={P.bgDeep} stroke={P.coral} strokeWidth={3.5} />
            <text x={-100} y={4} textAnchor="middle" fontSize={22} fill={P.coral}
              fontFamily={FONT_HEAD} fontWeight={800}>酒</text>
            <rect x={0} y={-34} width={240} height={56} rx={14} fill={P.bgDeep} stroke={P.coral} strokeWidth={3.5} />
            <text x={120} y={4} textAnchor="middle" fontSize={22} fill={P.coral}
              fontFamily={FONT_HEAD} fontWeight={800}>安眠药 / 太累</text>
            <text x={25} y={56} textAnchor="middle" fontSize={21} fill={P.muted}
              fontFamily={FONT_HEAD}>翻得更少 · 压得更死</text>
          </g>
        </g>
        {/* 自测三招（cue1） */}
        {listK > 0 && (
          <g opacity={listK} transform={`translate(0,${(1 - listK) * 60})`}>
            <text x={140} y={668} fill={P.nerve} fontSize={30} fontFamily={FONT_HEAD} fontWeight={900}>
              分不清要不要紧 · 自测三招</text>
            {checks.map((c, i) => (
              <g key={i} transform={`translate(${140 + i * 566},694) scale(${listPop[i]}) translate(${-(140 + i * 566)},-694)`}
                opacity={clamp01(listPop[i] * 2)}>
                <rect x={140 + i * 566} y={694} width={536} height={180} rx={16}
                  fill={P.panel} stroke={c.col} strokeWidth={4} />
                <circle cx={140 + i * 566 + 46} cy={740} r={24} fill={c.col} />
                <text x={140 + i * 566 + 46} y={750} textAnchor="middle" fontSize={26}
                  fill={P.ink} fontFamily={FONT_HEAD} fontWeight={900}>{i + 1}</text>
                <text x={140 + i * 566 + 88} y={750} fill={c.col} fontSize={30}
                  fontFamily={FONT_HEAD} fontWeight={900}>{c.q}</text>
                {c.a.split('\n').map((line, k) => (
                  <text key={k} x={140 + i * 566 + 34} y={806 + k * 36} fill={P.cream}
                    fontSize={23} fontFamily={FONT_HEAD}>{line}</text>
                ))}
              </g>
            ))}
          </g>
        )}
      </svg>
      <Chip text="保护反射睡着就下班" x={1250} y={72} at={beat(0) + 25} accent={P.nerve} />
      <Stamp text="压麻换姿势 · 慢麻看医生" x={560} y={86} at={b1 + 60} color={P.coralD} size={36} rot={-2} />
    </FadeIn>
  );
};

/* ============ 结尾卡 ============ */
const EndCard = () => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 14}});
  const pop = Math.max(0.001, spring({frame: f - 5, fps, config: {damping: 10, stiffness: 140}}));
  return (
    <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
      <Backdrop top={P.bgDeep} bottom={P.bgDeep} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <Stars n={20} from={0} dur={22} />
        <Moon x={330} y={220} scale={0.8} />
        {/* 手臂不再受压 · 安稳睡 */}
        <g transform={`translate(960,790) scale(${pop})`} opacity={clamp01(pop)}>
          <Sleeper x={0} y={0} s={1.05} mode="free" zzz={false} />
          <Zzz x={140} y={-140} k={0.9} />
        </g>
        {/* 神经畅通小牌 */}
        <g transform={`translate(1470,740) scale(${pop})`} opacity={clamp01(pop)}>
          <rect x={-170} y={-52} width={340} height={104} rx={18} fill={P.panel} stroke={P.teal} strokeWidth={4} />
          <path d="M -140 0 C -100 -18 -60 -18 -20 0 C 20 18 60 18 100 0" fill="none"
            stroke={P.nerve} strokeWidth={7} strokeLinecap="round" />
          <text x={0} y={40} textAnchor="middle" fontSize={24} fill={P.teal}
            fontFamily={FONT_HEAD} fontWeight={800}>神经畅通</text>
        </g>
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -300}}>
        <div style={{fontSize: 54, fontWeight: 800, color: C.text}}>麻是神经在抗议 · 听懂它的语言</div>
        <div style={{fontSize: 36, color: P.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E235 = ({data = tl}: {data?: any}) => {
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
