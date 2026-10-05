import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_HEAD, clamp01, rand} from '../lib/style';
import {Backdrop, Ripple, Stamp, SceneTag, Chip, Subtitles, ProgressBar, FadeIn} from '../lib/atoms';
import tl from '../timeline_216.json';

type Sec = {name: string; keywords: string[]; audio: string; start: number; dur: number; cues: {t0: number; t1: number; text: string}[]};

/* ============ 本期色板：深青蓝底 × 薄荷噬菌体 × 橙杆菌 × 琥珀DNA × 珊瑚病基因 ============ */
const P = {
  bg: '#243447',
  bgDeep: '#1b2735',
  bgLite: '#2b3f56',
  card: '#31455c',
  cream: '#f0e9d6',
  ink: '#121a26',
  phage: '#7fe3c8',    // 噬菌体：薄荷绿
  phageD: '#3fae8f',
  phageL: '#b8f2e0',
  dna: '#ffc94d',      // DNA：琥珀
  dnaD: '#c98f1f',
  bac: '#f0a35e',      // 细菌：暖橙
  bacD: '#bd7437',
  bacL: '#ffcf9e',
  seg: '#ff6b6b',      // 潜伏基因 / 目标菌：珊瑚红
  segD: '#c94f4f',
  sky: '#6cc9ff',
  grey: '#93a2b4',     // 阵亡菌
  enzyme: '#8fb8ff',   // 内切酶
  water: '#2f8291',
};

/* ---------- 噬菌体：着陆器造型，头朝下 ---------- */
const Phage = ({x, y, s = 1, rot = 0, op = 1, dnaK = 1, sheathK = 1, col = P.phage}: {
  x: number; y: number; s?: number; rot?: number; op?: number; dnaK?: number; sheathK?: number; col?: string;
}) => {
  const sheathH = 12 + 44 * sheathK;
  const tubeH = 46 + (1 - sheathK) * 52;
  return (
    <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`} opacity={op}>
      {/* 尾丝 6 条 */}
      <g stroke={P.phageD} strokeWidth={7} strokeLinecap="round">
        <line x1={-16} y1={104} x2={-60} y2={164} />
        <line x1={16} y1={104} x2={60} y2={164} />
        <line x1={-9} y1={108} x2={-26} y2={172} />
        <line x1={9} y1={108} x2={26} y2={172} />
        <line x1={-18} y1={100} x2={-88} y2={146} />
        <line x1={18} y1={100} x2={88} y2={146} />
      </g>
      {/* 尾管（鞘收缩后戳出） */}
      <rect x={-7} y={64} width={14} height={tubeH} fill={P.phageD} />
      {/* 尾鞘 */}
      <rect x={-21} y={64} width={42} height={sheathH} rx={10} fill={col} stroke={P.phageD} strokeWidth={5} />
      <line x1={-21} y1={64 + sheathH * 0.5} x2={21} y2={64 + sheathH * 0.5} stroke={P.phageD} strokeWidth={4} opacity={0.6} />
      {/* 底板 */}
      <ellipse cx={0} cy={114} rx={30} ry={9} fill={P.phageD} />
      {/* 领圈 */}
      <rect x={-24} y={46} width={48} height={20} rx={9} fill={P.phageD} />
      {/* 二十面体头壳 */}
      <polygon points="0,-62 54,-30 54,22 0,52 -54,22 -54,-30" fill={col} stroke={P.phageD} strokeWidth={6} strokeLinejoin="round" />
      <polygon points="0,-62 54,-30 0,-4 -54,-30" fill={P.phageL} opacity={0.55} />
      <line x1={0} y1={-62} x2={0} y2={-4} stroke={P.phageD} strokeWidth={3} opacity={0.5} />
      {/* 头壳里的DNA */}
      <path d="M -26 -22 Q -8 -36 12 -22 Q 28 -10 10 4 Q -8 16 -26 4" fill="none"
        stroke={P.dna} strokeWidth={8} strokeLinecap="round" opacity={dnaK} />
    </g>
  );
};

/* ---------- 迷你噬菌体：群众演员 ---------- */
const MiniPhage = ({x, y, s = 1, rot = 0, op = 1}: {x: number; y: number; s?: number; rot?: number; op?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`} opacity={op}>
    <polygon points="0,-17 15,-8 15,7 0,16 -15,7 -15,-8" fill={P.phage} stroke={P.phageD} strokeWidth={3} />
    <rect x={-3} y={14} width={6} height={17} fill={P.phageD} />
    <line x1={-3} y1={30} x2={-13} y2={40} stroke={P.phageD} strokeWidth={3} strokeLinecap="round" />
    <line x1={3} y1={30} x2={13} y2={40} stroke={P.phageD} strokeWidth={3} strokeLinecap="round" />
  </g>
);

/* ---------- 杆菌：圆头长条 + 菌毛 + 鞭毛 + 染色体环 + 表情 ---------- */
const Bac = ({x, y, s = 1, rot = 0, op = 1, col = P.bac, face = 'ok', chromo = 1, seg = 0, cracked = 0}: {
  x: number; y: number; s?: number; rot?: number; op?: number; col?: string;
  face?: 'ok' | 'sick' | 'dead'; chromo?: number; seg?: number; cracked?: number;
}) => {
  const edge = face === 'dead' ? '#7c8a9c' : P.bacD;
  return (
    <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`} opacity={op}>
      {/* 鞭毛 */}
      <path d="M -156 8 Q -194 -26 -228 2 Q -254 24 -282 -4" fill="none" stroke={edge} strokeWidth={10} strokeLinecap="round" />
      {/* 菌毛 */}
      {[-120, -72, -24, 24, 72, 120].map((px, i) => (
        <line key={'t' + i} x1={px} y1={-86} x2={px + (i % 2 ? -8 : 8)} y2={-110} stroke={edge} strokeWidth={7} strokeLinecap="round" />
      ))}
      {[-96, -48, 0, 48, 96].map((px, i) => (
        <line key={'b' + i} x1={px} y1={86} x2={px + (i % 2 ? 7 : -7)} y2={108} stroke={edge} strokeWidth={7} strokeLinecap="round" />
      ))}
      {/* 菌体 */}
      <rect x={-160} y={-90} width={320} height={180} rx={90} fill={col} stroke={edge} strokeWidth={8} />
      <rect x={-140} y={-72} width={280} height={76} rx={38} fill={P.bacL} opacity={0.32} />
      {/* 染色体环（可嵌病毒基因） */}
      {chromo > 0 && (
        <g opacity={chromo}>
          <path d="M -62 24 Q -62 -22 -20 -24 Q 22 -24 22 8 Q 22 36 -14 38 Q -60 40 -62 24 Z"
            fill="none" stroke={P.dna} strokeWidth={9} />
          {seg > 0 && <path d="M -20 -24 Q 22 -24 22 8" fill="none" stroke={P.seg}
            strokeWidth={13} strokeLinecap="round" opacity={seg} />}
        </g>
      )}
      {/* 核糖体点 */}
      {[[-92, -42], [-34, 52], [42, -52], [84, 34], [-84, 48]].map(([rx, ry], i) => (
        <circle key={i} cx={rx} cy={ry} r={9} fill={edge} opacity={0.45} />
      ))}
      {/* 裂纹 */}
      {cracked > 0 && (
        <g opacity={cracked} stroke={P.ink} strokeWidth={7} fill="none" strokeLinecap="round">
          <path d="M -50 -86 L -26 -46 L -52 -12 L -30 24" />
          <path d="M 30 -88 L 48 -50 L 26 -18 L 44 18" />
        </g>
      )}
      {/* 表情（菌头朝右） */}
      {face === 'ok' && (
        <g>
          <circle cx={58} cy={-30} r={17} fill="#fff" />
          <circle cx={106} cy={-30} r={17} fill="#fff" />
          <circle cx={62} cy={-28} r={7} fill={P.ink} />
          <circle cx={110} cy={-28} r={7} fill={P.ink} />
          <path d="M 66 18 Q 84 34 102 16" stroke={P.ink} strokeWidth={7} fill="none" strokeLinecap="round" />
        </g>
      )}
      {face === 'sick' && (
        <g>
          <path d="M 42 -36 Q 58 -48 74 -34" stroke={P.ink} strokeWidth={7} fill="none" strokeLinecap="round" />
          <path d="M 92 -36 Q 108 -48 124 -34" stroke={P.ink} strokeWidth={7} fill="none" strokeLinecap="round" />
          <path d="M 64 26 Q 84 12 104 26" stroke={P.ink} strokeWidth={7} fill="none" strokeLinecap="round" />
          <path d="M 138 -66 Q 148 -48 138 -40 Q 128 -48 138 -66 Z" fill={P.sky} />
        </g>
      )}
      {face === 'dead' && (
        <g stroke={P.ink} strokeWidth={7} strokeLinecap="round">
          <line x1={48} y1={-40} x2={70} y2={-18} />
          <line x1={70} y1={-40} x2={48} y2={-18} />
          <line x1={94} y1={-40} x2={116} y2={-18} />
          <line x1={116} y1={-40} x2={94} y2={-18} />
          <line x1={66} y1={26} x2={104} y2={26} />
        </g>
      )}
    </g>
  );
};

/* ---------- 迷你细菌：群众演员 ---------- */
const MiniBac = ({x, y, s = 1, rot = 0, op = 1, col = P.bac, face = 'ok'}: {
  x: number; y: number; s?: number; rot?: number; op?: number; col?: string; face?: 'ok' | 'dead';
}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`} opacity={op}>
    <line x1={-56} y1={4} x2={-72} y2={-8} stroke={face === 'dead' ? '#7c8a9c' : P.bacD} strokeWidth={6} strokeLinecap="round" />
    <rect x={-56} y={-28} width={112} height={56} rx={28} fill={col}
      stroke={face === 'dead' ? '#7c8a9c' : P.bacD} strokeWidth={5} />
    {face === 'ok' ? (
      <g>
        <circle cx={14} cy={-8} r={7} fill="#fff" />
        <circle cx={34} cy={-8} r={7} fill="#fff" />
        <circle cx={15} cy={-7} r={3} fill={P.ink} />
        <circle cx={35} cy={-7} r={3} fill={P.ink} />
      </g>
    ) : (
      <g stroke={P.ink} strokeWidth={4} strokeLinecap="round">
        <line x1={8} y1={-14} x2={20} y2={-2} /><line x1={20} y1={-14} x2={8} y2={-2} />
        <line x1={30} y1={-14} x2={42} y2={-2} /><line x1={42} y1={-14} x2={30} y2={-2} />
      </g>
    )}
  </g>
);

/* ---------- 游离DNA片段 ---------- */
const DnaBit = ({x, y, s = 1, rot = 0, op = 1, col = P.dna}: {x: number; y: number; s?: number; rot?: number; op?: number; col?: string}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`} opacity={op}>
    <path d="M -42 0 Q -21 -22 0 0 Q 21 22 42 0" fill="none" stroke={col} strokeWidth={9} strokeLinecap="round" />
    <path d="M -42 11 Q -21 -11 0 11 Q 21 33 42 11" fill="none" stroke={col} strokeWidth={5} opacity={0.55} />
  </g>
);

/* ---------- 内切酶：吃豆人剪刀嘴 ---------- */
const Enzyme = ({x, y, s = 1, rot = 0, op = 1, chomp = 1}: {x: number; y: number; s?: number; rot?: number; op?: number; chomp?: number}) => {
  const f = useCurrentFrame();
  const a = (chomp ? (0.35 + Math.abs(Math.sin(f / 4)) * 0.55) : 0.3) * Math.PI / 4;
  const r = 40;
  return (
    <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`} opacity={op}>
      <path d={`M 0 0 L ${Math.cos(-a) * r} ${Math.sin(-a) * r} A ${r} ${r} 0 1 0 ${Math.cos(a) * r} ${Math.sin(a) * r} Z`}
        fill={P.enzyme} stroke="#5d84c9" strokeWidth={5} />
      <circle cx={-8} cy={-16} r={6} fill={P.ink} />
    </g>
  );
};

/* ---------- CRISPR 档案卡 ---------- */
const FileCard = ({x, y, rot = 0, s = 1, op = 1, col = P.dna, hit = 0}: {x: number; y: number; rot?: number; s?: number; op?: number; col?: string; hit?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`} opacity={op}>
    <rect x={-48} y={-58} width={96} height={116} rx={14} fill={P.cream} stroke={hit ? P.seg : P.card} strokeWidth={hit ? 7 : 4} />
    <rect x={-48} y={-58} width={96} height={26} rx={13} fill={col} opacity={0.85} />
    <DnaBit x={0} y={12} s={0.85} col={col} />
    {hit > 0 && (
      <g transform="translate(30,-46) scale(0.9)" opacity={hit}>
        <circle r={16} fill={P.seg} />
        <path d="M -6 0 L -1 6 L 8 -6" stroke="#fff" strokeWidth={5} fill="none" strokeLinecap="round" />
      </g>
    )}
  </g>
);

/* ---------- 闪电 ---------- */
const Bolt = ({x, y, s = 1, op = 1}: {x: number; y: number; s?: number; op?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
    <polygon points="8,-70 -26,6 -2,6 -14,66 26,-16 2,-16 20,-70" fill={P.dna} stroke={P.dnaD} strokeWidth={4} strokeLinejoin="round" />
  </g>
);

/* ---------- 奖章 ---------- */
const Medal = ({x, y, s = 1, op = 1}: {x: number; y: number; s?: number; op?: number}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={op}>
    <polygon points="-30,-86 -10,-40 10,-40 30,-86 12,-86 0,-56 -12,-86" fill={P.seg} />
    <polygon points="-30,-86 -12,-86 0,-56 -14,-40 -30,-60" fill="#4f7fd9" opacity={0.9} />
    <circle r={44} fill={P.dna} stroke={P.dnaD} strokeWidth={6} />
    <circle r={32} fill="none" stroke={P.dnaD} strokeWidth={4} opacity={0.6} />
    <polygon points="0,-18 5,-6 18,-6 8,3 12,16 0,9 -12,16 -8,3 -18,-6 -5,-6" fill={P.dnaD} />
  </g>
);

/* ---------- 药瓶 ---------- */
const Vial = ({x, y, s = 1, rot = 0, op = 1}: {x: number; y: number; s?: number; rot?: number; op?: number}) => {
  const f = useCurrentFrame();
  return (
    <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`} opacity={op}>
      <rect x={-56} y={-110} width={112} height={170} rx={24} fill="rgba(207,232,242,0.25)" stroke="#cfe4f0" strokeWidth={7} />
      <path d="M -48 -20 L 48 -20 L 48 38 Q 48 52 34 52 L -34 52 Q -48 52 -48 38 Z" fill={P.water} />
      <ellipse cx={0} cy={-20} rx={48} ry={8} fill="#3b9aa9" />
      {[0, 1, 2].map((i) => (
        <MiniPhage key={i} x={-26 + i * 26} y={14 + Math.sin(f / 8 + i * 2) * 5} s={0.55} rot={i * 40 - 20} />
      ))}
      <rect x={-34} y={-142} width={68} height={34} rx={9} fill={P.dnaD} />
      <rect x={-22} y={-112} width={44} height={10} fill={P.dnaD} opacity={0.7} />
      <rect x={-40} y={-96} width={80} height={40} rx={8} fill={P.cream} />
      <text x={0} y={-68} textAnchor="middle" fill={P.ink} fontSize={26} fontFamily={FONT_HEAD} fontWeight={700}>噬菌体</text>
    </g>
  );
};

/* ---------- 胶囊（抗生素） ---------- */
const Capsule = ({x, y, s = 1, rot = 0, op = 1}: {x: number; y: number; s?: number; rot?: number; op?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`} opacity={op}>
    <g transform="rotate(90)">
      <path d="M -34 -78 A 34 34 0 0 1 34 -78 L 34 0 L -34 0 Z" fill={P.seg} />
      <path d="M -34 0 L 34 0 L 34 78 A 34 34 0 0 1 -34 78 Z" fill={P.cream} />
      <rect x={-34} y={-4} width={68} height={8} fill={P.ink} opacity={0.25} />
      <rect x={-20} y={-70} width={12} height={34} rx={6} fill="#fff" opacity={0.4} />
    </g>
  </g>
);

/* ---------- 步骤签 ---------- */
const StepPill = ({n, text, x, y, lit}: {n: string; text: string; x: number; y: number; lit: boolean}) => (
  <div style={{
    position: 'absolute', left: x, top: y, padding: '10px 26px', borderRadius: 999,
    border: `3px solid ${P.phage}`, background: lit ? P.phage : 'rgba(18,26,38,0.72)',
    color: lit ? P.ink : C.text, fontSize: 30, fontFamily: FONT_HEAD, letterSpacing: 3,
    fontWeight: lit ? 800 : 500, whiteSpace: 'nowrap',
  }}>
    {`${n} ${text}`}
  </div>
);

/* ============ 标题卡 ============ */
const TitleCard = ({data}: {data: any}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f, fps, config: {damping: 13, mass: 0.7}});
  const o = interpolate(f, [0, 8], [0, 1]);
  const p0 = Math.max(0.001, spring({frame: f - 4, fps, config: {damping: 10, stiffness: 150}}));
  const p1 = Math.max(0.001, spring({frame: f - 12, fps, config: {damping: 10, stiffness: 150}}));
  const bob = Math.sin(f / 16) * 8;
  return (
    <AbsoluteFill>
      <Backdrop top={P.bg} bottom={P.bg} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <circle cx={960} cy={352} r={218} fill={P.cream} />
        <circle cx={960} cy={352} r={196} fill={P.bgLite} />
        <g transform={`translate(960,${322 + bob}) scale(${p0})`}>
          <Phage x={0} y={0} s={1.02} rot={-7} />
        </g>
        <g transform={`translate(756,${430 - bob * 0.7}) scale(${p1})`}>
          <MiniBac x={0} y={0} s={0.92} rot={-14} />
        </g>
        <g transform={`translate(1178,${268 + bob}) scale(${p1})`}>
          <MiniPhage x={0} y={0} s={0.8} rot={26} />
        </g>
        <g transform={`translate(1150,${452 - bob * 0.6}) scale(${p1})`}>
          <DnaBit x={0} y={0} s={0.6} rot={-16} />
        </g>
      </svg>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', marginTop: 270}}>
          <div style={{color: P.phage, fontSize: 36, fontFamily: FONT_HEAD, letterSpacing: 14, marginBottom: 26}}>
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

/* ============ S0 病毒猎人：海水烧杯 + 噬菌体落到细菌背上 ============ */
const S0 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b0 = beat(0), b1 = beat(1), b2 = beat(2);

  const landT = b1 + 55;                                     // 落到菌背
  const phageY = interpolate(f, [b0 + 25, landT], [-160, 300], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const phageX = 1245 - clamp01((f - b0 - 25) / (landT - b0 - 25)) * 60;
  const docked = f >= landT;
  const bobY = docked ? Math.sin(f / 10) * 3 : 0;
  const sickK = clamp01((f - landT - 16) / 20);
  const pop = Math.max(0.001, spring({frame: f - 6, fps, config: {damping: 11, stiffness: 130}}));
  const cupK = clamp01((f - b2 + 30) / 25);                  // 烧杯高亮
  const countK = clamp01((f - b2 - 8) / 20);

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={0} name={sec.name} accent={P.phage} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 海水里漂的零星噬菌体 */}
        {[0, 1, 2, 3, 4, 5, 6, 7].map((i) => (
          <MiniPhage key={i}
            x={rand(i, 31) * 1720 + 100 + Math.sin(f / 34 + i * 1.9) * 26}
            y={rand(i, 32) * 460 + 180 + Math.cos(f / 40 + i * 2.3) * 14}
            s={0.5 + rand(i, 33) * 0.4} rot={rand(i, 34) * 70 - 35} op={0.35} />
        ))}

        {/* 左：一升海水烧杯 */}
        <ellipse cx={390} cy={845} rx={210} ry={28} fill={P.bgDeep} />
        <g transform={`translate(390,620)`}>
          <path d="M -128 -185 L -128 140 Q -128 172 -96 172 L 96 172 Q 128 172 128 140 L 128 -185"
            fill="rgba(207,232,242,0.12)" stroke="#cfe4f0" strokeWidth={8} strokeLinecap="round" />
          <path d="M -118 -80 L 118 -80 L 118 138 Q 118 162 94 162 L -94 162 Q -118 162 -118 138 Z"
            fill={P.water} opacity={cupK > 0 ? 0.95 : 0.8} />
          <ellipse cx={0} cy={-80} rx={118} ry={12} fill="#3b9aa9" />
          {Array.from({length: 11}).map((_, i) => (
            <MiniPhage key={i}
              x={-88 + rand(i, 41) * 176}
              y={-50 + rand(i, 42) * 190 + Math.sin(f / 9 + i * 2.2) * 7}
              s={0.5 + rand(i, 43) * 0.35} rot={rand(i, 44) * 80 - 40} />
          ))}
          <text x={0} y={-210} textAnchor="middle" fill={P.cream} fontSize={32} fontFamily={FONT_HEAD}>随手一升海水</text>
          {/* 计数牌 */}
          {countK > 0 && (
            <g transform={`translate(150,-120) scale(${Math.max(0.001, spring({frame: f - b2 - 8, fps, config: {damping: 9, stiffness: 170}}))})`} opacity={countK}>
              <rect x={-10} y={-34} width={216} height={66} rx={16} fill={P.card} stroke={P.phage} strokeWidth={5} />
              <text x={98} y={12} textAnchor="middle" fill={P.phage} fontSize={34} fontFamily={FONT_HEAD} fontWeight={800}>≈ 几亿颗</text>
            </g>
          )}
        </g>

        {/* 右：大细菌 + 落地阴影 */}
        <ellipse cx={1240} cy={872} rx={310} ry={36} fill={P.bgDeep} />
        <g transform={`translate(1240,620) scale(${1.5 * pop})`}>
          <Bac x={0} y={0} face={sickK > 0.5 ? 'sick' : 'ok'} />
        </g>
        {/* 噬菌体降落 */}
        <Phage x={phageX} y={phageY + bobY} s={1.35} rot={docked ? 0 : Math.sin(f / 9) * 6} />
        {docked && <Ripple x={1200} y={500} startFrame={landT} color={P.phage} />}
        {/* 病号标签 */}
        {sickK > 0 && (
          <g opacity={sickK} transform="translate(1460,330) rotate(10)">
            <rect x={-72} y={-30} width={144} height={56} rx={12} fill={P.seg} />
            <text x={0} y={10} textAnchor="middle" fill="#fff" fontSize={32} fontFamily={FONT_HEAD} fontWeight={800}>中招了</text>
          </g>
        )}
      </svg>
      <Chip text="噬菌体 = 专吃细菌的病毒" x={140} y={150} at={b1 + 8} accent={P.phage} />
      <Chip text="比所有细菌加起来还多" x={140} y={248} at={b1 + 55} accent={P.dna} />
      <Stamp text="病毒界 · 数量冠军" x={1330} y={150} at={b1 + 100} color={P.phageD} size={44} rot={5} />
      <Chip text="不感染人 · 零存在感" x={1420} y={780} at={b2 + 90} accent={P.sky} />
    </FadeIn>
  );
};

/* ============ S1 注射入侵：吸附→注射→复制→撑破 ============ */
const S1 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b0 = beat(0), b1 = beat(1);

  const hoverY = 168;
  const dockT = b1 - 10;                                     // 着陆时刻
  const phageY = interpolate(f, [b0 + 10, dockT - 55], [-140, hoverY], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})
    + interpolate(f, [dockT - 55, dockT], [0, 300 - hoverY], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const docked = f >= dockT;

  const injT = b1 + 50;                                      // 注射
  const sheathK = 1 - clamp01((f - injT) / 40) * 0.62;
  const headDna = 1 - clamp01((f - injT - 30) / 55) * 0.85;
  const dropY = interpolate(f, [injT + 35, injT + 90], [430, 585], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const dnaIn = clamp01((f - injT - 85) / 25);

  const repT = b1 + 155;                                     // 复制
  const chromoK = 1 - clamp01((f - repT) / 45) * 0.78;
  const burstT = b1 + 335;                                   // 爆发
  const swellK = f > burstT - 40 ? Math.sin(clamp01((f - burstT + 40) / 40) * Math.PI) * 0.07 : 0;
  const gone = clamp01((f - burstT - 8) / 16);
  const sickK = clamp01((f - injT - 40) / 25);

  const inside = [[-118, -28], [-58, 34], [-8, -44], [34, 40], [84, -32], [122, 18], [0, 2]];
  const burstDirs = [[-1, -0.35], [-0.75, -0.75], [-0.35, -1], [0.1, -1.05], [0.55, -0.85], [0.95, -0.5], [1, 0.1], [0.7, 0.65], [-0.85, 0.5], [-0.2, 0.95]];
  const callout = clamp01((f - b0 - 40) / 18) * (1 - clamp01((f - dockT + 60) / 15));

  const steps: [string, string, number][] = [['1', '吸附', dockT], ['2', '注射', injT], ['3', '复制', repT], ['4', '爆发', burstT]];

  return (
    <FadeIn>
      <Backdrop top={P.bgLite} bottom={P.bgLite} />
      <SceneTag index={1} name={sec.name} accent={P.dna} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <ellipse cx={960} cy={884} rx={330} ry={38} fill={P.bgDeep} />
        {/* 大细菌 */}
        <g transform={`translate(960,640) scale(${1.85 * (1 + swellK)})`} opacity={1 - gone}>
          <Bac x={0} y={0} face={gone > 0 ? 'dead' : (sickK > 0.5 ? 'sick' : 'ok')}
            chromo={chromoK} cracked={clamp01((f - burstT + 30) / 25)} />
        </g>
        {/* 注入体内的DNA */}
        {dnaIn > 0 && gone < 1 && (
          <DnaBit x={1002} y={596} s={0.8} rot={-20 + Math.sin(f / 12) * 8} op={dnaIn * (1 - gone) * chromoK} />
        )}
        {/* 体内复制出的新噬菌体 */}
        {gone < 1 && inside.map(([ix, iy], i) => {
          const at = repT + i * 11;
          const sp = Math.max(0.001, spring({frame: f - at, fps, config: {damping: 9, stiffness: 180}}));
          return f > at ? <MiniPhage key={i} x={960 + ix} y={640 + iy} s={0.62 * sp} rot={rand(i, 55) * 60 - 30} op={1 - gone} /> : null;
        })}
        {/* 噬菌体 */}
        {gone < 0.9 && (
          <Phage x={960} y={phageY + (docked ? Math.sin(f / 10) * 3 : 0)} s={1.3}
            rot={docked ? 0 : Math.sin(f / 8) * 5} dnaK={headDna} sheathK={sheathK} op={1 - gone} />
        )}
        {/* DNA 滴落注入 */}
        {f > injT + 35 && f < injT + 95 && (
          <g>
            <circle cx={960} cy={dropY} r={11} fill={P.dna} />
            <line x1={960} y1={dropY - 34} x2={960} y2={dropY - 10} stroke={P.dna} strokeWidth={6} strokeLinecap="round" opacity={0.6} />
          </g>
        )}
        {docked && <Ripple x={960} y={492} startFrame={dockT} color={P.phage} />}
        {/* 外形标注（着陆前） */}
        {callout > 0 && (
          <g opacity={callout}>
            <path d="M 800 210 L 902 178" stroke={P.cream} strokeWidth={4} strokeDasharray="8 8" />
            <text x={790} y={222} textAnchor="end" fill={P.cream} fontSize={30} fontFamily={FONT_HEAD}>蛋白头壳 · 装着DNA</text>
            <path d="M 1120 268 L 986 258" stroke={P.cream} strokeWidth={4} strokeDasharray="8 8" />
            <text x={1130} y={276} fill={P.cream} fontSize={30} fontFamily={FONT_HEAD}>中空尾巴</text>
          </g>
        )}
        {/* 爆发 */}
        {f > burstT && (
          <g>
            {burstDirs.map(([dx, dy], i) => {
              const t = f - burstT - i * 2;
              if (t < 0) return null;
              const p = clamp01(t / 55);
              return <MiniPhage key={i} x={960 + dx * 460 * p} y={640 + dy * 430 * p}
                s={0.7} rot={t * 9 + i * 40} op={1 - clamp01((t - 40) / 25)} />;
            })}
            {[0, 1, 2].map((i) => {
              const t = f - burstT;
              const p = clamp01(t / 40);
              const a = -40 - i * 100;
              return <path key={i} d="M -60 0 A 60 60 0 0 1 60 0" fill="none" stroke={P.bacD} strokeWidth={14}
                transform={`translate(${960 + Math.cos((a - 90) * Math.PI / 180) * 340 * p},${640 + Math.sin((a - 90) * Math.PI / 180) * 320 * p}) rotate(${a + t * 6})`}
                opacity={0.9 * (1 - p)} />;
            })}
            <Ripple x={960} y={640} startFrame={burstT} color={P.seg} />
          </g>
        )}
      </svg>
      {steps.map(([n, tx, at], i) => (
        <StepPill key={n} n={n} text={tx} x={560 + i * 205} y={132} lit={f >= at} />
      ))}
      <Chip text="像皮下注射一样注DNA" x={140} y={150} at={injT + 20} accent={P.dna} />
      <Chip text="接管工厂 · 批量复制" x={140} y={760} at={repT + 30} accent={P.phage} />
      <Stamp text="从内部撑破" x={1420} y={760} at={burstT + 18} color={P.seg} size={48} rot={-5} />
      <Chip text="几十分钟一锅端" x={1420} y={150} at={burstT + 60} accent={P.dna} />
    </FadeIn>
  );
};

/* ============ S2 卧底模式：基因嵌进染色体 → 分裂传代 → 反水爆发 ============ */
const S2 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b0 = beat(0), b1 = beat(1), b2 = beat(2);

  const dockT = b0 + 60;
  const phageY = interpolate(f, [b0 + 5, dockT], [-120, 252], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const phageOp = 1 - clamp01((f - b0 - 150) / 40);
  const segInY = interpolate(f, [dockT + 20, dockT + 70], [390, 545], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const segK = clamp01((f - dockT - 80) / 30);
  const segPulse = segK * (0.72 + 0.28 * Math.sin(f / 6));

  const divT = b0 + 235;                                     // 分裂
  const origOp = 1 - clamp01((f - divT) / 14);
  const dPop = Math.max(0.001, spring({frame: f - divT - 8, fps, config: {damping: 10, stiffness: 140}}));
  const revT = b1 + 20;                                      // 反水
  const deadK = clamp01((f - revT - 55) / 25);
  const burstT2 = revT + 75;
  const zOp = clamp01((f - b0 - 175) / 20) * (1 - clamp01((f - revT + 20) / 15));
  const burstDirs = [[-1, -0.4], [-0.6, -0.85], [-0.15, -1], [0.4, -0.9], [0.85, -0.5], [1, 0.05], [0.6, 0.6], [-0.7, 0.6]];

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={2} name={sec.name} accent={P.seg} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <ellipse cx={640} cy={852} rx={280} ry={32} fill={P.bgDeep} />
        {/* 母体细菌（分裂前） */}
        <g transform="translate(640,560) scale(1.5)" opacity={origOp}>
          <Bac x={0} y={0} face="ok" seg={segPulse} />
        </g>
        {/* 潜入的红色基因片段 */}
        {f > dockT + 20 && f < dockT + 95 && (
          <DnaBit x={640} y={segInY} s={0.65} rot={90} col={P.seg} />
        )}
        {/* 噬菌体送完货就走 */}
        {phageOp > 0 && <Phage x={640} y={phageY} s={0.95} op={phageOp} dnaK={1 - segK} />}
        {f >= dockT && f < dockT + 40 && <Ripple x={640} y={430} startFrame={dockT} color={P.seg} />}
        {/* 睡着 Zzz */}
        {zOp > 0 && [0, 1, 2].map((i) => {
          const t = ((f / 26 + i * 0.9) % 3);
          return (
            <text key={i} x={800 + i * 52 + t * 26} y={400 - t * 60} fill={C.muted}
              fontSize={40 - i * 6} fontFamily={FONT_HEAD} opacity={zOp * (1 - t / 3)}>Z</text>
          );
        })}
        {/* 嵌合基因标注 */}
        {segK > 0.5 && origOp > 0.3 && (
          <g>
            <path d="M 668 545 L 900 730" stroke={P.seg} strokeWidth={4} strokeDasharray="8 8" />
            <text x={916} y={748} fill={P.seg} fontSize={30} fontFamily={FONT_HEAD} fontWeight={700}>病毒基因 · 潜伏中</text>
          </g>
        )}
        {/* 分裂成两个带卧底的子代 */}
        <g transform={`translate(460,560) scale(${1.12 * dPop})`}>
          <Bac x={0} y={0} face={deadK > 0.4 ? 'dead' : (f > revT ? 'sick' : 'ok')}
            col={deadK > 0.4 ? P.grey : P.bac} seg={segK} chromo={1 - deadK * 0.6} />
        </g>
        <g transform={`translate(880,560) scale(${1.12 * dPop})`}>
          <Bac x={0} y={0} face="ok" seg={segK} />
        </g>
        {/* 环境恶化 → 雷劈左菌 */}
        {f > revT && (
          <g>
            <Bolt x={330} y={300} s={1.2} op={0.5 + Math.abs(Math.sin(f / 5)) * 0.5} />
            <path d="M 352 372 L 452 470" stroke={P.dna} strokeWidth={9} strokeLinecap="round"
              opacity={clamp01((f - revT - 8) / 10) * (1 - clamp01((f - revT - 40) / 20))} />
          </g>
        )}
        {/* 左菌反水爆发 */}
        {f > burstT2 && (
          <g>
            {burstDirs.map(([dx, dy], i) => {
              const t = f - burstT2 - i * 2;
              if (t < 0) return null;
              const p = clamp01(t / 50);
              return <MiniPhage key={i} x={460 + dx * 360 * p} y={560 + dy * 340 * p}
                s={0.62} rot={t * 8 + i * 45} op={1 - clamp01((t - 36) / 22)} />;
            })}
            <Ripple x={460} y={560} startFrame={burstT2} color={P.seg} />
          </g>
        )}
        {/* 霍乱借基因：右侧小毒素分子 */}
        {(() => {
          const toxK = clamp01((f - b2 - 10) / 25);
          return toxK > 0 ? (
            <g transform="translate(1240,660)" opacity={toxK}>
              <circle cx={0} cy={0} r={22} fill={P.seg} />
              <circle cx={38} cy={-26} r={14} fill={P.segD} />
              <circle cx={40} cy={22} r={14} fill={P.segD} />
              <line x1={14} y1={-10} x2={30} y2={-20} stroke={P.segD} strokeWidth={7} />
              <line x1={15} y1={10} x2={31} y2={18} stroke={P.segD} strokeWidth={7} />
              <text x={20} y={82} textAnchor="middle" fill={P.seg} fontSize={30} fontFamily={FONT_HEAD} fontWeight={700}>霍乱毒素基因</text>
            </g>
          ) : null;
        })()}
      </svg>
      <Chip text="基因整合进染色体" x={140} y={150} at={dockT + 60} accent={P.seg} />
      <Stamp text="卧底 · 代代相传" x={1150} y={150} at={divT + 30} color={P.phageD} size={44} rot={5} />
      <Chip text="环境恶化 → 突然反水" x={140} y={770} at={revT + 10} accent={P.dna} />
      <Stamp text="连毒素都是借来的" x={1350} y={770} at={b2 + 40} color={P.seg} size={44} rot={-4} />
    </FadeIn>
  );
};

/* ============ S3 细菌反击：左·分子剪刀剪碎DNA / 右·CRISPR档案比对 ============ */
const S3 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b0 = beat(0), b1 = beat(1);

  // 左侧三条入侵DNA：飞入→被酶咬断
  const strands = [
    {at: b0 + 50, y: 480, col: P.dna},
    {at: b0 + 170, y: 600, col: P.seg},
    {at: b0 + 290, y: 710, col: P.sky},
  ];
  // 右侧档案卡 + 一次比对拦截
  const cards = [
    {x: 1290, y: 300, col: P.dna},
    {x: 1410, y: 284, col: P.sky},
    {x: 1530, y: 300, col: P.seg},
  ];
  const inAt = b0 + 330, snipT = inAt + 52;
  const inX = interpolate(f, [inAt, snipT], [2030, 1700], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const cardFly = clamp01((f - inAt - 25) / 18);
  const medalK = clamp01((f - b1 - 10) / 25);

  const cutStrand = (sd: typeof strands[0], i: number) => {
    const t = f - sd.at;
    if (t < 0) return null;
    const snT = 44;
    if (t < snT) {
      const x = interpolate(t, [0, snT], [-90, 300], {extrapolateRight: 'clamp'});
      return <DnaBit key={i} x={x} y={sd.y} col={sd.col} s={0.85} />;
    }
    const bt = t - snT;
    const fade = 1 - clamp01(bt / 22);
    return (
      <g key={i} opacity={fade}>
        <DnaBit x={300 - 14 - bt * 4} y={sd.y - bt * 1.5} col={sd.col} s={0.45} rot={-bt * 3} />
        <DnaBit x={300 + 14 + bt * 4} y={sd.y + bt * 1.5} col={sd.col} s={0.45} rot={bt * 3} />
      </g>
    );
  };

  return (
    <FadeIn>
      <Backdrop top={P.bgDeep} bottom={P.bgDeep} />
      <SceneTag index={3} name={sec.name} accent={P.enzyme} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        <line x1={960} y1={150} x2={960} y2={850} stroke={P.cream} strokeWidth={3} strokeDasharray="4 18" opacity={0.3} />
        {/* 左：内切酶拦截 */}
        <ellipse cx={560} cy={858} rx={260} ry={30} fill={P.bg} />
        <g transform="translate(560,620) scale(1.35)"><Bac x={0} y={0} face="ok" /></g>
        {strands.map(cutStrand)}
        {strands.map((sd, i) => {
          const at = sd.at + 32;
          const o = clamp01((f - at) / 10) * (1 - clamp01((f - at - 34) / 14));
          return o > 0 ? <Enzyme key={i} x={300} y={sd.y - 6} s={0.95} rot={0} op={o} chomp={f > sd.at + 40 && f < sd.at + 66 ? 1 : 0} /> : null;
        })}
        <text x={120} y={428} fill={P.enzyme} fontSize={30} fontFamily={FONT_HEAD} fontWeight={700}>限制性内切酶 · 分子剪刀</text>
        {/* 右：CRISPR 档案库 */}
        <ellipse cx={1440} cy={858} rx={260} ry={30} fill={P.bg} />
        <g transform="translate(1440,620) scale(1.35)"><Bac x={0} y={0} face="ok" seg={0} /></g>
        {cards.map((cd, i) => {
          const at = b0 + 90 + i * 22;
          const o = clamp01((f - at) / 15);
          const isMatch = i === 2;
          return (
            <g key={i} transform={`translate(${cd.x + (isMatch ? (1700 - cd.x) * cardFly : 0)},${cd.y + (isMatch ? (455 - cd.y) * cardFly : 0)})`} opacity={o}>
              <FileCard x={0} y={0} rot={i * 10 - 10} col={cd.col} hit={isMatch ? cardFly : 0} />
            </g>
          );
        })}
        <text x={1100} y={210} fill={P.phage} fontSize={30} fontFamily={FONT_HEAD} fontWeight={700}
          opacity={clamp01((f - b0 - 80) / 15)}>CRISPR · 入侵者档案库</text>
        {/* 同款DNA再来 → 照档案拦截 */}
        {f > inAt && f < snipT && <DnaBit x={inX} y={560} col={P.seg} s={0.85} rot={0} />}
        {f >= snipT && f < snipT + 26 && (
          <g opacity={1 - (f - snipT) / 26}>
            <DnaBit x={1700 - 12 - (f - snipT) * 4} y={560 - (f - snipT)} col={P.seg} s={0.45} rot={-(f - snipT) * 3} />
            <DnaBit x={1700 + 12 + (f - snipT) * 4} y={560 + (f - snipT)} col={P.seg} s={0.45} rot={(f - snipT) * 3} />
          </g>
        )}
        {f > snipT - 12 && f < snipT + 30 && (
          <Enzyme x={1710} y={548} s={0.95} rot={180} op={clamp01((f - snipT + 12) / 8) * (1 - clamp01((f - snipT - 16) / 14))} />
        )}
        {f > snipT + 10 && f < snipT + 60 && (
          <text x={1735} y={665} textAnchor="middle" fill={P.seg} fontSize={32} fontFamily={FONT_HEAD}
            fontWeight={800} opacity={1 - (f - snipT - 10) / 50}>匹配！剪掉</text>
        )}
        {/* 诺贝尔奖章 */}
        {medalK > 0 && (
          <g transform={`translate(960,250) scale(${Math.max(0.001, spring({frame: f - b1 - 10, fps, config: {damping: 9, stiffness: 150}}))})`} opacity={medalK}>
            <Medal x={0} y={0} />
          </g>
        )}
      </svg>
      <Chip text="外来DNA → 当场剪碎" x={140} y={150} at={b0 + 30} accent={P.enzyme} />
      <Chip text="免疫记忆 · 比疫苗早几十亿年" x={1230} y={760} at={b0 + 210} accent={P.phage} />
      <Stamp text="人类借走 → 基因编辑" x={140} y={770} at={b1 + 40} color={P.dnaD} size={44} rot={-4} />
    </FadeIn>
  );
};

/* ============ S4 借刀杀菌：左抗生素无差别轰炸 / 右噬菌体精准点杀 ============ */
const S4 = ({sec}: {sec: Sec}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = (i: number) => Math.round((sec.cues[Math.min(i, sec.cues.length - 1)].t0 - sec.start) * fps);
  const b0 = beat(0), b1 = beat(1), b2 = beat(2);

  const vialK = clamp01((f - b0 - 10) / 20) * (1 - clamp01((f - b1 + 10) / 18));
  const splitK = clamp01((f - b1) / 25);

  const dropT = b1 + 40;
  const capY = interpolate(f, [dropT, dropT + 34], [-120, 330], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const boomT = dropT + 36;
  const leftDead = clamp01((f - boomT) / 30);
  const phageIn = b1 + 95;
  const phX = interpolate(f, [phageIn, phageIn + 48], [1980, 1590], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const phY = interpolate(f, [phageIn, phageIn + 48], [395, 424], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const rightDead = clamp01((f - phageIn - 62) / 25);

  const leftCrowd = [[270, 540], [430, 520], [590, 550], [330, 665], [490, 690], [650, 655]];
  const rightCrowd = [[1290, 540], [1450, 520], [1610, 550], [1350, 665], [1510, 690], [1670, 655]];
  const redIdx = 2;

  return (
    <FadeIn>
      <Backdrop top={P.bg} bottom={P.bg} />
      <SceneTag index={4} name={sec.name} accent={P.dna} />
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
        {/* 阶段A：老药瓶 */}
        {vialK > 0 && (
          <g opacity={vialK}>
            <ellipse cx={960} cy={700} rx={190} ry={26} fill={P.bgDeep} />
            <g transform={`translate(960,500) rotate(${Math.sin(f / 12) * 4})`}>
              <Vial x={0} y={0} s={1.35} />
            </g>
            <text x={960} y={760} textAnchor="middle" fill={P.cream} fontSize={34} fontFamily={FONT_HEAD} fontWeight={700}>1917 年 · 治好痢疾</text>
          </g>
        )}
        {/* 阶段B：对照实验 */}
        {splitK > 0 && (
          <g opacity={splitK}>
            <line x1={960} y1={170} x2={960} y2={850} stroke={P.cream} strokeWidth={3} strokeDasharray="4 18" opacity={0.3} />
            {/* 左：抗生素 */}
            {leftCrowd.map(([cx, cy], i) => (
              <MiniBac key={i} x={cx} y={cy} s={1.05} rot={rand(i, 61) * 30 - 15}
                col={i === redIdx ? P.seg : P.bac}
                face={leftDead > 0.5 ? 'dead' : 'ok'}
                op={leftDead > 0.5 ? 0.45 : 1} />
            ))}
            {f > dropT && f < boomT + 8 && <Capsule x={480} y={capY} s={1.1} rot={Math.sin(f / 7) * 10} />}
            {f >= boomT && (
              <g opacity={1 - clamp01((f - boomT - 30) / 25)}>
                <circle cx={480} cy={600} r={60 + clamp01((f - boomT) / 30) * 180} fill="none"
                  stroke={P.seg} strokeWidth={8 * (1 - clamp01((f - boomT) / 30))} />
              </g>
            )}
            <text x={480} y={830} textAnchor="middle" fill={P.seg} fontSize={32} fontFamily={FONT_HEAD} fontWeight={700}>好坏通杀</text>
            {/* 右：噬菌体点杀 */}
            {rightCrowd.map(([cx, cy], i) => (
              <MiniBac key={i} x={cx} y={cy} s={1.05} rot={rand(i, 71) * 30 - 15}
                col={i === redIdx ? P.seg : P.bac}
                face={i === redIdx && rightDead > 0.5 ? 'dead' : 'ok'}
                op={i === redIdx && rightDead > 0.5 ? 0.4 : 1} />
            ))}
            {f > phageIn && f < phageIn + 90 && (
              <Phage x={phX} y={phY + (f > phageIn + 48 ? Math.sin(f / 9) * 3 : 0)} s={0.55} rot={f < phageIn + 48 ? 30 : 0} />
            )}
            {f >= phageIn + 62 && <Ripple x={1610} y={540} startFrame={phageIn + 62} color={P.phage} />}
            <text x={1500} y={830} textAnchor="middle" fill={P.phage} fontSize={32} fontFamily={FONT_HEAD} fontWeight={700}>只杀目标株</text>
          </g>
        )}
        {/* 阶段C：结论 */}
        {f > b2 && (
          <g transform={`translate(960,205) scale(${Math.max(0.001, spring({frame: f - b2, fps, config: {damping: 10, stiffness: 140}}))})`}>
            <rect x={-330} y={-46} width={660} height={88} rx={22} fill={P.card} stroke={P.phage} strokeWidth={5} />
            <text x={0} y={12} textAnchor="middle" fill={C.text} fontSize={40} fontFamily={FONT_HEAD} fontWeight={800}>会"感冒"，但剧本完全不同</text>
          </g>
        )}
      </svg>
      <Chip text="噬菌体疗法" x={140} y={150} at={b0 + 30} accent={P.phage} />
      <Chip text="抗生素 = 轰炸" x={140} y={200 + 60} at={b1 + 15} accent={P.seg} />
      <Chip text="噬菌体 = 狙击" x={1620} y={200 + 60} at={phageIn + 20} accent={P.phage} />
      <Chip text="不伤肠道菌群" x={1420} y={150} at={rightDead > 0.5 ? phageIn + 90 : 99999} accent={P.dna} />
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
        <ellipse cx={960} cy={1010} rx={1080} ry={120} fill={P.bg} />
        <g transform={`translate(700,880) rotate(72) scale(${pop})`}>
          <Phage x={0} y={0} s={0.9} rot={0} />
        </g>
        <g transform={`translate(1200,930) rotate(-8) scale(${pop})`}>
          <Bac x={0} y={0} s={0.85} face="ok" />
        </g>
        {[0, 1, 2].map((i) => (
          <MiniPhage key={i} x={480 + i * 480 + Math.sin(f / 14 + i * 2) * 24}
            y={660 + Math.cos(f / 16 + i * 1.7) * 20} s={0.6} rot={i * 40 - 20} op={0.8} />
        ))}
      </svg>
      <div style={{transform: `scale(${s})`, textAlign: 'center', fontFamily: FONT_HEAD, marginTop: -240}}>
        <div style={{fontSize: 70, fontWeight: 800, color: C.text}}>细菌的感冒，是被噬菌体揍的</div>
        <div style={{fontSize: 38, color: C.muted, marginTop: 22, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ============ 主组件 ============ */
const SCENES = [S0, S1, S2, S3, S4];

export const E216 = ({data = tl}: {data?: any}) => {
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
