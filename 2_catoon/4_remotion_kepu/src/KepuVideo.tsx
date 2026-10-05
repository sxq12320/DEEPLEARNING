import {AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';

const BG = '#0f1e3a';
const TEXT = '#f1f5f9';
const MUTED = '#94a3b8';
const ACCENTS = ['#60a5fa', '#fbbf24', '#34d399', '#f472b6', '#a78bfa'];
const FONT = '"Microsoft YaHei", "PingFang SC", sans-serif';

type Cue = {t0: number; t1: number; text: string};
type Section = {name: string; keywords: string[]; text: string; audio: string; start: number; dur: number; cues: Cue[]};
type Timeline = {title: string; episode: string; titleCardSec: number; endCardSec: number; totalSec: number; sections: Section[]};

/* ---------- 背景装饰：缓慢旋转的虚线环 + 漂浮粒子 ---------- */
const Backdrop = ({accent}: {accent: string}) => {
  const frame = useCurrentFrame();
  return (
    <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute'}}>
      <defs>
        <radialGradient id="glow" cx="70%" cy="30%" r="70%">
          <stop offset="0%" stopColor={accent} stopOpacity={0.14} />
          <stop offset="100%" stopColor={BG} stopOpacity={0} />
        </radialGradient>
      </defs>
      <rect width={1920} height={1080} fill={BG} />
      <rect width={1920} height={1080} fill="url(#glow)" />
      {[260, 400, 540].map((r, i) => (
        <g key={r} transform={`rotate(${frame * (i % 2 ? -0.12 : 0.1) + i * 40} 1500 360)`}>
          <circle cx={1500} cy={360} r={r} fill="none" stroke={accent} strokeOpacity={0.12} strokeWidth={2} strokeDasharray="4 18" />
        </g>
      ))}
      {Array.from({length: 18}).map((_, i) => (
        <circle
          key={i}
          cx={(i * 331) % 1900 + 10}
          cy={(i * 197) % 1060 + 10 + Math.sin(frame / 30 + i) * 12}
          r={3}
          fill={TEXT}
          opacity={0.08}
        />
      ))}
    </svg>
  );
};

/* ---------- 顶部进度条 ---------- */
const ProgressBar = ({accent}: {accent: string}) => {
  const frame = useCurrentFrame();
  const {durationInFrames} = useVideoConfig();
  const p = frame / durationInFrames;
  return (
    <div style={{position: 'absolute', top: 0, left: 0, width: '100%', height: 8, background: 'rgba(255,255,255,0.08)'}}>
      <div style={{width: `${p * 100}%`, height: '100%', background: accent}} />
    </div>
  );
};

/* ---------- 分节卡 ---------- */
const SectionCard = ({sec, index, accent}: {sec: Section; index: number; accent: string}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const headScale = spring({frame, fps, config: {damping: 14}});
  const headOpacity = interpolate(frame, [0, 8], [0, 1]);

  return (
    <AbsoluteFill>
      <Backdrop accent={accent} />
      {/* 大幽灵序号 */}
      <div style={{
        position: 'absolute', right: 120, bottom: 60, fontSize: 380, fontWeight: 900,
        color: accent, opacity: 0.07, fontFamily: FONT, lineHeight: 1,
      }}>
        {String(index + 1).padStart(2, '0')}
      </div>

      {/* 小节标题 */}
      <div style={{
        position: 'absolute', top: 200, left: 160,
        transform: `scale(${headScale})`, transformOrigin: 'left center', opacity: headOpacity,
      }}>
        <div style={{color: accent, fontSize: 40, fontFamily: FONT, letterSpacing: 10, marginBottom: 18}}>
          PART {String(index + 1).padStart(2, '0')}
        </div>
        <div style={{color: TEXT, fontSize: 92, fontWeight: 800, fontFamily: FONT, letterSpacing: 4}}>
          {sec.name}
        </div>
        <div style={{width: 140, height: 10, background: accent, marginTop: 28, borderRadius: 5}} />
      </div>

      {/* 关键词弹片 */}
      <div style={{position: 'absolute', left: 160, top: 560, display: 'flex', gap: 28}}>
        {sec.keywords.map((kw, i) => {
          const s = spring({frame: frame - 14 - i * 8, fps, config: {damping: 12, mass: 0.5}});
          const o = interpolate(frame, [14 + i * 8, 22 + i * 8], [0, 1], {extrapolateLeft: 'clamp'});
          return (
            <div key={i} style={{
              transform: `scale(${s})`, opacity: o,
              padding: '22px 44px', borderRadius: 999,
              border: `3px solid ${accent}`, color: TEXT,
              background: 'rgba(255,255,255,0.06)',
              fontSize: 44, fontWeight: 600, fontFamily: FONT, letterSpacing: 3,
            }}>
              {kw}
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};

/* ---------- 标题卡 / 结尾卡 ---------- */
const TitleCard = ({data}: {data: Timeline}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame, fps, config: {damping: 12, mass: 0.7}});
  const o = interpolate(frame, [0, 10], [0, 1]);
  return (
    <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
      <Backdrop accent={ACCENTS[0]} />
      <div style={{transform: `scale(${s})`, opacity: o, textAlign: 'center', padding: '0 160px'}}>
        <div style={{color: ACCENTS[0], fontSize: 38, fontFamily: FONT, letterSpacing: 14, marginBottom: 30}}>
          冷知识 · 第 {data.episode} 期
        </div>
        <div style={{color: TEXT, fontSize: 110, fontWeight: 900, fontFamily: FONT, lineHeight: 1.25}}>
          {data.title}
        </div>
      </div>
    </AbsoluteFill>
  );
};

const EndCard = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame, fps, config: {damping: 14}});
  return (
    <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
      <Backdrop accent={ACCENTS[2]} />
      <div style={{transform: `scale(${s})`, textAlign: 'center', color: TEXT, fontFamily: FONT}}>
        <div style={{fontSize: 72, fontWeight: 800}}>涨知识了?</div>
        <div style={{fontSize: 40, color: MUTED, marginTop: 24, letterSpacing: 6}}>关注 · 下期更冷</div>
      </div>
    </AbsoluteFill>
  );
};

/* ---------- 字幕轨 ---------- */
const Subtitles = ({data}: {data: Timeline}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const t = frame / fps;
  const cue = data.sections.flatMap((s) => s.cues).find((c) => t >= c.t0 && t <= c.t1 + 0.15);
  if (!cue) return null;
  return (
    <div style={{position: 'absolute', bottom: 54, width: '100%', textAlign: 'center'}}>
      <span style={{
        display: 'inline-block', maxWidth: '80%',
        background: 'rgba(0,0,0,0.62)', color: '#fff',
        fontSize: 38, fontFamily: FONT, padding: '14px 34px', borderRadius: 14, lineHeight: 1.5,
      }}>
        {cue.text}
      </span>
    </div>
  );
};

/* ---------- 主组件 ---------- */
export const KepuVideo = ({data}: {data: Timeline}) => {
  const {fps} = useVideoConfig();
  return (
    <AbsoluteFill style={{background: BG}}>
      <Sequence durationInFrames={Math.round(data.titleCardSec * fps)}>
        <TitleCard data={data} />
      </Sequence>

      {data.sections.map((sec, i) => (
        <Sequence key={i} from={Math.round(sec.start * fps)} durationInFrames={Math.round(sec.dur * fps)}>
          <SectionCard sec={sec} index={i} accent={ACCENTS[i % ACCENTS.length]} />
          <Audio src={staticFile(sec.audio)} />
        </Sequence>
      ))}

      <Sequence from={Math.round((data.totalSec - data.endCardSec) * fps)} durationInFrames={Math.round(data.endCardSec * fps)}>
        <EndCard />
      </Sequence>

      <Subtitles data={data} />
      <ProgressBar accent={ACCENTS[0]} />
    </AbsoluteFill>
  );
};
