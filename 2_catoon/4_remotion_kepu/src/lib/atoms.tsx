import {AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT_BODY, FONT_HEAD, rand, clamp01} from './style';

/* ---------- 背景：纯色平涂 ---------- */
export const Backdrop = ({top, bottom}: {top: string; bottom: string; glow?: string}) => {
  return (
    <AbsoluteFill style={{background: bottom}} />
  );
};

/* ---------- svg 内漂浮容器 ---------- */
export const Float = ({amp = 6, speed = 0.05, phase = 0, x = 0, y = 0, children}: {
  amp?: number; speed?: number; phase?: number; x?: number; y?: number; children: React.ReactNode;
}) => {
  const f = useCurrentFrame();
  return (
    <g transform={`translate(${x + Math.sin(f * speed + phase) * amp},${y + Math.cos(f * speed * 0.8 + phase) * amp * 0.6})`}>
      {children}
    </g>
  );
};

/* ---------- 触摸涟漪 ---------- */
export const Ripple = ({x, y, startFrame, color = C.amber}: {x: number; y: number; startFrame: number; color?: string}) => {
  const f = useCurrentFrame();
  const t = f - startFrame;
  if (t < 0) return null;
  return (
    <g>
      {[0, 10].map((d) => {
        const p = clamp01((t - d) / 26);
        if (p <= 0 || p >= 1) return null;
        return (
          <circle key={d} cx={x} cy={y} r={20 + p * 120} fill="none" stroke={color}
            strokeWidth={6 * (1 - p)} opacity={0.9 * (1 - p)} />
        );
      })}
    </g>
  );
};

/* ---------- 弹性文字贴片（小印章） ---------- */
export const Stamp = ({text, x, y, at, color = C.coral, size = 56, rot = -8}: {
  text: string; x: number; y: number; at: number; color?: string; size?: number; rot?: number;
}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f - at, fps, config: {damping: 11, mass: 0.6, stiffness: 160}});
  const o = interpolate(f, [at, at + 6], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  if (o <= 0) return null;
  return (
    <div style={{
      position: 'absolute', left: x, top: y, transform: `scale(${s}) rotate(${rot}deg)`, opacity: o,
      background: color, color: '#fff', fontFamily: FONT_HEAD, fontSize: size, fontWeight: 700,
      padding: '10px 30px', borderRadius: 18, boxShadow: '0 10px 30px rgba(0,0,0,0.35)',
      letterSpacing: 4, whiteSpace: 'nowrap',
    }}>
      {text}
    </div>
  );
};

/* ---------- 小节标签：左上 PART 01 · 名 ---------- */
export const SceneTag = ({index, name, accent}: {index: number; name: string; accent: string}) => {
  const f = useCurrentFrame();
  const o = interpolate(f, [0, 12], [0, 1], {extrapolateRight: 'clamp'});
  const x = interpolate(f, [0, 12], [-20, 0], {extrapolateRight: 'clamp'});
  return (
    <div style={{
      position: 'absolute', top: 56, left: 80, opacity: o, transform: `translateX(${x}px)`,
      display: 'flex', alignItems: 'center', gap: 14, fontFamily: FONT_HEAD,
    }}>
      <div style={{width: 14, height: 14, borderRadius: 4, background: accent, transform: 'rotate(45deg)'}} />
      <div style={{color: accent, fontSize: 30, letterSpacing: 6}}>
        {String(index + 1).padStart(2, '0')}
      </div>
      <div style={{color: C.muted, fontSize: 30, letterSpacing: 4}}>{name}</div>
    </div>
  );
};

/* ---------- 关键词小胶囊 ---------- */
export const Chip = ({text, x, y, at, accent}: {text: string; x: number; y: number; at: number; accent: string}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: f - at, fps, config: {damping: 13, mass: 0.5}});
  const o = interpolate(f, [at, at + 5], [0, 1], {extrapolateLeft: 'clamp'});
  if (o <= 0) return null;
  return (
    <div style={{
      position: 'absolute', left: x, top: y, transform: `scale(${s})`, opacity: o,
      padding: '14px 30px', borderRadius: 999, border: `3px solid ${accent}`,
      background: 'rgba(20,24,50,0.75)', color: C.text, fontSize: 34,
      fontFamily: FONT_HEAD, letterSpacing: 3, whiteSpace: 'nowrap',
    }}>
      {text}
    </div>
  );
};

/* ---------- 字幕 ---------- */
export const Subtitles = ({data}: {data: any}) => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  const t = f / fps;
  const cue = data.sections.flatMap((s: any) => s.cues).find((c: any) => t >= c.t0 && t <= c.t1 + 0.15);
  if (!cue) return null;
  return (
    <div style={{position: 'absolute', bottom: 52, width: '100%', textAlign: 'center', zIndex: 50}}>
      <span style={{
        display: 'inline-block', maxWidth: '82%',
        background: 'rgba(8,10,26,0.72)', color: '#fff',
        fontSize: 36, fontFamily: FONT_BODY, fontWeight: 600,
        padding: '13px 32px', borderRadius: 18, lineHeight: 1.5,
        boxShadow: '0 6px 24px rgba(0,0,0,0.3)',
      }}>
        {cue.text}
      </span>
    </div>
  );
};

/* ---------- 顶部进度条 ---------- */
export const ProgressBar = () => {
  const f = useCurrentFrame();
  const {durationInFrames} = useVideoConfig();
  return (
    <div style={{position: 'absolute', top: 0, left: 0, width: '100%', height: 7, background: 'rgba(255,255,255,0.07)', zIndex: 60}}>
      <div style={{width: `${(f / durationInFrames) * 100}%`, height: '100%', background: C.amber, borderRadius: 4}} />
    </div>
  );
};

/* ---------- 手指 ---------- */
export const Finger = ({x, y, rot = -35, scale = 1}: {x: number; y: number; rot?: number; scale?: number}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${scale})`}>
    <rect x={-42} y={-30} width={190} height={86} rx={43} fill={C.skin} />
    <rect x={-42} y={20} width={190} height={36} rx={18} fill={C.skinDark} opacity={0.55} />
    <ellipse cx={-16} cy={-6} rx={26} ry={20} fill="#ffe0bd" />
  </g>
);

/* ---------- 太阳 / 月亮 / 星星 ---------- */
export const Sun = ({x, y, r = 90, opacity = 1}: {x: number; y: number; r?: number; opacity?: number}) => {
  const f = useCurrentFrame();
  return (
    <g transform={`translate(${x},${y})`} opacity={opacity}>
      <g transform={`rotate(${f * 0.4})`}>
        {Array.from({length: 8}).map((_, i) => (
          <rect key={i} x={-8} y={-r - 42} width={16} height={34} rx={8} fill={C.amber}
            transform={`rotate(${i * 45})`} />
        ))}
      </g>
      <circle r={r} fill={C.amber} />
      <circle cy={r * 0.28} r={r * 0.72} fill="#ffb03a" opacity={0.6} />
    </g>
  );
};

export const Moon = ({x, y, scale = 1, opacity = 1}: {x: number; y: number; scale?: number; opacity?: number}) => (
  <g transform={`translate(${x},${y}) scale(${scale})`} opacity={opacity}>
    <circle r={64} fill="#ffe9a8" />
    <circle cx={-24} cy={-16} r={50} fill={C.night0} opacity={0.92} />
    <circle cx={18} cy={22} r={8} fill="#f0d68a" />
    <circle cx={30} cy={-6} r={5} fill="#f0d68a" />
  </g>
);

export const Stars = ({n = 24, from = 0, dur = 40}: {n?: number; from?: number; dur?: number}) => {
  const f = useCurrentFrame();
  const o = interpolate(f, [from, from + dur], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  return (
    <g opacity={o}>
      {Array.from({length: n}).map((_, i) => {
        const tw = 0.5 + Math.sin(f / 9 + i * 2.4) * 0.5;
        return <circle key={i} cx={rand(i, 7) * 1880 + 20} cy={rand(i, 8) * 560 + 30}
          r={rand(i, 9) * 2.6 + 1.4} fill="#fff" opacity={0.35 + tw * 0.55} />;
      })}
    </g>
  );
};

/* ---------- 小甲虫 ---------- */
export const Bug = ({x, y, scale = 1, rot = 0, walk = 0}: {x: number; y: number; scale?: number; rot?: number; walk?: number}) => {
  const f = useCurrentFrame();
  const legW = Math.sin(f * 0.9) * 8 * walk;
  return (
    <g transform={`translate(${x},${y}) rotate(${rot}) scale(${scale})`}>
      {[-1, 1].map((s) =>
        [0, 1, 2].map((i) => (
          <line key={`${s}${i}`} x1={i * 22 - 22} y1={s * 16} x2={i * 22 - 26 + legW * (i % 2 ? 1 : -1)}
            y2={s * 40} stroke="#5a3d8f" strokeWidth={7} strokeLinecap="round" />
        ))
      )}
      <ellipse rx={46} ry={30} fill="#7c5cc4" />
      <ellipse cy={-8} rx={40} ry={20} fill="#9b7ede" />
      <circle cx={42} cy={-4} r={14} fill="#5a3d8f" />
      <line x1={48} y1={-14} x2={62} y2={-30} stroke="#5a3d8f" strokeWidth={5} strokeLinecap="round" />
      <line x1={40} y1={-16} x2={50} y2={-34} stroke="#5a3d8f" strokeWidth={5} strokeLinecap="round" />
      <circle cx={46} cy={-8} r={4} fill="#fff" />
    </g>
  );
};

/* ---------- 雨丝 ---------- */
export const Rain = ({on, n = 46}: {on: number; n?: number}) => {
  const f = useCurrentFrame();
  if (on <= 0) return null;
  return (
    <g opacity={on}>
      {Array.from({length: n}).map((_, i) => {
        const x = rand(i, 11) * 1920;
        const speed = 26 + rand(i, 12) * 10;
        const y = ((f * speed + i * 173) % 1250) - 80;
        return <line key={i} x1={x} y1={y} x2={x - 14} y2={y + 46}
          stroke={C.sky} strokeWidth={4} strokeLinecap="round" opacity={0.55} />;
      })}
    </g>
  );
};

/* ---------- 场景渐隐包装 ---------- */
export const FadeIn = ({children}: {children: React.ReactNode}) => {
  const f = useCurrentFrame();
  const o = interpolate(f, [0, 9], [0, 1], {extrapolateRight: 'clamp'});
  return <AbsoluteFill style={{opacity: o}}>{children}</AbsoluteFill>;
};
