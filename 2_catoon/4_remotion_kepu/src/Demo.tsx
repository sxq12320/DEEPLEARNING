import {AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig} from 'remotion';

// 配色：Kurzgesagt 风的深蓝夜空 + 暖黄 + 亮蓝
const BG = '#0f1e3a';
const SUN = '#ffd166';
const ACCENT = '#60a5fa';
const TEXT = '#f1f5f9';

const Particle = ({x, delay}: {x: number; delay: number}) => {
  const frame = useCurrentFrame();
  const t = frame - delay;
  const y = interpolate(t, [0, 90], [280, 700], {extrapolateRight: 'clamp'});
  const drift = Math.sin(t / 12) * 60;
  const opacity = interpolate(t, [0, 8, 70, 90], [0, 1, 1, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  return (
    <circle
      cx={x + drift}
      cy={y}
      r={9}
      fill={ACCENT}
      opacity={opacity}
    />
  );
};

export const Demo = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();

  // 标题弹性入场
  const titleScale = spring({frame, fps, config: {damping: 12, mass: 0.6}});
  const titleOpacity = interpolate(frame, [0, 10], [0, 1]);

  // 副标题稍后入场
  const subY = spring({frame: frame - 20, fps, config: {damping: 15}});
  const subOpacity = interpolate(frame, [20, 32], [0, 1]);

  // 太阳脉动
  const sunPulse = 1 + Math.sin(frame / 10) * 0.04;

  // 结尾淡出
  const fadeOut = interpolate(frame, [160, 179], [1, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  return (
    <AbsoluteFill style={{background: BG, opacity: fadeOut}}>
      <svg width={1920} height={1080} viewBox="0 0 1920 1080">
        {/* 太阳 */}
        <circle cx={1560} cy={320} r={140 * sunPulse} fill={SUN} />
        <circle cx={1560} cy={320} r={170 * sunPulse} fill={SUN} opacity={0.15} />

        {/* 太阳光线斜射 */}
        {[-60, 0, 60].map((off, i) => {
          const len = interpolate(frame - 40 - i * 6, [0, 30], [0, 1], {
            extrapolateLeft: 'clamp',
            extrapolateRight: 'clamp',
          });
          return (
            <line
              key={off}
              x1={1560 - len * 900}
              y1={320 + off - len * 260}
              x2={1560}
              y2={320 + off}
              stroke={SUN}
              strokeWidth={8}
              strokeLinecap="round"
              opacity={0.6}
            />
          );
        })}

        {/* 大气层横线 */}
        <rect x={0} y={640} width={1920} height={6} fill={ACCENT} opacity={0.4} rx={3} />

        {/* 散射的蓝色粒子 */}
        {[500, 620, 740, 860, 980, 1100, 1220, 1340].map((x, i) => (
          <Particle key={x} x={x} delay={50 + i * 7} />
        ))}
      </svg>

      {/* 标题 */}
      <div
        style={{
          position: 'absolute',
          top: 120,
          width: '100%',
          textAlign: 'center',
          transform: `scale(${titleScale})`,
          opacity: titleOpacity,
          color: TEXT,
          fontFamily: '"Microsoft YaHei", "PingFang SC", sans-serif',
          fontSize: 96,
          fontWeight: 800,
          letterSpacing: 4,
        }}
      >
        为什么天空是蓝色的?
      </div>

      {/* 副标题 */}
      <div
        style={{
          position: 'absolute',
          top: 260,
          width: '100%',
          textAlign: 'center',
          transform: `translateY(${(1 - subY) * 40}px)`,
          opacity: subOpacity,
          color: ACCENT,
          fontFamily: '"Microsoft YaHei", "PingFang SC", sans-serif',
          fontSize: 44,
          fontWeight: 500,
          letterSpacing: 8,
        }}
      >
        三分钟看懂瑞利散射
      </div>

      {/* 底部期号标签 */}
      <div
        style={{
          position: 'absolute',
          bottom: 60,
          width: '100%',
          textAlign: 'center',
          opacity: subOpacity,
          color: '#64748b',
          fontFamily: '"Microsoft YaHei", sans-serif',
          fontSize: 30,
          letterSpacing: 6,
        }}
      >
        第 1 期 · 光的秘密
      </div>
    </AbsoluteFill>
  );
};
