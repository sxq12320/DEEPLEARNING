import React from 'react';
import {Easing, interpolate, interpolateColors, staticFile} from 'remotion';

export interface FrameAudio {
  rms: number;
  bass: number;
  mid: number;
  high: number;
  onset: number;
}

export const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

export const easeInOut = (frame: number, start: number, duration: number) =>
  interpolate(frame, [start, start + Math.max(1, duration)], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: Easing.inOut(Easing.cubic),
  });

export const easeOut = (frame: number, start: number, duration: number) =>
  interpolate(frame, [start, start + Math.max(1, duration)], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: Easing.out(Easing.cubic),
  });

// Deterministic pseudo-random helper [0, 1)
export const hash = (seed: number) => {
  const x = Math.sin(seed * 127.1 + 311.7) * 43758.5453123;
  return x - Math.floor(x);
};

// Shared 6-fold snowflake / crystal geometry helper
export const SnowflakeGlyph: React.FC<{
  cx: number;
  cy: number;
  r: number;
  rot: number;
  color: string;
  opacity: number;
  strokeWidth?: number;
}> = ({cx, cy, r, rot, color, opacity, strokeWidth = 1.8}) => {
  const arms = [0, 60, 120, 180, 240, 300];
  return (
    <g
      transform={`translate(${cx.toFixed(1)} ${cy.toFixed(1)}) rotate(${rot.toFixed(1)})`}
      opacity={opacity}
      stroke={color}
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      fill="none"
    >
      {arms.map((deg) => (
        <g key={deg} transform={`rotate(${deg})`}>
          <line x1={0} y1={0} x2={0} y2={-r} />
          <line x1={0} y1={-r * 0.45} x2={-r * 0.22} y2={-r * 0.65} />
          <line x1={0} y1={-r * 0.45} x2={r * 0.22} y2={-r * 0.65} />
          <line x1={0} y1={-r * 0.72} x2={-r * 0.15} y2={-r * 0.86} />
          <line x1={0} y1={-r * 0.72} x2={r * 0.15} y2={-r * 0.86} />
        </g>
      ))}
    </g>
  );
};

// Artistic Silhouette of a Male Coat Figure (Jay) & Female Gown/Coat Figure (aMEI)
const ArtisticFigureJay: React.FC<{glowColor?: string; scarfWave?: number}> = ({
  glowColor = '#7DD3FC',
  scarfWave = 0,
}) => (
  <g>
    {/* Soft Aura Backlight */}
    <ellipse cx={0} cy={-75} rx={42} ry={95} fill={glowColor} opacity={0.14} />
    {/* Head & Hair Profile */}
    <path
      d="M -11 -162 C -14 -176, 8 -180, 14 -166 C 18 -156, 12 -142, 2 -140 C -8 -140, -11 -150, -11 -162 Z"
      fill="#0B1325"
      stroke={glowColor}
      strokeWidth={2}
    />
    {/* Long Winter Trench Coat & Shoulders */}
    <path
      d="M -24 -134 C -12 -140, 14 -140, 26 -132 L 36 -55 L 28 18 L 12 18 L 8 -42 L -6 -42 L -12 18 L -28 18 L -34 -55 Z"
      fill="#091020"
      stroke={glowColor}
      strokeWidth={2}
      strokeLinejoin="round"
    />
    {/* Collar & Flowing Winter Scarf */}
    <path
      d={`M -12 -136 Q 18 -132 ${48 + scarfWave * 16} ${-118 + scarfWave * 8} Q ${
        78 + scarfWave * 24
      } ${-105 - scarfWave * 10} ${102 + scarfWave * 28} ${-92 + scarfWave * 12}`}
      fill="none"
      stroke="#38BDF8"
      strokeWidth={5.5}
      strokeLinecap="round"
    />
  </g>
);

const ArtisticFigureAmei: React.FC<{glowColor?: string; hairWave?: number}> = ({
  glowColor = '#FDA4AF',
  hairWave = 0,
}) => (
  <g>
    <ellipse cx={0} cy={-72} rx={42} ry={92} fill={glowColor} opacity={0.15} />
    {/* Head & Flowing Long Hair */}
    <path
      d={`M -10 -156 C -12 -170, 8 -172, 12 -158 C 16 -146, 26 -130, ${
        34 + hairWave * 14
      } -112 C 18 -118, 6 -132, -6 -138 Z`}
      fill="#1A0B1E"
      stroke={glowColor}
      strokeWidth={2}
    />
    {/* Elegant Winter Cape / Dress Silhouette */}
    <path
      d="M -20 -130 C -8 -136, 10 -136, 22 -128 L 38 18 L -36 18 Z"
      fill="#16091C"
      stroke={glowColor}
      strokeWidth={2}
      strokeLinejoin="round"
    />
  </g>
);

// Multi-layer Snow & Blossom Particle Field
export const SnowAndPetalsField: React.FC<{
  frame: number;
  audio: FrameAudio;
  snowDensity?: number;
  petalDensity?: number;
  reverseTime?: boolean;
  freezeFactor?: number;
  windBias?: number;
}> = ({
  frame,
  audio,
  snowDensity = 1,
  petalDensity = 0,
  reverseTime = false,
  freezeFactor = 0,
  windBias = 1,
}) => {
  const effectiveFrame =
    freezeFactor > 0 ? frame * (1 - freezeFactor * 0.88) : frame;
  const dir = reverseTime ? -1 : 1;
  const snowCount = Math.round(78 * clamp01(snowDensity));
  const petalCount = Math.round(52 * clamp01(petalDensity));

  return (
    <g>
      {/* Layer 1: Snowflakes (3 depth tiers) */}
      {Array.from({length: snowCount}).map((_, i) => {
        const depth = i % 3;
        const speedY = (depth === 0 ? 1.1 : depth === 1 ? 2.2 : 3.6) * dir;
        const speedX = (depth === 0 ? 0.35 : depth === 1 ? 0.8 : 1.35) * windBias;
        const x0 = hash(i * 13 + 1) * 1920;
        const y0 = hash(i * 29 + 7) * 1080;
        const sway =
          Math.sin(effectiveFrame * 0.025 + i * 1.7) * (14 + depth * 10);
        const x =
          (((x0 + effectiveFrame * speedX + sway) % 1980) + 1980) % 1980 - 30;
        const y =
          (((y0 + effectiveFrame * speedY) % 1140) + 1140) % 1140 - 30;
        const r =
          depth === 0
            ? 2.2 + hash(i) * 1.8
            : depth === 1
            ? 4.5 + hash(i) * 3.5
            : 8 + hash(i) * 6;
        const baseAlpha = depth === 0 ? 0.28 : depth === 1 ? 0.55 : 0.22;
        const alpha =
          baseAlpha * clamp01(snowDensity) * (0.85 + audio.high * 0.3);

        if (depth === 1 && i % 4 === 0) {
          return (
            <SnowflakeGlyph
              key={`s-${i}`}
              cx={x}
              cy={y}
              r={r * 1.65}
              rot={effectiveFrame * (i % 2 === 0 ? 0.6 : -0.6) + i * 30}
              color="#E0F2FE"
              opacity={alpha}
              strokeWidth={1.3}
            />
          );
        }
        return (
          <circle
            key={`s-${i}`}
            cx={x.toFixed(1)}
            cy={y.toFixed(1)}
            r={r.toFixed(1)}
            fill={depth === 2 ? '#BAE6FD' : '#F8FAFC'}
            opacity={alpha.toFixed(3)}
          />
        );
      })}

      {/* Layer 2: Glowing Spring Blossom Petals */}
      {Array.from({length: petalCount}).map((_, i) => {
        const x0 = hash(i * 41 + 100) * 1920;
        const y0 = hash(i * 53 + 200) * 1080;
        const speedY = -1.4 - hash(i * 7) * 1.5;
        const speedX = 1.6 + hash(i * 11) * 1.8;
        const flutter = Math.sin(effectiveFrame * 0.05 + i) * 26;
        const x =
          (((x0 + effectiveFrame * speedX + flutter) % 1980) + 1980) % 1980 - 30;
        const y =
          (((y0 + effectiveFrame * speedY) % 1140) + 1140) % 1140 - 30;
        const rot = effectiveFrame * (1.2 + (i % 3) * 0.5) + i * 47;
        const scale = 0.65 + hash(i * 19) * 0.85 + audio.bass * 0.2;
        const color =
          i % 3 === 0 ? '#FDA4AF' : i % 3 === 1 ? '#FBCFE8' : '#FDE68A';
        const alpha = (0.35 + hash(i * 3) * 0.45) * clamp01(petalDensity);

        return (
          <g
            key={`p-${i}`}
            transform={`translate(${x.toFixed(1)} ${y.toFixed(1)}) rotate(${rot.toFixed(
              1
            )}) scale(${scale.toFixed(2)})`}
            opacity={alpha.toFixed(3)}
          >
            <path
              d="M 0 -10 C 7 -8 10 -1 0 11 C -10 -1 -7 -8 0 -10 Z"
              fill={color}
            />
            <path
              d="M 0 -7 C 3 -5 4 -1 0 7"
              stroke="#FFF"
              strokeWidth={0.8}
              opacity={0.5}
              fill="none"
            />
          </g>
        );
      })}
    </g>
  );
};

// Dynamic Aurora Borealis Sky
export const AuroraSky: React.FC<{
  frame: number;
  audio: FrameAudio;
  jayWeight: number;
  ameiWeight: number;
  climaxBoost?: number;
}> = ({frame, audio, jayWeight, ameiWeight, climaxBoost = 0}) => {
  const ribbons = [0, 1, 2, 3];
  return (
    <g>
      {ribbons.map((idx) => {
        const phase = frame * 0.012 + idx * 1.4;
        const yBase = 170 + idx * 75;
        const amp = 48 + audio.rms * 55 + climaxBoost * 45;
        const y1 = yBase + Math.sin(phase) * amp;
        const y2 = yBase + Math.cos(phase * 1.3 + 1) * amp;
        const y3 = yBase + Math.sin(phase * 0.9 + 2.2) * amp;
        const y4 = yBase + Math.cos(phase * 1.1 + 3.1) * amp;

        const color =
          idx % 2 === 0
            ? interpolateColors(jayWeight, [0, 1], ['#C084FC', '#38BDF8'])
            : interpolateColors(ameiWeight, [0, 1], ['#60A5FA', '#FB7185']);

        const op =
          (0.11 + audio.rms * 0.16 + climaxBoost * 0.15) * (1 - idx * 0.15);

        return (
          <path
            key={idx}
            d={`M -100 ${y1.toFixed(1)} C 480 ${(y2 - 90).toFixed(
              1
            )}, 960 ${(y3 + 90).toFixed(1)}, 1440 ${(y2 - 50).toFixed(
              1
            )} S 2020 ${y4.toFixed(1)}, 2020 ${(y4 + 140).toFixed(
              1
            )} L 2020 ${(y4 + 270).toFixed(1)} C 1440 ${(y3 + 210).toFixed(
              1
            )}, 640 ${(y1 + 240).toFixed(1)}, -100 ${(y1 + 220).toFixed(1)} Z`}
            fill={color}
            opacity={op}
          />
        );
      })}
    </g>
  );
};

// ============================================================================
// SCENE 1 (0s - 26.67s): 序幕 · 床边故事与冰封八音盒 (Intro)
// ============================================================================
export const Scene1Intro: React.FC<{
  frame: number;
  audio: FrameAudio;
}> = ({frame, audio}) => {
  const openProgress = easeOut(frame, 0, 90);
  const ringRot1 = frame * 0.18;
  const ringRot2 = -frame * 0.12;
  const pulse = 1 + audio.bass * 0.045 + audio.high * 0.03;
  const coverUrl = staticFile('mv-bugai/cover.jpg');

  return (
    <g>
      <defs>
        <clipPath id="intro-cover-clip">
          <circle cx={0} cy={0} r={168} />
        </clipPath>
      </defs>

      {/* Distant Gothic Cathedral / Storybook Arch Silhouette */}
      <g opacity={0.28 + audio.rms * 0.2}>
        <path
          d="M 260 980 L 260 360 Q 260 110 960 70 Q 1660 110 1660 360 L 1660 980"
          fill="none"
          stroke="#7DD3FC"
          strokeWidth={2}
          strokeDasharray="10 14"
        />
        <path
          d="M 340 980 L 340 390 Q 340 160 960 120 Q 1580 160 1580 390 L 1580 980"
          fill="none"
          stroke="#FDE68A"
          strokeWidth={1.2}
          opacity={0.5}
        />
      </g>

      {/* Ornate Celestial Music-Box / Clockwork Rings around Album Medallion */}
      <g
        transform={`translate(960 435) scale(${(
          (0.88 + openProgress * 0.12) *
          pulse
        ).toFixed(3)})`}
      >
        {/* Outer Halo */}
        <circle
          cx={0}
          cy={0}
          r={310 + audio.rms * 25}
          fill="none"
          stroke="#38BDF8"
          strokeWidth={1.5}
          opacity={0.22 + audio.high * 0.35}
        />
        <g transform={`rotate(${ringRot1.toFixed(2)})`}>
          <circle
            cx={0}
            cy={0}
            r={272}
            fill="none"
            stroke="#FDE68A"
            strokeWidth={2}
            strokeDasharray="4 18 28 18"
            opacity={0.55}
          />
          {[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330].map(
            (deg) => (
              <g key={deg} transform={`rotate(${deg})`}>
                <line
                  x1={0}
                  y1={-252}
                  x2={0}
                  y2={-288}
                  stroke="#FDE68A"
                  strokeWidth={1.6}
                  opacity={0.45}
                />
                <circle cx={0} cy={-296} r={3} fill="#7DD3FC" opacity={0.7} />
              </g>
            )
          )}
        </g>

        <g transform={`rotate(${ringRot2.toFixed(2)})`}>
          <SnowflakeGlyph
            cx={0}
            cy={0}
            r={235}
            rot={0}
            color="#BAE6FD"
            opacity={0.32 + audio.high * 0.3}
            strokeWidth={2}
          />
        </g>

        {/* Circular Sound-Wave Bars Responding to Music-Box Piano Notes */}
        {Array.from({length: 48}).map((_, i) => {
          const deg = (i / 48) * 360;
          const wave = Math.abs(Math.sin(frame * 0.08 + i * 0.65));
          const len = 10 + (audio.high * 42 + audio.mid * 28) * wave;
          return (
            <g key={i} transform={`rotate(${deg})`}>
              <line
                x1={0}
                y1={-192}
                x2={0}
                y2={-192 - len}
                stroke={i % 2 === 0 ? '#7DD3FC' : '#FDE68A'}
                strokeWidth={2.5}
                strokeLinecap="round"
                opacity={0.45 + wave * 0.45}
              />
            </g>
          );
        })}

        {/* Gold Medallion Backing */}
        <circle
          cx={0}
          cy={0}
          r={174}
          fill="#0F172A"
          stroke="#FDE68A"
          strokeWidth={4}
          opacity={0.95}
        />

        {/* Native SVG Image with Circular Clip */}
        <g
          clipPath="url(#intro-cover-clip)"
          transform={`rotate(${(frame * 0.06).toFixed(2)})`}
        >
          <image
            href={coverUrl}
            x={-168}
            y={-168}
            width={336}
            height={336}
            preserveAspectRatio="xMidYMid slice"
          />
        </g>
        <circle
          cx={0}
          cy={0}
          r={168}
          fill="none"
          stroke="#7DD3FC"
          strokeWidth={2}
          opacity={0.6}
        />
      </g>
    </g>
  );
};

// ============================================================================
// SCENE 2 (26.67s - 50.73s): 独白 I · 杰伦 —— 雪夜街灯与假装的电影帧
// ============================================================================
export const Scene2JayVerse: React.FC<{
  frame: number;
  localFrame: number;
  audio: FrameAudio;
}> = ({frame, localFrame, audio}) => {
  const enter = easeOut(localFrame, 0, 45);
  const lampFlicker = 0.86 + Math.sin(frame * 0.37) * 0.06 + audio.mid * 0.12;
  const phantomOpacity = interpolate(
    localFrame,
    [0, 260, 460, 650],
    [0.72, 0.55, 0.18, 0],
    {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    }
  );
  const scarfWave = Math.sin(frame * 0.07);

  return (
    <g opacity={enter}>
      {/* Snowy Ground Horizon */}
      <path
        d="M 0 770 Q 520 735 960 760 Q 1460 785 1920 750 L 1920 1080 L 0 1080 Z"
        fill="#081121"
        stroke="#38BDF8"
        strokeWidth={1.5}
        opacity={0.85}
      />

      {/* Vintage Film Strip Frame in Sky ("假装我们还在一块 / 我真的演不出来") */}
      <g
        transform={`translate(1080 165) rotate(-2.5)`}
        opacity={0.58 + audio.mid * 0.28}
      >
        <rect
          x={0}
          y={0}
          width={610}
          height={350}
          rx={14}
          fill="#0B1325"
          stroke="#7DD3FC"
          strokeWidth={2.5}
        />
        {/* Film sprockets */}
        {Array.from({length: 10}).map((_, i) => (
          <React.Fragment key={i}>
            <rect
              x={22 + i * 58}
              y={14}
              width={30}
              height={16}
              rx={3}
              fill="#38BDF8"
              opacity={0.45}
            />
            <rect
              x={22 + i * 58}
              y={320}
              width={30}
              height={16}
              rx={3}
              fill="#38BDF8"
              opacity={0.45}
            />
          </React.Fragment>
        ))}
        {/* Inside the film frame: Two figures together in warm memory */}
        <rect
          x={28}
          y={44}
          width={554}
          height={260}
          fill="#152238"
          opacity={0.78}
        />
        <circle
          cx={305}
          cy={165}
          r={92 + audio.mid * 22}
          fill="#FDE68A"
          opacity={0.16 * phantomOpacity}
        />

        {/* Artistic Silhouettes inside memory reel */}
        <g transform="translate(265 258) scale(0.76)">
          <ArtisticFigureJay glowColor="#7DD3FC" scarfWave={scarfWave * 0.5} />
          <g transform="translate(95 0)" opacity={phantomOpacity}>
            <ArtisticFigureAmei glowColor="#FDA4AF" hairWave={scarfWave * 0.5} />
            <path
              d="M -68 -68 Q -42 -48 -18 -68"
              stroke="#FDE68A"
              strokeWidth={3.5}
              fill="none"
            />
          </g>
        </g>

        <text
          x={305}
          y={288}
          textAnchor="middle"
          fill="#BAE6FD"
          fontSize={16}
          letterSpacing={4}
          opacity={0.8}
        >
          MEMORY REEL · SCENE 01 // 假装我们还在一块
        </text>
      </g>

      {/* Streetlamp & Volumetric Warm Light Cone on the Left */}
      <g transform="translate(470 0)">
        <polygon
          points="88,215 -290,785 380,785"
          fill="#FDE68A"
          opacity={0.09 * lampFlicker}
        />
        <polygon
          points="88,215 -150,785 250,785"
          fill="#7DD3FC"
          opacity={0.08 * lampFlicker}
        />
        {/* Lamp post */}
        <line
          x1={0}
          y1={190}
          x2={0}
          y2={775}
          stroke="#94A3B8"
          strokeWidth={7}
        />
        <path
          d="M 0 230 Q 45 175 90 210"
          fill="none"
          stroke="#94A3B8"
          strokeWidth={6}
        />
        {/* Glowing Lantern */}
        <circle
          cx={92}
          cy={220}
          r={32 + audio.rms * 14}
          fill="#FDE68A"
          opacity={0.32 * lampFlicker}
        />
        <polygon
          points="78,206 106,206 100,238 84,238"
          fill="#FEF08A"
          stroke="#F59E0B"
          strokeWidth={2}
        />

        {/* Empty Wooden Park Bench in the Snow */}
        <g transform="translate(135 705)">
          <line x1={0} y1={0} x2={205} y2={0} stroke="#64748B" strokeWidth={5} />
          <line
            x1={-8}
            y1={24}
            x2={215}
            y2={24}
            stroke="#94A3B8"
            strokeWidth={6}
          />
          <line x1={18} y1={0} x2={12} y2={58} stroke="#475569" strokeWidth={5} />
          <line
            x1={188}
            y1={0}
            x2={194}
            y2={58}
            stroke="#475569"
            strokeWidth={5}
          />
        </g>

        {/* Jay's Artistic Silhouette Standing Under the Lamp */}
        <g transform="translate(-45 745) scale(1.15)">
          <ellipse cx={0} cy={20} rx={65} ry={10} fill="#020617" opacity={0.7} />
          <ArtisticFigureJay glowColor="#7DD3FC" scarfWave={scarfWave} />
        </g>
      </g>
    </g>
  );
};

// ============================================================================
// SCENE 3 (50.73s - 76.0s): 独白 II · 阿妹 —— 霜窗暗涌与一片空白
// ============================================================================
export const Scene3AmeiVerse: React.FC<{
  frame: number;
  localFrame: number;
  audio: FrameAudio;
}> = ({frame, localFrame, audio}) => {
  const whiteout = interpolate(
    localFrame,
    [575, 625, 710, 755],
    [0, 0.72, 0.58, 0.12],
    {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    }
  );

  return (
    <g>
      {/* Emotional Rose-Crimson Ocean Waves Surging ("是我情绪涌了上来") */}
      {[0, 1, 2, 3].map((w) => {
        const yBase = 590 + w * 58 - audio.rms * 65;
        const phase = frame * (0.045 + w * 0.01) + w * 1.3;
        const amp = 32 + audio.bass * 75 + w * 12;
        const d = `M 0 ${yBase} Q 320 ${
          yBase - Math.sin(phase) * amp
        }, 640 ${yBase + Math.cos(phase) * amp} T 1280 ${
          yBase - Math.sin(phase * 0.8) * amp
        } T 1920 ${
          yBase + Math.cos(phase * 1.1) * amp
        } L 1920 1080 L 0 1080 Z`;
        return (
          <path
            key={w}
            d={d}
            fill={w % 2 === 0 ? '#FB7185' : '#E11D48'}
            opacity={(0.14 + audio.mid * 0.18) * (1 - w * 0.16)}
          />
        );
      })}

      {/* Center Frosted Cathedral Window Frame */}
      <g transform="translate(960 435)">
        <rect
          x={-520}
          y={-285}
          width={1040}
          height={570}
          rx={24}
          fill="#0F172A"
          fillOpacity={0.38}
          stroke="#FDA4AF"
          strokeWidth={2.5}
          strokeOpacity={0.55}
        />
        <line
          x1={0}
          y1={-285}
          x2={0}
          y2={285}
          stroke="#FDA4AF"
          strokeWidth={2}
          opacity={0.35}
        />
        <line
          x1={-520}
          y1={0}
          x2={520}
          y2={0}
          stroke="#FDA4AF"
          strokeWidth={2}
          opacity={0.35}
        />

        {/* Frost Crystals Growing in the Window Corners */}
        <SnowflakeGlyph
          cx={-460}
          cy={-225}
          r={62}
          rot={frame * 0.15}
          color="#BAE6FD"
          opacity={0.45}
        />
        <SnowflakeGlyph
          cx={460}
          cy={-225}
          r={62}
          rot={-frame * 0.15}
          color="#BAE6FD"
          opacity={0.45}
        />
        <SnowflakeGlyph
          cx={-460}
          cy={225}
          r={62}
          rot={-frame * 0.12}
          color="#FDA4AF"
          opacity={0.45}
        />
        <SnowflakeGlyph
          cx={460}
          cy={225}
          r={62}
          rot={frame * 0.12}
          color="#FDA4AF"
          opacity={0.45}
        />

        {/* Vocal Aura Rings & aMEI Silhouette */}
        {[1, 2, 3].map((ring) => (
          <circle
            key={ring}
            cx={0}
            cy={10}
            r={105 + ring * 52 + audio.mid * 48 + audio.bass * 35}
            fill="none"
            stroke={ring % 2 === 0 ? '#FDA4AF' : '#FDE68A'}
            strokeWidth={2.2 - ring * 0.4}
            opacity={(0.42 - ring * 0.1) * (0.6 + audio.rms * 0.6)}
          />
        ))}

        <g transform="translate(0 120) scale(1.25)">
          <ArtisticFigureAmei
            glowColor="#FDA4AF"
            hairWave={Math.sin(frame * 0.06)}
          />
        </g>

        {/* Rising Tear / Emotion Pearls */}
        {Array.from({length: 18}).map((_, i) => {
          const px = (hash(i * 17) - 0.5) * 760;
          const py =
            240 - ((localFrame * (1.8 + hash(i * 31) * 1.6) + i * 40) % 480);
          return (
            <circle
              key={i}
              cx={px}
              cy={py}
              r={3 + hash(i) * 4 + audio.high * 3}
              fill={i % 2 === 0 ? '#FDA4AF' : '#FFF1F2'}
              opacity={0.4 + audio.mid * 0.4}
            />
          );
        })}
      </g>

      {/* "想哭却一遍空白" Whiteout Veil */}
      {whiteout > 0.01 && (
        <rect
          x={0}
          y={0}
          width={1920}
          height={1080}
          fill="#F8FAFC"
          opacity={whiteout}
        />
      )}
    </g>
  );
};

// ============================================================================
// SCENE 4 (76.0s - 101.0s): 预副歌 · 零下结晶的誓言与融化 (Pre-Chorus 1)
// ============================================================================
export const Scene4CrystalVow: React.FC<{
  frame: number;
  localFrame: number;
  audio: FrameAudio;
  shatterBoost?: number;
}> = ({frame, localFrame, audio, shatterBoost = 0}) => {
  const crackProgress = clamp01((localFrame - 240) / 240 + shatterBoost * 0.6);
  const meltProgress = clamp01((localFrame - 450) / 260 + shatterBoost * 0.8);
  const crystalScale = 1 + audio.bass * 0.09 + audio.rms * 0.05;

  return (
    <g>
      {/* Dual Intertwining Vocal Ribbons */}
      {Array.from({length: 2}).map((_, s) => {
        const isJay = s === 0;
        const color = isJay ? '#38BDF8' : '#FB7185';
        const pts: string[] = [];
        for (let x = 80; x <= 1840; x += 40) {
          const norm = (x - 960) / 880;
          const envelope = 1 - Math.abs(norm) * 0.55;
          const y =
            435 +
            Math.sin(
              x * 0.008 + frame * 0.07 * (isJay ? 1 : -1) + s * Math.PI
            ) *
              (90 + audio.mid * 90) *
              envelope;
          pts.push(`${x},${y.toFixed(1)}`);
        }
        return (
          <polyline
            key={s}
            points={pts.join(' ')}
            fill="none"
            stroke={color}
            strokeWidth={4 + audio.rms * 4}
            strokeLinecap="round"
            opacity={0.65}
          />
        );
      })}

      {/* Centerpiece: Giant 6-Fold Crystalline Vow */}
      <g transform={`translate(960 430) scale(${crystalScale.toFixed(3)})`}>
        <circle
          cx={0}
          cy={0}
          r={235 + audio.rms * 35}
          fill="#38BDF8"
          opacity={0.11 + audio.high * 0.12}
        />
        <circle
          cx={0}
          cy={0}
          r={165 + audio.bass * 25}
          fill="#FB7185"
          opacity={0.1 + audio.mid * 0.14}
        />

        <g
          transform={`rotate(${(frame * 0.25).toFixed(2)})`}
          opacity={1 - meltProgress * 0.45}
        >
          <SnowflakeGlyph
            cx={0}
            cy={0}
            r={215}
            rot={0}
            color="#E0F2FE"
            opacity={0.85}
            strokeWidth={3.2}
          />
          <SnowflakeGlyph
            cx={0}
            cy={0}
            r={148}
            rot={30}
            color="#7DD3FC"
            opacity={0.75}
            strokeWidth={2.4}
          />
          <polygon
            points="0,-180 156,-90 156,90 0,180 -156,90 -156,-90"
            fill="#0EA5E9"
            fillOpacity={0.14}
            stroke="#BAE6FD"
            strokeWidth={2.5}
          />
          <polygon
            points="0,-110 95,-55 95,55 0,110 -95,55 -95,-55"
            fill="#F43F5E"
            fillOpacity={0.12}
            stroke="#FDA4AF"
            strokeWidth={2}
          />
        </g>

        {/* Glowing Fracture / Crack Lines */}
        {crackProgress > 0.02 && (
          <g opacity={crackProgress}>
            {[0, 60, 120, 180, 240, 300].map((deg, idx) => {
              const drift = meltProgress * (28 + (idx % 2) * 22);
              return (
                <g
                  key={deg}
                  transform={`rotate(${deg}) translate(0 ${-drift.toFixed(1)})`}
                >
                  <path
                    d="M 0 0 L 14 -45 L -9 -92 L 18 -145 L -6 -180"
                    fill="none"
                    stroke="#FDE68A"
                    strokeWidth={2.8}
                    strokeLinecap="round"
                  />
                  {meltProgress > 0.1 && (
                    <polygon
                      points="-12,-120 16,-135 4,-168"
                      fill="#E0F2FE"
                      stroke="#7DD3FC"
                      strokeWidth={1.5}
                      opacity={0.85 - meltProgress * 0.4}
                      transform={`translate(${(meltProgress * 35).toFixed(
                        1
                      )} ${(-meltProgress * 45).toFixed(1)}) rotate(${(
                        meltProgress * 45
                      ).toFixed(1)})`}
                    />
                  )}
                </g>
              );
            })}
          </g>
        )}

        {/* Melting Luminous Water Droplets */}
        {meltProgress > 0.05 &&
          Array.from({length: 16}).map((_, i) => {
            const dropX = (hash(i * 23) - 0.5) * 260;
            const dropCycle = (frame * 3.8 + i * 28) % 300;
            const dropY = 55 + dropCycle;
            const alpha = (1 - dropCycle / 300) * meltProgress;
            return (
              <path
                key={i}
                d={`M ${dropX.toFixed(1)} ${(dropY - 12).toFixed(1)} C ${(
                  dropX + 6
                ).toFixed(1)} ${dropY.toFixed(1)}, ${(dropX - 6).toFixed(
                  1
                )} ${dropY.toFixed(1)}, ${dropX.toFixed(1)} ${(
                  dropY - 12
                ).toFixed(1)} Z`}
                fill={i % 2 === 0 ? '#7DD3FC' : '#FDA4AF'}
                opacity={alpha}
              />
            );
          })}
      </g>
    </g>
  );
};

// ============================================================================
// SCENE 5 (101.0s - 154.7s): 副歌 I · 雪融花开与松开的手 (Chorus 1 Climax)
// ============================================================================
export const Scene5ChorusBloomAndHands: React.FC<{
  frame: number;
  localFrame: number;
  audio: FrameAudio;
}> = ({frame, localFrame, audio}) => {
  const handPhase = easeInOut(localFrame, 750, 90);
  const bloomProgress = easeOut(localFrame, 60, 480);
  const unclasp = easeInOut(localFrame, 1220, 220);

  return (
    <g>
      {/* PART A: Memory Prism & Winter-to-Spring Blossoming Tree (shifted up for clean lyric clearance) */}
      <g
        opacity={1 - handPhase * 0.88}
        transform="translate(960 420) scale(0.86)"
      >
        <g transform={`rotate(${(Math.sin(frame * 0.015) * 5).toFixed(2)})`}>
          <polygon
            points="0,-340 310,-170 310,150 0,300 -310,150 -310,-170"
            fill="#0F172A"
            fillOpacity={0.36}
            stroke="#7DD3FC"
            strokeWidth={2.5}
          />
          <polyline
            points="-310,-170 0,-10 310,-170"
            fill="none"
            stroke="#BAE6FD"
            strokeWidth={1.8}
            opacity={0.45}
          />
          <line
            x1={0}
            y1={-10}
            x2={0}
            y2={300}
            stroke="#BAE6FD"
            strokeWidth={1.8}
            opacity={0.45}
          />
        </g>

        <ellipse
          cx={0}
          cy={245}
          rx={350}
          ry={36}
          fill={interpolateColors(
            bloomProgress,
            [0, 1],
            ['#1E293B', '#31102F']
          )}
          opacity={0.75}
        />

        <path
          d="M 0 245 C -8 150, -18 75, 0 -20 C -45 -85, -125 -135, -185 -195 M 0 -20 C 45 -90, 130 -140, 195 -195 M -75 -95 C -95 -155, -70 -215, -110 -255 M 75 -95 C 95 -155, 80 -210, 120 -255 M 0 -20 C 5 -110, -10 -190, 0 -265"
          fill="none"
          stroke="#E2E8F0"
          strokeWidth={9}
          strokeLinecap="round"
        />

        {[
          [-185, -195],
          [-140, -155],
          [-110, -255],
          [-65, -190],
          [-45, -120],
          [0, -265],
          [15, -210],
          [-25, -235],
          [55, -145],
          [120, -255],
          [195, -195],
          [150, -160],
          [85, -195],
          [-95, -95],
          [95, -95],
          [-155, -230],
          [160, -230],
          [0, -155],
        ].map(([bx, by], i) => {
          const itemOpen = clamp01(bloomProgress * 1.4 - i * 0.03);
          const r =
            (12 + (i % 4) * 5) * itemOpen * (1 + audio.bass * 0.22);
          if (r <= 0.5) return null;
          return (
            <g key={i} transform={`translate(${bx} ${by})`}>
              <circle
                cx={0}
                cy={0}
                r={r * 1.7}
                fill="#FB7185"
                opacity={0.22 * itemOpen}
              />
              {[0, 72, 144, 216, 288].map((deg) => (
                <ellipse
                  key={deg}
                  cx={0}
                  cy={-r * 0.55}
                  rx={r * 0.42}
                  ry={r * 0.62}
                  fill={i % 2 === 0 ? '#FDA4AF' : '#FBCFE8'}
                  transform={`rotate(${deg + frame * 0.3})`}
                />
              ))}
              <circle cx={0} cy={0} r={r * 0.25} fill="#FDE68A" />
            </g>
          );
        })}
      </g>

      {/* PART B: Two Constellation Light-Hands Unclasping */}
      {handPhase > 0.02 && (
        <g opacity={handPhase} transform="translate(960 425)">
          <g opacity={1 - unclasp * 0.85}>
            {[-2, -1, 0, 1, 2].map((k) => {
              const yOff = k * 28;
              const wave =
                Math.sin(frame * 0.09 + k) * (18 + audio.mid * 35);
              return (
                <path
                  key={k}
                  d={`M ${-220 - unclasp * 180} ${yOff} Q 0 ${
                    yOff + wave
                  } ${220 + unclasp * 180} ${yOff}`}
                  fill="none"
                  stroke={k % 2 === 0 ? '#FB7185' : '#FDE68A'}
                  strokeWidth={2.5}
                  strokeDasharray="8 8"
                />
              );
            })}
          </g>

          <g
            transform={`translate(${(-55 - unclasp * 235).toFixed(1)} ${(
              unclasp * 18
            ).toFixed(1)}) rotate(${(-unclasp * 12).toFixed(2)})`}
          >
            <path
              d="M -380 90 L -150 40 Q -65 15 -15 -18 Q 25 -42 42 -26 Q 50 -14 15 12 L 58 -8 Q 76 0 58 18 L 45 28 Q 68 35 48 52 L 15 62 Q 35 72 15 86 L -120 115 L -380 170"
              fill="#0EA5E9"
              fillOpacity={0.22}
              stroke="#7DD3FC"
              strokeWidth={4}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <circle cx={12} cy={12} r={8 + audio.rms * 8} fill="#E0F2FE" />
          </g>

          <g
            transform={`translate(${(55 + unclasp * 235).toFixed(1)} ${(
              -unclasp * 18
            ).toFixed(1)}) rotate(${(unclasp * 12).toFixed(2)})`}
          >
            <path
              d="M 380 -90 L 150 -40 Q 65 -15 15 18 Q -25 42 -42 26 Q -50 14 -15 -12 L -58 8 Q -76 0 -58 -18 L -45 -28 Q -68 -35 -48 -52 L -15 -62 Q -35 -72 -15 -86 L 120 -115 L 380 -170"
              fill="#F43F5E"
              fillOpacity={0.22}
              stroke="#FDA4AF"
              strokeWidth={4}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <circle cx={-12} cy={-12} r={8 + audio.rms * 8} fill="#FFE4E6" />
          </g>

          {unclasp > 0.05 &&
            Array.from({length: 24}).map((_, i) => {
              const angle = (i / 24) * Math.PI * 2 + frame * 0.015;
              const dist =
                unclasp * (60 + (i % 5) * 68) + audio.rms * 25;
              const px = Math.cos(angle) * dist * 1.3;
              const py = Math.sin(angle) * dist * 0.85 - unclasp * 85;
              return (
                <circle
                  key={i}
                  cx={px.toFixed(1)}
                  cy={py.toFixed(1)}
                  r={3 + (i % 3) * 2.2}
                  fill={i % 2 === 0 ? '#FDE68A' : '#7DD3FC'}
                  opacity={(1 - unclasp * 0.25) * 0.85}
                />
              );
            })}
        </g>
      )}
    </g>
  );
};

// ============================================================================
// SCENE 6 (154.7s - 191.5s): 间奏与第二主歌 · 逆时怀表与释怀之湖
// ============================================================================
export const Scene6ClockAndLake: React.FC<{
  frame: number;
  localFrame: number;
  audio: FrameAudio;
}> = ({frame, localFrame, audio}) => {
  const isLakePhase = easeInOut(localFrame, 710, 75);
  const ccwAngle = -frame * (localFrame < 365 ? 2.4 : 0.65);

  return (
    <g>
      <g
        transform={`translate(960 ${410 - isLakePhase * 70}) scale(${
          0.92 - isLakePhase * 0.18
        })`}
        opacity={1 - isLakePhase * 0.35}
      >
        <circle
          cx={0}
          cy={0}
          r={250 + audio.bass * 28}
          fill="#0F172A"
          fillOpacity={0.55}
          stroke="#FDE68A"
          strokeWidth={3}
        />
        <circle
          cx={0}
          cy={0}
          r={218}
          fill="none"
          stroke="#7DD3FC"
          strokeWidth={1.5}
          strokeDasharray="6 14"
        />

        {Array.from({length: 12}).map((_, h) => {
          const deg = h * 30;
          return (
            <g key={h} transform={`rotate(${deg})`}>
              <line
                x1={0}
                y1={-218}
                x2={0}
                y2={-242}
                stroke="#FDE68A"
                strokeWidth={3}
              />
            </g>
          );
        })}

        <g transform={`rotate(${ccwAngle.toFixed(1)})`}>
          <line
            x1={0}
            y1={35}
            x2={0}
            y2={-195}
            stroke="#7DD3FC"
            strokeWidth={4.5}
            strokeLinecap="round"
          />
        </g>
        <g transform={`rotate(${(ccwAngle * 0.25).toFixed(1)})`}>
          <line
            x1={0}
            y1={22}
            x2={0}
            y2={-130}
            stroke="#FDA4AF"
            strokeWidth={6.5}
            strokeLinecap="round"
          />
        </g>
        <circle cx={0} cy={0} r={12} fill="#FDE68A" />

        <g transform={`rotate(${(Math.sin(frame * 0.08) * 22).toFixed(2)})`}>
          <line
            x1={0}
            y1={0}
            x2={0}
            y2={280}
            stroke="#FDE68A"
            strokeWidth={2.5}
            opacity={0.7}
          />
          <circle
            cx={0}
            cy={280}
            r={22 + audio.high * 10}
            fill="#FDE68A"
            opacity={0.85}
          />
        </g>
      </g>

      {isLakePhase > 0.02 && (
        <g opacity={isLakePhase} transform="translate(960 685)">
          {[1, 2, 3, 4, 5].map((r) => {
            const prog = ((frame * 0.7 + r * 38) % 200) / 200;
            const rx = 120 + prog * 680;
            const ry = 20 + prog * 75;
            return (
              <ellipse
                key={r}
                cx={0}
                cy={0}
                rx={rx}
                ry={ry}
                fill="none"
                stroke={r % 2 === 0 ? '#7DD3FC' : '#FDA4AF'}
                strokeWidth={2.2}
                opacity={(1 - prog) * 0.65}
              />
            );
          })}
        </g>
      )}
    </g>
  );
};

// ============================================================================
// SCENE 7 (191.5s - 270.0s): 终章大副歌 · 宿命绝唱与"最爱"超新星爆发
// ============================================================================
export const Scene7FinalClimax: React.FC<{
  frame: number;
  localFrame: number;
  audio: FrameAudio;
}> = ({frame, localFrame, audio}) => {
  const suspension =
    localFrame >= 1560 && localFrame < 1848
      ? easeInOut(localFrame, 1560, 45) * (1 - easeInOut(localFrame, 1825, 23))
      : 0;
  const supernova = localFrame >= 1845 ? easeOut(localFrame, 1845, 35) : 0;
  const walkApart = clamp01((localFrame - 750) / 1400);
  const breeze = Math.sin(frame * 0.06);

  return (
    <g>
      {/* Volumetric God-Rays & Supernova Burst at 04:13.16 ("最爱") */}
      {supernova > 0.01 && (
        <g transform="translate(960 395)" opacity={supernova}>
          {Array.from({length: 24}).map((_, i) => {
            const deg = i * 15 + frame * 0.22;
            const len = 650 + audio.rms * 380;
            return (
              <g key={i} transform={`rotate(${deg})`}>
                <polygon
                  points={`0,0 -28,-${len} 28,-${len}`}
                  fill={
                    i % 3 === 0
                      ? '#FDE68A'
                      : i % 3 === 1
                      ? '#7DD3FC'
                      : '#FDA4AF'
                  }
                  opacity={0.12 + audio.high * 0.14}
                />
              </g>
            );
          })}
          {[0, 1, 2].map((k) => {
            const r = ((localFrame - 1845 + k * 45) % 180) * 5.8;
            const op = Math.max(0, 1 - r / 980);
            return (
              <circle
                key={k}
                cx={0}
                cy={0}
                r={Math.max(1, r)}
                fill="none"
                stroke={k === 0 ? '#FDE68A' : '#E0F2FE'}
                strokeWidth={4}
                opacity={op * 0.75}
              />
            );
          })}
        </g>
      )}

      {/* Center Shattered Crystal & Blooming Star Mandala */}
      <g
        transform={`translate(960 385) scale(${
          0.95 + audio.bass * 0.12 + supernova * 0.18
        })`}
      >
        <SnowflakeGlyph
          cx={0}
          cy={0}
          r={210}
          rot={frame * (1 - suspension * 0.85) * 0.35}
          color={supernova > 0.3 ? '#FDE68A' : '#BAE6FD'}
          opacity={0.68}
          strokeWidth={2.6}
        />
        <SnowflakeGlyph
          cx={0}
          cy={0}
          r={142}
          rot={-frame * (1 - suspension * 0.85) * 0.45}
          color="#FDA4AF"
          opacity={0.68}
          strokeWidth={2.2}
        />
      </g>

      {/* Horizon Bridge & Two Artistic Silhouettes Walking Apart */}
      <g transform="translate(960 725)">
        <ellipse
          cx={0}
          cy={32}
          rx={840}
          ry={36}
          fill="#0B1325"
          stroke="#38BDF8"
          strokeWidth={1.8}
          opacity={0.75}
        />

        {/* Golden Thread of Memory Connecting Them Even After Letting Go */}
        <path
          d={`M ${(-90 - walkApart * 340).toFixed(1)} -75 Q 0 ${(
            -210 -
            supernova * 75
          ).toFixed(1)} ${(90 + walkApart * 340).toFixed(1)} -75`}
          fill="none"
          stroke="#FDE68A"
          strokeWidth={2.4}
          strokeDasharray="6 10"
          opacity={0.78}
        />

        {/* Jay Artistic Silhouette walking left */}
        <g
          transform={`translate(${(-90 - walkApart * 340).toFixed(
            1
          )} 10) scale(0.85)`}
        >
          <ArtisticFigureJay glowColor="#7DD3FC" scarfWave={breeze} />
        </g>

        {/* aMEI Artistic Silhouette walking right */}
        <g
          transform={`translate(${(90 + walkApart * 340).toFixed(
            1
          )} 10) scale(0.85)`}
        >
          <ArtisticFigureAmei glowColor="#FDA4AF" hairWave={breeze} />
        </g>
      </g>
    </g>
  );
};

// ============================================================================
// SCENE 8 (270.0s - 291.41s): 尾奏 · 故事合上，雪落无声 (Outro)
// ============================================================================
export const Scene8Outro: React.FC<{
  frame: number;
  localFrame: number;
  audio: FrameAudio;
}> = ({frame, localFrame, audio}) => {
  const fadeOut = 1 - easeInOut(localFrame, 420, 210);
  const coverUrl = staticFile('mv-bugai/cover.jpg');

  return (
    <g opacity={fadeOut}>
      <defs>
        <clipPath id="outro-cover-clip">
          <circle cx={0} cy={0} r={120} />
        </clipPath>
      </defs>
      <g transform="translate(960 425)">
        <circle
          cx={0}
          cy={0}
          r={210 + audio.rms * 20}
          fill="none"
          stroke="#FDE68A"
          strokeWidth={1.8}
          strokeDasharray="8 16"
          opacity={0.45}
        />
        <SnowflakeGlyph
          cx={0}
          cy={0}
          r={185}
          rot={frame * 0.12}
          color="#7DD3FC"
          opacity={0.35}
        />
        <circle
          cx={0}
          cy={0}
          r={124}
          fill="#0F172A"
          stroke="#FDE68A"
          strokeWidth={3}
        />
        <g clipPath="url(#outro-cover-clip)">
          <image
            href={coverUrl}
            x={-120}
            y={-120}
            width={240}
            height={240}
            preserveAspectRatio="xMidYMid slice"
          />
        </g>
      </g>
    </g>
  );
};
