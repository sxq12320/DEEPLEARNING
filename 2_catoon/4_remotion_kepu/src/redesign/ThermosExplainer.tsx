import React from 'react';
import {
  AbsoluteFill, Audio, Easing, interpolate, interpolateColors,
  Sequence, staticFile, useCurrentFrame,
} from 'remotion';
import timeline from './thermos-timeline.json';

// One object, six explanatory beats. No dependence on playback history or wall time.
const P = {
  paper: '#F4F0E7', ink: '#202E38', muted: '#65747A', blue: '#28647C',
  blueDark: '#17465A', paleBlue: '#E0E9E9', coral: '#D66A4D',
  warm: '#F0B393', metal: '#CED7D5', metalLight: '#F7F8F2', cold: '#80BACC',
};
const FONT = '"Microsoft YaHei", "PingFang SC", sans-serif';
const p = (frame: number, start: number, duration: number) => interpolate(
  frame, [start, start + duration], [0, 1],
  {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.inOut(Easing.cubic)},
);

const body = 'M -126 22 L -126 65 Q -212 96 -212 162 L -212 495 Q -212 570 -142 578 L 142 578 Q 212 570 212 495 L 212 162 Q 212 96 126 65 L 126 22 Z';
const liner = 'M -102 58 L -102 114 Q -157 132 -157 173 L -157 488 Q -157 528 -112 532 L 112 532 Q 157 528 157 488 L 157 173 Q 157 132 102 114 L 102 58 Z';

const Pointer = ({points, label, x, y, color = P.ink, anchor = 'start'}: {
  points: string; label: string; x: number; y: number; color?: string;
  anchor?: 'start' | 'middle' | 'end';
}) => (
  <g fill="none" stroke={color} strokeWidth={3} strokeLinecap="round" strokeLinejoin="round">
    <polyline points={points} />
    <text x={x} y={y} fill={color} stroke="none" fontFamily={FONT}
      fontWeight={600} fontSize={34} textAnchor={anchor}>{label}</text>
  </g>
);

const Flask = ({cut, cold, leak, inward, frame, lidOpen}: {
  cut: number; cold: number; leak: number; inward: boolean; frame: number;
  lidOpen: number;
}) => {
  const waterColor = interpolateColors(cold, [0, 1], [P.warm, P.cold]);
  // Heat has the same color in both directions; blue denotes the cold liquid.
  const flowColor = P.coral;
  const phase = (frame % 85) / 85;
  const flowY = inward ? -123 + phase * 386 : 263 - phase * 386;
  return (
    <g transform="translate(1254 262)">
      <defs>
        <clipPath id="thermos-liquid"><path d={liner} /></clipPath>
      </defs>
      <ellipse cx={0} cy={603} rx={241} ry={21} fill={P.ink} opacity={0.09} />
      <path d={body} fill={P.paper} stroke={P.ink} strokeWidth={8} strokeLinejoin="round" />
      <g opacity={cut}>
        <path d={liner} fill={P.metalLight} stroke={P.ink} strokeWidth={7} />
        <g clipPath="url(#thermos-liquid)">
          <path d="M-157 247 Q-81 235 0 247 Q81 259 157 247 L157 533 L-157 533 Z"
            fill={waterColor} />
          <path d="M-157 247 Q-81 235 0 247 Q81 259 157 247"
            fill="none" stroke={cold > 0.5 ? P.blue : P.coral} strokeWidth={5} />
          <path d="M-125 284 L-125 474 Q-125 500 -99 500" stroke="#FFF"
            strokeWidth={7} opacity={0.35} fill="none" strokeLinecap="round" />
        </g>
        <path d="M-166 190 L-166 486 Q-166 541 -116 544 L116 544 Q166 541 166 486 L166 190"
          stroke={P.metal} strokeWidth={7} fill="none" />
        <path d="M-201 165 L-201 486 Q-201 556 -140 564 L140 564 Q201 556 201 486 L201 165"
          stroke={P.metal} strokeWidth={6} fill="none" />
      </g>
      <g opacity={1 - cut}>
        <path d={body} fill={P.blue} stroke={P.ink} strokeWidth={8} />
        <path d="M-173 184 L-173 480 Q-173 526 -136 531"
          fill="none" stroke="#6593A3" strokeWidth={14} strokeLinecap="round" />
        <path d="M142 157 L142 494 Q142 537 103 545" fill="none"
          stroke={P.blueDark} strokeWidth={23} strokeLinecap="round" />
        <text x={0} y={306} textAnchor="middle" fontFamily={FONT} fontSize={35}
          fontWeight={700} fill={P.paper}>保温杯</text>
        <path d="M-34 346 L34 346" stroke={P.paper} strokeWidth={3} opacity={0.5} />
      </g>
      <g opacity={lidOpen * 0.65}>
        {[-50, 0, 50].map((x, i) => (
          <path key={x} d={`M${x} 5 q-17 -26 0 -52 t0 -52`}
            transform={`translate(0 ${-p(frame, 0, 60) * (i + 1) * 8})`}
            stroke={P.muted} strokeWidth={4} fill="none" strokeLinecap="round" />
        ))}
      </g>
      <g transform={`translate(0 ${-lidOpen * 115})`}>
        <rect x={-141} y={-50} width={282} height={86} rx={19} fill={P.blueDark}
          stroke={P.ink} strokeWidth={8} />
        <path d="M-113 -20 L113 -20 M-113 -3 L113 -3" stroke="#45778A" strokeWidth={5} />
      </g>
      <path d="M-130 43 L130 43" stroke={P.ink} strokeWidth={7} />
      {leak > 0 && (
        <g opacity={leak}>
          <path d="M85 263 L85 -123" stroke={flowColor} strokeWidth={4}
            strokeDasharray="10 12" opacity={0.35} fill="none" />
          <g transform={`translate(85 ${flowY})`}>
            <path d={inward ? 'M-9 -9 L0 2 L9 -9' : 'M-9 9 L0 -2 L9 9'}
              stroke={flowColor} strokeWidth={5} fill="none" strokeLinecap="round" strokeLinejoin="round" />
          </g>
          {inward ? <path d="M72 247 L85 264 L98 247" stroke={flowColor} fill="none" strokeWidth={4} />
            : <path d="M72 -106 L85 -123 L98 -106" stroke={flowColor} fill="none" strokeWidth={4} />}
        </g>
      )}
    </g>
  );
};

// Magnified wall: the radiation arrows originate at the gap-facing metal surface,
// rather than incorrectly passing straight from water through opaque metal.
const WallDetail = ({frame, gas, radiation, lowE}: {
  frame: number; gas: number; radiation: number; lowE: number;
}) => (
  <g transform="translate(994 226)">
    <rect x={0} y={0} width={151} height={620} fill={P.warm} />
    <rect x={563} y={0} width={151} height={620} fill={P.paleBlue} />
    <rect x={151} y={0} width={38} height={620} fill={P.metal} stroke={P.ink} strokeWidth={5} />
    <rect x={525} y={0} width={38} height={620} fill={P.metal} stroke={P.ink} strokeWidth={5} />
    <path d="M181 22 L181 598 M533 22 L533 598" stroke={P.metalLight} strokeWidth={6} />
    <g>
      {[0, 1, 2, 3, 4, 5, 6, 7].map((i) => (
        <circle key={i} cx={235 + (i % 3) * 82}
          cy={126 + Math.floor(i / 3) * 130 - (1 - gas) * 60}
          r={7} fill={P.muted} opacity={gas * 0.65} />
      ))}
    </g>
    {[0, 1, 2].map((i) => {
      const start = 197;
      const y = 217 + i * 100;
      // Fewer and weaker paths describe reduced net transfer, never zero transfer.
      const strength = radiation * (i === 1 ? 1 - lowE * 0.55 : 1 - lowE);
      return (
        <g key={i} opacity={strength}>
          <path d={`M${start} ${y} q18 -18 36 0 t36 0 t36 0 t36 0 t36 0 t36 0 t36 0 t36 0`}
            stroke={P.coral} strokeWidth={4} fill="none" />
          <path d={`M504 ${y - 8} L514 ${y} L504 ${y + 8}`}
            stroke={P.coral} strokeWidth={4} fill="none" strokeLinecap="round" />
          <circle cx={start + ((frame + i * 26) % 110) / 110 * 311}
            cy={y} r={4} fill={P.coral} />
        </g>
      );
    })}
    <path d="M189 28 L189 592" stroke={P.coral} strokeWidth={6} opacity={lowE} />
    <text x={75} y={674} textAnchor="middle" fill={P.coral} fontFamily={FONT}
      fontSize={32} fontWeight={600}>热的一侧</text>
    <text x={358} y={674} textAnchor="middle" fill={P.ink} fontFamily={FONT}
      fontSize={36} fontWeight={700}>真空层</text>
    <text x={638} y={674} textAnchor="middle" fill={P.blue} fontFamily={FONT}
      fontSize={32} fontWeight={600}>外界</text>
    <text x={358} y={79} textAnchor="middle" fill={P.muted} fontFamily={FONT}
      fontSize={29}>{gas > 0.12 ? '制造时抽走空气' : '杯壁局部 · 放大示意'}</text>
  </g>
);

const scenes = timeline.scenes;
export const THERMOS_DURATION = timeline.totalFrames;

export const ThermosExplainer: React.FC = () => {
  const frame = useCurrentFrame();
  const index = scenes.findIndex((scene) => frame >= scene.from && frame < scene.from + scene.frames);
  const scene = scenes[Math.max(0, index)];
  const local = frame - scene.from;
  const radiationBeat = scenes[2].from + scenes[2].cues[2].from;
  const radiationNow = frame >= radiationBeat;
  const cutsAt = scenes[1].from + scenes[1].cues[1].from;
  const cut = p(frame, cutsAt, 32);
  const detailStart = scenes[2].from;
  const detailEnd = scenes[4].from;
  const detail = p(frame, detailStart, 18) * (1 - p(frame, detailEnd, 18));
  const gas = 1 - 0.94 * p(frame, detailStart + 13, 47);
  const lowE = p(frame, scenes[3].from + scenes[3].cues[0].to - 3, 34);
  const cold = p(frame, scenes[5].from + scenes[5].cues[0].from + 4, 36);
  const leak = p(frame, scenes[4].from + scenes[4].cues[0].from, 25);
  const headlineOpacity = p(local, 0, 10) * (1 - p(local, scene.frames - 9, 9));
  const cue = scene.cues.find((item) => local >= item.from && local < item.to);
  const primaryLines = [
    ['热水为什么', '凉得这么慢？'],
    ['秘密藏在', '两层杯壁里。'],
    radiationNow ? ['真空也挡不住', '热辐射。'] : ['真空层，', '减少气体传热。'],
    ['再让辐射', '少传一点。'],
    ['热量依然', '会慢慢漏出。'],
    ['同一只杯子，', '也能保冷。'],
  ][scene.index];
  const notes = [
    '一个关于热量去向的小问题',
    '外壳 → 间隙 → 内胆',
    radiationNow ? '波线表示辐射换热的方向' : '气体传导与对流，都被减弱',
    '低辐射率表面，减少辐射换热',
    '杯口与杯盖，是残留的传热通道',
    '外界较暖时，热量向冷水传递',
  ][scene.index];
  return (
    <AbsoluteFill style={{background: P.paper, color: P.ink, fontFamily: FONT}}>
      <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{position: 'absolute', inset: 0}}>
        <path d="M927 152 L1807 152 L1807 941 L927 941 Z"
          fill={interpolateColors(cold, [0, 1], [P.paleBlue, '#F1DFD2'])} opacity={0.30} />
        <path d="M104 133 L1816 133" stroke={P.ink} strokeWidth={2} opacity={0.2} />
        <g opacity={1 - detail}>
          <Flask cut={cut} cold={cold} leak={leak * (scene.index === 5 ? cold : 1)}
            inward={scene.index === 5} frame={frame}
            lidOpen={scene.index === 0 ? 1 - p(local, 19, 39) : 0} />
          {scene.index === 1 && (
            <g opacity={p(local, scene.cues[1].from + 20, 20)}>
              <Pointer points="1463,594 1532,594 1567,560" label="双层杯壁" x={1576} y={565} />
            </g>
          )}
          {scene.index === 4 && (
            <g opacity={p(local, scene.cues[0].from + 18, 16)}>
              <Pointer points="1363,231 1502,231 1534,280" label="杯口 / 杯盖" x={1541} y={293} color={P.coral} />
            </g>
          )}
          {scene.index === 5 && (
            <g opacity={cold}>
              <text x={1534} y={316} fill={P.coral} fontFamily={FONT}
                fontSize={34} fontWeight={600}>外界较暖</text>
              <Pointer points="1224,695 1501,695 1531,658" label="冷水" x={1545} y={669} color={P.blue} />
            </g>
          )}
        </g>
        <g opacity={detail}>
          <WallDetail frame={frame} gas={gas} radiation={p(frame, radiationBeat, 14)} lowE={lowE} />
        </g>
      </svg>
      <div style={{position: 'absolute', top: 66, left: 105, fontSize: 30,
        fontWeight: 600, color: P.muted, letterSpacing: 2}}>日常物理 / 保温杯</div>
      <div style={{position: 'absolute', top: 281, left: 103, width: 810,
        opacity: headlineOpacity, transform: `translateY(${(1 - p(local, 0, 14)) * 8}px)`}}>
        <div style={{fontSize: 86, lineHeight: 1.38, fontWeight: 800, letterSpacing: -1}}>
          {primaryLines.map((line, i) => <div key={i}>{line}</div>)}
        </div>
        <div style={{width: 75, height: 6, background: scene.index === 5 ? P.blue : P.coral,
          marginTop: 38, marginBottom: 28}} />
        <div style={{fontSize: 32, lineHeight: 1.7, color: P.muted}}>{notes}</div>
        {scene.index === 3 && (
          <div style={{opacity: lowE, marginTop: 39, fontSize: 42, fontWeight: 600, color: P.coral}}>
            辐射换热减少 ↓
          </div>
        )}
        {scene.index === 4 && local >= scene.cues[1].from && (
          <div style={{marginTop: 39, fontSize: 42, fontWeight: 600, color: P.coral}}>保温 = 减慢传热</div>
        )}
      </div>
      {cue && <div style={{position: 'absolute', left: 150, right: 150, top: 973,
        textAlign: 'center', fontSize: 43, fontWeight: 600, lineHeight: 1.25}}>{cue.text}</div>}
      <div style={{position: 'absolute', left: 105, bottom: 17, fontSize: 24, color: P.muted}}>
        结构与热流为示意，非等比例
      </div>
      {scenes.map((part) => (
        <Sequence key={part.index} from={part.from + part.preroll}
          durationInFrames={part.frames - part.preroll} name={`配音 ${part.index + 1}`}>
          <Audio src={staticFile(part.audio)} />
        </Sequence>
      ))}
    </AbsoluteFill>
  );
};
