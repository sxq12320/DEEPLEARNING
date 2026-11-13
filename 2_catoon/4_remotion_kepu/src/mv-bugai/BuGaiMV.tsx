import React from 'react';
import {
  AbsoluteFill,
  Audio,
  interpolate,
  interpolateColors,
  staticFile,
  useCurrentFrame,
} from 'remotion';
import musicData from './music-data.json';
import {
  AuroraSky,
  clamp01,
  easeInOut,
  FrameAudio,
  Scene1Intro,
  Scene2JayVerse,
  Scene3AmeiVerse,
  Scene4CrystalVow,
  Scene5ChorusBloomAndHands,
  Scene6ClockAndLake,
  Scene7FinalClimax,
  Scene8Outro,
  SnowAndPetalsField,
} from './scenes';

export const BUGAI_FPS = musicData.fps;
export const BUGAI_DURATION_FRAMES = musicData.totalFrames; // 8743 frames (291.41s)

const FONT_SERIF = '"Songti SC", "STSong", "Noto Serif SC", "SimSun", "Microsoft YaHei", serif';
const FONT_SANS = '"Microsoft YaHei", "PingFang SC", "Helvetica Neue", sans-serif';

interface ChapterDef {
  id: number;
  code: string;
  title: string;
  subtitle: string;
  startSec: number;
  endSec: number;
  bgTop: string;
  bgBottom: string;
  snowDensity: number;
  petalDensity: number;
}

const CHAPTERS: ChapterDef[] = [
  {
    id: 1,
    code: 'ACT I · PRELUDE',
    title: '序幕 · 床边故事与冰封八音盒',
    subtitle: 'A Frozen Bedtime Story Begins',
    startSec: 0,
    endSec: 26.67,
    bgTop: '#040814',
    bgBottom: '#0F223D',
    snowDensity: 0.65,
    petalDensity: 0,
  },
  {
    id: 2,
    code: 'ACT II · SOLILOQUY (JAY)',
    title: '独白 · 雪夜街灯与假装的戏',
    subtitle: 'Pretending We Are Still Together',
    startSec: 26.67,
    endSec: 50.73,
    bgTop: '#060D1E',
    bgBottom: '#162A45',
    snowDensity: 0.8,
    petalDensity: 0,
  },
  {
    id: 3,
    code: 'ACT III · UNDERCURRENT (aMEI)',
    title: '暗涌 · 霜窗之外的一片空白',
    subtitle: 'Unspoken Emotions Behind Frosted Glass',
    startSec: 50.73,
    endSec: 76.0,
    bgTop: '#14091D',
    bgBottom: '#2D1436',
    snowDensity: 0.75,
    petalDensity: 0,
  },
  {
    id: 4,
    code: 'ACT IV · CRYSTALLIZED VOW',
    title: '结晶 · 零下誓言与碎裂之梦',
    subtitle: 'Frozen Promises Begin to Melt',
    startSec: 76.0,
    endSec: 101.0,
    bgTop: '#07152B',
    bgBottom: '#1E294B',
    snowDensity: 0.95,
    petalDensity: 0.1,
  },
  {
    id: 5,
    code: 'ACT V · FIRST CHORUS',
    title: '盛放 · 雪融花开与松开的手',
    subtitle: 'When the Snow Melts, Flowers Should Bloom',
    startSec: 101.0,
    endSec: 154.7,
    bgTop: '#0D132A',
    bgBottom: '#31173E',
    snowDensity: 0.55,
    petalDensity: 0.9,
  },
  {
    id: 6,
    code: 'ACT VI · RETROGRADE TIME',
    title: '逆时 · 买不到的时间与释怀',
    subtitle: 'Time That Cannot Be Bought Back',
    startSec: 154.7,
    endSec: 191.5,
    bgTop: '#080C18',
    bgBottom: '#1C1B2E',
    snowDensity: 0.7,
    petalDensity: 0.2,
  },
  {
    id: 7,
    code: 'ACT VII · CLIMAX SUPERNOVA',
    title: '绝唱 · 极光裂空与擦肩而过',
    subtitle: 'Letting Go to Embrace the Future',
    startSec: 191.5,
    endSec: 270.0,
    bgTop: '#0F152D',
    bgBottom: '#3E1642',
    snowDensity: 0.85,
    petalDensity: 1.0,
  },
  {
    id: 8,
    code: 'ACT VIII · EPILOGUE',
    title: '终章 · 故事合上，雪落无声',
    subtitle: 'Closing the Book of Memories',
    startSec: 270.0,
    endSec: 291.5,
    bgTop: '#03060E',
    bgBottom: '#09101D',
    snowDensity: 0.4,
    petalDensity: 0.3,
  },
];

// Poetic English translations matched to lyric text
const ENGLISH_SUBS: Record<string, string> = {
  '假装我们还在一块': 'Pretending that we are still together',
  '我真的演不出来': 'I truly cannot act out this play anymore',
  '还是不习惯你不在': 'Still unaccustomed to your absence beside me',
  '这身份转变太快': 'Our roles shifted from lovers to strangers too fast',
  '画面里不需要旁白': 'No narration is needed inside this frame',
  '却谁都看的出来': 'Yet anyone who looks can see the heartbreak',
  '是我情绪涌了上来': 'It is my surging tide of emotion rising up',
  '想哭却一遍空白': 'Wanting to cry, yet my mind turns into blinding white',
  '雪地里相爱': 'Falling in love upon the boundless snowfield',
  '他们说零下已结晶的誓言不会坏': 'They said vows crystallized below zero would never decay',
  '但爱的状态': 'Yet the state of love itself',
  '却不会永远都冰封': 'Can never remain frozen in eternity',
  '而透明的存在': 'As a fragile, transparent existence',
  '轻轻飘 落下来': 'Drifting softly down from the winter sky',
  '许下的梦 融化的太快': 'The dreams we wished upon melted all too soon',
  '许下的梦融化的太快': 'The dreams we wished upon melted all too soon',
  '或许我们都不该醒来': 'Perhaps neither of us should have ever woken up',
  '你还是住在我的回忆里不出来': 'You still live deep inside my memories, refusing to leave',
  '让我们微笑离开让故事留下来': 'Let us depart with a smile and leave only the story behind',
  '放手后爱依然在': 'Even after letting go, the love remains',
  '雪融了就应该花开': 'Once the snow melts, the spring flowers ought to bloom',
  '缘若尽了': 'If our destined bond has reached its end',
  '就不该再重来': 'Then we should never try to start over again',
  '你依旧住在我的回忆里不出来': 'You still dwell inside my memories, never walking out',
  '我离开将你的手交给下个最爱': 'I step away and entrust your hand to your next true love',
  '我离开将你的手交给下个': 'I step away and entrust your hand to the next—',
  '最爱': '—True Love',
  '纠缠与': 'Entanglement and—',
  'Oh 固执等待': 'Oh, stubborn waiting in vain',
  'Oh固执等待': 'Oh, stubborn waiting in vain',
  '反而是另一种伤害': 'Only turns into another kind of wound',
  '彼此紧握的手松开': 'Unclasping the hands we once held so tight',
  '去拥抱更多未来': 'To embrace a wider future apart',
  '错过的时间怎么买': 'How could missed time ever be bought back?',
  '谁都付不出来': 'No one in this world can afford that price',
  '或许我们学会释怀': 'Perhaps we have finally learned to let go',
  '让过去安静下来': 'And allow the past to settle into silence',
};

const formatTime = (sec: number) => {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
};

export const BuGaiMV: React.FC = () => {
  const frame = useCurrentFrame();
  const timeSec = frame / BUGAI_FPS;
  const idx = Math.min(musicData.frames.rms.length - 1, Math.max(0, frame));

  const audio: FrameAudio = {
    rms: musicData.frames.rms[idx] ?? 0,
    bass: musicData.frames.bass[idx] ?? 0,
    mid: musicData.frames.mid[idx] ?? 0,
    high: musicData.frames.high[idx] ?? 0,
    onset: musicData.frames.onset[idx] ?? 0,
  };

  // Determine active chapter
  const chapter =
    CHAPTERS.find((c) => timeSec >= c.startSec && timeSec < c.endSec) ??
    CHAPTERS[CHAPTERS.length - 1];
  const chapterLocalFrame = Math.max(0, Math.round((timeSec - chapter.startSec) * BUGAI_FPS));
  const chapterFrames = Math.round((chapter.endSec - chapter.startSec) * BUGAI_FPS);

  // Smooth fade-in/out for chapter scene transitions
  const sceneEnter = chapter.id === 1 ? 1 : easeInOut(chapterLocalFrame, 0, 24);
  const sceneExit =
    chapter.id === 8 ? 1 : 1 - easeInOut(chapterLocalFrame, chapterFrames - 24, 24);
  const sceneOpacity = clamp01(sceneEnter * sceneExit);

  // Active lyric lookup
  const activeLyricIndex = musicData.lyrics.findIndex(
    (l) => timeSec >= l.start && timeSec < l.end
  );
  const activeLyric = activeLyricIndex >= 0 ? musicData.lyrics[activeLyricIndex] : null;
  const nextLyric =
    activeLyricIndex >= 0 && activeLyricIndex + 1 < musicData.lyrics.length
      ? musicData.lyrics[activeLyricIndex + 1]
      : null;

  // Singer weights for color & aura modulation
  const singer = activeLyric?.singer ?? (chapter.id === 2 ? 'Jay' : chapter.id === 3 ? 'aMEI' : '合');
  const jayWeight = singer === 'Jay' ? 1 : singer === '合' ? 0.65 : 0.15;
  const ameiWeight = singer === 'aMEI' ? 1 : singer === '合' ? 0.65 : 0.15;

  // Subtle cinematic camera drift + bass impulse on chorus hits
  const isChorus = chapter.id === 5 || chapter.id === 7;
  const camScale =
    1.015 +
    Math.sin(frame * 0.008) * 0.012 +
    (isChorus ? audio.bass * 0.025 + audio.onset * 0.012 : audio.rms * 0.008);
  const camX = Math.sin(frame * 0.011) * 10;
  const camY = Math.cos(frame * 0.009) * 7;

  // Master fade-in (first 1.5s) and fade-out (last 3.5s)
  const masterFade =
    easeInOut(frame, 0, 45) *
    (1 - easeInOut(frame, BUGAI_DURATION_FRAMES - 105, 105));

  // Singer badge styling
  const singerMeta =
    singer === 'Jay'
      ? {label: 'JAY CHOU · 周杰伦', color: '#7DD3FC', glow: 'rgba(56, 189, 248, 0.55)'}
      : singer === 'aMEI'
      ? {label: 'aMEI · 张惠妹', color: '#FDA4AF', glow: 'rgba(251, 113, 133, 0.55)'}
      : singer === '合'
      ? {label: 'JAY × aMEI · 双声合唱', color: '#FDE68A', glow: 'rgba(253, 230, 138, 0.6)'}
      : {label: 'BEDTIME STORIES · 序章', color: '#BAE6FD', glow: 'rgba(186, 230, 253, 0.45)'};

  // Lyric animation progress within current line
  let lyricOpacity = 0;
  let lyricTranslateY = 0;
  let lyricProgress = 0;
  if (activeLyric) {
    const lineDur = Math.max(0.4, activeLyric.end - activeLyric.start);
    const elapsed = timeSec - activeLyric.start;
    lyricProgress = clamp01(elapsed / lineDur);
    const fadeIn = clamp01(elapsed / 0.28);
    const fadeOut = clamp01((activeLyric.end - timeSec) / 0.28);
    lyricOpacity = Math.min(fadeIn, fadeOut);
    lyricTranslateY = (1 - fadeIn) * 14 - (1 - fadeOut) * 8;
  }

  // Check if we are in the breathless drum-drop suspension (242.3s - 253.1s)
  const isSuspended = timeSec >= 242.3 && timeSec < 253.1;
  // Check if we are in the retrograde interlude (154.7s - 166.9s)
  const isRetrograde = timeSec >= 154.7 && timeSec < 166.9;

  return (
    <AbsoluteFill
      style={{
        backgroundColor: '#02040A',
        fontFamily: FONT_SANS,
        color: '#F8FAFC',
        overflow: 'hidden',
      }}
    >
      <Audio src={staticFile('mv-bugai/bugai.mp3')} />

      {/* MAIN CINEMATIC STAGE WITH CAMERA DRIFT */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          opacity: masterFade,
          transform: `translate(${camX.toFixed(2)}px, ${camY.toFixed(2)}px) scale(${camScale.toFixed(4)})`,
          transformOrigin: 'center center',
        }}
      >
        {/* Dynamic Emotional Sky Gradient */}
        <div
          style={{
            position: 'absolute',
            inset: 0,
            background: `radial-gradient(circle at 50% 38%, ${interpolateColors(
              audio.rms,
              [0, 1],
              [chapter.bgBottom, '#3B1D54']
            )} 0%, ${chapter.bgTop} 78%, #02040A 100%)`,
          }}
        />

        {/* SVG Visual Canvas */}
        <svg
          width={1920}
          height={1080}
          viewBox="0 0 1920 1080"
          style={{position: 'absolute', inset: 0}}
        >
          {/* Distant Starfield */}
          {Array.from({length: 42}).map((_, i) => {
            const sx = ((i * 197 + 83) % 1880) + 20;
            const sy = ((i * 113 + 29) % 520) + 60;
            const twinkle = 0.25 + 0.55 * Math.abs(Math.sin(frame * 0.03 + i));
            return (
              <circle
                key={i}
                cx={sx}
                cy={sy}
                r={i % 4 === 0 ? 2.1 : 1.3}
                fill={i % 3 === 0 ? '#FDE68A' : '#E0F2FE'}
                opacity={twinkle}
              />
            );
          })}

          {/* Aurora Borealis Sky Layer */}
          <AuroraSky
            frame={frame}
            audio={audio}
            jayWeight={jayWeight}
            ameiWeight={ameiWeight}
            climaxBoost={chapter.id === 7 ? 0.85 : chapter.id === 5 ? 0.45 : 0}
          />

          {/* Chapter-Specific Narrative Visual Scene */}
          <g opacity={sceneOpacity}>
            {chapter.id === 1 && <Scene1Intro frame={frame} audio={audio} />}
            {chapter.id === 2 && (
              <Scene2JayVerse
                frame={frame}
                localFrame={chapterLocalFrame}
                audio={audio}
              />
            )}
            {chapter.id === 3 && (
              <Scene3AmeiVerse
                frame={frame}
                localFrame={chapterLocalFrame}
                audio={audio}
              />
            )}
            {chapter.id === 4 && (
              <Scene4CrystalVow
                frame={frame}
                localFrame={chapterLocalFrame}
                audio={audio}
              />
            )}
            {chapter.id === 5 && (
              <Scene5ChorusBloomAndHands
                frame={frame}
                localFrame={chapterLocalFrame}
                audio={audio}
              />
            )}
            {chapter.id === 6 && (
              <Scene6ClockAndLake
                frame={frame}
                localFrame={chapterLocalFrame}
                audio={audio}
              />
            )}
            {chapter.id === 7 && (
              <Scene7FinalClimax
                frame={frame}
                localFrame={chapterLocalFrame}
                audio={audio}
              />
            )}
            {chapter.id === 8 && (
              <Scene8Outro
                frame={frame}
                localFrame={chapterLocalFrame}
                audio={audio}
              />
            )}
          </g>

          {/* Foreground Multi-Depth Snow & Blossom Petals */}
          <SnowAndPetalsField
            frame={frame}
            audio={audio}
            snowDensity={chapter.snowDensity}
            petalDensity={chapter.petalDensity}
            reverseTime={isRetrograde}
            freezeFactor={isSuspended ? 0.9 : 0}
            windBias={chapter.id === 5 || chapter.id === 7 ? 1.45 : 0.85}
          />
        </svg>

        {/* Subtle Radial Vignette */}
        <div
          style={{
            position: 'absolute',
            inset: 0,
            background:
              'radial-gradient(ellipse at center, rgba(0,0,0,0) 55%, rgba(2,4,10,0.68) 100%)',
            pointerEvents: 'none',
          }}
        />
      </div>

      {/* =================================================================== */}
      {/* INTRO / INTERLUDE / OUTRO CINEMATIC TITLE OVERLAYS                  */}
      {/* =================================================================== */}
      {timeSec < 25.5 && (
        <div
          style={{
            position: 'absolute',
            left: 0,
            right: 0,
            bottom: 138,
            textAlign: 'center',
            opacity:
              easeInOut(frame, 18, 45) * (1 - easeInOut(frame, 23 * 30, 60)),
          }}
        >
          <div
            style={{
              fontSize: 22,
              letterSpacing: 10,
              color: '#FDE68A',
              fontWeight: 600,
              marginBottom: 10,
            }}
          >
            JAY CHOU × aMEI · 2016 BEDTIME STORIES
          </div>
          <div
            style={{
              fontFamily: FONT_SERIF,
              fontSize: 74,
              fontWeight: 700,
              letterSpacing: 18,
              color: '#F8FAFC',
              textShadow: '0 0 36px rgba(125, 211, 252, 0.55)',
            }}
          >
            不该
          </div>
          <div
            style={{
              fontSize: 23,
              letterSpacing: 6,
              color: '#BAE6FD',
              marginTop: 8,
            }}
          >
            词：方文山 &nbsp;·&nbsp; 曲：周杰伦 &nbsp;·&nbsp; 编曲：黄雨勋
          </div>
        </div>
      )}

      {/* Instrumental Interlude Poetic Callout (155.2s - 166.5s) */}
      {timeSec >= 155.2 && timeSec < 166.5 && (
        <div
          style={{
            position: 'absolute',
            left: 0,
            right: 0,
            bottom: 156,
            textAlign: 'center',
            opacity:
              easeInOut(frame, Math.round(155.2 * 30), 30) *
              (1 - easeInOut(frame, Math.round(165.0 * 30), 40)),
          }}
        >
          <div
            style={{
              fontSize: 20,
              letterSpacing: 8,
              color: '#FDE68A',
              marginBottom: 10,
            }}
          >
            INSTRUMENTAL BRIDGE · 逆时吉他与弦乐间奏
          </div>
          <div
            style={{
              fontFamily: FONT_SERIF,
              fontSize: 42,
              letterSpacing: 6,
              color: '#E0F2FE',
              textShadow: '0 0 28px rgba(125, 211, 252, 0.45)',
            }}
          >
            当风雪倒流，故事停在尚未说出口的告别
          </div>
        </div>
      )}

      {/* Outro Closing Credits (272.5s - 290.5s) */}
      {timeSec >= 272.5 && (
        <div
          style={{
            position: 'absolute',
            left: 0,
            right: 0,
            bottom: 148,
            textAlign: 'center',
            opacity:
              easeInOut(frame, Math.round(272.5 * 30), 45) *
              (1 - easeInOut(frame, Math.round(288.0 * 30), 90)),
          }}
        >
          <div
            style={{
              fontFamily: FONT_SERIF,
              fontSize: 48,
              fontWeight: 700,
              letterSpacing: 12,
              color: '#F8FAFC',
              marginBottom: 12,
            }}
          >
            雪融了，就应该花开
          </div>
          <div
            style={{
              fontSize: 22,
              letterSpacing: 6,
              color: '#94A3B8',
            }}
          >
            周杰伦 × 张惠妹《不该》 · 专辑《周杰伦的床边故事》
          </div>
        </div>
      )}

      {/* =================================================================== */}
      {/* ACTIVE LYRIC STAGE (WITH SINGER AURA, BILINGUAL SUBS & NEXT PREVIEW)*/}
      {/* =================================================================== */}
      {activeLyric && activeLyric.singer !== 'Credits' && (
        <div
          style={{
            position: 'absolute',
            left: 140,
            right: 140,
            bottom: 112,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            opacity: lyricOpacity * masterFade,
            transform: `translateY(${lyricTranslateY.toFixed(1)}px)`,
            pointerEvents: 'none',
          }}
        >
          {/* Singer Badge Pill */}
          <div
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 10,
              padding: '5px 20px',
              borderRadius: 999,
              background: 'rgba(15, 23, 42, 0.72)',
              border: `1.5px solid ${singerMeta.color}`,
              boxShadow: `0 0 20px ${singerMeta.glow}`,
              marginBottom: 14,
            }}
          >
            <span
              style={{
                width: 9,
                height: 9,
                borderRadius: '50%',
                backgroundColor: singerMeta.color,
                boxShadow: `0 0 10px ${singerMeta.color}`,
              }}
            />
            <span
              style={{
                fontSize: 17,
                fontWeight: 700,
                letterSpacing: 4,
                color: singerMeta.color,
              }}
            >
              {singerMeta.label}
            </span>
          </div>

          {/* Primary Chinese Lyric Line */}
          <div
            style={{
              fontFamily: FONT_SERIF,
              fontSize: activeLyric.text === '最爱' ? 84 : 58,
              fontWeight: 700,
              letterSpacing: activeLyric.text === '最爱' ? 18 : 6,
              color: '#FFFFFF',
              textAlign: 'center',
              lineHeight: 1.25,
              textShadow: `0 4px 24px rgba(0, 0, 0, 0.9), 0 0 ${
                24 + audio.rms * 32
              }px ${singerMeta.glow}`,
            }}
          >
            {activeLyric.text}
          </div>

          {/* Animated Underline Sweep */}
          <div
            style={{
              width: 320,
              height: 3,
              background: 'rgba(255,255,255,0.14)',
              borderRadius: 2,
              marginTop: 12,
              marginBottom: 10,
              overflow: 'hidden',
            }}
          >
            <div
              style={{
                width: `${(lyricProgress * 100).toFixed(1)}%`,
                height: '100%',
                background: `linear-gradient(90deg, transparent, ${singerMeta.color}, #FFFFFF)`,
              }}
            />
          </div>

          {/* Poetic English Subtitle */}
          <div
            style={{
              fontSize: 22,
              fontWeight: 500,
              letterSpacing: 2.5,
              color: '#CBD5E1',
              textShadow: '0 2px 10px rgba(0,0,0,0.85)',
            }}
          >
            {ENGLISH_SUBS[activeLyric.text] ?? ''}
          </div>

          {/* Subtle Next Line Preview */}
          {nextLyric && nextLyric.start - timeSec < 4.5 && (
            <div
              style={{
                marginTop: 10,
                fontSize: 18,
                letterSpacing: 3,
                color: 'rgba(148, 163, 184, 0.55)',
              }}
            >
              NEXT · {nextLyric.text}
            </div>
          )}
        </div>
      )}

      {/* =================================================================== */}
      {/* ANAMORPHIC WIDESCREEN LETTERBOX BARS (TOP & BOTTOM)                 */}
      {/* =================================================================== */}
      {/* Top Letterbox Matte */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: 76,
          background: 'linear-gradient(180deg, rgba(2,4,10,0.95) 65%, rgba(2,4,10,0) 100%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 64px',
          opacity: masterFade,
        }}
      >
        <div style={{display: 'flex', alignItems: 'center', gap: 18}}>
          <span
            style={{
              fontSize: 15,
              fontWeight: 700,
              letterSpacing: 3,
              color: '#7DD3FC',
              padding: '4px 12px',
              border: '1px solid rgba(125,211,252,0.4)',
              borderRadius: 4,
            }}
          >
            {chapter.code}
          </span>
          <span
            style={{
              fontSize: 20,
              fontWeight: 600,
              letterSpacing: 2,
              color: '#E2E8F0',
            }}
          >
            {chapter.title}
          </span>
          <span
            style={{
              fontSize: 15,
              letterSpacing: 1.5,
              color: '#64748B',
            }}
          >
            / {chapter.subtitle}
          </span>
        </div>

        <div
          style={{
            fontSize: 16,
            letterSpacing: 4,
            color: '#94A3B8',
            fontWeight: 600,
          }}
        >
          周杰伦 × 张惠妹 《不该》 · HI-RES FLAC MV
        </div>
      </div>

      {/* Bottom Letterbox Matte with Symmetric Audio Spectrum & Timeline */}
      <div
        style={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          right: 0,
          height: 82,
          background: 'linear-gradient(0deg, rgba(2,4,10,0.96) 72%, rgba(2,4,10,0) 100%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 64px',
          opacity: masterFade,
        }}
      >
        {/* Left: Song Metadata */}
        <div style={{display: 'flex', alignItems: 'center', gap: 14, width: 420}}>
          <div
            style={{
              width: 10,
              height: 10,
              borderRadius: '50%',
              backgroundColor: singerMeta.color,
              boxShadow: `0 0 12px ${singerMeta.color}`,
            }}
          />
          <span style={{fontSize: 16, letterSpacing: 2, color: '#CBD5E1'}}>
            ALBUM · 周杰伦的床边故事
          </span>
        </div>

        {/* Center: 48-Bar Symmetric Audio Spectrum Visualizer */}
        <div
          style={{
            display: 'flex',
            alignItems: 'flex-end',
            justifyContent: 'center',
            gap: 5,
            height: 36,
          }}
        >
          {Array.from({length: 48}).map((_, b) => {
            const distFromCenter = Math.abs(b - 23.5) / 24;
            const bandEnergy =
              distFromCenter < 0.33
                ? audio.bass
                : distFromCenter < 0.66
                ? audio.mid
                : audio.high;
            const wave =
              0.45 + 0.55 * Math.abs(Math.sin(frame * 0.16 + b * 0.45));
            const h = Math.max(4, Math.min(34, bandEnergy * wave * 36));
            const barColor =
              b < 24
                ? interpolateColors(jayWeight, [0, 1], ['#64748B', '#38BDF8'])
                : interpolateColors(ameiWeight, [0, 1], ['#64748B', '#FB7185']);
            return (
              <div
                key={b}
                style={{
                  width: 4,
                  height: h,
                  borderRadius: 2,
                  backgroundColor: barColor,
                  opacity: 0.75,
                }}
              />
            );
          })}
        </div>

        {/* Right: Progress Bar & Timecode */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'flex-end',
            gap: 16,
            width: 420,
          }}
        >
          <div
            style={{
              width: 210,
              height: 4,
              backgroundColor: 'rgba(148, 163, 184, 0.22)',
              borderRadius: 2,
              overflow: 'hidden',
            }}
          >
            <div
              style={{
                width: `${((frame / BUGAI_DURATION_FRAMES) * 100).toFixed(2)}%`,
                height: '100%',
                background: 'linear-gradient(90deg, #38BDF8, #FB7185, #FDE68A)',
              }}
            />
          </div>
          <span
            style={{
              fontFamily: 'monospace',
              fontSize: 16,
              letterSpacing: 1.5,
              color: '#E2E8F0',
            }}
          >
            {formatTime(timeSec)} / {formatTime(musicData.duration)}
          </span>
        </div>
      </div>
    </AbsoluteFill>
  );
};
