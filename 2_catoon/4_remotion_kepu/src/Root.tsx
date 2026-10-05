import {Composition} from 'remotion';
import {Demo} from './Demo';
import {KepuVideo} from './KepuVideo';
import {E201} from './episodes/E201';
import {E202} from './episodes/E202';
import {E203} from './episodes/E203';
import {E204} from './episodes/E204';
import {E205} from './episodes/E205';
import timeline024 from './timeline_024.json';
import timeline201 from './timeline_201.json';
import timeline202 from './timeline_202.json';
import timeline203 from './timeline_203.json';
import timeline204 from './timeline_204.json';
import timeline205 from './timeline_205.json';
import {COMPS} from './episodes/all';
import {DURATIONS} from './durations';
import {ThermosExplainer, THERMOS_DURATION} from './redesign/ThermosExplainer';

const FPS = 30;

export const Root = () => {
  return (
    <>
      <Composition
        id="ThermosRedesign"
        component={ThermosExplainer}
        durationInFrames={THERMOS_DURATION}
        fps={FPS}
        width={1920}
        height={1080}
      />
      <Composition
        id="Kepu024"
        component={KepuVideo as any}
        durationInFrames={Math.ceil(timeline024.totalSec * FPS)}
        fps={FPS}
        width={1920}
        height={1080}
        defaultProps={{data: timeline024}}
      />
      <Composition
        id="Kepu201"
        component={E201 as any}
        durationInFrames={Math.ceil(timeline201.totalSec * FPS)}
        fps={FPS}
        width={1920}
        height={1080}
        defaultProps={{data: timeline201}}
      />
      {(['202', '203', '204', '205'] as const).map((ep) => {
        const comp = {'202': E202, '203': E203, '204': E204, '205': E205}[ep];
        const tl = {'202': timeline202, '203': timeline203, '204': timeline204, '205': timeline205}[ep];
        return (
          <Composition
            key={ep}
            id={`Kepu${ep}`}
            component={comp as any}
            durationInFrames={Math.ceil(tl.totalSec * FPS)}
            fps={FPS}
            width={1920}
            height={1080}
            defaultProps={{data: tl}}
          />
        );
      })}
      {Object.keys(DURATIONS).map((k) => {
        const n = Number(k);
        return (
          <Composition
            key={n}
            id={`Kepu${n}`}
            component={COMPS[n] as any}
            durationInFrames={DURATIONS[n]}
            fps={FPS}
            width={1920}
            height={1080}
          />
        );
      })}
      <Composition
        id="Demo"
        component={Demo}
        durationInFrames={180}
        fps={30}
        width={1920}
        height={1080}
      />
    </>
  );
};
