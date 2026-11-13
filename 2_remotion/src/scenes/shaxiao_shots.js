// shaxiao_shots.js: 14 narrative shots across 6 chapters for 周杰伦 × 袁咏琳《傻笑》(0.00s – 289.96s).
// Every shot is a pure function of (t, lt, dur) and follows ANIMATION_GUIDE.md.
(() => {
  const L_RAIN = LOOK.rainDusk;
  const L_HER  = LOOK.herRoom;
  const L_CAFE = LOOK.cafeNight;
  const L_SEP  = LOOK.sepia;
  const L_BAKE = LOOK.bakery;
  const L_CINE = LOOK.cinema;
  const L_CLIM = LOOK.climaxNight;

  // ---------- 1 · 0.00–19.50s · Intro: Rainy Café Window & Empty Chair ----------
  function shot01_intro(t, lt, dur) {
    const pushK = ease(seg(lt, 0, dur));
    camBegin(lerp(960, 860, pushK) + 18 * Math.sin(lt * .45), lerp(560, 535, pushK), lerp(1.02, 1.12, pushK));
    skyBg(L_RAIN, t);
    streetGround(L_RAIN);
    lampPost(1520, L_RAIN, .7 + .3 * seg(lt, 1.5, 4.0));
    storefront(220, 980, L_RAIN, 'cafe', t);
    bistroTable(960, GROUND - 20, L_RAIN);
    coffeeCup(925, GROUND - 122, 1.25, t, true);

    // Him sitting by the table, looking at the empty space across from him
    const m = emotions(lt, [
      [0, 'sleepy', { lookX: .6, emote: null }],
      [5.5, 'sad', { lookX: .8, lookY: .1, emote: null }],
      [12.5, 'thinking', { lookX: .7, lookY: -.2 }],
    ], { take: .55 });
    him(720, GROUND - 18, 25, { ...m, view: 'q' }, L_RAIN, t);

    rainOverlay(L_RAIN, t, .9);
    const focus = toScreen(925, GROUND - 130);
    camEnd();
    if (lt < 1.8) iris(focus[0], focus[1], lerp(0, 1600, easeOut(lt / 1.8)), PAL.ink);
  }

  // ---------- 2 · 19.50–38.20s · Verse 1A (Jay): No Signal & The Smiley Symbol ----------
  function shot02_phone(t, lt, dur) {
    // 19.69 眼看着手机里没讯号 / 23.90 担心你的回信我收不到 / 28.75 哪怕只传来一个微笑的符号 / 33.97 我都会舍不得删掉
    const tSmile = 28.75 - 19.50; // 9.25s into shot
    const tHold  = 33.97 - 19.50; // 14.47s into shot
    const zoomK  = ease(seg(lt, 0, tHold));
    camBegin(960 + 15 * Math.sin(lt * .5), lerp(560, 520, zoomK), lerp(1.08, 1.22, zoomK));
    skyBg(L_RAIN, t);
    streetGround(L_RAIN);
    lampPost(520, L_RAIN, 1);

    const m = emotions(lt, [
      [0, 'nervous', { lookX: .6, lookY: .4, emote: '?' }],
      [4.4, 'sad', { lookX: .5, lookY: .5, emote: 'sweat' }],
      [tSmile, 'surprised', { lookX: .3, lookY: -.3, emote: '!' }],
      [tSmile + 1.4, 'hopeful', { lookX: .2, lookY: -.4, emote: null }],
      [tHold, 'love', { lookX: .1, lookY: -.2 }],
    ], { take: .65 });

    const cradle = ease(seg(lt, tHold, tHold + 1.6));
    const phoneArm = (u, sw) => {
      push();
      paint(rrPts(0, -u * .9, u * 1.6, u * 2.4, u * .25), { wash: '#2B3554', ink: PAL.ink, sw: sw * .7 });
      paint(rectPts(u * .2, -u * .65, u * 1.2, u * 1.8), { wash: lt > tSmile ? '#FFE27A' : '#5E729C', ink: null });
      pop();
    };

    him(860, GROUND - 15, 30, {
      ...m,
      view: lt > tHold - .4 ? 'front' : 'q',
      aR: lerp(.35, .85, cradle),
      aL: lerp(-.3, .85, cradle),
      armR: lt < tHold ? phoneArm : null,
    }, L_RAIN, t);

    // The glowing smiley symbol rising from the phone and floating above his hands
    if (lt > tSmile) {
      const rise = easeOut(seg(lt, tSmile, tSmile + 2.2));
      const settle = ease(seg(lt, tHold, tHold + 1.5));
      const sx = lerp(995, 860, settle);
      const sy = lerp(GROUND - 135, GROUND - 265, rise) + lerp(0, 35, settle) + Math.sin(t * 3.5) * 8;
      smileyBubble(sx, sy, 34 * backOut(seg(lt, tSmile, tSmile + .6)), t, 1);
    }

    rainOverlay(L_RAIN, t, .75);
    camEnd();
    boilSeed('wipe2');
    if (lt > dur - .45) brushWipe((lt - (dur - .45)) / .9, ['#48395A', '#E8A658']);
  }

  // ---------- 3 · 38.20–48.00s · Verse 1B (Cindy): Weather Forecast & Raincoat Memory ----------
  function shot03_raincoat(t, lt, dur) {
    // 38.44 电台传来天气的预报 / 43.44 想起你为我遮雨的外套
    const tMem = 43.44 - 38.20; // 5.24s
    const inMem = ease(seg(lt, tMem - .4, tMem + .6));
    const L = inMem > .5 ? L_SEP : L_HER;

    camBegin(960 + 20 * Math.sin(lt * .5), 540, lerp(1.04, 1.14, ease(seg(lt, 0, dur))));
    skyBg(L, t);
    streetGround(L);

    if (inMem < .55) {
      // Her by the window with a vintage radio
      storefront(420, 1080, L_HER, 'cafe', t);
      // Radio on table
      boilSeed('radio');
      bistroTable(760, GROUND - 20, L_HER);
      paint(rrPts(710, GROUND - 155, 100, 58, 10), { wash: '#8C583A', ink: PAL.ink, sw: .95 });
      paint(ellPts(738, GROUND - 126, 16, 16, 12), { wash: '#F5C452', ink: PAL.ink, sw: .7 });
      inkLine([[725, GROUND - 155], [765, GROUND - 195]], 1.4, PAL.ink, 'ink', 0);
      emote('music', 780, GROUND - 210, 1.1, 1, t);

      const mf = emotions(lt, [
        [0, 'neutral', { lookX: -.6, emote: null }],
        [2.2, 'sad', { lookX: -.8, lookY: -.2, emote: 'cloud' }],
      ], { take: .55 });
      her(1060, GROUND - 18, 28, { ...mf, view: 'q', flip: true }, L_HER);
    } else {
      // Memory in sepia: Him holding his brown coat high over both of them in the pouring rain
      lampPost(420, L_SEP, 1);
      const coatY = GROUND - 285 + Math.sin(t * 3) * 4;
      boilSeed('coat');
      paint([
        [720, coatY], [1200, coatY - 12],
        [1240, coatY + 75], [680, coatY + 75],
      ], { wash: '#7A5238', fill: '#5C3B24', fillOp: 70, tex: .4, ink: PAL.ink, sw: 1.2, curv: .2 });

      him(835, GROUND - 18, 27, {
        ...feel('happy', t),
        view: 'q',
        aL: 1.35,
        aR: 1.35,
        blush: .6,
      }, L_SEP, t);

      her(1085, GROUND - 18, 26, {
        ...feel('starstruck', t),
        view: 'q',
        flip: true,
        lookX: -.8,
        lookY: -.3,
      }, L_SEP);
    }

    rainOverlay(L_RAIN, t, 1.1);
    camEnd();
    if (lt < .45) brushWipe(.5 + lt / .9, ['#48395A', '#E8A658']);
  }

  // ---------- 4 · 48.00–60.50s · Verse 1C (Duet): Idol Drama on the Living Room TV ----------
  function shot04_tvdrama(t, lt, dur) {
    // 48.20 客厅的电视播放偶像剧频道 / 53.75 羡慕他们为爱在争吵
    camBegin(960 + 14 * Math.sin(lt * .5), 545, lerp(1.0, 1.1, ease(seg(lt, 0, dur))));
    boilSeed('living');
    paint(rectPts(-300, -200, W + 600, 960), { wash: '#2C274C', ink: null });
    paint(rectPts(-300, 740, W + 600, 420), { wash: '#3D2E42', ink: PAL.ink, sw: 1 });

    // Vintage TV cabinet on the right
    const tx = 1060, ty = 250, tw = 520, th = 360;
    glow(tx + tw / 2, ty + th / 2, 360, '#F7C96E', .55);
    paint(rrPts(tx - 30, ty - 30, tw + 60, th + 80, 28), { wash: '#6B4838', ink: PAL.ink, sw: 1.4 });
    paint(rectPts(tx + 90, ty + th + 48, tw - 180, 140), { wash: '#523528', ink: PAL.ink, sw: 1.1 });
    paint(rrPts(tx, ty, tw, th, 18), { wash: '#FBE6C2', ink: PAL.ink, sw: 1.6 });

    // Inside the TV: two little drama characters arguing then hugging with hearts!
    const argue = lt < 6.2;
    him(tx + 175, ty + 295, 14, argue
      ? { ...feel('angry', t), view: 'q', aR: .9 * Math.sin(t * 14), emote: 'anger' }
      : { ...feel('love', t), view: 'q', aR: .6, emote: 'hearts' }, L_CAFE, t);
    her(tx + 345, ty + 295, 14, argue
      ? { ...feel('angry', t + .3), view: 'q', flip: true, aL: .9 * Math.cos(t * 14), emote: '!' }
      : { ...feel('love', t + .3), view: 'q', flip: true, aL: .6, emote: 'heart' }, L_CAFE);

    // Him on the left watching wistfully
    const m = emotions(lt, [
      [0, 'bored', { lookX: .9, lookY: -.2, emote: null }],
      [5.5, 'sad', { lookX: .9, lookY: -.2, emote: 'dots' }],
      [9.2, 'hopeful', { lookX: .8, lookY: -.1, emote: null }],
    ], { take: .55 });
    him(540, 835, 30, { ...m, view: 'q' }, L_CAFE, t);

    camEnd();
    boilSeed('wipe4');
    if (lt > dur - .45) brushWipe((lt - (dur - .45)) / .9, ['#1A213D', '#F6C572']);
  }

  // ---------- 5 · 60.50–79.80s · Chorus 1A: Street-Corner Reunion & Friend's Hug ----------
  function shot05_chorus1A(t, lt, dur) {
    // 60.75 你在我的面前微笑装不知道 / 65.24 其实你告白的简讯我有收到
    // 70.14 你说我爱太晚到 / 71.96 为何你爱要晚到 / 75.18 只能像朋友般拥抱
    const tMsg = 65.24 - 60.50;  // 4.74s
    const tLate = 70.14 - 60.50; // 9.64s
    const tHug = 75.18 - 60.50;  // 14.68s

    const hugK = ease(seg(lt, tHug, tHug + 1.4)) * (1 - .45 * ease(seg(lt, tHug + 3.0, dur)));
    camBegin(960, lerp(545, 515, hugK), lerp(1.0, 1.15, hugK));
    skyBg(L_CAFE, t);
    destinyClock(960, 255, 165, t, ease(seg(lt, tLate - .3, tLate + .8)));
    streetGround(L_CAFE);
    storefront(380, 1160, L_CAFE, 'cafe', t);
    lampPost(250, L_CAFE, 1);
    lampPost(1670, L_CAFE, 1);

    const hx = lerp(660, 885, hugK);
    const fx = lerp(1260, 1035, hugK);

    const mh = emotions(lt, [
      [0, 'happy', { lookX: .8, blush: .75, emote: 'sweat' }],
      [tLate, 'sad', { lookX: .8, emote: null }],
      [tHug, 'relieved', { eyes: 'closed', mouth: 'smile', blush: .6, emote: null }],
    ], { take: .6 });

    const mf = emotions(lt, [
      [0, 'shy', { lookX: -.8, blush: .65, emote: null }],
      [tMsg, 'love', { lookX: -.8, emote: 'heart' }],
      [tLate + 1.5, 'sad', { eyes: 'teary', lookX: -.8, emote: null }],
      [tHug, 'relieved', { eyes: 'closed', mouth: 'smile', blush: .65, emote: null }],
    ], { take: .6 });

    him(hx, GROUND - 16, 27, {
      ...mh,
      view: 'q',
      aR: lerp(.2, .95, hugK),
      aL: lerp(-.2, .65, hugK),
    }, L_CAFE, t);

    her(fx, GROUND - 16, 26, {
      ...mf,
      view: 'q',
      flip: true,
      aL: lerp(.2, .95, hugK),
      aR: lerp(-.2, .65, hugK),
    }, L_CAFE);

    rainOverlay(L_CAFE, t, .7);
    camEnd();
    if (lt < .45) brushWipe(.5 + lt / .9, ['#1A213D', '#F6C572']);
  }

  // ---------- 6 · 79.80–101.50s · Chorus 1B: Someone Arrived Earlier & Keeping Tenderness Safe ----------
  function shot06_chorus1B(t, lt, dur) {
    // 80.00 你说他的用心付出比我早到 / 84.83 对你的温柔只能放心里收好
    // 89.94 你说我爱太晚到 / 91.73 为何你爱要晚到 / 94.86 怎么做我们彼此才不会伤到
    const tKeep = 84.83 - 79.80; // 5.03s
    const tClock = 89.94 - 79.80; // 10.14s
    const tHurt = 94.86 - 79.80; // 15.06s

    camBegin(960 + 20 * Math.sin(lt * .35), 540, lerp(1.02, 1.1, ease(seg(lt, 0, dur))));
    skyBg(L_CAFE, t);
    destinyClock(960, 250, 175, t, ease(seg(lt, tClock - .4, tClock + .8)));
    streetGround(L_CAFE);
    storefront(380, 1160, L_CAFE, 'cafe', t);

    // In the warm shop window on the right: the other guy who arrived earlier
    if (lt < tClock) {
      otherGuy(1290, GROUND - 105, 18, { ...feel('neutral', t), view: 'q', flip: true, noShadow: true }, L_CAFE);
    }

    const mh = emotions(lt, [
      [0, 'sad', { lookX: .8, emote: null }],
      [tClock, 'nervous', { lookX: .7, emote: null }],
      [tHurt, 'sad', { eyes: 'teary', mouth: 'flat', lookX: .6, emote: null }],
    ], { take: .55 });

    const mf = emotions(lt, [
      [0, 'sad', { lookX: -.7, emote: null }],
      [tKeep, 'hopeful', { lookX: -.4, lookY: .3, emote: 'heart' }],
      [tClock + 1.5, 'cry', { lookX: -.7, emote: null }],
      [tHurt, 'sad', { eyes: 'teary', mouth: 'wobble', emote: null }],
    ], { take: .55 });

    const reach = ease(seg(lt, tHurt, tHurt + 2.0)) * (1 - .5 * ease(seg(lt, tHurt + 3.5, dur)));
    him(640, GROUND - 16, 27, { ...mh, view: 'q', aR: lerp(-.2, .85, reach) }, L_CAFE, t);
    her(1280, GROUND - 16, 26, { ...mf, view: 'q', flip: true, aL: lerp(-.2, .85, reach) }, L_CAFE);

    // A fragile glowing heart floating between them that both hesitate to touch
    if (lt > tKeep) {
      const hk = ease(seg(lt, tKeep, tKeep + 1.0));
      glow(960, GROUND - 165, 150 * hk, '#FF94B2', .6 * hk);
      paint(heartPts(960, GROUND - 165 + Math.sin(t * 3) * 8, 30 * hk), {
        wash: '#F79BB4',
        ink: PAL.ink,
        sw: .9,
      });
    }

    rainOverlay(L_CAFE, t, .65);
    camEnd();
    if (lt > dur - .6) ginkgoWipe((lt - (dur - .6)) / 1.2, '#E8AA38', '#C7AB82');
  }

  // ---------- 7 · 101.50–119.40s · Interlude 1: Sepia Autumn Bridge & Paper Planes ----------
  function shot07_interlude1(t, lt, dur) {
    camBegin(960 + 40 * Math.sin(lt * .3), 520, lerp(.96, 1.06, ease(seg(lt, 0, dur))));
    skyBg(L_SEP, t);
    streetGround(L_SEP);
    lampPost(360, L_SEP, 0);
    lampPost(1560, L_SEP, 0);

    // Stone bridge arch in sepia memory
    boilSeed('bridge');
    paint(rectPts(260, GROUND - 95, 1400, 80, 2), { wash: '#B89A70', ink: PAL.ink, sw: 1.1 });
    for (let i = 0; i < 7; i++) {
      paint(rectPts(320 + i * 210, GROUND - 155, 24, 62), { wash: '#9C7E54', ink: PAL.ink, sw: .8 });
    }
    inkLine([[240, GROUND - 155], [1680, GROUND - 155]], 2.2, PAL.ink, 'ink', 0);

    // Him and Her standing on opposite ends of the bridge in memory, watching two paper planes / ginkgo leaves
    him(560, GROUND - 95, 22, { ...feel('hopeful', t), view: 'q', lookY: -.6 }, L_SEP, t);
    her(1360, GROUND - 95, 21, { ...feel('hopeful', t + .4), view: 'q', flip: true, lookY: -.6 }, L_SEP);

    // Swirling golden ginkgo leaves / paper planes across the sky
    for (let i = 0; i < 16; i++) {
      boilSeed('leaf' + i);
      const prog = frac(lt / dur * 1.2 + hash(i * 7));
      const lx = -100 + prog * (W + 200);
      const ly = 220 + 260 * hash(i * 13) + Math.sin(t * 2.5 + i) * 45;
      const r = 14 + 10 * hash(i);
      paint(starPts(lx, ly, r, .45, 3, t * 1.5 + i), { wash: i % 2 === 0 ? '#E8AA38' : '#D97757', ink: PAL.ink, sw: .55 });
    }

    camEnd();
    if (lt < .6) ginkgoWipe(.5 + lt / 1.2, '#E8AA38', '#C7AB82');
    boilSeed('wipe7');
    if (lt > dur - .45) brushWipe((lt - (dur - .45)) / .9, ['#F2C49B', '#6B88B8']);
  }

  // ---------- 8 · 119.40–138.60s · Verse 2A: Favorite Drink, Bread & Distant Smile ----------
  function shot08_bakery(t, lt, dur) {
    // 119.64 我记得你爱喝的饮料 / 124.27 也排队买你爱吃的面包 / 129.04 却只能对你远远的会心一笑 / 134.32 连你的手都牵不到
    const tBread = 124.27 - 119.40; // 4.87s
    const tFar   = 129.04 - 119.40; // 9.64s
    const tHand  = 134.32 - 119.40; // 14.92s

    // Camera starts close on Him outside the bakery, then pulls wide at tFar to reveal Her across the street
    const wideK = ease(seg(lt, tFar - .4, tFar + 1.2));
    camBegin(lerp(620, 960, wideK), lerp(560, 540, wideK), lerp(1.22, .96, wideK));
    skyBg(L_BAKE, t);
    streetGround(L_BAKE);
    storefront(160, 760, L_BAKE, 'bakery', t);
    lampPost(1060, L_BAKE, 0);

    const mh = emotions(lt, [
      [0, 'happy', { lookX: .4, emote: 'music' }],
      [tBread, 'proud', { lookX: .6, blush: .5, emote: 'spark' }],
      [tFar, 'relieved', { eyes: 'happy', mouth: 'smile', blush: .75, lookX: .9, emote: null }],
      [tHand, 'sad', { eyes: 'happy', mouth: 'smile', blush: .6, lookX: .9, emote: null }],
    ], { take: .6 });

    const reachK = ease(seg(lt, tHand, tHand + 1.5));
    him(520, GROUND - 16, 27, {
      ...mh,
      view: 'q',
      aL: .45,
      aR: lerp(.55, 1.05, reachK),
      armL: () => drinkProp(0, 0, 1.05),
      armR: lt > tBread - .3 ? () => breadProp(0, 0, 1.05) : null,
    }, L_BAKE, t);

    // Her across the street on the right, turning to smile back from afar
    if (lt > tFar - .8) {
      const mf = emotions(lt, [
        [tFar - .8, 'surprised', { lookX: -.8, emote: null }],
        [tFar + .8, 'happy', { lookX: -.9, blush: .7, emote: 'heart' }],
      ], { take: .55 });
      her(1460, GROUND - 16, 25, { ...mf, view: 'q', flip: true }, L_BAKE);
    }

    camEnd();
    if (lt < .45) brushWipe(.5 + lt / .9, ['#F2C49B', '#6B88B8']);
    boilSeed('wipe8');
    if (lt > dur - .45) brushWipe((lt - (dur - .45)) / .9, ['#2C2552', '#F598B2']);
  }

  // ---------- 9 · 138.60–161.20s · Verse 2B: Extra Movie Tickets & Synchronized Heartbeat ----------
  function shot09_cinema(t, lt, dur) {
    // 138.90 你说你有多的电影票 / 143.67 你知道事情没有那么巧
    // 148.82 但很窝心的是你怕我无聊 / 154.11 你比他清楚我的心跳 / 156.20 我比他清楚你的心跳
    const tWarm = 148.82 - 138.60; // 10.22s
    const tBeat = 154.11 - 138.60; // 15.51s

    const pushK = ease(seg(lt, tWarm, tBeat + 1.5));
    camBegin(960, lerp(545, 520, pushK), lerp(1.02, 1.18, pushK));
    skyBg(L_CINE, t);
    streetGround(L_CINE);
    storefront(380, 1160, L_CINE, 'cinema', t);

    const mh = emotions(lt, [
      [0, 'surprised', { lookX: .8, emote: '!' }],
      [4.8, 'playful', { lookX: .8, blush: .6, emote: null }],
      [tWarm, 'love', { lookX: .8, blush: .75, emote: 'heart' }],
      [tBeat, 'starstruck', { lookX: .8, blush: .85, emote: null }],
    ], { take: .6 });

    const mf = emotions(lt, [
      [0, 'shy', { lookX: .5, blush: .85, emote: 'sweat' }],
      [5.5, 'playful', { lookX: -.8, blush: .75, emote: null }],
      [tWarm, 'love', { lookX: -.8, blush: .8, emote: 'hearts' }],
      [tBeat, 'love', { lookX: -.8, blush: .9, emote: null }],
    ], { take: .6 });

    him(690, GROUND - 16, 28, { ...mh, view: 'q' }, L_CINE, t);
    her(1230, GROUND - 16, 27, {
      ...mf,
      view: 'q',
      flip: true,
      aL: .65 + .1 * Math.sin(t * 4),
      armL: () => ticketsProp(0, 0, 1.15, -.15),
    }, L_CINE);

    // Synchronized glowing heartbeat wave between their chests
    if (lt > tWarm) {
      heartbeatBridge(730, GROUND - 140, 1190, GROUND - 140, t, ease(seg(lt, tWarm, tWarm + 1.5)));
    }

    camEnd();
    if (lt < .45) brushWipe(.5 + lt / .9, ['#2C2552', '#F598B2']);
    boilSeed('wipe9');
    if (lt > dur - .45) brushWipe((lt - (dur - .45)) / .9, ['#1A213D', '#F6C572']);
  }

  // ---------- 10 · 161.20–201.80s · Chorus 2: Shared Umbrella, Embrace & Parting Lamps ----------
  function shot10_chorus2(t, lt, dur) {
    // 161.42 你在我的面前微笑装不知道 ... 175.58 只能像朋友般拥抱 ... 195.06 怎么做我们彼此才不会伤到
    const tHug  = 175.58 - 161.20; // 14.38s
    const tPart = 181.00 - 161.20; // 19.80s
    const tHurt = 195.06 - 161.20; // 33.86s

    const together = ease(seg(lt, 2.0, tHug + 1.0)) * (1 - ease(seg(lt, tPart, tPart + 3.0)));
    camBegin(960 + 22 * Math.sin(lt * .35), lerp(540, 510, together), lerp(1.0, 1.14, together));
    skyBg(L_CAFE, t);
    destinyClock(960, 240, 185, t, ease(seg(lt, 8.5, 11.0)));
    streetGround(L_CAFE);
    storefront(380, 1160, L_CAFE, 'cafe', t);
    lampPost(280, L_CAFE, 1);
    lampPost(1640, L_CAFE, 1);

    const hx = lerp(590, 880, together);
    const fx = lerp(1330, 1040, together);

    // Umbrella held over them while together, or over Him as they step apart
    umbrellaProp(lerp(hx + 20, 960, together), GROUND - 265, 195, '#3B5B8C', .08 * (1 - together));

    const mh = emotions(lt, [
      [0, 'happy', { lookX: .8, blush: .7, emote: 'sweat' }],
      [9.4, 'sad', { lookX: .8, emote: null }],
      [tHug, 'relieved', { eyes: 'closed', mouth: 'smile', blush: .65, emote: null }],
      [tPart + 2.0, 'sad', { eyes: 'teary', lookX: .8, emote: null }],
      [tHurt, 'cry', { lookX: .7, emote: null }],
    ], { take: .6 });

    const mf = emotions(lt, [
      [0, 'shy', { lookX: -.8, blush: .7, emote: 'heart' }],
      [11.0, 'sad', { eyes: 'teary', lookX: -.8, emote: null }],
      [tHug, 'relieved', { eyes: 'closed', mouth: 'smile', blush: .7, emote: null }],
      [tPart + 2.0, 'sad', { eyes: 'teary', lookX: -.8, emote: null }],
      [tHurt, 'cry', { lookX: -.7, emote: null }],
    ], { take: .6 });

    him(hx, GROUND - 16, 27, { ...mh, view: 'q', aR: lerp(.2, .9, together) }, L_CAFE, t);
    her(fx, GROUND - 16, 26, { ...mf, view: 'q', flip: true, aL: lerp(.2, .9, together) }, L_CAFE);

    rainOverlay(L_CAFE, t, .95);
    camEnd();
    if (lt < .45) brushWipe(.5 + lt / .9, ['#1A213D', '#F6C572']);
    if (lt > dur - .7) ginkgoWipe((lt - (dur - .7)) / 1.4, '#2D2B63', '#FFD87A');
  }

  // ---------- 11 · 201.80–220.00s · Interlude 2: Floating Memory Polaroids in the Night Sky ----------
  function shot11_interlude2(t, lt, dur) {
    camBegin(960, lerp(480, 540, ease(seg(lt, 0, dur))), lerp(.95, 1.08, ease(seg(lt, 0, dur))));
    skyBg(L_CLIM, t);

    // Distant hilltop silhouette at the bottom
    boilSeed('hill11');
    paint(ellPts(960, 1020, 1450, 240, 36, 3), { wash: '#1B2042', ink: PAL.ink, sw: 1 });

    // Three floating Polaroid snapshots drifting across the starry sky
    for (const [k, x0, y0, rot0] of [[1, 460, 380, -.12], [2, 960, 320, .06], [3, 1460, 380, -.08]]) {
      const floatY = y0 + Math.sin(t * 1.8 + k * 1.5) * 18;
      const enter = backOut(seg(lt, (k - 1) * 2.2, (k - 1) * 2.2 + 1.6));
      if (enter > .02) {
        glow(x0, floatY, 210 * enter, '#FFD87A', .45 * enter);
        memoryPhoto(x0, floatY, 1.05 * enter, rot0 + .04 * Math.sin(t + k), k, t);
      }
    }

    // Him and Her walking slowly toward the hilltop crest
    him(720, 815, 21, { ...feel('hopeful', t), view: 'q', lookY: -.6 }, L_CLIM, t);
    her(1200, 815, 20, { ...feel('hopeful', t + .5), view: 'q', flip: true, lookY: -.6 }, L_CLIM);

    camEnd();
    if (lt < .7) ginkgoWipe(.5 + lt / 1.4, '#2D2B63', '#FFD87A');
  }

  // ---------- 12 · 220.00–239.00s · Final Chorus A: Missing the Scent of Happiness ----------
  function shot12_climaxA(t, lt, dur) {
    // 220.18 你在我的面前微笑装不知道 / 224.42 他没有不好只是话越聊越少
    // 229.36 你说我爱太晚到 / 231.16 为何你爱要晚到 / 234.26 错过了幸福的味道 Yeah
    const tScent = 234.26 - 220.00; // 14.26s

    camBegin(960 + 18 * Math.sin(lt * .4), 530, lerp(1.0, 1.12, ease(seg(lt, 0, dur))));
    skyBg(L_CLIM, t);
    destinyClock(960, 245, 195, t, ease(seg(lt, 8.8, 11.2)));

    boilSeed('hill12');
    paint(ellPts(960, 995, 1480, 230, 36, 3), { wash: '#1E2448', ink: PAL.ink, sw: 1.1 });
    bistroTable(960, 790, L_CLIM);
    coffeeCup(960, 688, 1.35, t, true);

    // On "错过了幸福的味道", golden aroma stars spiral up from the cup into the night sky
    if (lt > tScent - .5) {
      const sk = seg(lt, tScent - .5, dur);
      for (let i = 0; i < 12; i++) {
        boilSeed('scent' + i);
        const q = frac(sk * 1.4 + i / 12);
        const sx = 960 + Math.sin(q * 8 + i) * (40 + 180 * q);
        const sy = 665 - q * 480;
        paint(starPts(sx, sy, 14 * (1 - q * .5), .38, 4, t * 2 + i), {
          wash: i % 2 === 0 ? '#FFE27A' : '#F79BB4',
          washOp: Math.round(240 * (1 - q)),
          ink: null,
        });
      }
    }

    const mh = emotions(lt, [
      [0, 'happy', { lookX: .8, blush: .65, emote: null }],
      [9.2, 'sad', { eyes: 'teary', lookX: .8, emote: null }],
      [tScent, 'sad', { eyes: 'teary', mouth: 'smile', lookY: -.6, emote: null }],
    ], { take: .6 });

    const mf = emotions(lt, [
      [0, 'sad', { lookX: -.8, emote: null }],
      [4.4, 'thinking', { lookX: -.6, lookY: .2, emote: 'dots' }],
      [11.0, 'cry', { lookX: -.8, emote: null }],
      [tScent, 'sad', { eyes: 'teary', lookY: -.6, emote: null }],
    ], { take: .6 });

    him(660, 792, 28, { ...mh, view: 'q' }, L_CLIM, t);
    her(1260, 792, 27, { ...mf, view: 'q', flip: true }, L_CLIM);

    camEnd();
  }

  // ---------- 13 · 239.00–263.00s · Final Chorus B: Photographing with Memory & Right Person Wrong Time ----------
  function shot13_climaxB(t, lt, dur) {
    // 239.14 用记忆在对你拍照让人难熬 / 243.89 其实我对于你的好也曾动摇
    // 250.43 你说我爱太晚到 / 252.83 为何你爱要晚到 / 255.03 对的人 不对的时间 却放不掉
    const tSnap  = 2.1;             // camera shutter flash at 241.1s
    const tWaver = 243.89 - 239.00; // 4.89s
    const tFinal = 255.03 - 239.00; // 16.03s

    const climaxK = ease(seg(lt, tFinal - 1.0, tFinal + 2.5));
    camBegin(960, lerp(535, 495, climaxK), lerp(1.04, 1.16, climaxK));
    skyBg(L_CLIM, t);

    // Giant celestial destiny clock blazing behind them on "对的人 不对的时间 却放不掉"
    destinyClock(960, 340, lerp(210, 310, climaxK), t, .65 + .35 * climaxK);

    boilSeed('hill13');
    paint(ellPts(960, 995, 1480, 230, 36, 3), { wash: '#1E2448', ink: PAL.ink, sw: 1.1 });

    // Floating Polaroid photo captured at tSnap (drifts aside when the destiny clock blazes)
    if (lt > tSnap) {
      const pk = backOut(seg(lt, tSnap, tSnap + .8));
      memoryPhoto(lerp(960, 410, climaxK), lerp(350, 275, climaxK), lerp(.85, .68, climaxK) * pk, -.08 + .03 * Math.sin(t * 2), 1, t);
    }

    const mh = emotions(lt, [
      [0, 'determined', { lookX: .8, emote: null }],
      [tSnap + .3, 'sad', { eyes: 'teary', mouth: 'smile', blush: .7, emote: null }],
      [tFinal, 'cry', { eyes: 'teary', mouth: 'smile', blush: .8, emote: 'heart' }],
    ], { take: .65 });

    const mf = emotions(lt, [
      [0, 'shy', { lookX: -.8, blush: .8, emote: null }],
      [tWaver, 'sad', { eyes: 'teary', lookX: -.8, emote: 'heart' }],
      [tFinal, 'cry', { eyes: 'teary', mouth: 'smile', blush: .85, emote: 'hearts' }],
    ], { take: .65 });

    const reach = ease(seg(lt, tFinal, tFinal + 3.0));
    him(640, 792, 29, {
      ...mh,
      view: 'q',
      aR: lt < tSnap + .4 ? 1.1 : lerp(.2, .95, reach),
    }, L_CLIM, t);

    her(1280, 792, 28, {
      ...mf,
      view: 'q',
      flip: true,
      aL: lerp(.2, .95, reach),
    }, L_CLIM);

    // Golden star-petals rising around them during the final line
    if (lt > tFinal - 1.0) {
      for (let i = 0; i < 18; i++) {
        boilSeed('fp' + i);
        const py = 850 - ((t * 95 + i * 55) % 760);
        const px = 220 + hash(i * 17) * 1480 + Math.sin(t * 2 + i) * 35;
        paint(starPts(px, py, 9 + 6 * hash(i), .4, 4, t * 2 + i), {
          wash: i % 2 === 0 ? '#FFE27A' : '#F79BB4',
          ink: null,
        });
      }
    }

    camEnd();
    // Warm shutter flash at tSnap
    if (lt > tSnap - .06 && lt < tSnap + .45) {
      flash(Math.exp(-Math.abs(lt - tSnap) * 9) * .75, '#FFF5E2');
    }
    boilSeed('wipe13');
    if (lt > dur - .5) brushWipe((lt - (dur - .5)) / 1.0, ['#141936', '#F4B860']);
  }

  // ---------- 14 · 263.00–289.96s · Outro: The Last Silly Smile & Iris Out ----------
  function shot14_outro(t, lt, dur) {
    // Rhymes with Shot 1: Back at the street-corner café table after the rain has cleared.
    // She walks gently away into the distance and waves goodbye; He holds the glowing photo & smiles his last "傻笑".
    const pushK = ease(seg(lt, 8.0, dur - 2.0));
    camBegin(lerp(960, 760, pushK), lerp(545, 515, pushK), lerp(1.0, 1.28, pushK));
    skyBg({ ...L_RAIN, rain: 0, stars: .85 }, t);
    streetGround(L_RAIN);
    lampPost(620, L_RAIN, 1);
    storefront(160, 920, L_RAIN, 'cafe', t);
    bistroTable(920, GROUND - 20, L_RAIN);
    coffeeCup(920, GROUND - 122, 1.25, t, true);

    // Her walking gently off to the right (0..12s), pausing to wave goodbye
    if (lt < 13.5) {
      const w = stroll(lt, .5, 8.5, 1080, 1620, 25);
      const waveOn = lt > 8.5 && lt < 11.8;
      her(w.x, GROUND - 16, 25, {
        ...feel('relieved', t),
        eyes: 'happy',
        mouth: 'smile',
        blush: .7,
        view: waveOn ? 'q' : 'side',
        flip: waveOn,
        walk: lt < 8.5 ? w.walk : 0,
        dy: lt < 8.5 ? w.dy : 0,
        aL: waveOn ? 1.2 + .35 * Math.sin(t * 10) : .2,
      }, L_RAIN);
    }

    // Him staying under the warm streetlamp, looking at the glowing memory and giving a tender silly smile (傻笑)
    const mh = emotions(lt, [
      [0, 'sad', { lookX: .9, emote: null }],
      [8.5, 'relieved', { lookX: .8, mouth: 'smile', blush: .65, emote: null }],
      [14.0, 'happy', { eyes: 'happy', mouth: 'smile', blush: .85, emote: 'heart' }],
    ], { take: .55 });

    him(760, GROUND - 16, 29, {
      ...mh,
      view: lt > 13.5 ? 'front' : 'q',
      aL: lt > 8.5 && lt < 12.0 ? 1.15 + .3 * Math.sin(t * 9) : .25,
    }, L_RAIN, t);

    const sp = toScreen(760, GROUND - 16 - 4.5 * 29);
    camEnd();
    if (lt < .5) brushWipe(.5 + lt / 1.0, ['#141936', '#F4B860']);
    if (lt > dur - 4.2) {
      const ik = seg(lt, dur - 4.2, dur - .2);
      iris(sp[0], sp[1], ik < .65 ? lerp(1600, 210, ease(ik / .65)) : lerp(210, 0, easeIn((ik - .65) / .35)), PAL.ink);
    }
  }

  shots([
    [0.00,   shot01_intro],
    [19.50,  shot02_phone],
    [38.20,  shot03_raincoat],
    [48.00,  shot04_tvdrama],
    [60.50,  shot05_chorus1A],
    [79.80,  shot06_chorus1B],
    [101.50, shot07_interlude1],
    [119.40, shot08_bakery],
    [138.60, shot09_cinema],
    [161.20, shot10_chorus2],
    [201.80, shot11_interlude2],
    [220.00, shot12_climaxA],
    [239.00, shot13_climaxB],
    [263.00, shot14_outro],
  ]);
})();
