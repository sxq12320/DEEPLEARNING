// shaxiao_set.js: Shared world, cast, props, and transitions for 周杰伦 × 袁咏琳《傻笑》.
// Every function is a pure function of time `t` and uses boilSeed() so static elements boil cleanly at 12 fps.

const GROUND = 845;

const LOOK = {
  rainDusk: {
    sky: '#222B4A', sky2: '#3B4876', glow: '#F4B860',
    bldg: '#2F3659', bldg2: '#3D3B60', roof: '#1C2038',
    winLit: '#F7C96E', road: '#25283E', curb: '#434868',
    awning1: '#B85042', awning2: '#F2E3C9',
    rain: 0.85, stars: 0.2, sepia: false,
  },
  herRoom: {
    sky: '#3C2F52', sky2: '#6E5078', glow: '#F5B971',
    bldg: '#47385C', bldg2: '#5B466E', roof: '#2A203B',
    winLit: '#FAD48B', road: '#322844', curb: '#57476B',
    awning1: '#C46252', awning2: '#F5E8D0',
    rain: 0.65, stars: 0.1, sepia: false,
  },
  cafeNight: {
    sky: '#1A213D', sky2: '#2D3A68', glow: '#F6C572',
    bldg: '#293052', bldg2: '#383457', roof: '#161A30',
    winLit: '#FFD278', road: '#1E2236', curb: '#3B405E',
    awning1: '#B84A3E', awning2: '#F4E6CE',
    rain: 0.75, stars: 0.45, sepia: false,
  },
  sepia: {
    sky: '#EBD7B2', sky2: '#F6E8CC', glow: '#FFF3D6',
    bldg: '#C7AB82', bldg2: '#B89A70', roof: '#8C6B4A',
    winLit: '#F7E2B8', road: '#A68963', curb: '#D1B892',
    awning1: '#A6654B', awning2: '#F2E5C8',
    rain: 0, stars: 0, sepia: true,
  },
  bakery: {
    sky: '#6B88B8', sky2: '#F2C49B', glow: '#FFE4A8',
    bldg: '#8C6D62', bldg2: '#A68372', roof: '#5E443D',
    winLit: '#FFF0C2', road: '#5D5666', curb: '#8A8090',
    awning1: '#C95B48', awning2: '#FFF3DC',
    rain: 0, stars: 0, sepia: false,
  },
  cinema: {
    sky: '#1E1A3C', sky2: '#3D2C66', glow: '#F598B2',
    bldg: '#2D244C', bldg2: '#423166', roof: '#17132B',
    winLit: '#FFD479', road: '#221D38', curb: '#453B66',
    awning1: '#C44569', awning2: '#F8E9D2',
    rain: 0.25, stars: 0.75, sepia: false,
  },
  climaxNight: {
    sky: '#141936', sky2: '#2D2B63', glow: '#FFD87A',
    bldg: '#23274A', bldg2: '#2E2B52', roof: '#121528',
    winLit: '#FFE08A', road: '#1C2038', curb: '#363C60',
    awning1: '#B84A3E', awning2: '#F4E6CE',
    rain: 0.3, stars: 1.0, sepia: false,
  },
};

const SEP_TONE = '#B99468';
const HIM_COL = { col: PAL.clay, dk: PAL.clayDk, lt: '#F5B394' };
const HER_COL = { col: '#EC94AA', dk: '#BC5D76', lt: '#FCD2DC' };
const OTHER_COL = { col: '#6E7A9C', dk: '#49526E', lt: '#9BA6C4' };

const sepiaShift = c => ({
  col: mixCol(c.col, SEP_TONE, .42),
  dk: mixCol(c.dk, SEP_TONE, .42),
  lt: mixCol(c.lt, SEP_TONE, .42),
});

// ---------- Characters ----------
// Him: wears a warm slate-blue knit scarf that sways slightly with t
function knitScarf(u, sw, t = 0, sepia = false) {
  const sc = sepia ? '#7A6855' : '#436B96';
  const hi = sepia ? '#A69076' : '#74A2CF';
  // Neck wrap
  paint(rrPts(-3.6 * u, -3.2 * u, 7.2 * u, 1.15 * u, .35 * u), { wash: sc, ink: PAL.ink, sw: sw * .65 });
  inkLine([[-2.8 * u, -2.65 * u], [2.8 * u, -2.65 * u]], sw * .45, hi, 'inkfine', 0);
  // Hanging scarf tail with gentle breeze wave
  const wave = Math.sin(t * 3.2) * .35 * u;
  paint([
    [1.2 * u, -2.3 * u],
    [2.5 * u, -2.3 * u],
    [2.8 * u + wave, -.3 * u],
    [1.3 * u + wave, -.2 * u],
  ], { wash: sc, ink: PAL.ink, sw: sw * .6, curv: .2 });
}

// Her: wears the camellia flower hat + a delicate cream capelet collar
function capeCollar(u, sw) {
  paint([[-2.6 * u, -2.9 * u], [0, -2.1 * u], [-1.2 * u, -1.7 * u]], { wash: PAL.cream, ink: PAL.ink, sw: sw * .5 });
  paint([[2.6 * u, -2.9 * u], [0, -2.1 * u], [1.2 * u, -1.7 * u]], { wash: PAL.cream, ink: PAL.ink, sw: sw * .5 });
  paint(ellPts(0, -2.35 * u, .38 * u, .38 * u, 8), { wash: '#F2B84B', ink: PAL.ink, sw: sw * .45 });
}

function him(x, y, u, o = {}, L = LOOK.cafeNight, t = 0) {
  const userDraw = o.draw;
  clawd(x, y, u, {
    ...o,
    ...(L.sepia ? sepiaShift(HIM_COL) : HIM_COL),
    tint: null,
    draw: (uu, sw) => {
      knitScarf(uu, sw, t, L.sepia);
      if (userDraw) userDraw(uu, sw);
    },
  });
}

function her(x, y, u, o = {}, L = LOOK.cafeNight) {
  const userDraw = o.draw;
  clawd(x, y, u, {
    ...o,
    ...(L.sepia ? sepiaShift(HER_COL) : HER_COL),
    tint: null,
    hat: 'flower',
    draw: (uu, sw) => {
      capeCollar(uu, sw);
      if (userDraw) userDraw(uu, sw);
    },
  });
}

function otherGuy(x, y, u, o = {}, L = LOOK.cafeNight) {
  clawd(x, y, u, { ...o, ...(L.sepia ? sepiaShift(OTHER_COL) : OTHER_COL), tint: null });
}

// ---------- Organic Cloud / Foliage Shape Helper ----------
function lumpy(cx, cy, rx, ry, seed, n = 20) {
  const p = [];
  for (let k = 0; k < n; k++) {
    const a = (k / n) * TAU;
    const w = 1 + .12 * Math.sin(a * 5 + seed) + .06 * Math.sin(a * 9 + seed * 2);
    p.push([cx + Math.cos(a) * rx * w, cy + Math.sin(a) * ry * w]);
  }
  return p;
}

// ---------- Sky, Stars, Clouds & Rain ----------
function skyBg(L, t) {
  boilSeed('sky');
  paint(rectPts(-400, -300, W + 800, H + 600), { wash: L.sky, ink: null });
  glow(W * .5, 620, 950, L.sky2, .55);
  glow(W * .55, 520, 520, L.glow, .42);

  if (L.stars > 0) {
    const nStars = Math.round(34 * L.stars);
    for (let i = 0; i < nStars; i++) {
      boilSeed('st' + i);
      const x = hash(i * 3.1) * (W + 400) - 200;
      const y = hash(i * 7.3 + 20) * 520 - 60;
      const tw = .5 + .5 * Math.sin(t * (1.8 + 1.5 * hash(i + 9)) + i);
      paint(starPts(x, y, (3 + 4.5 * hash(i + 3)) * tw, .35, 4), {
        wash: PAL.cream,
        washOp: Math.min(255, Math.round((130 + 115 * tw) * L.stars)),
        ink: null,
      });
    }
  }

  for (let i = 0; i < 3; i++) {
    boilSeed('cloud' + i);
    const x = ((hash(i + 20) * 2200 + t * (10 + 5 * i)) % 2500) - 300;
    const y = 135 + i * 105;
    paint(lumpy(x, y, 185 + 50 * i, 42, 15 + i), {
      wash: mixCol(L.sky2, PAL.cream, L.sepia ? .35 : .18),
      washOp: 140,
      ink: null,
    });
  }
}

// Rain streaks + ground puddle ripples (drawn in front of characters/buildings)
function rainOverlay(L, t, density = 1) {
  const amt = (L.rain || 0) * density;
  if (amt <= 0.02) return;
  const nDrops = Math.round(32 * amt);
  for (let i = 0; i < nDrops; i++) {
    boilSeed('rain' + i);
    const speed = 1150 + 350 * (i % 3);
    const cycle = (t * speed + hash(i * 19) * 1400) % 1350;
    const x = hash(i * 37 + Math.floor((t * speed + hash(i * 19) * 1400) / 1350)) * (W + 300) - 100 - cycle * .14;
    const y = cycle - 150;
    const len = 32 + 18 * (i % 2);
    inkLine([[x, y], [x - len * .14, y + len]], .75, mixCol(PAL.sky, PAL.cream, .5), 'inkfine', 0);
  }
  // Puddle ripples on the road
  for (let i = 0; i < 5; i++) {
    const prog = frac(t * (.7 + .15 * i) + hash(i * 11));
    const rx = 220 + i * 360 + 50 * hash(i);
    const ry = GROUND + 45 + (i % 2) * 32;
    boilSeed('rip' + i);
    paint(ellPts(rx, ry, 12 + 48 * prog, 3 + 10 * prog, 14), {
      ink: mixCol(PAL.sky, PAL.cream, .4),
      sw: .65 * (1 - prog),
    });
  }
}

// ---------- Street, Café, Bakery & Cinema Exteriors ----------
function streetGround(L) {
  boilSeed('ground');
  // Sidewalk
  paint(rectPts(-400, GROUND - 45, W + 800, 65), { wash: L.curb, ink: PAL.ink, sw: .9 });
  for (let i = -2; i < 14; i++) {
    inkLine([[i * 165, GROUND - 43], [i * 165 - 35, GROUND + 18]], .55, mixCol(L.curb, PAL.ink, .35), 'inkfine', 0);
  }
  // Cobblestone road
  paint(rectPts(-400, GROUND + 18, W + 800, 320), { wash: L.road, ink: PAL.ink, sw: 1 });
}

function lampPost(x, L, on = 1) {
  boilSeed('lamp' + x);
  const top = 360, col = mixCol(L.road, PAL.ink, .35);
  if (on > 0) glow(x + 58, top + 36, 210 * on, '#FFC96B', .78 * on);
  paint(rectPts(x - 8, top, 16, GROUND - top), { wash: col, ink: PAL.ink, sw: .85 });
  paint(rectPts(x - 22, GROUND - 16, 44, 18), { wash: col, ink: PAL.ink, sw: .85 });
  inkLine([[x, top + 18], [x + 36, top - 12], [x + 62, top + 8]], 2.8, col, 'ink', .5);
  paint([[x + 42, top + 8], [x + 82, top + 8], [x + 74, top + 56], [x + 50, top + 56]], {
    wash: on > 0 ? '#FFE6A3' : mixCol(L.bldg, PAL.cream, .25),
    ink: PAL.ink,
    sw: .85,
  });
  paint([[x + 36, top + 4], [x + 88, top + 4], [x + 62, top - 16]], { wash: col, ink: PAL.ink, sw: .85 });
}

// Street-corner Café / Bakery / Cinema storefront
function storefront(x0, w, L, kind = 'cafe', t = 0) {
  boilSeed('store' + x0 + kind);
  const top = 270, h = GROUND - 45 - top;
  // Main building facade
  paint(rectPts(x0, top, w, h, 2), { wash: L.bldg, ink: PAL.ink, sw: 1.1 });
  paint(rectPts(x0 - 16, top - 24, w + 32, 26, 1.5), { wash: L.roof, ink: PAL.ink, sw: 1 });

  // Upper floor windows
  for (let i = 0; i < 3; i++) {
    const wx = x0 + 65 + i * ((w - 130) / 2.3), wy = top + 48;
    const lit = i === 1 || kind === 'cinema';
    if (lit) glow(wx + 42, wy + 55, 95, L.winLit, .45);
    paint(rrPts(wx, wy, 84, 110, 12), {
      wash: lit ? L.winLit : mixCol(L.bldg, PAL.ink, .35),
      ink: PAL.ink,
      sw: .85,
    });
    inkLine([[wx + 42, wy], [wx + 42, wy + 110]], .7, PAL.ink, 'ink', 0);
    inkLine([[wx, wy + 55], [wx + 84, wy + 55]], .7, PAL.ink, 'ink', 0);
  }

  // Lower shop window (warm lit interior)
  const sx = x0 + 55, sy = top + 235, sw = w - 110, sh = 240;
  glow(sx + sw / 2, sy + sh / 2, 240, L.winLit, .55);
  paint(rrPts(sx, sy, sw, sh, 14), { wash: L.winLit, hatch: 14, hatchCol: '#E89B48', hatchOp: 45, ink: PAL.ink, sw: 1.1 });

  // If bakery, draw golden loaves & cake stands inside the window
  if (kind === 'bakery') {
    inkLine([[sx + 20, sy + 165], [sx + sw - 20, sy + 165]], 1.2, PAL.ink, 'ink', 0);
    for (let k = 0; k < 4; k++) {
      paint(ellPts(sx + 75 + k * 95, sy + 150, 34, 18, 12), { wash: '#D9823B', ink: PAL.ink, sw: .75 });
    }
  }

  // Striped awning over the shop window
  const awY = top + 205, awH = 54, nStripes = 7, stW = (w - 40) / nStripes;
  for (let i = 0; i < nStripes; i++) {
    const ax = x0 + 20 + i * stW;
    paint([[ax, awY], [ax + stW + 2, awY], [ax + stW - 4, awY + awH], [ax - 4, awY + awH]], {
      wash: i % 2 === 0 ? L.awning1 : L.awning2,
      ink: PAL.ink,
      sw: .8,
    });
  }

  // If cinema, draw a glowing bulb marquee on top of the awning
  if (kind === 'cinema') {
    paint(rrPts(x0 + 40, awY - 58, w - 80, 52, 8), { wash: '#2B193E', ink: PAL.ink, sw: 1.1 });
    for (let b = 0; b < 12; b++) {
      const bx = x0 + 62 + b * ((w - 124) / 11);
      const pulseB = .55 + .45 * Math.sin(t * 6 + b * .8);
      paint(ellPts(bx, awY - 32, 6 * pulseB, 6 * pulseB, 8), { wash: '#FFE27A', ink: null });
    }
  }
}

// Bistro table & chairs outside the café
function bistroTable(x, y, L) {
  boilSeed('table' + x);
  const iron = mixCol(L.road, PAL.ink, .5);
  inkLine([[x, y - 95], [x, y]], 2.2, iron, 'ink', 0);
  inkLine([[x - 38, y], [x + 38, y]], 2.2, iron, 'ink', 0);
  paint(ellPts(x, y - 98, 78, 16, 18), { wash: PAL.cream, ink: PAL.ink, sw: .95 });
}

// ---------- Signature Props ----------
// Porcelain coffee cup with rising steam
function coffeeCup(x, y, s = 1, t = 0, steamOn = true) {
  boilSeed('cup' + Math.round(x));
  paint(ellPts(x, y + 6 * s, 26 * s, 6 * s, 12), { wash: PAL.cream, ink: PAL.ink, sw: .75 });
  paint([[x - 16 * s, y - 18 * s], [x + 16 * s, y - 18 * s], [x + 12 * s, y + 5 * s], [x - 12 * s, y + 5 * s]], {
    wash: PAL.cream,
    ink: PAL.ink,
    sw: .85,
  });
  paint(ellPts(x + 17 * s, y - 7 * s, 6 * s, 7 * s, 8), { ink: PAL.ink, sw: .8 });
  if (steamOn) {
    for (let i = 0; i < 2; i++) {
      const ph = frac(t * .55 + i * .5);
      const sy = y - 24 * s - ph * 42 * s;
      const sx = x + (i === 0 ? -5 : 5) * s + Math.sin(t * 3 + i) * 6 * s;
      inkLine([[sx, sy + 12 * s], [sx + 4 * s, sy], [sx - 3 * s, sy - 12 * s]], .65 * (1 - ph), PAL.cream, 'inkfine', .5);
    }
  }
}

// Glowing smiley bubble ("哪怕只传来一个微笑的符号 / 我都会舍不得删掉")
function smileyBubble(x, y, r, t, alpha = 1) {
  if (alpha <= .02 || r <= 2) return;
  boilSeed('smiley');
  glow(x, y, r * 2.4, '#FFD96A', .75 * alpha);
  paint(ellPts(x, y, r, r, 18, 1), { wash: '#FFE27A', ink: PAL.ink, sw: clamp(r / 28, .6, 1.2) });
  const e = r * .28;
  for (const s of [-1, 1]) {
    inkLine([[x + s * e - e * .35, y - e * .2], [x + s * e, y - e * .55], [x + s * e + e * .35, y - e * .2]], .9, PAL.ink, 'ink', .5);
  }
  inkLine([[x - e * 1.05, y + e * .35], [x, y + e * 1.1], [x + e * 1.05, y + e * .35]], 1.0, PAL.ink, 'ink', .6);
  for (const s of [-1, 1]) {
    paint(ellPts(x + s * e * 1.45, y + e * .25, e * .32, e * .18, 8), { wash: PAL.rose, ink: null });
  }
}

// Strawberry drink cup & warm bread loaf ("我记得你爱喝的饮料 / 也排队买你爱吃的面包")
function drinkProp(x, y, s = 1) {
  boilSeed('drink');
  inkLine([[x + 4 * s, y - 22 * s], [x + 12 * s, y - 42 * s]], 1.4, '#E25B78', 'ink', 0);
  paint([[x - 14 * s, y - 20 * s], [x + 14 * s, y - 20 * s], [x + 10 * s, y + 14 * s], [x - 10 * s, y + 14 * s]], {
    wash: '#F79BB4',
    ink: PAL.ink,
    sw: .8,
  });
  paint(ellPts(x, y - 20 * s, 15 * s, 6 * s, 10), { wash: PAL.cream, ink: PAL.ink, sw: .75 });
}

function breadProp(x, y, s = 1) {
  boilSeed('bread');
  // Paper bakery bag + golden croissant/loaf peeking out
  paint(rectPts(x - 14 * s, y - 10 * s, 28 * s, 26 * s, 1), { wash: '#E6CBA5', ink: PAL.ink, sw: .8 });
  paint(ellPts(x, y - 14 * s, 16 * s, 11 * s, 12), { wash: '#E89A3C', ink: PAL.ink, sw: .85 });
  inkLine([[x - 7 * s, y - 16 * s], [x - 3 * s, y - 10 * s]], .65, PAL.ink, 'inkfine', 0);
  inkLine([[x + 3 * s, y - 16 * s], [x + 7 * s, y - 10 * s]], .65, PAL.ink, 'inkfine', 0);
}

// Pair of golden movie tickets ("你说你有多的电影票")
function ticketsProp(x, y, s = 1, rot = 0) {
  boilSeed('tickets');
  push();
  translate(x, y);
  rotate(rot);
  for (const [dx, dy, ang, col] of [[-6 * s, -4 * s, -.18, '#F5C452'], [6 * s, 4 * s, .12, '#F79BB4']]) {
    push();
    translate(dx, dy);
    rotate(ang);
    paint(rrPts(-22 * s, -11 * s, 44 * s, 22 * s, 3 * s), { wash: col, ink: PAL.ink, sw: .8 });
    inkLine([[-8 * s, -10 * s], [-8 * s, 10 * s]], .6, PAL.ink, 'inkfine', 0);
    pop();
  }
  pop();
}

// Umbrella held over one or two characters
function umbrellaProp(x, y, r, col = '#3B5B8C', tilt = 0) {
  boilSeed('umb');
  push();
  translate(x, y);
  rotate(tilt);
  // Handle shaft
  inkLine([[0, -r * .45], [0, r * .95]], 1.8, PAL.ink, 'ink', 0);
  inkLine([[0, r * .95], [-r * .12, r * 1.02], [-r * .18, r * .92]], 1.8, PAL.ink, 'ink', .5);
  // Canopy dome
  const pts = [];
  for (let i = 0; i <= 14; i++) {
    const a = Math.PI + (i / 14) * Math.PI;
    pts.push([Math.cos(a) * r, Math.sin(a) * r * .52]);
  }
  pts.push([0, -r * .06]);
  paint(pts, { wash: col, ink: PAL.ink, sw: 1.1, curv: .2 });
  pop();
}

// Synchronized Heartbeat Ribbon ("你比他清楚我的心跳 / 我比他清楚你的心跳")
function heartbeatBridge(x1, y1, x2, y2, t, k = 1) {
  if (k <= .02) return;
  boilSeed('hb');
  const midX = (x1 + x2) / 2, midY = (y1 + y2) / 2 - 25;
  glow(midX, midY, 220 * k, '#FF8DA6', .65 * k);
  const pts = [];
  const N = 28;
  for (let i = 0; i <= N; i++) {
    const u = i / N;
    const px = lerp(x1, x2, u);
    let py = lerp(y1, y2, u) - Math.sin(u * Math.PI) * 28;
    // ECG pulse spike travelling along the line
    const beatPos = frac(t * 1.64);
    const d = (u - beatPos) * 9;
    const ecg = Math.exp(-d * d * 2.5) * Math.sin(d * 8) * 42 * k;
    py += ecg;
    pts.push([px, py]);
  }
  inkLine(pts, 2.2 * k, '#FF94B2', 'ink', .2);
  // Two glowing hearts at their chests and one at the center
  const beatScale = 1 + .22 * pulse(t * 2, 5);
  paint(heartPts(x1, y1, 22 * k * beatScale), { wash: '#7CCDF4', ink: PAL.ink, sw: .75 });
  paint(heartPts(x2, y2, 22 * k * beatScale), { wash: '#F79BB4', ink: PAL.ink, sw: .75 });
  paint(heartPts(midX, midY - 12, 32 * k * beatScale), { wash: '#FFE27A', ink: PAL.ink, sw: .9 });
}

// Polaroid Memory Snapshot Frame ("用记忆在对你拍照让人难熬")
function memoryPhoto(x, y, s, rot, kind, t) {
  boilSeed('photo' + kind);
  push();
  translate(x, y);
  rotate(rot);
  // White Polaroid border
  paint(rrPts(-110 * s, -125 * s, 220 * s, 255 * s, 8 * s), { wash: PAL.cream, ink: PAL.ink, sw: 1.1 });
  // Inner sepia canvas
  paint(rectPts(-92 * s, -106 * s, 184 * s, 175 * s), { wash: '#EBD5B3', ink: PAL.ink, sw: .85 });
  if (kind === 1) {
    // Snapshot 1: Sharing the coat in the rain
    umbrellaProp(0, -35 * s, 68 * s, '#8A5E48', -.08);
    paint(heartPts(0, -10 * s, 20 * s), { wash: '#E27A92', ink: PAL.ink, sw: .6 });
  } else if (kind === 2) {
    // Snapshot 2: Strawberry drink & warm bread
    drinkProp(-32 * s, 12 * s, 1.4 * s);
    breadProp(32 * s, 15 * s, 1.4 * s);
  } else {
    // Snapshot 3: Two golden movie tickets & camellia flower
    ticketsProp(0, 0, 1.8 * s, -.1);
  }
  pop();
}

// Giant Celestial Destiny Clock ("你说我爱太晚到 / 为何你爱要晚到 / 对的人不对的时间")
function destinyClock(cx, cy, r, t, k = 1) {
  if (k <= .02) return;
  boilSeed('clock');
  glow(cx, cy, r * 1.45, '#FFD87A', .42 * k);
  paint(ellPts(cx, cy, r, r, 32, 2), { wash: '#1E2448', washOp: Math.round(185 * k), ink: '#F4C258', sw: 1.6 * k });
  paint(ellPts(cx, cy, r * .86, r * .86, 28, 1.5), { ink: '#F4C258', sw: .85 * k });
  for (let i = 0; i < 12; i++) {
    const a = (i / 12) * TAU - Math.PI / 2;
    inkLine([
      [cx + Math.cos(a) * r * .76, cy + Math.sin(a) * r * .76],
      [cx + Math.cos(a) * r * .88, cy + Math.sin(a) * r * .88],
    ], 1.4 * k, '#FFE28A', 'ink', 0);
  }
  // Clock hands turning
  const aMin = -t * .45, aHr = -t * .08 + 2.1;
  inkLine([[cx, cy], [cx + Math.cos(aMin) * r * .68, cy + Math.sin(aMin) * r * .68]], 2.2 * k, '#7CCDF4', 'ink', 0);
  inkLine([[cx, cy], [cx + Math.cos(aHr) * r * .46, cy + Math.sin(aHr) * r * .46]], 2.8 * k, '#F79BB4', 'ink', 0);
  paint(ellPts(cx, cy, 8 * k, 8 * k, 10), { wash: '#FFE28A', ink: PAL.ink, sw: .8 });
}

// Autumn Ginkgo Leaves / Mist Wipe Transition
function ginkgoWipe(p, col1 = '#E8AA38', col2 = '#D97757') {
  if (p <= 0 || p >= 1) return;
  const cover = p < .5 ? easeOut(p * 2) : 1 - ease((p - .5) * 2);
  for (let i = 0; i < 14; i++) {
    boilSeed('gw' + i);
    const x = (i % 5) * 460 - 20 + 60 * hash(i) + (p - .5) * 320;
    const y = Math.floor(i / 5) * 410 + 50 * hash(i + 7);
    const r = (130 + 360 * cover) * (.8 + .35 * hash(i + 3));
    if (r < 12) continue;
    paint(lumpy(x, y, r, r * .82, i * 2.1), {
      wash: i % 2 === 0 ? col1 : col2,
      washOp: Math.round(255 * clamp(cover * 1.5)),
      ink: cover < .8 ? PAL.ink : null,
      sw: .85,
    });
  }
}

// Darkness spotlight outside a circle
function spotDark(cx, cy, r, a, col = '#14162B') {
  if (a <= .01) return;
  const pts = ellPts(cx, cy, r, r * .92, 32), far = 3600;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i], q = pts[(i + 1) % pts.length];
    const o = pp => {
      const dx = pp[0] - cx, dy = pp[1] - cy, d = Math.hypot(dx, dy) || 1;
      return [cx + (dx / d) * far, cy + (dy / d) * far];
    };
    const e = [(q[0] - p[0]) * .08, (q[1] - p[1]) * .08];
    const p2 = [p[0] - e[0], p[1] - e[1]], q2 = [q[0] + e[0], q[1] + e[1]];
    paint([p2, q2, o(q2), o(p2)], { wash: col, washOp: Math.round(255 * a), ink: null });
  }
}
