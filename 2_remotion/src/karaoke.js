// karaoke.js: brush-font karaoke with a character-by-character fill, drawn on the 2D compositor through
// window.overlayHook in core.js (crisp, above the paper grain). The bar covers y ≈ 975–1070 while a line shows.
//
// LY = [[start, end, "line", opts?]].
// opts.sing = seconds the line takes to sing (for the fill)
// opts.hold = seconds the line stays up after it ends, lifted above the next line (used for Jay & Cindy call-and-response!)
// opts.who  = 'm' (Jay, warm amber-sky tint), 'f' (Cindy, rose-gold tint), 'b' (Duet, bright gold)
const LY = [
  // Intro credits
  [1.20, 5.10, '傻笑 · 周杰伦 × 袁咏琳', { sing: 2.8, who: 'b' }],
  [5.29, 11.20, '词：方文山   曲：周杰伦   编曲：黄雨勋', { sing: 3.8, who: 'b' }],

  // Verse 1A (Jay)
  [19.69, 23.82, '眼看着手机里 没讯号', { sing: 3.1, who: 'm' }],
  [23.90, 28.65, '担心你的回信我 收不到', { sing: 3.4, who: 'm' }],
  [28.75, 33.89, '哪怕只传来一个 微笑的符号', { sing: 4.2, who: 'm' }],
  [33.97, 38.20, '我都会舍不得 删掉', { sing: 3.3, who: 'm' }],

  // Verse 1B (Cindy & Duet)
  [38.44, 43.35, '电台传来天气 的预报', { sing: 3.6, who: 'f' }],
  [43.44, 48.10, '想起你为我遮雨的外套', { sing: 3.6, who: 'f' }],
  [48.20, 53.65, '客厅的电视播放偶像剧频道', { sing: 4.4, who: 'b' }],
  [53.75, 58.80, '羡慕他们为爱 在争吵', { sing: 3.6, who: 'b' }],

  // Chorus 1 (Duet Call & Response)
  [60.75, 65.16, '你在我的面前微笑装不知道', { sing: 3.9, who: 'm' }],
  [65.24, 70.06, '其实你告白的简讯我有收到', { sing: 4.1, who: 'f' }],
  [70.14, 71.92, '你说我爱太晚到', { sing: 1.6, hold: 3.1, who: 'm' }],
  [71.96, 75.10, '为何你爱要晚到', { sing: 2.6, who: 'f' }],
  [75.18, 79.90, '只能像朋友般拥抱 Yeah', { sing: 3.8, who: 'b' }],
  [80.00, 84.75, '你说他的用心付出比我早到', { sing: 4.1, who: 'm' }],
  [84.83, 89.86, '对你的温柔只能放心里收好', { sing: 4.2, who: 'f' }],
  [89.94, 91.68, '你说我爱太晚到', { sing: 1.6, hold: 3.0, who: 'm' }],
  [91.73, 94.78, '为何你爱要晚到', { sing: 2.5, who: 'f' }],
  [94.86, 101.47, '怎么做 我们彼此才不会伤到', { sing: 5.2, who: 'b' }],

  // Verse 2A (Bakery & Street)
  [119.64, 122.95, '我记得你爱喝的饮料', { sing: 2.9, hold: 1.25, who: 'm' }],
  [123.00, 124.22, '（的饮料）', { sing: 1.0, who: 'f' }],
  [124.27, 127.80, '也排队买你爱吃的面包', { sing: 3.1, hold: 1.2, who: 'm' }],
  [127.87, 128.98, '（的面包）', { sing: 0.9, who: 'f' }],
  [129.04, 131.30, '却只能对你远远', { sing: 2.0, hold: 2.8, who: 'm' }],
  [131.37, 134.24, '的会心一笑', { sing: 2.4, who: 'b' }],
  [134.32, 138.60, '连你的手都牵 不到', { sing: 3.4, who: 'b' }],

  // Verse 2B (Cinema Tickets & Heartbeat)
  [138.90, 143.55, '你说你有多的电影票', { sing: 3.6, who: 'f' }],
  [143.67, 146.10, '你知道 事情没有', { sing: 2.1, hold: 2.1, who: 'm' }],
  [146.17, 148.60, '那么巧', { sing: 1.9, who: 'b' }],
  [148.82, 151.52, '但很窝心的是你怕我无聊', { sing: 2.4, hold: 2.4, who: 'f' }],
  [151.58, 154.05, '但很窝心的是我怕你无聊', { sing: 2.2, who: 'm' }],
  [154.11, 156.15, '你比他清楚我的心跳', { sing: 1.8, hold: 2.6, who: 'f' }],
  [156.20, 159.50, '我比他清楚你的心跳', { sing: 2.6, who: 'm' }],

  // Chorus 2
  [161.42, 165.83, '你在我的面前微笑装不知道', { sing: 3.9, who: 'm' }],
  [165.91, 170.51, '其实你告白的简讯我有收到', { sing: 4.1, who: 'f' }],
  [170.59, 172.65, '你说我 爱太晚到', { sing: 1.8, hold: 2.8, who: 'm' }],
  [172.71, 175.50, '为何你 爱要晚到', { sing: 2.4, who: 'f' }],
  [175.58, 180.17, '只能像朋友般拥抱 Yeah', { sing: 3.8, who: 'b' }],
  [180.25, 185.04, '你说他的用心付出比我早到', { sing: 4.1, who: 'm' }],
  [185.12, 190.00, '对你的温柔只能放心里收好', { sing: 4.2, who: 'f' }],
  [190.07, 192.05, '你说我爱太晚到', { sing: 1.7, hold: 2.9, who: 'm' }],
  [192.12, 194.98, '为何你爱要晚到', { sing: 2.4, who: 'f' }],
  [195.06, 201.88, '怎么做 我们彼此才不会伤到', { sing: 5.2, who: 'b' }],

  // Final Chorus Climax
  [220.18, 224.34, '你在我的面前微笑装不知道', { sing: 3.7, who: 'm' }],
  [224.42, 229.28, '他没有不好只是话越聊越少', { sing: 4.2, who: 'f' }],
  [229.36, 231.10, '你说我爱太晚到', { sing: 1.6, hold: 3.0, who: 'm' }],
  [231.16, 234.18, '为何你爱要晚到', { sing: 2.5, who: 'f' }],
  [234.26, 239.06, '错过了幸福的味道 Yeah', { sing: 4.0, who: 'b' }],
  [239.14, 243.81, '用记忆在对你拍照让人难熬', { sing: 4.1, who: 'm' }],
  [243.89, 250.35, '其实我对于你的好也曾动摇', { sing: 4.8, who: 'f' }],
  [250.43, 252.76, '你说我爱太晚到', { sing: 1.9, hold: 2.2, who: 'm' }],
  [252.83, 254.96, '为何你爱要晚到', { sing: 1.9, who: 'f' }],
  [255.03, 263.50, '对的人 不对的时间 却放不掉', { sing: 6.2, who: 'b' }],
];

const KFONT = '"STXingkai", "STKaiti", "KaiTi", "Kaiti SC", serif';
window.EXTRA_FONTS = [];

window.overlayHook = (c, t) => {
  const i = LY.findIndex(l => t >= l[0] && t < l[1]);
  const P = LY.find(l => l[3] && l[3].hold && t >= l[1] && t < l[1] + l[3].hold);
  if (P) {
    const a = t - P[1], k = ease(a / .15);
    kLine(c, P, t, 1020 - 102 * k, 1 - .26 * k, 1 - ease((a - (P[3].hold - .3)) / .3), true);
  }
  if (i >= 0) kLine(c, LY[i], t, 1020, 1, 1, false);
};

function kLine(c, L, t, y, sc, alpha, held) {
  const [a, b, txt, opt = {}] = L;
  const grow = held ? 1 : easeOut((t - a) / .18) * (opt.hold ? 1 : 1 - ease((t - (b - .12)) / .12));
  if (grow < .02 || alpha <= .01) return;
  c.save();
  c.globalAlpha = alpha;
  c.translate(960, y);
  c.scale(sc, sc);
  c.translate(-960, -y);
  c.font = `58px ${KFONT}`;
  c.textBaseline = 'middle';
  c.textAlign = 'left';
  const total = c.measureText(txt).width;

  // Ragged hand-painted ink lozenge bar that boils with BOILN
  const w = (total + 130) * grow, x0 = 960 - w / 2, y0 = y - 46;
  const r = n => hash(BOILN * 13 + n) * 7 - 3.5;
  c.globalAlpha = .84 * alpha;
  c.fillStyle = PAL.ink;
  c.beginPath();
  c.moveTo(x0 + r(1), y0 + r(2));
  c.lineTo(x0 + w / 2, y0 - 5 + r(3));
  c.lineTo(x0 + w + r(4), y0 + r(5));
  c.lineTo(x0 + w + 16, y0 + 44);
  c.lineTo(x0 + w + r(6), y0 + 88 + r(7));
  c.lineTo(x0 + w / 2, y0 + 92 + r(8));
  c.lineTo(x0 + r(9), y0 + 88 + r(10));
  c.lineTo(x0 - 16, y0 + 44);
  c.closePath();
  c.fill();

  c.globalAlpha = alpha;
  if (grow > .82) {
    const n = [...txt].length;
    const singDur = opt.sing ?? Math.min((b - a) * .85, n * .3);
    const f = held ? 1 : clamp((t - a) / singDur);
    const x = 960 - total / 2;
    c.fillStyle = PAL.cream;
    c.fillText(txt, x, y);
    c.save();
    c.beginPath();
    c.rect(x - 2, y - 50, total * f + 4, 100);
    c.clip();
    // Color-coded fill: Jay = warm sky-gold, Cindy = rose-coral, Duet = rich ochre-gold
    c.fillStyle = opt.who === 'm' ? '#7CCDF4' : opt.who === 'f' ? '#F79BB4' : '#F4B844';
    c.fillText(txt, x, y);
    c.restore();
  }
  c.restore();
}
