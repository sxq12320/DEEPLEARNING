export const C = {
  bg0: '#141832',
  bg1: '#232a52',
  bgDeep: '#0e1126',
  leaf: '#6fdc8c',
  leafDark: '#3aa56d',
  stem: '#57c98b',
  stemDark: '#2e7d5b',
  petal: '#ff8fa3',
  amber: '#ffc94d',
  coral: '#ff6b6b',
  sky: '#5cc8ff',
  water: '#4fc3f7',
  text: '#f5f7ff',
  muted: '#aab3d6',
  cell: '#63e2c6',
  cellDark: '#37b59c',
  skin: '#ffcf9e',
  skinDark: '#e8a76f',
  night0: '#12152e',
  night1: '#1c2247',
  dusk0: '#3a2a5e',
  dusk1: '#e0714f',
};

export const FONT_HEAD = '"YouYuan","Microsoft YaHei",sans-serif';
export const FONT_BODY = '"Microsoft YaHei","PingFang SC",sans-serif';

export const rand = (i: number, salt = 0) => {
  const x = Math.sin(i * 127.1 + salt * 311.7) * 43758.5453;
  return x - Math.floor(x);
};

export const clamp01 = (v: number) => Math.max(0, Math.min(1, v));
