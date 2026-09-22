/**
 * Original maze layout for Fly Maze. Legend:
 *   '#' wall, '.' pellet, 'o' power pellet, 'P' player start, ' ' ghost start (inside
 *   the house), 'G' ghost-only door. Every open cell is reachable from the player start.
 */
export const LAYOUT: readonly string[] = [
  "###################",
  "#........#........#",
  "#.##.###.#.###.##.#",
  "#o...............o#",
  "#.##.#.#####.#.##.#",
  "#....#...#...#....#",
  "####.###.#.###.####",
  "####.#.......#.####",
  "####.#.##G##.#.####",
  "#o.....#   #.....o#",
  "####.#.#####.#.####",
  "####.#.......#.####",
  "####.#.#####.#.####",
  "#........#........#",
  "#.##.###.#.###.##.#",
  "#..#.....P.....#..#",
  "##.#.#.#####.#.#.##",
  "#o...#...#...#...o#",
  "#.######.#.######.#",
  "#.................#",
  "###################",
];

export const UP = 0;
export const RIGHT = 1;
export const DOWN = 2;
export const LEFT = 3;
export const DIRS: ReadonlyArray<readonly [number, number]> = [
  [0, -1],
  [1, 0],
  [0, 1],
  [-1, 0],
];
export const DIR_NAMES = ["up", "right", "down", "left"] as const;
export type DirName = (typeof DIR_NAMES)[number];

export interface Maze {
  readonly width: number;
  readonly height: number;
  readonly wall: Uint8Array;
  readonly door: Uint8Array;
  readonly pellet: Uint8Array;
  readonly power: Uint8Array;
  readonly pelletCount: number;
  readonly start: readonly [number, number];
  readonly ghostStarts: ReadonlyArray<readonly [number, number]>;
}

export function parseMaze(rows: readonly string[] = LAYOUT): Maze {
  const height = rows.length;
  const width = rows[0].length;
  for (const row of rows) {
    if (row.length !== width) throw new Error(`ragged maze row: "${row}"`);
  }
  const wall = new Uint8Array(width * height);
  const door = new Uint8Array(width * height);
  const pellet = new Uint8Array(width * height);
  const power = new Uint8Array(width * height);
  const ghostStarts: Array<readonly [number, number]> = [];
  let start: readonly [number, number] | null = null;
  let pelletCount = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      switch (rows[y][x]) {
        case "#":
          wall[i] = 1;
          break;
        case ".":
          pellet[i] = 1;
          pelletCount++;
          break;
        case "o":
          pellet[i] = 1;
          power[i] = 1;
          pelletCount++;
          break;
        case "P":
          start = [x, y];
          break;
        case " ":
          ghostStarts.push([x, y]);
          break;
        case "G":
          door[i] = 1;
          break;
        default:
          throw new Error(`unknown maze glyph "${rows[y][x]}" at ${x},${y}`);
      }
    }
  }
  if (!start) throw new Error("maze has no player start");
  if (ghostStarts.length === 0) throw new Error("maze has no ghost start");
  return { width, height, wall, door, pellet, power, pelletCount, start, ghostStarts };
}

export function cellIndex(m: Maze, x: number, y: number): number {
  return y * m.width + x;
}

/** Whether a cell can be entered. Ghosts may pass the house door; the player may not. */
export function isOpen(m: Maze, x: number, y: number, ghost = false): boolean {
  if (x < 0 || y < 0 || x >= m.width || y >= m.height) return false;
  const i = y * m.width + x;
  if (m.wall[i]) return false;
  if (m.door[i] && !ghost) return false;
  return true;
}

/** Multi-source breadth-first distances in steps; -1 marks unreachable cells. */
export function bfsDistances(m: Maze, sources: Iterable<number>, ghost = false): Int32Array {
  const dist = new Int32Array(m.width * m.height).fill(-1);
  const queue: number[] = [];
  for (const s of sources) {
    if (dist[s] === -1) {
      dist[s] = 0;
      queue.push(s);
    }
  }
  for (let head = 0; head < queue.length; head++) {
    const i = queue[head];
    const x = i % m.width;
    const y = (i - x) / m.width;
    for (const [dx, dy] of DIRS) {
      const nx = x + dx;
      const ny = y + dy;
      if (!isOpen(m, nx, ny, ghost)) continue;
      const j = ny * m.width + nx;
      if (dist[j] === -1) {
        dist[j] = dist[i] + 1;
        queue.push(j);
      }
    }
  }
  return dist;
}
