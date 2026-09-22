import { generateMaze } from "./generate";
import { DIRS, LEFT, UP, type Maze, bfsDistances, cellIndex, isOpen, parseMaze } from "./maze";
import { Rng } from "./rng";
import { DEFAULT_RULES, type GameRules } from "./rules";

export const PELLET_SCORE = 10;
export const POWER_SCORE = 50;
export const CLEAR_BONUS = 1000;

export interface Ghost {
  x: number;
  y: number;
  dir: number;
  readonly startX: number;
  readonly startY: number;
  readonly chase: number;
  release: number;
  /** Fleeing and edible until the frightened timer ends or the ghost is eaten. */
  frightened: boolean;
  /** Ghost-walkable distance from every cell to this ghost's scatter corner. */
  readonly cornerDist: Int32Array;
}

export interface GameState {
  readonly maze: Maze;
  readonly rules: GameRules;
  readonly seed: number;
  px: number;
  py: number;
  pdir: number;
  readonly ghosts: Ghost[];
  readonly pellet: Uint8Array;
  remaining: number;
  score: number;
  steps: number;
  alive: boolean;
  cleared: boolean;
  done: boolean;
  /** Remaining ticks during which ghosts flee and can be eaten. */
  frightened: number;
  ghostsEaten: number;
  /** Player-walkable distance from each cell to the nearest remaining pellet. */
  pelletDist: Int32Array;
  /** Ghost-walkable distance from each cell to the nearest hunting ghost. */
  ghostDist: Int32Array;
  /** Ghost-walkable distance from each cell to the nearest frightened ghost; all -1 when none. */
  preyDist: Int32Array;
  readonly rng: Rng;
}

let defaultMaze: Maze | null = null;
export function getDefaultMaze(): Maze {
  return (defaultMaze ??= parseMaze());
}

export function isScatterPhase(steps: number, rules: GameRules): boolean {
  return steps % (rules.scatterTicks + rules.chaseTicks) < rules.scatterTicks;
}

function scatterCorners(maze: Maze): number[] {
  const corners: Array<[number, number]> = [
    [1, 1],
    [maze.width - 2, 1],
    [1, maze.height - 2],
    [maze.width - 2, maze.height - 2],
  ];
  return corners.map(([x, y]) => cellIndex(maze, x, y));
}

/** Open cells the player may start on: reachable, outside the ghost house, and not too close to it. */
function startCandidates(maze: Maze, minGhostDistance: number): number[] {
  const fromStart = bfsDistances(maze, [cellIndex(maze, maze.start[0], maze.start[1])], false);
  const fromGhosts = bfsDistances(
    maze,
    maze.ghostStarts.map(([x, y]) => cellIndex(maze, x, y)),
    true,
  );
  const cells: number[] = [];
  for (let i = 0; i < fromStart.length; i++) {
    if (fromStart[i] >= 0 && fromGhosts[i] >= minGhostDistance) cells.push(i);
  }
  return cells;
}

export function createGame(seed: number, rules: GameRules = DEFAULT_RULES, mazeArg?: Maze): GameState {
  const maze = mazeArg ?? (rules.randomMaze ? generateMaze(seed) : getDefaultMaze());
  const corners = scatterCorners(maze);
  const setup = new Rng((seed ^ 0x5bf03635) >>> 0 || 1);
  let startX = maze.start[0];
  let startY = maze.start[1];
  if (rules.randomStart) {
    const candidates = startCandidates(maze, rules.startGhostDistance);
    const cell = candidates[setup.int(candidates.length)];
    startX = cell % maze.width;
    startY = (cell - startX) / maze.width;
  }
  const ghosts: Ghost[] = maze.ghostStarts.map(([x, y], i) => ({
    x,
    y,
    dir: UP,
    startX: x,
    startY: y,
    chase: rules.ghostChase[i % rules.ghostChase.length],
    release: rules.ghostRelease[i % rules.ghostRelease.length] + (rules.releaseJitter > 0 ? setup.int(rules.releaseJitter + 1) : 0),
    frightened: false,
    cornerDist: bfsDistances(maze, [corners[i % corners.length]], true),
  }));
  const state: GameState = {
    maze,
    rules,
    seed,
    px: startX,
    py: startY,
    pdir: LEFT,
    ghosts,
    pellet: maze.pellet.slice(),
    remaining: maze.pelletCount,
    score: 0,
    steps: 0,
    alive: true,
    cleared: false,
    done: false,
    frightened: 0,
    ghostsEaten: 0,
    pelletDist: new Int32Array(0),
    ghostDist: new Int32Array(0),
    preyDist: new Int32Array(0),
    rng: new Rng(seed),
  };
  refreshPelletDist(state);
  refreshGhostDist(state);
  return state;
}

function refreshPelletDist(s: GameState): void {
  const sources: number[] = [];
  for (let i = 0; i < s.pellet.length; i++) if (s.pellet[i]) sources.push(i);
  s.pelletDist = bfsDistances(s.maze, sources, false);
}

function refreshGhostDist(s: GameState): void {
  const hunting = s.ghosts.filter((g) => !g.frightened).map((g) => cellIndex(s.maze, g.x, g.y));
  const fleeing = s.ghosts.filter((g) => g.frightened).map((g) => cellIndex(s.maze, g.x, g.y));
  s.ghostDist = bfsDistances(s.maze, hunting, true);
  s.preyDist = bfsDistances(s.maze, fleeing, true);
}

function movePlayer(s: GameState, action: number): void {
  const m = s.maze;
  if (action >= 0 && action < 4) {
    const [dx, dy] = DIRS[action];
    if (isOpen(m, s.px + dx, s.py + dy)) {
      s.pdir = action;
      s.px += dx;
      s.py += dy;
      return;
    }
  }
  const [cx, cy] = DIRS[s.pdir];
  if (isOpen(m, s.px + cx, s.py + cy)) {
    s.px += cx;
    s.py += cy;
  }
}

function ghostMovesThisTick(s: GameState, g: Ghost): boolean {
  if (s.steps < g.release) return false;
  if (g.frightened) return s.steps % 2 === 0;
  const rest = s.rules.ghostRestEvery;
  return rest <= 0 || s.steps % rest !== rest - 1;
}

function moveGhost(s: GameState, g: Ghost, playerDist: Int32Array): void {
  const m = s.maze;
  const fleeing = g.frightened;
  const target = fleeing || !isScatterPhase(s.steps, s.rules) ? playerDist : g.cornerDist;
  const atTarget = !fleeing && target[cellIndex(m, g.x, g.y)] === 0;
  const reverse = (g.dir + 2) % 4;
  const options: number[] = [];
  for (let d = 0; d < 4; d++) {
    if (d === reverse) continue;
    const [dx, dy] = DIRS[d];
    if (isOpen(m, g.x + dx, g.y + dy, true)) options.push(d);
  }
  if (options.length === 0) {
    const [dx, dy] = DIRS[reverse];
    if (isOpen(m, g.x + dx, g.y + dy, true)) options.push(reverse);
    else return;
  }
  let choice: number;
  if (!atTarget && s.rng.next() < g.chase) {
    choice = options[0];
    let best = fleeing ? -Infinity : Infinity;
    for (const d of options) {
      const [dx, dy] = DIRS[d];
      const dist = target[cellIndex(m, g.x + dx, g.y + dy)];
      const value = dist < 0 ? (fleeing ? -Infinity : Infinity) : dist;
      if (fleeing ? value > best : value < best) {
        best = value;
        choice = d;
      }
    }
  } else {
    choice = options[s.rng.int(options.length)];
  }
  g.dir = choice;
  g.x += DIRS[choice][0];
  g.y += DIRS[choice][1];
}

/** Player and ghost occupy the same cell: the player eats a frightened ghost, or is caught. */
function resolveContact(s: GameState, g: Ghost): void {
  if (!s.alive) return;
  if (g.frightened) {
    s.score += s.rules.ghostEatScore;
    s.ghostsEaten++;
    sendHome(s, g);
  } else {
    s.alive = false;
  }
}

function sendHome(s: GameState, g: Ghost): void {
  g.x = g.startX;
  g.y = g.startY;
  g.dir = UP;
  g.frightened = false;
  g.release = s.steps + s.rules.respawnTicks;
}

/** Advance one tick. `action` is a direction 0..3, or -1 to keep the current heading. */
export function stepGame(s: GameState, action: number): GameState {
  if (s.done) return s;
  const m = s.maze;
  const prevPx = s.px;
  const prevPy = s.py;
  movePlayer(s, action);

  const here = cellIndex(m, s.px, s.py);
  if (s.pellet[here]) {
    s.pellet[here] = 0;
    s.remaining--;
    if (m.power[here] && s.rules.powerPellets) {
      s.score += POWER_SCORE;
      s.frightened = s.rules.frightenedTicks;
      for (const g of s.ghosts) if (s.steps >= g.release) g.frightened = true;
    } else {
      s.score += PELLET_SCORE;
    }
    refreshPelletDist(s);
  }

  // Contact is resolved twice per tick: once where the player lands, and once after each
  // ghost moves. Without the first pass a player stepping onto a ghost's cell would pass
  // through whenever that ghost left for a third cell in the same tick.
  for (const g of s.ghosts) {
    if (s.steps >= g.release && g.x === s.px && g.y === s.py) resolveContact(s, g);
  }
  const playerDist = bfsDistances(m, [here], true);
  for (const g of s.ghosts) {
    const gx0 = g.x;
    const gy0 = g.y;
    if (ghostMovesThisTick(s, g)) moveGhost(s, g, playerDist);
    const onPlayer = g.x === s.px && g.y === s.py;
    const swapped = g.x === prevPx && g.y === prevPy && gx0 === s.px && gy0 === s.py;
    if (onPlayer || swapped) resolveContact(s, g);
  }
  if (s.frightened > 0 && --s.frightened === 0) for (const g of s.ghosts) g.frightened = false;

  s.steps++;
  if (!s.alive) {
    s.done = true;
  } else if (s.remaining === 0) {
    s.cleared = true;
    s.score += CLEAR_BONUS;
    s.done = true;
  } else if (s.steps >= s.rules.maxSteps) {
    s.done = true;
  }
  refreshGhostDist(s);
  return s;
}
