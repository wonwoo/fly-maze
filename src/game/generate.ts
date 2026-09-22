import { DIRS, type Maze, parseMaze } from "./maze";
import { Rng } from "./rng";

/**
 * Generates a seeded maze with the same footprint and conventions as the built-in one:
 * a 19 x 21 grid, an outer wall, a central ghost house behind a one-tile door, pellets on
 * every open cell, and six power pellets. Corridors come from a randomised depth-first
 * carve on the odd lattice, then a few extra openings so the maze has loops rather than a
 * single tree, which would trap any controller that cannot back out.
 */
export function generateMaze(seed: number, width = 19, height = 21): Maze {
  const rng = new Rng((seed ^ 0x27d4eb2f) >>> 0 || 1);
  const open: boolean[][] = Array.from({ length: height }, () => new Array<boolean>(width).fill(false));

  const carve = (x: number, y: number): void => {
    open[y][x] = true;
    const order = [0, 1, 2, 3];
    for (let i = order.length - 1; i > 0; i--) {
      const j = rng.int(i + 1);
      [order[i], order[j]] = [order[j], order[i]];
    }
    for (const d of order) {
      const nx = x + DIRS[d][0] * 2;
      const ny = y + DIRS[d][1] * 2;
      if (nx <= 0 || ny <= 0 || nx >= width - 1 || ny >= height - 1 || open[ny][nx]) continue;
      open[y + DIRS[d][1]][x + DIRS[d][0]] = true;
      carve(nx, ny);
    }
  };
  carve(1, 1);

  // A carved tree is all dead ends, which is unfair in a chase: every pocket is a trap.
  // Open walls until the maze is mostly loops, then remove whatever dead ends remain.
  addLoops(open, width, height, rng);

  connect(open, width, height, rng);

  // Ghost house: a 3 x 2 chamber at the centre with a door on top.
  const houseX = Math.floor(width / 2) - 1;
  const houseY = Math.floor(height / 2) - 1;
  // Three closed rows: door row, chamber row, and a floor so the player cannot enter from below.
  for (let y = houseY; y < houseY + 3; y++) {
    for (let x = houseX - 1; x <= houseX + 3; x++) {
      if (x >= 0 && x < width && y >= 0 && y < height) open[y][x] = false;
    }
  }
  // Carving the house cuts corridors, so re-loop and then seal whatever dead ends remain.
  const houseBlock = { x: houseX - 1, y: houseY, w: 5, h: 3 };
  connect(open, width, height, rng, houseBlock);
  addLoops(open, width, height, rng, 0.2, houseBlock);
  removeDeadEnds(open, width, height, [[houseX + 1, houseY - 1]], houseBlock);
  connect(open, width, height, rng, houseBlock);

  const rows: string[][] = open.map((row) => row.map((cell) => (cell ? "." : "#")));
  rows[houseY][houseX + 1] = "G";
  for (let x = houseX; x <= houseX + 2; x++) rows[houseY + 1][x] = " ";

  // Make sure the chamber is reachable from outside by opening the cell above the door.
  if (houseY - 1 > 0) rows[houseY - 1][houseX + 1] = ".";

  const openCells: Array<[number, number]> = [];
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) if (rows[y][x] === ".") openCells.push([x, y]);
  }
  if (openCells.length < 60) return parseMaze();
  const [sx, sy] = openCells[rng.int(openCells.length)];
  rows[sy][sx] = "P";

  // Six power pellets, spread by picking the open cell farthest from those already chosen.
  const chosen: Array<[number, number]> = [];
  const candidates = openCells.filter(([x, y]) => rows[y][x] === ".");
  for (let n = 0; n < 6 && candidates.length > 0; n++) {
    let best = 0;
    let bestScore = -1;
    candidates.forEach(([x, y], i) => {
      const score = chosen.length === 0 ? (x - width / 2) ** 2 + (y - height / 2) ** 2 : Math.min(...chosen.map(([cx, cy]) => (x - cx) ** 2 + (y - cy) ** 2));
      if (score > bestScore) {
        bestScore = score;
        best = i;
      }
    });
    const [px, py] = candidates.splice(best, 1)[0];
    rows[py][px] = "o";
    chosen.push([px, py]);
  }
  return parseMaze(rows.map((row) => row.join("")));
}


/**
 * Opens wall cells until every open cell is reachable from every other. Carving the ghost
 * house can cut corridors in two, and an unreachable pocket of pellets makes a course
 * impossible to clear. Cells inside `keepClosed` are never opened.
 */
function connect(
  open: boolean[][],
  width: number,
  height: number,
  rng: Rng,
  keepClosed?: { x: number; y: number; w: number; h: number },
): void {
  const label = (): number[][] => {
    const ids = Array.from({ length: height }, () => new Array<number>(width).fill(-1));
    let next = 0;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (!open[y][x] || ids[y][x] >= 0) continue;
        const id = next++;
        const stack: Array<[number, number]> = [[x, y]];
        ids[y][x] = id;
        while (stack.length) {
          const [cx, cy] = stack.pop()!;
          for (const [dx, dy] of DIRS) {
            const nx = cx + dx;
            const ny = cy + dy;
            if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
            if (!open[ny][nx] || ids[ny][nx] >= 0) continue;
            ids[ny][nx] = id;
            stack.push([nx, ny]);
          }
        }
      }
    }
    return ids;
  };
  const reserved = (x: number, y: number): boolean =>
    !!keepClosed && x >= keepClosed.x && x < keepClosed.x + keepClosed.w && y >= keepClosed.y && y < keepClosed.y + keepClosed.h;

  for (let pass = 0; pass < 40; pass++) {
    const ids = label();
    const sizes = new Map<number, number>();
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) if (ids[y][x] >= 0) sizes.set(ids[y][x], (sizes.get(ids[y][x]) ?? 0) + 1);
    }
    if (sizes.size <= 1) return;
    const main = [...sizes.entries()].sort((a, b) => b[1] - a[1])[0][0];
    const joins: Array<[number, number]> = [];
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        if (open[y][x] || reserved(x, y)) continue;
        const neighbours = new Set<number>();
        for (const [dx, dy] of DIRS) {
          const id = ids[y + dy][x + dx];
          if (id >= 0) neighbours.add(id);
        }
        if (neighbours.size > 1 && neighbours.has(main)) joins.push([x, y]);
      }
    }
    if (joins.length === 0) {
      // Nothing bridges directly; drop the stranded pockets so no pellet is unreachable.
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) if (ids[y][x] >= 0 && ids[y][x] !== main) open[y][x] = false;
      }
      return;
    }
    const [jx, jy] = joins[rng.int(joins.length)];
    open[jy][jx] = true;
  }
}


/** Counts a cell's open orthogonal neighbours. */
function degree(open: boolean[][], width: number, height: number, x: number, y: number): number {
  let n = 0;
  for (const [dx, dy] of DIRS) {
    const nx = x + dx;
    const ny = y + dy;
    if (nx >= 0 && ny >= 0 && nx < width && ny < height && open[ny][nx]) n++;
  }
  return n;
}

/**
 * Opens wall cells that join two existing corridors, turning the carved tree into a
 * looped maze. `target` is the share of open cells that should sit at a junction, which
 * is what makes a chase survivable: with loops, every corridor has a way out.
 */
function addLoops(open: boolean[][], width: number, height: number, rng: Rng, target = 0.2, keepClosed?: { x: number; y: number; w: number; h: number }): void {
  const openCount = (): number => {
    let n = 0;
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) if (open[y][x]) n++;
    return n;
  };
  const junctions = (): number => {
    let n = 0;
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) if (open[y][x] && degree(open, width, height, x, y) >= 3) n++;
    }
    return n;
  };
  for (let pass = 0; pass < 400; pass++) {
    if (junctions() >= target * openCount()) return;
    const candidates: Array<[number, number]> = [];
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        if (open[y][x]) continue;
        if (keepClosed && x >= keepClosed.x && x < keepClosed.x + keepClosed.w && y >= keepClosed.y && y < keepClosed.y + keepClosed.h) continue;
        // Opening a cell with two or more open neighbours creates a loop, not a spur.
        if (degree(open, width, height, x, y) >= 2) candidates.push([x, y]);
      }
    }
    if (candidates.length === 0) return;
    const [x, y] = candidates[rng.int(candidates.length)];
    open[y][x] = true;
  }
}

/**
 * Removes dead ends by extending them into the rest of the maze rather than walling them
 * off, so the corridor count stays high. A dead end is opened outward through one wall
 * cell until it meets another corridor; only if no such extension exists is it sealed.
 */
function removeDeadEnds(
  open: boolean[][],
  width: number,
  height: number,
  keep: Array<[number, number]> = [],
  keepClosed?: { x: number; y: number; w: number; h: number },
): void {
  const kept = new Set(keep.map(([x, y]) => `${x},${y}`));
  const reserved = (x: number, y: number): boolean =>
    !!keepClosed && x >= keepClosed.x && x < keepClosed.x + keepClosed.w && y >= keepClosed.y && y < keepClosed.y + keepClosed.h;

  for (let pass = 0; pass < 200; pass++) {
    let changed = false;
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        if (!open[y][x] || kept.has(`${x},${y}`)) continue;
        if (degree(open, width, height, x, y) > 1) continue;
        // Prefer to dig on: open a neighbouring wall whose far side reaches another corridor.
        let dug = false;
        for (const [dx, dy] of DIRS) {
          const wx = x + dx;
          const wy = y + dy;
          if (wx <= 0 || wy <= 0 || wx >= width - 1 || wy >= height - 1) continue;
          if (open[wy][wx] || reserved(wx, wy)) continue;
          const fx = wx + dx;
          const fy = wy + dy;
          const reachesCorridor =
            fx > 0 && fy > 0 && fx < width - 1 && fy < height - 1 && (open[fy][fx] || degree(open, width, height, wx, wy) >= 2);
          if (reachesCorridor) {
            open[wy][wx] = true;
            dug = true;
            changed = true;
            break;
          }
        }
        if (!dug) {
          open[y][x] = false;
          changed = true;
        }
      }
    }
    if (!changed) return;
  }
}

