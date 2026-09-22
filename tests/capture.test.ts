import { describe, expect, it } from "vitest";
import { DIRS, cellIndex, isOpen } from "../src/game/maze";
import { DEFAULT_RULES } from "../src/game/rules";
import { createGame, stepGame, type GameState } from "../src/game/sim";
import { GreedyController, RandomController, type Controller } from "../src/brain/controller";
import { heldOutSeeds } from "../src/train/rollout";

/**
 * Count ticks where the player lands on a hunting ghost's cell and survives without
 * eating it. Any such tick means a ghost was walked through.
 */
function countPassThroughs(controller: Controller, seeds: readonly number[]): number {
  let passes = 0;
  for (const seed of seeds) {
    const s = createGame(seed, DEFAULT_RULES);
    controller.reset(s);
    while (!s.done) {
      const before = s.ghosts.map((g) => ({ x: g.x, y: g.y, frightened: g.frightened, out: s.steps >= g.release }));
      const eatenBefore = s.ghostsEaten;
      const aliveBefore = s.alive;
      stepGame(s, controller.act(s));
      for (const g of before) {
        const landedOnGhost = g.out && !g.frightened && g.x === s.px && g.y === s.py;
        if (landedOnGhost && aliveBefore && s.alive && s.ghostsEaten === eatenBefore) passes++;
      }
    }
  }
  return passes;
}

describe("capture rules", () => {
  it("never lets the player walk through a hunting ghost", () => {
    const seeds = heldOutSeeds(30);
    expect(countPassThroughs(new GreedyController(), seeds)).toBe(0);
    expect(countPassThroughs(new RandomController(), seeds)).toBe(0);
  });

  it("catches the player who steps onto a ghost that leaves for a third cell", () => {
    const s = createGame(1, DEFAULT_RULES);
    const ghost = s.ghosts[0];
    // Place the ghost directly ahead of the player, free to move on, and released.
    const [dx, dy] = DIRS[0];
    while (!isOpen(s.maze, s.px + dx, s.py + dy)) s.pdir = (s.pdir + 1) % 4;
    ghost.x = s.px + dx;
    ghost.y = s.py + dy;
    ghost.release = 0;
    ghost.frightened = false;
    s.frightened = 0;
    stepGame(s, 0);
    expect(s.alive).toBe(false);
    expect(s.done).toBe(true);
  });

  it("eats a frightened ghost instead of dying, and sends it home", () => {
    const s = createGame(2, DEFAULT_RULES);
    const ghost = s.ghosts[0];
    ghost.release = 0;
    ghost.frightened = true;
    s.frightened = 10;
    let dir = -1;
    for (let d = 0; d < 4; d++) if (isOpen(s.maze, s.px + DIRS[d][0], s.py + DIRS[d][1])) dir = d;
    ghost.x = s.px + DIRS[dir][0];
    ghost.y = s.py + DIRS[dir][1];
    const scoreBefore = s.score;
    stepGame(s, dir);
    expect(s.alive).toBe(true);
    expect(s.ghostsEaten).toBe(1);
    expect(s.score).toBeGreaterThanOrEqual(scoreBefore + DEFAULT_RULES.ghostEatScore);
    expect(cellIndex(s.maze, ghost.x, ghost.y)).toBe(cellIndex(s.maze, ghost.startX, ghost.startY));
  });
});

describe("course variation", () => {
  it("starts different seeds in different places", () => {
    const starts = new Set<string>();
    for (let i = 0; i < 30; i++) {
      const s = createGame(500_000 + i, DEFAULT_RULES);
      starts.add(`${s.px},${s.py}`);
    }
    expect(starts.size).toBeGreaterThan(10);
  });

  it("keeps every start reachable and away from the ghost house", () => {
    for (let i = 0; i < 30; i++) {
      const s = createGame(600_000 + i, DEFAULT_RULES);
      expect(isOpen(s.maze, s.px, s.py)).toBe(true);
      for (const g of s.ghosts) expect(`${s.px},${s.py}`).not.toBe(`${g.startX},${g.startY}`);
    }
  });

  it("gives controllers different opening moves across seeds", () => {
    const openings = new Set<string>();
    for (let i = 0; i < 20; i++) {
      const s = createGame(700_000 + i, DEFAULT_RULES);
      const c: Controller = new GreedyController();
      c.reset(s);
      const path: string[] = [];
      while (!s.done && path.length < 12) {
        path.push(`${s.px},${s.py}`);
        stepGame(s, c.act(s));
      }
      openings.add(path.join("|"));
    }
    expect(openings.size).toBeGreaterThan(10);
  });
});
