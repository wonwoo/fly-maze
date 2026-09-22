import { describe, expect, it } from "vitest";
import { generateMaze } from "../src/game/generate";
import { bfsDistances, cellIndex, isOpen } from "../src/game/maze";
import { DEFAULT_RULES } from "../src/game/rules";
import { createGame } from "../src/game/sim";

describe("generated mazes", () => {
  const seeds = Array.from({ length: 25 }, (_, i) => 4_000_001 + i);

  it("reaches every pellet from the player start", () => {
    for (const seed of seeds) {
      const maze = generateMaze(seed);
      const dist = bfsDistances(maze, [cellIndex(maze, maze.start[0], maze.start[1])]);
      for (let i = 0; i < maze.pellet.length; i++) {
        if (maze.pellet[i]) expect(dist[i]).toBeGreaterThanOrEqual(0);
      }
      expect(maze.pelletCount).toBeGreaterThan(120);
    }
  });

  it("lets ghosts leave the house and reach the player", () => {
    for (const seed of seeds) {
      const maze = generateMaze(seed);
      const fromGhosts = bfsDistances(
        maze,
        maze.ghostStarts.map(([x, y]) => cellIndex(maze, x, y)),
        true,
      );
      expect(maze.ghostStarts.length).toBe(3);
      expect(fromGhosts[cellIndex(maze, maze.start[0], maze.start[1])]).toBeGreaterThan(0);
      for (const [x, y] of maze.ghostStarts) expect(isOpen(maze, x, y, true)).toBe(true);
    }
  });

  it("keeps the ghost house closed to the player", () => {
    for (const seed of seeds) {
      const maze = generateMaze(seed);
      const fromStart = bfsDistances(maze, [cellIndex(maze, maze.start[0], maze.start[1])], false);
      for (const [x, y] of maze.ghostStarts) expect(fromStart[cellIndex(maze, x, y)]).toBe(-1);
    }
  });

  it("places six power pellets and is deterministic per seed", () => {
    for (const seed of seeds.slice(0, 8)) {
      const a = generateMaze(seed);
      const b = generateMaze(seed);
      expect(Array.from(a.wall)).toEqual(Array.from(b.wall));
      expect(Array.from(a.power).filter(Boolean).length).toBe(6);
    }
    expect(Array.from(generateMaze(seeds[0]).wall)).not.toEqual(Array.from(generateMaze(seeds[1]).wall));
  });

  it("gives each course its own maze when the rule is on", () => {
    const shapes = new Set<string>();
    for (const seed of seeds.slice(0, 10)) shapes.add(createGame(seed, DEFAULT_RULES).maze.wall.join(""));
    expect(shapes.size).toBe(10);
    const fixed = { ...DEFAULT_RULES, randomMaze: false };
    const fixedShapes = new Set(seeds.slice(0, 10).map((s) => createGame(s, fixed).maze.wall.join("")));
    expect(fixedShapes.size).toBe(1);
  });
});
