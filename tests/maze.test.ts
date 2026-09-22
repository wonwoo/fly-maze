import { describe, expect, it } from "vitest";
import { LAYOUT, bfsDistances, cellIndex, parseMaze } from "../src/game/maze";

describe("maze", () => {
  const maze = parseMaze();

  it("has rectangular rows", () => {
    for (const row of LAYOUT) expect(row.length).toBe(LAYOUT[0].length);
  });

  it("reaches every pellet from the player start", () => {
    const dist = bfsDistances(maze, [cellIndex(maze, maze.start[0], maze.start[1])]);
    for (let i = 0; i < maze.pellet.length; i++) {
      if (maze.pellet[i]) expect(dist[i]).toBeGreaterThanOrEqual(0);
    }
    expect(maze.pelletCount).toBeGreaterThan(100);
  });

  it("keeps the ghost house behind a door only ghosts can pass", () => {
    expect(maze.ghostStarts.length).toBe(3);
    const doors = Array.from(maze.door).filter(Boolean).length;
    expect(doors).toBe(1);
    const start = [cellIndex(maze, maze.start[0], maze.start[1])];
    const playerDist = bfsDistances(maze, start, false);
    const ghostDist = bfsDistances(maze, start, true);
    for (const [x, y] of maze.ghostStarts) {
      expect(playerDist[cellIndex(maze, x, y)]).toBe(-1);
      expect(ghostDist[cellIndex(maze, x, y)]).toBeGreaterThan(0);
    }
  });
});
