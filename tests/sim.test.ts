import { describe, expect, it } from "vitest";
import { GreedyController, IdleController, RandomController, type Controller } from "../src/brain/controller";
import { FEATURE_COUNT, computeFeatures } from "../src/game/features";
import { DIRS, isOpen } from "../src/game/maze";
import { DEFAULT_RULES } from "../src/game/rules";
import { createGame, stepGame } from "../src/game/sim";
import { evaluate, runEpisode } from "../src/train/rollout";

describe("simulation", () => {
  it("is deterministic for a seed and controller", () => {
    const a = runEpisode(new RandomController(), 4242);
    const b = runEpisode(new RandomController(), 4242);
    expect(a).toEqual(b);
    expect(a.steps).toBeGreaterThan(0);
  });

  it("ends only by capture, clearing, or the step cap", () => {
    for (const seed of [1, 2, 3]) {
      const s = createGame(seed, { ...DEFAULT_RULES, maxSteps: 120 });
      const random = new RandomController();
      random.reset(s);
      while (!s.done) stepGame(s, random.act());
      expect(!s.alive || s.remaining === 0 || s.steps >= 120).toBe(true);
      if (s.cleared) expect(s.score).toBeGreaterThanOrEqual(1000);
    }
  });

  it("produces bounded features that match wall openness", () => {
    const s = createGame(7);
    const controller = new GreedyController();
    for (let t = 0; t < 40 && !s.done; t++) {
      const f = computeFeatures(s);
      expect(f.length).toBe(FEATURE_COUNT);
      for (let d = 0; d < 4; d++) {
        const open = isOpen(s.maze, s.px + DIRS[d][0], s.py + DIRS[d][1]);
        expect(f[8 + d]).toBe(open ? 1 : 0);
        if (!open) {
          expect(f[d]).toBe(0);
          expect(f[4 + d]).toBe(0);
          expect(f[12 + d]).toBe(0);
        }
        if (s.frightened === 0) expect(f[12 + d]).toBe(0);
      }
      for (const v of f) {
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThanOrEqual(1);
      }
      stepGame(s, controller.act(s));
    }
  });

  it("lets the greedy baseline eat more than idling", () => {
    const seeds = [11, 12, 13, 14, 15];
    const greedy = evaluate(new GreedyController(), seeds);
    const idle = evaluate(new IdleController(), seeds);
    expect(greedy.meanPellets).toBeGreaterThan(idle.meanPellets);
  });
});

describe("food encodings", () => {
  it("gives the relative channel a sharper direction preference than the absolute one", () => {
    const seeds = [21, 22, 23];
    const spread = (encoding: "absolute" | "relative" | "hybrid"): number => {
      let total = 0;
      let samples = 0;
      for (const seed of seeds) {
        const s = createGame(seed, { ...DEFAULT_RULES, foodEncoding: encoding });
        const controller: Controller = new GreedyController();
        controller.reset(s);
        for (let t = 0; t < 30 && !s.done; t++) {
          const f = computeFeatures(s);
          const open = [0, 1, 2, 3].filter((d) => isOpen(s.maze, s.px + DIRS[d][0], s.py + DIRS[d][1]));
          if (open.length > 1) {
            const values = open.map((d) => f[d]);
            total += Math.max(...values) - Math.min(...values);
            samples++;
          }
          stepGame(s, controller.act(s));
        }
      }
      return total / samples;
    };
    expect(spread("relative")).toBeGreaterThan(spread("absolute"));
    expect(spread("hybrid")).toBeGreaterThan(spread("absolute"));
  });

  it("keeps every encoding inside [0, 1]", () => {
    for (const encoding of ["absolute", "relative", "hybrid"] as const) {
      const s = createGame(31, { ...DEFAULT_RULES, foodEncoding: encoding });
      const controller: Controller = new GreedyController();
      controller.reset(s);
      for (let t = 0; t < 40 && !s.done; t++) {
        for (const v of computeFeatures(s)) {
          expect(v).toBeGreaterThanOrEqual(0);
          expect(v).toBeLessThanOrEqual(1);
        }
        stepGame(s, controller.act(s));
      }
    }
  });
});
