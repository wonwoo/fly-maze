import { describe, expect, it } from "vitest";
import { GreedyController, RandomController, type Controller } from "../src/brain/controller";
import { computeFeatures } from "../src/game/features";
import { DEFAULT_RULES } from "../src/game/rules";
import { createGame, stepGame, type GameState } from "../src/game/sim";
import { renderGame } from "../src/ui/render";

/** Minimal 2D context stub: renderGame may call anything on it, but must not change state. */
function stubContext(): CanvasRenderingContext2D {
  const handler: ProxyHandler<object> = {
    get: (_t, prop) => (prop === "canvas" ? { width: 456, height: 504 } : () => undefined),
    set: () => true,
  };
  return new Proxy({}, handler) as unknown as CanvasRenderingContext2D;
}

function trace(controller: Controller, seed: number, draw: boolean): string[] {
  const s = createGame(seed, DEFAULT_RULES);
  controller.reset(s);
  const ctx = draw ? stubContext() : null;
  const steps: string[] = [];
  while (!s.done && steps.length < 300) {
    if (ctx) renderGame(ctx, s);
    stepGame(s, controller.act(s));
    steps.push(snapshot(s));
  }
  return steps;
}

function snapshot(s: GameState): string {
  return [s.px, s.py, s.pdir, s.score, s.remaining, s.frightened, s.ghostsEaten, ...s.ghosts.flatMap((g) => [g.x, g.y, Number(g.frightened)])].join(",");
}

describe("render parity", () => {
  it("produces identical trajectories with and without a drawing surface", () => {
    for (const seed of [11, 4242, 900_123]) {
      expect(trace(new GreedyController(), seed, true)).toEqual(trace(new GreedyController(), seed, false));
      expect(trace(new RandomController(), seed, true)).toEqual(trace(new RandomController(), seed, false));
    }
  });

  it("leaves the observation encoder unaffected by rendering", () => {
    const a = createGame(777, DEFAULT_RULES);
    const b = createGame(777, DEFAULT_RULES);
    renderGame(stubContext(), a);
    expect(Array.from(computeFeatures(a))).toEqual(Array.from(computeFeatures(b)));
  });
});
