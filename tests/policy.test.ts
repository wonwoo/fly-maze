import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { Circuit, type CircuitData } from "../src/brain/circuit";
import { ConnectomeController, GreedyController, type Controller } from "../src/brain/controller";
import { paramCount, randomParams, readoutBackward, readoutForward, readoutInputCount, ACTIONS } from "../src/brain/readout";
import { Rng } from "../src/game/rng";
import { DEFAULT_RULES } from "../src/game/rules";
import { createGame, stepGame } from "../src/game/sim";
import { DIRS, cellIndex, isOpen } from "../src/game/maze";
import { episodeFitness } from "../src/train/rollout";
import { avoidanceTarget, safetyPotential, stepReward } from "../src/train/policy";

const circuit = new Circuit(JSON.parse(readFileSync("public/data/circuit.json", "utf8")) as CircuitData);

/** Replays an episode, summing the per-step rewards the policy trainer would have paid. */
function rewardSum(controller: Controller, seed: number): { rewards: number; fitness: number } {
  const s = createGame(seed, DEFAULT_RULES);
  controller.reset(s);
  let rewards = 0;
  while (!s.done) {
    const remainingBefore = s.remaining;
    const eatenBefore = s.ghostsEaten;
    stepGame(s, controller.act(s));
    rewards += stepReward(remainingBefore - s.remaining, s.ghostsEaten - eatenBefore, s);
  }
  const fitness = episodeFitness({
    seed,
    score: s.score,
    steps: s.steps,
    maxSteps: DEFAULT_RULES.maxSteps,
    pellets: s.maze.pelletCount - s.remaining,
    ghostsEaten: s.ghostsEaten,
    cleared: s.cleared,
    alive: s.alive,
  });
  return { rewards, fitness };
}

describe("per-step reward", () => {
  it("sums to the episode fitness the cross-entropy method optimises", () => {
    // Two controllers with different endings, so a caught episode and a longer one both count.
    for (const seed of [2_100_001, 2_100_002, 2_100_003, 2_100_004]) {
      const greedy = rewardSum(new GreedyController(), seed);
      expect(greedy.rewards).toBeCloseTo(greedy.fitness, 6);
    }
  });

  it("holds for a randomly parameterised connectome controller", () => {
    const inputs = readoutInputCount(circuit.readoutCells.length);
    const params = randomParams(new Rng(7), 0.5, inputs, 0);
    for (const seed of [2_100_010, 2_100_011]) {
      const r = rewardSum(new ConnectomeController(circuit, params), seed);
      expect(r.rewards).toBeCloseTo(r.fitness, 6);
    }
  });
});

describe("readout gradient", () => {
  /** Central difference against the analytic gradient, for both readout shapes. */
  for (const hidden of [0, 5]) {
    it(`matches finite differences with hidden ${hidden}`, () => {
      const inputs = 9;
      const rng = new Rng(11 + hidden);
      const params = randomParams(rng, 0.6, inputs, hidden);
      const x = Float64Array.from({ length: inputs }, () => rng.gaussian());
      const dScores = Float64Array.from({ length: ACTIONS }, () => rng.gaussian());
      // Loss = sum_a dScores[a] * score[a], whose gradient in the scores is dScores itself.
      const loss = (p: Float64Array): number => {
        const out = readoutForward(p, x);
        let total = 0;
        for (let a = 0; a < ACTIONS; a++) total += dScores[a] * out[a];
        return total;
      };
      const analytic = new Float64Array(paramCount(inputs, hidden));
      readoutBackward(params, x, dScores, analytic);
      const step = 1e-6;
      for (let i = 0; i < analytic.length; i += 7) {
        const up = params.slice();
        const down = params.slice();
        up[i] += step;
        down[i] -= step;
        expect(analytic[i]).toBeCloseTo((loss(up) - loss(down)) / (2 * step), 5);
      }
    });
  }
});

describe("avoidance target", () => {
  it("only ever points at legal moves that increase the distance to a hunting ghost", () => {
    const out = new Float64Array(ACTIONS);
    const controller: Controller = new GreedyController();
    let taught = 0;
    let silent = 0;
    for (const seed of [2_100_020, 2_100_021, 2_100_022, 2_100_023]) {
      const s = createGame(seed, DEFAULT_RULES);
      controller.reset(s);
      while (!s.done) {
        const here = s.ghostDist[cellIndex(s.maze, s.px, s.py)];
        if (avoidanceTarget(s, out)) {
          taught++;
          expect(here).toBeGreaterThanOrEqual(1);
          expect(here).toBeLessThanOrEqual(DEFAULT_RULES.nearDistance);
          expect(s.frightened).toBe(0);
          let mass = 0;
          for (let d = 0; d < ACTIONS; d++) {
            mass += out[d];
            if (out[d] === 0) continue;
            const nx = s.px + DIRS[d][0];
            const ny = s.py + DIRS[d][1];
            expect(isOpen(s.maze, nx, ny)).toBe(true);
            expect(s.ghostDist[cellIndex(s.maze, nx, ny)]).toBeGreaterThan(here);
          }
          expect(mass).toBeCloseTo(1, 12);
        } else {
          silent++;
        }
        stepGame(s, controller.act(s));
      }
    }
    // Both branches have to be exercised for the invariants above to mean anything.
    expect(taught).toBeGreaterThan(0);
    expect(silent).toBeGreaterThan(0);
  });

  it("says nothing while the ghosts are fleeing", () => {
    const out = new Float64Array(ACTIONS);
    const s = createGame(2_100_030, DEFAULT_RULES);
    s.frightened = 10;
    expect(avoidanceTarget(s, out)).toBe(false);
  });
});

describe("safety potential", () => {
  it("with openness weight, never leaves [0, 1] and prefers the cell with more ways out", () => {
    const controller: Controller = new GreedyController();
    let comparisons = 0;
    for (const seed of [2_100_050, 2_100_051]) {
      const s = createGame(seed, DEFAULT_RULES);
      controller.reset(s);
      while (!s.done) {
        for (const w of [0, 0.5, 1]) {
          const phi = safetyPotential(s, w);
          expect(phi).toBeGreaterThanOrEqual(0);
          expect(phi).toBeLessThanOrEqual(1);
        }
        // Openness can only add safety, and adds none once the ghost is out of range.
        const plain = safetyPotential(s, 0);
        const open = safetyPotential(s, 1);
        expect(open).toBeGreaterThanOrEqual(plain - 1e-12);
        if (plain === 1) expect(open).toBeCloseTo(1, 12);
        else if (plain < 1) comparisons++;
        stepGame(s, controller.act(s));
      }
    }
    expect(comparisons).toBeGreaterThan(0);
  });

  it("stays in [0, 1], reads zero at the end of a course, and rises with ghost distance", () => {
    const controller: Controller = new GreedyController();
    let atTerminal = -1;
    const byDistance = new Map<number, number>();
    for (const seed of [2_100_040, 2_100_041, 2_100_042]) {
      const s = createGame(seed, DEFAULT_RULES);
      controller.reset(s);
      while (!s.done) {
        const phi = safetyPotential(s);
        expect(phi).toBeGreaterThanOrEqual(0);
        expect(phi).toBeLessThanOrEqual(1);
        if (s.frightened === 0) {
          const d = s.ghostDist[cellIndex(s.maze, s.px, s.py)];
          if (d >= 0 && !byDistance.has(d)) byDistance.set(d, phi);
        }
        stepGame(s, controller.act(s));
      }
      // Shaping only telescopes to a constant if the potential is zero once the course is over.
      atTerminal = safetyPotential(s);
      expect(atTerminal).toBe(0);
    }
    const seen = [...byDistance.entries()].sort((a, b) => a[0] - b[0]);
    expect(seen.length).toBeGreaterThan(2);
    for (let i = 1; i < seen.length; i++) expect(seen[i][1]).toBeGreaterThanOrEqual(seen[i - 1][1]);
  });
});
