import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { Circuit, type CircuitData } from "../src/brain/circuit";
import { paramCount, readoutInputCount } from "../src/brain/readout";
import { DEFAULT_RULES } from "../src/game/rules";
import { DEFAULT_CEM, trainCem } from "../src/train/cem";
import { OBJECTIVE, episodeFitness } from "../src/train/rollout";

const data = JSON.parse(readFileSync("public/data/circuit.json", "utf8")) as CircuitData;
const tiny = { ...DEFAULT_CEM, seed: 99, generations: 2, candidates: 6, elites: 2, coursesPerGeneration: 1, validationSeeds: [1_100_001], rules: { ...DEFAULT_RULES, maxSteps: 120 } };

describe("cross-entropy training", () => {
  it("is reproducible for a seed and only touches readout parameters", () => {
    const a = trainCem(new Circuit(data), tiny);
    const b = trainCem(new Circuit(data), tiny);
    expect(Array.from(a.champion)).toEqual(Array.from(b.champion));
    expect(a.champion.length).toBe(paramCount(readoutInputCount(32)));
    expect(a.history.length).toBe(2);
    expect(a.history.map((h) => h.courseSeeds)).toEqual(b.history.map((h) => h.courseSeeds));
    expect(Number.isFinite(a.championValidation)).toBe(true);
  });
});

describe("training objective", () => {
  const base = { seed: 1, score: 0, steps: 100, maxSteps: 1000, pellets: 50, ghostsEaten: 0, cleared: false, alive: true };

  it("prefers clearing over collecting, and clearing sooner over later", () => {
    const cleared = episodeFitness({ ...base, pellets: 182, cleared: true, steps: 400 });
    const nearlyCleared = episodeFitness({ ...base, pellets: 181, steps: 400 });
    const clearedLater = episodeFitness({ ...base, pellets: 182, cleared: true, steps: 800 });
    expect(cleared).toBeGreaterThan(nearlyCleared);
    expect(cleared).toBeGreaterThan(clearedLater);
  });

  it("makes being caught a loss rather than a neutral ending", () => {
    const survived = episodeFitness({ ...base, alive: true });
    const caught = episodeFitness({ ...base, alive: false });
    expect(caught).toBeLessThan(survived);
    expect(survived - caught).toBe(OBJECTIVE.caught);
  });

  it("still rewards pellets and eaten ghosts", () => {
    expect(episodeFitness({ ...base, pellets: 60 })).toBeGreaterThan(episodeFitness(base));
    expect(episodeFitness({ ...base, ghostsEaten: 1 })).toBeGreaterThan(episodeFitness(base));
  });

  it("does not let idle survival beat collecting", () => {
    const idleToTheEnd = episodeFitness({ ...base, pellets: 3, steps: 1000 });
    const collectedAndDied = episodeFitness({ ...base, pellets: 90, steps: 200, alive: false });
    expect(collectedAndDied).toBeGreaterThan(idleToTheEnd);
  });
});
