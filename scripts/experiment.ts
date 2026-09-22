import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { Circuit, type CircuitData } from "../src/brain/circuit";
import { EscapePathway } from "../src/brain/escape";
import { ConnectomeController, GreedyController, RandomController } from "../src/brain/controller";
import { DEFAULT_RULES, type GameRules } from "../src/game/rules";
import { DEFAULT_CEM, trainCem, type CemConfig } from "../src/train/cem";
import { evaluate, heldOutSeeds, type EvaluationSummary } from "../src/train/rollout";

interface Variant {
  description: string;
  rules?: Partial<GameRules>;
  circuit?: string;
  generations?: number;
  seed?: number;
}

/**
 * One-factor-at-a-time variants against the published configuration in src/game/rules.ts.
 * Each writes public/benchmarks/experiments/<name>.json with its own rules, circuit and
 * held-out results, so every row of the README's comparison table has a matching file.
 */
export const VARIANTS: Record<string, Variant> = {
  "self-seed1": { description: "20 channels with heading, seed 20260917" },
  "self-seed2": { description: "20 channels with heading, seed 20260918", seed: 20260918 },
  "fixed-maze": { description: "train and evaluate on the single built-in maze", rules: { randomMaze: false } },
  "replica-seed2": { description: "published configuration, training seed 20260918", seed: 20260918 },
  "replica-seed3": { description: "published configuration, training seed 20260919", seed: 20260919 },
  gens200: { description: "200 training generations instead of 400", generations: 200 },
  "slow-ghosts": { description: "ghosts rest every 3rd tick (2/3 speed)", rules: { ghostRestEvery: 3 } },
  "low-chase": { description: "chase probabilities 0.6 / 0.45 / 0.3", rules: { ghostChase: [0.6, 0.45, 0.3] } },
  "no-power-pellets": { description: "power pellets score as plain pellets", rules: { powerPellets: false } },
  "food-absolute": { description: "absolute food channel 1 - dist/40", rules: { foodEncoding: "absolute" } },
  "food-relative-6": { description: "relative food channel, contrast 6", rules: { foodContrast: 6 } },
  "fixed-start": { description: "every course starts at the same cell, no release jitter", rules: { randomStart: false, releaseJitter: 0 } },
  readout16: { description: "16 descending readout cells instead of 32", circuit: "public/data/variants/circuit-readout16.json" },
  "readout16-seed2": { description: "16 readout cells, seed 20260918", circuit: "public/data/variants/circuit-readout16.json", seed: 20260918 },
  "readout16-seed3": { description: "16 readout cells, seed 20260919", circuit: "public/data/variants/circuit-readout16.json", seed: 20260919 },
};

const name = process.argv[2];
const variant = VARIANTS[name];
if (!variant) throw new Error(`unknown variant ${name}; choose one of ${Object.keys(VARIANTS).join(", ")}`);

const rules: GameRules = { ...DEFAULT_RULES, ...variant.rules };
const circuitPath = variant.circuit ?? "public/data/circuit.json";
const circuit = new Circuit(JSON.parse(readFileSync(circuitPath, "utf8")) as CircuitData);
const config: CemConfig = { ...DEFAULT_CEM, rules, generations: variant.generations ?? DEFAULT_CEM.generations, seed: variant.seed ?? DEFAULT_CEM.seed };

const started = Date.now();
const escape = new EscapePathway(new Circuit(JSON.parse(readFileSync("public/data/pathway/circuit.json", "utf8")) as CircuitData));
const result = trainCem(circuit, config, undefined, escape);
const trainSeconds = (Date.now() - started) / 1000;
const seeds = heldOutSeeds();
const compact = (s: EvaluationSummary) => ({ cleared: s.cleared, meanScore: s.meanScore, meanPellets: s.meanPellets, meanSteps: s.meanSteps });
const summary = {
  name,
  description: variant.description,
  rules,
  circuit: circuitPath,
  readoutCells: circuit.readoutCells.length,
  generations: config.generations,
  seed: config.seed,
  championGeneration: result.championGeneration,
  championValidation: result.championValidation,
  trainSeconds,
  trained: compact(evaluate(new ConnectomeController(circuit, result.champion, false, escape), seeds, rules)),
  silenced: compact(evaluate(new ConnectomeController(circuit, result.champion, true, escape), seeds, rules)),
  greedy: compact(evaluate(new GreedyController(), seeds, rules)),
  random: compact(evaluate(new RandomController(), seeds, rules)),
  championCurve: result.history.map((h) => h.championValidation),
  params: Array.from(result.champion),
};
mkdirSync("public/benchmarks/experiments", { recursive: true });
writeFileSync(`public/benchmarks/experiments/${name}.json`, JSON.stringify(summary, null, 1));
console.log(JSON.stringify({ name, trained: summary.trained, greedy: summary.greedy, random: summary.random, silenced: summary.silenced, championValidation: summary.championValidation, trainSeconds }));
