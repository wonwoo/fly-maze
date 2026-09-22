/**
 * How much of the trained readout's score survives on mazes it never trained on?
 *
 * Evaluates the published checkpoint on three course sets that share the same seeds but
 * differ in maze: the mazes the run trained under, freshly generated unseen mazes, and
 * the single built-in maze. Controls run on every set so the comparison is like for like.
 */
import { readFileSync } from "node:fs";
import { Circuit, type CircuitData } from "../src/brain/circuit";
import { EscapePathway } from "../src/brain/escape";
import { ConnectomeController, DEPLOY_TEMPERATURE, GreedyController, IdleController, RandomController, type Controller } from "../src/brain/controller";
import { DEFAULT_RULES, type GameRules } from "../src/game/rules";
import { parseCheckpoint } from "../src/train/checkpoint";
import { evaluate, heldOutSeeds } from "../src/train/rollout";
import { randomParams, readoutInputCount } from "../src/brain/readout";
import { Rng } from "../src/game/rng";

const dir = process.argv[2] ?? "public";
const circuit = new Circuit(JSON.parse(readFileSync("public/data/circuit.json", "utf8")) as CircuitData);
const checkpoint = parseCheckpoint(JSON.parse(readFileSync(`${dir}/checkpoints/readout.json`, "utf8")), readoutInputCount(circuit.readoutCells.length));
const escape = new EscapePathway(new Circuit(JSON.parse(readFileSync("public/data/pathway/circuit.json", "utf8")) as CircuitData));
const trained = Float64Array.from(checkpoint.params);
const rules: GameRules = { ...DEFAULT_RULES, ...checkpoint.config.rules };
const seeds = heldOutSeeds();

const sets: Array<[string, GameRules]> = [
  ["unseen generated mazes", { ...rules, randomMaze: true }],
  ["the single built-in maze", { ...rules, randomMaze: false }],
];
const makeControllers = (): Array<[string, Controller]> => [
  ["connectome + trained readout", new ConnectomeController(circuit, trained, false, escape, DEPLOY_TEMPERATURE)],
  ["same readout, circuit silenced", new ConnectomeController(circuit, trained, true, escape, DEPLOY_TEMPERATURE)],
  // Drawn the way a from-scratch run draws it. A run continued from an earlier checkpoint starts
  // already trained, so its own starting parameters are not an untrained control.
  ["untrained readout, random initialisation", new ConnectomeController(circuit, randomParams(new Rng(checkpoint.config.seed), 0.1, readoutInputCount(circuit.readoutCells.length), checkpoint.config.hidden), false, escape, DEPLOY_TEMPERATURE)],
  ["handwritten greedy baseline", new GreedyController()],
  ["uniform random actions", new RandomController()],
  ["idle", new IdleController()],
];

const report: Record<string, unknown> = { seeds: { first: seeds[0], count: seeds.length }, trainedOnRandomMazes: rules.randomMaze, sets: {} };
for (const [label, set] of sets) {
  console.log(`\n== ${label}`);
  console.log(`| Controller | Cleared / ${seeds.length} | Mean pellets | Mean survival |`);
  console.log("| --- | ---: | ---: | ---: |");
  const rows: Record<string, unknown> = {};
  for (const [name, controller] of makeControllers()) {
    const e = evaluate(controller, seeds, set);
    console.log(`| ${name} | ${e.cleared} | ${e.meanPellets.toFixed(2)} | ${e.meanSteps.toFixed(2)} |`);
    rows[name] = { cleared: e.cleared, meanPellets: e.meanPellets, meanSteps: e.meanSteps, meanScore: e.meanScore };
  }
  (report.sets as Record<string, unknown>)[label] = rows;
}
console.log(`\nwrote ${dir}/benchmarks/generalization.json`);
const { writeFileSync } = await import("node:fs");
writeFileSync(`${dir}/benchmarks/generalization.json`, JSON.stringify(report, null, 1));
