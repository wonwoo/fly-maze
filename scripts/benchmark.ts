import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { Circuit, type CircuitData } from "../src/brain/circuit";
import { EscapePathway } from "../src/brain/escape";
import { ConnectomeController, DEPLOY_TEMPERATURE, GreedyController, IdleController, RandomController, type Controller } from "../src/brain/controller";
import { parseCheckpoint } from "../src/train/checkpoint";
import { evaluate, heldOutSeeds, type EvaluationSummary } from "../src/train/rollout";
import { randomParams, readoutInputCount } from "../src/brain/readout";
import { Rng } from "../src/game/rng";

const dir = process.argv[2] ?? "public";
const circuitData = JSON.parse(readFileSync("public/data/circuit.json", "utf8")) as CircuitData;
const training = JSON.parse(readFileSync(join(dir, "benchmarks", "training.json"), "utf8")) as { initialParams: number[] };
const seeds = heldOutSeeds();
const circuit = new Circuit(circuitData);
const checkpoint = parseCheckpoint(JSON.parse(readFileSync(join(dir, "checkpoints", "readout.json"), "utf8")), readoutInputCount(circuit.readoutCells.length));
const rules = checkpoint.config.rules;
const escape = new EscapePathway(new Circuit(JSON.parse(readFileSync("public/data/pathway/circuit.json", "utf8")) as CircuitData));
const trained = Float64Array.from(checkpoint.params);
const started = Float64Array.from(training.initialParams);
// The untrained control is drawn the way a from-scratch run draws it, not taken from this run's
// starting point: a run continued from an earlier checkpoint starts already trained, and calling
// that "untrained" would turn the control into a second copy of the previous champion.
const untrained = randomParams(new Rng(checkpoint.config.seed), 0.1, readoutInputCount(circuit.readoutCells.length), checkpoint.config.hidden);
const warmStarted = started.length === untrained.length && started.some((v, i) => v !== untrained[i]);

const controllers: Controller[] = [
  new ConnectomeController(circuit, trained, false, escape, DEPLOY_TEMPERATURE),
  new ConnectomeController(circuit, trained, true, escape, DEPLOY_TEMPERATURE),
  new ConnectomeController(circuit, untrained, false, escape, DEPLOY_TEMPERATURE),
  ...(warmStarted ? [new ConnectomeController(circuit, started, false, escape, DEPLOY_TEMPERATURE)] : []),
  new GreedyController(),
  new RandomController(),
  new IdleController(),
];
const labels = [
  "connectome + trained readout",
  "same readout, circuit silenced",
  "untrained readout, random initialisation",
  ...(warmStarted ? ["readout this run continued from"] : []),
  "handwritten greedy-pellet baseline",
  "uniform random actions",
  "idle",
];

const summaries: Array<EvaluationSummary & { label: string }> = controllers.map((c, i) => ({ ...evaluate(c, seeds, rules), label: labels[i] }));
console.log(`| Controller | Cleared / ${seeds.length} | Mean score | Mean pellets | Mean steps |`);
console.log("| --- | ---: | ---: | ---: | ---: |");
for (const s of summaries) {
  console.log(`| ${s.label} | ${s.cleared} | ${s.meanScore.toFixed(2)} | ${s.meanPellets.toFixed(2)} | ${s.meanSteps.toFixed(2)} |`);
}
writeFileSync(
  join(dir, "benchmarks", "benchmark.json"),
  JSON.stringify({ seeds: { first: seeds[0], count: seeds.length }, rules, checkpoint: { championGeneration: checkpoint.championGeneration, championValidation: checkpoint.championValidation, circuitSha256: checkpoint.circuitSha256 }, controllers: summaries }, null, 1),
);
