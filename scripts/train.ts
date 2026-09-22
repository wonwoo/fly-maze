import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { Circuit, type CircuitData } from "../src/brain/circuit";
import { EscapePathway } from "../src/brain/escape";
import { DEFAULT_CEM, trainCem } from "../src/train/cem";
import type { Checkpoint } from "../src/train/checkpoint";

const [seedArg, generationsArg, outDirArg, hiddenArg] = process.argv.slice(2);
const outDir = outDirArg ?? "public";
const circuitData = JSON.parse(readFileSync("public/data/circuit.json", "utf8")) as CircuitData;
const manifest = JSON.parse(readFileSync("public/data/manifest.json", "utf8")) as { circuit_sha256: string };
const config = {
  ...DEFAULT_CEM,
  seed: seedArg ? Number(seedArg) : DEFAULT_CEM.seed,
  generations: generationsArg ? Number(generationsArg) : DEFAULT_CEM.generations,
  hidden: hiddenArg !== undefined ? Number(hiddenArg) : DEFAULT_CEM.hidden,
};

console.log(`training seed ${config.seed}, ${config.generations} generations, ${config.candidates} candidates, ${config.elites} elites, hidden ${config.hidden}`);
const started = Date.now();
const pathway = new EscapePathway(new Circuit(JSON.parse(readFileSync("public/data/pathway/circuit.json", "utf8")) as CircuitData));
const result = trainCem(new Circuit(circuitData), config, (r) => {
  const flag = r.championUpdated ? " *" : "";
  console.log(
    `gen ${String(r.generation).padStart(3)}  best ${r.bestTrainFitness.toFixed(1).padStart(7)}  mean ${r.meanTrainFitness.toFixed(1).padStart(7)}  val ${r.validationScore.toFixed(1).padStart(7)}  champ ${r.championValidation.toFixed(1).padStart(7)}  sigma ${r.meanSigma.toFixed(3)}  ${(r.elapsedMs / 1000).toFixed(1)}s${flag}`,
  );
}, pathway);
const elapsed = (Date.now() - started) / 1000;

mkdirSync(join(outDir, "checkpoints"), { recursive: true });
mkdirSync(join(outDir, "benchmarks"), { recursive: true });
const checkpoint: Checkpoint = {
  version: 1,
  params: Array.from(result.champion),
  championGeneration: result.championGeneration,
  championValidation: result.championValidation,
  config,
  circuitSha256: manifest.circuit_sha256,
};
writeFileSync(join(outDir, "checkpoints", "readout.json"), JSON.stringify(checkpoint));
writeFileSync(
  join(outDir, "benchmarks", "training.json"),
  JSON.stringify({ config, elapsedSeconds: elapsed, championGeneration: result.championGeneration, championValidation: result.championValidation, initialParams: Array.from(result.initialParams), history: result.history }, null, 1),
);
console.log(`champion from generation ${result.championGeneration}, validation ${result.championValidation.toFixed(2)}, ${elapsed.toFixed(1)}s`);
