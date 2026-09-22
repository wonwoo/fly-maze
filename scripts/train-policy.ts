import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { Circuit, type CircuitData } from "../src/brain/circuit";
import { EscapePathway } from "../src/brain/escape";
import { DEFAULT_POLICY, trainPolicy } from "../src/train/policy";
import type { Checkpoint } from "../src/train/checkpoint";
import type { GameRules } from "../src/game/rules";

const [seedArg, updatesArg, outDirArg, hiddenArg, perStepArg] = process.argv.slice(2);
const outDir = outDirArg ?? "public";
const circuitData = JSON.parse(readFileSync("public/data/circuit.json", "utf8")) as CircuitData;
const manifest = JSON.parse(readFileSync("public/data/manifest.json", "utf8")) as { circuit_sha256: string };
const config = {
  ...DEFAULT_POLICY,
  seed: seedArg ? Number(seedArg) : DEFAULT_POLICY.seed,
  updates: updatesArg ? Number(updatesArg) : DEFAULT_POLICY.updates,
  hidden: hiddenArg !== undefined ? Number(hiddenArg) : DEFAULT_POLICY.hidden,
  avoidance: process.env.FLY_MAZE_AVOIDANCE ? Number(process.env.FLY_MAZE_AVOIDANCE) : DEFAULT_POLICY.avoidance,
  shaping: process.env.FLY_MAZE_SHAPING ? Number(process.env.FLY_MAZE_SHAPING) : DEFAULT_POLICY.shaping,
  openness: process.env.FLY_MAZE_OPENNESS ? Number(process.env.FLY_MAZE_OPENNESS) : DEFAULT_POLICY.openness,
  learningRate: process.env.FLY_MAZE_LR ? Number(process.env.FLY_MAZE_LR) : DEFAULT_POLICY.learningRate,
  objective: {
    ...DEFAULT_POLICY.objective,
    perStep: perStepArg !== undefined ? Number(perStepArg) : DEFAULT_POLICY.objective.perStep,
    ...(process.env.FLY_MAZE_CAUGHT ? { caught: Number(process.env.FLY_MAZE_CAUGHT) } : {}),
  },
  // Rule overrides for one-factor experiments; the checkpoint records whatever they end up as.
  rules: {
    ...DEFAULT_POLICY.rules,
    ...(process.env.FLY_MAZE_THREAT_ENCODING ? { threatEncoding: process.env.FLY_MAZE_THREAT_ENCODING as GameRules["threatEncoding"] } : {}),
    ...(process.env.FLY_MAZE_THREAT_RANGE ? { threatRange: Number(process.env.FLY_MAZE_THREAT_RANGE) } : {}),
    ...(process.env.FLY_MAZE_ESCAPE_CHANNELS ? { escapeChannels: process.env.FLY_MAZE_ESCAPE_CHANNELS as GameRules["escapeChannels"] } : {}),
  },
};

console.log(`policy gradient, seed ${config.seed}, ${config.updates} updates, ${config.episodesPerUpdate} episodes each, hidden ${config.hidden}, perStep ${config.objective.perStep}, threat ${config.rules.threatEncoding}/${config.rules.threatRange}, caught ${config.objective.caught}, escape ${config.rules.escapeChannels}, avoidance ${config.avoidance}, shaping ${config.shaping}/openness ${config.openness}, lr ${config.learningRate}${process.env.FLY_MAZE_INIT ? `, continuing from ${process.env.FLY_MAZE_INIT}` : ""}`);
const started = Date.now();
const pathway = new EscapePathway(new Circuit(JSON.parse(readFileSync("public/data/pathway/circuit.json", "utf8")) as CircuitData));
// Continuing from a finished readout keeps the pellet-collecting weights that already work and
// lets the run spend its budget on what the shaping adds, rather than relearning both at once.
const initial = process.env.FLY_MAZE_INIT
  ? Float64Array.from((JSON.parse(readFileSync(process.env.FLY_MAZE_INIT, "utf8")) as { params: number[] }).params)
  : null;
const result = trainPolicy(new Circuit(circuitData), config, (r) => {
  if (r.update % 10 !== 0 && !r.championUpdated) return;
  const flag = r.championUpdated ? " *" : "";
  console.log(
    `upd ${String(r.update).padStart(4)}  return ${r.meanReturn.toFixed(1).padStart(8)}  pellets ${r.meanPellets.toFixed(1).padStart(6)}  entropy ${r.meanEntropy.toFixed(3)}  val ${r.validationScore.toFixed(1).padStart(8)}  champ ${r.championValidation.toFixed(1).padStart(8)}  ${(r.elapsedMs / 1000).toFixed(1)}s${flag}`,
  );
}, pathway, initial);
const elapsed = (Date.now() - started) / 1000;

mkdirSync(join(outDir, "checkpoints"), { recursive: true });
mkdirSync(join(outDir, "benchmarks"), { recursive: true });
const checkpoint: Checkpoint = {
  version: 1,
  params: Array.from(result.champion),
  championGeneration: result.championUpdate,
  championValidation: result.championValidation,
  config,
  circuitSha256: manifest.circuit_sha256,
};
writeFileSync(join(outDir, "checkpoints", "readout.json"), JSON.stringify(checkpoint));
writeFileSync(
  join(outDir, "benchmarks", "training.json"),
  JSON.stringify({ config, elapsedSeconds: elapsed, championGeneration: result.championUpdate, championValidation: result.championValidation, initialParams: Array.from(result.initialParams), history: result.history }, null, 1),
);
console.log(`champion from update ${result.championUpdate}, validation ${result.championValidation.toFixed(2)}, ${elapsed.toFixed(1)}s`);
