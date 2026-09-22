/**
 * Picks the softmax temperature the finished game should run at.
 *
 * The temperature is a deployment choice, not a trained parameter, so it has to be selected the
 * same way the champion was: on validation courses. Choosing it on the held-out courses would
 * make those courses part of the fitting procedure, and the score they then report would be an
 * overstatement. An earlier sweep showed exactly that failure — 0.6 looked best on 100 held-out
 * courses and lost to 0.25 on 300 fresh ones.
 *
 * The sweep runs on 512 validation courses, four times the 128 the champion was chosen on, so the
 * standard error is halved. They are still validation seeds; the held-out range is untouched.
 *
 * Every temperature plays the same 512 courses, so the comparisons are paired: the difference is
 * taken course by course and course difficulty cancels out. The unpaired standard error of a
 * single column is dominated by that difficulty and is far too wide to resolve the gap between
 * two temperatures.
 *
 * Both paired gaps are recorded, against the training temperature and against the temperature the
 * sweep picks. The second is what shows whether the pick is a measured optimum or one value out of
 * a band the sweep cannot separate, and the README quotes it, so it has to live in the artifact
 * rather than in a one-off script.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { Circuit, type CircuitData } from "../src/brain/circuit";
import { EscapePathway } from "../src/brain/escape";
import { ConnectomeController } from "../src/brain/controller";
import { readoutInputCount } from "../src/brain/readout";
import { parseCheckpoint } from "../src/train/checkpoint";
import { episodeFitness, runEpisode } from "../src/train/rollout";

const dir = process.argv[2] ?? "public";
const courses = Number(process.argv[3] ?? 512);
const TEMPERATURES = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1, 1.4];

const circuit = new Circuit(JSON.parse(readFileSync("public/data/circuit.json", "utf8")) as CircuitData);
const escape = new EscapePathway(new Circuit(JSON.parse(readFileSync("public/data/pathway/circuit.json", "utf8")) as CircuitData));
const checkpoint = parseCheckpoint(JSON.parse(readFileSync(join(dir, "checkpoints", "readout.json"), "utf8")), readoutInputCount(circuit.readoutCells.length));
const { rules, objective, validationSeeds } = checkpoint.config;
const params = Float64Array.from(checkpoint.params);
const seeds = Array.from({ length: courses }, (_, i) => validationSeeds[0] + i);

const rows = TEMPERATURES.map((temperature) => {
  const controller = new ConnectomeController(circuit, params, false, escape, temperature);
  const results = seeds.map((seed) => runEpisode(controller, seed, rules));
  const fitness = results.map((r) => episodeFitness(r, objective));
  const mean = fitness.reduce((a, f) => a + f, 0) / fitness.length;
  const variance = fitness.reduce((a, f) => a + (f - mean) ** 2, 0) / (fitness.length - 1);
  return {
    temperature,
    fitness,
    meanFitness: mean,
    standardError: Math.sqrt(variance / fitness.length),
    cleared: results.filter((r) => r.cleared).length,
    meanPellets: results.reduce((a, r) => a + r.pellets, 0) / results.length,
  };
});

const training = rows.find((r) => r.temperature === 1)!;
const best = rows.reduce((a, r) => (r.meanFitness > a.meanFitness ? r : a));

/** Paired difference between two temperatures, course by course. */
function pairedGap(row: (typeof rows)[number], baseline: (typeof rows)[number]): { gap: number; error: number } {
  const d = row.fitness.map((f, i) => f - baseline.fitness[i]);
  const gap = d.reduce((a, x) => a + x, 0) / d.length;
  const variance = d.reduce((a, x) => a + (x - gap) ** 2, 0) / (d.length - 1);
  return { gap, error: Math.sqrt(variance / d.length) };
}

// Two baselines, because they answer different questions. Against the training temperature the
// sweep shows that sampling less is worth doing at all; against the temperature it picked, it
// shows that no other low temperature is distinguishable from it, which is what keeps the choice
// from being read as a measured optimum.
const table = rows.map((r) => {
  const vsTraining = pairedGap(r, training);
  const vsBest = pairedGap(r, best);
  return {
    gap: vsTraining.gap,
    gapError: vsTraining.error,
    gapAgainstBest: vsBest.gap,
    gapAgainstBestError: vsBest.error,
    temperature: r.temperature,
    meanFitness: r.meanFitness,
    standardError: r.standardError,
    cleared: r.cleared,
    meanPellets: r.meanPellets,
  };
});

const signed = (v: number) => `${v >= 0 ? "+" : ""}${v.toFixed(1)}`;
console.log(`| Temperature | Mean validation fitness | Paired gap against 1 | Paired gap against ${best.temperature} | Cleared / ${courses} | Mean pellets |`);
console.log("| ---: | ---: | ---: | ---: | ---: | ---: |");
for (const r of table) {
  console.log(
    `| ${r.temperature} | ${r.meanFitness.toFixed(1)} ± ${r.standardError.toFixed(1)} | ${signed(r.gap)} ± ${r.gapError.toFixed(1)} | ${signed(r.gapAgainstBest)} ± ${r.gapAgainstBestError.toFixed(1)} | ${r.cleared} | ${r.meanPellets.toFixed(2)} |`,
  );
}

console.log(`\nbest temperature ${best.temperature}, paired gap against the training temperature of ${signed(pairedGap(best, training).gap)} ± ${pairedGap(best, training).error.toFixed(1)}`);
const contenders = table.filter((r) => r.temperature !== best.temperature && Math.abs(r.gapAgainstBest) < 2 * r.gapAgainstBestError);
console.log(`temperatures indistinguishable from it at two standard errors: ${contenders.map((r) => r.temperature).join(", ") || "none"}`);

// The sweep only describes the weights it swept, so it names them: check:assets rejects a sweep
// left behind by an earlier checkpoint.
writeFileSync(
  join(dir, "benchmarks", "temperature.json"),
  JSON.stringify({ seeds: { first: seeds[0], count: seeds.length }, checkpoint: { championGeneration: checkpoint.championGeneration, championValidation: checkpoint.championValidation, circuitSha256: checkpoint.circuitSha256 }, best: best.temperature, rows: table }, null, 1),
);
