/**
 * Why the cross-entropy method stalled: measuring the noise it selected on.
 *
 * `src/train/cem.ts` scores a candidate on whole games, so a few hundred decisions collapse into
 * one number and the number is dominated by which mazes the generation happened to draw. Three
 * quantities say how bad that is, and this script measures all three rather than quoting them:
 *
 *   draw noise     the spread of one parameter vector's own score across independent draws of
 *                  `courses` mazes. This is the error bar the selection is reading through
 *   candidate spread  the spread of true fitness across the candidates of one generation. This is
 *                  the signal. When draw noise exceeds it, the ranking is mostly the draw
 *   rank fidelity  Spearman correlation between the generation's ranking and the true ranking,
 *                  and how much of the available improvement the elites actually capture
 *
 * "True" fitness means the mean over `TRUTH_COURSES` fresh mazes, wide enough that its own error
 * is small next to the effects being measured. The population is drawn the way the trainer draws
 * one, around a champion warmed up by a real short run, using the mean sigma that run reported;
 * the trainer keeps a per-parameter sigma, so this is that sigma averaged, not reproduced exactly.
 *
 * All three are measured at several points along a run, because they are not one pair of numbers:
 * as the proposal narrows, the candidates crowd together while the draw noise does not shrink with
 * them, and a ranking that starts out informative ends up close to a coin toss. A single figure
 * quoted without saying where in the run it was taken does not describe the method.
 *
 * Nothing here trains anything. It measures the selection signal the method had to work with.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { Circuit, type CircuitData } from "../src/brain/circuit";
import { EscapePathway } from "../src/brain/escape";
import { ConnectomeController } from "../src/brain/controller";
import { paramCount, readoutInputCount } from "../src/brain/readout";
import { Rng } from "../src/game/rng";
import { DEFAULT_CEM, trainCem } from "../src/train/cem";
import { OBJECTIVE, episodeFitness, runEpisode } from "../src/train/rollout";
import { DEFAULT_RULES } from "../src/game/rules";

const dir = process.argv[2] ?? "public";
const WARMUP_POINTS = (process.argv[3] ?? "1,10,40").split(",").map(Number);
const CANDIDATES = DEFAULT_CEM.candidates;
const ELITES = DEFAULT_CEM.elites;
const REPEATS = 40;
const TRUTH_COURSES = 256;
const COURSE_COUNTS = [6, 24];

const circuit = new Circuit(JSON.parse(readFileSync("public/data/circuit.json", "utf8")) as CircuitData);
const escape = new EscapePathway(new Circuit(JSON.parse(readFileSync("public/data/pathway/circuit.json", "utf8")) as CircuitData));

/** Mean fitness of one parameter vector over the given mazes, exactly as the trainer scores. */
const score = (params: Float64Array, seeds: readonly number[]): number => {
  const controller = new ConnectomeController(circuit, params, false, escape);
  let total = 0;
  for (const seed of seeds) total += episodeFitness(runEpisode(controller, seed, DEFAULT_RULES), OBJECTIVE);
  return total / seeds.length;
};

const sd = (xs: number[]): number => {
  const m = xs.reduce((a, x) => a + x, 0) / xs.length;
  return Math.sqrt(xs.reduce((a, x) => a + (x - m) ** 2, 0) / (xs.length - 1));
};

/** Spearman correlation; there are no ties in these scores, so plain ranks suffice. */
function spearman(a: number[], b: number[]): number {
  const rank = (xs: number[]): number[] => {
    const order = xs.map((_, i) => i).sort((i, j) => xs[i] - xs[j]);
    const r = new Array<number>(xs.length);
    order.forEach((idx, pos) => (r[idx] = pos));
    return r;
  };
  const ra = rank(a);
  const rb = rank(b);
  const n = a.length;
  let s = 0;
  for (let i = 0; i < n; i++) s += (ra[i] - rb[i]) ** 2;
  return 1 - (6 * s) / (n * (n * n - 1));
}

const courseRng = new Rng(777);
const draw = (n: number): number[] => Array.from({ length: n }, () => 1 + courseRng.int(900_000));
const size = paramCount(readoutInputCount(circuit.readoutCells.length), DEFAULT_CEM.hidden);

/** One operating point: warm up that many generations, then measure the population it leaves. */
function measure(generations: number) {
  // A champion and a sigma from a real run, so the population sits where the method puts it
  // rather than somewhere arbitrary in parameter space.
  const warm = trainCem(circuit, { ...DEFAULT_CEM, generations }, undefined, escape);
  const sigma = warm.history[warm.history.length - 1].meanSigma;
  const rng = new Rng(20260922);
  const population: Float64Array[] = [warm.champion.slice()];
  for (let k = 1; k < CANDIDATES; k++) {
    const p = new Float64Array(size);
    for (let i = 0; i < size; i++) p[i] = warm.champion[i] + sigma * rng.gaussian();
    population.push(p);
  }

  // 1. Draw noise: one vector, many independent draws of the same number of mazes.
  const drawNoise = COURSE_COUNTS.map((courses) => ({
    courses,
    standardDeviation: sd(Array.from({ length: REPEATS }, () => score(population[0], draw(courses)))),
  }));

  // 2. Candidate spread and 3. rank fidelity, both against fitness on TRUTH_COURSES fresh mazes.
  const truthSeeds = draw(TRUTH_COURSES);
  const truth = population.map((p) => score(p, truthSeeds));
  const truthOrder = truth.map((_, i) => i).sort((a, b) => truth[b] - truth[a]);
  const populationMean = truth.reduce((a, x) => a + x, 0) / truth.length;
  const bestPossible = truthOrder.slice(0, ELITES).reduce((a, i) => a + truth[i], 0) / ELITES - populationMean;

  const ranking = COURSE_COUNTS.map((courses) => {
    const seeds = draw(courses);
    const noisy = population.map((p) => score(p, seeds));
    const picked = noisy.map((_, i) => i).sort((a, b) => noisy[b] - noisy[a]).slice(0, ELITES);
    const captured = picked.reduce((a, i) => a + truth[i], 0) / ELITES - populationMean;
    return { courses, rankCorrelation: spearman(noisy, truth), eliteCapture: captured / bestPossible };
  });

  return { generations, championGeneration: warm.championGeneration, meanSigma: sigma, drawNoise, candidateSpread: sd(truth), ranking };
}

const points = WARMUP_POINTS.map((g) => {
  console.log(`warming up ${g} generations at ${DEFAULT_CEM.coursesPerGeneration} courses...`);
  const p = measure(g);
  console.log(`  champion from generation ${p.championGeneration}, mean sigma ${p.meanSigma.toFixed(3)}`);
  return p;
});

console.log("\n| After | Mean sigma | Courses | Draw noise | Candidate spread | Rank correlation | Elite capture |");
console.log("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
for (const p of points) {
  for (let i = 0; i < COURSE_COUNTS.length; i++) {
    const d = p.drawNoise[i];
    const r = p.ranking[i];
    console.log(`| ${p.generations} gen | ${p.meanSigma.toFixed(3)} | ${d.courses} | ${d.standardDeviation.toFixed(1)} | ${p.candidateSpread.toFixed(1)} | ${r.rankCorrelation.toFixed(2)} | ${(r.eliteCapture * 100).toFixed(0)}% |`);
  }
}
console.log(`\nspread over ${CANDIDATES} candidates scored on ${TRUTH_COURSES} fresh courses; draw noise over ${REPEATS} repeats`);

writeFileSync(join(dir, "benchmarks", "cem-noise.json"), JSON.stringify({ population: { candidates: CANDIDATES, elites: ELITES, truthCourses: TRUTH_COURSES, repeats: REPEATS }, points }, null, 1));
