/**
 * How much of what the readout needs is present in what it is given?
 *
 * Three least-squares probes are fitted on states collected from the published checkpoint playing
 * the held-out courses at the shipped temperature. A probe is not part of the game: it is fitted
 * afterwards, purely to ask what a linear function of a given input could have recovered. Where a
 * probe succeeds and the policy does not, the information was there and the training failed to
 * use it; where the probe fails too, the input never carried it.
 *
 *   distance   ghost distance from the four threat channels, and from the four escape channels,
 *              reported as R^2. The threat channel is relative by construction, so this measures
 *              how much absolute distance survives that encoding
 *   direction  which direction is best, from the 32 readout-cell activities and, as a ceiling,
 *              from the 20 raw feature channels, reported as top-1 accuracy over held-out states
 *   escape     which directions increase distance from the nearest hunting ghost, from the same
 *              32 activities, over exactly the states `avoidanceTarget` would teach
 *
 * Each probe is fitted on the first half of the collected states and scored on the second, so the
 * numbers are out-of-sample. Ridge regularisation keeps the normal equations solvable when a
 * channel is constant; the ridge is small enough not to move the reported figures.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { Circuit, type CircuitData } from "../src/brain/circuit";
import { EscapePathway } from "../src/brain/escape";
import { ConnectomeController, DEPLOY_TEMPERATURE } from "../src/brain/controller";
import { ACTIONS, readoutInputCount } from "../src/brain/readout";
import { DIRS, cellIndex, isOpen } from "../src/game/maze";
import { createGame, stepGame } from "../src/game/sim";
import { parseCheckpoint } from "../src/train/checkpoint";
import { heldOutSeeds } from "../src/train/rollout";

const dir = process.argv[2] ?? "public";
const RIDGE = 1e-6;

/** Solves (X'X + ridge I) w = X'y by Gaussian elimination with partial pivoting. */
function fit(X: number[][], y: number[]): number[] {
  const n = X[0].length;
  const A = Array.from({ length: n }, () => new Float64Array(n + 1));
  for (let r = 0; r < X.length; r++) {
    const x = X[r];
    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) A[i][j] += x[i] * x[j];
      A[i][n] += x[i] * y[r];
    }
  }
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < i; j++) A[i][j] = A[j][i];
    A[i][i] += RIDGE;
  }
  for (let c = 0; c < n; c++) {
    let pivot = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[pivot][c])) pivot = r;
    [A[c], A[pivot]] = [A[pivot], A[c]];
    if (Math.abs(A[c][c]) < 1e-12) continue;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = A[r][c] / A[c][c];
      for (let k = c; k <= n; k++) A[r][k] -= f * A[c][k];
    }
  }
  return Array.from({ length: n }, (_, i) => (Math.abs(A[i][i]) < 1e-12 ? 0 : A[i][n] / A[i][i]));
}

const dot = (w: number[], x: number[]): number => w.reduce((a, v, i) => a + v * x[i], 0);
const half = <T,>(a: T[]): [T[], T[]] => [a.slice(0, a.length >> 1), a.slice(a.length >> 1)];

/** Out-of-sample R^2 of a linear map from `X` to `y`. */
function rSquared(X: number[][], y: number[]): number {
  const [Xa, Xb] = half(X);
  const [ya, yb] = half(y);
  const w = fit(Xa, ya);
  const mean = yb.reduce((a, v) => a + v, 0) / yb.length;
  let ss = 0;
  let tot = 0;
  for (let i = 0; i < yb.length; i++) {
    ss += (yb[i] - dot(w, Xb[i])) ** 2;
    tot += (yb[i] - mean) ** 2;
  }
  return 1 - ss / tot;
}

/**
 * Out-of-sample top-1 accuracy of one linear score per action, fitted one-versus-rest.
 * `mask` marks the actions a state allows; the argmax is taken over those only.
 */
function accuracy(X: number[][], target: number[][], mask: number[][]): number {
  const [Xa, Xb] = half(X);
  const [ta, tb] = half(target);
  const [, mb] = half(mask);
  const w = Array.from({ length: ACTIONS }, (_, a) => fit(Xa, ta.map((t) => t[a])));
  let hit = 0;
  for (let i = 0; i < Xb.length; i++) {
    let best = -1;
    let bestScore = -Infinity;
    for (let a = 0; a < ACTIONS; a++) {
      if (!mb[i][a]) continue;
      const s = dot(w[a], Xb[i]);
      if (s > bestScore) {
        bestScore = s;
        best = a;
      }
    }
    if (best >= 0 && tb[i][best] > 0) hit++;
  }
  return hit / Xb.length;
}

const circuit = new Circuit(JSON.parse(readFileSync("public/data/circuit.json", "utf8")) as CircuitData);
const escape = new EscapePathway(new Circuit(JSON.parse(readFileSync("public/data/pathway/circuit.json", "utf8")) as CircuitData));
const checkpoint = parseCheckpoint(JSON.parse(readFileSync(join(dir, "checkpoints", "readout.json"), "utf8")), readoutInputCount(circuit.readoutCells.length));
const rules = checkpoint.config.rules;
const controller = new ConnectomeController(circuit, Float64Array.from(checkpoint.params), false, escape, DEPLOY_TEMPERATURE);
const base = circuit.readoutCells.length;

// Every row carries a trailing 1 so each probe fits its own intercept.
const threatX: number[][] = [];
const escapeX: number[][] = [];
const distanceY: number[] = [];
const cellsX: number[][] = [];
const featuresX: number[][] = [];
const bestTarget: number[][] = [];
const legalMask: number[][] = [];
const escCellsX: number[][] = [];
const escTarget: number[][] = [];
const escMask: number[][] = [];

for (const seed of heldOutSeeds()) {
  const s = createGame(seed, rules);
  controller.reset(s);
  while (!s.done) {
    const here = s.ghostDist[cellIndex(s.maze, s.px, s.py)];
    const legal = new Array<number>(ACTIONS).fill(0);
    const away = new Array<number>(ACTIONS).fill(0);
    const nearest = new Array<number>(ACTIONS).fill(0);
    let anyAway = 0;
    let legalCount = 0;
    let bestFood = Infinity;
    for (let d = 0; d < ACTIONS; d++) {
      const nx = s.px + DIRS[d][0];
      const ny = s.py + DIRS[d][1];
      if (!isOpen(s.maze, nx, ny)) continue;
      legal[d] = 1;
      legalCount++;
      const i = cellIndex(s.maze, nx, ny);
      const food = s.pelletDist[i];
      if (food >= 0 && food < bestFood) bestFood = food;
      nearest[d] = food;
      const there = s.ghostDist[i];
      if (here >= 0 && there >= 0 && there > here) {
        away[d] = 1;
        anyAway++;
      }
    }
    controller.act(s);
    const features = Array.from(controller.features);
    const cells = Array.from(controller.input.subarray(0, base));
    const esc = Array.from(controller.input.subarray(base));

    // Distance probes: only where a hunting ghost is actually reachable.
    if (s.frightened === 0 && here >= 0) {
      threatX.push([...features.slice(4, 8), 1]);
      escapeX.push([...esc, 1]);
      distanceY.push(here);
    }
    // Direction probe: the best direction is the legal one nearest a pellet.
    if (legalCount > 0 && Number.isFinite(bestFood)) {
      const target = new Array<number>(ACTIONS).fill(0);
      for (let d = 0; d < ACTIONS; d++) if (legal[d] && nearest[d] === bestFood) target[d] = 1;
      cellsX.push([...cells, 1]);
      featuresX.push([...features, 1]);
      bestTarget.push(target);
      legalMask.push(legal);
    }
    // Escape probe: exactly the states avoidanceTarget would teach.
    if (s.frightened === 0 && here >= 1 && here <= rules.nearDistance && anyAway > 0 && anyAway < legalCount) {
      escCellsX.push([...cells, 1]);
      escTarget.push(away);
      escMask.push(legal);
    }
    stepGame(s, controller.lastAction);
  }
}

const report = {
  temperature: DEPLOY_TEMPERATURE,
  seeds: { first: heldOutSeeds()[0], count: heldOutSeeds().length },
  checkpoint: { championGeneration: checkpoint.championGeneration, championValidation: checkpoint.championValidation, circuitSha256: checkpoint.circuitSha256 },
  distance: {
    states: distanceY.length,
    threatChannelsR2: rSquared(threatX, distanceY),
    escapeChannelsR2: rSquared(escapeX, distanceY),
  },
  direction: {
    states: bestTarget.length,
    readoutCellsAccuracy: accuracy(cellsX, bestTarget, legalMask),
    rawFeaturesAccuracy: accuracy(featuresX, bestTarget, legalMask),
  },
  escape: {
    states: escTarget.length,
    readoutCellsAccuracy: accuracy(escCellsX, escTarget, escMask),
  },
};

console.log(`probes fitted on the first half of each set and scored on the second, at temperature ${DEPLOY_TEMPERATURE}\n`);
console.log("| Probe | Input | Score | States |");
console.log("| --- | --- | ---: | ---: |");
console.log(`| Ghost distance | 4 threat channels | R² ${report.distance.threatChannelsR2.toFixed(3)} | ${report.distance.states} |`);
console.log(`| Ghost distance | 4 escape channels | R² ${report.distance.escapeChannelsR2.toFixed(3)} | ${report.distance.states} |`);
console.log(`| Best direction | 32 readout cells | ${(report.direction.readoutCellsAccuracy * 100).toFixed(1)}% | ${report.direction.states} |`);
console.log(`| Best direction | 20 raw features | ${(report.direction.rawFeaturesAccuracy * 100).toFixed(1)}% | ${report.direction.states} |`);
console.log(`| An escaping direction | 32 readout cells | ${(report.escape.readoutCellsAccuracy * 100).toFixed(1)}% | ${report.escape.states} |`);

writeFileSync(join(dir, "benchmarks", "probe.json"), JSON.stringify(report, null, 1));
