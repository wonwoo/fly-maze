import type { Circuit } from "../brain/circuit";
import { ConnectomeController } from "../brain/controller";
import type { EscapePathway } from "../brain/escape";
import { paramCount, randomParams, readoutInputCount } from "../brain/readout";
import { Rng } from "../game/rng";
import { DEFAULT_RULES, type GameRules } from "../game/rules";
import { OBJECTIVE, episodeFitness, runEpisode } from "./rollout";
import type { TrainingConfig } from "./checkpoint";

export interface CemConfig extends TrainingConfig {
  generations: number;
  candidates: number;
  elites: number;
  coursesPerGeneration: number;
  sigmaInit: number;
  initialChampionSigma: number;
  sigmaFloor: number;
  smoothing: number;
}

/**
 * Course and validation counts are set by measurement rather than taste.
 *
 * A candidate's score on one maze is dominated by which maze it drew, and `scripts/cem-noise.ts`
 * measures by how much. Ten generations in, repeating one candidate over six courses swings with
 * a standard deviation of 140.1 while the true spread between the generation's candidates is
 * 91.4; at 24 courses the swing falls to 58.8. That moves the rank correlation against the true
 * ordering from 0.83 to 0.91 and the share of the available improvement the elites capture from
 * 68% to 90%, which is why 24 is the default. It gets worse as the proposal narrows: by forty
 * generations six courses rank at 0.12 and capture 17%. The champion is kept on a fixed
 * validation set, so that set has to be wide enough that beating it means playing better rather
 * than fitting those particular mazes.
 */
export const DEFAULT_CEM: CemConfig = {
  seed: 20260917,
  generations: 400,
  candidates: 64,
  elites: 8,
  coursesPerGeneration: 24,
  hidden: 12,
  sigmaInit: 0.8,
  initialChampionSigma: 0.7,
  sigmaFloor: 0.07,
  smoothing: 0.7,
  validationSeeds: Array.from({ length: 32 }, (_, i) => 1_100_001 + i),
  rules: DEFAULT_RULES,
  objective: OBJECTIVE,
};

export interface GenerationRecord {
  generation: number;
  courseSeeds: number[];
  bestTrainFitness: number;
  meanTrainFitness: number;
  validationScore: number;
  championUpdated: boolean;
  championValidation: number;
  meanSigma: number;
  elapsedMs: number;
}

export interface CemResult {
  champion: Float64Array;
  championGeneration: number;
  championValidation: number;
  initialParams: Float64Array;
  history: GenerationRecord[];
  config: CemConfig;
}

export type GenerationCallback = (record: GenerationRecord, champion: Float64Array) => void;

/**
 * Diagonal Gaussian cross-entropy method over the readout parameters only.
 * Candidate 0 always carries the current validation champion. The champion is
 * replaced only when the generation's best training candidate strictly improves
 * the mean score on the fixed validation seeds.
 */
export function trainCem(circuit: Circuit, config: CemConfig = DEFAULT_CEM, onGeneration?: GenerationCallback, escape: EscapePathway | null = null): CemResult {
  const rng = new Rng(config.seed);
  const courseRng = new Rng((config.seed ^ 0x9e3779b9) >>> 0);
  const evaluateMean = (params: Float64Array, seeds: readonly number[]): number => {
    const controller = new ConnectomeController(circuit, params, false, escape);
    let total = 0;
    for (const seed of seeds) total += episodeFitness(runEpisode(controller, seed, config.rules), config.objective);
    return total / seeds.length;
  };

  const inputs = readoutInputCount(circuit.readoutCells.length);
  const size = paramCount(inputs, config.hidden);
  const mean = new Float64Array(size);
  const sigma = new Float64Array(size).fill(config.sigmaInit);
  const initialParams = randomParams(rng, config.initialChampionSigma, inputs, config.hidden);
  let champion = initialParams.slice();
  let championValidation = evaluateMean(champion, config.validationSeeds);
  let championGeneration = 0;
  const history: GenerationRecord[] = [];

  for (let generation = 1; generation <= config.generations; generation++) {
    const started = Date.now();
    const courseSeeds = Array.from({ length: config.coursesPerGeneration }, () => 1 + courseRng.int(900_000));
    const population: Float64Array[] = [champion.slice()];
    for (let k = 1; k < config.candidates; k++) {
      const p = new Float64Array(size);
      for (let i = 0; i < size; i++) p[i] = mean[i] + sigma[i] * rng.gaussian();
      population.push(p);
    }
    const fitness = population.map((p) => evaluateMean(p, courseSeeds));
    const order = fitness.map((_, i) => i).sort((a, b) => fitness[b] - fitness[a] || a - b);
    const elites = order.slice(0, config.elites).map((i) => population[i]);

    for (let i = 0; i < size; i++) {
      let m = 0;
      for (const e of elites) m += e[i];
      m /= elites.length;
      let v = 0;
      for (const e of elites) v += (e[i] - m) ** 2;
      const sd = Math.sqrt(v / elites.length);
      mean[i] = (1 - config.smoothing) * mean[i] + config.smoothing * m;
      sigma[i] = Math.max(config.sigmaFloor, (1 - config.smoothing) * sigma[i] + config.smoothing * sd);
    }

    const best = population[order[0]];
    const validationScore = evaluateMean(best, config.validationSeeds);
    let championUpdated = false;
    if (validationScore > championValidation) {
      champion = best.slice();
      championValidation = validationScore;
      championGeneration = generation;
      championUpdated = true;
    }
    const record: GenerationRecord = {
      generation,
      courseSeeds,
      bestTrainFitness: fitness[order[0]],
      meanTrainFitness: fitness.reduce((a, b) => a + b, 0) / fitness.length,
      validationScore,
      championUpdated,
      championValidation,
      meanSigma: sigma.reduce((a, b) => a + b, 0) / size,
      elapsedMs: Date.now() - started,
    };
    history.push(record);
    onGeneration?.(record, champion);
  }
  return { champion, championGeneration, championValidation, initialParams, history, config };
}
