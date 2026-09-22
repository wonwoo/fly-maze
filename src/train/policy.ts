import type { Circuit } from "../brain/circuit";
import { ConnectomeController } from "../brain/controller";
import type { EscapePathway } from "../brain/escape";
import { ACTIONS, paramCount, randomParams, readoutBackward, readoutInputCount } from "../brain/readout";
import { DIRS, cellIndex, isOpen } from "../game/maze";
import { Rng } from "../game/rng";
import { DEFAULT_RULES, type GameRules } from "../game/rules";
import { createGame, stepGame, type GameState } from "../game/sim";
import type { TrainingConfig } from "./checkpoint";
import { OBJECTIVE, episodeFitness, runEpisode, type Objective } from "./rollout";

export interface PolicyConfig extends TrainingConfig {
  updates: number;
  /** Episodes rolled out per parameter update. */
  episodesPerUpdate: number;
  learningRate: number;
  /** Discount on future reward, which is what lets a step be credited for what follows it. */
  discount: number;
  /** Softmax temperature on the direction scores while collecting experience. */
  temperature: number;
  /** Weight on the entropy bonus, which keeps the policy from committing before it has looked. */
  entropy: number;
  /**
   * Weight on potential-based reward shaping over the distance to the nearest hunting ghost.
   *
   * The supervised term below taught the wrong lesson: at weight 8 it lifted escapes at one to
   * three cells from 40.1% to 80.8% and still lost every one of the hundred held-out courses,
   * because always taking the step that gains a cell walks into dead ends. Shaping asks for
   * nothing in particular. It only moves reward the game already pays to the tick where the
   * distance changed, and because it is the difference of a potential it leaves the best policy
   * exactly where it was (Ng, Harada and Russell, 1999). Getting caught still costs what it costs,
   * so the fly is free to discover that a corridor it cannot leave is worse than a cell closer.
   */
  shaping: number;
  /**
   * How much of the safety potential comes from the number of ways out of the current cell,
   * rather than from distance to the ghost alone. Zero is the distance-only potential.
   */
  openness: number;
  /**
   * Weight on the auxiliary avoidance loss. Zero leaves the objective exactly as published.
   *
   * Measured and rejected: 0.5, 2 and 8 all cleared 0 of 100 held-out courses against the
   * published 9, while raising escape rates. Kept at zero, and kept reproducible.
   *
   * Reward alone does not teach avoidance: a pellet pays ten points a hundred times a course and
   * being caught costs five hundred once, so the policy converges on the frequent signal and
   * stops. Measured on the published checkpoint over the 300 held-out courses at the shipped
   * temperature, it moves away from a ghost four to six cells off 62.7% of the time against the
   * 53.0% a uniform draw over the legal moves would reach, but only 55.5% at one to three cells
   * against the same 53.0%, and 1.5% with the ghost one cell away. It is careful at the distance
   * where being careful is cheap and fails where the course is decided. The information is there — a linear probe on the same
   * readout cells picks an escaping direction 97.0% of the time in those states — so this term
   * supplies the gradient the reward does not. It is a training signal only: it enters no
   * controller, and the finished game still chooses from circuit activity and learned weights.
   */
  avoidance: number;
  validateEvery: number;
}

export const DEFAULT_POLICY: PolicyConfig = {
  seed: 20260917,
  hidden: 0,
  updates: 4000,
  episodesPerUpdate: 32,
  learningRate: 0.02,
  discount: 0.99,
  temperature: 1,
  entropy: 0.01,
  shaping: 0,
  openness: 0,
  avoidance: 0,
  // Every 200 updates over 512 courses costs exactly what every 50 over 128 did, and buys a
  // champion that is chosen rather than drawn. At 128 courses the standard error of the score is
  // about 200, and taking the best of 240 such measurements selects a lucky draw: the checkpoint
  // recorded at 1584.8 scores 1275.6 ± 84.8 over 512 courses, the one recorded at 1973.5 scores
  // 1446.4 ± 87.1, and four runs continued from the latter failed to beat its recorded number
  // even once in twelve thousand updates. At 512 courses the standard error is about 87.
  validateEvery: 200,
  validationSeeds: Array.from({ length: 512 }, (_, i) => 1_100_001 + i),
  rules: DEFAULT_RULES,
  objective: OBJECTIVE,
};

export interface PolicyRecord {
  update: number;
  meanReturn: number;
  meanPellets: number;
  meanEntropy: number;
  validationScore: number;
  championUpdated: boolean;
  championValidation: number;
  elapsedMs: number;
}

export interface PolicyResult {
  champion: Float64Array;
  championUpdate: number;
  championValidation: number;
  initialParams: Float64Array;
  history: PolicyRecord[];
  config: PolicyConfig;
}

/**
 * The reward paid at the tick it is earned.
 *
 * Every term is a term of `episodeFitness`, placed on the step that produced it, so the
 * rewards of an episode sum to exactly the fitness the cross-entropy method optimises. The
 * objective is unchanged; only the moment it is delivered is. That is the whole point: a
 * game-long score says one thing about three hundred decisions, while this says something
 * about each of them.
 */
export function stepReward(pelletsEaten: number, ghostsEaten: number, after: GameState, objective: Objective = OBJECTIVE): number {
  let r = objective.pellet * pelletsEaten + objective.ghost * ghostsEaten + objective.perStep;
  if (after.cleared) r += objective.clear + objective.timeSaved * Math.max(0, after.rules.maxSteps - after.steps);
  if (!after.alive) r -= objective.caught;
  return r;
}

interface Transition {
  input: Float64Array;
  scores: Float64Array;
  action: number;
  advantage: number;
  /** Where the avoidance loss wants the probability mass, or null where it has nothing to say. */
  target: Float64Array | null;
}

/**
 * Safety of a state, as a number in [0, 1]: zero when a hunting ghost shares the cell, one when
 * none is within `threatRange`. Terminal states are zero by the convention shaping requires, and
 * so are states where every ghost is frightened, since there is then nothing to run from and the
 * potential has to return to zero before the course ends. A state no hunting ghost can reach at
 * all scores one.
 *
 * Distance alone is not safety, which is what the first attempt at teaching escapes got wrong:
 * a policy that always took the step gaining a cell reached 80.8% escapes at one to three cells
 * and still lost all hundred held-out courses, because the cell it gained was often a dead end.
 * `openness` mixes in how many ways out the cell has, weighted so that it matters only as the
 * ghost closes: at full distance the state is safe whatever its shape, and with a ghost on top
 * the only thing left is whether there is somewhere to go. Zero reproduces the distance-only
 * potential exactly.
 */
export function safetyPotential(s: GameState, openness = 0): number {
  if (s.done || s.frightened > 0) return 0;
  const d = s.ghostDist[cellIndex(s.maze, s.px, s.py)];
  if (d < 0) return 1;
  const distance = Math.min(d, s.rules.threatRange) / s.rules.threatRange;
  if (openness <= 0) return distance;
  let exits = 0;
  for (let dir = 0; dir < ACTIONS; dir++) if (isOpen(s.maze, s.px + DIRS[dir][0], s.py + DIRS[dir][1])) exits++;
  // A dead end scores zero and a four-way junction one; a plain corridor sits a third of the way.
  const ways = Math.max(0, exits - 1) / (ACTIONS - 1);
  return distance + openness * ways * (1 - distance);
}

/**
 * Spreads a unit of probability evenly over the legal moves that increase the distance to the
 * nearest hunting ghost, and reports whether such a move exists to teach.
 *
 * It says nothing in three cases: no ghost is hunting within `nearDistance`, the ghosts are
 * fleeing, or every legal move is equally good or equally bad. Teaching the last case would push
 * the policy towards a direction that buys nothing, which is how an auxiliary loss starts
 * competing with the pellets instead of complementing them.
 */
export function avoidanceTarget(s: GameState, out: Float64Array): boolean {
  const here = s.ghostDist[cellIndex(s.maze, s.px, s.py)];
  if (here < 1 || here > s.rules.nearDistance || s.frightened > 0) return false;
  out.fill(0);
  let legal = 0;
  let away = 0;
  for (let d = 0; d < ACTIONS; d++) {
    const nx = s.px + DIRS[d][0];
    const ny = s.py + DIRS[d][1];
    if (!isOpen(s.maze, nx, ny)) continue;
    legal++;
    const there = s.ghostDist[cellIndex(s.maze, nx, ny)];
    if (there >= 0 && there > here) {
      out[d] = 1;
      away++;
    }
  }
  if (away === 0 || away === legal) return false;
  for (let d = 0; d < ACTIONS; d++) out[d] /= away;
  return true;
}

/** Numerically stable softmax in place. */
function softmax(scores: Float64Array, temperature: number, out: Float64Array): void {
  let max = -Infinity;
  for (let a = 0; a < ACTIONS; a++) if (scores[a] > max) max = scores[a];
  let sum = 0;
  for (let a = 0; a < ACTIONS; a++) {
    out[a] = Math.exp((scores[a] - max) / temperature);
    sum += out[a];
  }
  for (let a = 0; a < ACTIONS; a++) out[a] /= sum;
}

/**
 * REINFORCE with a whitened return baseline, training the same readout on the same circuit.
 *
 * Experience is collected by sampling from a softmax over the four direction scores, so the
 * fly explores by itself rather than by being told what to try. Each step is credited with
 * the discounted reward that followed it, the batch's returns are centred and scaled, and
 * the readout is moved by Adam. Nothing here encodes a strategy: the only knowledge injected
 * is when a reward happened, which the game already decides.
 */
export function trainPolicy(
  circuit: Circuit,
  config: PolicyConfig = DEFAULT_POLICY,
  onUpdate?: (record: PolicyRecord, champion: Float64Array) => void,
  escape: EscapePathway | null = null,
  /** Parameters to start from, for continuing a finished run instead of searching from scratch. */
  initial: Float64Array | null = null,
): PolicyResult {
  const rng = new Rng(config.seed);
  const courseRng = new Rng((config.seed ^ 0x9e3779b9) >>> 0);
  const inputs = readoutInputCount(circuit.readoutCells.length);
  const size = paramCount(inputs, config.hidden);
  const params = randomParams(rng, 0.1, inputs, config.hidden);
  if (initial) {
    if (initial.length !== size) throw new Error(`initial parameters have length ${initial.length}, not ${size}`);
    params.set(initial);
  }
  const initialParams = params.slice();

  const grad = new Float64Array(size);
  const m = new Float64Array(size);
  const v = new Float64Array(size);
  const beta1 = 0.9;
  const beta2 = 0.999;
  const epsilon = 1e-8;

  // The same controller, at the same temperature, that the finished game will run: experience
  // is collected through the exact code path being trained.
  const controller = new ConnectomeController(circuit, params, false, escape, config.temperature);
  const probability = new Float64Array(ACTIONS);
  const dScores = new Float64Array(ACTIONS);
  const avoidBuffer = new Float64Array(ACTIONS);

  const validate = (p: Float64Array): number => {
    // Left at the controller's default temperature of 1, so validation samples the softmax the
    // same way the finished game does rather than taking an argmax the game never plays.
    const sampled = new ConnectomeController(circuit, p, false, escape);
    let total = 0;
    for (const seed of config.validationSeeds) total += episodeFitness(runEpisode(sampled, seed, config.rules), config.objective);
    return total / config.validationSeeds.length;
  };

  let champion = params.slice();
  let championValidation = validate(champion);
  let championUpdate = 0;
  const history: PolicyRecord[] = [];

  for (let update = 1; update <= config.updates; update++) {
    const started = Date.now();
    const batch: Transition[] = [];
    let returnSum = 0;
    let pelletSum = 0;
    let entropySum = 0;
    let entropyCount = 0;

    for (let e = 0; e < config.episodesPerUpdate; e++) {
      const seed = 1 + courseRng.int(900_000);
      const state = createGame(seed, config.rules);
      controller.reset(state);
      const steps: Array<{ input: Float64Array; scores: Float64Array; action: number; reward: number; target: Float64Array | null }> = [];
      while (!state.done) {
        // act() runs the circuit, fills `input` and `scores`, and draws the direction from its
        // own softmax, so the policy being improved is the policy that will be shipped.
        const action = controller.act(state);
        let entropy = 0;
        for (let a = 0; a < ACTIONS; a++) {
          const q = controller.probability[a];
          if (q > 0) entropy -= q * Math.log(q);
        }
        entropySum += entropy;
        entropyCount++;
        const input = Float64Array.from(controller.input);
        const scores = Float64Array.from(controller.scores);
        // Read before the move, because the label describes the choice that was just faced.
        const target = config.avoidance > 0 && avoidanceTarget(state, avoidBuffer) ? Float64Array.from(avoidBuffer) : null;
        const remainingBefore = state.remaining;
        const eatenBefore = state.ghostsEaten;
        const safetyBefore = config.shaping > 0 ? safetyPotential(state, config.openness) : 0;
        stepGame(state, action);
        let reward = stepReward(remainingBefore - state.remaining, state.ghostsEaten - eatenBefore, state, config.objective);
        // Shaping is added here rather than inside stepReward, so the fitness the champion is
        // selected on stays the objective the benchmark reports.
        if (config.shaping > 0) reward += config.shaping * (config.discount * safetyPotential(state, config.openness) - safetyBefore);
        steps.push({ input, scores, action, reward, target });
      }
      pelletSum += state.maze.pelletCount - state.remaining;

      let ret = 0;
      for (let t = steps.length - 1; t >= 0; t--) {
        ret = steps[t].reward + config.discount * ret;
        batch.push({ input: steps[t].input, scores: steps[t].scores, action: steps[t].action, advantage: ret, target: steps[t].target });
      }
      returnSum += steps.reduce((a, s) => a + s.reward, 0);
    }

    // Centre and scale the returns across the batch, so the update direction depends on which
    // decisions did better than the batch average rather than on the scale of the score.
    const mean = batch.reduce((a, b) => a + b.advantage, 0) / batch.length;
    let variance = 0;
    for (const t of batch) variance += (t.advantage - mean) ** 2;
    const sd = Math.sqrt(variance / batch.length) || 1;

    grad.fill(0);
    for (const t of batch) {
      softmax(t.scores, config.temperature, probability);
      const advantage = (t.advantage - mean) / sd;
      for (let a = 0; a < ACTIONS; a++) {
        const policyTerm = (probability[a] - (a === t.action ? 1 : 0)) * advantage;
        // Entropy bonus, maximised, so its gradient is subtracted from the loss gradient.
        let entropyTerm = 0;
        if (config.entropy > 0) {
          let sum = 0;
          for (let b = 0; b < ACTIONS; b++) {
            const logp = probability[b] > 0 ? Math.log(probability[b]) : 0;
            sum += probability[b] * logp * ((a === b ? 1 : 0) - probability[a]);
          }
          entropyTerm = config.entropy * sum;
        }
        // Cross-entropy against the escaping directions, minimised, so it pulls probability
        // towards them exactly as the policy term pulls it towards what paid off.
        const avoidTerm = t.target ? config.avoidance * (probability[a] - t.target[a]) : 0;
        dScores[a] = (policyTerm + entropyTerm + avoidTerm) / (config.temperature * batch.length);
      }
      readoutBackward(params, t.input, dScores, grad);
    }

    const correction1 = 1 - Math.pow(beta1, update);
    const correction2 = 1 - Math.pow(beta2, update);
    for (let i = 0; i < size; i++) {
      m[i] = beta1 * m[i] + (1 - beta1) * grad[i];
      v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i];
      params[i] -= (config.learningRate * (m[i] / correction1)) / (Math.sqrt(v[i] / correction2) + epsilon);
    }

    let validationScore = championValidation;
    let championUpdated = false;
    if (update % config.validateEvery === 0 || update === config.updates) {
      validationScore = validate(params);
      if (validationScore > championValidation) {
        champion = params.slice();
        championValidation = validationScore;
        championUpdate = update;
        championUpdated = true;
      }
    }
    const record: PolicyRecord = {
      update,
      meanReturn: returnSum / config.episodesPerUpdate,
      meanPellets: pelletSum / config.episodesPerUpdate,
      meanEntropy: entropySum / Math.max(1, entropyCount),
      validationScore,
      championUpdated,
      championValidation,
      elapsedMs: Date.now() - started,
    };
    history.push(record);
    onUpdate?.(record, champion);
  }

  return { champion, championUpdate, championValidation, initialParams, history, config };
}
