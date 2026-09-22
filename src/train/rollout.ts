import type { Controller } from "../brain/controller";
import { DEFAULT_RULES, type GameRules } from "../game/rules";
import { createGame, stepGame, type GameState } from "../game/sim";

export interface EpisodeResult {
  seed: number;
  score: number;
  steps: number;
  maxSteps: number;
  pellets: number;
  ghostsEaten: number;
  cleared: boolean;
  alive: boolean;
}

export function runEpisode(controller: Controller, seed: number, rules: GameRules = DEFAULT_RULES, onStep?: (s: GameState) => void): EpisodeResult {
  const s = createGame(seed, rules);
  controller.reset(s);
  while (!s.done) {
    stepGame(s, controller.act(s));
    onStep?.(s);
  }
  return { seed, score: s.score, steps: s.steps, maxSteps: rules.maxSteps, pellets: s.maze.pelletCount - s.remaining, ghostsEaten: s.ghostsEaten, cleared: s.cleared, alive: s.alive };
}

export interface EvaluationSummary {
  controller: string;
  courses: number;
  cleared: number;
  meanScore: number;
  meanSteps: number;
  meanPellets: number;
  results: EpisodeResult[];
}

export function evaluate(controller: Controller, seeds: readonly number[], rules: GameRules = DEFAULT_RULES): EvaluationSummary {
  const results = seeds.map((seed) => runEpisode(controller, seed, rules));
  const mean = (f: (r: EpisodeResult) => number) => results.reduce((a, r) => a + f(r), 0) / results.length;
  return {
    controller: controller.name,
    courses: results.length,
    cleared: results.filter((r) => r.cleared).length,
    meanScore: mean((r) => r.score),
    meanSteps: mean((r) => r.steps),
    meanPellets: mean((r) => r.pellets),
    results,
  };
}

/**
 * What training optimises. The game keeps its arcade score; this is the separate objective,
 * and it states the goal of the game rather than any strategy for reaching it: clear the maze,
 * stay alive, and do not dawdle. Nothing here tells the readout how to behave.
 *
 * `perStep` is the price of one tick. It is positive in the shape this project first shipped,
 * which turned out to say the opposite of what was meant: a wasted step earned its reward and
 * cost nothing, while the reward for finishing early was only ever paid on a cleared maze, and
 * mazes were cleared five times in a hundred. A checkpoint records the objective it trained
 * under, so runs made under different wordings of the goal stay distinguishable.
 */
export interface Objective {
  /** Per pellet, matching the arcade score so collection stays the base signal. */
  pellet: number;
  /** Per frightened ghost eaten. */
  ghost: number;
  /** Paid once for emptying the maze. */
  clear: number;
  /** Per tick left on the clock when the maze is cleared, so finishing sooner is better. */
  timeSaved: number;
  /** Paid every tick. Positive rewards staying alive; negative charges for time spent. */
  perStep: number;
  /** Charged once when a ghost catches the player, making death a real loss. */
  caught: number;
}

export const OBJECTIVE: Objective = {
  pellet: 10,
  ghost: 200,
  clear: 3000,
  timeSaved: 2,
  perStep: 0.1,
  caught: 500,
};

export function episodeFitness(r: EpisodeResult, objective: Objective = OBJECTIVE): number {
  const timeBonus = r.cleared ? objective.timeSaved * Math.max(0, r.maxSteps - r.steps) : 0;
  return (
    objective.pellet * r.pellets +
    objective.ghost * r.ghostsEaten +
    (r.cleared ? objective.clear : 0) +
    timeBonus +
    objective.perStep * r.steps -
    (r.alive ? 0 : objective.caught)
  );
}

export function heldOutSeeds(count = 300, first = 2_100_001): number[] {
  return Array.from({ length: count }, (_, i) => first + i);
}
