import { computeFeatures, FEATURE_COUNT } from "../game/features";
import { EscapePathway, directionThreat } from "./escape";
import { DIRS, cellIndex, isOpen } from "../game/maze";
import { Rng } from "../game/rng";
import type { GameState } from "../game/sim";
import { READOUT_GAIN, type Circuit } from "./circuit";
import { ACTIONS, argmax, readoutForward, readoutInputCount } from "./readout";

/**
 * Softmax temperature the finished game and every published measurement run at.
 *
 * It is a deployment choice rather than a trained parameter, so it is picked on validation
 * courses by `scripts/tune-temperature.ts` and recorded in `public/benchmarks/temperature.json`;
 * `npm run check:assets` fails when that sweep no longer names this value or the checkpoint that
 * ships. The constructor default below stays at 1 because that is the temperature experience is
 * collected at while training, and lowering it here would silently change both trainers and the
 * reward sums `tests/policy.test.ts` pins.
 */
export const DEPLOY_TEMPERATURE = 0.1;

export interface Controller {
  readonly name: string;
  /** Called once at the start of every episode with the fresh game state. */
  reset(state: GameState): void;
  act(state: GameState): number;
}

/**
 * Engineered features -> fixed circuit -> trained readout -> direction.
 *
 * The 20 engineered channels drive the main circuit, whose 32 readout cells report what it
 * made of them. Four more inputs follow, and `rules.escapeChannels` decides what fills them.
 * The published checkpoint sets it to "proximity", so they carry the binary gate documented
 * at `proximityGate` below and the measured escape pathway is not in the loop; the rest of
 * this comment describes "pathway", the alternative the gate replaced.
 *
 * Under "pathway", two measured circuits feed the same readout. Separately, the looming seen in
 * the player's left and right visual halves drives the left and right looming populations of
 * the measured escape pathway, and the two giant-fibre cells answer.
 *
 * Those two answers are written back into the same four-direction frame the other channels
 * use: the left giant fibre lands on the direction lying to the player's left, the right one
 * on the direction to its right, and their common mode on the heading, which is the part of
 * the drive both sides shared. Nothing behind the fly reaches its eyes, so that channel is
 * zero. The readout then sees 36 numbers and decides on its own how much weight the escape
 * answer deserves; it is not overruled, so avoiding and collecting are learned together
 * rather than fighting each other.
 *
 * The pathway itself contributes little (see src/brain/escape.ts): it squashes each side
 * independently, and the side a threat is on is worked out by `directionThreat` and the
 * rotation below, not by the wiring. What these four channels do carry, and no other input
 * does, is how near the nearest ghost actually is.
 *
 * The direction is drawn from a softmax over the four scores rather than taken as the highest
 * one, at `temperature`. Zero restores the argmax. The shipped value is `DEPLOY_TEMPERATURE`
 * above, chosen on validation courses; every controller in the published benchmark, controls
 * included, runs at that same temperature so the rows differ only in their wiring and parameters.
 */
export class ConnectomeController implements Controller {
  readonly features = new Float64Array(FEATURE_COUNT);
  readonly scores = new Float64Array(ACTIONS);
  readonly threat = new Float64Array(ACTIONS);
  /** Readout-cell activities followed by the four escape channels; what the readout consumes. */
  readonly input: Float64Array;
  /** Softmax probability of each direction on the last decision. */
  readonly probability = new Float64Array(ACTIONS);
  private readonly passable = new Uint8Array(ACTIONS);
  private readonly escapeBase: number;
  private rng = new Rng(1);
  lastAction = -1;
  lastEscape: Float64Array | null = null;

  constructor(
    readonly circuit: Circuit,
    public params: Float64Array,
    public silenced = false,
    public escape: EscapePathway | null = null,
    /** Softmax temperature; zero takes the highest-scoring direction instead of sampling. */
    public temperature = 1,
  ) {
    this.escapeBase = circuit.readoutCells.length;
    this.input = new Float64Array(readoutInputCount(this.escapeBase));
  }

  get name(): string {
    return this.silenced ? "connectome-silenced" : "connectome";
  }

  reset(state?: GameState): void {
    this.circuit.reset();
    this.scores.fill(0);
    this.input.fill(0);
    this.probability.fill(0);
    this.lastAction = -1;
    // Seeded from the course, so a sampled run is as reproducible as a deterministic one.
    this.rng = new Rng(((state?.seed ?? 1) * 2654435761) >>> 0 || 1);
  }

  act(state: GameState): number {
    computeFeatures(state, this.features);
    const readout = this.circuit.step(this.features, this.silenced);
    this.input.set(readout, 0);
    this.input.fill(0, this.escapeBase);
    if (state.rules.escapeChannels === "proximity") {
      if (!this.silenced) this.proximityGate(state);
      this.lastEscape = null;
    } else if (this.escape && !this.silenced) {
      directionThreat(state, this.threat, this.passable);
      // Split the four directions into the player's two visual halves, relative to its
      // heading: what lies to its left drives the left eye, what lies right drives the right.
      // A threat straight ahead is seen by both, which is what makes the common mode mean
      // "closing head-on" rather than "off to one side".
      const heading = state.pdir;
      const leftDir = (heading + 3) % ACTIONS;
      const rightDir = (heading + 1) % ACTIONS;
      const leftDrive = Math.max(this.threat[leftDir], this.threat[heading] * 0.5);
      const rightDrive = Math.max(this.threat[rightDir], this.threat[heading] * 0.5);
      const gf = this.escape.evaluate(leftDrive, rightDrive);
      this.input[this.escapeBase + leftDir] = READOUT_GAIN * gf[0];
      this.input[this.escapeBase + rightDir] = READOUT_GAIN * gf[1];
      this.input[this.escapeBase + heading] = READOUT_GAIN * 0.5 * (gf[0] + gf[1]);
      this.lastEscape = gf;
    } else {
      this.lastEscape = null;
    }
    readoutForward(this.params, this.input, this.scores);
    this.lastAction = this.temperature > 0 ? this.sample() : argmax(this.scores);
    return this.lastAction;
  }

  /** One binary channel per direction: is a hunting ghost already this close that way? */
  private proximityGate(state: GameState): void {
    const { maze, rules } = state;
    for (let d = 0; d < ACTIONS; d++) {
      const nx = state.px + DIRS[d][0];
      const ny = state.py + DIRS[d][1];
      if (!isOpen(maze, nx, ny) || state.frightened > 0) continue;
      const dist = state.ghostDist[cellIndex(maze, nx, ny)];
      if (dist >= 0 && dist <= rules.nearDistance) this.input[this.escapeBase + d] = READOUT_GAIN;
    }
  }

  /** Draws a direction from the softmax over the scores and records the distribution. */
  private sample(): number {
    let max = -Infinity;
    for (let a = 0; a < ACTIONS; a++) if (this.scores[a] > max) max = this.scores[a];
    let sum = 0;
    for (let a = 0; a < ACTIONS; a++) {
      this.probability[a] = Math.exp((this.scores[a] - max) / this.temperature);
      sum += this.probability[a];
    }
    for (let a = 0; a < ACTIONS; a++) this.probability[a] /= sum;
    let pick = this.rng.next();
    for (let a = 0; a < ACTIONS; a++) {
      pick -= this.probability[a];
      if (pick <= 0) return a;
    }
    return ACTIONS - 1;
  }
}

/** Uniform random directions, seeded from the episode seed so runs are reproducible. */
export class RandomController implements Controller {
  readonly name = "random";
  private rng = new Rng(1);
  reset(state: GameState): void {
    this.rng = new Rng((state.seed * 2654435761) >>> 0 || 1);
  }
  act(): number {
    return this.rng.int(ACTIONS);
  }
}

/** Handwritten baseline: step toward the nearest pellet, ignoring ghosts. */
export class GreedyController implements Controller {
  readonly name = "greedy-pellet";
  reset(): void {}
  act(s: GameState): number {
    let best = s.pdir;
    let bestDist = Infinity;
    for (let d = 0; d < 4; d++) {
      const nx = s.px + DIRS[d][0];
      const ny = s.py + DIRS[d][1];
      if (!isOpen(s.maze, nx, ny)) continue;
      const dist = s.pelletDist[cellIndex(s.maze, nx, ny)];
      if (dist >= 0 && dist < bestDist) {
        bestDist = dist;
        best = d;
      }
    }
    return best;
  }
}

/** No input: the player keeps its current heading until a wall stops it. */
export class IdleController implements Controller {
  readonly name = "idle";
  reset(): void {}
  act(s: GameState): number {
    return s.pdir;
  }
}
