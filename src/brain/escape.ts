import type { Circuit } from "./circuit";
import { DIRS, cellIndex, isOpen } from "../game/maze";
import type { GameState } from "../game/sim";

/** Samples per axis of the drive plane. The response is smooth, so this is plenty. */
const RESOLUTION = 65;

/**
 * The measured looming-escape pathway, driven as a pair of eyes.
 *
 * In the fly, the two hemispheres see different halves of the world, and the left and right
 * giant-fibre cells are driven by their own side's looming population. The measured drive onto
 * the giant fibres keeps that separation: of the 284 retained edges from looming cells onto the
 * two DNp01 cells, 155 run left to left and 129 right to right, and none cross. The graph around
 * them does cross, with 14,468 of its 74,697 edges running between the hemispheres, so what is
 * separate is the looming-to-giant-fibre drive rather than the circuit as a whole.
 *
 * What that separation buys is narrow, and the comment here used to claim more than it should.
 * Those crossing edges reach the giant fibres with almost nothing: raising the opposite drive
 * from 0 to 1 moves each one by 0.006 to 0.013 across the whole drive plane. So each giant fibre
 * is a function of its own side in all but the third decimal, and the pathway applies a saturating
 * squash to each side and little else. Which side the threat is on is decided by
 * `directionThreat` and the heading rotation in the controller, not here. Measured against
 * replacements on the hundred held-out courses, the pathway is worth little: 112.55 pellets
 * with it, 110.27 with the channels held at zero, 111.84 with its edges shuffled.
 *
 * It is kept because it is the only place the readout learns absolute distance from. The
 * `threat` feature channel is relative, so the most dangerous direction reads 1 whether the
 * ghost is two steps away or fifty; a least-squares probe recovers ghost distance from those
 * four channels at R2 -0.001 and from these four at R2 0.181 (scripts/probe.ts).
 */
export class EscapePathway {
  private readonly leftLooming: number[] = [];
  private readonly rightLooming: number[] = [];
  private leftGiantFiber = -1;
  private rightGiantFiber = -1;
  private readonly drive: Float64Array;
  /** Left and right giant-fibre activity sampled over the whole drive plane, measured once. */
  private readonly table: Float64Array;
  /** Left and right giant-fibre activity from the most recent evaluation. */
  readonly giantFiber = new Float64Array(2);

  constructor(private readonly circuit: Circuit) {
    circuit.data.cells.forEach((c, i) => {
      if (c.modality === "threat") {
        if (c.side === "L") this.leftLooming.push(i);
        else if (c.side === "R") this.rightLooming.push(i);
      }
      if (c.giantFiber) {
        if (c.side === "L") this.leftGiantFiber = i;
        else if (c.side === "R") this.rightGiantFiber = i;
      }
    });
    if (this.leftLooming.length === 0 || this.rightLooming.length === 0) {
      throw new Error("pathway circuit lacks a looming population on both sides");
    }
    if (this.leftGiantFiber < 0 || this.rightGiantFiber < 0) {
      throw new Error("pathway circuit lacks a giant-fibre cell on both sides");
    }
    this.drive = new Float64Array(circuit.size);
    // The pathway is reset before each evaluation, so its output depends only on the pair
    // of drive levels. Sampling that plane once turns every later evaluation into a table
    // read with the same numbers, which keeps training fast.
    this.table = new Float64Array(RESOLUTION * RESOLUTION * 2);
    for (let i = 0; i < RESOLUTION; i++) {
      for (let j = 0; j < RESOLUTION; j++) {
        const [l, r] = this.propagate(i / (RESOLUTION - 1), j / (RESOLUTION - 1));
        this.table[(i * RESOLUTION + j) * 2] = l;
        this.table[(i * RESOLUTION + j) * 2 + 1] = r;
      }
    }
  }

  /**
   * Drives the two looming populations and returns the left and right giant-fibre activity.
   * `left` and `right` are the looming strengths in each half of the player's view.
   */
  evaluate(left: number, right: number): Float64Array {
    const x = Math.max(0, Math.min(1, left)) * (RESOLUTION - 1);
    const y = Math.max(0, Math.min(1, right)) * (RESOLUTION - 1);
    const i = Math.min(RESOLUTION - 2, Math.floor(x));
    const j = Math.min(RESOLUTION - 2, Math.floor(y));
    const tx = x - i;
    const ty = y - j;
    for (let k = 0; k < 2; k++) {
      const a = this.table[(i * RESOLUTION + j) * 2 + k];
      const b = this.table[(i * RESOLUTION + j + 1) * 2 + k];
      const c = this.table[((i + 1) * RESOLUTION + j) * 2 + k];
      const d = this.table[((i + 1) * RESOLUTION + j + 1) * 2 + k];
      this.giantFiber[k] = a * (1 - tx) * (1 - ty) + b * (1 - tx) * ty + c * tx * (1 - ty) + d * tx * ty;
    }
    return this.giantFiber;
  }

  /** Propagates one pair of drive levels through the measured pathway. */
  private propagate(left: number, right: number): [number, number] {
    this.circuit.reset();
    this.drive.fill(0);
    for (const cell of this.leftLooming) this.drive[cell] = left;
    for (const cell of this.rightLooming) this.drive[cell] = right;
    this.circuit.stepWithDrive(this.drive);
    return [this.circuit.h[this.leftGiantFiber], this.circuit.h[this.rightGiantFiber]];
  }
}

/**
 * Looming strength of each direction, as optical expansion: it grows as a threat closes in
 * and is strongest for the direction that is worst relative to the alternatives.
 */
export function directionThreat(s: GameState, out: Float64Array = new Float64Array(4), passable: Uint8Array = new Uint8Array(4)): Float64Array {
  let worst = Infinity;
  for (let d = 0; d < 4; d++) {
    const nx = s.px + DIRS[d][0];
    const ny = s.py + DIRS[d][1];
    passable[d] = isOpen(s.maze, nx, ny) ? 1 : 0;
    if (!passable[d]) continue;
    const dist = s.ghostDist[cellIndex(s.maze, nx, ny)];
    if (dist >= 0 && dist < worst) worst = dist;
  }
  const { threatRange, threatContrast } = s.rules;
  for (let d = 0; d < 4; d++) {
    if (!passable[d] || s.frightened > 0) {
      out[d] = 0;
      continue;
    }
    const dist = s.ghostDist[cellIndex(s.maze, s.px + DIRS[d][0], s.py + DIRS[d][1])];
    if (dist < 0) {
      out[d] = 0;
      continue;
    }
    const nearness = 1 - Math.min(dist / threatRange, 1);
    const relative = Math.max(0, 1 - (dist - worst) / threatContrast);
    out[d] = nearness * relative;
  }
  return out;
}
