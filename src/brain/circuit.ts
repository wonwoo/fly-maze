import { DIR_NAMES } from "../game/maze";
import { FEATURE_COUNT, MODALITIES } from "../game/features";

export interface CircuitCell {
  bodyId: number;
  type: string | null;
  class: string | null;
  superclass: string | null;
  side: string | null;
  soma: number[] | null;
  nt: string;
  sign: number;
  role: "input" | "bridge" | "readout";
  modality: "food" | "threat" | "wall" | "prey" | "self" | null;
  /** Pathway circuits mark the giant-fibre descending cells. */
  giantFiber?: boolean;
  channel: "up" | "right" | "down" | "left" | null;
}

export interface CircuitData {
  source: string;
  selection: Record<string, string>;
  counts: Record<string, unknown>;
  reachability: Record<string, number>;
  channels?: string[];
  cells: CircuitCell[];
  edges: Array<[number, number, number]>;
}

/** Leak retained from the previous state at every synchronous iteration. */
export const LEAK = 0.3;
/** Gain on the normalized recurrent input. */
export const GAIN = 1.4;
/** Synchronous iterations per game decision. */
export const ITERATIONS = 3;
/** Scale applied to readout-cell activity before the trained readout. */
export const READOUT_GAIN = 4;

/**
 * Fixed-wiring rate model of the extracted MaleCNS circuit.
 *
 *   W[j,i]     = c[j,i] * s[j] / sum_k c[k,i] * |s[k]|
 *   h_new[i]   = LEAK * h[i] + (1 - LEAK) * tanh(u[i] + GAIN * sum_j W[j,i] * h[j])
 *
 * c is the measured contact count, s the presynaptic transmitter sign
 * (acetylcholine +1, GABA and glutamate -1, anything else 0). u is the external
 * drive 2 * (feature - 0.5) on driven input cells and zero elsewhere. The state is
 * dimensionless; it is neither a firing rate nor a membrane voltage.
 */
export class Circuit {
  readonly size: number;
  readonly h: Float64Array;
  readonly readout: Float64Array;
  readonly channelCells: number[][];
  readonly readoutCells: number[];
  private readonly next: Float64Array;
  private readonly drive: Float64Array;
  private readonly rowStart: Int32Array;
  private readonly preIndex: Int32Array;
  private readonly weight: Float64Array;

  constructor(readonly data: CircuitData) {
    const n = data.cells.length;
    this.size = n;
    this.h = new Float64Array(n);
    this.next = new Float64Array(n);
    this.drive = new Float64Array(n);

    const sign = data.cells.map((c) => c.sign);
    const denom = new Float64Array(n);
    const count = new Int32Array(n + 1);
    for (const [pre, post, c] of data.edges) {
      denom[post] += c * Math.abs(sign[pre]);
      count[post + 1]++;
    }
    this.rowStart = new Int32Array(n + 1);
    for (let i = 0; i < n; i++) this.rowStart[i + 1] = this.rowStart[i] + count[i + 1];
    this.preIndex = new Int32Array(data.edges.length);
    this.weight = new Float64Array(data.edges.length);
    const fill = this.rowStart.slice(0, n);
    for (const [pre, post, c] of data.edges) {
      const k = fill[post]++;
      this.preIndex[k] = pre;
      this.weight[k] = denom[post] > 0 ? (c * sign[pre]) / denom[post] : 0;
    }

    // A pathway circuit has no per-direction channels; it is driven cell-by-cell instead.
    this.channelCells = [];
    if (data.channels) {
      if (data.channels.join() !== DIR_NAMES.join()) {
        throw new Error(`circuit channel order ${data.channels.join()} does not match ${DIR_NAMES.join()}`);
      }
      for (const modality of MODALITIES) {
        for (const dir of DIR_NAMES) {
          const cells: number[] = [];
          data.cells.forEach((c, i) => {
            if (c.role === "input" && c.modality === modality && c.channel === dir) cells.push(i);
          });
          if (cells.length === 0) throw new Error(`no input cells for channel ${modality}/${dir}`);
          this.channelCells.push(cells);
        }
      }
      if (this.channelCells.length !== FEATURE_COUNT) throw new Error("channel count mismatch");
    }
    this.readoutCells = [];
    data.cells.forEach((c, i) => {
      if (c.role === "readout") this.readoutCells.push(i);
    });
    this.readout = new Float64Array(this.readoutCells.length);
  }

  reset(): void {
    this.h.fill(0);
    this.readout.fill(0);
  }

  /** Signed normalized weight from cell `pre` onto cell `post`, or 0 when no edge exists. */
  weightBetween(pre: number, post: number): number {
    for (let k = this.rowStart[post]; k < this.rowStart[post + 1]; k++) {
      if (this.preIndex[k] === pre) return this.weight[k];
    }
    return 0;
  }

  /** Number of (modality, direction) channels this circuit is driven through; 0 for a pathway circuit. */
  get channelCount(): number {
    return this.channelCells.length;
  }

  /**
   * Run ITERATIONS synchronous updates from an explicit per-cell external drive.
   * Used to stimulate a named population directly, without the game's channel encoder.
   */
  stepWithDrive(drive: Float64Array): void {
    const { h, next, rowStart, preIndex, weight, size } = this;
    for (let it = 0; it < ITERATIONS; it++) {
      for (let i = 0; i < size; i++) {
        let sum = 0;
        for (let k = rowStart[i]; k < rowStart[i + 1]; k++) sum += weight[k] * h[preIndex[k]];
        next[i] = LEAK * h[i] + (1 - LEAK) * Math.tanh(drive[i] + GAIN * sum);
      }
      h.set(next);
    }
  }

  /** Run ITERATIONS synchronous updates and return READOUT_GAIN * h on the readout cells. */
  step(features: Float64Array, silenced = false): Float64Array {
    if (silenced) {
      this.h.fill(0);
      this.readout.fill(0);
      return this.readout;
    }
    const { drive, h, next, rowStart, preIndex, weight, size } = this;
    drive.fill(0);
    for (let ch = 0; ch < FEATURE_COUNT; ch++) {
      const u = 2 * (features[ch] - 0.5);
      for (const cell of this.channelCells[ch]) drive[cell] = u;
    }
    for (let it = 0; it < ITERATIONS; it++) {
      for (let i = 0; i < size; i++) {
        let sum = 0;
        for (let k = rowStart[i]; k < rowStart[i + 1]; k++) sum += weight[k] * h[preIndex[k]];
        next[i] = LEAK * h[i] + (1 - LEAK) * Math.tanh(drive[i] + GAIN * sum);
      }
      h.set(next);
    }
    for (let r = 0; r < this.readoutCells.length; r++) this.readout[r] = READOUT_GAIN * h[this.readoutCells[r]];
    return this.readout;
  }
}
