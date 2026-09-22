/** Tunable game rules. A checkpoint records the rules it was trained under. */
export interface GameRules {
  /** Per-ghost probability of taking the target-shortening step at a junction. */
  ghostChase: readonly number[];
  /** Tick at which each ghost first leaves the house. */
  ghostRelease: readonly number[];
  /** Ghosts rest on every Nth tick (0 = never rest, i.e. player speed). */
  ghostRestEvery: number;
  /** Ticks spent heading to a home corner, then ticks spent hunting, repeating. */
  scatterTicks: number;
  chaseTicks: number;
  /** Whether the six large pellets frighten the ghosts. Otherwise they score as plain pellets. */
  powerPellets: boolean;
  frightenedTicks: number;
  ghostEatScore: number;
  /** Ticks an eaten ghost waits in the house before returning. */
  respawnTicks: number;
  maxSteps: number;
  /** How the food channel encodes pellet direction (see src/game/features.ts). */
  foodEncoding: "absolute" | "relative" | "hybrid";
  /** Absolute mode: pellet distance beyond which the channel reads zero. */
  foodRange: number;
  /** Relative mode: extra path length versus the best direction at which the channel reads zero. */
  foodContrast: number;
  /** How the threat channel encodes ghost proximity, mirroring foodEncoding. */
  threatEncoding: "absolute" | "relative" | "hybrid";
  /** Absolute mode: ghost distance beyond which the threat channel reads zero. */
  threatRange: number;
  /** Relative mode: how many steps safer than the most dangerous direction still reads as threat. */
  threatContrast: number;
  /** Start the player on a seeded random open cell instead of the fixed start, so courses differ from tick 1. */
  randomStart: boolean;
  /** Minimum ghost-walkable distance from the ghost house to a candidate start cell. */
  startGhostDistance: number;
  /** Maximum seeded delay added to each ghost's release tick. */
  releaseJitter: number;
  /** Generate a fresh maze per course instead of using the built-in layout. */
  randomMaze: boolean;
  /**
   * What the four extra readout inputs carry. "pathway" runs the measured looming-to-giant-
   * fibre circuit; "proximity" replaces it with a per-direction binary gate, 1 when a hunting
   * ghost is within `nearDistance` of that neighbour. A linear readout cannot form the product
   * of two channels, so a condition it is meant to act on has to arrive already thresholded.
   */
  escapeChannels: "pathway" | "proximity";
  /** Gate distance for the "proximity" escape channels. */
  nearDistance: number;
}

export const DEFAULT_RULES: GameRules = {
  ghostChase: [0.8, 0.6, 0.4],
  ghostRelease: [0, 50, 100],
  ghostRestEvery: 4,
  scatterTicks: 40,
  chaseTicks: 80,
  powerPellets: true,
  frightenedTicks: 40,
  ghostEatScore: 200,
  respawnTicks: 30,
  maxSteps: 1000,
  foodEncoding: "relative",
  foodRange: 40,
  foodContrast: 3,
  threatEncoding: "relative",
  threatRange: 12,
  threatContrast: 3,
  randomStart: true,
  startGhostDistance: 8,
  releaseJitter: 30,
  randomMaze: true,
  escapeChannels: "proximity",
  nearDistance: 3,
};
