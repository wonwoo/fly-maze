import { DIRS, cellIndex, isOpen } from "./maze";
import type { GameState } from "./sim";

export const MODALITIES = ["food", "threat", "wall", "prey", "self"] as const;
export type Modality = (typeof MODALITIES)[number];
export const FEATURE_COUNT = MODALITIES.length * 4;
/** Ghost distance beyond which the prey channel reads zero. */
export const THREAT_RANGE = 12;

/**
 * Twenty engineered observations in [0, 1], one per (modality, direction):
 *   food[d]   = pellet direction signal. "absolute" is 1 - dist/foodRange, "relative" is
 *               1 - (dist - bestNeighbourDist)/foodContrast, "hybrid" averages the two
 *   threat[d] = danger signal; "absolute" is 1 - dist/threatRange, "relative" is
 *               1 - (dist - worstNeighbourDist)/threatContrast, so the most dangerous
 *               direction reads 1 and a clearly safer one reads 0
 *   wall[d]   = 1 when the neighbour cell is open to the player
 *   prey[d]   = closeness of the nearest frightened (edible) ghost, scaled by the
 *               fraction of frightened time remaining; zero when no ghost is frightened
 *   self[d]   = 1 for the direction the player is heading, 0 for its reverse, 0.5 either
 *               side, so the readout can tell continuing from turning back
 * Channel index = modality * 4 + direction. Blocked directions read zero on every channel.
 */
/**
 * Danger of one direction. The two encodings carry different things and neither carries both.
 *
 * "absolute" is how near the ghost is, which goes flat at zero once every direction is beyond
 * `threatRange`, so past that distance it says nothing about direction. "relative" is how much
 * worse this direction is than the best one, so the most dangerous direction reads 1 whether
 * the ghost is two steps away or fifty, and the channel carries no distance at all. "hybrid"
 * averages them, keeping the direction contrast while letting nearness raise the whole channel.
 */
function threatSignal(dist: number, worst: number, s: GameState): number {
  const { threatEncoding, threatRange, threatContrast } = s.rules;
  const absolute = 1 - Math.min(dist / threatRange, 1);
  const relative = Math.max(0, 1 - (dist - worst) / threatContrast);
  if (threatEncoding === "absolute") return absolute;
  if (threatEncoding === "relative") return relative;
  return 0.5 * absolute + 0.5 * relative;
}

function foodSignal(dist: number, best: number, s: GameState): number {
  const { foodEncoding, foodRange, foodContrast } = s.rules;
  const absolute = 1 - Math.min(dist / foodRange, 1);
  const relative = 1 - Math.min((dist - best) / foodContrast, 1);
  if (foodEncoding === "absolute") return absolute;
  if (foodEncoding === "relative") return relative;
  return 0.5 * absolute + 0.5 * relative;
}

export function computeFeatures(s: GameState, out: Float64Array = new Float64Array(FEATURE_COUNT)): Float64Array {
  const m = s.maze;
  const urgency = s.frightened > 0 ? s.frightened / s.rules.frightenedTicks : 0;
  let bestFood = Infinity;
  let worstGhost = Infinity;
  for (let d = 0; d < 4; d++) {
    const nx = s.px + DIRS[d][0];
    const ny = s.py + DIRS[d][1];
    if (!isOpen(m, nx, ny)) continue;
    const i = cellIndex(m, nx, ny);
    const food = s.pelletDist[i];
    if (food >= 0 && food < bestFood) bestFood = food;
    const ghost = s.ghostDist[i];
    if (ghost >= 0 && ghost < worstGhost) worstGhost = ghost;
  }
  for (let d = 0; d < 4; d++) {
    const nx = s.px + DIRS[d][0];
    const ny = s.py + DIRS[d][1];
    if (!isOpen(m, nx, ny)) {
      out[d] = 0;
      out[4 + d] = 0;
      out[8 + d] = 0;
      out[12 + d] = 0;
      out[16 + d] = 0;
      continue;
    }
    const i = cellIndex(m, nx, ny);
    const food = s.pelletDist[i];
    const ghost = s.ghostDist[i];
    const prey = s.preyDist[i];
    out[d] = food < 0 ? 0 : foodSignal(food, bestFood, s);
    out[4 + d] = ghost < 0 ? 0 : threatSignal(ghost, worstGhost, s);
    out[8 + d] = 1;
    out[12 + d] = prey < 0 ? 0 : urgency * (1 - Math.min(prey / THREAT_RANGE, 1));
    out[16 + d] = d === s.pdir ? 1 : d === (s.pdir + 2) % 4 ? 0 : 0.5;
  }
  return out;
}
