import type { Rng } from "../game/rng";

/** Default hidden width. Zero means a direct linear map from inputs to direction scores. */
export const HIDDEN = 12;
export const ACTIONS = 4;

/**
 * Extra readout inputs carrying the measured escape pathway's answer, one per direction.
 * They sit after the readout-cell activities in the input vector.
 */
export const ESCAPE_INPUTS = ACTIONS;

/** Length of the vector the readout consumes: every readout cell plus the escape channels. */
export function readoutInputCount(readoutCells: number): number {
  return readoutCells + ESCAPE_INPUTS;
}

/**
 * Parameter count of a readout over an input vector of length `inputs`.
 * With a hidden layer: W1, b1, W2, b2. With `hidden` zero: W, b alone.
 */
export function paramCount(inputs: number, hidden: number = HIDDEN): number {
  if (hidden === 0) return inputs * ACTIONS + ACTIONS;
  return inputs * hidden + hidden + hidden * ACTIONS + ACTIONS;
}

/**
 * Recovers the hidden width a parameter vector was built for, so a checkpoint carries its
 * own shape in its length and every caller keeps passing the parameters alone.
 */
export function hiddenWidth(paramsLength: number, inputs: number): number {
  if (paramsLength === paramCount(inputs, 0)) return 0;
  const h = (paramsLength - ACTIONS) / (inputs + ACTIONS + 1);
  if (!Number.isInteger(h) || h <= 0) throw new Error(`${paramsLength} parameters do not fit a readout over ${inputs} inputs`);
  return h;
}

/**
 * Trainable readout, in the shape the parameter vector's length implies.
 *
 * Width zero, which is what the published checkpoint uses, is a direct linear map from the N
 * inputs to the 4 direction scores, laid out as W[a*N+i] then b[a]. A nonzero width inserts a
 * tanh hidden layer, laid out as W1[h*N+i], b1[h], W2[a*width+h], b2[a].
 */
export function readoutForward(params: Float64Array, x: Float64Array, out: Float64Array = new Float64Array(ACTIONS)): Float64Array {
  const inputs = x.length;
  const width = hiddenWidth(params.length, inputs);
  if (width === 0) {
    const b = inputs * ACTIONS;
    for (let a = 0; a < ACTIONS; a++) {
      let sum = params[b + a];
      for (let i = 0; i < inputs; i++) sum += params[a * inputs + i] * x[i];
      out[a] = sum;
    }
    return out;
  }
  const hidden = new Float64Array(width);
  const b1 = inputs * width;
  const w2 = b1 + width;
  const b2 = w2 + width * ACTIONS;
  for (let h = 0; h < width; h++) {
    let sum = params[b1 + h];
    for (let i = 0; i < inputs; i++) sum += params[h * inputs + i] * x[i];
    hidden[h] = Math.tanh(sum);
  }
  for (let a = 0; a < ACTIONS; a++) {
    let sum = params[b2 + a];
    for (let h = 0; h < width; h++) sum += params[w2 + a * width + h] * hidden[h];
    out[a] = sum;
  }
  return out;
}

export function argmax(v: Float64Array): number {
  let best = 0;
  for (let i = 1; i < v.length; i++) if (v[i] > v[best]) best = i;
  return best;
}

export function randomParams(rng: Rng, sigma: number, inputs: number, hidden: number = HIDDEN): Float64Array {
  const n = paramCount(inputs, hidden);
  const p = new Float64Array(n);
  for (let i = 0; i < n; i++) p[i] = sigma * rng.gaussian();
  return p;
}

/**
 * Accumulates the gradient of a scalar loss with respect to the readout parameters.
 *
 * `dScores` is the gradient of that loss with respect to the four direction scores; the
 * chain rule back through the readout is written out here because the readout is the only
 * trained part of the system and nothing else needs autodiff.
 */
export function readoutBackward(params: Float64Array, x: Float64Array, dScores: Float64Array, grad: Float64Array): void {
  const inputs = x.length;
  const width = hiddenWidth(params.length, inputs);
  if (width === 0) {
    const b = inputs * ACTIONS;
    for (let a = 0; a < ACTIONS; a++) {
      const d = dScores[a];
      if (d === 0) continue;
      for (let i = 0; i < inputs; i++) grad[a * inputs + i] += d * x[i];
      grad[b + a] += d;
    }
    return;
  }
  const b1 = inputs * width;
  const w2 = b1 + width;
  const b2 = w2 + width * ACTIONS;
  const hidden = new Float64Array(width);
  for (let h = 0; h < width; h++) {
    let sum = params[b1 + h];
    for (let i = 0; i < inputs; i++) sum += params[h * inputs + i] * x[i];
    hidden[h] = Math.tanh(sum);
  }
  const dHidden = new Float64Array(width);
  for (let a = 0; a < ACTIONS; a++) {
    const d = dScores[a];
    grad[b2 + a] += d;
    if (d === 0) continue;
    for (let h = 0; h < width; h++) {
      grad[w2 + a * width + h] += d * hidden[h];
      dHidden[h] += d * params[w2 + a * width + h];
    }
  }
  for (let h = 0; h < width; h++) {
    const dz = dHidden[h] * (1 - hidden[h] * hidden[h]);
    if (dz === 0) continue;
    for (let i = 0; i < inputs; i++) grad[h * inputs + i] += dz * x[i];
    grad[b1 + h] += dz;
  }
}
