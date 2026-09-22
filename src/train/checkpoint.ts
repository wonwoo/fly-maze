import { paramCount } from "../brain/readout";
import { DEFAULT_RULES, type GameRules } from "../game/rules";
import { OBJECTIVE, type Objective } from "./rollout";
/** What every trainer must record, whatever method it used, for its readout to be re-runnable. */
export interface TrainingConfig {
  seed: number;
  /** Hidden width of the readout; zero is a direct linear map. */
  hidden: number;
  validationSeeds: number[];
  rules: GameRules;
  /** The wording of the goal this readout was trained against. */
  objective: Objective;
}

export interface Checkpoint {
  version: 1;
  params: number[];
  /** Generation for the cross-entropy method, update for the policy gradient. */
  championGeneration: number;
  championValidation: number;
  config: TrainingConfig;
  circuitSha256: string;
}

function checkRules(rules: unknown): asserts rules is GameRules {
  if (!rules || typeof rules !== "object") throw new Error("checkpoint has no game rules");
  const r = rules as Record<string, unknown>;
  for (const key of Object.keys(DEFAULT_RULES) as Array<keyof GameRules>) {
    if (!(key in r)) throw new Error(`checkpoint rules are missing "${key}"`);
  }
  if (!Array.isArray(r.ghostChase) || r.ghostChase.length === 0 || !r.ghostChase.every((v) => typeof v === "number" && v >= 0 && v <= 1)) {
    throw new Error("checkpoint rules have an invalid ghostChase list");
  }
  if (!Array.isArray(r.ghostRelease) || r.ghostRelease.length === 0 || !r.ghostRelease.every((v) => typeof v === "number" && v >= 0)) {
    throw new Error("checkpoint rules have an invalid ghostRelease list");
  }
  if (typeof r.maxSteps !== "number" || r.maxSteps < 1) throw new Error("checkpoint rules have an invalid maxSteps");
  if (!["absolute", "relative", "hybrid"].includes(String(r.foodEncoding))) throw new Error("checkpoint rules have an unknown foodEncoding");
  if (!["absolute", "relative", "hybrid"].includes(String(r.threatEncoding))) throw new Error("checkpoint rules have an unknown threatEncoding");
}

/**
 * Validate a checkpoint before it can drive the game.
 *
 * `readoutInputs` is the number of readout cells in the circuit that will run it, so a
 * checkpoint trained against a different circuit is rejected by parameter count. When
 * `circuitSha256` is supplied, the checkpoint must also name that exact circuit: the
 * readout is only meaningful for the wiring it was trained on.
 */
export function parseCheckpoint(raw: unknown, readoutInputs: number, circuitSha256?: string): Checkpoint {
  if (!raw || typeof raw !== "object") throw new Error("checkpoint is not an object");
  const c = raw as Checkpoint;
  if (c.version !== 1) throw new Error(`unsupported checkpoint version ${String(c.version)}`);
  const expected = paramCount(readoutInputs, c.config?.hidden ?? 12);
  if (!Array.isArray(c.params) || c.params.length !== expected) {
    throw new Error(`checkpoint has ${Array.isArray(c.params) ? c.params.length : 0} parameters, expected ${expected}`);
  }
  if (!c.params.every((v) => typeof v === "number" && Number.isFinite(v))) {
    throw new Error("checkpoint parameters contain a non-finite value");
  }
  checkRules(c.config?.rules);
  // A readout is only interpretable next to the goal it was trained against, so the checkpoint
  // has to name it: two readouts trained under different wordings are not comparable.
  const objective = c.config?.objective as unknown as Record<string, unknown> | undefined;
  if (!objective || typeof objective !== "object") throw new Error("checkpoint does not record the objective it was trained against");
  for (const key of Object.keys(OBJECTIVE)) {
    if (typeof objective[key] !== "number" || !Number.isFinite(objective[key])) {
      throw new Error(`checkpoint objective has an invalid "${key}"`);
    }
  }
  if (circuitSha256 && c.circuitSha256 !== circuitSha256) {
    throw new Error(`checkpoint was trained on circuit ${c.circuitSha256.slice(0, 12)}, but this circuit is ${circuitSha256.slice(0, 12)}`);
  }
  return c;
}
