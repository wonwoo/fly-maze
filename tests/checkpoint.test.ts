import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import { paramCount, readoutInputCount } from "../src/brain/readout";
import { DEFAULT_CEM } from "../src/train/cem";
import { parseCheckpoint } from "../src/train/checkpoint";
import { heldOutSeeds } from "../src/train/rollout";
import { DEFAULT_RULES } from "../src/game/rules";

const valid = () => ({
  version: 1 as const,
  params: new Array(paramCount(readoutInputCount(32))).fill(0.1),
  championGeneration: 1,
  championValidation: 1,
  config: { ...DEFAULT_CEM },
  circuitSha256: "abc",
});

describe("parseCheckpoint", () => {
  it("accepts a well-formed checkpoint", () => {
    expect(() => parseCheckpoint(valid(), readoutInputCount(32))).not.toThrow();
    expect(() => parseCheckpoint(valid(), readoutInputCount(32), "abc")).not.toThrow();
  });

  it("rejects the wrong parameter count, circuit, version and shape", () => {
    expect(() => parseCheckpoint(valid(), readoutInputCount(16))).toThrow(/parameters/);
    expect(() => parseCheckpoint(valid(), readoutInputCount(32), "other")).toThrow(/circuit/);
    expect(() => parseCheckpoint({ ...valid(), version: 2 }, readoutInputCount(32))).toThrow(/version/);
    expect(() => parseCheckpoint(null, readoutInputCount(32))).toThrow();
  });

  it("rejects non-finite parameters and broken rules", () => {
    const nan = valid();
    nan.params[3] = Number.NaN;
    expect(() => parseCheckpoint(nan, readoutInputCount(32))).toThrow(/non-finite/);
    const noRules = valid();
    (noRules.config as { rules?: unknown }).rules = undefined;
    expect(() => parseCheckpoint(noRules, readoutInputCount(32))).toThrow(/rules/);
    const emptyChase = valid();
    emptyChase.config = { ...DEFAULT_CEM, rules: { ...DEFAULT_RULES, ghostChase: [] } };
    expect(() => parseCheckpoint(emptyChase, readoutInputCount(32))).toThrow(/ghostChase/);
    const badEncoding = valid();
    badEncoding.config = { ...DEFAULT_CEM, rules: { ...DEFAULT_RULES, foodEncoding: "nearest" as never } };
    expect(() => parseCheckpoint(badEncoding, readoutInputCount(32))).toThrow(/foodEncoding/);
  });

  // The browser trainer produces a wider readout than the published one, so an export has to
  // carry the config that run used. Pairing browser-trained parameters with the published
  // checkpoint's config wrote a file that could not be imported again.
  it("round-trips a browser-trained readout, and rejects one exported under the published config", () => {
    const inputs = readoutInputCount(32);
    const browserTrained = new Array(paramCount(inputs, DEFAULT_CEM.hidden)).fill(0.01);
    expect(browserTrained.length).not.toBe(paramCount(inputs, 0));
    const exported = { ...valid(), params: browserTrained, config: { ...DEFAULT_CEM } };
    expect(parseCheckpoint(exported, inputs).params).toHaveLength(browserTrained.length);
    const mislabelled = { ...exported, config: { ...DEFAULT_CEM, hidden: 0 } };
    expect(() => parseCheckpoint(mislabelled, inputs)).toThrow(/parameters/);
  });

  it("matches the published checkpoint to the published circuit", () => {
    const circuitHash = createHash("sha256").update(readFileSync("public/data/circuit.json")).digest("hex");
    const manifest = JSON.parse(readFileSync("public/data/manifest.json", "utf8")) as { circuit_sha256: string };
    expect(manifest.circuit_sha256).toBe(circuitHash);
    const raw = JSON.parse(readFileSync("public/checkpoints/readout.json", "utf8"));
    expect(() => parseCheckpoint(raw, readoutInputCount(32), circuitHash)).not.toThrow();
  });
});

describe("course seed ranges", () => {
  it("keeps training, validation and held-out seeds disjoint", () => {
    const validation = new Set(DEFAULT_CEM.validationSeeds);
    const held = new Set(heldOutSeeds(1000));
    for (const seed of validation) expect(held.has(seed)).toBe(false);
    // Training course seeds are drawn from 1..900,000 in src/train/cem.ts.
    for (const seed of [...validation, ...held]) expect(seed).toBeGreaterThan(900_000);
  });
});
