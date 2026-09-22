import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { Circuit, type CircuitData } from "../src/brain/circuit";
import { ConnectomeController, DEPLOY_TEMPERATURE } from "../src/brain/controller";
import { parseCheckpoint } from "../src/train/checkpoint";
import { runEpisode, type EvaluationSummary } from "../src/train/rollout";
import { readoutInputCount } from "../src/brain/readout";
import { EscapePathway } from "../src/brain/escape";

const hasArtifacts = existsSync("public/checkpoints/readout.json") && existsSync("public/benchmarks/benchmark.json");

describe.skipIf(!hasArtifacts)("published benchmark", () => {
  it("reproduces the first held-out courses exactly", () => {
    const data = JSON.parse(readFileSync("public/data/circuit.json", "utf8")) as CircuitData;
    const circuit = new Circuit(data);
    const checkpoint = parseCheckpoint(JSON.parse(readFileSync("public/checkpoints/readout.json", "utf8")), readoutInputCount(circuit.readoutCells.length));
    const benchmark = JSON.parse(readFileSync("public/benchmarks/benchmark.json", "utf8")) as { controllers: EvaluationSummary[] };
    const published = benchmark.controllers[0];
    // Built exactly as scripts/benchmark.ts builds it, escape pathway included: reproducing a
    // different controller would not check what the published table reports.
    const escape = new EscapePathway(new Circuit(JSON.parse(readFileSync("public/data/pathway/circuit.json", "utf8")) as CircuitData));
    const controller = new ConnectomeController(circuit, Float64Array.from(checkpoint.params), false, escape, DEPLOY_TEMPERATURE);
    for (const expected of published.results.slice(0, 5)) {
      expect(runEpisode(controller, expected.seed, checkpoint.config.rules)).toEqual(expected);
    }
  });

  it("depends on circuit activity", () => {
    const benchmark = JSON.parse(readFileSync("public/benchmarks/benchmark.json", "utf8")) as { controllers: EvaluationSummary[] };
    const [trained, silenced] = benchmark.controllers;
    expect(trained.meanScore).toBeGreaterThan(silenced.meanScore);
  });
});
