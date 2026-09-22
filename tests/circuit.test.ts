import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { Circuit, type CircuitData } from "../src/brain/circuit";
import { FEATURE_COUNT } from "../src/game/features";

const data = JSON.parse(readFileSync("public/data/circuit.json", "utf8")) as CircuitData;

describe("extracted circuit", () => {
  const circuit = new Circuit(data);

  it("has the documented shape", () => {
    expect(data.cells.length).toBe(160);
    expect(circuit.readoutCells.length).toBe(32);
    expect(circuit.channelCells.length).toBe(FEATURE_COUNT);
    for (const cells of circuit.channelCells) expect(cells.length).toBe(4);
    expect(data.edges.length).toBeGreaterThan(1000);
  });

  it("reaches every readout cell from the inputs through signed edges", () => {
    const adjacency = new Map<number, number[]>();
    for (const [pre, post] of data.edges) {
      if (data.cells[pre].sign === 0) continue;
      adjacency.set(pre, [...(adjacency.get(pre) ?? []), post]);
    }
    const seen = new Set<number>(circuit.channelCells.flat());
    const stack = [...seen];
    while (stack.length) {
      const u = stack.pop()!;
      for (const v of adjacency.get(u) ?? []) if (!seen.has(v)) (seen.add(v), stack.push(v));
    }
    for (const r of circuit.readoutCells) expect(seen.has(r)).toBe(true);
  });

  it("normalizes incoming weights to unit absolute mass", () => {
    const mass = new Float64Array(data.cells.length);
    for (const [pre, post] of data.edges) mass[post] += Math.abs(circuit.weightBetween(pre, post));
    for (const m of mass) expect(m).toBeLessThanOrEqual(1 + 1e-9);
    expect(Math.max(...mass)).toBeGreaterThan(0.99);
  });

  it("is silent when silenced and deterministic otherwise", () => {
    const features = new Float64Array(FEATURE_COUNT).fill(1);
    circuit.reset();
    const silenced = circuit.step(features, true);
    expect(Array.from(silenced).every((v) => v === 0)).toBe(true);
    circuit.reset();
    const first = Array.from(circuit.step(features));
    circuit.reset();
    const second = Array.from(circuit.step(features));
    expect(first).toEqual(second);
    expect(first.some((v) => v !== 0)).toBe(true);
    for (const v of circuit.h) expect(Math.abs(v)).toBeLessThanOrEqual(1);
  });
});
