/**
 * Does the measured looming pathway actually drive the giant fiber?
 *
 * Stimulates the looming population and measures activity at DNp01 against three
 * controls: a degree-preserving rewiring of the same graph, stimulation of an equally
 * sized non-looming sensory population, and no stimulus at all. Nothing is trained.
 */
import { readFileSync } from "node:fs";
import { Circuit, type CircuitData } from "../src/brain/circuit";
import { Rng } from "../src/game/rng";

const path = process.argv[2] ?? "public/data/pathway/circuit.json";
const data = JSON.parse(readFileSync(path, "utf8")) as CircuitData;
const cells = data.cells;
const giantFiber = cells.map((c, i) => [c, i] as const).filter(([c]) => c.giantFiber).map(([, i]) => i);
const looming = cells.map((c, i) => [c, i] as const).filter(([c]) => c.modality === "threat").map(([, i]) => i);
const wall = cells.map((c, i) => [c, i] as const).filter(([c]) => c.modality === "wall").map(([, i]) => i);
console.log(`cells ${cells.length}, edges ${data.edges.length}, giant fiber ${giantFiber.length}, looming ${looming.length}, wall ${wall.length}`);

/** Degree-preserving rewiring: shuffle presynaptic partners, keeping each cell's in-degree and contact counts. */
function rewire(edges: CircuitData["edges"], seed: number): CircuitData["edges"] {
  const rng = new Rng(seed);
  const pres = edges.map((e) => e[0]);
  for (let i = pres.length - 1; i > 0; i--) {
    const j = rng.int(i + 1);
    [pres[i], pres[j]] = [pres[j], pres[i]];
  }
  return edges.map((e, i) => [pres[i], e[1], e[2]] as [number, number, number]);
}

/** Run the fixed dynamics with a drive on `stimulated`, return mean |activity| on `probe`. */
function respond(circuit: Circuit, stimulated: number[], probe: number[], ticks: number): number {
  circuit.reset();
  const drive = new Float64Array(circuit.size);
  let total = 0;
  for (let t = 0; t < ticks; t++) {
    drive.fill(0);
    for (const cell of stimulated) drive[cell] = 1;
    circuit.stepWithDrive(drive);
    let sum = 0;
    for (const p of probe) sum += Math.abs(circuit.h[p]);
    total += sum / probe.length;
  }
  return total / ticks;
}

const TICKS = 12;
const base = new Circuit(data);
const loomingResponse = respond(base, looming, giantFiber, TICKS);
const wallResponse = respond(base, wall, giantFiber, TICKS);
const silent = respond(base, [], giantFiber, TICKS);

const rewired: number[] = [];
for (const seed of [1, 2, 3, 4, 5]) {
  const shuffled = new Circuit({ ...data, edges: rewire(data.edges, seed) });
  rewired.push(respond(shuffled, looming, giantFiber, TICKS));
}
const rewiredMean = rewired.reduce((a, b) => a + b, 0) / rewired.length;

const show = (label: string, value: number) => console.log(`${label.padEnd(38)} ${value.toExponential(3)}`);
show("looming stimulus, measured wiring", loomingResponse);
show("looming stimulus, rewired controls", rewiredMean);
show("wall stimulus, measured wiring", wallResponse);
show("no stimulus", silent);
console.log(`\nselectivity vs rewired: ${(loomingResponse / Math.max(rewiredMean, 1e-12)).toFixed(2)}x`);
console.log(`selectivity vs wall population: ${(loomingResponse / Math.max(wallResponse, 1e-12)).toFixed(2)}x`);
console.log(`selectivity vs silence: ${(loomingResponse / Math.max(silent, 1e-12)).toFixed(2)}x`);
