/**
 * Does the measured wiring carry the play, or would any graph of the same shape do?
 *
 * The trained readout is held fixed and only the wiring underneath it is replaced, so nothing
 * about the learned parameters changes between rows. Three replacements are compared with the
 * measured circuit on the same held-out courses:
 *
 *   rewired   degree-preserving shuffle of the presynaptic partners, which keeps every cell's
 *             in-degree and every contact count and destroys only who connects to whom
 *   silenced  the circuit held at zero, which is the floor
 *   measured  the shipped circuit
 *
 * This shows whether the trained readout depends on the measured wiring it was trained on. It
 * does not show that measured wiring is easier to learn from than an arbitrary graph of the
 * same shape: that would need a rewired graph trained from scratch on the same budget, which
 * this project has not run.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { Circuit, type CircuitData } from "../src/brain/circuit";
import { EscapePathway } from "../src/brain/escape";
import { ConnectomeController, DEPLOY_TEMPERATURE } from "../src/brain/controller";
import { readoutInputCount } from "../src/brain/readout";
import { Rng } from "../src/game/rng";
import { parseCheckpoint } from "../src/train/checkpoint";
import { evaluate, heldOutSeeds } from "../src/train/rollout";

const dir = process.argv[2] ?? "public";
const replicates = Number(process.argv[3] ?? 5);

/** Degree-preserving rewiring: shuffle presynaptic partners, keeping in-degrees and contact counts. */
function rewire(edges: CircuitData["edges"], seed: number): CircuitData["edges"] {
  const rng = new Rng(seed);
  const pres = edges.map((e) => e[0]);
  for (let i = pres.length - 1; i > 0; i--) {
    const j = rng.int(i + 1);
    [pres[i], pres[j]] = [pres[j], pres[i]];
  }
  return edges.map((e, i) => [pres[i], e[1], e[2]] as [number, number, number]);
}

const circuitData = JSON.parse(readFileSync("public/data/circuit.json", "utf8")) as CircuitData;
const pathwayData = JSON.parse(readFileSync("public/data/pathway/circuit.json", "utf8")) as CircuitData;
const measured = new Circuit(circuitData);
const checkpoint = parseCheckpoint(JSON.parse(readFileSync(join(dir, "checkpoints", "readout.json"), "utf8")), readoutInputCount(measured.readoutCells.length));
const rules = checkpoint.config.rules;
const params = Float64Array.from(checkpoint.params);
const seeds = heldOutSeeds();
const escape = new EscapePathway(new Circuit(pathwayData));

const rows: Array<{ label: string; cleared: number; meanScore: number; meanPellets: number }> = [];
const add = (label: string, circuit: Circuit, pathway: EscapePathway | null, silenced = false): void => {
  const summary = evaluate(new ConnectomeController(circuit, params, silenced, pathway, DEPLOY_TEMPERATURE), seeds, rules);
  rows.push({ label, cleared: summary.cleared, meanScore: summary.meanScore, meanPellets: summary.meanPellets });
};

add("measured circuit and pathway", measured, escape);
for (let r = 0; r < replicates; r++) {
  add(`both rewired, replicate ${r + 1}`, new Circuit({ ...circuitData, edges: rewire(circuitData.edges, 8_000 + r) }), new EscapePathway(new Circuit({ ...pathwayData, edges: rewire(pathwayData.edges, 9_000 + r) })));
}
add("main circuit silenced", measured, escape, true);

console.log(`| Wiring under the same trained readout | Cleared / ${seeds.length} | Mean score | Mean pellets |`);
console.log("| --- | ---: | ---: | ---: |");
for (const r of rows) console.log(`| ${r.label} | ${r.cleared} | ${r.meanScore.toFixed(2)} | ${r.meanPellets.toFixed(2)} |`);

const shuffled = rows.filter((r) => r.label.startsWith("both rewired"));
const mean = shuffled.reduce((a, r) => a + r.meanPellets, 0) / shuffled.length;
const spread = Math.sqrt(shuffled.reduce((a, r) => a + (r.meanPellets - mean) ** 2, 0) / shuffled.length);
console.log(`\nmeasured ${rows[0].meanPellets.toFixed(2)} pellets against rewired ${mean.toFixed(2)} ± ${spread.toFixed(2)} over ${shuffled.length} replicates`);

writeFileSync(join(dir, "benchmarks", "rewire-control.json"), JSON.stringify({ seeds: { first: seeds[0], count: seeds.length }, replicates: shuffled.length, rows }, null, 1));
