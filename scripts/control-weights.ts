/**
 * Does the measured synapse count carry the play, or would the graph alone do?
 *
 * The rewiring control in scripts/control-rewire.ts destroys who connects to whom while keeping
 * every contact count. This one is its complement: every edge keeps both endpoints, so the graph
 * is the measured graph down to the last recurrent and single-contact edge, and only the number
 * measured on each edge is replaced. What is left is the question of whether the readout needs
 * the strengths that were measured in the fly, or merely a circuit of the right shape.
 *
 * The trained readout is held fixed, so nothing about the learned parameters changes between
 * rows. Two replacements are compared with the measured circuit on the same held-out courses:
 *
 *   shuffled  the measured contact counts dealt back out over the same edges in a random order,
 *             which keeps the multiset of strengths and destroys only which edge carries which
 *   flat      every edge set to a single contact, which removes strength entirely and leaves
 *             the normalisation to divide each cell's input equally among its presynaptic cells
 *   measured  the shipped circuit
 *
 * Like the rewiring control, this shows dependence rather than superiority: it does not show
 * that measured strengths are easier to learn from than arbitrary ones, which would need a
 * retrained readout on an equal budget.
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

/** Deal the measured contact counts back out over the same edges in a random order. */
function shuffleWeights(edges: CircuitData["edges"], seed: number): CircuitData["edges"] {
  const rng = new Rng(seed);
  const counts = edges.map((e) => e[2]);
  for (let i = counts.length - 1; i > 0; i--) {
    const j = rng.int(i + 1);
    [counts[i], counts[j]] = [counts[j], counts[i]];
  }
  return edges.map((e, i) => [e[0], e[1], counts[i]] as [number, number, number]);
}

/** Every edge reduced to a single contact, so only the graph is left. */
function flatWeights(edges: CircuitData["edges"]): CircuitData["edges"] {
  return edges.map((e) => [e[0], e[1], 1] as [number, number, number]);
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
const add = (label: string, circuit: Circuit, pathway: EscapePathway): void => {
  const summary = evaluate(new ConnectomeController(circuit, params, false, pathway, DEPLOY_TEMPERATURE), seeds, rules);
  rows.push({ label, cleared: summary.cleared, meanScore: summary.meanScore, meanPellets: summary.meanPellets });
};

add("measured contact counts", measured, escape);
for (let r = 0; r < replicates; r++) {
  add(
    `contact counts shuffled, replicate ${r + 1}`,
    new Circuit({ ...circuitData, edges: shuffleWeights(circuitData.edges, 6_000 + r) }),
    new EscapePathway(new Circuit({ ...pathwayData, edges: shuffleWeights(pathwayData.edges, 7_000 + r) })),
  );
}
add("every edge flattened to one contact", new Circuit({ ...circuitData, edges: flatWeights(circuitData.edges) }), new EscapePathway(new Circuit({ ...pathwayData, edges: flatWeights(pathwayData.edges) })));

console.log(`| Contact counts under the same trained readout and the same graph | Cleared / ${seeds.length} | Mean score | Mean pellets |`);
console.log("| --- | ---: | ---: | ---: |");
for (const r of rows) console.log(`| ${r.label} | ${r.cleared} | ${r.meanScore.toFixed(2)} | ${r.meanPellets.toFixed(2)} |`);

const shuffled = rows.filter((r) => r.label.startsWith("contact counts shuffled"));
const mean = shuffled.reduce((a, r) => a + r.meanPellets, 0) / shuffled.length;
const spread = Math.sqrt(shuffled.reduce((a, r) => a + (r.meanPellets - mean) ** 2, 0) / shuffled.length);
console.log(`\nmeasured ${rows[0].meanPellets.toFixed(2)} pellets against shuffled ${mean.toFixed(2)} ± ${spread.toFixed(2)} over ${shuffled.length} replicates`);

writeFileSync(join(dir, "benchmarks", "weight-control.json"), JSON.stringify({ seeds: { first: seeds[0], count: seeds.length }, replicates: shuffled.length, rows }, null, 1));
