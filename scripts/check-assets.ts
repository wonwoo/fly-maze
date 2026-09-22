/**
 * Verify that the published artifacts describe each other consistently:
 * the manifest pins the circuit that ships, and the checkpoint and benchmark
 * name that same circuit. Exits non-zero on any mismatch.
 */
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { DEPLOY_TEMPERATURE } from "../src/brain/controller";
import { paramCount, readoutInputCount } from "../src/brain/readout";

const sha256 = (path: string): string => createHash("sha256").update(readFileSync(path)).digest("hex");
const read = (path: string): any => JSON.parse(readFileSync(path, "utf8"));

const problems: string[] = [];
const circuitPath = "public/data/circuit.json";
const actual = sha256(circuitPath);

const manifest = read("public/data/manifest.json");
if (manifest.circuit_sha256 !== actual) {
  problems.push(`manifest.json pins circuit ${String(manifest.circuit_sha256).slice(0, 12)} but ${circuitPath} hashes to ${actual.slice(0, 12)}`);
}

const checkpointPath = "public/checkpoints/readout.json";
if (existsSync(checkpointPath)) {
  const checkpoint = read(checkpointPath);
  if (checkpoint.circuitSha256 !== actual) {
    problems.push(`${checkpointPath} was trained on circuit ${String(checkpoint.circuitSha256).slice(0, 12)}, not the published ${actual.slice(0, 12)}`);
  }
  const circuit = read(circuitPath);
  const readouts = circuit.cells.filter((c: { role: string }) => c.role === "readout").length;
  // The shape is whatever the checkpoint declares it trained, so the two must agree here.
  const hidden = checkpoint.config?.hidden;
  if (typeof hidden !== "number") {
    problems.push(`${checkpointPath} does not record the readout width it was trained with`);
  } else {
    const expected = paramCount(readoutInputCount(readouts), hidden);
    if (checkpoint.params?.length !== expected) {
      problems.push(`${checkpointPath} has ${checkpoint.params?.length} parameters, but ${readouts} readout cells at hidden width ${hidden} need ${expected}`);
    }
  }
}

// The escape pathway is a second measured circuit with its own manifest. It ships whether or
// not the published checkpoint gates it in, so a silent edit to it should fail the check too.
const pathwayPath = "public/data/pathway/circuit.json";
const pathwayManifestPath = "public/data/pathway/circuit-manifest.json";
if (existsSync(pathwayPath) && existsSync(pathwayManifestPath)) {
  const pathwayActual = sha256(pathwayPath);
  const pathwayManifest = read(pathwayManifestPath);
  if (pathwayManifest.circuit_sha256 !== pathwayActual) {
    problems.push(`${pathwayManifestPath} pins circuit ${String(pathwayManifest.circuit_sha256).slice(0, 12)} but ${pathwayPath} hashes to ${pathwayActual.slice(0, 12)}`);
  }
  // Both circuits have to come from the same release, or the two halves describe different flies.
  for (const [key, entry] of Object.entries(pathwayManifest.raw ?? {}) as Array<[string, { sha256: string }]>) {
    const main = manifest.raw?.[key];
    if (main && main.sha256 !== entry.sha256) {
      problems.push(`the pathway was built from a different ${key} table (${entry.sha256.slice(0, 12)}) than the main circuit (${String(main.sha256).slice(0, 12)})`);
    }
  }
}

// The temperature the game ships at is chosen by a sweep, so the sweep has to describe the
// weights that ship and still name the value the code uses. A stale sweep otherwise justifies a
// deployment choice measured on a checkpoint that was replaced.
const temperaturePath = "public/benchmarks/temperature.json";
if (existsSync(temperaturePath) && existsSync(checkpointPath)) {
  const sweep = read(temperaturePath);
  const checkpoint = read(checkpointPath);
  if (sweep.checkpoint?.circuitSha256 !== actual || sweep.checkpoint?.championValidation !== checkpoint.championValidation) {
    problems.push(`${temperaturePath} swept a different checkpoint than the one that ships; re-run npm run tune:temperature`);
  }
  if (sweep.best !== DEPLOY_TEMPERATURE) {
    problems.push(`${temperaturePath} picked temperature ${String(sweep.best)} but DEPLOY_TEMPERATURE is ${DEPLOY_TEMPERATURE}`);
  }
}

// The probes describe one checkpoint played at one temperature, and both are recorded in the
// artifact, so both can go stale silently after a retrain or a temperature change.
const probePath = "public/benchmarks/probe.json";
if (existsSync(probePath) && existsSync(checkpointPath)) {
  const probe = read(probePath);
  const trained = read(checkpointPath);
  if (probe.checkpoint?.circuitSha256 !== actual || probe.checkpoint?.championValidation !== trained.championValidation) {
    problems.push(`${probePath} probed a different checkpoint than the one that ships; re-run npm run probe`);
  }
  if (probe.temperature !== DEPLOY_TEMPERATURE) {
    problems.push(`${probePath} was measured at temperature ${String(probe.temperature)} but DEPLOY_TEMPERATURE is ${DEPLOY_TEMPERATURE}`);
  }
}

const benchmarkPath = "public/benchmarks/benchmark.json";
if (existsSync(benchmarkPath)) {
  const benchmark = read(benchmarkPath);
  if (benchmark.checkpoint?.circuitSha256 !== actual) {
    problems.push(`${benchmarkPath} reports results for circuit ${String(benchmark.checkpoint?.circuitSha256).slice(0, 12)}, not the published ${actual.slice(0, 12)}`);
  }
}

if (problems.length > 0) {
  console.error("asset check failed:");
  for (const p of problems) console.error(`  - ${p}`);
  process.exit(1);
}
console.log(`asset check passed: circuit ${actual.slice(0, 12)} is pinned consistently`);
