import "./style.css";
import { Circuit, type CircuitData } from "./brain/circuit";
import { ConnectomeController, DEPLOY_TEMPERATURE, GreedyController, RandomController, type Controller } from "./brain/controller";
import { EscapePathway } from "./brain/escape";
import { paramCount, readoutInputCount } from "./brain/readout";
import { DOWN, LEFT, RIGHT, UP } from "./game/maze";
import { DEFAULT_RULES } from "./game/rules";
import { createGame, stepGame, type GameState } from "./game/sim";
import { DEFAULT_CEM } from "./train/cem";
import { parseCheckpoint, type Checkpoint } from "./train/checkpoint";
import type { WorkerCommand, WorkerMessage } from "./train/protocol";
import { BrainView } from "./ui/brainView";
import { renderGame } from "./ui/render";

const el = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;

class ManualController implements Controller {
  readonly name = "manual";
  desired = -1;
  reset(): void {
    this.desired = -1;
  }
  act(s: GameState): number {
    return this.desired >= 0 ? this.desired : s.pdir;
  }
}

async function loadJson<T>(path: string): Promise<T | null> {
  try {
    const response = await fetch(path);
    return response.ok ? ((await response.json()) as T) : null;
  } catch {
    return null;
  }
}

/**
 * Gives a canvas a backing store that matches the pixels it is actually painted on.
 *
 * The markup sizes both canvases in layout pixels and the stylesheet stretches them to the width
 * of their panel, so on a wide window or a high-resolution display the drawing was being scaled up
 * from fewer pixels than the screen has. The transform keeps every drawing routine in the same
 * coordinates it was written for.
 */
function matchCanvasToDisplay(canvas: HTMLCanvasElement, width: number, height: number): void {
  const displayed = canvas.getBoundingClientRect().width || width;
  const ratio = (displayed / width) * Math.min(3, window.devicePixelRatio || 1);
  const backingWidth = Math.round(width * ratio);
  if (canvas.width === backingWidth) return;
  canvas.width = backingWidth;
  canvas.height = Math.round(height * ratio);
  canvas.getContext("2d")!.setTransform(ratio, 0, 0, ratio, 0, 0);
}

const KEY_DIRECTIONS: Record<string, number> = {
  ArrowUp: UP, ArrowRight: RIGHT, ArrowDown: DOWN, ArrowLeft: LEFT,
  w: UP, d: RIGHT, s: DOWN, a: LEFT, W: UP, D: RIGHT, S: DOWN, A: LEFT,
};

async function main(): Promise<void> {
  const base = import.meta.env.BASE_URL;
  const circuitData = await loadJson<CircuitData>(`${base}data/circuit.json`);
  if (!circuitData) throw new Error("public/data/circuit.json is missing; run scripts/build-connectome.py");
  const pathwayData = await loadJson<CircuitData>(`${base}data/pathway/circuit.json`);
  const manifest = await loadJson<{ circuit_sha256: string }>(`${base}data/manifest.json`);
  const circuit = new Circuit(circuitData);

  let checkpoint: Checkpoint | null = null;
  try {
    const raw = await loadJson<unknown>(`${base}checkpoints/readout.json`);
    checkpoint = raw ? parseCheckpoint(raw, readoutInputCount(circuit.readoutCells.length), manifest?.circuit_sha256) : null;
  } catch {
    checkpoint = null;
  }
  const rules = checkpoint?.config.rules ?? DEFAULT_RULES;
  // The page must run the controller the published benchmark measured, escape pathway included;
  // without it the last four readout inputs are always zero and the numbers no longer apply.
  const pathway = pathwayData ? new EscapePathway(new Circuit(pathwayData)) : null;
  const connectome = new ConnectomeController(circuit, checkpoint ? Float64Array.from(checkpoint.params) : new Float64Array(paramCount(readoutInputCount(circuit.readoutCells.length))), false, pathway, DEPLOY_TEMPERATURE);
  const manual = new ManualController();
  const factories: Record<string, () => Controller> = {
    connectome: () => ((connectome.silenced = false), connectome),
    silenced: () => ((connectome.silenced = true), connectome),
    greedy: () => new GreedyController(),
    random: () => new RandomController(),
    manual: () => manual,
  };

  const gameCanvas = el<HTMLCanvasElement>("game");
  const gctx = gameCanvas.getContext("2d")!;
  const brain = new BrainView(el<HTMLCanvasElement>("brain"), circuit);
  const modeSelect = el<HTMLSelectElement>("mode");
  const bars = Array.from(document.querySelectorAll<HTMLDivElement>(".bar"));
  // Counted from the shipped circuit rather than written down, so the claim and the file agree.
  const measuredChip = `${circuitData.cells.length} MaleCNS cells \u00b7 ${circuitData.edges.length.toLocaleString("en-US")} measured connections \u00b7 fixed`;
  const checkpointInfo = el<HTMLParagraphElement>("checkpointInfo");

  // The course count comes from the checkpoint that produced the score, not from the current
  // default, so a readout trained under a narrower validation set is not described as a wider one.
  const describeCheckpoint = (label: string, generation: number, validation: number, courses: number): void => {
    checkpointInfo.textContent = `${label}: champion from generation ${generation}, validation score ${validation.toFixed(1)} on ${courses} fixed courses.`;
  };
  if (checkpoint) describeCheckpoint("Published checkpoint", checkpoint.championGeneration, checkpoint.championValidation, checkpoint.config.validationSeeds.length);
  else checkpointInfo.textContent = "No checkpoint found. The readout is all zeros until you train or import one.";

  // The panel runs the cross-entropy trainer over a hidden-layer readout, while the published
  // checkpoint is a linear readout trained by policy gradient. Both counts are computed here so
  // the page cannot drift from the code the way a written-in number does.
  const browserParams = paramCount(readoutInputCount(circuit.readoutCells.length), DEFAULT_CEM.hidden);
  el("trainChip").textContent = `${browserParams} parameters`;
  if (checkpoint) {
    el("trainNote").textContent =
      `Runs a cross-entropy method trainer in a Web Worker over ${browserParams} parameters. The published checkpoint is a different, ` +
      `smaller readout of ${checkpoint.params.length} parameters trained by policy gradient. The measured wiring is untouched by either.`;
  }

  // Read rather than written into the page, so re-running the benchmark updates what is shown.
  interface BenchmarkFile { seeds: { count: number }; controllers: Array<{ label: string; cleared: number; meanPellets: number }> }
  const benchmark = await loadJson<BenchmarkFile>(`${base}benchmarks/benchmark.json`);
  const scoreboard = el<HTMLUListElement>("scoreboard");
  if (benchmark) {
    const shown = ["connectome + trained readout", "same readout, circuit silenced", "untrained readout, random initialisation", "handwritten greedy-pellet baseline"];
    scoreboard.innerHTML = "";
    const caption = document.createElement("li");
    caption.className = "scoreboard-caption";
    caption.textContent = `${benchmark.seeds.count} unseen mazes`;
    scoreboard.append(caption);
    for (const label of shown) {
      const row = benchmark.controllers.find((c) => c.label === label);
      if (!row) continue;
      const item = document.createElement("li");
      item.innerHTML = `<b>${row.meanPellets.toFixed(1)}</b><span>pellets</span><em>${row.label}</em><i>${row.cleared} cleared</i>`;
      scoreboard.append(item);
    }
  }

  let mode = modeSelect.value;
  let controller = factories[mode]();
  let state = createGame(1, rules);
  let paused = false;
  let stepsPerSecond = Number(el<HTMLInputElement>("speed").value);
  let accumulator = 0;
  let last = performance.now();
  let restartAt: number | null = null;
  // Counted per controller: a tally that survived a controller change would credit one controller
  // with courses another one played.
  let cleared = 0;
  let finished = 0;
  const resetTally = (): void => {
    cleared = 0;
    finished = 0;
  };

  // Demo seeds sit above the training (1..900,000), validation and held-out ranges.
  const newGame = (seed = 3_100_001 + Math.floor(Math.random() * 900_000)): void => {
    state = createGame(seed, rules);
    controller = factories[mode]();
    controller.reset(state);
    restartAt = null;
  };
  newGame();

  const draw = (): void => {
    renderGame(gctx, state);
    const usesCircuit = controller === connectome;
    brain.draw(usesCircuit ? connectome.lastAction : -1);
    el("score").textContent = String(state.score);
    el("pellets").textContent = `${state.maze.pelletCount - state.remaining} / ${state.maze.pelletCount}`;
    el("steps").textContent = `${state.steps} / ${state.rules.maxSteps}`;
    el("cleared").textContent = `${cleared} / ${finished}`;
    el("seed").textContent = String(state.seed);
    el("brainChip").textContent = usesCircuit ? (connectome.silenced ? "circuit silenced" : measuredChip) : "readout idle";
    const status = !state.alive ? "caught" : state.cleared ? "cleared" : state.done ? "time up" : paused ? "paused" : state.frightened > 0 ? "ghosts fleeing" : "running";
    el("status").textContent = status;
    el("status").classList.toggle("subtle", status !== "running" && status !== "ghosts fleeing");
    const scores = usesCircuit ? connectome.scores : null;
    const scale = scores ? Math.max(1e-6, ...Array.from(scores, Math.abs)) : 1;
    bars.forEach((bar, d) => {
      const v = scores ? scores[d] : 0;
      const half = (Math.abs(v) / scale) * 50;
      bar.style.setProperty("--w", `${half}%`);
      bar.style.setProperty("--tx", v < 0 ? "-100%" : "0");
      bar.style.setProperty("--c", v < 0 ? "#60a5fa" : "#f2b134");
      bar.classList.toggle("chosen", usesCircuit && connectome.lastAction === d);
      bar.querySelector("b")!.textContent = scores ? v.toFixed(2) : "-";
    });
  };

  const tick = (now: number): void => {
    const dt = Math.min(250, now - last);
    last = now;
    if (!paused) {
      accumulator += dt;
      const interval = 1000 / stepsPerSecond;
      while (accumulator >= interval) {
        accumulator -= interval;
        if (!state.done) {
          stepGame(state, controller.act(state));
          // The episode can only end here, so each finished course is counted exactly once.
          if (state.done) {
            finished++;
            if (state.cleared) cleared++;
          }
        } else if (mode !== "manual") {
          restartAt ??= now + 1500;
          if (now >= restartAt) newGame();
        }
      }
    }
    draw();
    requestAnimationFrame(tick);
  };
  const fitCanvases = (): void => {
    matchCanvasToDisplay(gameCanvas, 456, 504);
    matchCanvasToDisplay(el<HTMLCanvasElement>("brain"), 456, 504);
  };
  fitCanvases();
  window.addEventListener("resize", fitCanvases);
  requestAnimationFrame(tick);

  modeSelect.addEventListener("change", () => {
    mode = modeSelect.value;
    resetTally();
    newGame(state.seed);
  });
  el("restart").addEventListener("click", () => newGame());
  el("pause").addEventListener("click", () => {
    paused = !paused;
    el("pause").textContent = paused ? "Resume" : "Pause";
  });
  el<HTMLInputElement>("speed").addEventListener("input", (e) => {
    stepsPerSecond = Number((e.target as HTMLInputElement).value);
    el("speedValue").textContent = String(stepsPerSecond);
  });
  window.addEventListener("keydown", (e) => {
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
    const dir = KEY_DIRECTIONS[e.key];
    if (dir !== undefined) {
      e.preventDefault();
      if (mode !== "manual") {
        modeSelect.value = "manual";
        mode = "manual";
        resetTally();
        controller = manual;
        manual.reset();
      }
      manual.desired = dir;
      if (state.done) newGame();
    } else if (e.key === " " && state.done) {
      e.preventDefault();
      newGame();
    }
  });

  // Training in a Web Worker: only the readout parameters change.
  const log = el<HTMLPreElement>("trainLog");
  const trainButton = el<HTMLButtonElement>("train");
  const stopButton = el<HTMLButtonElement>("stopTrain");
  let worker: Worker | null = null;
  // What an export should say about the parameters currently in the controller. It has to follow
  // them: the browser trainer produces a wider readout than the published one, so exporting the
  // published checkpoint's config next to browser-trained parameters writes a file that
  // `parseCheckpoint` then rejects on its parameter count, and that mislabels a browser run as
  // the published champion.
  let provenance: Pick<Checkpoint, "championGeneration" | "championValidation" | "config"> = {
    championGeneration: checkpoint?.championGeneration ?? 0,
    championValidation: checkpoint?.championValidation ?? 0,
    config: checkpoint?.config ?? DEFAULT_CEM,
  };
  const finishTraining = (): void => {
    worker?.terminate();
    worker = null;
    trainButton.disabled = false;
    stopButton.disabled = true;
  };
  trainButton.addEventListener("click", () => {
    const config = {
      ...DEFAULT_CEM,
      rules,
      seed: Number(el<HTMLInputElement>("trainSeed").value) || DEFAULT_CEM.seed,
      generations: Math.max(1, Number(el<HTMLInputElement>("trainGenerations").value) || DEFAULT_CEM.generations),
    };
    // The controller adopts this run's champion from its first generation, so the export has to
    // describe this run from the moment it starts, not from the moment it finishes.
    provenance = { championGeneration: 0, championValidation: 0, config };
    worker = new Worker(new URL("./train/worker.ts", import.meta.url), { type: "module" });
    trainButton.disabled = true;
    stopButton.disabled = false;
    log.textContent = `Training seed ${config.seed} for ${config.generations} generations (${config.candidates} candidates × ${config.coursesPerGeneration} courses each)...\n`;
    worker.onmessage = (event: MessageEvent<WorkerMessage>) => {
      const msg = event.data;
      if (msg.type === "generation") {
        const r = msg.record;
        log.textContent += `gen ${String(r.generation).padStart(3)}  best ${r.bestTrainFitness.toFixed(0).padStart(5)}  val ${r.validationScore.toFixed(0).padStart(5)}  champion ${r.championValidation.toFixed(0).padStart(5)}${r.championUpdated ? " *" : ""}\n`;
        log.scrollTop = log.scrollHeight;
        connectome.params = Float64Array.from(msg.champion);
        if (r.championUpdated) provenance = { championGeneration: r.generation, championValidation: r.championValidation, config };
      } else if (msg.type === "done") {
        connectome.params = Float64Array.from(msg.champion);
        provenance = { championGeneration: msg.championGeneration, championValidation: msg.championValidation, config };
        describeCheckpoint("Browser-trained readout", msg.championGeneration, msg.championValidation, DEFAULT_CEM.validationSeeds.length);
        log.textContent += `Done. Champion from generation ${msg.championGeneration}, validation ${msg.championValidation.toFixed(1)}. The game now uses it.\n`;
        finishTraining();
      } else {
        log.textContent += `Error: ${msg.message}\n`;
        finishTraining();
      }
    };
    const command: WorkerCommand = { type: "start", circuit: circuitData, config };
    worker.postMessage(command);
  });
  stopButton.addEventListener("click", () => {
    log.textContent += "Stopped. The latest champion stays in use.\n";
    finishTraining();
  });

  el("export").addEventListener("click", () => {
    const payload: Checkpoint = {
      version: 1,
      params: Array.from(connectome.params),
      championGeneration: provenance.championGeneration,
      championValidation: provenance.championValidation,
      config: provenance.config,
      circuitSha256: checkpoint?.circuitSha256 ?? manifest?.circuit_sha256 ?? "",
    };
    const blob = new Blob([JSON.stringify(payload)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "fly-maze-readout.json";
    a.click();
    URL.revokeObjectURL(a.href);
  });
  el<HTMLInputElement>("import").addEventListener("change", async (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) return;
    try {
      const imported = parseCheckpoint(JSON.parse(await file.text()), readoutInputCount(circuit.readoutCells.length), manifest?.circuit_sha256);
      connectome.params = Float64Array.from(imported.params);
      checkpoint = imported;
      provenance = { championGeneration: imported.championGeneration, championValidation: imported.championValidation, config: imported.config };
      describeCheckpoint("Imported readout", imported.championGeneration, imported.championValidation, imported.config.validationSeeds.length);
      log.textContent += `Imported ${file.name}.\n`;
    } catch (error) {
      log.textContent += `Could not import ${file.name}: ${error instanceof Error ? error.message : "not a valid checkpoint"}.\n`;
    }
  });

}

main().catch((error: unknown) => {
  document.body.insertAdjacentHTML("beforeend", `<pre class="error">${error instanceof Error ? error.message : String(error)}</pre>`);
});
