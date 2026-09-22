import { Circuit } from "../brain/circuit";
import { trainCem } from "./cem";
import type { WorkerCommand, WorkerMessage } from "./protocol";

const post = (message: WorkerMessage): void => {
  (self as unknown as { postMessage(m: WorkerMessage): void }).postMessage(message);
};

self.onmessage = (event: MessageEvent<WorkerCommand>) => {
  const command = event.data;
  if (command.type !== "start") return;
  try {
    const circuit = new Circuit(command.circuit);
    const result = trainCem(circuit, command.config, (record, champion) => {
      post({ type: "generation", record, champion: Array.from(champion) });
    });
    post({
      type: "done",
      champion: Array.from(result.champion),
      championGeneration: result.championGeneration,
      championValidation: result.championValidation,
      history: result.history,
    });
  } catch (error) {
    post({ type: "error", message: error instanceof Error ? error.message : String(error) });
  }
};
