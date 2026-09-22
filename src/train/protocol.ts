import type { CircuitData } from "../brain/circuit";
import type { CemConfig, GenerationRecord } from "./cem";

export type WorkerCommand = { type: "start"; circuit: CircuitData; config: CemConfig };

export type WorkerMessage =
  | { type: "generation"; record: GenerationRecord; champion: number[] }
  | { type: "done"; champion: number[]; championGeneration: number; championValidation: number; history: GenerationRecord[] }
  | { type: "error"; message: string };
