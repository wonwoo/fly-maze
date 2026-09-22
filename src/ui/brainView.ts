import type { Circuit, CircuitCell } from "../brain/circuit";
import { DIR_NAMES } from "../game/maze";

interface Placed {
  index: number;
  x: number;
  y: number;
  radius: number;
}

const MODALITY_COLORS: Record<string, string> = {
  food: "#f2b134",
  threat: "#f87171",
  wall: "#94a2bd",
  prey: "#4ade80",
  self: "#c084fc",
};
const BRIDGE_COLOR = "#5f7196";
const READOUT_COLOR = "#e9edf6";
/** Only the strongest measured connections are drawn; the full graph is unreadable. */
const EDGES_DRAWN = 260;

/**
 * Anatomical view of the extracted circuit.
 *
 * Cells sit at their measured MaleCNS soma coordinates, projected as a frontal view
 * (x medio-lateral, y dorso-ventral), so position carries anatomy rather than layout
 * convenience. The gustatory and proprioceptive cells have no CNS soma and are drawn in
 * a peripheral strip below the brain, which is where they enter from. Underneath, the
 * descending cells the readout actually reads are shown as signed bars.
 */
export class BrainView {
  private readonly ctx: CanvasRenderingContext2D;
  private readonly placed: Placed[];
  private readonly byIndex: Map<number, Placed>;
  private readonly edgeOrder: number[];
  private readonly edgeWeights: Float64Array;
  private readonly readouts: Array<{ index: number; label: string }>;
  private readonly midline: number;
  private readonly anatomyBottom: number;
  private readonly barTop: number;
  /** Layout size in CSS pixels. The backing store may be larger on a high-resolution display. */
  private readonly width: number;
  private readonly height: number;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly circuit: Circuit,
  ) {
    this.ctx = canvas.getContext("2d")!;
    const cells = circuit.data.cells;
    const { width, height } = canvas;
    this.width = width;
    this.height = height;
    this.anatomyBottom = Math.round(height * 0.6);
    this.barTop = this.anatomyBottom + 54;

    const withSoma = cells.map((c, i) => ({ c, i })).filter(({ c }) => c.soma);
    const xs = withSoma.map(({ c }) => c.soma![0]);
    const ys = withSoma.map(({ c }) => c.soma![1]);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const padX = 30;
    const top = 30;
    const scale = Math.min((width - padX * 2) / (maxX - minX), (this.anatomyBottom - top - 14) / (maxY - minY));
    const offsetX = (width - (maxX - minX) * scale) / 2;
    const offsetY = top + (this.anatomyBottom - top - 14 - (maxY - minY) * scale) / 2;
    // Mirror x so the fly's left hemisphere appears on the viewer's left.
    const project = (soma: number[]): [number, number] => [
      width - (offsetX + (soma[0] - minX) * scale),
      offsetY + (soma[1] - minY) * scale,
    ];
    this.midline = width - (offsetX + ((minX + maxX) / 2 - minX) * scale);

    this.placed = [];
    for (const { c, i } of withSoma) {
      const [x, y] = project(c.soma!);
      this.placed.push({ index: i, x, y, radius: c.role === "readout" ? 5.5 : 3.4 });
    }
    // Peripheral sensory cells, spread evenly in a strip under the brain.
    const peripheral = cells.map((c, i) => ({ c, i })).filter(({ c }) => !c.soma);
    const stripY = this.anatomyBottom + 18;
    peripheral.forEach(({ i }, k) => {
      const t = peripheral.length === 1 ? 0.5 : k / (peripheral.length - 1);
      this.placed.push({ index: i, x: padX + t * (width - padX * 2), y: stripY, radius: 3.4 });
    });
    this.byIndex = new Map(this.placed.map((p) => [p.index, p]));

    this.edgeWeights = Float64Array.from(circuit.data.edges, ([pre, post]) => circuit.weightBetween(pre, post));
    this.edgeOrder = circuit.data.edges
      .map((_, i) => i)
      .sort((a, b) => Math.abs(this.edgeWeights[b]) - Math.abs(this.edgeWeights[a]))
      .slice(0, EDGES_DRAWN)
      .reverse();

    this.readouts = circuit.readoutCells.map((index) => ({
      index,
      label: cells[index].type ?? String(cells[index].bodyId),
    }));
  }

  private cellColor(cell: CircuitCell): string {
    if (cell.role === "readout") return READOUT_COLOR;
    if (cell.role === "bridge") return BRIDGE_COLOR;
    return MODALITY_COLORS[cell.modality ?? ""] ?? BRIDGE_COLOR;
  }

  draw(chosen: number): void {
    const { ctx, circuit } = this;
    const cells = circuit.data.cells;
    ctx.fillStyle = "#070a12";
    ctx.fillRect(0, 0, this.width, this.height);

    ctx.strokeStyle = "#1b2540";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 5]);
    ctx.beginPath();
    ctx.moveTo(this.midline, 26);
    ctx.lineTo(this.midline, this.anatomyBottom - 6);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.font = "9px ui-sans-serif, system-ui";
    ctx.fillStyle = "#4e5d7d";
    ctx.textAlign = "center";
    ctx.fillText("left hemisphere", this.midline / 2, 16);
    ctx.fillText("right hemisphere", this.midline + (this.width - this.midline) / 2, 16);
    ctx.textAlign = "left";
    ctx.fillText("gustatory and proprioceptive cells (no central soma)", 22, this.anatomyBottom + 6);

    ctx.lineWidth = 0.7;
    for (const e of this.edgeOrder) {
      const [pre, post] = circuit.data.edges[e];
      const a = this.byIndex.get(pre);
      const b = this.byIndex.get(post);
      const w = this.edgeWeights[e];
      if (!a || !b || w === 0) continue;
      const active = Math.min(1, Math.abs(circuit.h[pre]));
      // The resting term is kept small on purpose: with 260 curves on screen, a floor high enough
      // to read individually turns the whole field into a thicket and the carrying paths stop
      // standing out. What the eye should follow is the connections a live cell is driving.
      const alpha = 0.015 + 0.5 * active * Math.min(1, Math.abs(w) * 3);
      ctx.strokeStyle = w > 0 ? `rgba(242,177,52,${alpha})` : `rgba(96,165,250,${alpha})`;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.quadraticCurveTo((a.x + b.x) / 2, (a.y + b.y) / 2 - 18, b.x, b.y);
      ctx.stroke();
    }

    for (const p of this.placed) {
      const cell = cells[p.index];
      const h = circuit.h[p.index];
      const mag = Math.min(1, Math.abs(h));
      if (mag > 0.12) {
        ctx.fillStyle = h >= 0 ? `rgba(242,177,52,${0.2 * mag})` : `rgba(96,165,250,${0.2 * mag})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius + 8 * mag, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 0.35 + 0.65 * mag;
      ctx.fillStyle = this.cellColor(cell);
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
      if (cell.role === "readout") {
        ctx.strokeStyle = h >= 0 ? "rgba(242,177,52,0.85)" : "rgba(96,165,250,0.85)";
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius + 2.5, 0, Math.PI * 2);
        ctx.stroke();
      }
    }

    this.drawReadoutBars(chosen);
  }

  /** The sixteen descending activities, exactly the vector the trained readout consumes. */
  private drawReadoutBars(chosen: number): void {
    const { ctx, canvas, circuit } = this;
    const top = this.barTop;
    ctx.font = "9px ui-sans-serif, system-ui";
    ctx.fillStyle = "#6c7c9c";
    ctx.textAlign = "left";
    ctx.fillText("descending activity read by the trained readout", 22, top - 12);
    ctx.textAlign = "right";
    ctx.fillStyle = chosen >= 0 ? "#f2b134" : "#4e5d7d";
    ctx.fillText(chosen >= 0 ? `chosen: ${DIR_NAMES[chosen]}` : "readout idle", this.width - 22, top - 12);

    const columns = 2;
    const rows = Math.ceil(this.readouts.length / columns);
    const columnWidth = (this.width - 44 - 16) / columns;
    const rowHeight = Math.min(16, (this.height - top - 12) / rows);
    ctx.textAlign = "left";
    this.readouts.forEach((r, k) => {
      const column = Math.floor(k / rows);
      const row = k % rows;
      const left = 22 + column * (columnWidth + 16);
      const y = top + row * rowHeight + rowHeight / 2;
      ctx.fillStyle = "#5d6b8c";
      ctx.font = "8.5px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.fillText(r.label, left, y + 3);

      const trackLeft = left + 48;
      const trackWidth = columnWidth - 48;
      const center = trackLeft + trackWidth / 2;
      ctx.fillStyle = "#151d33";
      ctx.fillRect(trackLeft, y - 3.5, trackWidth, 7);
      ctx.fillStyle = "#26324f";
      ctx.fillRect(center, y - 4.5, 1, 9);
      const value = Math.max(-1, Math.min(1, circuit.h[r.index]));
      const span = (Math.abs(value) * trackWidth) / 2;
      ctx.fillStyle = value >= 0 ? "#f2b134" : "#60a5fa";
      ctx.fillRect(value >= 0 ? center : center - span, y - 3.5, span, 7);
    });
  }
}
