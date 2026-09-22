import { DIRS, cellIndex, isOpen } from "../game/maze";
import type { GameState } from "../game/sim";

export const CELL = 24;
const GHOST_COLORS = ["#4ade80", "#c084fc", "#fb923c"];
const WALL_FILL = "#141c34";
const WALL_EDGE = "#3b5bdb";
const FRIGHTENED = "#3b5bdb";
const FRIGHTENED_FLASH = "#e8ecf5";

/**
 * Draws the maze with outlined walls rather than filled tiles: each wall cell is stroked
 * only along the sides that face an open cell, which reads as a continuous corridor wall.
 * Everything is solid colour and stateless so headless rollouts and drawn play stay identical.
 */
export function renderGame(ctx: CanvasRenderingContext2D, s: GameState): void {
  const m = s.maze;
  ctx.fillStyle = "#070a12";
  ctx.fillRect(0, 0, m.width * CELL, m.height * CELL);

  ctx.lineWidth = 2;
  ctx.lineCap = "round";
  for (let y = 0; y < m.height; y++) {
    for (let x = 0; x < m.width; x++) {
      const i = cellIndex(m, x, y);
      if (m.wall[i]) drawWallCell(ctx, m, x, y);
      else if (m.door[i]) {
        ctx.fillStyle = "#8b5cf6";
        ctx.fillRect(x * CELL + 3, y * CELL + CELL / 2 - 2, CELL - 6, 3);
      }
    }
  }

  for (let y = 0; y < m.height; y++) {
    for (let x = 0; x < m.width; x++) {
      const i = cellIndex(m, x, y);
      if (!s.pellet[i]) continue;
      const cx = x * CELL + CELL / 2;
      const cy = y * CELL + CELL / 2;
      if (m.power[i] && s.rules.powerPellets) {
        ctx.fillStyle = "rgba(242, 177, 52, 0.22)";
        ctx.beginPath();
        ctx.arc(cx, cy, 10, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "#f2b134";
        ctx.beginPath();
        ctx.arc(cx, cy, 6, 0, Math.PI * 2);
        ctx.fill();
      } else {
        ctx.fillStyle = "#f7e9c4";
        ctx.beginPath();
        ctx.arc(cx, cy, 2.6, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  s.ghosts.forEach((g, k) => {
    // Ghosts waiting in the house are drawn faded rather than hidden, so the maze never looks empty.
    const waiting = s.steps < g.release;
    ctx.globalAlpha = waiting ? 0.3 : 1;
    drawGhost(ctx, g.x * CELL + CELL / 2, g.y * CELL + CELL / 2, g.dir, g.frightened, s.frightened, s.steps, GHOST_COLORS[k % GHOST_COLORS.length]);
    ctx.globalAlpha = 1;
  });

  drawFly(ctx, s.px * CELL + CELL / 2, s.py * CELL + CELL / 2, s.pdir, s.alive, s.steps);
}

function drawWallCell(ctx: CanvasRenderingContext2D, m: GameState["maze"], x: number, y: number): void {
  const left = x * CELL;
  const top = y * CELL;
  ctx.fillStyle = WALL_FILL;
  ctx.fillRect(left, top, CELL, CELL);
  ctx.strokeStyle = WALL_EDGE;
  ctx.beginPath();
  // Stroke each side that borders a non-wall cell, so only corridor-facing edges glow.
  if (!isWall(m, x, y - 1)) {
    ctx.moveTo(left + 1, top + 1);
    ctx.lineTo(left + CELL - 1, top + 1);
  }
  if (!isWall(m, x, y + 1)) {
    ctx.moveTo(left + 1, top + CELL - 1);
    ctx.lineTo(left + CELL - 1, top + CELL - 1);
  }
  if (!isWall(m, x - 1, y)) {
    ctx.moveTo(left + 1, top + 1);
    ctx.lineTo(left + 1, top + CELL - 1);
  }
  if (!isWall(m, x + 1, y)) {
    ctx.moveTo(left + CELL - 1, top + 1);
    ctx.lineTo(left + CELL - 1, top + CELL - 1);
  }
  ctx.stroke();
}

function isWall(m: GameState["maze"], x: number, y: number): boolean {
  if (x < 0 || y < 0 || x >= m.width || y >= m.height) return true;
  return m.wall[cellIndex(m, x, y)] === 1;
}

function drawGhost(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  dir: number,
  frightened: boolean,
  frightenedLeft: number,
  steps: number,
  color: string,
): void {
  const flashing = frightened && frightenedLeft < 10 && steps % 2 === 1;
  ctx.fillStyle = frightened ? (flashing ? FRIGHTENED_FLASH : FRIGHTENED) : color;
  ctx.beginPath();
  ctx.arc(cx, cy - 2, 8, Math.PI, 0);
  ctx.lineTo(cx + 8, cy + 7);
  // Scalloped skirt.
  for (let i = 0; i < 3; i++) {
    const x0 = cx + 8 - i * (16 / 3);
    const x1 = x0 - 16 / 6;
    const x2 = x0 - 16 / 3;
    ctx.quadraticCurveTo(x1, cy + 11, x2, cy + 7);
  }
  ctx.closePath();
  ctx.fill();

  if (frightened) {
    ctx.fillStyle = flashing ? "#b4341f" : "#e8ecf5";
    for (const ex of [-3.5, 3.5]) {
      ctx.beginPath();
      ctx.arc(cx + ex, cy - 2, 1.8, 0, Math.PI * 2);
      ctx.fill();
    }
    return;
  }
  const [dx, dy] = DIRS[dir];
  ctx.fillStyle = "#f4f7ff";
  for (const ex of [-3.5, 3.5]) {
    ctx.beginPath();
    ctx.ellipse(cx + ex, cy - 3, 3, 3.6, 0, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.fillStyle = "#1b2440";
  for (const ex of [-3.5, 3.5]) {
    ctx.beginPath();
    ctx.arc(cx + ex + dx * 1.4, cy - 3 + dy * 1.6, 1.6, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawFly(ctx: CanvasRenderingContext2D, cx: number, cy: number, dir: number, alive: boolean, steps: number): void {
  const [dx, dy] = DIRS[dir];
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(Math.atan2(dy, dx));
  ctx.globalAlpha = alive ? 1 : 0.3;
  // Wings alternate each tick so motion reads at low frame rates.
  const beat = steps % 2 === 0 ? 1 : -1;
  ctx.fillStyle = "rgba(186, 214, 255, 0.5)";
  ctx.beginPath();
  ctx.ellipse(-3, -6, 7.5, 3.4, -0.45 * beat, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.ellipse(-3, 6, 7.5, 3.4, 0.45 * beat, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#20242e";
  ctx.beginPath();
  ctx.ellipse(-1.5, 0, 8.5, 4.8, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#39404f";
  ctx.beginPath();
  ctx.ellipse(-6, 0, 3.4, 4.2, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#f2b134";
  ctx.beginPath();
  ctx.arc(6.5, -2.6, 2.6, 0, Math.PI * 2);
  ctx.arc(6.5, 2.6, 2.6, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#1b1f28";
  ctx.beginPath();
  ctx.arc(7.4, -2.6, 1, 0, Math.PI * 2);
  ctx.arc(7.4, 2.6, 1, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}
