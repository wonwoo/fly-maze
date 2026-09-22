# Fly Maze

*English · [한국어](README.ko.md)*

A fixed 160-cell circuit measured in the **MaleCNS v1.0** fruit-fly connectome steers an original maze game through a small trained readout. Twenty maze observations drive gustatory, visual and proprioceptive cells, the activity propagates over 2,319 measured connections, and 32 descending cells feed a linear readout that picks a direction. **Only the 148 readout parameters learn. The biological wiring never changes.**

- Original maze, rules, artwork and code. No third-party game assets. A fresh maze is generated per course.
- **20 engineered observations → 80 driven MaleCNS cells → 48 bridge cells → 32 descending cells, plus 4 ghost-proximity channels → 36 → 4 linear readout → softmax over 4 directions.**
- Policy gradient (REINFORCE with a whitened return baseline and Adam) over the readout, with potential-based reward shaping on distance to the nearest hunting ghost.
- Held-out benchmark on 300 unseen generated courses, with silenced-circuit, untrained, handwritten, random and idle controls, plus degree-preserving rewiring and contact-count controls.
- Static Vite site; a cross-entropy trainer also runs in a Web Worker in the browser.

## Held-out results

300 held-out courses (seeds 2,100,001 to 2,100,300), 1,000-step cap, about 195 pellets per generated course. Weights were frozen before this evaluation; these seeds appear in neither training nor validation.

| Controller | Cleared / 300 | Mean score | Mean pellets | Mean survival (steps) |
| --- | ---: | ---: | ---: | ---: |
| Connectome + trained readout | **61** | **1816.53** | **133.65** | **213.38** |
| Same readout, circuit silenced | 0 | 40.87 | 3.58 | 80.34 |
| Untrained readout, random initialisation | 0 | 74.43 | 5.67 | 93.83 |
| Readout this run continued from | 60 | 1817.83 | 133.17 | 211.92 |
| Handwritten greedy-pellet baseline | 34 | 1440.60 | 112.50 | 141.13 |
| Uniform random actions | 0 | 261.63 | 23.54 | 95.55 |
| Idle (keep heading) | 0 | 23.17 | 2.08 | 78.77 |

Published champion: training seed **20260921**, update **800** of 12,000, validation score **1574.80** on 512 fixed validation courses at the training temperature of 1. The finished game and every row above play those same parameters at temperature **0.1**, chosen afterwards on those validation courses and documented below. Training took 2,590 seconds on the development Mac (not a portable performance guarantee). Per-course results and the full learning curve are in [`public/benchmarks`](public/benchmarks).

**It clears nearly twice as often as the handwritten baseline and collects more on the way**: 61 courses of 300 against 34, at 133.65 pellets against 112.50. Both run the same 300 seeds, so the comparison is paired, and the two still clear largely different sets: 53 courses only the readout finishes, 26 only the baseline finishes, 8 that both do and 213 that neither does. A 27-course margin against a standard error of `sqrt(53 + 26)` = 8.9 clears is about three standard errors, so this one is a real gap rather than a coin flip. What has not changed is how the courses end: 239 of the readout's 300 end in capture and none reach the step cap.

**The evaluation was widened for a reason.** On the first 100 held-out courses this checkpoint scores 9 clears and its predecessor 10, which would have ranked the predecessor first; on all 300 they score 33 and 28. All four counts are quoted at temperature 1, which is what both were measured at when the comparison was made. A clear count on 100 courses carries a standard error of about 2.9, so every 100-course number in this file should be read with that width.

### Why 512 validation courses

The champion is chosen **by validation score**, never by held-out score, and the width of that validation set turned out to matter more than any change to the objective:

| Validation courses | Standard error of the score | Held-out clears / 300 |
| ---: | ---: | ---: |
| 128 | ~200 | 28 |
| 512 | ~87 | 33 |

Both clear counts are at temperature 1, the temperature the two checkpoints were compared at when the width was chosen. **At the deployment temperature of 0.1 the same two clear 60 and 61, so this five-course gap is one course.** The clear count is the weaker half of the case and it does not survive the temperature change; what does survive is the structural argument below, that selecting the best of 240 measurements with a standard error of 200 selects the measurement, and the 1584.8 that fell to 1275.6 when re-measured. The two rows are held at one temperature because the table isolates validation width, not because 33 is the current figure.

At 128 courses the trainer takes the best of 240 measurements whose standard error is about 200, which selects a lucky draw rather than a better readout. Two checkpoints recorded at 1584.8 and 1973.5 on 128 courses scored 1275.6 ± 84.8 and 1446.4 ± 87.1 when re-measured on 512, and four runs continued from the 1973.5 checkpoint failed to beat its recorded number **even once in 12,000 updates**, because that number was a peak rather than a level. Those five figures are the one claim in this file that cannot be rechecked here: neither checkpoint's weights were kept, and the 128-course regime they were selected under is no longer in [`src/train/policy.ts`](src/train/policy.ts), so rerunning the stages today lands on different weights. They are reported as measured at the time and nothing in the repository reproduces them. Every later measurement keeps its weights for exactly this reason. Validating every 200 updates over 512 courses costs exactly what every 50 over 128 did.

### Does the measured wiring carry the play?

The trained readout is held fixed and only the wiring underneath it is replaced ([`scripts/control-rewire.ts`](scripts/control-rewire.ts)), on the same held-out courses:

| Wiring under the same trained readout | Cleared / 300 | Mean pellets |
| --- | ---: | ---: |
| Measured circuit | 61 | 133.65 |
| Degree-preserving rewiring, 5 replicates | 0 | 5.04 ± 2.09 |
| Circuit silenced | 0 | 3.58 |

Shuffling who connects to whom, while keeping every cell's in-degree and every contact count, costs almost all of the play: 133.65 pellets fall to 5.04, close to the 3.58 that silencing the circuit outright leaves. A scrambled graph and no graph at all are worth about the same to this readout, and neither clears a single course.

This is the control that isolates the wiring. It leaves the readout, the observation encoder and the proximity gate exactly as they ship and changes nothing but which cell connects to which, whereas the silenced row above removes the circuit and the gate together. One caveat on the shuffle: moving the presynaptic endpoints can land an edge on its own target or onto a pair another edge already occupies, so each of the five replicates carries 17 to 25 self-connections and 166 to 192 pairs holding more than one edge, where the measured graph has neither.

Its complement keeps every edge's two endpoints and replaces only the number measured on it ([`scripts/control-weights.ts`](scripts/control-weights.ts)), so the graph stays the measured graph down to the last recurrent and single-contact edge:

| Contact counts under the same graph and the same trained readout | Cleared / 300 | Mean pellets |
| --- | ---: | ---: |
| Measured counts | 61 | 133.65 |
| Measured counts dealt back out at random, 5 replicates | 0 | 5.87 ± 1.94 |
| Every edge flattened to one contact | 0 | 8.83 |

The graph alone is not enough. Dealing the same multiset of strengths back over the same edges in a different order costs as much as scrambling the graph did, so what the readout learned to read is the strength measured on each particular connection, not the shape of the network around it.

**This does not show that biological topology beats an artificial network of the same size.** It shows that this readout depends on the wiring it was trained on. The stronger claim needs a rewired graph trained from scratch on an equal budget, which this project has not run.

### Generated mazes against the one built-in maze

[`scripts/generalization.ts`](scripts/generalization.ts) reruns the same controllers on the same
seeds with maze generation turned off, so only the layout differs:

| Controller | Generated mazes | The single built-in maze |
| --- | ---: | ---: |
| Connectome + trained readout | 61 / 300 | **79 / 300** |
| Handwritten greedy-pellet baseline | 34 / 300 | 39 / 300 |

On one fixed layout the readout clears about a third more often, and its lead over the baseline widens. It never
trained on that layout — training generates a maze per course — so this is not memorisation of it;
the fair reading is that a fixed layout is an easier problem, and that the readout gains more from
that than a baseline which recomputes a shortest path every tick. Every headline number in this
file is the harder generated-maze case.

### What the escape pathway was, and what replaced it

The four extra readout inputs now carry a per-direction binary gate: 1 when a hunting ghost is within three steps that way, 0 otherwise. A linear readout cannot form the product of two channels, so a condition it is meant to act on has to arrive already thresholded. The published checkpoint uses these; `escapeChannels: "pathway"` in [`src/game/rules.ts`](src/game/rules.ts) restores what they replaced, described below.

That predecessor is a second measured circuit, the 5,241-cell looming-to-giant-fibre pathway ([`scripts/build-escape-circuit.py`](scripts/build-escape-circuit.py)), driven separately: threat in the player's left visual half drives the left looming population, threat on its right drives the right one, and the two DNp01 giant-fibre cells answer. The drive onto the giant fibre is hemisphere-pure: of the 284 measured edges from those looming cells onto the two DNp01 cells, 155 run left to left and 129 right to right, and none cross. The pathway graph around them is not, with 14,468 of its 74,697 edges crossing, so what is separate is the looming-to-giant-fibre drive rather than the circuit as a whole.

That separation buys less than it sounds like. The crossing edges reach the giant fibres with almost nothing: raising the opposite drive from 0 to 1 moves each one by 0.006 to 0.013 across the whole drive plane, so each giant fibre is a function of its own side in all but the third decimal and the pathway applies a saturating squash per side and little more. Which side a threat is on is decided by `directionThreat` and a heading rotation in TypeScript, not by the wiring. Replacing the pathway confirms it:

Measured on 100 courses under the last checkpoint that was trained with the pathway in place, three
checkpoints before the one shipping now:

| Escape channel source | Cleared / 100 | Mean pellets |
| --- | ---: | ---: |
| Measured pathway (as trained) | 5 | 112.55 |
| Channels held at zero | 5 | 110.27 |
| Raw direction threat, pathway skipped | 4 | 115.47 |
| Pathway edges shuffled | 6 | 111.84 |

It is kept for one reason: it is the only place the readout learns **absolute** distance from. The `threat` feature channel is relative, so the most dangerous direction reads 1 whether a ghost is two steps away or fifty — over all 10,653 held-out states with the nearest hunting ghost 15 or more steps away, the most dangerous direction read a full 1.0 every single time, which the relative encoding guarantees: it scores each direction against the most dangerous one, so that one always comes out at exactly 1. A least-squares probe recovers ghost distance from the four threat channels at R² −0.001, which is no better than predicting the mean, and from these four escape channels at R² 0.181 ([`scripts/probe.ts`](scripts/probe.ts)).

Supplying that distance through the threat channel as a continuous value was tried and made things worse (see **What was tried**). Supplying it as a binary gate instead is what the published checkpoint now does.

## What is real, and what is modeled?

**Real source data.** MaleCNS v1.0 cell identities, classes, types, soma sides and positions, transmitter annotations, and all 2,319 selected directed connections (42,965 synaptic contacts). Data creators: FlyEM / HHMI Janelia and collaborators, CC BY 4.0. Raw files are downloaded separately (about 1.1 GB) and pinned by SHA-256 in [`public/data/manifest.json`](public/data/manifest.json). Rebuilding from those files reproduces `public/data/circuit.json` byte for byte.

**Engineered model.** Which 160 cells to keep, the mapping from game state to driven cells, the signed normalized leaky-tanh dynamics, the trainable readout and the game itself. Sixteen cells per channel and five channels are choices, not measurements: the fly has no Pac-Man neurons, and everything connecting these cells to this game was written here. Computed activity is dimensionless; it is neither firing rate nor membrane voltage. No fly, muscle, physiological receptive field, synaptic plasticity or biological learning is claimed.

## Measured circuit and selection

Selection is deterministic and uses anatomy and transmitter annotations only, before any training ([`scripts/build-connectome.py`](scripts/build-connectome.py)):

1. **Food inputs (16 cells).** Gustatory sensory cells (`class = gustatory`, subclass labellar bristle, taste peg or pharyngeal sensillum) ranked by two-hop contact reach onto descending neurons. Selected: PhG1c ×4, PhG9 ×4, PhG1b ×2, PhG5 ×2, and one each of PhG8, LB1c, LB3c, LB3d.
2. **Threat inputs (16 cells).** Looming-sensitive visual projection neurons, ranked by direct contacts onto descending neurons. Candidate pool `LPLC2`, `LC4`, `LC16`; selected: 16 × LC4.
3. **Wall inputs (16 cells).** Visual projection neurons ranked the same way. Candidate pool `LC11`, `LC15`, `LC17`, `LC21`; selected: 15 × LC11, 1 × LC21.
4. **Prey inputs (16 cells).** Small-target pursuit neurons ranked the same way. Candidate pool the `LC10` group; selected: 14 × LC10d, 1 × LC10c-1, 1 × LC10c-2. They carry the fleeing-ghost channel.
5. **Self inputs (16 cells).** `mechanosensory_proprioceptive` cells ranked the same way, carrying the current heading. Selected: SApp10 ×12, SApp19 ×2, SApp ×2. Enabled by `FLY_MAZE_SELF_CHANNEL=1`.
6. **Readout (32 descending neurons).** Sixteen strongest direct targets of the selected visual inputs, then the strongest two-hop targets of the selected gustatory inputs, then whatever combined rank is needed to reach 32. The gustatory pass is not a clean sixteen: its quota counts every cell already picked that appears anywhere in the gustatory ranking, so each visual pick that also reaches gustatory targets consumes one of its sixteen slots in advance, and the combined pass makes up the difference ([`scripts/build-connectome.py`](scripts/build-connectome.py)). The split is therefore sixteen visual and a smaller gustatory block; the exact division needs the raw tables to recover. The shipped set is unaffected, but `READOUT_COUNT` cannot be changed without re-reading that condition. Two each of DNp01, DNp02, DNp04, DNp05, DNp35, DNa10, DNg27, DNg38, DNg60, DNg103, DNpe049, DNge036, DNge055, and one each of DNp03, DNp06, DNp11, DNg40, DNg68, DNg70.
7. **Bridges (48 cells).** Sixteen strongest gustatory-to-readout carriers and thirty-two strongest general input-to-readout carriers, scored by the minimum of summed input and output contacts.
8. **Every** measured directed edge among the 160 cells is retained, including recurrent and single-contact edges. Nothing is synthesized.

Every one of the twenty channels reaches all 32 readout cells through nonzero-sign edges. Transmitter annotations: 125 acetylcholine, 25 GABA, 10 glutamate. Each channel's four cells are assigned round-robin by rank; the assignment of directions to particular cells is an arbitrary engineering encoder with no biological interpretation. The threat, wall and prey pools and the bridge candidates are drawn only from cells with a recorded `somaLocation`; the gustatory and proprioceptive pools are not filtered that way.

## Observation encoder

Twenty features in [0, 1], one per (modality, direction), in [`src/game/features.ts`](src/game/features.ts). Directions are up, right, down, left. `n` is the neighbouring cell in that direction; every channel reads 0 when `n` is a wall.

| Modality | Feature | Driven cells |
| --- | --- | --- |
| food | `1 - clamp((pelletDistance(n) - bestNeighbourDistance) / 3)`, so the direction on the shortest path to a pellet reads 1 and one three steps longer reads 0 | gustatory |
| threat | `max(0, 1 - (ghostDistance(n) - worstNeighbourDistance) / 3)`, relative danger, so the most dangerous direction always reads 1 regardless of how far the ghost actually is | LC4 |
| wall | 1 when `n` is open to the player | LC11 / LC21 |
| prey | `(frightenedTicksLeft / 40) * (1 - clamp(fleeingGhostDistance(n) / 12))`, closeness of the nearest frightened ghost weighted by remaining frightened time; 0 when no ghost is frightened | LC10 |
| self | 1 for the current heading, 0 for its reverse, 0.5 to either side | SApp |

These are structured game observations, not pixels. A driven cell receives `u = 2 * (feature - 0.5)`; every other cell receives zero external drive. The threat channel's relative encoding is a known weakness, measured above.

## Dynamics

With measured contact count `c[j,i]` and presynaptic sign `s[j]` (acetylcholine +1, GABA and glutamate −1, anything else 0):

```text
W[j,i]   = c[j,i] * s[j] / sum_k( c[k,i] * |s[k]| )
h_new[i] = 0.3 * h[i] + 0.7 * tanh( u[i] + 1.4 * sum_j W[j,i] * h[j] )
```

Three synchronous iterations run per game decision. State resets to zero at the start of each course and persists between decisions. The readout receives `4 * h` from the 32 descending cells and four escape channels, and computes 36 → 4 linear scores; a softmax over those scores draws the direction. Silencing the circuit forces `h = 0` **and holds the four escape channels at zero as well** ([`src/brain/controller.ts`](src/brain/controller.ts)), so the silenced readout sees an all-zero input and plays on its four biases alone. That control therefore removes the circuit and the proximity gate together, and does not isolate either one. The shape of this model is the one Fly Dino uses; the equations, gains and iteration count here are simplified assumptions of this project's own, not calibrated physiology.

**The direction is drawn from a softmax at temperature 0.1, not taken as the highest score.** The temperature is a deployment choice rather than a trained parameter, so [`scripts/tune-temperature.ts`](scripts/tune-temperature.ts) picks it on the 512 validation courses and never on the held-out ones, writing the sweep to [`public/benchmarks/temperature.json`](public/benchmarks/temperature.json). Every temperature plays the same courses, so the gaps are paired course by course:

| Temperature | Mean validation fitness | Paired gap against 1 | Cleared / 512 |
| ---: | ---: | ---: | ---: |
| 0, the argmax | 1795.2 ± 100.0 | +220.4 ± 138.7 | 90 |
| **0.1, shipped** | **1865.1 ± 102.6** | **+290.4 ± 138.1** | **97** |
| 0.25 | 1785.3 ± 98.9 | +210.5 ± 131.3 | 88 |
| 0.5 | 1730.8 ± 98.9 | +156.0 ± 131.4 | 85 |
| 1, the training temperature | 1574.8 ± 93.1 | 0 | 74 |
| 1.4 | 1128.1 ± 75.6 | −446.7 ± 113.8 | 43 |

A little sampling beats both extremes. Pure argmax leaves the fly stuck where two directions score nearly the same, and temperature 1 throws away too much of what was learned; every temperature below 1 beats 1, and 1.4 is worse than all of them. What the sweep does **not** resolve is which low temperature is best. Taken pairwise against 0.1 on the same courses, the argmax is 70.0 ± 116.6 behind, 0.25 is 79.9 ± 125.9 behind, 0.15 is 119.3 ± 94.7 and 0.5 is 134.3 ± 133.4: not one of them reaches two standard errors, and the column means are not monotone in temperature either. So what the sweep supports is that the low band beats temperature 1 and is otherwise indistinguishable within itself; 0.1 is the value taken from that band rather than a measured optimum. This is the same trap the widened validation set was introduced to avoid, one level down: taking the largest of eleven noisy measurements selects a lucky draw as readily as taking the largest of 240 did. Every controller in the benchmark, controls included, runs at 0.1 ([`DEPLOY_TEMPERATURE`](src/brain/controller.ts)), so the rows differ only in wiring and parameters, and `npm run check:assets` fails when the sweep no longer names that value or the checkpoint that ships. Sampling is seeded from the course seed, so a sampled run is as reproducible as a deterministic one.

## Game

A fresh maze is generated per course ([`src/game/generate.ts`](src/game/generate.ts)): a depth-first carve on an odd lattice, extra loops added to a target junction share, a three-row ghost house with a door, and dead ends extended rather than walled off, giving about 200 open cells, 52 junctions, 2.5 dead ends and 195 pellets, six of them large. Each course also seeds its own player start, drawn uniformly from the open cells at least eight ghost-steps from the house, and adds up to 30 ticks of seeded jitter to each ghost's release.

The player moves one cell per tick and keeps its heading when the requested direction is blocked. Three ghosts leave the house at ticks 0, 50 and 100, move on three of every four ticks, and alternate between 40 ticks of heading to a home corner and 80 ticks of chasing. At each junction a ghost takes the path-shortening step with probability 0.8, 0.6 or 0.4 (one value per ghost), otherwise a random open direction, never reversing unless cornered. A pellet scores 10, a power pellet 50, and clearing the maze adds 1,000. Power pellets frighten every ghost outside the house for 40 ticks; frightened ghosts flee at half speed and can be eaten for 200 points, after which they wait 30 ticks in the house. Contact is resolved twice per tick, once where the player lands and once after each ghost moves, so a player stepping onto a ghost cannot pass through it when that ghost leaves in the same tick. A course ends on capture, on clearing, or after 1,000 ticks.

All of these rules live in [`src/game/rules.ts`](src/game/rules.ts); a checkpoint records the rules and the objective it was trained under, and the page plays under the published checkpoint's rules. Every course is seeded and rendering never touches the update path, so browser play and headless rollouts follow identical trajectories.

## Training protocol

Policy gradient over the 148 readout parameters ([`src/train/policy.ts`](src/train/policy.ts)):

- 12,000 updates, 32 episodes per update, course seeds drawn from 1 to 900,000 by a seeded generator.
- Experience is collected through the same controller the finished game runs, sampling its own softmax at the training temperature of 1. The game plays those parameters at 0.1, a deployment choice made afterwards on validation courses, so training explores more widely than the shipped policy acts. Each step is credited with the discounted reward that followed it (discount 0.99), the batch's returns are centred and scaled, and Adam moves the parameters with an entropy bonus of 0.01.
- The reward is paid at the tick it is earned and **sums over an episode to exactly the fitness** the cross-entropy trainer maximises, which [`tests/policy.test.ts`](tests/policy.test.ts) pins to six decimal places. The objective is unchanged; only the moment of delivery is.
- Every 200 updates the parameters are evaluated on validation seeds **1,100,001 to 1,100,512**. The champion is replaced only on a strictly higher mean validation score.
- Nothing except the readout parameters changes. Gameplay uses inference only.

The objective ([`src/train/rollout.ts`](src/train/rollout.ts)) states the goal, not a strategy: 10 per pellet, 200 per frightened ghost eaten, 3,000 for clearing plus 2 per tick left on the clock, 0.1 per tick alive, and 500 charged on capture.

**Reward shaping.** On top of that objective each step also pays `10 × (0.99 × Φ(next) − Φ(now))`, where Φ is the distance to the nearest hunting ghost divided by the threat range and capped at one. It is zero in a terminal state and zero while the ghosts are frightened, and it is one when no hunting ghost can reach the player at all ([`safetyPotential`](src/train/policy.ts)). Being the difference of a potential, this leaves the best policy exactly where it was (Ng, Harada and Russell, 1999) while turning a signal that arrived once a course — the 500 charged on capture — into one that arrives every tick. It enters training only: validation and the benchmark score the unshaped objective, so a shaped run cannot flatter itself.

**The published checkpoint is the third run in a chain**, not a single training run from random weights:

| Stage | Seed | Change | Learning rate | Validation set |
| --- | --- | --- | ---: | ---: |
| 1 | 20260917 | from random weights, no shaping | 0.02 | 128 courses |
| 2 | 20260919 | continued, shaping 10 | 0.002 | 128 courses |
| 3 (published) | 20260921 | continued, shaping 10 | 0.002 | 512 courses |

Continuing costs less than starting over because the pellet-collecting weights are already there. Rerunning stage 3's settings from random weights, same seed and same 12,000 updates but no `FLY_MAZE_INIT`, reaches 657.4 validation and clears 5 of 300 at temperature 1, against the continued run's 1574.8 and 33; at the shipped 0.1 it clears 16 against 61. Its weights are kept in [`public/benchmarks/experiments/policy/from-random-seed20260921.json`](public/benchmarks/experiments/policy/from-random-seed20260921.json), so the row can be rechecked. An earlier version of this file put that run at 1360.4 and 18 clears, roughly twice what the rerun gets; the original seed was never recorded, so this is an independent run under the same settings rather than a contradiction of that one, and the conclusion it was offered for holds by a wider margin than claimed. The last stage, by contrast, earned little: at 0.1 the checkpoint stage 3 continued from clears 60 of 300 against the published 61, so the chain's gain was made in the first two stages. `FLY_MAZE_INIT` points a run at the checkpoint it should continue from.

A cross-entropy method trainer remains in [`src/train/cem.ts`](src/train/cem.ts) and is what the browser training panel runs.

## What was tried

Each row is a one-factor change, trained to completion and evaluated on the held-out courses. Rows
from before the evaluation was widened are marked; those were measured on 100 courses, where the
standard error on a clear count is about 2.9.

| Change | Result |
| --- | --- |
| **Cross-entropy method → policy gradient** | **63.6 → 112.6 pellets, 0 → 5 clears.** The single largest gain. |
| Escape pathway as an override → as readout inputs | Validation 428.9 → 833.0 |
| Readout 36-12-4 → 36-4 linear | Champion 434.5 → 818.4 at 150 generations (CEM) |
| Readout 36-4-4 / 36-24-4 | 403.7 / 473.3 at 150 generations (CEM) |
| Argmax → sampling the learned softmax | 83.9 → 97.3 pellets, 1 → 3 clears. **The published checkpoint reverses this**: it scores worst at temperature 1 and best at 0.1, which is why the shipped temperature is now chosen by a sweep. |
| Courses per generation 6 → 24, validation 8 → 32 (CEM) | Elite selection captures 90% of the available gain instead of 68% ten generations in, remeasured by [`scripts/cem-noise.ts`](scripts/cem-noise.ts) |

Seven one-factor changes were then trained to the full 12,000 updates from seed 20260917 and benchmarked on the held-out courses **as they then stood, 100 of them**. **Every one of them lost to the configuration published at the time**, and each has its artifact in [`public/benchmarks/experiments/policy`](public/benchmarks/experiments/policy):

| Change | Cleared / 100 | Mean pellets | Mean steps |
| --- | ---: | ---: | ---: |
| **Configuration published at the time** | **5** | **112.55** | 185.82 |
| Per-tick reward +0.1 → −0.5 | 4 | 104.79 | 180.97 |
| Threat channel relative → hybrid | 3 | 102.82 | 175.97 |
| Threat channel relative → absolute, range 12 | 2 | 101.14 | 158.49 |
| Capture penalty 500 → 2000 | 2 | 100.82 | 202.88 |
| Per-tick reward +0.1 → 0 | 0 | 99.43 | 163.51 |
| Threat channel relative → absolute, range 20 | 2 | 93.87 | 144.19 |
| Readout 36-4 linear → 36-12-4 | 2 | 88.51 | 153.18 |

### Trying to teach avoidance

The readout barely avoids ghosts where it counts, and the information to do so is there: a least-squares probe on the same 32 readout-cell activities picks a direction that increases distance from the nearest ghost **97.0%** of the time in the states that matter, while the policy itself manages 55.5% over one to three cells, and 1.5% with the ghost one cell away. Two ways of closing that gap were trained to completion and neither worked.

| Attempt | Escapes at 1-3 cells | Cleared | Pellets |
| --- | ---: | ---: | ---: |
| Supervised auxiliary loss, weight 0.5 | 63.8% | 0 / 100 | 72.22 |
| Supervised auxiliary loss, weight 2 | 76.9% | 0 / 100 | 66.39 |
| Supervised auxiliary loss, weight 8 | 80.8% | 0 / 100 | 53.12 |
| Reward shaping on ghost distance | 46.8% | 10 / 100 | 114.32 |
| Shaping with a ways-out term, weight 0.6 | 47.5% | 29 / 300 | 121.24 |
| Shaping with no ways-out term (published) | 50.3% | 33 / 300 | 120.35 |

Every row of that table was measured at temperature 1, the temperature training samples at, so the rows compare training configurations rather than the shipped deployment; the published row reads 61 clears and 133.65 pellets at the shipped 0.1.

**Teaching the escape directly taught the wrong thing.** Naming the escaping directions as targets and adding a cross-entropy term on them lifted escapes at one to three cells from the 40.1% of the checkpoint published at the time to 80.8% — and lost every single course, all hundred of them ending in capture. Taking the step that gains a cell is not the same as surviving: done every tick it walks into dead ends. A label that is right about each step and wrong about the sequence is worse than no label.

**Shaping helped, but not with avoidance.** It bought pace rather than escapes — 172.5 steps per course became 189.4 — and time near ghosts went up, not down: close encounters rose from 1.79 to 2.05 per course and the mean distance to a ghost fell from 12.91 to 12.44. What improved was surviving one, from 56.4% to 62.4%.

**Adding the number of ways out of the current cell to the potential did not help**, at any of three weights, against a control that was identical but for that term. The idea follows from the first failure — a dead end should score as dangerous even at a distance — but this encoding of it did not carry.

So the shipped readout avoids at the two distances where the proximity gate still tells directions apart, and nowhere else. Broken out by the distance to the nearest hunting ghost, rather than grouped, the picture is sharper than the grouped numbers suggest:

| Ghost distance | Escapes | Uniform draw | States |
| ---: | ---: | ---: | ---: |
| 1 cell | **1.5%** | 53.7% | 269 |
| 2 cells | **6.4%** | 52.2% | 283 |
| 3 cells | **80.1%** | 53.1% | 1,157 |
| 4 cells | **82.7%** | 52.8% | 2,036 |
| 5 cells | 47.3% | 52.8% | 1,726 |
| 6 cells | 52.3% | 53.3% | 1,353 |
| 11 cells or more | 55.0% | 52.4% | 13,209 |

The proximity gate accounts for that shape, though not by firing or not firing. The gate measures each *neighbouring* cell against `nearDistance`, so it is open whenever the player itself stands one to four cells from a hunting ghost and shut from five out: it fires in 100% of states at all four of those distances. What changes across them is whether it is selective. At one and two cells every legal direction is already within range, so the gate is on everywhere at once and says nothing about which way the danger lies. At three and four cells it is on for only some of them, in 91.4% and 96.1% of states. The escape rate tracks that selectivity and not the firing: 1.5% and 6.4% where the gate saturates, 80.1% and 82.7% where it discriminates, chance again beyond its range. Grouping one to three cells into a single 55.5% against 53.0% averages the collapse at one and two cells together with the escape at three and hides both, which is why the rows are broken out. Of its 239 deaths, 230 happened with an escaping move available. The only honest reading is that it collects pellets well, steers away only while the gate can still point somewhere, and is caught once the ghost is close enough that it cannot.

Three of the earlier results were predictions that the measurements above seemed to support, and all three were wrong.

**Giving the fly absolute distance did not help.** The threat channel really does carry no distance — R² −0.001, and a full 1.0 on the most dangerous direction in every one of the 10,653 held-out states with the nearest hunting ghost 15 or more steps away. Supplying it, by either encoding, cost 10 to 19 pellets and 2 or 3 clears. A plausible reading is that distance is only worth having if avoidance can be learned from it; with 95 of 100 courses ending in capture either way, the channel spent its dynamic range on information the readout could not act on. The escape channels still carry absolute distance, and removing them costs little, which points the same way.

**Charging more for dying did not help.** At a 2,000 penalty the fly survived longer than any other variant (202.9 steps) and still collected less and cleared less. Its validation score went to −16.9, meaning the death term had come to dominate the objective and the collection signal was barely visible inside it.

**Adding a hidden layer did not help.** The circuit is already nonlinear — tanh, three iterations, state carried between decisions — so the readout does not have to be. A 12-unit hidden layer triples the search space and scored worst of all eight.

Earlier experiments against the cross-entropy configuration are in [`public/benchmarks/experiments`](public/benchmarks/experiments), each written by `npm run experiment -- <name>` ([`scripts/experiment.ts`](scripts/experiment.ts), which lists its variants when run without one). They were run under a different trainer, readout shape and observation encoder, and do not describe the system documented here.

Three of these are worth spelling out.

**Why the cross-entropy method stalled.** It scores a candidate on whole games, so 300-odd decisions get one number, and that number is mostly the mazes the generation drew. [`scripts/cem-noise.ts`](scripts/cem-noise.ts) measures the selection signal it had to work with: the spread of one fixed candidate's own score across independent draws of the same mazes, the spread of true fitness across a generation's candidates, and how faithfully a generation's ranking matches the true one. True fitness here is the mean over 256 fresh mazes.

| Warmed up | Mean sigma | Courses | Draw noise | Candidate spread | Rank correlation | Elite capture |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 generation | 0.743 | 6 | 93.1 | 67.3 | 0.87 | 77% |
| 1 generation | 0.743 | 24 | 35.0 | 67.3 | 0.88 | 93% |
| 10 generations | 0.368 | 6 | 140.1 | 91.4 | 0.83 | 68% |
| 10 generations | 0.368 | 24 | 58.8 | 91.4 | 0.91 | 90% |
| 40 generations | 0.085 | 6 | 211.2 | 37.2 | 0.12 | 17% |
| 40 generations | 0.085 | 24 | 93.7 | 37.2 | 0.18 | 20% |

The draw noise exceeds the spread it is supposed to resolve at every point, and the gap widens as the run proceeds: the proposal narrows, the candidates crowd together, and the noise does not shrink with them. By forty generations a six-course ranking correlates 0.12 with the truth and its elites capture 17% of the improvement that was available, which is selection on almost nothing. Widening the generation from six courses to twenty-four buys back a good deal of that early and little of it late. Paying the reward per step instead put the credit where the decision was and roughly doubled the held-out score without touching the circuit.

Earlier versions of this file quoted a single pair, a standard deviation of 17.2 against a spread of 5.6, and a rank correlation of 0.58. No code in this repository produced those, and the script above does not reproduce them at any of the three points; the argument they were offered for survives the remeasurement, but the numbers were not reproducible and have been replaced.

**The circuit was never the bottleneck.** A least-squares probe fitted afterwards on the 32 readout-cell activities picks the best direction 98.8% of the time, against 100.0% from the raw feature channels, both scored out of sample ([`scripts/probe.ts`](scripts/probe.ts), artifact in [`public/benchmarks/probe.json`](public/benchmarks/probe.json)). The information survives the wiring nearly intact; what was missing was a way to learn from it.

**Slowing the fly down does not speed it up.** Removing the per-tick reward, or charging for time, shortened courses (185.8 → 163.5 steps) but cost pellets and clears, so the 11.5% reversal rate is not simply wasted motion — 59.3% of reversals do move toward food.

## Run and reproduce

Node 22 or newer. Python with `uv` is needed only to rebuild the circuit from raw data.

```sh
npm ci
npm run dev                            # play at the printed local address
npm test                               # 47 tests plus the published-asset hash check
npm run benchmark                      # re-evaluate on the 300 held-out courses
npm run control:rewire                 # rewiring control, 5 replicates
npm run control:weights                # contact-count control, 5 replicates
npm run generalization                 # generated mazes against the one built-in maze
npm run validate:escape                # looming selectivity of the measured escape pathway
npm run tune:temperature               # pick the softmax temperature on validation courses
npm run probe                          # least-squares probes on the readout cells and channels
npm run cem-noise                      # selection noise the cross-entropy method worked against
npm run build                          # static site in dist/
```

Training writes into the directory given as the third argument, so pass one to keep the published
checkpoint intact. Reproducing the published readout means repeating the three stages in order,
each continuing from the last; a single run from random weights does not reproduce it.

```sh
npm run train-policy -- 20260917 12000 /tmp/stage1
FLY_MAZE_SHAPING=10 FLY_MAZE_LR=0.002 FLY_MAZE_INIT=/tmp/stage1/checkpoints/readout.json \
  npm run train-policy -- 20260919 12000 /tmp/stage2
FLY_MAZE_SHAPING=10 FLY_MAZE_LR=0.002 FLY_MAZE_INIT=/tmp/stage2/checkpoints/readout.json \
  npm run train-policy -- 20260921 12000 /tmp/stage3
npm run benchmark -- /tmp/stage3
```

Stages 1 and 2 were validated on 128 courses, which [`src/train/policy.ts`](src/train/policy.ts) no
longer does; rerunning them today validates on 512 and will not land on the same weights. Only two
sets of weights from the chain still exist: the published readout in
[`public/checkpoints/readout.json`](public/checkpoints/readout.json) and stage 2's, kept as
`initialParams` in [`public/benchmarks/training.json`](public/benchmarks/training.json). Stage 1's
weights were not preserved, so **the chain cannot be replayed from its own artifacts** — the
benchmark it produced is in
[`public/benchmarks/experiments/policy/prox128-seed20260917.json`](public/benchmarks/experiments/policy/prox128-seed20260917.json)
and that is all that remains of it. One-factor
experiments take `FLY_MAZE_SHAPING`, `FLY_MAZE_OPENNESS`, `FLY_MAZE_AVOIDANCE`, `FLY_MAZE_LR`,
`FLY_MAZE_CAUGHT`, `FLY_MAZE_THREAT_ENCODING`, `FLY_MAZE_THREAT_RANGE` and
`FLY_MAZE_ESCAPE_CHANNELS` without a rebuild.

### Rebuilding the circuit from raw MaleCNS data

```sh
mkdir -p /tmp/malecns
base=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome
curl -fL "$base/body-annotations-male-cns-v1.0-minconf-0.5.feather" -o /tmp/malecns/body-annotations-male-cns-v1.0-minconf-0.5.feather
curl -fL "$base/connectome-weights-male-cns-v1.0-minconf-0.5.feather" -o /tmp/malecns/connectome-weights-male-cns-v1.0-minconf-0.5.feather
curl -fL "$base/body-neurotransmitters-male-cns-v1.0.feather" -o /tmp/malecns/body-neurotransmitters-male-cns-v1.0.feather
FLY_MAZE_SELF_CHANNEL=1 npm run build:connectome -- /tmp/malecns public/data 32
npm run build:escape -- /tmp/malecns public/data/pathway/circuit.json 10
```

The builder writes `public/data/circuit.json` and `public/data/manifest.json` with SHA-256 hashes of the raw tables and the exported graph. A changed circuit invalidates the published checkpoint: `npm run check:assets` fails when the manifest, checkpoint and benchmark do not all name the circuit that ships, `parseCheckpoint` refuses a checkpoint trained on a different circuit or under a different readout shape, and `npm test` runs both. Retrain after any circuit change.

## Controls

Arrow keys or WASD take over manually at any time; Space restarts a finished manual course. The controller selector switches between the trained connectome, the silenced-circuit control, the handwritten baseline, random actions and manual play; automatic modes restart 1.5 seconds after a course ends. The page runs the same controller the benchmark measures, proximity channels included; the escape pathway is loaded and passed as well, and takes over whenever a checkpoint records `escapeChannels: "pathway"`. The training panel runs the cross-entropy trainer in a Web Worker, adopts each new champion immediately, and can export or import readout checkpoints as JSON.

## Repository layout

| Path | Contents |
| --- | --- |
| `scripts/build-connectome.py`, `scripts/build-escape-circuit.py` | Deterministic circuit extraction from MaleCNS raw tables |
| `scripts/train-policy.ts`, `scripts/train.ts`, `scripts/benchmark.ts` | Reproducible training and held-out evaluation |
| `scripts/control-rewire.ts`, `scripts/control-weights.ts`, `scripts/validate-escape.ts` | Wiring, contact-count and looming-selectivity controls |
| `scripts/tune-temperature.ts`, `scripts/generalization.ts`, `scripts/experiment.ts` | Deployment temperature sweep, generalization report, one-factor variant runner |
| `scripts/probe.ts`, `scripts/cem-noise.ts` | Least-squares probes on the readout, and the selection noise the cross-entropy method worked against |
| `src/game/` | Maze generator, seeded simulation, observation encoder |
| `src/brain/` | Circuit dynamics, escape pathway, readout, controllers and controls |
| `src/train/` | Policy gradient, cross-entropy method, rollouts, checkpoint format, Web Worker |
| `src/ui/`, `src/main.ts` | Canvas renderer, anatomical circuit view, page logic |
| `public/data/` | Extracted circuit, escape pathway and hash manifest |
| `public/checkpoints/`, `public/benchmarks/` | Published readout, learning curve, benchmark, wiring and contact-count controls, temperature sweep, probes, cross-entropy selection noise |
| `tests/` | Vitest suite, including exact reproduction of the first held-out courses |

## Known limitations

- **It avoids ghosts only while the proximity gate can still tell directions apart.** At three and four cells it escapes 80.1% and 82.7% of the time against about 53% for a uniform draw, and at one and two cells, where the gate is on for every legal direction at once and so points nowhere, it escapes 1.5% and 6.4%, far below chance. 239 of 300 courses end in capture, 230 of those with an escape available. Two attempts to teach it, described above, failed in opposite directions.
- **The shipped temperature is the maximum of an eleven-point sweep.** Paired against 0.1 on the same validation courses, no other temperature below 1 comes within two standard errors, so the sweep establishes that the low band beats temperature 1, not that 0.1 is the best value inside it.
- The threat channel carries no absolute distance (R² −0.001 for recovering ghost distance), so the fly knows which way danger lies but not how near it is.
- The measured escape pathway contributed almost nothing and has been replaced by an engineered proximity gate, so one of the two measured circuits is no longer in the loop.
- **The published checkpoint has one seed per stage.** Nothing here shows the three-stage chain reproduces; the gain from widening validation is the only result measured against a like-for-like control.
- Its champion arrived at update 800 of 12,000 and was never beaten, which is a short search however wide the validation set.
- Ten one-factor changes have been tried against the published configuration and all ten lost, so the current settings are a local best that nothing has beaten, not a tuned optimum.
- The rewiring control shows dependence on the trained wiring, not that biological topology is easier to learn from than an arbitrary graph.

## Attribution and licenses

- **FlyEM / HHMI Janelia and MaleCNS collaborators**: measured connectome and annotations, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Cite the dataset and the paper below when publishing results derived from this circuit.

### Citing the source data

- Dataset: **MaleCNS v1.0**, released 8 June 2026 by FlyEM (HHMI Janelia) with the University of Cambridge, the MRC Laboratory of Molecular Biology and Google Research, CC BY 4.0. Project page <https://male-cns.janelia.org/>, downloads <https://male-cns.janelia.org/download/>, interactive query <https://neuprint.janelia.org/?dataset=male-cns:v1.0>.
- Paper: Berg S., Beckett I. R., Costa M., Schlegel P., Januszewski M., Marin E. C., Nern A., *et al.* (111 authors). "Sexual dimorphism in the complete *Drosophila* male central nervous system connectome." *Cell*, 3 September 2026, doi [10.1016/j.cell.2026.08.015](https://doi.org/10.1016/j.cell.2026.08.015). Preprint, under its earlier title "Sexual dimorphism in the complete connectome of the *Drosophila* male central nervous system": bioRxiv, doi [10.1101/2025.10.09.680999](https://doi.org/10.1101/2025.10.09.680999).
- The three raw files this project consumes, and their SHA-256 digests, are listed in [`public/data/manifest.json`](public/data/manifest.json).
- **[Fly Dino](https://flydino.cobanov.dev/)**: prior work in the same vein; referenced while building this. No code copied.
- Maze, artwork and code in this repository: MIT License (see [`LICENSE`](LICENSE)). The list text of any index that links here does not relicense this repository.

Independent project. Not affiliated with HHMI Janelia, Google or any game publisher.
