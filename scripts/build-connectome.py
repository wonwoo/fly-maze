"""Extract a fixed MaleCNS v1.0 circuit for Fly Maze.

Usage: python scripts/build-connectome.py <raw-dir> [out-dir] [readout-count]

Selection uses anatomy and neurotransmitter annotations only. No game data or
training result influences which cells are retained. Every measured directed
edge among the selected cells is kept; no edge is synthesized.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

RAW = {
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "weights": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
}
CHANNELS = ["up", "right", "down", "left"]
CELLS_PER_CHANNEL = 4
READOUT_COUNT = int(sys.argv[3]) if len(sys.argv) > 3 else 16
BRIDGE_FOOD = 16
BRIDGE_GENERAL = 32
SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1}

FOOD_SUBCLASSES = {"labellar bristle", "taste peg", "pharyngeal sensillum"}
THREAT_TYPES = ["LPLC2", "LC4", "LC16"]
WALL_TYPES = ["LC11", "LC15", "LC17", "LC21"]
# Small-moving-target pursuit neurons carry the "edible fleeing ghost" channel.
PREY_TYPES = ["LC10a", "LC10b", "LC10c-1", "LC10c-2", "LC10d", "LC10e"]
# Proprioceptive mechanosensory cells carry the "current heading" (self-motion) channel.
SELF_CLASS = "mechanosensory_proprioceptive"
# The heading channel measurably hurt performance, so the code stays but the flag is off by
# default. The published circuit was built with FLY_MAZE_SELF_CHANNEL=1 and does contain all 16
# self cells; README.md gives the rebuild command that sets it.
INCLUDE_SELF = os.environ.get("FLY_MAZE_SELF_CHANNEL") == "1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main(raw_dir: Path, out_dir: Path) -> None:
    ann = feather.read_feather(raw_dir / RAW["annotations"])
    nt = feather.read_feather(raw_dir / RAW["neurotransmitters"])
    w = feather.read_feather(raw_dir / RAW["weights"])
    pre_col = next(c for c in w.columns if "pre" in c.lower())
    post_col = next(c for c in w.columns if "post" in c.lower())
    wt_col = next(c for c in w.columns if c not in (pre_col, post_col))
    edges = w[[pre_col, post_col, wt_col]].rename(columns={pre_col: "pre", post_col: "post", wt_col: "c"})
    edges = edges[edges["c"] > 0]
    print(f"edges {len(edges):,} using columns {pre_col},{post_col},{wt_col}")

    ann = ann.set_index("bodyId")
    nt_map = nt.set_index("body")["consensus_nt"].to_dict()

    dn_ids = set(ann.index[(ann["superclass"] == "descending_neuron") & ann["somaLocation"].notna()])
    print(f"descending neurons with soma: {len(dn_ids):,}")

    out_total = edges.groupby("pre")["c"].sum()
    to_dn = edges[edges["post"].isin(dn_ids)].groupby("pre")["c"].sum()

    def rank_by_dn_contact(pool: pd.Index) -> list[int]:
        s = to_dn.reindex(pool).fillna(0)
        df = pd.DataFrame({"score": s, "body": pool})
        df = df[df["score"] > 0].sort_values(["score", "body"], ascending=[False, True])
        return [int(b) for b in df["body"]]

    has_soma = ann["somaLocation"].notna()
    threat_pool = ann.index[ann["type"].isin(THREAT_TYPES) & has_soma]
    wall_pool = ann.index[ann["type"].isin(WALL_TYPES) & has_soma]
    food_pool = ann.index[(ann["class"] == "gustatory") & ann["subclass"].isin(FOOD_SUBCLASSES)]

    prey_pool = ann.index[ann["type"].isin(PREY_TYPES) & has_soma]
    threat_cells = rank_by_dn_contact(threat_pool)[: CELLS_PER_CHANNEL * 4]
    wall_cells = rank_by_dn_contact(wall_pool)[: CELLS_PER_CHANNEL * 4]
    self_pool = ann.index[(ann["class"] == SELF_CLASS)]
    prey_cells = rank_by_dn_contact(prey_pool)[: CELLS_PER_CHANNEL * 4]
    self_cells = rank_by_dn_contact(self_pool)[: CELLS_PER_CHANNEL * 4]
    prey_selection = "direct contacts onto descending neurons"
    self_selection = "direct contacts onto descending neurons"
    if len(self_cells) < CELLS_PER_CHANNEL * 4:
        se = edges[edges["pre"].isin(set(self_pool))]
        se = se.assign(reach=np.minimum(se["c"].to_numpy(), to_dn.reindex(se["post"]).fillna(0).to_numpy()))
        ss = se.groupby("pre")["reach"].sum()
        ss = ss[ss > 0].sort_values(ascending=False)
        self_cells = [int(b) for b in ss.index[: CELLS_PER_CHANNEL * 4]]
        self_selection = "two-hop contact reach onto descending neurons"
    if len(prey_cells) < CELLS_PER_CHANNEL * 4:
        pe = edges[edges["pre"].isin(set(prey_pool))]
        pe = pe.assign(reach=np.minimum(pe["c"].to_numpy(), to_dn.reindex(pe["post"]).fillna(0).to_numpy()))
        ps = pe.groupby("pre")["reach"].sum()
        ps = ps[ps > 0].sort_values(ascending=False)
        prey_cells = [int(b) for b in ps.index[: CELLS_PER_CHANNEL * 4]]
        prey_selection = "two-hop contact reach onto descending neurons"

    # Gustatory cells: two-hop reach onto descending neurons.
    food_edges = edges[edges["pre"].isin(set(food_pool))]
    mid_to_dn = to_dn.reindex(food_edges["post"]).fillna(0).to_numpy()
    food_edges = food_edges.assign(reach=np.minimum(food_edges["c"].to_numpy(), mid_to_dn))
    food_score = food_edges.groupby("pre")["reach"].sum()
    food_score = food_score[food_score > 0].sort_values(ascending=False)
    food_cells = [int(b) for b in food_score.index[: CELLS_PER_CHANNEL * 4]]
    if not INCLUDE_SELF:
        self_cells = []
    assert len(threat_cells) == len(wall_cells) == len(food_cells) == len(prey_cells) == 16, (len(threat_cells), len(wall_cells), len(food_cells), len(prey_cells))

    inputs = {"food": food_cells, "threat": threat_cells, "wall": wall_cells, "prey": prey_cells, "self": self_cells}
    input_set = set(food_cells) | set(threat_cells) | set(wall_cells) | set(prey_cells) | set(self_cells)

    # Readout descending neurons: direct visual contact + two-hop gustatory reach.
    vis_direct = edges[edges["pre"].isin(set(threat_cells) | set(wall_cells) | set(prey_cells)) & edges["post"].isin(dn_ids)].groupby("post")["c"].sum()
    fe = edges[edges["pre"].isin(set(food_cells))]
    mids = set(fe["post"])
    m2dn = edges[edges["pre"].isin(mids) & edges["post"].isin(dn_ids)]
    in_from_food = fe.groupby("post")["c"].sum()
    m2dn = m2dn.assign(reach=np.minimum(m2dn["c"].to_numpy(), in_from_food.reindex(m2dn["pre"]).fillna(0).to_numpy()))
    food_2hop = m2dn.groupby("post")["reach"].sum()
    readout: list[int] = []
    # The quota below counts every cell already chosen that APPEARS in this series' index, not the
    # ones chosen FROM it. The visual pass is unaffected because it runs first on an empty list, but
    # each visual pick that also reaches gustatory targets spends one of the gustatory pass's slots
    # in advance, and the combined pass makes up the shortfall. The shipped 32 are what this rule
    # produced; changing READOUT_COUNT without revisiting it will not split the halves evenly.
    # The wording emitted into circuit.json below is left as it shipped: rewriting it would change
    # the exported bytes, and with them the SHA-256 the published checkpoint is pinned to.
    for series in (vis_direct.sort_values(ascending=False), food_2hop.sort_values(ascending=False)):
        for b in series.index:
            if len([r for r in readout if r in series.index]) >= READOUT_COUNT // 2:
                break
            if int(b) not in readout and int(b) not in input_set:
                readout.append(int(b))
    combined = vis_direct.add(food_2hop, fill_value=0).sort_values(ascending=False)
    for b in combined.index:
        if len(readout) >= READOUT_COUNT:
            break
        if int(b) not in readout and int(b) not in input_set:
            readout.append(int(b))
    readout = readout[:READOUT_COUNT]
    readout_set = set(readout)

    # Bridge cells: strongest two-hop carriers from inputs to readouts.
    from_inputs = edges[edges["pre"].isin(input_set)].groupby("post")["c"].sum()
    from_food = edges[edges["pre"].isin(set(food_cells))].groupby("post")["c"].sum()
    to_readout = edges[edges["post"].isin(readout_set)].groupby("pre")["c"].sum()
    cand = pd.DataFrame({"inp": from_inputs, "food": from_food, "out": to_readout}).fillna(0)
    cand = cand[~cand.index.isin(input_set | readout_set) & cand.index.isin(ann.index[has_soma])]
    cand["general"] = np.minimum(cand["inp"], cand["out"])
    cand["foodpath"] = np.minimum(cand["food"], cand["out"])
    bridge_food = [int(b) for b in cand[cand["foodpath"] > 0].sort_values("foodpath", ascending=False).index[:BRIDGE_FOOD]]
    rest = cand[~cand.index.isin(bridge_food) & (cand["general"] > 0)].sort_values("general", ascending=False)
    bridge_general = [int(b) for b in rest.index[:BRIDGE_GENERAL]]
    bridges = bridge_food + bridge_general

    order = food_cells + threat_cells + wall_cells + prey_cells + self_cells + bridges + readout
    assert len(set(order)) == len(order), "overlapping selection"
    index = {b: i for i, b in enumerate(order)}

    sub = edges[edges["pre"].isin(index) & edges["post"].isin(index)]
    cells = []
    for b in order:
        row = ann.loc[b]
        soma = row["somaLocation"]
        if b in inputs["food"]:
            role, ch = "input", ("food", CHANNELS[inputs["food"].index(b) % 4])
        elif b in inputs["threat"]:
            role, ch = "input", ("threat", CHANNELS[inputs["threat"].index(b) % 4])
        elif b in inputs["wall"]:
            role, ch = "input", ("wall", CHANNELS[inputs["wall"].index(b) % 4])
        elif b in inputs["prey"]:
            role, ch = "input", ("prey", CHANNELS[inputs["prey"].index(b) % 4])
        elif b in inputs["self"]:
            role, ch = "input", ("self", CHANNELS[inputs["self"].index(b) % 4])
        elif b in readout_set:
            role, ch = "readout", None
        else:
            role, ch = "bridge", None
        ntx = nt_map.get(b)
        ntx = str(ntx) if isinstance(ntx, str) else "unknown"
        cells.append({
            "bodyId": int(b),
            "type": None if pd.isna(row["type"]) else str(row["type"]),
            "class": None if pd.isna(row["class"]) else str(row["class"]),
            "superclass": None if pd.isna(row["superclass"]) else str(row["superclass"]),
            "side": None if pd.isna(row["somaSide"]) else str(row["somaSide"]),
            "soma": None if soma is None or (isinstance(soma, float) and np.isnan(soma)) else [int(v) for v in soma],
            "nt": ntx,
            "sign": SIGN.get(ntx, 0),
            "role": role,
            "modality": ch[0] if ch else None,
            "channel": ch[1] if ch else None,
        })
    edge_list = [[index[int(p)], index[int(q)], int(c)] for p, q, c in sub[["pre", "post", "c"]].itertuples(index=False)]
    edge_list.sort()

    # Reachability through nonzero-sign edges.
    adj = defaultdict(list)
    for p, q, _ in edge_list:
        if cells[p]["sign"] != 0:
            adj[p].append(q)

    def reach(starts):
        seen = set(starts)
        stack = list(starts)
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return seen

    readout_idx = [index[b] for b in readout]
    report = {}
    for modality in ("food", "threat", "wall", "prey") + (("self",) if INCLUDE_SELF else ()):
        for ch in CHANNELS:
            starts = [i for i, c in enumerate(cells) if c["role"] == "input" and c["modality"] == modality and c["channel"] == ch]
            r = reach(starts)
            report[f"{modality}/{ch}"] = sum(1 for i in readout_idx if i in r)
    all_reach = reach([i for i, c in enumerate(cells) if c["role"] == "input"])
    unreachable = [cells[i]["bodyId"] for i in readout_idx if i not in all_reach]
    print("readout cells reached per input channel:", report)
    print("unreachable readout cells:", unreachable)
    assert not unreachable, "every readout cell must be reachable from inputs"
    assert all(v > 0 for v in report.values()), "every input channel must reach at least one readout cell"

    nt_counts = defaultdict(int)
    for c in cells:
        nt_counts[c["nt"]] += 1
    circuit = {
        "source": "MaleCNS v1.0 flat connectome, minimum confidence 0.5 (FlyEM / HHMI Janelia and collaborators, CC BY 4.0)",
        "selection": {
            "food": f"class=gustatory, subclass in {sorted(FOOD_SUBCLASSES)}, top 16 by two-hop contact reach onto descending neurons",
            "threat": f"type in {THREAT_TYPES}, top 16 by direct contacts onto descending neurons",
            "wall": f"type in {WALL_TYPES}, top 16 by direct contacts onto descending neurons",
            "prey": f"type in {PREY_TYPES}, top 16 by {prey_selection}",
            **({"self": f"class = {SELF_CLASS}, top 16 by {self_selection}"} if INCLUDE_SELF else {}),
            "readout": f"{READOUT_COUNT} descending neurons: top {READOUT_COUNT // 2} by direct contacts from selected visual inputs, top {READOUT_COUNT // 2} by two-hop reach from selected gustatory inputs, filled by combined rank",
            "bridge": f"{BRIDGE_FOOD} strongest gustatory-to-readout carriers plus {BRIDGE_GENERAL} strongest general input-to-readout carriers (min of summed input and output contacts)",
            "edges": "every measured directed edge among selected cells is retained",
        },
        "counts": {
            "cells": len(cells),
            "edges": len(edge_list),
            "contacts": int(sum(e[2] for e in edge_list)),
            "inputs": len(input_set),
            "bridges": len(bridges),
            "readouts": len(readout),
            "neurotransmitters": dict(nt_counts),
        },
        "reachability": report,
        "channels": CHANNELS,
        "cells": cells,
        "edges": edge_list,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "circuit.json").write_text(json.dumps(circuit, separators=(",", ":")))
    manifest = {
        "raw": {k: {"file": v, "sha256": sha256(raw_dir / v), "bytes": (raw_dir / v).stat().st_size} for k, v in RAW.items()},
        "circuit_sha256": hashlib.sha256((out_dir / "circuit.json").read_bytes()).hexdigest(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(circuit["counts"], indent=1))


if __name__ == "__main__":
    raw = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "public" / "data"
    main(raw, out)
