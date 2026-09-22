"""Extract the looming-escape and gustatory-approach pathways from MaleCNS v1.0.

Usage: python scripts/build-escape-circuit.py <raw-dir> [out-path] [min-contacts]

Unlike scripts/build-connectome.py, which crops a small subgraph for a trained readout,
this keeps whole measured pathways: every cell on a one- or two-hop path from a sensory
population to a descending neuron, above a contact threshold. Selection uses anatomy and
transmitter annotations only, before any simulation.
"""
from __future__ import annotations

import hashlib
import json
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
SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1}
LOOMING_TYPES = ["LC4", "LPLC2", "LC6", "LC16", "LPLC1"]
GIANT_FIBER = ["DNp01"]
FOOD_SUBCLASSES = {"labellar bristle", "taste peg", "pharyngeal sensillum"}
WALL_TYPES = ["LC11", "LC15", "LC17", "LC21"]
PREY_TYPES = ["LC10a", "LC10b", "LC10c-1", "LC10c-2", "LC10d", "LC10e"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main(raw_dir: Path, out_path: Path, min_contacts: int) -> None:
    ann = feather.read_feather(raw_dir / RAW["annotations"]).set_index("bodyId")
    nt = feather.read_feather(raw_dir / RAW["neurotransmitters"])
    nt_map = nt.set_index("body")["consensus_nt"].to_dict()
    w = feather.read_feather(raw_dir / RAW["weights"])
    edges = w.rename(columns={"body_pre": "pre", "body_post": "post", "weight": "c"})
    edges = edges[edges["c"] >= min_contacts]
    print(f"edges at >= {min_contacts} contacts: {len(edges):,}")

    def by_type(types: list[str]) -> set[int]:
        return set(ann.index[ann["type"].isin(types)])

    looming = by_type(LOOMING_TYPES)
    wall = by_type(WALL_TYPES)
    prey = by_type(PREY_TYPES)
    food = set(ann.index[(ann["class"] == "gustatory") & ann["subclass"].isin(FOOD_SUBCLASSES)])
    descending = set(ann.index[ann["superclass"] == "descending_neuron"])
    giant_fiber = by_type(GIANT_FIBER)
    sensory = {"threat": looming, "wall": wall, "prey": prey, "food": food}
    all_sensory = set().union(*sensory.values())

    # Bridges: cells receiving from a sensory population and projecting onto a descending neuron.
    from_sensory = set(edges[edges["pre"].isin(all_sensory)]["post"])
    onto_descending = set(edges[edges["post"].isin(descending)]["pre"])
    bridges = (from_sensory & onto_descending) - all_sensory - descending
    nodes = all_sensory | bridges | descending
    sub = edges[edges["pre"].isin(nodes) & edges["post"].isin(nodes)]
    # Drop cells that ended up with no retained edge at all.
    connected = set(sub["pre"]) | set(sub["post"])
    nodes &= connected
    sub = sub[sub["pre"].isin(nodes) & sub["post"].isin(nodes)]

    order = sorted(nodes)
    index = {b: i for i, b in enumerate(order)}
    print(f"cells {len(order):,}  edges {len(sub):,}  contacts {int(sub['c'].sum()):,}")

    cells = []
    for b in order:
        row = ann.loc[b]
        ntx = nt_map.get(b)
        ntx = str(ntx) if isinstance(ntx, str) else "unknown"
        modality = next((m for m, ids in sensory.items() if b in ids), None)
        role = "input" if modality else "readout" if b in descending else "bridge"
        soma = row["somaLocation"]
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
            "modality": modality,
            "giantFiber": b in giant_fiber,
        })

    edge_list = [[index[int(p)], index[int(q)], int(c)] for p, q, c in sub[["pre", "post", "c"]].itertuples(index=False)]
    edge_list.sort()

    adjacency = defaultdict(list)
    for p, q, _ in edge_list:
        if cells[p]["sign"] != 0:
            adjacency[p].append(q)

    def reach(starts: list[int]) -> set[int]:
        seen = set(starts)
        stack = list(starts)
        while stack:
            u = stack.pop()
            for v in adjacency[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return seen

    readout_idx = [i for i, c in enumerate(cells) if c["role"] == "readout"]
    report = {}
    for modality in sensory:
        starts = [i for i, c in enumerate(cells) if c["modality"] == modality]
        r = reach(starts) if starts else set()
        report[modality] = {"cells": len(starts), "descendingReached": sum(1 for i in readout_idx if i in r)}
    gf_idx = [i for i, c in enumerate(cells) if c["giantFiber"]]
    direct_to_gf = sum(1 for p, q, _ in edge_list if cells[p]["modality"] == "threat" and q in gf_idx)
    print("reachability:", json.dumps(report))
    print(f"giant fiber cells retained: {len(gf_idx)}, direct looming->GF edges: {direct_to_gf}")

    counts = defaultdict(int)
    for c in cells:
        counts[c["role"]] += 1
    circuit = {
        "source": "MaleCNS v1.0 flat connectome, minimum confidence 0.5 (FlyEM / HHMI Janelia and collaborators, CC BY 4.0)",
        "kind": "pathway",
        "minContacts": min_contacts,
        "selection": {
            "threat": f"type in {LOOMING_TYPES}, the looming-sensitive lobula populations",
            "wall": f"type in {WALL_TYPES}",
            "prey": f"type in {PREY_TYPES}",
            "food": f"class = gustatory, subclass in {sorted(FOOD_SUBCLASSES)}",
            "readout": "every descending neuron reached by the retained graph",
            "bridge": "every cell that receives from a selected sensory population and projects onto a descending neuron",
            "edges": f"every measured directed edge of at least {min_contacts} contacts among the retained cells",
        },
        "counts": {
            "cells": len(cells),
            "edges": len(edge_list),
            "contacts": int(sub["c"].sum()),
            **dict(counts),
            "giantFiber": len(gf_idx),
            "loomingToGiantFiberEdges": direct_to_gf,
        },
        "reachability": report,
        "cells": cells,
        "edges": edge_list,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(circuit, separators=(",", ":")))
    manifest = {
        "raw": {k: {"file": v, "sha256": sha256(raw_dir / v)} for k, v in RAW.items()},
        "minContacts": min_contacts,
        "circuit_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
    }
    out_path.with_name(out_path.stem + "-manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    raw = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("public/data/pathway/circuit.json")
    mc = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    main(raw, out, mc)
