#!/usr/bin/env python3
"""
2-class whole-episode segment labels for the merged mix_block_0528 dataset.

Each merged episode's correct answer (block color) comes from the provenance
mapping written by merge_block_datasets.py (meta/episode_origin.csv). Because
green-answer and yellow-answer episodes are interleaved in ONE dataset, the
episode index / recording-batch position no longer correlates with the answer.

One segment per episode: [0 .. length-1] -> put_<color>_block_to_plate.

Run with the trossen_data env from /iris/projects/humanoid/trossen_data.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


COLOR_TO_LABEL = {
    "green": "put_green_block_to_plate",
    "yellow": "put_yellow_block_to_plate",
}
COLOR_TO_TASK = {
    "green": "put the green block to the plate",
    "yellow": "put the yellow block to the plate",
}
CSV_COLUMNS = ["dataset", "episode_id", "task", "start_frame",
               "end_frame", "subtask", "source", "notes"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/iris/projects/humanoid/trossen_data"))
    ap.add_argument("--dataset", default="mix_block_0528")
    ap.add_argument("--out", type=Path,
                    default=Path("scripts/labels/subtask_segments_mix_block_0528_auto.csv"))
    args = ap.parse_args()

    ds_root = args.root / args.dataset
    origin = ds_root / "meta" / "episode_origin.csv"
    lengths = {
        int(r["episode_index"]): int(r["length"])
        for r in (json.loads(l) for l in (ds_root / "meta" / "episodes.jsonl").read_text().splitlines() if l.strip())
    }

    rows = []
    counts = {"green": 0, "yellow": 0}
    with origin.open(newline="") as f:
        for r in csv.DictReader(f):
            ep = int(r["new_episode_id"])
            color = r["block_color"]
            length = lengths[ep]
            rows.append({
                "dataset": args.dataset,
                "episode_id": f"{ep:06d}",
                "task": COLOR_TO_TASK[color],
                "start_frame": "0",
                "end_frame": str(length - 1),
                "subtask": COLOR_TO_LABEL[color],
                "source": "auto",
                "notes": "scheme=block_2cls_mix; whole_episode",
            })
            counts[color] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.out}  rows={len(rows)}  green={counts['green']} yellow={counts['yellow']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
