#!/usr/bin/env python3
"""
Merge the green + yellow block datasets into one LeRobot v2.1 dataset
`mix_block_0528`, interleaving episodes so the block color (the answer) is NOT
correlated with episode order / recording batch position.

Source episodes are taken in interleaved order:
    new ep0 = green ep0, new ep1 = yellow ep0, new ep2 = green ep1, ...
(when one dataset runs out, the rest of the longer one is appended).

For each merged episode we:
  - copy all 3 camera mp4s to the new episode number,
  - rewrite the parquet's episode_index (-> new id) and index (-> contiguous
    global row counter), keeping frame_index / task_index / action / state,
  - record provenance (origin dataset + original episode id + block color) in
    a sidecar mapping CSV.

Meta files are rebuilt: episodes.jsonl, episodes_stats.jsonl (renumbered),
tasks.jsonl (single shared task), info.json (totals).

Run from /iris/projects/humanoid/trossen_data with the trossen_data env
(needs pyarrow):
  /iris/projects/humanoid/miniconda3/envs/trossen_data/bin/python \
      scripts/merge_block_datasets.py
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow as pa


CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)

# Each source dataset and the block color that is the correct answer for it.
SOURCES = [
    ("0528_green_block_mem", "green"),
    ("0528_yellow_block_mem", "yellow"),
]
SHARED_TASK = "Put the block the human points at to the plate."


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def episode_order(root: Path) -> dict[str, list[int]]:
    """Episode indices present in each source, sorted."""
    order = {}
    for ds, _ in SOURCES:
        eps = [int(r["episode_index"]) for r in load_jsonl(root / ds / "meta" / "episodes.jsonl")]
        order[ds] = sorted(eps)
    return order


def interleave(order: dict[str, list[int]]) -> list[tuple[str, str, int]]:
    """Return [(dataset, color, src_ep), ...] interleaved across sources."""
    queues = {ds: list(eps) for ds, eps in order.items()}
    color_by_ds = dict(SOURCES)
    merged: list[tuple[str, str, int]] = []
    while any(queues[ds] for ds, _ in SOURCES):
        for ds, _ in SOURCES:
            if queues[ds]:
                merged.append((ds, color_by_ds[ds], queues[ds].pop(0)))
    return merged


def src_parquet(root: Path, ds: str, ep: int, chunks_size: int) -> Path:
    chunk = ep // chunks_size
    return root / ds / "data" / f"chunk-{chunk:03d}" / f"episode_{ep:06d}.parquet"


def src_video(root: Path, ds: str, cam: str, ep: int, chunks_size: int) -> Path:
    chunk = ep // chunks_size
    return root / ds / "videos" / f"chunk-{chunk:03d}" / cam / f"episode_{ep:06d}.mp4"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/iris/projects/humanoid/trossen_data"))
    ap.add_argument("--out-name", default="mix_block_0528")
    ap.add_argument("--chunks-size", type=int, default=1000)
    args = ap.parse_args()

    root = args.root
    out = root / args.out_name
    if out.exists():
        print(f"ERROR: {out} already exists; remove it first.", flush=True)
        return 2

    # Build interleaved plan.
    order = episode_order(root)
    plan = interleave(order)
    print(f"Merging {len(plan)} episodes: " +
          ", ".join(f"{c}{e}" for _, c, e in plan[:8]) + " ...")

    # Reference info.json from the first source (schema is identical).
    base_info = json.loads((root / SOURCES[0][0] / "meta" / "info.json").read_text())
    chunks_size = int(base_info.get("chunks_size", args.chunks_size))

    # Output dirs.
    (out / "data" / "chunk-000").mkdir(parents=True)
    for cam in CAMERAS:
        (out / "videos" / "chunk-000" / cam).mkdir(parents=True)
    (out / "meta").mkdir(parents=True)

    episodes_meta = []
    episodes_stats = []
    mapping_rows = []
    global_index = 0
    total_frames = 0

    # Cache source stats keyed by (ds, src_ep).
    stats_cache: dict[str, dict[int, dict]] = {}
    for ds, _ in SOURCES:
        stats_cache[ds] = {
            int(r["episode_index"]): r
            for r in load_jsonl(root / ds / "meta" / "episodes_stats.jsonl")
        }

    for new_ep, (ds, color, src_ep) in enumerate(plan):
        # --- parquet: rewrite episode_index + index ---
        sp = src_parquet(root, ds, src_ep, chunks_size)
        table = pq.read_table(sp)
        n = table.num_rows
        ep_col = pa.array([new_ep] * n, type=pa.int64())
        idx_col = pa.array(list(range(global_index, global_index + n)), type=pa.int64())
        cols = {name: table.column(name) for name in table.column_names}
        cols["episode_index"] = ep_col
        cols["index"] = idx_col
        new_table = pa.table(cols)
        dp = out / "data" / "chunk-000" / f"episode_{new_ep:06d}.parquet"
        pq.write_table(new_table, dp)

        # --- videos: copy all cameras ---
        for cam in CAMERAS:
            sv = src_video(root, ds, cam, src_ep, chunks_size)
            dv = out / "videos" / "chunk-000" / cam / f"episode_{new_ep:06d}.mp4"
            shutil.copy2(sv, dv)

        # --- meta rows ---
        episodes_meta.append({
            "episode_index": new_ep,
            "tasks": [SHARED_TASK],
            "length": n,
        })
        st = dict(stats_cache[ds][src_ep])
        st["episode_index"] = new_ep
        episodes_stats.append(st)
        mapping_rows.append({
            "new_episode_id": f"{new_ep:06d}",
            "origin_dataset": ds,
            "origin_episode_id": f"{src_ep:06d}",
            "block_color": color,
            "length": n,
        })

        global_index += n
        total_frames += n
        if new_ep % 10 == 0:
            print(f"  ep {new_ep:>3} <- {ds} {src_ep:06d} ({color}, {n} frames)")

    # --- meta files ---
    write_jsonl(out / "meta" / "episodes.jsonl", episodes_meta)
    write_jsonl(out / "meta" / "episodes_stats.jsonl", episodes_stats)
    write_jsonl(out / "meta" / "tasks.jsonl", [{"task_index": 0, "task": SHARED_TASK}])

    info = json.loads(json.dumps(base_info))  # deep copy
    info["total_episodes"] = len(plan)
    info["total_frames"] = total_frames
    info["total_tasks"] = 1
    info["total_videos"] = len(plan) * len(CAMERAS)
    info["total_chunks"] = 1
    info["splits"] = {"train": f"0:{len(plan)}"}
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    # --- provenance mapping ---
    with (out / "meta" / "episode_origin.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["new_episode_id", "origin_dataset",
                                          "origin_episode_id", "block_color", "length"])
        w.writeheader()
        w.writerows(mapping_rows)

    print(f"\nDone: {out}")
    print(f"  episodes={len(plan)}  frames={total_frames}  videos={len(plan)*len(CAMERAS)}")
    ngreen = sum(1 for r in mapping_rows if r["block_color"] == "green")
    nyellow = sum(1 for r in mapping_rows if r["block_color"] == "yellow")
    print(f"  green={ngreen}  yellow={nyellow}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
