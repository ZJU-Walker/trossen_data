#!/usr/bin/env python3
"""
Merge two LeRobot-format datasets (same schema) into a single dataset.

Concatenates SOURCES in order. Episodes from the second source are renumbered
to continue after the first; the global frame `index` column inside every
parquet is also renumbered so the merged dataset is contiguous and valid for
downstream training (e.g. pi0.5 finetuning).

Edit SOURCES and OUT_ROOT below, then run:
    python merge_lerobot_datasets.py            # dry run, no files touched
    python merge_lerobot_datasets.py --apply    # actually write OUT_ROOT

Assumes both sources share:
  - same `features` schema in info.json
  - same `fps`, `chunks_size`, `codebase_version`, `robot_type`
  - same `tasks.jsonl` task definitions (script will merge & remap task_index
    if they differ, but in practice they should match)

The output is written as a fresh dataset at OUT_ROOT — source datasets are
not modified. Videos are HARDLINKED when possible (instant, no extra disk),
and fall back to copy across filesystems.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow as pa

# ----------------------------------------------------------------------
# Edit these for the merge you want to perform.
# SOURCES is an ordered list -- episodes from sources[0] come first.
# ----------------------------------------------------------------------
SOURCES = [
    "/iris/projects/humanoid/trossen_data/pack_part1",
    "/iris/projects/humanoid/trossen_data/pack_part2",
]
OUT_ROOT = "/iris/projects/humanoid/trossen_data/pack_with_human"
# ----------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def chunk_for(ep_index: int, chunks_size: int) -> int:
    return ep_index // chunks_size


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def rewrite_parquet(
    src: Path,
    dst: Path,
    new_ep_index: int,
    new_index_start: int,
    task_index_remap: dict[int, int] | None,
) -> int:
    """Read src parquet, replace episode_index / global index / task_index
    columns, write to dst. Returns the next global index."""
    table = pq.read_table(src)
    n = table.num_rows

    new_cols = {}
    for name in table.column_names:
        if name == "episode_index":
            new_cols[name] = pa.array([new_ep_index] * n, type=table.schema.field(name).type)
        elif name == "index":
            new_cols[name] = pa.array(
                list(range(new_index_start, new_index_start + n)),
                type=table.schema.field(name).type,
            )
        elif name == "task_index" and task_index_remap:
            old = table.column(name).to_pylist()
            new_cols[name] = pa.array(
                [task_index_remap.get(v, v) for v in old],
                type=table.schema.field(name).type,
            )
        else:
            new_cols[name] = table.column(name)

    new_table = pa.table(new_cols, schema=table.schema)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_table, dst)
    return new_index_start + n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", nargs="+", default=None,
                   help=f"Ordered list of source dataset roots (default: SOURCES in script)")
    p.add_argument("--out", type=Path, default=Path(OUT_ROOT),
                   help="Output dataset root (default: OUT_ROOT in script)")
    p.add_argument("--apply", action="store_true",
                   help="Actually write files (default is dry-run)")
    p.add_argument("--overwrite", action="store_true",
                   help="If output exists, remove it first (only with --apply)")
    args = p.parse_args()

    src_roots = [Path(s).resolve() for s in (args.sources or SOURCES)]
    out_root: Path = args.out.resolve()

    if len(src_roots) < 2:
        print("ERROR: need at least 2 sources to merge", file=sys.stderr)
        return 2

    for r in src_roots:
        if not r.is_dir():
            print(f"ERROR: source not a dir: {r}", file=sys.stderr)
            return 2

    # Load and validate metadata from each source
    infos = [json.loads((r / "meta" / "info.json").read_text()) for r in src_roots]

    base = infos[0]
    must_match = ("features", "fps", "chunks_size", "codebase_version", "robot_type")
    for r, info in zip(src_roots[1:], infos[1:]):
        for key in must_match:
            if info.get(key) != base.get(key):
                print(f"ERROR: '{key}' differs between {src_roots[0]} and {r}", file=sys.stderr)
                return 2

    chunks_size = int(base["chunks_size"])
    video_keys = [k for k, v in base["features"].items() if v.get("dtype") == "video"]

    # Build merged task table (remap task_index per source if needed)
    merged_tasks: list[dict] = []
    task_text_to_new_idx: dict[str, int] = {}
    per_source_remap: list[dict[int, int]] = []
    for r in src_roots:
        rows = load_jsonl(r / "meta" / "tasks.jsonl")
        remap = {}
        for row in rows:
            text = row["task"]
            if text not in task_text_to_new_idx:
                task_text_to_new_idx[text] = len(merged_tasks)
                merged_tasks.append({"task_index": len(merged_tasks), "task": text})
            remap[int(row["task_index"])] = task_text_to_new_idx[text]
        per_source_remap.append(remap)

    # Compute episode mapping
    plan: list[tuple[int, Path, int]] = []  # (source_idx, source_root, old_ep_index)
    for s_idx, r in enumerate(src_roots):
        n_eps = int(infos[s_idx]["total_episodes"])
        for old in range(n_eps):
            plan.append((s_idx, r, old))

    new_total = len(plan)

    print(f"Sources:")
    for r, info in zip(src_roots, infos):
        print(f"  {r}  ({info['total_episodes']} episodes, {info['total_frames']} frames)")
    print(f"Output:           {out_root}")
    print(f"Merged episodes:  {new_total}")
    print(f"Merged tasks:     {len(merged_tasks)}  -> {[t['task'] for t in merged_tasks]}")
    print(f"Video keys:       {video_keys}")
    print()

    if not args.apply:
        print("DRY RUN -- nothing written. Re-run with --apply to execute.")
        print()
        cur_ep = 0
        for s_idx, r, old in plan[:5]:
            print(f"  {r.name}:episode_{old:06d}  ->  episode_{cur_ep:06d}")
            cur_ep += 1
        if new_total > 10:
            print(f"  ... ({new_total - 10} more)")
            cur_ep = new_total - 5
            for s_idx, r, old in plan[-5:]:
                print(f"  {r.name}:episode_{old:06d}  ->  episode_{cur_ep:06d}")
                cur_ep += 1
        return 0

    # Apply
    if out_root.exists():
        if not args.overwrite:
            print(f"ERROR: output exists: {out_root}. Pass --overwrite to replace it.",
                  file=sys.stderr)
            return 2
        shutil.rmtree(out_root)

    (out_root / "data").mkdir(parents=True)
    (out_root / "videos").mkdir(parents=True)
    (out_root / "meta").mkdir(parents=True)

    # Process episodes in order
    src_episodes_rows = {i: load_jsonl(r / "meta" / "episodes.jsonl")
                        for i, r in enumerate(src_roots)}
    src_stats_rows = {i: load_jsonl(r / "meta" / "episodes_stats.jsonl")
                     for i, r in enumerate(src_roots)}
    src_ep_by_idx = {i: {row["episode_index"]: row for row in src_episodes_rows[i]}
                    for i in range(len(src_roots))}
    src_stats_by_idx = {i: {row["episode_index"]: row for row in src_stats_rows[i]}
                       for i in range(len(src_roots))}

    new_episodes_rows: list[dict] = []
    new_stats_rows: list[dict] = []
    new_global_index = 0

    for new_ep_idx, (s_idx, r, old) in enumerate(plan):
        # Parquet
        old_chunk = chunk_for(old, chunks_size)
        new_chunk = chunk_for(new_ep_idx, chunks_size)
        src_pq = r / "data" / f"chunk-{old_chunk:03d}" / f"episode_{old:06d}.parquet"
        dst_pq = out_root / "data" / f"chunk-{new_chunk:03d}" / f"episode_{new_ep_idx:06d}.parquet"
        new_global_index = rewrite_parquet(
            src_pq, dst_pq, new_ep_idx, new_global_index, per_source_remap[s_idx]
        )

        # Videos
        for vk in video_keys:
            src_v = r / "videos" / f"chunk-{old_chunk:03d}" / vk / f"episode_{old:06d}.mp4"
            dst_v = out_root / "videos" / f"chunk-{new_chunk:03d}" / vk / f"episode_{new_ep_idx:06d}.mp4"
            link_or_copy(src_v, dst_v)

        # Meta rows
        ep_row = dict(src_ep_by_idx[s_idx][old])
        ep_row["episode_index"] = new_ep_idx
        new_episodes_rows.append(ep_row)

        if old in src_stats_by_idx[s_idx]:
            st_row = dict(src_stats_by_idx[s_idx][old])
            st_row["episode_index"] = new_ep_idx
            new_stats_rows.append(st_row)

        if (new_ep_idx + 1) % 10 == 0 or new_ep_idx == new_total - 1:
            print(f"  merged {new_ep_idx + 1}/{new_total} episodes")

    # Write meta
    dump_jsonl(out_root / "meta" / "episodes.jsonl", new_episodes_rows)
    dump_jsonl(out_root / "meta" / "episodes_stats.jsonl", new_stats_rows)
    dump_jsonl(out_root / "meta" / "tasks.jsonl", merged_tasks)

    new_total_frames = sum(row["length"] for row in new_episodes_rows)
    new_info = dict(base)
    new_info["total_episodes"] = new_total
    new_info["total_frames"] = new_total_frames
    new_info["total_tasks"] = len(merged_tasks)
    new_info["total_videos"] = new_total * len(video_keys)
    max_chunk = chunk_for(new_total - 1, chunks_size)
    new_info["total_chunks"] = max_chunk + 1
    new_info["splits"] = {"train": f"0:{new_total}"}
    (out_root / "meta" / "info.json").write_text(json.dumps(new_info, indent=4) + "\n")

    print()
    print(f"Done. Merged dataset at {out_root}")
    print(f"  total_episodes = {new_total}")
    print(f"  total_frames   = {new_total_frames}")
    print(f"  total_tasks    = {len(merged_tasks)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
