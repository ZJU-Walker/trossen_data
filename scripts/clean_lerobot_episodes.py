#!/usr/bin/env python3
"""
Delete episodes from a LeRobot-format dataset and renumber the remaining
episodes so the dataset stays contiguous and valid for downstream training
(e.g. pi0.5 finetuning).

Edit DATA_ROOT and DELETE_NUMBER below, then run:
    python clean_lerobot_episodes.py            # dry run, no files touched
    python clean_lerobot_episodes.py --apply    # actually modify files

Layout assumed (LeRobot v2.1):
    <root>/
        data/chunk-XXX/episode_NNNNNN.parquet
        videos/chunk-XXX/<video_key>/episode_NNNNNN.mp4
        meta/episodes.jsonl
        meta/episodes_stats.jsonl
        meta/info.json
        meta/tasks.jsonl
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow as pa

# ----------------------------------------------------------------------
# Edit these for the dataset you want to clean.
# DELETE_NUMBER accepts ints and inclusive "lo-hi" range strings, e.g.
#   DELETE_NUMBER = [9, 21, 22, 23, 24, 25]
#   DELETE_NUMBER = [9, "21-25"]
# ----------------------------------------------------------------------
DATA_ROOT = "/iris/projects/humanoid/trossen_data/0528_yellow_block_mem"
DELETE_NUMBER = [13]
# ----------------------------------------------------------------------


def parse_delete_spec(tokens) -> set[int]:
    """Parse a list of ints and/or range strings ('21-25') into a set of ints."""
    out: set[int] = set()
    for tok in tokens:
        if isinstance(tok, int):
            out.add(tok)
            continue
        tok = str(tok).strip()
        if not tok:
            continue
        if "-" in tok:
            lo, hi = tok.split("-", 1)
            lo_i, hi_i = int(lo), int(hi)
            if hi_i < lo_i:
                raise ValueError(f"bad range: {tok}")
            out.update(range(lo_i, hi_i + 1))
        else:
            out.add(int(tok))
    return out


def chunk_for(ep_index: int, chunks_size: int) -> int:
    return ep_index // chunks_size


def parquet_path(root: Path, ep_index: int, chunks_size: int) -> Path:
    c = chunk_for(ep_index, chunks_size)
    return root / "data" / f"chunk-{c:03d}" / f"episode_{ep_index:06d}.parquet"


def video_path(root: Path, ep_index: int, chunks_size: int, video_key: str) -> Path:
    c = chunk_for(ep_index, chunks_size)
    return root / "videos" / f"chunk-{c:03d}" / video_key / f"episode_{ep_index:06d}.mp4"


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


def rewrite_parquet(src: Path, dst: Path, new_ep_index: int, new_index_start: int) -> int:
    """Read src parquet, replace episode_index and global index columns,
    write to dst. Returns the next global index (start + length)."""
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
        else:
            new_cols[name] = table.column(name)

    new_table = pa.table(new_cols, schema=table.schema)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_table, dst)
    return new_index_start + n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path(DATA_ROOT), help="Dataset root directory (default: DATA_ROOT in script)")
    p.add_argument(
        "--delete",
        nargs="+",
        default=None,
        help="Episode indices to delete, e.g. '9 21-25' (default: DELETE_NUMBER in script)",
    )
    p.add_argument("--apply", action="store_true", help="Actually modify files (default is dry-run)")
    p.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backing up meta/ before rewriting (only valid with --apply)",
    )
    args = p.parse_args()

    root: Path = args.root.resolve()
    if not root.is_dir():
        print(f"ERROR: root not a directory: {root}", file=sys.stderr)
        return 2

    info_path = root / "meta" / "info.json"
    episodes_path = root / "meta" / "episodes.jsonl"
    stats_path = root / "meta" / "episodes_stats.jsonl"

    for f in (info_path, episodes_path, stats_path):
        if not f.is_file():
            print(f"ERROR: missing {f}", file=sys.stderr)
            return 2

    info = json.loads(info_path.read_text())
    chunks_size = int(info["chunks_size"])
    total_eps = int(info["total_episodes"])
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]

    delete_spec = args.delete if args.delete is not None else DELETE_NUMBER
    to_delete = parse_delete_spec(delete_spec)
    bad = [e for e in to_delete if e < 0 or e >= total_eps]
    if bad:
        print(f"ERROR: out-of-range episodes (dataset has {total_eps} eps): {bad}", file=sys.stderr)
        return 2

    keep = [e for e in range(total_eps) if e not in to_delete]
    new_total = len(keep)

    print(f"Root: {root}")
    print(f"Existing episodes: {total_eps}  (chunks_size={chunks_size})")
    print(f"Video keys: {video_keys}")
    print(f"Deleting:  {sorted(to_delete)}")
    print(f"Keeping:   {new_total} episodes")
    print()

    episodes = load_jsonl(episodes_path)
    stats = load_jsonl(stats_path)
    ep_by_idx = {row["episode_index"]: row for row in episodes}
    stats_by_idx = {row["episode_index"]: row for row in stats}

    missing = [e for e in keep if e not in ep_by_idx]
    if missing:
        print(f"ERROR: episodes missing from episodes.jsonl: {missing}", file=sys.stderr)
        return 2

    mapping = []  # list of (old_idx, new_idx)
    for new_idx, old_idx in enumerate(keep):
        mapping.append((old_idx, new_idx))

    if not args.apply:
        print("DRY RUN -- no files will change. Re-run with --apply to execute.")
        print()
        print("Files that would be removed:")
        for old in sorted(to_delete):
            print(f"  rm {parquet_path(root, old, chunks_size).relative_to(root)}")
            for vk in video_keys:
                print(f"  rm {video_path(root, old, chunks_size, vk).relative_to(root)}")
        print()
        print("Renames (old -> new) for kept episodes:")
        changes = [(o, n) for o, n in mapping if o != n]
        if not changes:
            print("  (none)")
        else:
            for old, new in changes:
                print(f"  episode_{old:06d}  ->  episode_{new:06d}")
        print()
        new_total_frames = sum(ep_by_idx[e]["length"] for e in keep)
        print(f"info.json updates: total_episodes {total_eps} -> {new_total},  "
              f"total_frames -> {new_total_frames},  "
              f"total_videos -> {new_total * len(video_keys)},  "
              f"splits.train -> '0:{new_total}'")
        return 0

    if not args.no_backup:
        backup = root / "meta_backup"
        if backup.exists():
            print(f"ERROR: backup dir already exists: {backup}. Remove it or pass --no-backup.",
                  file=sys.stderr)
            return 2
        shutil.copytree(root / "meta", backup)
        print(f"Backed up meta/ -> {backup}")

    # 1. Delete dropped episodes (parquet + videos)
    print("Deleting dropped episodes...")
    for old in sorted(to_delete):
        pq_f = parquet_path(root, old, chunks_size)
        if pq_f.is_file():
            pq_f.unlink()
            print(f"  removed {pq_f.relative_to(root)}")
        for vk in video_keys:
            vf = video_path(root, old, chunks_size, vk)
            if vf.is_file():
                vf.unlink()
                print(f"  removed {vf.relative_to(root)}")

    # 2. Renumber kept episodes. Process in safe order to avoid clobbering:
    #    if new_idx < old_idx for every kept ep (which is true when we only
    #    delete -- never insert), ascending order is safe.
    print("Renumbering remaining episodes...")
    # Use a tmp dir for parquet rewrites to avoid in-place hazards.
    tmp_data = root / "data_tmp"
    if tmp_data.exists():
        shutil.rmtree(tmp_data)

    new_global_index = 0
    new_episodes_rows = []
    new_stats_rows = []

    for old_idx, new_idx in mapping:
        # parquet: read old, rewrite columns, write to new location in tmp
        src_pq = parquet_path(root, old_idx, chunks_size)
        # destination computed under tmp_data with new chunking
        new_chunk = chunk_for(new_idx, chunks_size)
        dst_pq = tmp_data / f"chunk-{new_chunk:03d}" / f"episode_{new_idx:06d}.parquet"
        new_global_index = rewrite_parquet(src_pq, dst_pq, new_idx, new_global_index)

        # videos: rename in place (old name -> new name within same camera dir).
        # Ascending order guarantees new name slot is free (it was either
        # already moved out, or never existed because new_idx <= old_idx).
        for vk in video_keys:
            old_v = video_path(root, old_idx, chunks_size, vk)
            new_v = video_path(root, new_idx, chunks_size, vk)
            new_v.parent.mkdir(parents=True, exist_ok=True)
            if old_idx != new_idx:
                old_v.rename(new_v)

        # meta rows
        ep_row = dict(ep_by_idx[old_idx])
        ep_row["episode_index"] = new_idx
        new_episodes_rows.append(ep_row)

        if old_idx in stats_by_idx:
            st_row = dict(stats_by_idx[old_idx])
            st_row["episode_index"] = new_idx
            new_stats_rows.append(st_row)

    # 3. Swap data dir with rewritten one
    old_data = root / "data"
    old_data_bak = root / "data_old"
    if old_data_bak.exists():
        shutil.rmtree(old_data_bak)
    old_data.rename(old_data_bak)
    tmp_data.rename(old_data)
    shutil.rmtree(old_data_bak)
    print(f"  rewrote data/ ({new_total} parquet files)")

    # 4. Update meta files
    dump_jsonl(episodes_path, new_episodes_rows)
    dump_jsonl(stats_path, new_stats_rows)

    new_total_frames = sum(row["length"] for row in new_episodes_rows)
    info["total_episodes"] = new_total
    info["total_frames"] = new_total_frames
    info["total_videos"] = new_total * len(video_keys)
    info["splits"] = {"train": f"0:{new_total}"}
    info_path.write_text(json.dumps(info, indent=4) + "\n")
    print(f"  updated meta/episodes.jsonl, meta/episodes_stats.jsonl, meta/info.json")

    print()
    print(f"Done. Dataset now has {new_total} episodes / {new_total_frames} frames.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
