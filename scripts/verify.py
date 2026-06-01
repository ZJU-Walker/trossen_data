#!/usr/bin/env python3
"""
Export per-frame actions from selected LeRobot demo episodes to CSV.

Examples:
  python3 scripts/verify.py
  python3 scripts/verify.py --dataset data_robot_give_0526 --episodes 0
  python3 scripts/verify.py --dataset data_robot_pull_0526 --episodes 0 5 10-12
  python3 scripts/verify.py --dataset data_robot_give_0526 --episodes 0 --start-frame 70 --end-frame 100
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_DATASET = REPO_ROOT / "data_robot_give_0526"
DEFAULT_OUT = SCRIPT_DIR / "labels" / "verify_actions.csv"


class ParquetReader:
    """Read parquet columns using either full pyarrow or the low-level module."""

    def __init__(self) -> None:
        self._mode = ""
        self._pq = None

        try:
            import pyarrow.parquet as pq  # type: ignore

            self._mode = "pyarrow.parquet"
            self._pq = pq
            return
        except Exception:
            pass

        try:
            import pyarrow._parquet as pq  # type: ignore

            self._mode = "pyarrow._parquet"
            self._pq = pq
            return
        except Exception:
            pass

    @property
    def available(self) -> bool:
        return self._pq is not None

    @property
    def mode(self) -> str:
        return self._mode or "unavailable"

    def read_column(self, path: Path, column_name: str) -> list:
        if self._pq is None:
            raise RuntimeError("No parquet reader is available. Install pyarrow with parquet support.")

        if self._mode == "pyarrow.parquet":
            table = self._pq.read_table(path, columns=[column_name])
            return table.column(column_name).to_pylist()

        reader = self._pq.ParquetReader()
        reader.open(str(path))
        try:
            column_index = None
            for idx, path_parts in enumerate(reader.column_paths):
                if path_parts and path_parts[0] == column_name:
                    column_index = idx
                    break
            if column_index is None:
                raise KeyError(f"Column not found in {path}: {column_name}")
            return reader.read_column(column_index).to_pylist()
        finally:
            reader.close()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def parse_episode_specs(specs: list[str]) -> list[int]:
    episodes: set[int] = set()
    for spec in specs:
        token = spec.strip()
        if not token:
            continue
        if "-" in token:
            lo_s, hi_s = token.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                raise ValueError(f"Bad episode range: {spec}")
            episodes.update(range(lo, hi + 1))
        else:
            episodes.add(int(token))
    return sorted(episodes)


def chunk_for(episode_id: int, chunks_size: int) -> int:
    return episode_id // chunks_size


def parquet_path(dataset_root: Path, episode_id: int, chunks_size: int) -> Path:
    chunk = chunk_for(episode_id, chunks_size)
    return dataset_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_id:06d}.parquet"


def safe_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_")


def vector_headers(prefix: str, names: list[str] | None, width: int) -> list[str]:
    headers = []
    for idx in range(width):
        suffix = names[idx] if names and idx < len(names) else str(idx)
        headers.append(f"{prefix}_{safe_name(suffix)}")
    return headers


def select_frame_indices(num_frames: int, start_frame: int | None, end_frame: int | None) -> range:
    start = 0 if start_frame is None else start_frame
    end = num_frames - 1 if end_frame is None else end_frame

    if start < 0:
        raise ValueError("--start-frame must be >= 0")
    if end < start:
        raise ValueError("--end-frame must be >= --start-frame")
    if start >= num_frames:
        raise ValueError(f"--start-frame {start} is outside episode with {num_frames} frames")

    end = min(end, num_frames - 1)
    return range(start, end + 1)


def export_actions(
    dataset_root: Path,
    episodes: list[int],
    out_path: Path,
    start_frame: int | None,
    end_frame: int | None,
    include_state: bool,
) -> int:
    info = load_json(dataset_root / "meta" / "info.json")
    chunks_size = int(info["chunks_size"])
    action_names = info["features"]["action"].get("names")
    state_names = info["features"].get("observation.state", {}).get("names")

    action_width = int(info["features"]["action"]["shape"][0])
    action_headers = vector_headers("action", action_names, action_width)
    state_headers = vector_headers("state", state_names, action_width) if include_state else []

    reader = ParquetReader()
    if not reader.available:
        raise RuntimeError("No parquet reader is available")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "dataset",
        "episode_id",
        "frame_index",
        "timestamp",
        "global_index",
        "task_index",
        "action_json",
        "action_left_gripper",
        "action_right_gripper",
        *action_headers,
        *state_headers,
    ]

    rows_written = 0
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for episode_id in episodes:
            path = parquet_path(dataset_root, episode_id, chunks_size)
            if not path.is_file():
                raise FileNotFoundError(f"Missing episode parquet: {path}")

            actions = reader.read_column(path, "action")
            timestamps = reader.read_column(path, "timestamp")
            frame_indices = reader.read_column(path, "frame_index")
            global_indices = reader.read_column(path, "index")
            task_indices = reader.read_column(path, "task_index")
            states = reader.read_column(path, "observation.state") if include_state else None

            for row_idx in select_frame_indices(len(actions), start_frame, end_frame):
                action = [float(value) for value in actions[row_idx]]
                row = {
                    "dataset": dataset_root.name,
                    "episode_id": f"{episode_id:06d}",
                    "frame_index": int(frame_indices[row_idx]),
                    "timestamp": float(timestamps[row_idx]),
                    "global_index": int(global_indices[row_idx]),
                    "task_index": int(task_indices[row_idx]),
                    "action_json": json.dumps(action),
                    "action_left_gripper": action[6] if len(action) > 6 else "",
                    "action_right_gripper": action[13] if len(action) > 13 else "",
                }

                for header, value in zip(action_headers, action):
                    row[header] = value

                if include_state and states is not None:
                    state = [float(value) for value in states[row_idx]]
                    for header, value in zip(state_headers, state):
                        row[header] = value

                writer.writerow(row)
                rows_written += 1

    print(f"Parquet reader: {reader.mode}")
    print(f"Wrote {rows_written} rows to {out_path}")
    return rows_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Dataset root. Default: {DEFAULT_DATASET}",
    )
    parser.add_argument(
        "--episodes",
        nargs="+",
        default=["0"],
        help="Episode ids or inclusive ranges, e.g. 0 5 10-12. Default: 0",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output CSV path. Default: {DEFAULT_OUT}",
    )
    parser.add_argument("--start-frame", type=int, default=None, help="Optional first frame to export.")
    parser.add_argument("--end-frame", type=int, default=None, help="Optional last frame to export.")
    parser.add_argument(
        "--include-state",
        action="store_true",
        help="Also include observation.state columns in the CSV.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset.resolve()
    if not dataset_root.is_dir():
        print(f"ERROR: dataset root does not exist: {dataset_root}", file=sys.stderr)
        return 2

    try:
        episodes = parse_episode_specs(args.episodes)
        export_actions(
            dataset_root=dataset_root,
            episodes=episodes,
            out_path=args.out.resolve(),
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            include_state=args.include_state,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
