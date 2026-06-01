#!/usr/bin/env python3
"""
Create and validate the 3-class block-handover subtask labels for the
2026-05-28 memory datasets (0528_green_block_mem, 0528_yellow_block_mem).

Per-frame label scheme (2 segments per episode):

  observe_human            = [0 .. onset-1]   robot stationary, human pointing
  put_<color>_block_to_plate = [onset .. end]  robot moving block to the plate

  green episodes: observe_human -> put_green_block_to_plate
  yellow episodes: observe_human -> put_yellow_block_to_plate

The split boundary `onset` is the first frame of sustained robot motion,
detected from observation.state velocity. During the observe phase the robot
is held still while the human points; once it commits to a block it starts
moving, which is a clean, automatically-detectable boundary.

Mirrors create_subtask_segments_0526_2cls.py (same CSV columns, same
auto/needs_manual_review provenance) but uses motion onset instead of a
gripper threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATASETS = [
    Path("0528_green_block_mem"),
    Path("0528_yellow_block_mem"),
]
DEFAULT_AUTO_OUT = Path("scripts/labels/subtask_segments_0528_blocks_auto.csv")
DEFAULT_SUMMARY_OUT = Path("scripts/labels/subtask_segments_0528_blocks_summary.csv")

CSV_COLUMNS = [
    "dataset",
    "episode_id",
    "task",
    "start_frame",
    "end_frame",
    "subtask",
    "source",
    "notes",
]

TASK_BY_DATASET = {
    "0528_green_block_mem": "put the green block to the plate",
    "0528_yellow_block_mem": "put the yellow block to the plate",
}

# 3-class scheme: every episode starts with observe_human, then transitions to
# the dataset-specific put_<color>_block_to_plate the moment the robot moves.
SEQUENCE_BY_DATASET = {
    "0528_green_block_mem": ("observe_human", "put_green_block_to_plate"),
    "0528_yellow_block_mem": ("observe_human", "put_yellow_block_to_plate"),
}

# Motion-onset detection parameters.
VEL_THRESHOLD_FRAC = 0.10   # fraction of the episode's peak velocity
SUSTAIN_FRAMES = 5          # require motion sustained over this many frames
SUSTAIN_FRAC = 0.6          # at least this fraction of the window moving


@dataclass(frozen=True)
class Episode:
    dataset: str
    episode_id: int
    task: str
    length: int


@dataclass(frozen=True)
class Detection:
    onset_frame: int
    peak_velocity: float
    method: str
    warning: str | None = None


class ParquetReader:
    """Small compatibility wrapper around full or low-level pyarrow parquet."""

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

    def read_vector_column(self, path: Path, column_name: str) -> list[list[float]]:
        if self._pq is None:
            raise RuntimeError("No parquet reader is available")
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
                raise KeyError(f"Column not found in parquet: {column_name}")
            return reader.read_column(column_index).to_pylist()
        finally:
            reader.close()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_info(dataset_root: Path) -> dict:
    path = dataset_root / "meta" / "info.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    return json.loads(path.read_text())


def load_episodes(dataset_root: Path) -> list[Episode]:
    dataset = dataset_root.name
    task = TASK_BY_DATASET.get(dataset, "")
    episodes = []
    for row in load_jsonl(dataset_root / "meta" / "episodes.jsonl"):
        episodes.append(
            Episode(
                dataset=dataset,
                episode_id=int(row["episode_index"]),
                task=task,
                length=int(row["length"]),
            )
        )
    return sorted(episodes, key=lambda ep: ep.episode_id)


def parquet_path(dataset_root: Path, episode_id: int, chunks_size: int) -> Path:
    chunk = episode_id // chunks_size
    return dataset_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_id:06d}.parquet"


def state_velocity(vectors: list[list[float]]) -> list[float]:
    """Per-step L2 norm of the state difference: ||s[t+1]-s[t]||."""
    vel = []
    for prev, cur in zip(vectors, vectors[1:]):
        s = 0.0
        for a, b in zip(prev, cur):
            d = float(b) - float(a)
            s += d * d
        vel.append(s ** 0.5)
    return vel


def detect_onset(vectors: list[list[float]], length: int) -> Detection:
    if len(vectors) < SUSTAIN_FRAMES + 2:
        return Detection(max(1, int(round(length * 0.35))), 0.0,
                         "fallback_short_episode", "too_few_frames")
    vel = state_velocity(vectors)
    peak = max(vel) if vel else 0.0
    if peak <= 1e-6:
        return Detection(max(1, int(round(length * 0.35))), peak,
                         "fallback_no_motion", "no_motion_detected")
    thr = peak * VEL_THRESHOLD_FRAC
    moving = [v > thr for v in vel]
    onset = -1
    for i in range(len(moving) - SUSTAIN_FRAMES):
        window = moving[i:i + SUSTAIN_FRAMES]
        if moving[i] and (sum(window) / len(window)) >= SUSTAIN_FRAC:
            onset = i
            break
    if onset < 0:
        onset = next((i for i, m in enumerate(moving) if m), int(round(length * 0.35)))
    # onset indexes into vel (between frame i and i+1); use i+1 as the first
    # "moving" frame so the observe segment captures the last still frame.
    onset_frame = onset + 1
    onset_frame = max(1, min(onset_frame, length - 1))
    return Detection(onset_frame, peak, "state_velocity_onset", None)


def generate_rows(dataset_roots: list[Path], signal_column: str,
                  reader: ParquetReader) -> list[dict]:
    rows: list[dict] = []
    for dataset_root in dataset_roots:
        dataset_root = dataset_root.resolve()
        dataset = dataset_root.name
        if dataset not in SEQUENCE_BY_DATASET:
            raise ValueError(f"Unsupported dataset name: {dataset}")
        info = load_info(dataset_root)
        chunks_size = int(info["chunks_size"])
        episodes = load_episodes(dataset_root)

        for ep in episodes:
            path = parquet_path(dataset_root, ep.episode_id, chunks_size)
            if reader.available and path.is_file():
                vectors = reader.read_vector_column(path, signal_column)
                detection = detect_onset(vectors, ep.length)
            else:
                detection = Detection(max(1, int(round(ep.length * 0.35))), 0.0,
                                      "fallback_parquet_unavailable",
                                      f"parquet_unavailable_for_{dataset}")

            state_before, state_after = SEQUENCE_BY_DATASET[dataset]
            boundary = max(1, min(detection.onset_frame, ep.length - 1))
            segments = [
                (0, boundary - 1, state_before),
                (boundary, ep.length - 1, state_after),
            ]
            warning_note = f"; warning={detection.warning}" if detection.warning else ""
            notes = (
                "needs_manual_review"
                "; scheme=block_3cls"
                f"; method={detection.method}"
                f"; signal={signal_column}"
                f"; onset={detection.onset_frame}"
                f"; peak_vel={detection.peak_velocity:.5f}"
                f"{warning_note}"
            )
            for start, end, subtask in segments:
                rows.append({
                    "dataset": dataset,
                    "episode_id": f"{ep.episode_id:06d}",
                    "task": ep.task,
                    "start_frame": str(start),
                    "end_frame": str(end),
                    "subtask": subtask,
                    "source": "auto",
                    "notes": notes,
                })
    return rows


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_segments(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = [col for col in CSV_COLUMNS if col not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        return list(reader)


def dataset_lengths(dataset_roots: list[Path]) -> dict[tuple[str, str], int]:
    lengths = {}
    for dataset_root in dataset_roots:
        for ep in load_episodes(dataset_root):
            lengths[(ep.dataset, f"{ep.episode_id:06d}")] = ep.length
    return lengths


def validate_rows(rows: list[dict],
                  lengths: dict[tuple[str, str], int]) -> tuple[list[str], list[dict]]:
    errors: list[str] = []
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["episode_id"])].append(row)

    expected_keys = set(lengths)
    actual_keys = set(grouped)
    for missing in sorted(expected_keys - actual_keys):
        errors.append(f"missing episode labels: {missing[0]} episode_{missing[1]}")
    for extra in sorted(actual_keys - expected_keys):
        errors.append(f"unknown episode labels: {extra[0]} episode_{extra[1]}")

    for key, ep_rows in sorted(grouped.items()):
        dataset, episode_id = key
        if dataset not in SEQUENCE_BY_DATASET:
            errors.append(f"unsupported dataset in CSV: {dataset}")
            continue
        ep_rows.sort(key=lambda row: int(row["start_frame"]))
        sequence = tuple(row["subtask"] for row in ep_rows)
        expected_sequence = SEQUENCE_BY_DATASET[dataset]
        if sequence != expected_sequence:
            errors.append(
                f"{dataset} episode_{episode_id}: expected sequence {expected_sequence}, got {sequence}"
            )
        if len(ep_rows) != 2:
            errors.append(f"{dataset} episode_{episode_id}: expected 2 segments, got {len(ep_rows)}")
        if key not in lengths:
            continue
        expected_start = 0
        for row in ep_rows:
            try:
                start = int(row["start_frame"])
                end = int(row["end_frame"])
            except ValueError:
                errors.append(f"{dataset} episode_{episode_id}: non-integer frame range")
                continue
            if start != expected_start:
                errors.append(
                    f"{dataset} episode_{episode_id}: gap/overlap before {row['subtask']} "
                    f"(expected start {expected_start}, got {start})"
                )
            if end < start:
                errors.append(f"{dataset} episode_{episode_id}: negative segment for {row['subtask']}")
            expected_start = end + 1
        final_expected = lengths[key]
        if expected_start != final_expected:
            errors.append(
                f"{dataset} episode_{episode_id}: final frame coverage ends at {expected_start - 1}, "
                f"expected {final_expected - 1}"
            )

    summary_acc: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        try:
            frames = int(row["end_frame"]) - int(row["start_frame"]) + 1
        except ValueError:
            continue
        summary_acc[(row["dataset"], row["subtask"])].append(frames)

    summary = []
    for (dataset, subtask), frame_counts in sorted(summary_acc.items()):
        total = sum(frame_counts)
        mean = total / len(frame_counts)
        summary.append({
            "dataset": dataset,
            "subtask": subtask,
            "num_segments": str(len(frame_counts)),
            "total_frames": str(total),
            "mean_frames": f"{mean:.2f}",
        })
    return errors, summary


def print_summary(summary: list[dict]) -> None:
    print("dataset,subtask,num_segments,total_frames,mean_frames")
    for row in summary:
        print(
            f"{row['dataset']},{row['subtask']},{row['num_segments']},"
            f"{row['total_frames']},{row['mean_frames']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS,
                        help="Dataset roots to label/validate.")
    parser.add_argument("--out-auto", type=Path, default=DEFAULT_AUTO_OUT)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--signal-column", default="observation.state",
                        choices=("action", "observation.state"))
    parser.add_argument("--validate", type=Path, default=None,
                        help="Validate an existing segment CSV instead of generating one.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_roots = [path.resolve() for path in args.datasets]
    for dataset_root in dataset_roots:
        if not dataset_root.is_dir():
            print(f"ERROR: dataset root does not exist: {dataset_root}", file=sys.stderr)
            return 2
    try:
        lengths = dataset_lengths(dataset_roots)
        if args.validate is not None:
            rows = read_segments(args.validate)
        else:
            reader = ParquetReader()
            print(f"Parquet reader: {reader.mode}")
            rows = generate_rows(dataset_roots, args.signal_column, reader)
            write_csv(args.out_auto, rows, CSV_COLUMNS)
            print(f"Wrote {args.out_auto}")
        errors, summary = validate_rows(rows, lengths)
        write_csv(args.summary_out, summary,
                  ["dataset", "subtask", "num_segments", "total_frames", "mean_frames"])
        print(f"Wrote {args.summary_out}")
        print_summary(summary)
        if errors:
            print("\nValidation errors:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            return 1
        print("\nValidation passed.")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
