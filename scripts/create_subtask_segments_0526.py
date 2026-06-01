#!/usr/bin/env python3
"""
Create and validate subtask segment labels for the 2026-05-26 marker datasets.

The script writes a segment-level CSV for:
  - data_robot_give_0526: keep_open -> close -> keep_closed
  - data_robot_pull_0526: keep_closed -> open -> keep_open

The first-pass labels are intentionally marked as source=auto and
notes=needs_manual_review. Review transition frames against video before using
the labels as ground truth.
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
from typing import Iterable


DEFAULT_DATASETS = [
    Path("data_robot_give_0526"),
    Path("data_robot_pull_0526"),
]
DEFAULT_AUTO_OUT = Path("labels/subtask_segments_0526_auto.csv")
DEFAULT_SUMMARY_OUT = Path("labels/subtask_segments_0526_summary.csv")

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
    "data_robot_give_0526": "take the marker from the human",
    "data_robot_pull_0526": "give the marker to the human",
}

SEQUENCE_BY_DATASET = {
    "data_robot_give_0526": ("keep_open", "close", "keep_closed"),
    "data_robot_pull_0526": ("keep_closed", "open", "keep_open"),
}

TRANSITION_BY_DATASET = {
    "data_robot_give_0526": "close",
    "data_robot_pull_0526": "open",
}

DIRECTION_BY_DATASET = {
    "data_robot_give_0526": "falling",
    "data_robot_pull_0526": "rising",
}


@dataclass(frozen=True)
class Episode:
    dataset: str
    episode_id: int
    task: str
    length: int


@dataclass(frozen=True)
class Detection:
    start_frame: int
    end_frame: int
    signal_column: str
    gripper_joint: int
    start_level: float | None
    end_level: float | None
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


def load_task(dataset_root: Path) -> str:
    rows = load_jsonl(dataset_root / "meta" / "tasks.jsonl")
    if not rows:
        raise ValueError(f"No tasks found for {dataset_root}")
    return str(rows[0]["task"])


def load_episodes(dataset_root: Path) -> list[Episode]:
    dataset = dataset_root.name
    task = load_task(dataset_root)
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


def chunk_for(episode_id: int, chunks_size: int) -> int:
    return episode_id // chunks_size


def parquet_path(dataset_root: Path, episode_id: int, chunks_size: int) -> Path:
    chunk = chunk_for(episode_id, chunks_size)
    return dataset_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_id:06d}.parquet"


def median(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        raise ValueError("Cannot compute median of empty values")
    return float(statistics.median(vals))


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values[:]

    half = window // 2
    smoothed = []
    for idx in range(len(values)):
        lo = max(0, idx - half)
        hi = min(len(values), idx + half + 1)
        smoothed.append(sum(values[lo:hi]) / (hi - lo))
    return smoothed


def movement_score(vectors: list[list[float]], joint: int) -> float:
    values = [float(row[joint]) for row in vectors]
    if not values:
        return 0.0
    return max(values) - min(values)


def infer_gripper_joint(
    dataset_root: Path,
    episodes: list[Episode],
    chunks_size: int,
    reader: ParquetReader,
    signal_column: str,
    candidates: tuple[int, ...],
) -> int:
    if not reader.available:
        return candidates[-1]

    scores = {joint: [] for joint in candidates}
    for ep in episodes[: min(8, len(episodes))]:
        path = parquet_path(dataset_root, ep.episode_id, chunks_size)
        vectors = reader.read_vector_column(path, signal_column)
        for joint in candidates:
            scores[joint].append(movement_score(vectors, joint))

    median_scores = {
        joint: median(values) if values else 0.0
        for joint, values in scores.items()
    }
    return max(candidates, key=lambda joint: median_scores[joint])


def fallback_detection(length: int, dataset: str, fps: int, signal_column: str, gripper_joint: int) -> Detection:
    center = int(round(length * 0.60))
    min_frames = max(9, int(round(fps * 0.40)))
    start = max(1, center - min_frames // 2)
    end = min(length - 2, start + min_frames - 1)
    start = max(1, end - min_frames + 1)
    return Detection(
        start_frame=start,
        end_frame=end,
        signal_column=signal_column,
        gripper_joint=gripper_joint,
        start_level=None,
        end_level=None,
        method="fallback_length_fraction",
        warning=f"parquet_unavailable_for_{dataset}",
    )


def detect_transition(
    values: list[float],
    dataset: str,
    fps: int,
    signal_column: str,
    gripper_joint: int,
) -> Detection:
    if len(values) < 3:
        raise ValueError("Need at least 3 frames to detect a transition")

    baseline_window = max(5, min(15, len(values) // 10))
    start_level = median(values[:baseline_window])
    end_level = median(values[-baseline_window:])
    delta = end_level - start_level
    expected_direction = DIRECTION_BY_DATASET[dataset]
    actual_direction = "rising" if delta > 0 else "falling"
    warning = None

    if abs(delta) < 0.002:
        fallback = fallback_detection(len(values), dataset, fps, signal_column, gripper_joint)
        return Detection(
            fallback.start_frame,
            fallback.end_frame,
            signal_column,
            gripper_joint,
            start_level,
            end_level,
            "fallback_small_delta",
            "small_gripper_delta",
        )

    if actual_direction != expected_direction:
        warning = f"expected_{expected_direction}_got_{actual_direction}"

    smoothed = moving_average(values, window=5)
    start_threshold = start_level + delta * 0.20
    end_threshold = start_level + delta * 0.80

    if delta < 0:
        raw_start = next(
            (idx for idx, value in enumerate(smoothed) if value <= start_threshold),
            int(round(len(values) * 0.60)),
        )
        raw_end = next(
            (idx for idx, value in enumerate(smoothed[raw_start:], raw_start) if value <= end_threshold),
            raw_start,
        )
    else:
        raw_start = next(
            (idx for idx, value in enumerate(smoothed) if value >= start_threshold),
            int(round(len(values) * 0.60)),
        )
        raw_end = next(
            (idx for idx, value in enumerate(smoothed[raw_start:], raw_start) if value >= end_threshold),
            raw_start,
        )

    if raw_end < raw_start:
        raw_end = raw_start

    min_transition_frames = max(9, int(round(fps * 0.40)))
    max_transition_frames = max(min_transition_frames, int(round(fps * 0.55)))
    width = raw_end - raw_start + 1

    if width < min_transition_frames:
        center = (raw_start + raw_end) // 2
        start = center - min_transition_frames // 2
        end = start + min_transition_frames - 1
    elif width > max_transition_frames:
        center = (raw_start + raw_end) // 2
        start = center - max_transition_frames // 2
        end = start + max_transition_frames - 1
    else:
        start, end = raw_start, raw_end

    start = max(1, start)
    end = min(len(values) - 2, end)
    if end < start:
        start = max(1, min(raw_start, len(values) - 2))
        end = start

    return Detection(
        start_frame=start,
        end_frame=end,
        signal_column=signal_column,
        gripper_joint=gripper_joint,
        start_level=start_level,
        end_level=end_level,
        method="gripper_threshold",
        warning=warning,
    )


def generate_rows(
    dataset_roots: list[Path],
    signal_column: str,
    gripper_joint_arg: str,
    reader: ParquetReader,
) -> list[dict]:
    rows: list[dict] = []

    for dataset_root in dataset_roots:
        dataset_root = dataset_root.resolve()
        dataset = dataset_root.name
        if dataset not in SEQUENCE_BY_DATASET:
            raise ValueError(f"Unsupported dataset name: {dataset}")

        info = load_info(dataset_root)
        fps = int(round(float(info["fps"])))
        chunks_size = int(info["chunks_size"])
        episodes = load_episodes(dataset_root)

        candidates = (6, 13)
        if gripper_joint_arg == "auto":
            gripper_joint = infer_gripper_joint(dataset_root, episodes, chunks_size, reader, signal_column, candidates)
        else:
            gripper_joint = int(gripper_joint_arg)

        for ep in episodes:
            path = parquet_path(dataset_root, ep.episode_id, chunks_size)
            if reader.available and path.is_file():
                vectors = reader.read_vector_column(path, signal_column)
                values = [float(row[gripper_joint]) for row in vectors]
                detection = detect_transition(values, dataset, fps, signal_column, gripper_joint)
            else:
                detection = fallback_detection(ep.length, dataset, fps, signal_column, gripper_joint)

            first, transition, last = SEQUENCE_BY_DATASET[dataset]
            segments = [
                (0, detection.start_frame - 1, first),
                (detection.start_frame, detection.end_frame, transition),
                (detection.end_frame + 1, ep.length - 1, last),
            ]

            level_note = ""
            if detection.start_level is not None and detection.end_level is not None:
                level_note = f"; levels={detection.start_level:.5f}->{detection.end_level:.5f}"
            warning_note = f"; warning={detection.warning}" if detection.warning else ""
            notes = (
                "needs_manual_review"
                f"; method={detection.method}"
                f"; signal={detection.signal_column}[{detection.gripper_joint}]"
                f"{level_note}{warning_note}"
            )

            for start, end, subtask in segments:
                rows.append(
                    {
                        "dataset": dataset,
                        "episode_id": f"{ep.episode_id:06d}",
                        "task": ep.task,
                        "start_frame": str(start),
                        "end_frame": str(end),
                        "subtask": subtask,
                        "source": "auto",
                        "notes": notes,
                    }
                )

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


def validate_rows(rows: list[dict], lengths: dict[tuple[str, str], int]) -> tuple[list[str], list[dict]]:
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
        if len(ep_rows) != 3:
            errors.append(f"{dataset} episode_{episode_id}: expected 3 segments, got {len(ep_rows)}")

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
        summary.append(
            {
                "dataset": dataset,
                "subtask": subtask,
                "num_segments": str(len(frame_counts)),
                "total_frames": str(total),
                "mean_frames": f"{mean:.2f}",
            }
        )

    return errors, summary


def print_summary(summary: list[dict]) -> None:
    print("dataset,subtask,num_segments,total_frames,mean_frames")
    for row in summary:
        print(
            f"{row['dataset']},{row['subtask']},{row['num_segments']},"
            f"{row['total_frames']},{row['mean_frames']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=Path,
        default=DEFAULT_DATASETS,
        help="Dataset roots to label/validate.",
    )
    parser.add_argument(
        "--out-auto",
        type=Path,
        default=DEFAULT_AUTO_OUT,
        help="Auto-generated segment CSV path.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=DEFAULT_SUMMARY_OUT,
        help="Summary CSV path.",
    )
    parser.add_argument(
        "--signal-column",
        default="action",
        choices=("action", "observation.state"),
        help="Parquet vector column used for rough transition detection.",
    )
    parser.add_argument(
        "--gripper-joint",
        default="auto",
        help="Gripper joint index to use, or 'auto' to choose between 6 and 13.",
    )
    parser.add_argument(
        "--validate",
        type=Path,
        default=None,
        help="Validate an existing segment CSV instead of generating one.",
    )
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
            rows = generate_rows(dataset_roots, args.signal_column, args.gripper_joint, reader)
            write_csv(args.out_auto, rows, CSV_COLUMNS)
            print(f"Wrote {args.out_auto}")

        errors, summary = validate_rows(rows, lengths)
        write_csv(args.summary_out, summary, ["dataset", "subtask", "num_segments", "total_frames", "mean_frames"])
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
