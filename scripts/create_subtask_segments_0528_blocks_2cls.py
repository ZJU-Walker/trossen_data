#!/usr/bin/env python3
"""
Create and validate the 2-class block-handover subtask labels for the
2026-05-28 memory datasets (0528_green_block_mem, 0528_yellow_block_mem).

No observe phase: each episode is a SINGLE segment covering the whole video,
labeled by its block color.

  green episodes:  [0 .. end] -> put_green_block_to_plate
  yellow episodes: [0 .. end] -> put_yellow_block_to_plate

Same CSV columns / provenance as the other generators.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATASETS = [
    Path("0528_green_block_mem"),
    Path("0528_yellow_block_mem"),
]
DEFAULT_AUTO_OUT = Path("scripts/labels/subtask_segments_0528_blocks_2cls_auto.csv")
DEFAULT_SUMMARY_OUT = Path("scripts/labels/subtask_segments_0528_blocks_2cls_summary.csv")

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

# 2-class scheme: the whole episode is the block color, one segment per episode.
LABEL_BY_DATASET = {
    "0528_green_block_mem": "put_green_block_to_plate",
    "0528_yellow_block_mem": "put_yellow_block_to_plate",
}


@dataclass(frozen=True)
class Episode:
    dataset: str
    episode_id: int
    task: str
    length: int


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def generate_rows(dataset_roots: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for dataset_root in dataset_roots:
        dataset_root = dataset_root.resolve()
        dataset = dataset_root.name
        if dataset not in LABEL_BY_DATASET:
            raise ValueError(f"Unsupported dataset name: {dataset}")
        label = LABEL_BY_DATASET[dataset]
        for ep in load_episodes(dataset_root):
            rows.append({
                "dataset": dataset,
                "episode_id": f"{ep.episode_id:06d}",
                "task": ep.task,
                "start_frame": "0",
                "end_frame": str(ep.length - 1),
                "subtask": label,
                "source": "auto",
                "notes": "scheme=block_2cls; whole_episode",
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
        if dataset not in LABEL_BY_DATASET:
            errors.append(f"unsupported dataset in CSV: {dataset}")
            continue
        if len(ep_rows) != 1:
            errors.append(f"{dataset} episode_{episode_id}: expected 1 segment, got {len(ep_rows)}")
        row = ep_rows[0]
        expected_label = LABEL_BY_DATASET[dataset]
        if row["subtask"] != expected_label:
            errors.append(
                f"{dataset} episode_{episode_id}: expected {expected_label}, got {row['subtask']}"
            )
        if key in lengths:
            try:
                start = int(row["start_frame"])
                end = int(row["end_frame"])
            except ValueError:
                errors.append(f"{dataset} episode_{episode_id}: non-integer frame range")
                continue
            if start != 0:
                errors.append(f"{dataset} episode_{episode_id}: start_frame {start} != 0")
            if end != lengths[key] - 1:
                errors.append(
                    f"{dataset} episode_{episode_id}: end_frame {end} != length-1 {lengths[key] - 1}"
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
    parser.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--out-auto", type=Path, default=DEFAULT_AUTO_OUT)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT)
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
            rows = generate_rows(dataset_roots)
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
