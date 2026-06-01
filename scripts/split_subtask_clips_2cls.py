#!/usr/bin/env python3
"""Cut per-subtask video clips from episode mp4s using the 2-class segment CSV.

For each segment row (dataset, episode_id, start_frame, end_frame, subtask) this
extracts EXACTLY frames [start_frame .. end_frame] (inclusive) from the source
cam_high episode video, frame-accurately, by re-encoding with ffmpeg's select
filter. Output clips are named:

  <dataset>_ep<episode_id>_<subtask>_f<start>-<end>_cam_high.mp4

Defaults to episodes 0 and 1 of both marker datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


DATA_ROOT = Path("/iris/projects/humanoid/trossen_data")
DEFAULT_CSV = DATA_ROOT / "scripts" / "labels" / "subtask_segments_0526_2cls_auto.csv"
DEFAULT_OUT = DATA_ROOT / "scripts" / "clips" / "subtask_clips_2cls"
CAMERA = "observation.images.cam_high"


def load_info(dataset: str) -> dict:
    return json.loads((DATA_ROOT / dataset / "meta" / "info.json").read_text())


def source_video(dataset: str, episode_id: str, chunks_size: int) -> Path:
    chunk = int(episode_id) // chunks_size
    return (
        DATA_ROOT
        / dataset
        / "videos"
        / f"chunk-{chunk:03d}"
        / CAMERA
        / f"episode_{episode_id}.mp4"
    )


def cut_clip(src: Path, dst: Path, start_frame: int, end_frame: int, fps: float) -> None:
    """Extract exactly frames [start_frame, end_frame] inclusive, re-encoded."""
    # select frames by index n; setpts to rebuild a clean timeline at the same fps.
    vf = (
        f"select='between(n,{start_frame},{end_frame})',"
        f"setpts=N/FRAME_RATE/TB"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vf",
        vf,
        "-r",
        f"{fps}",
        "-vsync",
        "vfr",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def count_frames(path: Path) -> int:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(out.stdout.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--episodes",
        nargs="+",
        default=["000000", "000001"],
        help="Episode ids (zero-padded) to cut.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["data_robot_give_0526", "data_robot_pull_0526"],
    )
    args = parser.parse_args()

    wanted_eps = {f"{int(e):06d}" for e in args.episodes}
    wanted_ds = set(args.datasets)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    info_cache: dict[str, dict] = {}
    manifest = []

    with args.csv.open(newline="") as f:
        rows = list(csv.DictReader(f))

    selected = [
        r
        for r in rows
        if r["dataset"] in wanted_ds and f"{int(r['episode_id']):06d}" in wanted_eps
    ]
    if not selected:
        print("No matching segments found.", file=sys.stderr)
        return 1

    for row in selected:
        dataset = row["dataset"]
        episode_id = f"{int(row['episode_id']):06d}"
        subtask = row["subtask"]
        start_frame = int(row["start_frame"])
        end_frame = int(row["end_frame"])

        if dataset not in info_cache:
            info_cache[dataset] = load_info(dataset)
        info = info_cache[dataset]
        fps = float(info["fps"])
        chunks_size = int(info["chunks_size"])

        src = source_video(dataset, episode_id, chunks_size)
        if not src.is_file():
            print(f"  [skip] missing source: {src}", file=sys.stderr)
            continue

        dst = (
            args.out_dir
            / f"{dataset}_ep{episode_id}_{subtask}_f{start_frame}-{end_frame}_cam_high.mp4"
        )
        expected = end_frame - start_frame + 1
        cut_clip(src, dst, start_frame, end_frame, fps)
        actual = count_frames(dst)
        ok = actual == expected
        print(
            f"{'OK ' if ok else 'WARN'} {dst.name}  "
            f"frames={actual} (expected {expected})"
        )
        manifest.append(
            {
                "clip": dst.name,
                "dataset": dataset,
                "episode_id": episode_id,
                "subtask": subtask,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "expected_frames": expected,
                "actual_frames": actual,
                "fps": fps,
                "source_video": str(src),
            }
        )

    (args.out_dir / "clips_manifest.json").write_text(json.dumps(manifest, indent=2))
    n_ok = sum(1 for m in manifest if m["actual_frames"] == m["expected_frames"])
    print(f"\nWrote {len(manifest)} clips ({n_ok} frame-exact) to {args.out_dir}")
    return 0 if n_ok == len(manifest) else 1


if __name__ == "__main__":
    raise SystemExit(main())
