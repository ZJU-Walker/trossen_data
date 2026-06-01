# 2026-05-26 Subtask Segment Labels

This directory contains first-pass segment labels for the marker handoff
datasets:

- `subtask_segments_0526_auto.csv`: auto-generated rough labels.
- `subtask_segments_0526_summary.csv`: validation/class-balance summary.

The auto CSV is not final ground truth. Every row is marked `source=auto` and
`notes=needs_manual_review`; copy it to `subtask_segments_0526_reviewed.csv`
after checking and correcting transition frames against the videos.

Generate labels:

```bash
python3 scripts/create_subtask_segments_0526.py
```

Validate an edited CSV:

```bash
python3 scripts/create_subtask_segments_0526.py --validate labels/subtask_segments_0526_reviewed.csv
```

Label sequences:

- `data_robot_give_0526`: `keep_open -> close -> keep_closed`
- `data_robot_pull_0526`: `keep_closed -> open -> keep_open`

The rough detector uses `action[13]`, the active right gripper joint detected
from these datasets.
