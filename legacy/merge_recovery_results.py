"""
merge_recovery_results.py
==========================

recovery_grid.py's checkpoint is a single JSON dict (unlike
identifiability_study.py's append-only JSON-lines), which is NOT safe
for concurrent array-job workers to share one results_path -- a
read-modify-write race could silently drop results. So each array task
writes to its OWN file (results_task${SLURM_ARRAY_TASK_ID}.json); this
script combines them back into one file for plot_recovery_grid.py.

Fails loudly (not silently) if the same cell key appears in more than
one input file with a DIFFERENT result -- that would mean two workers
somehow computed the same cell, which should never happen if the
--task-id/--n-tasks slicing was used consistently across all files, and
silently picking one would hide a real configuration mistake (e.g. two
array jobs launched with different --n-tasks and told they didn't
overlap when they did).
"""

from __future__ import annotations

import json
import argparse
import glob


def merge_recovery_results(input_paths: list[str], output_path: str) -> dict:
    merged: dict = {}
    conflicts = []

    for path in input_paths:
        with open(path) as f:
            data = json.load(f)
        for key, value in data.items():
            if key in merged and merged[key] != value:
                conflicts.append((key, path))
            merged[key] = value

    if conflicts:
        raise ValueError(
            f"{len(conflicts)} cell(s) appeared with DIFFERING results across input files -- "
            f"this should never happen with correct --task-id/--n-tasks slicing. "
            f"Conflicting keys (first 5): {conflicts[:5]}. "
            f"Check that all array tasks used the SAME --n-tasks value."
        )

    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Merged {len(input_paths)} file(s) -> {len(merged)} unique cell(s) -> {output_path}")
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", required=True,
                         help="Shell glob for input files, e.g. 'figures/recovery_task*.json' "
                              "(quote it so YOUR shell doesn't expand it before Python sees it).")
    parser.add_argument("--output-path", default="figures/recovery_grid_results_merged.json")
    args = parser.parse_args()

    input_paths = sorted(glob.glob(args.glob))
    if not input_paths:
        raise SystemExit(f"No files matched glob {args.glob!r}")
    print(f"Found {len(input_paths)} input file(s): {input_paths}")
    merge_recovery_results(input_paths, args.output_path)