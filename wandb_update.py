"""
Delete files from old Wandb projects to clean up space.

Author:
Collin McCarthy
https://github.com/collinmccarthy/wandb-scripts

Examples:
- Delete all checkpoints
    ```
    python wandb_update.py \
    --wandb-entity=$WANDB_ENTITY \
    --wandb-project=$WANDB_PROJECT \
    --delete_filename_regex=".*\.pth"
    ```
- Delete all bbox and segm checkpoints for runs matching 'm2f_.*_city_.*'
    ```
    python wandb_update.py \
    --wandb-entity=$WANDB_ENTITY \
    --wandb-project=$WANDB_PROJECT \
    --delete_filename_regex '.*_segm_.*\.pth' '.*_bbox_.*\.pth' \
    --select-run-names-regex 'm2f_.*_city.*'
    ```
- Delete all files in folder 'predictions/', except for runs matching 'm2f_.*_city_.*'
    ```
    python wandb_update.py \
    --wandb-entity=$WANDB_ENTITY \
    --wandb-project=$WANDB_PROJECT \
    --delete_filename_regex="^predictions/.*" \
    --skip-run-names-regex=$SKIP_RUN
    ```
- Delete all artifacts ending in '_pred_final', except for run $SKIP_RUN
    ```
    python wandb_update.py \
    --wandb-entity=$WANDB_ENTITY \
    --wandb-project=$WANDB_PROJECT \
    --delete-artifact-regex=".*_pred_final" \
    --skip-run-names-regex=$SKIP_RUN
    ```
"""

import argparse
import re
from argparse import Namespace

from tqdm import tqdm

import wandb


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Clean up Wandb runs remotely")
    parser.add_argument(
        "--wandb-entity",
        "--wandb_entity",
        required=True,
        help="Wandb entity name (e.g. username or team name)",
    )
    parser.add_argument(
        "--wandb-project",
        "--wandb_project",
        required=True,
        help="Wandb project name",
    )
    parser.add_argument(
        "--delete-filename-regex",
        "--delete_filename_regex",
        nargs="+",
        default=list(),
        help="Delete all files in all runs in the project that match this regular expression",
    )
    parser.add_argument(
        "--delete-artifact-regex",
        "--delete_artifact_regex",
        nargs="+",
        default=list(),
        help="Delete all artifacts in all runs in the project that match this regular expression",
    )
    parser.add_argument(
        "--skip-run-ids",
        "--skip_run_ids",
        nargs="+",
        default=list(),
        help="Run ids to skip",
    )
    parser.add_argument(
        "--select-run-ids",
        "--select_run_ids",
        nargs="+",
        default=list(),
        help="Run ids to filter / select",
    )
    parser.add_argument(
        "--skip-run-names-regex",
        "--skip_run_names_regex",
        nargs="+",
        default=list(),
        help="Run names to skip",
    )
    parser.add_argument(
        "--select-run-names-regex",
        "--select_run_names_regex",
        nargs="+",
        default=list(),
        help="Run names to filter / select",
    )

    args = parser.parse_args()

    if len(args.wandb_entity) == 0:
        raise RuntimeError(f"Found empty string for --wandb-entity")

    if len(args.wandb_project) == 0:
        raise RuntimeError(f"Found empty string for --wandb-project")

    return args


def delete_files(args: Namespace):
    api = wandb.Api()
    print(f"Querying Wandb entity: {args.wandb_entity}, project: {args.wandb_project}")
    runs = api.runs(f"{args.wandb_entity}/{args.wandb_project}")

    if len(args.delete_filename_regex) > 0:
        print(
            f"Searching {len(runs)} runs for filenames matching {args.delete_filename_regex}"
        )
    if len(args.delete_artifact_regex) > 0:
        print(
            f"Searching {len(runs)} runs for artifacts matching {args.delete_artifact_regex}"
        )

    all_delete_files = []
    all_delete_artifacts = []
    for run in tqdm(runs, desc="Run"):
        if len(args.skip_run_ids) > 0 and run.id in args.skip_run_ids:
            # print(f"Skipping run id {run.id} (found in --skip-run-ids)")
            continue

        if len(args.filter_run_ids) > 0 and run.id not in args.filter_run_ids:
            # print(f"Skipping run id {run.id} (not found in --select-run-ids)")
            continue

        if len(args.skip_run_names_regex) > 0 and any(
            [re.search(regex, run.name) for regex in args.skip_run_names_regex]
        ):
            # print(f"Skipping run name {run.name} (matched with --skip-run-names-regex)")
            continue

        if len(args.filter_run_names_regex) > 0 and not any(
            [re.search(regex, run.name) for regex in args.filter_run_names_regex]
        ):
            # print(f"Skipping run name {run.name} (not matched with --select-run-names-regex)")
            continue

        for regex in args.delete_filename_regex:
            files = run.files()
            delete_files = [f for f in files if re.search(regex, f.name)]
            if len(delete_files) > 0:
                delete_filenames = [f.name for f in delete_files]
                delete_filenames_str = "\n  " + "\n  ".join(delete_filenames)
                print(
                    f"Found {len(delete_files)} matching files from {run.name}:"
                    f"{delete_filenames_str}"
                )
                all_delete_files.extend(delete_files)

        for regex in args.delete_artifact_regex:
            artifacts = run.logged_artifacts()
            delete_artifacts = [f for f in artifacts if re.search(regex, f.name)]
            if len(delete_artifacts) > 0:
                delete_artifact_names = [f.name for f in delete_artifacts]
                delete_artifacts_str = "\n  " + "\n  ".join(delete_artifact_names)
                print(
                    f"Found {len(delete_artifacts)} matching artifacts from {run.name}:"
                    f"{delete_artifacts_str}"
                )
                all_delete_artifacts.extend(delete_artifacts)

    if len(all_delete_files + all_delete_artifacts) == 0:
        print(f"Found no matching files to delete")
    else:
        response = input(
            f"Delete {len(all_delete_files + all_delete_artifacts)} files? (y/N): "
        )
        if response.lower() == "y":
            print(f"Deleting files (response=y)")
            for f in tqdm(all_delete_files, desc="Files"):
                f.delete()
            for f in tqdm(all_delete_artifacts, desc="Artifacts"):
                f.delete(delete_aliases=True)
        else:
            print(f"Skipping deletion (response=n)")


if __name__ == "__main__":
    args = parse_args()
    if len(args.delete_filename_regex) > 0 or len(args.delete_artifact_regex) > 0:
        delete_files(args)
