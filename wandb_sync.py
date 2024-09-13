"""
Sync a checkpoint file from a Wandb project to a local directory.

Author:
Collin McCarthy
https://github.com/collinmccarthy/wandb-scripts

Examples:
- Upload checkpoints
    ```
    python sync_runs.py \
        --wandb-project=$WANDB_PROJECT \
        --wandb-entity=$WANDB_ENTITY \
        --results-dir=<PARENT_DIR_FOR_ALL_RUNS> \
        --checkpoint-name checkpoint-best.pth \
        --upload-checkpoints
    ```

- Download checkpoints
    ```
    python sync_runs.py \
        --wandb-project=$WANDB_PROJECT \
        --wandb-entity=$WANDB_ENTITY \
        --results-dir=<PARENT_DIR_FOR_ALL_RUNS> \
        --checkpoint-name checkpoint-best.pth \
        --download-checkpoints
    ```

- Upload (sync) runs (upload runs)
    - Files previously uploaded (added as symlinks in wandb dir) will be uploaded if symlink exists
    - To upload a new checkpoint, add --checkpoint_name as in uploading checkpoints above
    ```
    python sync_runs.py \
        --wandb-project=$WANDB_PROJECT \
        --wandb-entity=$WANDB_ENTITY \
        --results-dir=<PARENT_DIR_FOR_ALL_RUNS> \
        --run-names
        --upload-runs \
    ```

- Download runs
    - Will download all files, don't need --checkpoint_name
    ```
    python sync_runs.py \
        --wandb-project=$WANDB_PROJECT \
        --wandb-entity=$WANDB_ENTITY \
        --results-dir=<PARENT_DIR_FOR_ALL_RUNS> \
        --download-runs
    ```
"""

import sys
import os
import argparse
import subprocess
import re
from argparse import Namespace
from pathlib import Path
from typing import List, Union
from tqdm import tqdm

import wandb
from wandb.apis.public.runs import Run as ApiRun
from wandb.apis.public.runs import Runs as ApiRuns


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Sync wandb checkpoint or run locally")
    # Required
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
        "--results_dir",
        required=True,
        help="Path to results dir, parent directory containing all run folders.",
    )

    # Required to set ONE of the following flags
    parser.add_argument(
        "--download-checkpoints",
        "--download_checkpoints",
        action="store_true",
        default=False,
        help="Sync from Wandb to our local output dir (download checkpoints)",
    )
    parser.add_argument(
        "--download-runs",
        "--download_runs",
        action="store_true",
        default=False,
        help="Sync from Wandb to our local output dir (download full runs / all files)",
    )
    parser.add_argument(
        "--upload-checkpoints",
        "--upload_checkpoints",
        action="store_true",
        default=False,
        help="Sync from our local output dir to Wandb (upload checkpoints)",
    )
    parser.add_argument(
        "--upload-runs",
        "--upload_runs",
        action="store_true",
        default=False,
        help="Sync from our local output dir to Wandb (upload full runs/files whose symlinks exist)",
    )

    # Recommended (required for uploading/downloading checkpoints)
    parser.add_argument(
        "--checkpoint-name",
        "--checkpoint_name",
        default=None,
        help=(
            "Checkpoint name for uploading/downloading checkpoints, or forcing checkpoint upload"
            " during sync via (--upload-runs)"
        ),
    )

    # General
    parser.add_argument(
        "--dry-run",
        "--dry_run",
        action="store_true",
        default=False,
        help="Print the commands but do not run them",
    )
    parser.add_argument(
        "--run-names",
        "--run_names",
        nargs="+",
        default=None,
        help="Filter project runs to only sync these run names (if not provided, will sync all)",
    )
    parser.add_argument(
        "--overwrite-existing",
        "--overwrite_existing",
        action="store_true",
        default=False,
        help="Overwrite existing files (if False, will skip)",
    )

    args = parser.parse_args(sys.argv[1:])
    return args


def get_matching_runs(args: Namespace) -> tuple[Union[ApiRuns, List[ApiRun]], str]:
    api = wandb.Api()
    project_str = f"{args.wandb_entity}/{args.wandb_project}"
    runs: ApiRuns = api.runs(project_str)

    if args.run_names is None:
        return runs, project_str

    keep_runs = []
    run_names = [run.name for run in runs]
    for keyword in args.run_names:  # Treating as keywords not just hard-coded run names
        matching_runs = [
            run for run in runs if re.search(keyword, run.name) is not None
        ]
        if len(matching_runs) != 1:
            print(
                f"Failed to find match for run name / keyword {keyword} in run names: {run_names}"
            )
        keep_runs.append(matching_runs[0])

    # Verify the runs are still unique
    keep_run_names = [run.name for run in keep_runs]
    if len(set(keep_run_names)) != len(args.run_names):
        raise ValueError(
            f"One or more run names / keywords matched the same run, returning fewer runs than requested:"
            f"\n  run names / keywords: {args.run_names}\n  run names: {keep_run_names}"
        )
    return keep_runs, project_str


def download_runs(args: Namespace, checkpoints_only: bool):
    runs, project_str = get_matching_runs(args)

    def _download(wandb_file, run_dir):
        print(f"Downloading {wandb_file.name}", end="")
        if Path(run_dir, wandb_file.name).exists():
            if args.overwrite_existing:
                if not args.dry_run:
                    wandb_file.download(str(run_dir), replace=True)
                print("")
            else:
                print(" File exists and not overwriting, skipping...")
        else:
            if not args.dry_run:
                wandb_file.download(str(run_dir))
            print("")

    for run in tqdm(runs, desc="Run"):
        run_dir = Path(args.results_dir, run.name)
        run_path = f"{project_str}/{run.name}"
        print(f"Creating local directory for run {run_path}:\n  {run_dir}")
        run_dir.mkdir(exist_ok=True, parents=True)

        if not checkpoints_only:
            files = run.files()
            print(f"Syncing {len(files)} files (checkpoints_only=False)")
            for f in files:
                _download(wandb_file=f, run_dir=run_dir)
        else:
            try:
                print(
                    f"Downloading {run_path}/{args.checkpoint_name} to:\n  {run_dir}/{args.checkpoint_name}"
                )
                if not args.dry_run:
                    f = run.file(args.checkpoint_name)
                    _download(wandb_file=f, run_dir=run_dir)
            except Exception as e:
                print(
                    f"Failed to find checkpoint {args.checkpoint_name} in run {run_path}"
                )
                raise e


def upload_checkpoints(args: Namespace):
    # NOTE: Currently getting 403/permission errors, use --upload_runs with --checkpoint_name instead

    parent_dir = args.results_dir
    subdirs = [f for f in Path(parent_dir).iterdir() if f.is_dir()]

    # Get a mapping from run name to run id (from the runs we have uploaded already)
    runs, project_str = get_matching_runs(args)

    # Verify run names are unique
    num_run_names = len(set([run.name for run in runs]))
    if num_run_names != len(runs):
        raise ValueError(
            f"Found duplicate run names in project, cannot determine run from run name: {project_str}"
        )

    run_name_to_run = {run.name: run for run in runs}

    for subdir in tqdm(subdirs, desc="Dir"):
        matching_checkpoints = [
            f for f in subdir.iterdir() if f.name == args.checkpoint_name
        ]
        if len(matching_checkpoints) != 1:
            print(
                f"Failed to find checkpoint {args.checkpoint_name} in directory {subdir}. Skipping directory."
            )
            continue
        checkpoint = matching_checkpoints[0]

        run_name = subdir.name
        run = run_name_to_run[run_name]
        print(f"Uploading checkpoint to {project_str}/{run_name}:\n  {str(checkpoint)}")
        if not args.dry_run:
            assert checkpoint.parent.name == run.name, "Folder does not match run name"
            run.upload_file(str(checkpoint))


def upload_runs(args: Namespace, checkpoints_only: bool = False):
    # NOTE: Should use upload_checkpoints() if checkpoint only, but getting 403 error so added the flag here

    parent_dir = args.results_dir
    subdirs = [f for f in Path(parent_dir).iterdir() if f.is_dir()]

    # Get a mapping from run name to run id (from the runs we have uploaded already)
    runs, _project_str = get_matching_runs(args)
    run_name_to_id = {run.name: run.id for run in runs}

    if args.checkpoint_name is None:
        print(
            "WARNING: Not creating symbolic link for checkpoints, will sync runs using existing wandb file symlinks"
        )

    commands: List[List[str]] = []
    for subdir in subdirs:

        wandb_run_dir = subdir.joinpath("wandb", "latest-run")
        if args.checkpoint_name is not None:
            print(
                f"Creating symbolic link for checkpoing {args.checkpoint_name} to upload file during sync"
            )

            # Get the checkpoint we want to upload
            matching_checkpoints = [
                f for f in subdir.iterdir() if f.name == args.checkpoint_name
            ]
            if len(matching_checkpoints) != 1:
                print(
                    f"Failed to find checkpoint {args.checkpoint_name} in directory {subdir}. Skipping directory."
                )
                continue

            # Get the run directory that we should add this to (most recent wandb run directory matching the run id)
            if subdir.name in run_name_to_id:
                run_id = run_name_to_id[subdir.name]
                wandb_run_dirs = [
                    f
                    for f in subdir.joinpath("wandb").iterdir()
                    if f.is_dir() and run_id in f.name
                ]
                wandb_run_dirs.sort(key=lambda f: os.path.getmtime(f))
                wandb_run_dir = wandb_run_dirs[-1]  # Last created
            else:
                print(
                    f"Could not find run {subdir.name} in project runs. Adding checkpoint symlink to latest-run folder"
                )

            # Create symbolic link from wandb/latest-run/files/<FILENAME> to the checkpoint we want to upload
            checkpoint = matching_checkpoints[0]
            symlink = wandb_run_dir.joinpath("files", checkpoint.name)
            if not symlink.parent.exists():
                print(
                    f"Wandb files directory does not exist: {symlink.parent}. Skipping directory."
                )
                continue

            print(f"Creating symbolic link from:\n{checkpoint} to\n{symlink}")
            if not args.dry_run:
                if symlink.exists():
                    print("Symlink already exists, skipping symlink creation.")
                else:
                    os.symlink(src=str(checkpoint), dst=str(symlink))

        # Call 'wandb sync <directory>' on this directory
        command = [
            "wandb",
            "sync",
            "--project",
            args.wandb_project,
            "--entity",
            args.wandb_entity,
        ]

        if checkpoints_only:
            # Use current wandb_dir which points to the dir the checkpoint was symlinked to
            command.extend(["--include-globs", args.checkpoint_name])
        else:
            # Use the parent dir and pass --sync-all
            wandb_run_dir = wandb_run_dir.parent.joinpath("run-*")
            command.append("--sync-all")

        command.append(str(wandb_run_dir))
        commands.append(command)

    if args.dry_run:
        commands_list = [" ".join(command) for command in commands]
        commands_str = "\n".join(commands_list)
        print(f"Commands to be run (dry-run only):\n{commands_str}")
    else:
        for command in tqdm(commands, desc="Dir"):
            command_str = " ".join(command)
            print(f"Running: {command_str}")
            subprocess.run(command, check=True)


if __name__ == "__main__":
    args = parse_args()

    num_flags_set = sum(
        [
            args.download_checkpoints,
            args.download_runs,
            args.upload_checkpoints,
            args.upload_runs,
        ]
    )
    if num_flags_set != 1:
        raise ValueError(
            "Must set one of the following flags:\n  "
            "--download_checkpoints, --download_runs, --upload_checkpoints, --upload_runs"
        )

    if args.download_runs:
        download_runs(args, checkpoints_only=False)
    if args.download_checkpoints:
        download_runs(args, checkpoints_only=True)
    if args.upload_runs:
        upload_runs(args, checkpoints_only=False)
    if args.upload_checkpoints:
        # Getting 403 errror with upload_checkpoints(), use upload_runs(checkpoint_only=True) for now
        # upload_checkpoints(args)
        upload_runs(args, checkpoints_only=True)
