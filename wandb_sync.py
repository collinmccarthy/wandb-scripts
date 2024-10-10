"""
Sync a checkpoint file from a Wandb project to a local directory.

Author:
Collin McCarthy
https://github.com/collinmccarthy/wandb-scripts

Examples:
- Upload checkpoints
    ```
    python sync_runs.py \
        --wandb-entity=$WANDB_ENTITY \
        --wandb-project=$WANDB_PROJECT \
        --results-dir=<PARENT_DIR_FOR_ALL_RUNS> \
        --checkpoint-name checkpoint-best.pth \
        --upload-checkpoints
    ```

- Download checkpoints
    ```
    python sync_runs.py \
        --wandb-entity=$WANDB_ENTITY \
        --wandb-project=$WANDB_PROJECT \
        --results-dir=<PARENT_DIR_FOR_ALL_RUNS> \
        --checkpoint-name checkpoint-best.pth \
        --download-checkpoints
    ```

- Upload (sync) runs
    - Files previously uploaded (added as symlinks in wandb dir) will be uploaded if symlink exists
    - To upload a new checkpoint, add --checkpoint_name as in uploading checkpoints above
    ```
    python sync_runs.py \
        --wandb-entity=$WANDB_ENTITY \
        --wandb-project=$WANDB_PROJECT \
        --results-dir=<PARENT_DIR_FOR_ALL_RUNS> \
        --run-names <RUN_NAME_ONE> <RUN_NAME_TWO> \
        --upload-runs \
    ```

- Upload (sync) runs in a more complicated (typical) scenario:
    - The wandb dir is stored within a 'wandb_vis' folder: use --wandb-local-prefix-path
    - There may be duplicates, force run names to match run folder: use --force-run-name-match
    - There may be missing symlinks, delete them if there are: use --remove-missing-symlinks
    ```
    python sync_runs.py \
        --wandb-entity=$WANDB_ENTITY \
        --wandb-project=$WANDB_PROJECT \
        --results-dir=<PARENT_DIR_FOR_ALL_RUNS> \
        --run-names <RUN_NAME_ONE> <RUN_NAME_TWO> \
        --upload-runs \
        --wandb-local-prefix-path="wandb_vis" \
        --force-run-name-match \
        --remove-missing-symlinks
    ```

- Download runs
    - Will download all files, don't need --checkpoint_name
    ```
    python sync_runs.py \
        --wandb-entity=$WANDB_ENTITY \
        --wandb-project=$WANDB_PROJECT \
        --results-dir=<PARENT_DIR_FOR_ALL_RUNS> \
        --download-runs
    ```
"""

import sys
import os
import argparse
import subprocess
import re
import warnings
import pprint
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
        type=str,
        default=os.environ.get("WANDB_ENTITY", None),
        help="Wandb entity name (e.g. username or team name)",
    )
    parser.add_argument(
        "--wandb-project",
        "--wandb_project",
        type=str,
        required=True,
        help="Wandb project name",
    )

    # Must pass in either results-dir OR run-dirs
    parser.add_argument(
        "--results-dir",
        "--results_dir",
        type=str,
        help="Path to results dir, parent directory containing all run folders.",
    )
    parser.add_argument(
        "--run-names",
        "--run_names",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Filter project runs to only sync run ids corresponding to these run names"
            " (if not provided, will sync all)"
        ),
    )
    parser.add_argument(
        "--run-dirs",
        "--run_dirs",
        type=str,
        nargs="+",
        default=None,
        help="Filter project runs to only sync these run dirs",
    )
    parser.add_argument(
        "--force-run-name-match",
        "--force_run_name_match",
        action="store_true",
        help=(
            "Only sync run directories whose folder names match one of values in --run-names."
            " This may be necessary if duplicate run folders are found."
        ),
    )
    parser.add_argument(
        "--remove-missing-symlinks",
        "--remove_missing_symlinks",
        action="store_true",
        help="Remove missing symlinks when uploading runs, to prevent run syncing from failing.",
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
        type=str,
        default=None,
        help=(
            "Checkpoint name for uploading/downloading checkpoints, or forcing checkpoint upload"
            " during sync via (--upload-runs)"
        ),
    )

    # General
    parser.add_argument(
        "--wandb-local-prefix-path",
        "--wandb_local_prefix_path",
        type=str,
        help=(
            "Additional prefix path for wandb folder. E.g. if within the run directory the wandb"
            " folder is 'wandb_vis/wandb' then this should be 'wandb_vis'."
        ),
    )
    parser.add_argument(
        "--dry-run",
        "--dry_run",
        action="store_true",
        default=False,
        help="Print the commands but do not run them",
    )

    parser.add_argument(
        "--overwrite-existing",
        "--overwrite_existing",
        action="store_true",
        default=False,
        help="Overwrite existing files (if False, will skip)",
    )

    args = parser.parse_args()

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
            "Must specify exactly one of:\n  "
            "--download_checkpoints, --download_runs, --upload_checkpoints, --upload_runs"
        )

    if sum([val is not None for val in (args.results_dir, args.run_dirs)]) != 1:
        raise ValueError(f"Must specify exactly one of --results-dir, --run-dirs")

    if args.wandb_entity is None or len(args.wandb_entity) == 0:
        raise RuntimeError(f"Missing --wandb-entity (default: $WANDB_ENTITY env var)")

    if args.wandb_project is None or len(args.wandb_project) == 0:
        raise RuntimeError(f"Missing --wandb-project")

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
            continue
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

    if args.run_dirs is not None:
        all_run_dirs = [Path(run_dir) for run_dir in args.run_dirs]
    else:
        parent_dir = args.results_dir
        all_run_dirs = [f for f in Path(parent_dir).iterdir() if f.is_dir()]

    # Get a mapping from run name to run id (from the runs we have uploaded already)
    runs, _project_str = get_matching_runs(args)
    run_name_to_id = {run.name: run.id for run in runs}
    run_ids = [run.id for run in runs]

    if args.checkpoint_name is None:
        warnings.warn(
            f"Not creating symbolic link for any new checkpoints, will sync runs using"
            f" existing wandb file symlinks. To upload new checkpoints during sync, specify"
            f" --checkpoint-name <name>."
        )

    if args.wandb_local_prefix_path is not None:
        if "/" in args.wandb_local_prefix_path:
            wandb_subdir_paths = args.wandb_local_prefix_path.split("/") + ["wandb"]
        elif r"\\" in args.wandb_local_prefix_path:
            wandb_subdir_paths = args.wandb_local_prefix_path.split(r"\\") + ["wandb"]
        else:
            wandb_subdir_paths = [args.wandb_local_prefix_path] + ["wandb"]
    else:
        wandb_subdir_paths = ["wandb"]

    # Iterate over all folders and check the run ids for the individual run-* folders
    # This is more reliable than assuming the folder name matches the run name
    all_wandb_run_folders: List[List[Path]] = []
    all_wandb_dirs: List[Path] = []
    for run_dir in all_run_dirs:

        wandb_dir: Path = run_dir.joinpath(*wandb_subdir_paths)
        if not wandb_dir.exists():
            print(
                f"Missing 'wandb' folder in directory {run_dir}, skipping directory. If this is a"
                f" subfolder containing more runs, re-run this script with --results_dir=<dir>"
                f" to sync runs within this directory."
            )
            continue

        wandb_run_folders: List[Path] = []
        for run_folder in [
            f
            for f in Path(wandb_dir).iterdir()
            if f.is_dir() and f.name.startswith("run-")
        ]:
            run_id = run_folder.name.split("-")[-1]
            if run_id in run_ids:
                wandb_run_folders.append(run_folder)

        if len(wandb_run_folders) > 0:
            if (
                args.force_run_name_match
                and args.run_names is not None
                and run_dir.name not in args.run_names
            ):
                print(
                    f"Skipping directory {run_dir.name}, because this is not in specified run names"
                    f" and --force-run-name-match is present."
                )
                continue

            # Sort the run folders by name which will sync from oldest to newest
            # This shouldn't matter but it's possible it could
            wandb_run_folders = sorted(wandb_run_folders, key=lambda f: f.name)
            all_wandb_run_folders.append(wandb_run_folders)
            all_wandb_dirs.append(wandb_dir)

    # Verify all run folders are unique, and if they're not, tell the user to pass in a different
    #   path for --results-dir that has unique runs for this run id
    for idx, (wandb_dir, wandb_run_folders) in enumerate(
        zip(all_wandb_dirs, all_wandb_run_folders)
    ):
        current_folder_strs = [f.name for f in wandb_run_folders]
        other_folder_strs = [
            f.name
            for other_idx, other_run_folders in enumerate(all_wandb_run_folders)
            for f in other_run_folders
            if other_idx != idx
        ]
        duplicate_run_folders = set(current_folder_strs).intersection(other_folder_strs)
        if len(duplicate_run_folders) > 0:
            duplicate_run_dirs = [
                str(f)
                for wandb_run_folders in all_wandb_run_folders
                for f in wandb_run_folders
                if f.name in duplicate_run_folders
            ]
            raise RuntimeError(
                f"Found duplicate run folder names:\n{pprint.pformat(duplicate_run_folders)}."
                f"\nAll run folders with these names:\n{pprint.pformat(duplicate_run_dirs)}."
                f"\nThis means the folder was copied at some point, and syncing duplicates could"
                f" could lead to issues if the contents of the directories are not identical. Pass"
                f" in either --force-run-name-match to only use run directories whose folder names"
                f" match the corresponding --run-names, or use --run-dirs instead to sync just the"
                f" run dirs that don't contain duplicates."
            )

    # Clean up any old symlinks
    missing_symlinks: list[Path] = []
    for wandb_run_folders in all_wandb_run_folders:
        for wandb_run_folder in wandb_run_folders:
            files = Path(wandb_run_folder, "files")
            for f in files.iterdir():
                if f.is_symlink() and not f.resolve().exists():
                    missing_symlinks.append(f)

    if len(missing_symlinks) > 0:
        if not args.remove_missing_symlinks:
            raise RuntimeError(
                f"Found {len(missing_symlinks)} missing symlinks. Cannot sync runs with missing"
                f" symlinks or errors will be raised. Pass in --remove_missing_symlinks to remove these"
                f" missing symlinks, or manually delete them to sync the current runs"
            )
        for f in missing_symlinks:
            f.unlink()

    commands: List[List[str]] = []
    for wandb_dir, wandb_run_folders in zip(all_wandb_dirs, all_wandb_run_folders):

        # Add a symlink from requested checkpoint to latest run dir so it gets uploaded
        #   when syncing latest run (which is one of the "run-*" folders)
        if args.checkpoint_name is not None:
            wandb_run_dir = wandb_dir.joinpath("latest-run").resolve()
            latest_run_id = wandb_run_dir.name.split("-")[-1]
            if latest_run_id not in run_ids:
                raise RuntimeError(
                    f"Latest run has run id {latest_run_id} which does not match any run names"
                    f": {list(run_name_to_id.keys())}"
                )

            print(
                f"Creating symbolic link for checkpoing {args.checkpoint_name} to upload file during sync"
            )

            # Get the checkpoint we want to upload
            matching_checkpoints = [
                f for f in wandb_dir.iterdir() if f.name == args.checkpoint_name
            ]
            if len(matching_checkpoints) != 1:
                print(
                    f"Failed to find checkpoint {args.checkpoint_name} in directory {wandb_dir}."
                    f" Skipping directory."
                )
                continue

            # Create symbolic link from wandb/<run>/files/<FILENAME> to checkpoint we want to upload
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

        for run_folder in wandb_run_folders:
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

            command.append(str(run_folder))
            commands.append(command)

    if args.dry_run:
        commands_list = [" ".join(command) for command in commands]
        commands_str = "\n".join(commands_list)
        print("- " * 80)
        print("Commands to be run (dry-run only)")
        print("- " * 80)
        print(f"\n{commands_str}")
    else:
        for command in tqdm(commands, desc="Dir"):
            command_str = " ".join(command)
            print(f"Running: {command_str}")

            # Using subprocess.run(command, check=True) gives "no such file or dir" error
            # Using subprocess.run(command_str, check=True) gives "no such file or dir" error too
            # Using subprocess.run(command, shell=True, check=True) gives usage / argparse error
            # Using subprocess.run(command_str, shell=True, check=True) works
            subprocess.run(command_str, shell=True, check=True)


if __name__ == "__main__":
    args = parse_args()

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
