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

import os
import argparse
import re
import pprint
from argparse import Namespace
from datetime import datetime
from collections import namedtuple
from typing import Union, Sequence

import wandb
from wandb.apis.public.runs import Run, Runs
from wandb.apis.public.files import File, Files
from wandb.apis.public.artifacts import RunArtifacts
from wandb.sdk.artifacts.artifact import Artifact
from tqdm import tqdm


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Clean up Wandb runs remotely")
    parser.add_argument(
        "--wandb-entity",
        "--wandb_entity",
        default=os.environ.get("WANDB_ENTITY", None),
        help="Wandb entity name (e.g. username or team name)",
    )
    parser.add_argument(
        "--wandb-project",
        "--wandb_project",
        default=os.environ.get("WANDB_PROJECT", None),
        help="Wandb project name",
    )
    parser.add_argument("--username", help="Username to filter / select runs")
    parser.add_argument(
        "--delete-filename-regex",
        "--delete_filename_regex",
        nargs="+",
        default=list(),
        help="Delete all files in all runs in the project that match this regular expression",
    )
    parser.add_argument(
        "--delete-keep-latest",
        "--delete_keep_latest",
        action="store_true",
        help="If delete_filename_regex or delete_artifact_regex are used, keep the objects with"
        " the last modified time",
    )
    parser.add_argument(
        "--delete-artifact-regex",
        "--delete_artifact_regex",
        nargs="+",
        default=list(),
        help="Delete all artifacts in all runs in the project that match this regular expression",
    )
    parser.add_argument(
        "--find-filename-regex",
        "--find_filename_regex",
        nargs="+",
        default=list(),
        help="Print all files in all runs in the project that match this regular expression",
    )
    parser.add_argument(
        "--find-artifact-regex",
        "--find_artifact_regex",
        nargs="+",
        default=list(),
        help="Print all artifacts in all runs in the project that match this regular expression",
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
    parser.add_argument(
        "--dry-run",
        "--dry_run",
        action="store_true",
        help="Print all files that will be deleted but do not delete them",
    )

    args = parser.parse_args()

    if args.wandb_entity is None:
        raise RuntimeError(f"Missing --wandb-entity (default: $WANDB_ENTITY)")

    if args.wandb_project is None:
        raise RuntimeError(f"Missing --wandb-project (default: $WANDB_PROJECT)")

    if len(args.wandb_entity) == 0:
        raise RuntimeError(f"Found empty string for --wandb-entity")

    if len(args.wandb_project) == 0:
        raise RuntimeError(f"Found empty string for --wandb-project")

    return args


def remove_latest_file_or_artifact(
    delete_objs: Union[Sequence[File], Sequence[Artifact]]
) -> Union[Sequence[File], Sequence[Artifact]]:
    if len(delete_objs) == 1:
        last_mod_file = delete_objs[0]
    else:
        DatetimeTup = namedtuple("DatetimeTup", ["file", "datetime"])
        datetimes = []
        for f in delete_objs:
            if isinstance(f, File):
                updated_attr = "updatedAt"
            elif isinstance(f, Artifact):
                updated_attr = "updated_at"
            else:
                assert False, "Expected File or Artifact"

            dt = DatetimeTup(
                file=f,
                datetime=datetime.strptime(
                    getattr(f, updated_attr),
                    "%Y-%m-%dT%H:%M:%SZ",  # For version 0.18.6; throws exception if incorrect
                ),
            )
            datetimes.append(dt)

        datetimes = sorted(datetimes, key=lambda x: x.datetime, reverse=True)
        last_mod_file = datetimes[0].file

    print(f"Removing last modified file from deletion list: {last_mod_file.name}")
    num_delete_prev = len(delete_objs)
    delete_objs = [
        f for f in delete_objs if f != last_mod_file
    ]  # pyright: ignore[reportAssignmentType]
    assert len(delete_objs) == num_delete_prev - 1
    return delete_objs


def filter_runs(args: Namespace) -> Sequence[Run]:
    api = wandb.Api()
    print(f"Querying Wandb entity: {args.wandb_entity}, project: {args.wandb_project}")
    runs: Runs = api.runs(f"{args.wandb_entity}/{args.wandb_project}")

    if len(args.delete_filename_regex) > 0:
        print(
            f"Searching {len(runs)} runs for filenames matching {args.delete_filename_regex}"
        )
    if len(args.delete_artifact_regex) > 0:
        print(
            f"Searching {len(runs)} runs for artifacts matching {args.delete_artifact_regex}"
        )

    current_runs: Sequence[Run] = []
    for run in runs:
        run: Run
        if len(args.skip_run_ids) > 0 and run.id in args.skip_run_ids:
            # print(f"Skipping run id {run.id} (found in --skip-run-ids)")
            continue

        if len(args.select_run_ids) > 0 and run.id not in args.select_run_ids:
            # print(f"Skipping run id {run.id} (not found in --select-run-ids)")
            continue

        if len(args.skip_run_names_regex) > 0 and any(
            [re.search(regex, run.name) for regex in args.skip_run_names_regex]
        ):
            # print(f"Skipping run name {run.name} (matched with --skip-run-names-regex)")
            continue

        if len(args.select_run_names_regex) > 0 and not any(
            [re.search(regex, run.name) for regex in args.select_run_names_regex]
        ):
            # print(f"Skipping run name {run.name} (not matched with --select-run-names-regex)")
            continue

        if args.username is not None and run.user.username != args.username:
            continue

        current_runs.append(run)

    return current_runs


def find_objects(
    run: Run,
    objects: Union[Files, RunArtifacts],
    regex: str,
    remove_latest: bool,
    verbose: bool = True,
) -> Union[Sequence[File], Sequence[Artifact]]:
    found_objs: Union[Sequence[File], Sequence[Artifact]] = [
        f for f in objects if re.search(regex, f.name)
    ]
    if len(found_objs) > 0:
        found_filenames = [f.name for f in found_objs]
        found_filenames_str = "\n  " + "\n  ".join(sorted(found_filenames))
        if verbose:
            print(
                f"Found {len(found_objs)} matching objects from {run.name}:"
                f"{found_filenames_str}"
            )

        if remove_latest:
            found_objs = remove_latest_file_or_artifact(found_objs)

    return found_objs


def delete_objects(args: Namespace):
    runs = filter_runs(args)

    print(
        f"Using delete_filename_regex={args.delete_filename_regex},"
        f" delete_artifact_regex={args.delete_artifact_regex}"
    )

    all_delete_files = []
    all_delete_artifacts = []
    for run in tqdm(runs, desc="Run"):

        for regex in args.delete_filename_regex:
            files: Files = run.files()
            delete_files: Sequence[File] = find_objects(
                run=run,
                objects=files,
                regex=regex,
                remove_latest=args.delete_keep_latest,
            )  # pyright: ignore[reportAssignmentType]
            all_delete_files.extend(delete_files)

        for regex in args.delete_artifact_regex:
            artifacts = run.logged_artifacts()
            delete_artifacts: Sequence[File] = find_objects(
                run=run,
                objects=artifacts,
                regex=regex,
                remove_latest=args.delete_keep_latest,
            )  # pyright: ignore[reportAssignmentType]
            all_delete_artifacts.extend(delete_artifacts)

    if len(all_delete_files + all_delete_artifacts) == 0:
        print(f"Found no matching files or artifacts to delete")
    else:
        all_obj_strs = []
        all_obj_strs.extend([f.url for f in all_delete_files])
        all_obj_strs.extend([f.qualified_name for f in all_delete_artifacts])
        print("-" * 80)
        print(
            f"Found {len(all_obj_strs)} files / artifacts:"
            f"\n{pprint.pformat(all_obj_strs)}"
        )

        if not args.dry_run:
            response = input(
                f"Delete {len(all_delete_files + all_delete_artifacts)} files/artifacts? (y/N): "
            )
            if response.lower() == "y":
                print(f"Deleting files (response=y)")
                if len(all_delete_files) > 0:  # For tqdm output
                    for f in tqdm(all_delete_files, desc="Files"):
                        f.delete()
                if len(all_delete_artifacts) > 0:  # For tqdm output
                    for f in tqdm(all_delete_artifacts, desc="Artifacts"):
                        f.delete(delete_aliases=True)
            else:
                print(f"Skipping deletion (response=n)")


def find_all_objects(args: Namespace):
    runs = filter_runs(args)

    print(
        f"Using find_filename_regex={args.find_filename_regex},"
        f" find_artifact_regex={args.find_artifact_regex}"
    )

    all_found_files: list[File] = []
    all_found_artifacts: list[Artifact] = []
    for run in tqdm(runs, desc="Run"):

        for regex in args.find_filename_regex:
            files: Files = run.files()
            found_files: Sequence[File] = find_objects(
                run=run,
                objects=files,
                regex=regex,
                remove_latest=False,
                verbose=False,
            )  # pyright: ignore[reportAssignmentType]
            all_found_files.extend(found_files)

        for regex in args.find_artifact_regex:
            artifacts = run.logged_artifacts()
            found_artifacts: Sequence[Artifact] = find_objects(
                run=run,
                objects=artifacts,
                regex=regex,
                remove_latest=False,
                verbose=False,
            )  # pyright: ignore[reportAssignmentType]
            all_found_artifacts.extend(found_artifacts)

    if len(all_found_files + all_found_artifacts) == 0:
        print(f"Found no matching files or artifacts")
    else:
        all_obj_strs = []
        all_obj_strs.extend([f.url for f in all_found_files])
        all_obj_strs.extend([f.qualified_name for f in all_found_artifacts])
        print("-" * 80)
        print(
            f"Found {len(all_obj_strs)} files / artifacts:"
            f"\n{pprint.pformat(all_obj_strs)}"
        )

        unique_names = list(set(f.name for f in all_found_files + all_found_artifacts))
        print("-" * 80)
        print(
            f"Found {len(unique_names)} unique files / artifacts:"
            f"\n{pprint.pformat(unique_names)}"
        )


if __name__ == "__main__":
    args = parse_args()
    if len(args.find_filename_regex) > 0 or len(args.find_artifact_regex) > 0:
        find_all_objects(args)
    if len(args.delete_filename_regex) > 0 or len(args.delete_artifact_regex) > 0:
        delete_objects(args)
