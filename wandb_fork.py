"""
Fork a Wandb run independently from training the model.

This is mostly a sandbox to test/verify run forking works as expected. For context surrounding this
test see https://github.com/wandb/wandb/issues/8353

Author:
Collin McCarthy
https://github.com/collinmccarthy/wandb-scripts

Examples:
- Fork using run id, at step=15000
    ```
    python wandb_fork.py \
    --wandb-entity=$WANDB_ENTITY \
    --wandb-project=$WANDB_PROJECT \
    --run-id=<PREV_RUN_ID> \
    --forked-value=15000 \
    --forked-run-dir=<NEW_RUN_DIR> \
    --forked-run-name=<NEW_RUN_NAME> \
    ```
- Fork using run name, at step=15000
    ```
    python wandb_fork.py \
    --wandb-entity=$WANDB_ENTITY \
    --wandb-project=$WANDB_PROJECT \
    --run-name=<PREV_RUN_NAME> \
    --forked-value=15000 \
    --forked-run-dir=<NEW_RUN_DIR> \
    --forked-run-name=<NEW_RUN_NAME> \
    ```
- Fork using run name, at first step of epoch=30
    ```
    python wandb_fork.py \
    --wandb-entity=$WANDB_ENTITY \
    --wandb-project=$WANDB_PROJECT \
    --run-name=<PREV_RUN_NAME> \
    --forked-value=30 \
    --forked-metric-name="epoch" \
    --matching-steps-reduction-func="min" \
    --forked-run-dir=<NEW_RUN_DIR> \
    --forked-run-name=<NEW_RUN_NAME> \
    ```
- Fork using run name, at last step of epoch=29
    ```
    python wandb_fork.py \
    --wandb-entity=$WANDB_ENTITY \
    --wandb-project=$WANDB_PROJECT \
    --run-name=<PREV_RUN_NAME> \
    --forked-value=29 \
    --forked-metric-name="epoch" \
    --matching-steps-reduction-func="max" \
    --forked-run-dir=<NEW_RUN_DIR> \
    --forked-run-name=<NEW_RUN_NAME> \
    ```
"""

import os
import argparse
import re
import pprint
import warnings
from argparse import Namespace
from pathlib import Path
from tqdm import tqdm

import wandb
from wandb.apis.public.runs import Run as ApiRun
from wandb.apis.public.runs import Runs as ApiRuns


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
        required=True,
        help="Wandb project name",
    )
    parser.add_argument(
        "--forked-value",
        "--forked_value",
        type=int,
        required=True,
        help="Value to fork at. Default is 'step', but could be 'epochs', etc. and we'll convert.",
    )
    parser.add_argument(
        "--forked-metric-name",
        "--forked_metric_name",
        type=str,
        default="step",
        help="Step to fork at.",
    )
    parser.add_argument(
        "--matching-steps-reduction-func",
        "--matching_steps_reduction_func",
        type=str,
        default=None,
        choices=("min", "max"),
        help="Function to use if multiple steps match our metric.",
    )
    parser.add_argument(
        "--forked-run-dir",
        "--forked_run_dir",
        type=str,
        required=True,
        help="Directory for forked run",
    )
    parser.add_argument(
        "--forked-run-name",
        "--forked_run_name",
        type=str,
        help="Name for forked run. If not specified will use folder name of --forked-run-dir.",
    )
    parser.add_argument(
        "--run-id",
        "--run_id",
        type=str,
        help="Run id to fork",
    )
    parser.add_argument(
        "--run-name",
        "--run_name",
        type=str,
        help="Run name to fork. If name is not unique an error is thrown.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="auto",
        help="Resume flag to use during init. Default: 'auto' to create `wandb-resume.json`.",
    )
    parser.add_argument(
        "--no-resume",
        "--no_resume",
        action="store_true",
        help="Do not pass in resume flag to wandb.init()",
    )

    args = parser.parse_args()

    if not sum([val is not None for val in (args.run_id, args.run_name)]) == 1:
        raise RuntimeError(f"Must specify exactly one of --run-id and --run-name")

    if args.wandb_entity is None or len(args.wandb_entity) == 0:
        raise RuntimeError(f"Missing --wandb-entity (default: $WANDB_ENTITY env var)")

    if args.wandb_project is None or len(args.wandb_project) == 0:
        raise RuntimeError(f"Missing --wandb-project")

    return args


def fork_run(args: Namespace):
    print("-" * 80)

    api = wandb.Api()
    print(f"Querying Wandb entity: {args.wandb_entity}, project: {args.wandb_project}")
    runs: ApiRuns = api.runs(f"{args.wandb_entity}/{args.wandb_project}")

    run: ApiRun
    if args.run_id is not None:
        print(f"Searching for run with run.id={args.run_id}")
        matching_runs = [run for run in tqdm(runs, desc="Run") if run.id == args.run_id]
        if len(matching_runs) == 1:
            run = matching_runs[0]
        else:
            raise RuntimeError(  # Should only happen if run doesn't exist (e.g. deleted)
                f"Found {len(matching_runs)} runs with run.id == {args.run_id}."
                f" expected one. Verify entity and project are correct, and run id exists."
            )

    elif args.run_name is not None:
        print(f"Searching for run with run.name={args.run_name}")
        matching_runs = [
            run for run in tqdm(runs, desc="Run") if run.name == args.run_name
        ]
        if len(matching_runs) == 1:
            run = matching_runs[0]
        else:
            raise RuntimeError(  # Can easily have duplicate run names
                f"Found {len(matching_runs)} runs with run.name == {args.run_name},"
                f" expected one. Verify entity and project are correct, and if so, pass in --run-id"
                f" instead to guarantee a unique match."
            )

    else:
        assert False

    # Get the latest step corresponding to our metric, if metric is not "step"
    step_reduction = False
    if args.forked_metric_name == "step":
        step = args.forked_value
    else:
        steps = []
        for row in run.scan_history():
            if (
                args.forked_metric_name in row
                and row[args.forked_metric_name] == args.forked_value
            ):
                steps.append(row["_step"])

        if len(steps) == 1:
            step = steps[0]
        elif len(steps) > 1:
            step_reduction = True
            if args.matching_steps_reduction_func == "max":
                step = max(steps)
            elif args.matching_steps_reduction_func == "min":
                step = min(steps)
            else:
                raise RuntimeError(
                    f"Found {len(steps)} steps with {args.forked_metric_name}={args.forked_value}."
                    f" Must specify --matching-steps-reduction-func (with value 'min' or 'max')"
                    f" to choose which step to use."
                )

            print(
                f"Found {len(steps)} steps with {args.forked_metric_name}={args.forked_value}."
                f" Using {args.matching_steps_reduction_func}(step)={step} for step to fork."
            )
        else:  # len(steps) == 0:
            raise RuntimeError(
                f"Failed to find steps corresponding to metric {args.forked_metric_name}"
            )

    # Minimal init just to fork the run and create wandb-resume.json (by default)
    # See docs at https://docs.wandb.ai/guides/runs/forking
    # And any response to our question at https://github.com/wandb/wandb/issues/8353
    forked_run_dir = Path(args.forked_run_dir).expanduser().resolve()

    if not forked_run_dir.exists():
        if args.create_forked_run_dir:
            forked_run_dir.mkdir(parents=True)
            warnings.warn(
                f"Creating forked run directory {forked_run_dir} with no files from previous run"
                f" directory. You will need to manually copy any files over to this directory for"
                f" it to be self-contained with any logs and checkpoints from the original"
                f" directory."
            )
        else:
            raise RuntimeError(
                f"Forked run directory {forked_run_dir} does not exist. You should manually copy"
                f" any files over to this directory before forking into it, otherwise previous logs"
                f" and checkpoints will not be updated when resuming from this directory. To create"
                f" this directory from scratch pass in --create-forked-run-dir"
            )

    forked_run_name = (
        args.forked_run_name
        if args.forked_run_name is not None
        else forked_run_dir.name
    )

    init_kwargs = dict(
        project=args.wandb_project,
        entity=args.wandb_entity,
        fork_from=f"{run.id}?_step={step}",
        name=forked_run_name,
        dir=str(forked_run_dir),
    )

    print("- " * 40)
    print(
        f"Forking run:"
        f"\n  Orig name: {run.name} (id: {run.id})"
        f"\n  New name: {forked_run_name}"
        f"\n  New dir: {forked_run_dir}"
        f"\n  Forked metric name: {args.forked_metric_name}"
        f"\n  Forked metric value: {args.forked_value}"
    )

    if step_reduction:
        print(f"  Matching steps reduction func: {args.matching_steps_reduction_func}")
        print(f"  Corresponding step: {step}")

    print("- " * 40)
    print(f"Wandb init kwargs:\n{pprint.pformat(init_kwargs, sort_dicts=False)}")

    print("- " * 40)
    response = input("Continue? (y/N): ")
    if response.lower() not in ["y", "yes"]:
        print(f"Response={response}. Skipping fork run and exiting.")
        exit(0)

    forked_run = wandb.init(**init_kwargs)

    print("- " * 40)
    print(f"Forked run info:" f"\n  Name: {forked_run.name}" f"\n  ID: {forked_run.id}")

    # Verify resuming and/or create wandb-resume.json
    if not args.no_resume:
        print(
            f"Closing run to re-init with resume flag (cannot fork and resume at same time)"
        )
        forked_run.finish(exit_code=0, quiet=True)

        resume_kwargs = dict(**init_kwargs)
        resume_kwargs.pop("fork_from")
        resume_kwargs["resume"] = args.resume

        print("- " * 40)
        print(
            f"Re-initializing run to set resume flag, using kwargs:"
            f"\n{pprint.pformat(resume_kwargs, sort_dicts=False)}"
        )

        print("- " * 40)
        response = input("Continue? (y/N): ")
        if response.lower() not in ["y", "yes"]:
            print(f"Response={response}. Skipping fork run and exiting.")
            exit(0)

        resumed_run = wandb.init(**resume_kwargs)

        print("- " * 40)
        print(f"Run resumed successfully. Closing and exiting.")

        resumed_run.finish(exit_code=0, quiet=True)
        print("-" * 80)


if __name__ == "__main__":
    args = parse_args()
    fork_run(args)
