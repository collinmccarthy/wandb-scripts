# Wandb Scripts
Collection of Wandb scripts. Contributions welcome!

## Requirements

- Pip requirements: [requirements.txt](./requirements.txt)
    - Only `tqdm` and `wandb` currently

```bash
pip install -r requirements.txt
```

## Merge Runs

- Source: [wandb_merge.py](./wandb_merge.py)
- Merge runs that were accidentally split into a *new run*

Notes:
- Original run will not be modified except by adding a tag (see `--tag_partial_runs`, default `partial-run`)
- If using mmdetection, `$SAVE_DIR` defaults to the original save dir (`work_dir` in mmdetection)
- Examples use `--verify-overlap-metric=iter` which assumes `iter` has unique values for all runs
    - May need to change to something like `epoch` if iter has duplicate values in multiple runs

Examples:
- Merge runs with names `my_run` and `my_other_run`
    - If all runs share the same name, only need to specify a single run name

```bash
# Set WANDB_ENTITY, WANDB_PROJECT, SAVE_DIR
python tools/wandb/wandb_merge.py \
--wandb-entity=$WANDB_ENTITY \
--wandb-project=$WANDB_PROJECT \
--merge_run_save_dir=$SAVE_DIR \
--verify_overlap_metric iter \
--run-names my_run my_other_run
```

- Merge runs with run ids `1234` and `5678`

```bash
# Specify $WANDB_ENTITY, $WANDB_PROJECT, $SAVE_DIR
python tools/wandb/wandb_merge.py \
--wandb-entity=$WANDB_ENTITY \
--wandb-project=$WANDB_PROJECT \
--merge_run_save_dir=$SAVE_DIR \
--verify_overlap_metric iter \
--run-ids 1234 5678
```

## Delete Files / Artifacts

- Source: [wandb_update.py](./wandb_update.py)
- Delete files and/or artifacts from multiple runs

Examples:
- Delete all checkpoints

```bash
python tools/wandb/wandb_update.py \
--wandb-entity=$WANDB_ENTITY \
--wandb-project=$WANDB_PROJECT \
--delete_filename_regex=".*\.pth"
```

- Delete all bbox and segm checkpoints in runs matching 'r50_.\*\_coco_.*'

```bash
python tools/wandb/wandb_update.py \
--wandb-entity=$WANDB_ENTITY \
--wandb-project=$WANDB_PROJECT \
--delete_filename_regex '.*_segm_.*\.pth' '.*_bbox_.*\.pth' \
--select-run-names-regex 'r50_.*_coco.*'
```

- Delete all files in folder 'predictions/' for all runs except runs matching 'r50_.\*\_coco_.*'

```bash
python tools/wandb/wandb_update.py \
--wandb-entity=$WANDB_ENTITY \
--wandb-project=$WANDB_PROJECT \
--delete_filename_regex="^predictions/.*" \
--skip-run-names-regex 'r50_.*_coco.*'
```

- Delete all artifacts ending in '_pred_final' for all runs except runs matching 'r50_.\*\_coco_.*'

```bash
python tools/wandb/wandb_update.py \
--wandb-entity=$WANDB_ENTITY \
--wandb-project=$WANDB_PROJECT \
--delete-artifact-regex=".*_pred_final" \
--skip-run-names-regex 'r50_.*_coco.*'
```
