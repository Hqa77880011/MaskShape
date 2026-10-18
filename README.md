# MaskShape

Implementation of **Mask-guided shape-texture perception under background shift**.

## Installation

Requires Python `>=3.11`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5; numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, torchvision 0.22.1, scikit-learn 1.6.1, matplotlib 3.10.0, pillow 12.3.0. Checks: Python syntax, configuration parsing and CLI/source argument matching. Training and full evaluation were not run.

## Data and preprocessing

Supply image/mask/class records from your licensed SEM defect dataset. Preserve source_defect_id and predefined held-out background IDs; keep real fab images in their independent test stratum.

Fields are `x` float32 [N,3,H,W], `mask` binary [N,1,H,W], `y` integer zero-based class and `group` source defect ID. `x_pair` is the same defect on a different background, used for consistency. Optional numeric `background` and `domain` produce stratified metrics. Normalize RGB channels consistently before packing.

Create `data/manifest.jsonl` with one row per source sample, for example `{"x":"images/1.png","mask":"masks/1.png","x_pair":"images/1_pair.png","y":0,"group":"defect-1"}`; paths resolve relative to the manifest.

```bash
python -m maskshape.pack --manifest data/manifest.jsonl --output data/raw.npz --channels 3 --target class
python -m maskshape prepare --source data/raw.npz --output data/prepared --seed 17
```

Use `group` to keep every derived view of one source object in the same partition. An explicit integer `split` array (0=train, 1=validation, 2=test) preserves a registered split. Otherwise groups are assigned 60/20/20. Every stored field must share the sample axis. Files are loaded without pickle.

## Training and inference

Set input dimensions, device, training budget and `seed` in `configs/default.json` (default seed 17).

```bash
python -m maskshape train --config configs/default.json --data data/prepared --output runs/main
```

## Evaluation

```bash
python -m maskshape evaluate --checkpoint runs/main/maskshape/model.pt --data data/prepared/test.npz --output results/maskshape
```

## Ablations

```bash
python -m maskshape suite --config configs/default.json --data data/prepared --output runs/comparison
```

Select a subset with `--methods`: `maskshape`, `image_only`, `shape_only`, `no_consistency`, `binary_mask`.

## Outputs

`runs/main/maskshape/model.pt` stores the validation-selected weights, method and configuration; `history.jsonl` records measured epoch losses and time. Evaluation writes `predictions.npz` and `metrics.json` under `results/maskshape`.

```bash
python -m maskshape plot --inputs runs/main/maskshape results/maskshape --output figures
```

`figures/` contains loss curves and metric plots from the supplied run directories.
