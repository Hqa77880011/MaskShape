# MaskShape

Implementation of **Mask-guided shape-texture perception under background shift**.

## Installation

Requires Python `>=3.11`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Offline check: `python -m maskshape.smoke --output runs/smoke` uses generated pairs to check nine methods, gradients, checkpoint reload, metrics, evaluation and visualizations.

For the Windows Anaconda/PyTorch environment, set per-process `MKL_THREADING_LAYER=SEQUENTIAL`, `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`.

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

Select a subset with `--methods`: `maskshape`, `image_only`, `shape_only`, `no_consistency`, `binary_mask`, `mask_crop`, `masked_image`, `mask_channel`, `concatenation`.

## Outputs

`runs/main/maskshape/model.pt` stores the validation-selected weights, method and configuration; `history.jsonl` records measured epoch losses and time. Evaluation writes `predictions.npz` and `metrics.json` under `results/maskshape`.

```bash
python -m maskshape plot --inputs runs/main/maskshape results/maskshape --output figures
```

`figures/` contains loss curves and metric plots from the supplied run directories.

## Components and metrics

| Component / metric | Code entry | Setting / command |
|---|---|---|
| ConvNeXt-Tiny + ResNet-18 shape stream | `core.Model`, `representation` | `train`; binary region, one-pixel inside contour, signed Euclidean distance normalized per mask |
| Projected residual fusion | `core.Model.forward` | pooled 768/512 features project to 256 dimensions, addition, LayerNorm and classifier |
| Inverse-frequency CE + paired JS | `core.loss`, `cli.train` | weights estimated on train labels only, mean-one normalization; JS coefficient .1 |
| Full-image, tight box crop, image-mask product, four-channel mask input | `core.METHODS`, `mask_crop` | `suite --methods image_only mask_crop masked_image mask_channel` |
| Shape/fusion/consistency ablations | `core.METHODS` | `suite --methods shape_only concatenation no_consistency binary_mask` |
| Macro F1, balanced accuracy, accuracy, NLL, ECE, top-3 | `vision.classification_metrics` | `evaluate`; F1 and BAcc average over all configured classes, including absent strata classes as zero |
| Background/domain strata and paired stability | `core.metrics` | numeric `background`/`domain`, `x_pair`; mean per-class probability variance and paired prediction agreement |
| Spatial evidence and training curves | `activation`, `cli.plot` | commands below |

`prepare` requires source `group` IDs. For unseen-background tests, supply an instance-grouped `split` manifest and optionally `--heldout-backgrounds 3 4`; preparation rejects any listed background in training/validation. Keep real-fab test data separate from training and model selection. When masks are empty, the crop baseline uses the full image. The mask-channel baseline uses a four-channel ConvNeXt stem; image-only baselines do not execute or count a shape encoder. All models use random initialization.

```bash
python -m maskshape.activation --checkpoint runs/main/maskshape/model.pt --data data/prepared/test.npz --index 0 --output figures/activation.png
python -m maskshape.smoke --output runs/smoke
```

Grad-CAM displays predicted-class gradients through the appearance stream. Display scaling affects visualization only. Use an explicit grouped `split` manifest to preserve registered evaluation partitions.
