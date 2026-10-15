# MaskShape

Implementation of **Mask-guided shape-texture perception under background shift**.

Uses ConvNeXt-Tiny and ResNet-18 branches, mask/contour/signed-distance shape input, independent learned projections, residual addition with LayerNorm and paired-background JS consistency.

## Install

Use Python 3.11 or newer. Install a matching PyTorch/TorchVision build for the target CPU or CUDA device, then install the package.

```bash
git clone https://github.com/Hqa77880011/MaskShape.git
cd MaskShape
python -m venv .venv
```

Activate with `source .venv/bin/activate` or `.venv\Scripts\Activate.ps1` in PowerShell.

```bash
python -m pip install -e .
```

## Prepare data

Use the image/mask/class records from the SEM defect dataset. Preserve source_defect_id and predefined held-out background IDs; keep real fab images in their independent test stratum.

Fields are `x` float32 [N,3,H,W], `mask` binary [N,1,H,W], `y` integer zero-based class and `group` source defect ID. `x_pair` is the same defect on a different background, used for consistency. Optional numeric `background` and `domain` produce stratified metrics. Normalize RGB channels consistently before packing.

```bash
python -m maskshape prepare --source data/raw.npz --output data/prepared --seed 17
```

Use `group` to keep every derived view of one source object in the same partition. An explicit integer `split` array (0=train, 1=validation, 2=test) preserves a registered split. Otherwise groups are assigned 60/20/20. Every stored field must share the sample axis. Files are loaded without pickle.

## Train

Edit `configs/default.json` for the input dimensions, device and training budget.

```bash
python -m maskshape train --config configs/default.json --data data/prepared --output runs/main
python -m maskshape suite --config configs/default.json --data data/prepared --output runs/comparison
```

`--methods` selects a subset. maskshape, image_only, shape_only, no_consistency and binary_mask.

The common driver uses AdamW with cosine learning-rate decay, validation-objective early stopping and a fixed seed. Custom federated/continual drivers use their registered round/task budgets. It saves only a validation-selected model and measured epoch history. `learning_rate`, `weight_decay`, `batch_size`, `epochs`, `patience` and `device` control optimization. Use separate output roots for independent runs.

## Evaluate and visualize

```bash
python -m maskshape evaluate --checkpoint runs/main/maskshape/model.pt --data data/prepared/test.npz --output results/maskshape
python -m maskshape plot --inputs runs/main/maskshape results/maskshape --output figures
```

`model.pt` contains model parameters, configuration and method identity. `history.jsonl` contains measured training/validation loss and epoch time. Evaluation writes `predictions.npz` and `metrics.json`. Plotting creates loss curves and metric comparisons from supplied run directories.

Macro F1, balanced accuracy, accuracy, Top-3, ECE and NLL; optional background/domain breakdowns and exported fusion embeddings.

## Implementation conventions

Configure class_weights using inverse training frequencies normalized to mean one. Backbones initialize from scratch. The fusion dimension is an explicit 256-dimensional choice; JS weight is fixed to 0.1. The implementation follows the projected residual fusion in the text.


## Pack image records

A JSONL manifest can point to images, masks and NPY feature arrays. Paths are relative to the manifest. All rows use identical keys; group stores the source object ID. For example, an image-classification row is `{"x":"images/1.png","y":0,"group":"object-1"}`.

```bash
python -m maskshape.pack --manifest data/manifest.jsonl --output data/raw.npz --channels 3 --target class
```

Inference timing runs the deployment forward pass separately from calibration, commutation and other evaluator diagnostics, after one warmup. Batch P50/P95 and examples/s retain the configured batch size. MAC/FLOP counts cover Conv/Linear/GRU multiply-adds; process RSS includes the dataset and Python runtime.
