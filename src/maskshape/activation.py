"""Appearance-branch Grad-CAM for one frozen checkpoint and image pair."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from .core import Model
from .cli import Arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    payload = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    model = Model(payload["config"], payload["method"]).eval()
    model.load_state_dict(payload["state"])
    if model.image is None:
        raise ValueError("shape_only has no appearance branch for image Grad-CAM")
    sample = Arrays(args.data)[args.index]
    images = [sample["x"]]
    if "x_pair" in sample:
        images.append(sample["x_pair"])
    activation = {}

    def retain(module, inputs, value):
        activation["value"] = value
        value.retain_grad()

    hook = model.image.features[-1].register_forward_hook(retain)
    fig, axes = plt.subplots(len(images), 2, figsize=(6, 3 * len(images)), squeeze=False)
    try:
        for row, image in enumerate(images):
            model.zero_grad(set_to_none=True)
            logits = model(image[None], sample["mask"][None])[0]
            target = int(logits.argmax(-1))
            logits[0, target].backward()
            feature = activation["value"]
            cam = (feature * feature.grad.mean((-2, -1), keepdim=True)).sum(1, keepdim=True).relu()
            cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
            cam = (cam / cam.max().clamp_min(1e-8)).detach().numpy()
            # Display min/max scaling affects the visualization only.
            display = image.permute(1, 2, 0).numpy()
            display = (display - display.min()) / max(float(display.max() - display.min()), 1e-8)
            axes[row, 0].imshow(display)
            axes[row, 1].imshow(display)
            axes[row, 1].imshow(cam, cmap="viridis", alpha=.55, vmin=0, vmax=1)
            axes[row, 1].set_title(f"Predicted class {target}")
            for axis in axes[row]:
                axis.axis("off")
    finally:
        hook.remove()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
