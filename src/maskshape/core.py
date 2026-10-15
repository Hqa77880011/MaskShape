import numpy as np
import torch
from scipy import ndimage
from torch import nn
from torch.nn import functional as F
from torchvision.models import convnext_tiny, resnet18

from .vision import classification_metrics


METHODS = ["maskshape", "image_only", "shape_only", "no_consistency", "binary_mask"]


def representation(mask, device, binary_only=False):
    output = []
    for value in mask.detach().cpu().numpy():
        m = value.squeeze() > 0.5
        boundary = m & ~ndimage.binary_erosion(m)
        distance = ndimage.distance_transform_edt(m) - ndimage.distance_transform_edt(~m)
        distance /= max(np.abs(distance).max(), 1)
        output.append(np.stack([m, m if binary_only else boundary, m if binary_only else distance]))
    return torch.as_tensor(np.stack(output), device=device, dtype=torch.float32)


class Model(nn.Module):
    def __init__(self, cfg, method):
        super().__init__()
        self.method = method
        self.image = convnext_tiny(weights=None)
        self.image.classifier[-1] = nn.Identity()
        self.shape = resnet18(weights=None)
        self.shape.fc = nn.Identity()
        self.px = nn.Linear(768, cfg["fusion_dim"])
        self.ps = nn.Linear(512, cfg["fusion_dim"])
        self.norm = nn.LayerNorm(cfg["fusion_dim"])
        self.classifier = nn.Linear(cfg["fusion_dim"], cfg["classes"])

    def forward(self, x, mask):
        sx = representation(mask, x.device, self.method == "binary_mask")
        image = self.px(self.image(x.float()))
        shape = self.ps(self.shape(sx))
        combined = (
            image
            if self.method == "image_only"
            else shape
            if self.method == "shape_only"
            else image + shape
        )
        embedding = self.norm(combined)
        return self.classifier(embedding), embedding


def js(left, right):
    p, q = left.softmax(-1), right.softmax(-1)
    middle = (p + q) / 2
    return (
        0.5
        * (
            (p * (p.clamp_min(1e-8).log() - middle.clamp_min(1e-8).log())).sum(-1)
            + (q * (q.clamp_min(1e-8).log() - middle.clamp_min(1e-8).log())).sum(-1)
        ).mean()
    )


def loss(model, batch, cfg):
    logits, _ = model(batch["x"], batch["mask"])
    weights = torch.as_tensor(cfg["class_weights"], device=logits.device, dtype=logits.dtype)
    result = F.cross_entropy(logits, batch["y"].long(), weight=weights)
    if model.method not in ("no_consistency", "image_only", "shape_only"):
        paired, _ = model(batch["x_pair"], batch["mask"])
        result = result + 0.1 * js(logits, paired)
    return result


def predict(model, batch, cfg, calibration=None):
    logits, embedding = model(batch["x"], batch["mask"])
    return {"logits": logits, "embedding": embedding}


def metrics(prediction, data, cfg):
    report = classification_metrics(prediction["logits"], data["y"])
    for field in ("background", "domain"):
        if field in data:
            for value in np.unique(data[field]):
                mask = data[field] == value
                report[field + "_" + str(value)] = classification_metrics(
                    prediction["logits"][mask], data["y"][mask]
                )
    return report


def inference(model, batch, cfg):
    return model(batch["x"], batch["mask"])
