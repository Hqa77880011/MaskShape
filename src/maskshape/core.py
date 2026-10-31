import numpy as np
import torch
from scipy import ndimage
from torch import nn
from torch.nn import functional as F
from torchvision.models import convnext_tiny, resnet18

from .vision import classification_metrics


METHODS = ["maskshape", "image_only", "shape_only", "no_consistency", "binary_mask",
           "mask_crop", "masked_image", "mask_channel", "concatenation"]


def mask_crop(images, masks):
    """Resize each tight defect bounding box to the original image grid."""
    crops = []
    for image, mask in zip(images, masks):
        coordinates = torch.nonzero(mask.squeeze() > .5)
        if not len(coordinates):
            crops.append(image)
            continue
        low, high = coordinates.amin(0), coordinates.amax(0) + 1
        crop = image[:, low[0]:high[0], low[1]:high[1]][None]
        crops.append(F.interpolate(crop, size=image.shape[-2:], mode="bilinear", align_corners=False)[0])
    return torch.stack(crops)


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
        self.image = None
        self.shape = None
        if method != "shape_only":
            self.image = convnext_tiny(weights=None)
            self.image.classifier[-1] = nn.Identity()
            if method == "mask_channel":
                self.image.features[0][0] = nn.Conv2d(4, 96, 4, 4)
            self.px = nn.Linear(768, cfg["fusion_dim"])
        if method not in ("image_only", "mask_crop", "masked_image", "mask_channel"):
            self.shape = resnet18(weights=None)
            self.shape.fc = nn.Identity()
            self.ps = nn.Linear(512, cfg["fusion_dim"])
        if method == "concatenation":
            self.fusion = nn.Linear(2 * cfg["fusion_dim"], cfg["fusion_dim"])
        self.norm = nn.LayerNorm(cfg["fusion_dim"])
        self.classifier = nn.Linear(cfg["fusion_dim"], cfg["classes"])

    def forward(self, x, mask):
        image = shape = None
        if self.image is not None:
            value = x.float()
            if self.method == "mask_crop":
                value = mask_crop(value, mask)
            elif self.method == "masked_image":
                value = value * mask
            elif self.method == "mask_channel":
                value = torch.cat([value, mask.float()], 1)
            image = self.px(self.image(value))
        if self.shape is not None:
            sx = representation(mask, x.device, self.method == "binary_mask")
            shape = self.ps(self.shape(sx))
        combined = shape if image is None else image if shape is None else image + shape
        if self.method == "concatenation":
            combined = self.fusion(torch.cat([image, shape], -1))
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
    if model.method not in ("no_consistency", "image_only", "shape_only", "mask_crop", "masked_image", "mask_channel"):
        paired, _ = model(batch["x_pair"], batch["mask"])
        result = result + 0.1 * js(logits, paired)
    return result


def predict(model, batch, cfg, calibration=None):
    logits, embedding = model(batch["x"], batch["mask"])
    result = {"logits": logits, "embedding": embedding}
    if "x_pair" in batch:
        result["paired_logits"] = model(batch["x_pair"], batch["mask"])[0]
    return result


def metrics(prediction, data, cfg):
    report = classification_metrics(prediction["logits"], data["y"])
    if "paired_logits" in prediction:
        pair = torch.stack([torch.as_tensor(prediction["logits"]), torch.as_tensor(prediction["paired_logits"])], 1).softmax(-1)
        report["paired_probability_variance"] = float(pair.var(1, unbiased=False).mean())
        report["paired_prediction_agreement"] = float((pair[:, 0].argmax(-1) == pair[:, 1].argmax(-1)).float().mean())
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
