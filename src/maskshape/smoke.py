"""CPU training and inference checks with generated samples."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from . import core
from .cli import Arrays, evaluate, plot, prepare, train
from .vision import classification_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/smoke")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(17)
    rng = np.random.default_rng(17)
    x = rng.random((6, 3, 64, 64), dtype=np.float32)
    mask = np.zeros((6, 1, 64, 64), dtype=np.float32)
    mask[:, :, 15:45, 20:35] = 1
    np.savez_compressed(output / "raw.npz", x=x, x_pair=x * mask + (1 - x) * (1 - mask),
                        mask=mask, y=np.tile([0, 1], 3), group=np.arange(6), split=np.repeat([0, 1, 2], 2))
    prepare(output / "raw.npz", output / "data", 17)
    dataset = Arrays(output / "data/train.npz")
    batch = {key: torch.stack([dataset[i][key] for i in range(2)]) for key in dataset[0]}
    cfg = dict(seed=17, device="cpu", epochs=1, patience=1, batch_size=2,
               learning_rate=1e-4, weight_decay=1e-4, classes=2, fusion_dim=8,
               class_weights=[1, 1], balance_classes=True)
    for method in core.METHODS:
        model = core.Model(cfg, method).train()
        value = core.loss(model, batch, cfg)
        assert torch.isfinite(value)
        value.backward()
        if method == "maskshape":
            keys = ("image.features.0.0.weight", "shape.conv1.weight", "px.weight", "ps.weight", "classifier.weight")
            parameters = dict(model.named_parameters())
            before = {key: parameters[key].detach().clone() for key in keys}
            for key in keys:
                assert parameters[key].grad is not None and parameters[key].grad.abs().sum() > 0, key
            torch.optim.SGD(model.parameters(), lr=.001).step()
            for key in keys:
                assert not torch.equal(before[key], parameters[key]), key
            model.eval()
            with torch.inference_mode():
                before_save = model(batch["x"], batch["mask"])[0]
            torch.save(model.state_dict(), output / "roundtrip.pt")
            model.load_state_dict(torch.load(output / "roundtrip.pt", weights_only=True))
            with torch.inference_mode():
                torch.testing.assert_close(before_save, model(batch["x"], batch["mask"])[0], rtol=0, atol=1e-7)
        del model
    train(cfg, output / "data", output / "trained", "maskshape")
    payload = torch.load(output / "trained/model.pt", weights_only=False, map_location="cpu")
    first = core.Model(cfg, "maskshape").eval()
    first.load_state_dict(payload["state"])
    second = core.Model(cfg, "maskshape").eval()
    second.load_state_dict(torch.load(output / "trained/model.pt", weights_only=False)["state"])
    with torch.inference_mode():
        torch.testing.assert_close(first(batch["x"], batch["mask"])[0], second(batch["x"], batch["mask"])[0], rtol=0, atol=1e-7)
    hand = classification_metrics(np.log([[.75, .25], [.75, .25]]), [0, 1])
    np.testing.assert_allclose(hand["accuracy"], .5)
    np.testing.assert_allclose(hand["balanced_accuracy"], .5)
    np.testing.assert_allclose(hand["macro_f1"], 1 / 3)
    np.testing.assert_allclose(hand["nll"], -np.log(.75 * .25) / 2)
    metrics = evaluate(output / "trained/model.pt", output / "data/test.npz", output / "evaluation")
    plot([output / "trained", output / "evaluation"], output / "figures")
    del first, second
    from .activation import main as activation
    sys.argv = ["maskshape.activation", "--checkpoint", str(output / "trained/model.pt"),
                "--data", str(output / "data/test.npz"), "--output", str(output / "figures/activation.png")]
    activation()
    print(json.dumps(dict(checks="passed", methods=len(core.METHODS), loss=float(value.detach()),
                          gradient_and_update="image,shape,projections,classifier", evaluation_accuracy=metrics["accuracy"])))


if __name__ == "__main__":
    main()
