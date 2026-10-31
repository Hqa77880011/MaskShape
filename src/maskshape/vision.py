import numpy as np

import torch


from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def classification_metrics(logits, labels):
    probability = torch.softmax(torch.as_tensor(logits), -1).numpy()
    labels = np.asarray(labels).astype(int)
    prediction = probability.argmax(-1)
    confidence = probability.max(-1)
    correct = prediction == labels
    classes = np.arange(probability.shape[1])
    confusion = confusion_matrix(labels, prediction, labels=classes)
    ece = 0.0
    for lower in np.linspace(0, 1, 16)[:-1]:
        mask = (confidence > lower) & (confidence <= lower + 1 / 15)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(np.mean(np.diag(confusion) / np.maximum(confusion.sum(1), 1))),
        "macro_f1": float(f1_score(labels, prediction, labels=classes, average="macro", zero_division=0)),
        "top3": float(
            np.mean([y in row for y, row in zip(labels, np.argsort(-probability, axis=-1)[:, :3])])
        ),
        "ece": float(ece),
        "nll": float(
            -np.log(np.maximum(probability[np.arange(len(labels)), labels], 1e-12)).mean()
        ),
    }
