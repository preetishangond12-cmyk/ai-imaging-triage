"""Evaluate a trained CheXpert checkpoint on the untouched test manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from train_chexpert import CheXpertDataset, LABEL_COLUMNS, build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, default=Path("dataset/chexpert/test.csv"))
    parser.add_argument("--image-root", type=Path, default=Path("."))
    parser.add_argument("--checkpoint", type=Path, default=Path("ai model/checkpoints/chexpert_densenet121.pt"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if payload.get("labels") != LABEL_COLUMNS:
        raise ValueError("Checkpoint label metadata does not match the CheXpert label order")
    model = build_model().to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    loader = DataLoader(CheXpertDataset(
        args.test,
        args.image_root,
        uncertain_label_policy=payload.get("uncertain_label_policy", "zeros"),
    ), batch_size=args.batch_size)
    predictions, targets, masks = [], [], []
    with torch.no_grad():
        for images, labels, label_mask in loader:
            predictions.append(torch.sigmoid(model(images.to(device))).cpu().numpy())
            targets.append(labels.numpy())
            masks.append(label_mask.numpy())

    probabilities = np.concatenate(predictions)
    actual = np.concatenate(targets)
    observed = np.concatenate(masks).astype(bool)
    predicted = (probabilities >= 0.5).astype(int)
    flat_actual = actual[observed].astype(int)
    flat_predicted = predicted[observed]
    flat_probabilities = probabilities[observed]
    metrics = {
        "accuracy": accuracy_score(flat_actual, flat_predicted),
        "precision": precision_score(flat_actual, flat_predicted, zero_division=0),
        "recall": recall_score(flat_actual, flat_predicted, zero_division=0),
        "f1": f1_score(flat_actual, flat_predicted, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(flat_actual, flat_probabilities)
    except ValueError:
        metrics["roc_auc"] = None
    for name, value in metrics.items():
        print(f"{name}={value:.4f}" if value is not None else f"{name}=unavailable")
    for index, label in enumerate(LABEL_COLUMNS):
        label_observed = observed[:, index]
        label_actual = actual[label_observed, index].astype(int)
        label_predicted = predicted[label_observed, index]
        if label_actual.size == 0:
            print(f"{label}: support=0 positive_rate=unavailable f1=unavailable roc_auc=unavailable")
            continue
        if np.unique(label_actual).size < 2:
            label_auc = None
        else:
            try:
                label_auc = roc_auc_score(label_actual, probabilities[label_observed, index])
            except ValueError:
                label_auc = None
        auc_text = f"{label_auc:.4f}" if label_auc is not None else "unavailable"
        print(
            f"{label}: support={label_observed.sum()} "
            f"positive_rate={label_actual.mean():.4f} "
            f"f1={f1_score(label_actual, label_predicted, zero_division=0):.4f} "
            f"roc_auc={auc_text}"
        )


if __name__ == "__main__":
    main()
