"""Evaluate the binary model on the dataset's held-out test folder."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from train_pneumonia import LABELS, build_model, make_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = build_model().to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    test_dataset = make_dataset(args.data_dir / "test", training=False)
    loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=2)
    predictions, targets, positive_scores = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            probabilities = torch.softmax(model(images.to(device)), dim=1)
            positive_scores.extend(probabilities[:, 1].cpu().tolist())
            predictions.extend(probabilities.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())

    metrics = {
        "accuracy": accuracy_score(targets, predictions),
        "precision": precision_score(targets, predictions, zero_division=0),
        "recall": recall_score(targets, predictions, zero_division=0),
        "f1": f1_score(targets, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(targets, predictions).tolist(),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(targets, positive_scores)
    except ValueError:
        metrics["roc_auc"] = None
    for name, value in metrics.items():
        print(f"{name}={value:.4f}" if isinstance(value, float) else f"{name}={value}")


if __name__ == "__main__":
    main()
