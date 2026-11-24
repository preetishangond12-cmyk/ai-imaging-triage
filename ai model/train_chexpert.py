"""Train a 14-label CheXpert classifier from the prepared manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from labels import CHEXPERT_LABELS

LABEL_COLUMNS = CHEXPERT_LABELS


class CheXpertDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        image_root: Path = Path("."),
        image_size: int = 224,
        uncertain_label_policy: str = "zeros",
    ) -> None:
        self.data = pd.read_csv(manifest_path)
        if "path" not in self.data.columns and "Path" in self.data.columns:
            self.data = self.data.rename(columns={"Path": "path"})
        required_columns = {"path", *LABEL_COLUMNS}
        missing_columns = required_columns - set(self.data.columns)
        if missing_columns:
            raise ValueError(f"{manifest_path} is missing columns: {sorted(missing_columns)}")
        if uncertain_label_policy not in {"zeros", "ones", "ignore"}:
            raise ValueError("uncertain_label_policy must be zeros, ones, or ignore")
        self.image_root = Path(image_root)
        self.uncertain_label_policy = uncertain_label_policy
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.data.iloc[index]
        image_path = Path(str(row["path"]))
        if not image_path.is_absolute():
            image_path = self.image_root / image_path
        with Image.open(image_path) as image:
            image_tensor = self.transform(image.convert("L"))
        raw_labels = pd.to_numeric(row[LABEL_COLUMNS], errors="coerce").to_numpy(dtype=np.float32)
        observed = np.isfinite(raw_labels)
        uncertain = raw_labels == -1
        if self.uncertain_label_policy == "ignore":
            observed &= ~uncertain
            raw_labels[uncertain] = 0
        elif self.uncertain_label_policy == "ones":
            raw_labels[uncertain] = 1
        else:
            raw_labels[uncertain] = 0
        raw_labels[~observed] = 0
        labels = torch.tensor(raw_labels, dtype=torch.float32)
        label_mask = torch.tensor(observed, dtype=torch.float32)
        return image_tensor.repeat(3, 1, 1), labels, label_mask


def build_model(pretrained: bool = False) -> nn.Module:
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, len(LABEL_COLUMNS))
    return model


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    observed_labels = 0.0
    for images, labels, label_mask in loader:
        images = images.to(device)
        labels = labels.to(device)
        label_mask = label_mask.to(device)
        with torch.set_grad_enabled(training):
            per_label_loss = loss_function(model(images), labels)
            loss = (per_label_loss * label_mask).sum() / label_mask.sum().clamp_min(1)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += (per_label_loss.detach() * label_mask).sum().item()
        observed_labels += label_mask.sum().item()
    if observed_labels == 0:
        raise ValueError("The manifest contains no observed labels.")
    return total_loss / observed_labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("dataset/chexpert/train.csv"))
    parser.add_argument("--val", type=Path, default=Path("dataset/chexpert/val.csv"))
    parser.add_argument("--image-root", type=Path, default=Path("."))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Initialize from ImageNet weights (downloads them if not already cached).",
    )
    parser.add_argument(
        "--uncertain-label-policy",
        choices=("zeros", "ones", "ignore"),
        default="zeros",
        help="How to treat CheXpert -1 labels; blank labels are always ignored.",
    )
    parser.add_argument("--output", type=Path, default=Path("ai model/checkpoints/chexpert_densenet121.pt"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = CheXpertDataset(
        args.train, args.image_root, uncertain_label_policy=args.uncertain_label_policy
    )
    val_dataset = CheXpertDataset(
        args.val, args.image_root, uncertain_label_policy=args.uncertain_label_policy
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    model = build_model(pretrained=args.pretrained).to(device)
    loss_function = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, loss_function, optimizer, device)
        val_loss = run_epoch(model, val_loader, loss_function, None, device)
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "labels": LABEL_COLUMNS,
                "uncertain_label_policy": args.uncertain_label_policy,
                "image_size": 224,
            }, args.output)

    print(f"Best checkpoint: {args.output}")
    print(f"Device: {device}")


if __name__ == "__main__":
    main()
