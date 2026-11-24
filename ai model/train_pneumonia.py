"""Train the binary classifier using the Kaggle Chest X-Ray Images (Pneumonia) layout."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

LABELS = ["No Finding", "Pneumonia"]
IMAGE_SIZE = 224


def build_model() -> nn.Module:
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(LABELS))
    return model


def make_dataset(folder: Path, training: bool) -> ImageFolder:
    operations = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if training:
        operations.extend([
            transforms.RandomRotation(7),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
    operations.extend([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    dataset = ImageFolder(folder, transform=transforms.Compose(operations))
    expected = {"NORMAL": 0, "PNEUMONIA": 1}
    if dataset.class_to_idx != expected:
        raise ValueError(f"Expected NORMAL and PNEUMONIA folders, found {dataset.class_to_idx}")
    return dataset


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    loss_total = 0.0
    correct = 0
    count = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_function(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        loss_total += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += images.size(0)
    return loss_total / max(count, 1), correct / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder containing train/ and val/ directories")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, default=Path("pneumonia_densenet121.pt"))
    args = parser.parse_args()

    train_dataset = make_dataset(args.data_dir / "train", training=True)
    validation_folder = args.data_dir / "val"
    validation_dataset = make_dataset(validation_folder, training=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, num_workers=2)

    class_counts = torch.bincount(torch.tensor(train_dataset.targets), minlength=len(LABELS)).float()
    class_weights = class_counts.sum() / (len(LABELS) * class_counts.clamp_min(1))
    model = build_model().to(device)
    loss_function = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_validation_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, loss_function, optimizer, device)
        validation_loss, validation_accuracy = run_epoch(model, validation_loader, loss_function, None, device)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={validation_loss:.4f} val_acc={validation_accuracy:.4f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "labels": LABELS}, args.output)

    print(f"Best checkpoint saved to: {args.output}")
    print(f"Training device: {device}")


if __name__ == "__main__":
    main()
