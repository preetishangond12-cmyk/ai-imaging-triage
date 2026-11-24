"""Prepare the local CheXpert subset for model training and inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image


LABEL_COLUMNS = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
]


def resolve_image_path(path_value: str, image_root: Path) -> Path:
    """Resolve a CheXpert CSV path without depending on the current directory."""
    relative_path = Path(path_value)
    if relative_path.is_absolute():
        return relative_path
    return image_root / relative_path


def prepare_manifest(
    csv_path: Path,
    image_root: Path,
    output_path: Path,
    image_size: int = 224,
    uncertain_value: float = 0.0,
) -> pd.DataFrame:
    """Validate images, apply the paper's deterministic preprocessing contract, and save labels."""
    dataframe = pd.read_csv(csv_path)
    required_columns = {"Path", "patient_id", *LABEL_COLUMNS}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSV is missing required columns: {missing}")

    rows = []
    for row in dataframe.itertuples(index=False):
        image_path = resolve_image_path(row.Path, image_root)
        if not image_path.is_file():
            continue
        try:
            with Image.open(image_path) as image:
                image.convert("L").resize((image_size, image_size))
        except (OSError, ValueError):
            continue

        record = {
            "path": str(image_path.resolve()),
            "patient_id": row.patient_id,
        }
        for label in LABEL_COLUMNS:
            value = getattr(row, label.replace(" ", "_"))
            record[label] = uncertain_value if pd.isna(value) or value == -1 else float(value)
        rows.append(record)

    manifest = pd.DataFrame(rows, columns=["path", "patient_id", *LABEL_COLUMNS])
    if manifest.empty:
        raise FileNotFoundError(
            "No readable CheXpert images were found. Set --image-root to the folder "
            "containing the CheXpert-v1.0 directory."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("dataset/chexpert/chexpert_subset.csv"))
    parser.add_argument("--image-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("dataset/chexpert/processed_manifest.csv"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--uncertain-value", type=float, default=0.0)
    args = parser.parse_args()

    manifest = prepare_manifest(
        args.csv,
        args.image_root,
        args.output,
        args.image_size,
        args.uncertain_value,
    )
    print(f"Prepared {len(manifest)} readable frontal radiographs")
    print(f"Patients: {manifest['patient_id'].nunique()}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()