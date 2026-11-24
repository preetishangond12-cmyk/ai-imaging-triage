"""Create patient-level train/validation/test splits from CheXpert annotations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("dataset/chexpert/train_cheXbert.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/chexpert"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = pd.read_csv(args.source)
    if "Path" not in data.columns:
        raise ValueError(f"{args.source} must contain a Path column")
    if "Frontal/Lateral" in data.columns:
        data = data[data["Frontal/Lateral"] == "Frontal"].copy()
    data["patient_id"] = data["Path"].astype(str).str.extract(r"(patient\d+)", expand=False)
    if data["patient_id"].isna().any():
        raise ValueError("Some image paths do not contain a patient ID")
    patients = data["patient_id"].drop_duplicates().sample(frac=1, random_state=args.seed)
    train_end = int(len(patients) * 0.70)
    val_end = int(len(patients) * 0.85)
    train_patients = set(patients.iloc[:train_end])
    val_patients = set(patients.iloc[train_end:val_end])
    test_patients = set(patients.iloc[val_end:])
    splits = {
        "train": data[data["patient_id"].isin(train_patients)],
        "val": data[data["patient_id"].isin(val_patients)],
        "test": data[data["patient_id"].isin(test_patients)],
    }

    if (
        (train_patients & val_patients)
        or (train_patients & test_patients)
        or (val_patients & test_patients)
    ):
        raise RuntimeError("Patient leakage detected between dataset splits")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        split.to_csv(args.output_dir / f"{name}.csv", index=False)
        print(f"{name}: {len(split)} images, {split['patient_id'].nunique()} patients")
    print(f"Saved patient-level splits to {args.output_dir}")


if __name__ == "__main__":
    main()
