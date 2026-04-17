#!/usr/bin/env python3
"""
Utility script for generating multiple train/val/test splits for the copper
ore X-ray dataset that is organized in paired high/low energy folders.

The script will:
1. Scan the dataset root and pair up high/low images for each label.
2. Compute the mean grayscale (average of the high and low images) per sample.
3. Assign each sample into thin / medium / thick groups by mean grayscale.
4. Produce one stratified random split (train/val/test = 70/15/15).
5. Produce three cross-thickness splits where different groups act as the test set.
6. Save a master CSV plus per-split helper CSV files.

Author: Codex (GPT-5)
Date: March 5, 2026
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from PIL import Image


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_RANDOM_SEED = 2024


@dataclass
class SampleRecord:
    """Keeps the metadata we need for one paired sample."""

    sample_id: str
    label: int  # 1 = mine, 0 = waste
    high_path: Path
    low_path: Path
    mean_gray: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CSV splits for the copper ore X-ray dataset."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(r"C:\Users\朱爷\Desktop\All Datasets\原始购买的二分类数据集\高低能拆分好的"),
        help="Root directory that contains the mine/ and waste/ folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("split_outputs"),
        help="Directory where CSV files will be written (created if missing).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed used for reproducible shuffling.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation ratio for both the random split and cross-thickness split #1.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Training ratio for the random split (test ratio = 1 - train - val).",
    )
    parser.add_argument(
        "--intensity-source",
        choices=("high", "low", "both"),
        default="both",
        help="Which channel(s) to use when computing the mean grayscale.",
    )
    return parser.parse_args()


def derive_sample_id(stem: str) -> str:
    """
    Attempt to strip channel suffixes (_high / _low or -high / -low) from the filename stem.
    If no suffix is found we fall back to the original stem.
    """
    lowered = stem.lower()
    for suffix in ("_high", "_low", "-high", "-low"):
        if lowered.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def compute_mean_gray(image_path: Path) -> float:
    """Load a grayscale image and compute its mean pixel value (0-255)."""
    with Image.open(image_path) as img:
        grayscale = img.convert("L")
        array = np.asarray(grayscale, dtype=np.float32)
    return float(array.mean())


def gather_samples(
    root_dir: Path, intensity_source: str = "both"
) -> List[SampleRecord]:
    """
    Traverse the dataset structure and gather paired samples while computing mean grayscale.
    """
    records: List[SampleRecord] = []
    root_dir = root_dir.expanduser().resolve()

    if not root_dir.exists():
        raise FileNotFoundError(f"Dataset root not found: {root_dir}")

    subsets = [("mine", 1), ("waste", 0)]
    for subset_name, label_value in subsets:
        high_dir = root_dir / subset_name / "high"
        low_dir = root_dir / subset_name / "low"
        if not high_dir.exists() or not low_dir.exists():
            raise FileNotFoundError(
                f"Missing high/low folders under {root_dir / subset_name}"
            )

        for high_path in sorted(high_dir.iterdir()):
            if not high_path.is_file() or high_path.suffix.lower() not in IMG_EXTENSIONS:
                continue

            low_path = low_dir / high_path.name
            if not low_path.exists():
                print(f"[WARN] Low-energy image missing for {high_path.name}; skipping.")
                continue

            mean_values = []
            if intensity_source in ("high", "both"):
                mean_values.append(compute_mean_gray(high_path))
            if intensity_source in ("low", "both"):
                mean_values.append(compute_mean_gray(low_path))

            if not mean_values:
                raise ValueError("No intensity source selected. This should never happen.")

            mean_gray = float(sum(mean_values) / len(mean_values))
            sample_id = derive_sample_id(high_path.stem)
            records.append(
                SampleRecord(
                    sample_id=sample_id,
                    label=label_value,
                    high_path=high_path,
                    low_path=low_path,
                    mean_gray=mean_gray,
                )
            )

    if not records:
        raise RuntimeError("No samples were collected. Please check the dataset structure.")

    return records


def assign_thickness_groups(df: pd.DataFrame) -> pd.Series:
    """
    Sort samples by mean_gray (ascending) and split into thirds:
    - Lowest third (darkest) -> 'thick'
    - Middle third -> 'medium'
    - Highest third (brightest) -> 'thin'
    """
    ordered = df.sort_values("mean_gray").reset_index()
    n = len(ordered)
    third = n // 3

    group_labels = (
        ["thick"] * third
        + ["medium"] * third
        + ["thin"] * (n - 2 * third)
    )
    ordered["thickness_group"] = group_labels
    thickness_series = pd.Series(
        index=ordered["index"], data=ordered["thickness_group"]
    )
    return thickness_series.reindex(df.index)


def stratified_random_split(
    df: pd.DataFrame,
    label_col: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> pd.Series:
    """
    Assign each row a split (train/val/test) while preserving class balance.
    """
    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError("Train ratio must be between 0 and 1.")
    if val_ratio <= 0 or val_ratio >= 1:
        raise ValueError("Validation ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("Train ratio + val ratio must be less than 1.")

    rng = random.Random(seed)
    splits = pd.Series(index=df.index, dtype="object")

    for label_value, indices in df.groupby(label_col).groups.items():
        idx_list = list(indices)
        rng.shuffle(idx_list)
        n = len(idx_list)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_train = n - n_val

        train_idx = idx_list[:n_train]
        val_idx = idx_list[n_train : n_train + n_val]
        test_idx = idx_list[n_train + n_val :]

        splits.loc[train_idx] = "train"
        splits.loc[val_idx] = "val"
        splits.loc[test_idx] = "test"

    return splits


def stratified_sample(
    df: pd.DataFrame,
    indices: Iterable[int],
    label_col: str,
    fraction: float,
    seed: int,
) -> List[int]:
    """Pick a stratified subset from the provided indices."""
    rng = random.Random(seed)
    subset = df.loc[list(indices)]
    sampled: List[int] = []
    for _, group in subset.groupby(label_col):
        idx_list = list(group.index)
        rng.shuffle(idx_list)
        take = int(round(len(idx_list) * fraction))
        sampled.extend(idx_list[:take])
    return sampled


def assign_cross_splits(
    df: pd.DataFrame,
    thickness_col: str,
    val_ratio: float,
    seed: int,
) -> Dict[str, pd.Series]:
    """Create cross-thickness split columns."""
    rng_seed = seed

    # Split #1: thin+medium train, thick test, 15% validation from train.
    cross1 = pd.Series(index=df.index, dtype="object")
    train_mask_1 = df[thickness_col].isin(["thin", "medium"])
    test_mask_1 = df[thickness_col] == "thick"
    cross1.loc[test_mask_1] = "test"
    train_indices_1 = df.index[train_mask_1]

    val_indices_1 = stratified_sample(
        df, train_indices_1, "label", val_ratio, rng_seed + 1
    )
    val_set = set(val_indices_1)
    cross1.loc[list(val_set)] = "val"
    train_only_indices = [idx for idx in train_indices_1 if idx not in val_set]
    cross1.loc[train_only_indices] = "train"

    # Split #2: medium+thick train, thin test.
    cross2 = pd.Series(index=df.index, dtype="object")
    cross2.loc[df[thickness_col].isin(["medium", "thick"])] = "train"
    cross2.loc[df[thickness_col] == "thin"] = "test"

    # Split #3: thin+thick train, medium test.
    cross3 = pd.Series(index=df.index, dtype="object")
    cross3.loc[df[thickness_col].isin(["thin", "thick"])] = "train"
    cross3.loc[df[thickness_col] == "medium"] = "test"

    return {
        "split_cross1": cross1,
        "split_cross2": cross2,
        "split_cross3": cross3,
    }


def save_csv_views(df: pd.DataFrame, output_dir: Path) -> None:
    """Save the master CSV plus specific per-split CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Master file with every column.
    master_path = output_dir / "copper_xray_all_splits.csv"
    df.to_csv(master_path, index=False)

    # Helper CSVs focused on each experiment configuration.
    columns_common = [
        "sample_id",
        "label",
        "high_path",
        "low_path",
        "thickness_group",
    ]

    random_cols = columns_common + ["split"]
    df[random_cols].to_csv(output_dir / "split_random.csv", index=False)

    cross1_cols = columns_common + ["split_cross1"]
    df[cross1_cols].to_csv(output_dir / "split_cross1.csv", index=False)

    cross2_cols = columns_common + ["split_cross2"]
    df[cross2_cols].to_csv(output_dir / "split_cross2.csv", index=False)

    cross3_cols = columns_common + ["split_cross3"]
    df[cross3_cols].to_csv(output_dir / "split_cross3.csv", index=False)

    print(f"Saved master CSV to: {master_path}")
    print(f"Random split CSV: {output_dir / 'split_random.csv'}")
    print(f"Cross split CSVs: {[output_dir / f'split_cross{i}.csv' for i in (1, 2, 3)]}")


def main() -> None:
    args = parse_args()

    print(f"[INFO] Scanning dataset under: {args.root}")
    records = gather_samples(args.root, args.intensity_source)
    print(f"[INFO] Collected {len(records)} paired samples.")

    df = pd.DataFrame(
        {
            "sample_id": [rec.sample_id for rec in records],
            "label": [rec.label for rec in records],
            "high_path": [str(rec.high_path) for rec in records],
            "low_path": [str(rec.low_path) for rec in records],
            "mean_gray": [rec.mean_gray for rec in records],
        }
    )

    df["thickness_group"] = assign_thickness_groups(df)

    # Stratified random split.
    val_ratio = args.val_ratio
    test_ratio = 1.0 - args.train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError("Train ratio + val ratio must be less than 1.")
    df["split"] = stratified_random_split(
        df, label_col="label", train_ratio=args.train_ratio, val_ratio=val_ratio, seed=args.seed
    )

    # Cross-thickness splits.
    cross_splits = assign_cross_splits(df, "thickness_group", val_ratio, args.seed)
    for column, values in cross_splits.items():
        df[column] = values

    save_csv_views(df, args.output_dir)

    label_counts = df["label"].value_counts().to_dict()
    group_counts = df["thickness_group"].value_counts().to_dict()
    print(f"[INFO] Label counts: {label_counts}")
    print(f"[INFO] Thickness group counts: {group_counts}")


if __name__ == "__main__":
    main()
