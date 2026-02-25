import os
from typing import Literal

import pandas as pd

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

SplitType = Literal["train", "test"]


def _get_annotation_path(split: SplitType) -> str:
    if split == "train":
        filename = "Gardner_train_silver.csv"
    elif split == "test":
        filename = "Gardner_test_gold_onlyGardnerScores.csv"
    else:
        raise ValueError(f"Unsupported split: {split}")
    path = os.path.join(RAW_DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Annotation file not found: {path}")
    return path


def load_annotations(split: SplitType) -> pd.DataFrame:
    """
    Load and clean Gardner annotations for a given split.

    The raw CSV uses ';' as separator and has a single combined header.
    This function splits it into proper columns:
        - Image
        - EXP_gold (expansion)
        - ICM_gold
        - TE_gold
    """
    path = _get_annotation_path(split)
    raw_df = pd.read_csv(path, header=0)

    col = raw_df.columns[0]
    # Split the header into column names
    header_parts = [p for p in col.split(";") if p]
    if len(header_parts) < 4:
        raise RuntimeError(f"Unexpected header format in {path}: {col}")

    image_col, exp_col, icm_col, te_col = header_parts[:4]

    # Split each row on ';' and drop empty tail
    split_values = raw_df[col].str.split(";", expand=True)

    # Keep the first four parts (Image, EXP, ICM, TE)
    split_values = split_values.iloc[:, :4]
    split_values.columns = [image_col, exp_col, icm_col, te_col]

    # Strip spaces
    for c in split_values.columns:
        split_values[c] = split_values[c].astype(str).str.strip()

    # Rename to normalized column names
    df = split_values.rename(
        columns={
            image_col: "image",
            exp_col: "expansion",
            icm_col: "icm",
            te_col: "te",
        }
    )

    return df


def add_quality_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 3-class embryo quality label based on expansion, ICM, and TE grades.

    Rules (heuristic):
        - high:   expansion >= 3 and ICM >= 1 and TE >= 1
        - low:    expansion <= 1 or ICM == 0 or TE == 0
        - medium: everything else with numeric grades

    Rows with non-numeric (e.g. 'ND', 'NA') in any of the three fields are dropped.
    """

    def to_int_or_none(x: str) -> int | None:
        try:
            return int(x)
        except ValueError:
            return None

    exp = df["expansion"].apply(to_int_or_none)
    icm = df["icm"].apply(to_int_or_none)
    te = df["te"].apply(to_int_or_none)

    def classify(e: int | None, i: int | None, t: int | None) -> str | None:
        if e is None or i is None or t is None:
            return None

        if e >= 3 and i >= 1 and t >= 1:
            return "high"

        if e <= 1 or i == 0 or t == 0:
            return "low"

        return "medium"

    df = df.copy()
    df["quality"] = [
        classify(e, i, t) for e, i, t in zip(exp, icm, te)
    ]
    df = df[df["quality"].notnull()].reset_index(drop=True)
    return df


def load_multitask_annotations(split: SplitType) -> pd.DataFrame:
    """
    Load cleaned annotations for multi-task learning and keep only rows
    where expansion, ICM, and TE are all numeric.

    Uses the cleaned CSVs created under PROCESSED_DATA_DIR:
        - train_clean.csv
        - test_clean.csv

    Returns a DataFrame with columns:
        - image
        - expansion_int
        - icm_int
        - te_int
    """
    if split not in ("train", "test"):
        raise ValueError(f"Unsupported split for multitask: {split}")

    path = os.path.join(PROCESSED_DATA_DIR, f"{split}_clean.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Clean annotation file not found: {path}")

    df = pd.read_csv(path)

    def to_int_or_none(x) -> int | None:
        try:
            return int(x)
        except (ValueError, TypeError):
            return None

    exp = df["expansion"].apply(to_int_or_none)
    icm = df["icm"].apply(to_int_or_none)
    te = df["te"].apply(to_int_or_none)

    df = df.copy()
    df["expansion_int"] = exp
    df["icm_int"] = icm
    df["te_int"] = te

    df = df[
        df["expansion_int"].notnull()
        & df["icm_int"].notnull()
        & df["te_int"].notnull()
    ].reset_index(drop=True)

    df["expansion_int"] = df["expansion_int"].astype(int)
    df["icm_int"] = df["icm_int"].astype(int)
    df["te_int"] = df["te_int"].astype(int)

    return df[["image", "expansion_int", "icm_int", "te_int"]]