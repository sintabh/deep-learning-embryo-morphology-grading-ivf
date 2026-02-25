import os
import shutil
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RAW_DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, PROCESSED_DATA_DIR
from src.utils.annotation_utils import load_annotations, add_quality_label


RAW_IMAGE_DIR = os.path.join(RAW_DATA_DIR, "Human_Blastocyst_Dataset")


def _ensure_clean_dir(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def _split_train_val(
    df: pd.DataFrame, val_ratio: float = 0.2, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        random_state=seed,
        stratify=df["quality"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def _copy_images_for_split(df: pd.DataFrame, split_dir: str) -> None:
    for _, row in df.iterrows():
        img_name = row["image"]
        quality = row["quality"]

        src_path = os.path.join(RAW_IMAGE_DIR, img_name)
        if not os.path.exists(src_path):
            # Skip missing images
            continue

        dst_class_dir = os.path.join(split_dir, quality)
        os.makedirs(dst_class_dir, exist_ok=True)

        dst_path = os.path.join(dst_class_dir, img_name)
        shutil.copy2(src_path, dst_path)


def build_quality_dataset(val_ratio: float = 0.2, seed: int = 42) -> None:
    if not os.path.isdir(RAW_IMAGE_DIR):
        raise FileNotFoundError(f"Raw image directory not found: {RAW_IMAGE_DIR}")

    _ensure_clean_dir(PROCESSED_DATA_DIR)
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    train_ann = load_annotations("train")
    test_ann = load_annotations("test")

    train_ann = add_quality_label(train_ann)
    test_ann = add_quality_label(test_ann)

    train_df, val_df = _split_train_val(train_ann, val_ratio=val_ratio, seed=seed)
    test_df = test_ann

    _copy_images_for_split(train_df, TRAIN_DIR)
    _copy_images_for_split(val_df, VAL_DIR)
    _copy_images_for_split(test_df, TEST_DIR)


if __name__ == "__main__":
    build_quality_dataset()