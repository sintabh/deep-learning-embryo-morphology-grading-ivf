import os
import shutil
from typing import List, Tuple

from sklearn.model_selection import train_test_split

from src.config import RAW_DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR


def _get_class_dirs(raw_root: str) -> List[str]:
    entries = [os.path.join(raw_root, d) for d in os.listdir(raw_root)]
    dirs = [d for d in entries if os.path.isdir(d)]
    if len(dirs) == 1:
        nested_entries = [
            os.path.join(dirs[0], d) for d in os.listdir(dirs[0])
        ]
        nested_dirs = [d for d in nested_entries if os.path.isdir(d)]
        if nested_dirs:
            return nested_dirs
    return dirs


def _gather_images(class_dir: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = []
    for root, _, filenames in os.walk(class_dir):
        for f in filenames:
            if os.path.splitext(f.lower())[1] in exts:
                files.append(os.path.join(root, f))
    return files


def _split_indices(
    items: List[str], train_ratio: float, val_ratio: float, test_ratio: float, seed: int
) -> Tuple[List[str], List[str], List[str]]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    train_items, temp_items = train_test_split(
        items, test_size=(1.0 - train_ratio), random_state=seed, shuffle=True
    )
    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    val_items, test_items = train_test_split(
        temp_items,
        test_size=(1.0 - relative_val_ratio),
        random_state=seed,
        shuffle=True,
    )
    return train_items, val_items, test_items


def _copy_files(file_paths: List[str], dst_root: str, class_name: str) -> None:
    dst_class_dir = os.path.join(dst_root, class_name)
    os.makedirs(dst_class_dir, exist_ok=True)
    for src_path in file_paths:
        filename = os.path.basename(src_path)
        dst_path = os.path.join(dst_class_dir, filename)
        shutil.copy2(src_path, dst_path)


def split_dataset(
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    class_dirs = _get_class_dirs(RAW_DATA_DIR)
    if not class_dirs:
        raise RuntimeError(f"No class directories found in {RAW_DATA_DIR}")

    for class_dir in class_dirs:
        class_name = os.path.basename(class_dir.rstrip(os.sep))
        images = _gather_images(class_dir)
        if not images:
            continue

        train_imgs, val_imgs, test_imgs = _split_indices(
            images, train_ratio, val_ratio, test_ratio, seed
        )

        _copy_files(train_imgs, TRAIN_DIR, class_name)
        _copy_files(val_imgs, VAL_DIR, class_name)
        _copy_files(test_imgs, TEST_DIR, class_name)


if __name__ == "__main__":
    split_dataset()