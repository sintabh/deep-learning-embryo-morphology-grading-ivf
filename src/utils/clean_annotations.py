import os
import pandas as pd
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.utils.annotation_utils import load_annotations, add_quality_label

def clean_split(split: str):
    """Loads raw annotations for a split and produces a cleaned version."""
    print(f"\n=== Cleaning {split} annotations ===")
    
    df = load_annotations(split)
    df_clean = add_quality_label(df)

    out_path = os.path.join(PROCESSED_DATA_DIR, f"{split}_clean.csv")
    df_clean.to_csv(out_path, index=False)

    print(f"Saved cleaned annotations to: {out_path}")
    print(f"Shape: {df_clean.shape}")
    print(df_clean.head(10))


def main():
    for split in ["train", "test"]:
        clean_split(split)

    print("\nAll splits cleaned successfully.")

if __name__ == "__main__":
    main()