import os

# Root directory of the project (change this if the project is moved)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

TRAIN_DIR = os.path.join(PROCESSED_DATA_DIR, "train")
VAL_DIR = os.path.join(PROCESSED_DATA_DIR, "val")
TEST_DIR = os.path.join(PROCESSED_DATA_DIR, "test")

NOTEBOOKS_DIR = os.path.join(ROOT_DIR, "notebooks")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

MODELS_DIR = os.path.join(ROOT_DIR, "src", "models")
UTILS_DIR = os.path.join(ROOT_DIR, "src", "utils")
TRAINING_DIR = os.path.join(ROOT_DIR, "src", "training")