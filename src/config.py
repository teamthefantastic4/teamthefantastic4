import os
from pathlib import Path

DATASET_ROOT_PATH = str(Path(__file__).parent.parent / "dataset")
os.makedirs(DATASET_ROOT_PATH, exist_ok=True)

DATASET_TRAIN = str(Path(DATASET_ROOT_PATH) / "application_train_hma.csv")

DATASET_TEST = str(Path(DATASET_ROOT_PATH) / "application_test_hma.csv")

