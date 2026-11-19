import csv
import random
from pathlib import Path

DATA_DIR = Path("data/processed")
MANIFEST = Path("data/manifest.csv")

SEED = 42

files = sorted(DATA_DIR.glob("*.nii.gz"))


if not files:
    raise FileNotFoundError("No MRI files found in data/processed")

random.Random(SEED).shuffle(files)

n = len(files)

n_train = int(n * 0.70)
n_val = int(n * 0.15)

with MANIFEST.open("w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "subject_id",
        "path",
        "label",
        "split",
        "site"
    ])

    for index, path in enumerate(files):
        if index < n_train:
            split = "train"
        elif index < n_train + n_val:
            split = "val"
        else:
            split = "test"
        
        #ITX is a healthy volunteer dataset
        label = 0

        #Example: ITX002-Guys 0828 T1.nii.gz
        parts = path.name.split("-")
        subject_id = parts[0]
        site = parts[1]

        writer.writerow([
            subject_id, 
            f"processed/{path.name}", 
            label, 
            split, 
            site
        ])
    
    print(f"Total MRI scans: {n}")
    print(f"Training: {n_train}")
    print(f"Validation: {n_val}")
    print(f"Testing: {n - n_train - n_val}")
    print(f"Manifest created: {MANIFEST}")