import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from scipy.ndimage import zoom
from torch.utils.data import Dataset

class AtaxiaMRIDataset(Dataset):
    """
    Loads, preprocessed anatomically aligned T1 MRI volumes:
    Each subject provides four neighboring slicers 

    labels: 
    0 - healthy control
    1 - genetically confirmed SCA3

    Research use only.
    """

    def __init__(
        self, 
        manifest_path, 
        split="train", 
        image_size=128, 
        slice_indices=(60, 64, 68, 72),
    ):
        self.image_size = image_size
        self.slice_indices = slice_indices

        with open(manifest_path, newline="") as f:
            rows = list(csv.DictReader(f))
        
        self.record = [
            row for row in rows
            if row['split'] == split
        ]

        if not self.record:
            raise ValueError(
                f"No MRI records found for split: {split}"
            )
        self.root = Path(manifest_path).resolve().parent
    
    def __len__(self):
        return len(self.record)
    
    def __getitem__(self, index):
        record = self.record[index]
    
        path = Path(record["path"])
        if not path.is_absolute():
            path = self.root / path

        volume = nib.load(str(path)).get_fdata(dtype=np.float32)

        if volume.ndim != 3:
            raise ValueError(
                f"Expected 3D MRI volume: {path}"
            )
        
        if max(self.slice_indices) >= volume.shape[2]:
            raise ValueError(
                f"Volume too small to select slices: {path}"
            )
        # These indices assume that all volumes have
        # already been registered into the same anatomical space
        slices = []

        for slice_index in self.slice_indices:
            image = volume[:, :, slice_index]
            #robust intensity normalization
            low, high = np.percentile(image, [1, 99])

            image = np.clip(image, low, high)

            if high > low:
                image = (image - low) / (high - low)
            else: 
                image = np.zeros_like(image)
            
            # resize to a common spatial dimension
            scale = (
                self.image_size / image.shape[0], 
                self.image_size / image.shape[1]
            )

            image = zoom(
                image, 
                scale, 
                order=1
            )

            slices.append(image)
        
        x = np.stack(slices, axis=0)
        x = torch.from_numpy(np.ascontiguousarray(x)).float()
        y = torch.tensor(int(record["label"]), dtype=torch.long)

        return x, y