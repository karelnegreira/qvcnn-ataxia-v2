import torch
import nibabel as nib
import numpy as np

print("Pytorch", torch.__version__)
print("Nibabel:", nib.__version__)
print("NumPy:", np.__version__)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device", device)

#simulated mri tensor for testing the enviroment
mri = torch.randn(1, 4, 128, 128).to(device)

print("MRI tensor shape", mri.shape)