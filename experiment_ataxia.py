from mri_experiment import AtaxiaMRIDataset
import torch
from torch import nn
from torch.utils.data import DataLoader

from experiment import Classifier

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def build_loader(manifest_path, batch_size=8):
    train_dataset = AtaxiaMRIDataset(manifest_path, split="train")

    val_dataset = AtaxiaMRIDataset(manifest_path, split="val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def run_experiment(manifest_path, quaternion=False, epochs=20):
    device = torch.device("cpu")

    model = Classifier(
        quaternion=quaternion, 
        num_classes=2
    ).to(device)

    train_loader, val_loader = build_loader(manifest_path)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.001)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    
        print(
            f"Epoch {epoch + 1}/{epochs}"
            f"training loss: {loss:.4f}" 
        )

    return model

if __name__=="__main__":
    model = run_experiment(
        manifest_path="data/manifest.csv", 
        quaternion=False, 
        epochs=20
    )