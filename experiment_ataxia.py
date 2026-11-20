from pathlib import Path
from mri_experiment import AtaxiaMRIDataset
import torch
from torch import nn
from torch.utils.data import DataLoader
import time

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

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)

        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size
    
    average_loss = total_loss / total_samples

    accuracy = total_correct / total_samples

    return average_loss, accuracy


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

def run_experiment_v2(
        manifest_path, 
        quaternion=False, 
        epochs=20, 
        smoke_test=False
):
    torch.manual_seed(42)

    device = torch.device("cpu")

    model_name = "QCNN" if quaternion else "CNN"

    train_loader, val_loader = build_loader(manifest_path)

    train_labels = {
        int(row["label"])
        for row in val_loader.dataset.record
    }

    val_labels = {
        int(row["label"])
        for row in val_loader.dataset.record
    }

    if train_labels != {0, 1} or val_labels != {0, 1}:
        if not smoke_test:
            raise ValueError(
                "Both classes are required for "
                "ataxia classification"
            )
        
        print(
            "WARNING: One-class smoke test only. "
            "Results have no diagnostic validity"
        )

    model = Classifier(
        quaternion=quaternion, 
        num_classes=2
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=0.0003, 
        weight_decay=0.001
    )

    criterion = nn.CrossEntropyLoss()

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"\nModel: {model_name}")
    print(f"Device: {device}")
    print(f"Parameters: {parameters:,}")
    print(f"Training samples:  {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    best_val_loss = float("inf")

    start_time = time.perf_counter()

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, 
            train_loader, 
            optimizer, 
            criterion, 
            device
        )

        val_loss, val_accuracy = evaluate(
            model, 
            val_loader, 
            criterion, 
            device
        )

        print(
            f"Epoch {epoch + 1:02d}/{epochs} |"
            f"Train loss: {train_loss:.6f} | "
            f"Val loss: {val_loss:.6f} |"
            f"Val accuracy: {val_accuracy * 100:.2f}%"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            prefix = (
                "smoke_test_"
                if smoke_test
                else ""
            )

            torch.save(
                model.state_dict(), 
                checkpoint_dir / 
                f"{prefix}{model_name.lower()}_best.pt"
            )
    
    elapsed_time = time.perf_counter() - start_time

    print(f"\nTraining completed: {model_name}")
    print(f"Training time: {elapsed_time:.2f} seconds")
    print(f"Best validation loss: {best_val_loss:.6f}")

    return model



if __name__=="__main__":
    print("\n===============STANDARD CNN==================")
    cnn_model = run_experiment_v2(
        manifest_path="data/manifest.csv", 
        quaternion=False, 
        epochs=3,
        smoke_test=True
    )

    print("\n===============QUATERNION CNN=================")
    qcnn_model = run_experiment_v2(
        manifest_path="data/manifest.csv",
        quaternion=True, 
        epochs=3, 
        smoke_test=True
    )