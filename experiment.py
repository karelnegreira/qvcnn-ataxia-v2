"""
PathMIST CNN Vs Quaternion CNN benchmark (educational, not clinical)
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from medmnist import INFO, PathMNIST
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


console = Console()

class FourChannelDataset(Dataset):
    """RGB tensor plus a grayscale channel derived from RGB"""
    def __init__(self, split):
        self.base = PathMNIST(split=split, download=True, transform=transforms.ToTensor())

    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, index):
        rgb, label = self.base[index]
        gray = (0.2989 * rgb[0:1] + 0.5870*rgb[1:2] + 0.1140 * rgb[2:3])
        x = torch.cat((rgb, gray), dim=0)
        x = (x - 0.5) / 0.5

        return x, int(np.asarray(label).item())
    
class QuaternionConv2d(nn.Module):
    """Hamilton product convolution, grouping channels into (r, i, j, k)"""

    def __init__(self, in_q, out_q, kernel_size=3, padding=1):
        super().__init__()
        kwargs = dict(kernel_size=kernel_size, padding=padding, bias=False)
        self.wr = nn.Conv2d(in_q, out_q, **kwargs)
        self.wi = nn.Conv2d(in_q, out_q, **kwargs)
        self.wj = nn.Conv2d(in_q, out_q, **kwargs)
        self.wk = nn.Conv2d(in_q, out_q, **kwargs)
        self.bias = nn.Parameter(torch.zeros(4 * out_q))

    def forward(self, x):
        r, i, j, k = torch.chunk(x, 4, dim=1)
        yr = self.wr(r) - self.wi(i) - self.wj(j) - self.wk(k)
        yi = self.wr(i) + self.wi(r) + self.wj(k) - self.wk(j)
        yj = self.wr(j) - self.wi(k) + self.wj(r) + self.wk(i)
        yk = self.wr(k) + self.wi(j) - self.wj(i) + self.wk(r)
        return torch.cat((yr, yi, yj, yk), dim=1) + self.bias[None, :, None, None]

class Classifier(nn.Module):
    def __init__(self, quaternion=False, num_classes=9):
        super().__init__()
        if quaternion:
            layers = [
                 QuaternionConv2d(1, 4), nn.ReLU(), nn.MaxPool2d(2),
                 QuaternionConv2d(4, 8), nn.ReLU(), nn.MaxPool2d(2), 
            ]
        else:
            layers = [
                nn.Conv2d(4, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), 
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), 
            ]

        self.net = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, num_classes))
    
    def forward(self, x):
        return self.net(x)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def subset(dataset, limit, seed):
    if limit is None or limit >= len(dataset):
        return dataset
    if limit < 1:
        raise ValueError("Subset size must be possitive")
    
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:limit].tolist()
    return Subset(dataset, indices)

def evaluate(model, loader, device, criterion):
    model.eval()
    targets, predictions = [], []
    total_loss, total_count = 0.0, 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            total_count += y.size(0)
            targets.extend(y.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    
    return {
        "loss": total_loss / total_count,
        "accuracy": accuracy_score(targets, predictions),
        "macro_f1": f1_score(targets, predictions, labels=list(range(9)), average="macro", zero_division=0), 
        "targets": targets, 
        "predictions": predictions, 
    }

def train_model(name, model, train_loader, val_loader, test_loader, device, args, outdir, labels):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    history = []
    best_f1 = -1.0
    best_path = outdir/f"{name.lower()}_best.pt"
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, count = 0.0, 0
        with Progress(SpinnerColumn(), TextColumn(f"{name} epoch {epoch}/{args.epochs}"), BarColumn(),
                      TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(), console=console) as progress:
            task = progress.add_task("train", total=len(train_loader))
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * y.size(0)
                count += y.size(0)
                progress.advance(task)
        val = evaluate(model, val_loader, device, criterion)
        row = {"epoch": epoch, "train_loss": total_loss / count, "val_loss": val["loss"], "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"]}
        history.append(row)
        console.print(f"  [bold]{name}[/bold] train loss={row['train_loss']:.4f}  "
                      f"val loss={row['val_loss']:.4f}  val accuracy={row['val_accuracy']:.2%}  "
                      f"val macro-F1={row['val_macro_f1']:.4f}")
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            torch.save(model.state_dict(), best_path)
    
    train_seconds = time.perf_counter() - started
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test = evaluate(model, test_loader, device, criterion)
    report = classification_report(test["targets"], test["predictions"], 
                                    labels=list(range(9)), target_names=labels, 
                                    zero_division=0, output_dict=True)
    matrix = confusion_matrix(test["targets"], test["predictions"], labels=list(range(9)))
    parameters = sum(p.numel() for p in model.parameters())
    console.print(Panel(f"[bold]{name}[/bold] | Accuracy: {test['accuracy']:.2%} | "
                        f"Macro-F1: {test['macro_f1']:.4f} | Parameters: {parameters:,} | "
                        f"Training: {train_seconds:.1f}s", title="Held-out test results"))

    detail = Table(title=f"{name}: per-class test metrics")
    for col in {"Class", "Precision", "Recall", "F1", "Support"}:
        detail.add_column(col, justify="left" if col == "Class" else "right")
    
    for label in labels:
        item = report[label]
        detail.add_row(label, f"{item['precision']:.3f}", f"{item['recall']:.3f}", 
                       f"{item['f1-score']:.3f}", str(int(item['support'])))
        
    console.print(detail)

    with(outdir / f"{name.lower()}_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with(outdir / f"{name.lower()}_report.json").open("w") as handle:
        json.dump(report, handle, indent=2)

    np.savetxt(outdir / f"{name.lower()}_confusion.csv", matrix, fmt="%d", delimiter=",")
    return {"name": name, "parameters": parameters, "train_seconds": train_seconds,
            "accuracy": test["accuracy"], "macro_f1": test["macro_f1"],
            "history": history, "matrix": matrix}


def save_plots(results, labels, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for result in results:
        epochs = [r["epoch"] for r in result["history"]]
        axes[0].plot(epochs, [r["train_loss"] for r in result["history"]], marker="o", label=result["name"] + " train")
        axes[0].plot(epochs, [r["val_loss"] for r in result["history"]], marker=".", linestyle="--", label=result["name"] + " validation")
        axes[1].plot(epochs, [r["val_macro_f1"] for r in result["history"]], marker="o", label=result["name"])
    axes[0].set(title="Training and validation loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[1].set(title="Validation macro-F1", xlabel="Epoch", ylabel="Macro-F1", ylim=(0, 1.05))

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "training_curves.png", dpi=180)
    plt.close(fig)

    for result in results:
        fig, ax = plt.subplots(figsize=(10, 8))
        matrix = result["matrix"]
        image = ax.imshow(matrix, cmap="Blues")
        ax.set(title=f"{result['name']} confusion matrix", xlabel="Predicted class", ylabel="True class")
        ax.set_xticks(range(9), labels=labels, rotation=60, ha="right")
        ax.set_yticks(range(9), labels=labels)
        for i in range(9):
            for j in range(9):
                ax.text(j, i, str([i, j]), ha="center", va="center", color="white" 
            if matrix[i, j] > matrix.max() / 2 else "black", fontsize=8 )
        
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(outdir / f"{result['name'].lower()} confusion.png", dpi=180)
        plt.close(fig)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    names = [r["name"] for r in results]
    for ax, field, title in zip(axes, ("accuracy", "macro_f1", "parameters"), 
                                ("Test accuracy", "Test macro-F1", "Trainable parameters")):
        values = [r[field] for r in results]
        bars = ax.bar(names, values)
        ax.set_title(title)
        ax.bar_label(bars, fmt="%.3f" if field != "parameters" else "%.0f")
        if field != "parameters":
            ax.set_ylim(0, 1.1)
    
    fig.tight_layout()
    fig.savefig(outdir / "model_comparison.png", dpi=180)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("result"))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs and batch-size should be possitive and lr > 0")
    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    labels = [INFO["pathmnist"]["label"][str(i)] for i in range(9)]
    console.print(Panel.fit("[bold cyan]CNN vs Quaternion CNN[/bold cyan]\nPathMNIST research benchmark — not clinical", border_style="cyan"))
    console.print(f"Device: [bold]{device}[/bold] | Seed: {args.seed} | Epochs: {args.epochs}")

    train = subset(FourChannelDataset("train"), args.max_train, args.seed)
    val = subset(FourChannelDataset("val"), args.max_val, args.seed + 1)
    test = subset(FourChannelDataset("test"), args.max_test, args.seed + 2)
    console.print(f"Samples: train={len(train): } | validation={len(val): } | test={len(test): }")
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=args.batch_size)
    test_loader = DataLoader(test, batch_size=args.batch_size)

    results = []
    for name, quaternion in (("CNN", False), ("QCNN", True)):
        set_seed(args.seed)
        model = Classifier(quaternion=quaternion)
        results.append(train_model(name, model, train_loader, val_loader, test_loader, 
                                  device, args, args.output, labels))
        
        comparison = Table(title="Final comparison (held-out test)")
        for col in ("Model", "Accuracy", "Macro-F1", "Parameters", "Training seconds"):
            comparison.add_column(col)
        for r in results:
            comparison.add_row(r["name"], f"{r['accuracy']:.2%}", f"{r['macro_f1']:.4f}",
                           f"{r['parameters']:,}", f"{r['train_seconds']:.1f}")
        
        console.print(comparison)
        save_plots(results, labels, args.output)
        with (args.output / "summary.json").open("w") as handle:
            json.dump([{k: v for k, v in r.items() if k not in ("history", "matrix")} 
                       for r in results], handle, indent=2)
        
        console.print(f"[green]Saved reports, charts, and checkpoints to {args.output.resolve()}[/green]")

if __name__=="__main__":
    main()