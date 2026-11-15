
"""
Educational CNN vs quaternion CNN experiment on synthetic four-channel images
Not a clinical clasifier. Requires torch, numpy, scikit-learn
"""

import argparse
import random
from typing import Any
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix

class SynteticAtaxiaDataSet(Dataset):
    """Artificial spatial pattern, NOT patient MRI or a model of ataxia pathology"""
    def __init__(self, n=600, size=32, seed=42):
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[:size, :size]

        self.x = np.empty((n, 4, size, size), dtype=np.float32)
        self.y = np.empty(n, dtype=np.int64)

        for index in range(n):
            label = index % 2
            self.y[index] = label

            cx = size / 2 + rng.normal(0, 1)
            cy = size / 2 + rng.normal(0, 1)
            
            radius = (
                (5.0 if label else 9.0)
                + rng.normal(0, 0.5)
            )

            base = np.exp(
                -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * radius ** 2 )
            )

            for ch in range(4):
                shifted = np.roll(base, shift=ch - 1, axis=1)

                self.x[index, ch] = (
                    (1 + 0.12 * ch) * shifted
                    + rng.normal(0, 0.12, (size, size))
                )

        order = rng.permutation(n)

        self.x = self.x[order]
        self.y = self.y[order]

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return (
            torch.from_numpy(self.x[index]), 
            int(self.y[index])
        )
    

#=======================================================
#2. Quaternion Convolution
#=======================================================
class QuaternionConv2d(nn.Module):
    """
    Hamilton product convolution, 
    Input: 4* in_q real channels
    Output: 4* out_q real channels
    """

    def __init__(self, in_q, out_q, kernel_size=3, padding=1):
        super().__init__()

        kw = dict(kernel_size=kernel_size, padding=padding, bias=False)

        self.wr = nn.Conv2d(in_q, out_q, **kw)
        self.wi = nn.Conv2d(in_q, out_q, **kw)
        self.wj = nn.Conv2d(in_q, out_q, **kw)
        self.wk = nn.Conv2d(in_q, out_q, **kw)

        self.bias = nn.Parameter(
            torch.zeros(4 * out_q)
        )

    def forward(self, x):
        r, i, j, k = torch.chunk(x, 4, dim=1)

        # Hamilton product
        yr = (
            self.wr(r) - self.wi(i) - self.wj(j) - self.wk(k)
        )

        yi = (
            self.wr(i) + self.wi(r) + self.wj(k) - self.wk(j)
        )

        yj = (
            self.wr(j) - self.wi(k) + self.wj(r) + self.wk(i)
        )

        yk = (
            self.wr(k) + self.wi(j) - self.wj(i) + self.wk(r)
        )

        output = torch.cat(
            [yr, yi, yj, yk], dim=1
        )

        return (
            output + self.bias[None, :, None, None]
        )


#=============================================
# QUATERNION CLASSIFIER
#=============================================

class Classifier(nn.Module):

    def __init__(self, quaternion=False):
        super().__init__()

        if quaternion:
            layers = [
                QuaternionConv2d(1, 4), 
                nn.ReLU(), 
                nn.MaxPool2d(2),

                QuaternionConv2d(4, 8), 
                nn.ReLU(), 
                nn.MaxPool2d(2)
            ]

            channels = 32
        else:
            layers = [
                nn.Conv2d(4, 16, 3, padding=1), 
                nn.ReLU(), 
                nn.MaxPool2d(2),

                nn.Conv2d(16, 32, 3, padding=1), 
                nn.ReLU(), 
                nn.MaxPool2d(2)
            ]

            channels = 32
        
        self.net = nn.Sequential(
            *layers, 
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(), 
            nn.Linear(channels, 2)
        )
    
    def forward(self, x):
        return self.net(x)
    
#------------------------------------------
# 4. TRAINING AND EVALUATION
#------------------------------------------

def run(
        name, 
        model, 
        train_loader, 
        test_loader, 
        device, 
        epochs, 
        lr
):
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=lr
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        seen = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            seen += len(y)
        
        print(
            f"{name} epoch {epoch + 1}/{epochs}"
            f"train loss={total_loss/seen:.4f}"
        )
    # Evaluate on held-out synthetic data
    model.eval()

    truth = []
    predictions = []

    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x.to(device))

            pred = (
                logits.argmax(dim=1)
                .cpu()
                .tolist()
            )

            truth.extend(y.tolist())
            predictions.extend(pred)
    
    print(f"\n{name} classifications report: ")

    print(
        classification_report(
            truth, 
            predictions, 
            labels=[0, 1], 
            target_names=[
                "synthetic_control", 
                "synthetic_pattern"
            ], 
            zero_division=0
        )
    )

    print(
        "Confusion matrix:\n", 
        confusion_matrix(
            truth, 
            predictions, 
            labels=[0,1]
        )
    )

    return model

#==============================================
# MAIN EXPERIMENT
#==============================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs", 
        type=int, 
        default=8
    )

    parser.add_argument(
        "--samples", 
        type=int, 
        default=600
    )

    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32
    )

    parser.add_argument(
        "--lr", 
        type=float,
        default=0.01
    )

    parser.add_argument(
        "--seed",
        type=int, 
        default=42
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = SynteticAtaxiaDataSet(
        n=args.samples, 
        seed=args.seed
    )

    n_train = int(0.8*len(dataset))

    train_data, test_data = random_split(
        dataset, 
        [n_train, len(dataset) - n_train], 
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Device", device)
    print("Train samples", len(train_data))
    print("Test samples", len(test_data))

    # Compare both arquitectures.
    for name, quaternion in [("CNN", False), ("QCNN", "True")]:
        torch.manual_seed(args.seed)

        model = Classifier(quaternion=quaternion)

        parameters = sum(
            p.numel()
            for p in model.parameters()
        )

        print(
            f"\n{name} parameters: {parameters}"
        )

        run (
            name, model, train_loader, test_loader, device, args.epochs, args.lr
        )

if __name__ == "__main__":
    main()

        