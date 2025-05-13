import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import f_oneway
from advances_project.arc.fftmask import FFTMask
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_results(results):
    labels = [name for name, _ in results]
    means = [np.mean(accs) for _, accs in results]
    stds = [np.std(accs) for _, accs in results]

    plt.figure(figsize=(12, 6))
    plt.bar(labels, means, yerr=stds, capsize=5)
    plt.xticks(rotation=90, ha='right')
    plt.ylabel("Test Accuracy")
    plt.title("Accuracy per Frequency Band (Mean ± Std)")
    plt.tight_layout()
    plt.savefig("results.png")
    # plt.show()

class SyntheticTensorDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class EncodedDataset(Dataset):
    def __init__(self, encodings_path, labels_path, band, fftmask):
        self.encodings = torch.load(encodings_path)
        self.labels = torch.load(labels_path)
        self.band = band
        self.fftmask = fftmask

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.encodings[idx].unsqueeze(0).to(device)
        band_tensor = torch.tensor([self.band], device=device)
        masked = self.fftmask(x, provided_group_masks=band_tensor)
        return masked.squeeze(0), self.labels[idx].to(device)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc = nn.Linear(16 * 8 * 8, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class LinearClassifier(nn.Module):
    def __init__(self, input_shape=(4, 64, 64), num_classes=10):
        super().__init__()
        c, h, w = input_shape
        self.flattened_size = c * h * w
        self.fc = nn.Linear(self.flattened_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return nn.functional.softmax(x, dim=1)

def compute_loss_and_accuracy(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * targets.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
            total += targets.size(0)
    return total_loss / total, correct / total

def kfold_evaluate(dataset, in_model, k=2, epochs=5, lr=1e-3, batch_size=32, test_batch_size=64):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    indices = np.arange(len(dataset))
    test_accuracies = []

    for fold, (train_idx, test_idx) in enumerate(tqdm(kf.split(indices), desc="K-Fold Splits", total=k)):
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=test_batch_size, shuffle=False)

        model = in_model().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in tqdm(range(epochs), desc=f"Training Fold {fold+1}", leave=False):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        _, test_acc = compute_loss_and_accuracy(model, test_loader, criterion)
        test_accuracies.append(test_acc)

    return test_accuracies

if __name__ == "__main__":
    fftmask = FFTMask().to(device)
    k = 10
    epochs = 10

    band_lists = list(itertools.product([0, 1], repeat=6))  # 64 combinations

    results = []

    for band in band_lists:
        band_str = "".join(map(str, band))
        print(f"\n== Running k-fold on {band_str} ==")
        dataset = EncodedDataset(
            encodings_path="scripts/valid_encodings.pt",
            labels_path="scripts/valid_labels.pt",
            band=list(band),
            fftmask=fftmask
        )
        accs = kfold_evaluate(dataset, LinearClassifier, k=k, epochs=epochs, lr=1e-3)
        print(f"{band_str} accuracies: {accs}")
        results.append((band_str, accs))

    # Write all results to CSV at the end
    with open("accuracies_linear.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["band"] + [f"fold_{i+1}" for i in range(k)])
        for band_str, accs in results:
            writer.writerow([band_str] + accs)
