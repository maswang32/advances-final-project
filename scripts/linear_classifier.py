import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import ttest_ind

# --- Dataset Definition ---
class SyntheticTensorDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# --- Model ---
class LinearClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4 * 64 * 64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

# --- Evaluation ---
def compute_loss_and_accuracy(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += batch_size
    return total_loss / total, correct / total

# Doing k-fold validation might be overkill but it will allow us to do a statistical test between our different levels.
def kfold_evaluate(dataset, k=2, epochs=5, lr=1e-3, batch_size=32, test_batch_size=64):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    indices = np.arange(len(dataset))
    test_accuracies = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(indices)):
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=test_batch_size, shuffle=False)

        model = LinearClassifier()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for _ in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        _, test_acc = compute_loss_and_accuracy(model, test_loader, criterion)
        test_accuracies.append(test_acc)

    return test_accuracies

# These are just dummy data generators. The data is somewhat seperable for one and not seperable for the other.
def generate_separable_dataset(seed=42, num_samples_per_class=100):
    np.random.seed(seed)
    data = []
    labels = []
    for label in range(10):
        low = label / 10
        high = (label + 1) / 10
        for _ in range(num_samples_per_class):
            img = np.random.uniform(low, high, (4, 64, 64)).astype(np.float32)
            data.append(img)
            labels.append(label)
    return SyntheticTensorDataset(torch.tensor(data), torch.tensor(labels))

def generate_nonseparable_dataset(seed=1337, num_samples=1000):
    np.random.seed(seed)
    data = np.random.uniform(0, 1, size=(num_samples, 4, 64, 64)).astype(np.float32)
    labels = np.random.randint(0, 10, size=(num_samples,))
    return SyntheticTensorDataset(torch.tensor(data), torch.tensor(labels))

if __name__ == "__main__":
    k = 10
    epochs = 100

    # Mason: put in a list of datasets of each of the latents here. Expecting 64x64x4 with one of 10 labels.
    datasets = [
        ("Separable", generate_separable_dataset()),
        ("Non-separable", generate_nonseparable_dataset())
    ]

    results = []

    # Run k-fold CV on each dataset
    for name, dataset in datasets:
        print(f"\n== Running k-fold on {name} dataset ==")
        accs = kfold_evaluate(dataset, k=k, epochs=epochs)
        print(f"{name} accuracies: {accs}")
        results.append((name, accs))

    # Perform t-test
    accs_sep = results[0][1]
    accs_nonsep = results[1][1]
    t_stat, p_val = ttest_ind(accs_sep, accs_nonsep)

    print("\n== T-test results between separable and non-separable datasets ==")
    print(f"  t = {t_stat:.4f}")
    print(f"  p = {p_val:.4f}")
    if p_val < 0.05:
        print("  → Statistically significant difference in test accuracy.")
    else:
        print("  → No statistically significant difference in test accuracy.")
