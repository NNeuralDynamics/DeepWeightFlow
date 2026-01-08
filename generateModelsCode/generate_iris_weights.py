import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import os

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=16)

class MLP(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16, output_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def init_weights(m, init_type="xavier"):
    if isinstance(m, nn.Linear):
        if init_type == "xavier":
            nn.init.xavier_uniform_(m.weight)
        elif init_type == "kaiming":
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        elif init_type == "normal":
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif init_type == "uniform":
            nn.init.uniform_(m.weight, -0.1, 0.1)
        elif init_type == "zeros":
            nn.init.constant_(m.weight, 0.0)
        elif init_type == "default":
            pass
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def train_model(init_type, seed, train_loader, test_loader, epochs=100, lr=0.001, save_dir="iris_models"):
    torch.manual_seed(seed)
    model = MLP()
    model.apply(lambda m: init_weights(m, init_type))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb).argmax(1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    acc = correct / total
    print(f"Init={init_type}, Seed={seed}: Test Accuracy={acc:.4f}")

    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{save_dir}/mlp_{init_type}_seed{seed}.pt")

if __name__ == "__main__":
    inits = ["xavier", "he", "normal", "uniform", "default"]
    for init_type in inits:
        for seed in range(20):
            train_model(init_type, seed, train_loader, test_loader)

