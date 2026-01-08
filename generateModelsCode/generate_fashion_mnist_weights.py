import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import logging
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(stream=sys.stdout, format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.INFO, datefmt='%I:%M:%S')

def get_fashion_mnist_loaders(batch_size=128):
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader

class MLP(nn.Module):
    def __init__(self, init_type='he', seed=None):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)
        
        if seed is not None:
            torch.manual_seed(seed)
        self.init_weights(init_type)

    def init_weights(self, init_type):
        for layer in [self.fc1, self.fc2, self.fc3]:
            if init_type == 'xavier':
                nn.init.xavier_uniform_(layer.weight)
            elif init_type == 'he':
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            else:
                nn.init.normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(model, train_loader, optimizer, criterion, device, scaler, max_grad_norm=1.0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():  # mixed precision
            out = model(xb)
            loss = criterion(out, yb)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds = out.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

    return total_loss / len(train_loader), correct / total


def evaluate(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            preds = model(xb).argmax(1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total


def train_model(seed, train_loader, test_loader, epochs=25, lr=1e-3, save_dir="fashion_mnist_models"):
    torch.manual_seed(seed)
    model = MLP(seed=seed).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0.0
    patience, patience_counter = 5, 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, scaler)
        val_acc = evaluate(model, test_loader, device)

        logging.info(f"Seed:{seed} Epoch:{epoch} Train Loss:{train_loss:.4f} "
                     f"Train Acc:{train_acc:.4f} Val Acc:{val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{save_dir}/mlp_seed{seed}.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch} for seed {seed}")
                break

    logging.info(f"Seed:{seed} Best Val Accuracy={best_acc:.4f}")


if __name__ == "__main__":
    train_loader, test_loader = get_fashion_mnist_loaders(batch_size=128)

    for seed in range(100):
        train_model(seed, train_loader, test_loader)
