import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import sys
import logging
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(stream=sys.stdout, format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%I:%M:%S')

def get_cifar10_loaders(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    
    train_ds = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=transform_train)
    test_ds = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, test_loader

def get_resnet18(num_classes=10, pretrained=False):
    model = torchvision.models.resnet18(weights=None if not pretrained else "IMAGENET1K_V1")
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

def train_resnet18(seed, train_loader, test_loader, epochs=100, lr=0.1, save_dir="cifar10_models", accum_steps=4, patience=20):
    torch.manual_seed(seed)
    model = get_resnet18()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0
    best_model_state = None
    patience_counter = 0
    
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        
        for i, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets) / accum_steps
            loss.backward()
            running_loss += loss.item() * accum_steps  # accumulate real loss
            
            if (i + 1) % accum_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
        
        scheduler.step()
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                preds = model(inputs).argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        acc = correct / total
        logging.info(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f}, Test Accuracy: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    torch.save(best_model_state, f"{save_dir}/resnet18_seed{seed}.pt")
    logging.info(f"Best Test Accuracy={best_acc:.4f} for seed={seed}")

if __name__ == "__main__":
    train_loader, test_loader = get_cifar10_loaders(batch_size=128)
    
    for seed in range(100): 
        logging.info(f"Training seed={seed}")
        train_resnet18(seed, train_loader, test_loader, epochs=100, lr=0.1, accum_steps=4, patience=20)
