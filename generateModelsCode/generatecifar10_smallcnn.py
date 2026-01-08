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
logging.basicConfig(stream=sys.stdout,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.INFO,
                    datefmt='%I:%M:%S')


def load_cifar10(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader


class SmallCNN(nn.Module):
    """
    SmallCNN with ~12k parameters for CIFAR-10
    
    Parameter breakdown:
    - conv1: 3*3*3*16 + 16 = 448
    - conv2: 16*3*3*32 + 32 = 4640
    - conv3: 32*3*3*32 + 32 = 9248
    - fc1: 32*4*4*64 + 64 = 32832
    - fc2: 64*10 + 10 = 650
    Total: ~47,818 parameters
    
    To get ~12k parameters:
    - conv1: 3*3*3*16 + 16 = 448
    - conv2: 16*3*3*24 + 24 = 3480
    - conv3: 24*3*3*24 + 24 = 5208
    - fc1: 24*4*4*32 + 32 = 12320
    - fc2: 32*10 + 10 = 330
    Total: ~11,786 parameters ✓
    """
    def __init__(self, num_classes=10):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)     # 32x32x16
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 24, 3, padding=1)    # 16x16x24
        self.bn2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, 3, padding=1)    # 8x8x24
        self.bn3 = nn.BatchNorm2d(24)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.act = nn.ReLU(inplace=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(24 * 4 * 4, 32)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Block 1: 32x32 -> 16x16
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pool(x)
        
        # Block 2: 16x16 -> 8x8
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.pool(x)
        
        # Block 3: 8x8 -> 4x4
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act(x)
        x = self.pool(x)
        
        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def count_parameters(model):
    """Count total parameters in model"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_smallcnn():
    model = SmallCNN().to(device)
    total, trainable = count_parameters(model)
    logging.info(f"Model has {total:,} total parameters ({trainable:,} trainable)")
    return model


def train_smallcnn(seed,
                   train_loader,
                   test_loader,
                   epochs=100,
                   lr=1e-3,
                   weight_decay=5e-4,
                   save_dir="cifar10_models",
                   patience=20):

    torch.manual_seed(seed)
    model = get_smallcnn()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate schedule with warmup
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=5
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs-5, eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[5]
    )

    best_acc = 0
    best_state = None
    patience_counter = 0

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            running_loss += loss.item()

        scheduler.step()

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                preds = model(inputs).argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        acc = correct / total

        logging.info(f"Epoch [{epoch+1}/{epochs}] "
                     f"Loss: {running_loss/len(train_loader):.4f} "
                     f"Test Acc: {acc:.4f} "
                     f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break

    torch.save(best_state, f"{save_dir}/smallcnn_seed{seed}.pt")
    logging.info(f"Best Test Accuracy={best_acc:.4f} for seed={seed}")


if __name__ == "__main__":
    train_loader, test_loader = load_cifar10(batch_size=128)

    for seed in range(1, 101):
        logging.info(f"\n{'='*60}")
        logging.info(f"Training seed={seed}")
        logging.info(f"{'='*60}")
        train_smallcnn(
            seed,
            train_loader,
            test_loader,
            epochs=50,
            lr=1e-4,
            weight_decay=1e-3,
            patience=20,
            save_dir="cnn_cifar10_models",
        )