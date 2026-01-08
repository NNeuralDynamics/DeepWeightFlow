import os
from collections import defaultdict
from torch.utils.data import Subset
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from torchvision import datasets, transforms
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr
from transformers import BertTokenizer
from datasets import load_dataset

class Bunch:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def safe_deflatten(flat, batch_size, starts, ends):
    """Safely deflatten a tensor without index errors"""
    parts = []
    actual_batch_size = flat.size(0)
    
    safe_batch_size = min(actual_batch_size, batch_size)
    
    for i in range(safe_batch_size):
        batch_parts = []
        for si, ei in zip(starts, ends):
            if si < ei: 
                batch_parts.append(flat[i][si:ei])
        parts.append(batch_parts)
    
    return parts

def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class YelpReviewDataset(Dataset):
    def __init__(self, split='train', max_length=128, subset_size=10000):
        raw_data = load_dataset("yelp_review_full", split=split)
        self.data = raw_data.select(range(min(len(raw_data), subset_size)))
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        label = torch.tensor(item['label'] / 4.0, dtype=torch.float32)
        return input_ids, label

def load_cifar10(batch_size=128):

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return test_loader


def get_cifar100_loaders(batch_size=128, few_shot=False, num_samples_per_class=50):
    """CIFAR-100 (target domain for transfer learning)"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])

    train_ds = datasets.CIFAR100(root="data", train=True, download=True, transform=transform_train)
    test_ds = datasets.CIFAR100(root="data", train=False, download=True, transform=transform_test)
    
    if few_shot:
        class_indices = defaultdict(list)
        for idx in range(len(train_ds)):
            _, label = train_ds[idx]
            class_indices[label].append(idx)
        
        few_shot_indices = []
        for class_id in range(100):
            if class_id in class_indices:
                few_shot_indices.extend(class_indices[class_id][:num_samples_per_class])
        
        train_ds = Subset(train_ds, few_shot_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"CIFAR-100 loaded: {len(train_ds)} training samples, {len(test_ds)} test samples")
    return train_loader, test_loader

def load_mnist(batch_size=32):
    transform = transforms.Compose([transforms.ToTensor()])
    test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return test_loader

def load_fashion_mnist(batch_size=32):
    transform = transforms.Compose([transforms.ToTensor()])
    test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return test_loader

def load_iris_dataset(batch_size=32):
    iris = load_iris()
    X, y = iris.data, iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    test_ds = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    return test_loader

def evaluate_model(model, test_loader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return 100 * correct / total

def test_ensemble(models, test_loader, device="cuda"):
    """Test ensemble of models"""
    for m in models:
        m.eval()
        m.to(device)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            logits_sum = None
            for model in models:
                output = model(data)
                if logits_sum is None:
                    logits_sum = output
                else:
                    logits_sum += output
            
            avg_logits = logits_sum / len(models)
            _, predicted = torch.max(avg_logits, 1)
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return 100 * correct / total

def print_stats(models, test_loader, device):
    """Print statistics for generated models"""
    accuracies = []
    for model in models:
        acc = evaluate_model(model, test_loader, device)
        accuracies.append(acc)
    
    accuracies = np.array(accuracies)
    mean = accuracies.mean()
    std = accuracies.std()
    min_acc = accuracies.min()
    max_acc = accuracies.max()
    
    print("\n=== Summary ===")
    print(f"Average Accuracy: {mean:.2f}% ± {std:.2f}%")
    print(f"Min Accuracy: {min_acc:.2f}%")
    print(f"Max Accuracy: {max_acc:.2f}%")
    
    return mean, std

def print_regression_stats(model, loader, device):
    """
    Evaluate model and return MAE, R², and Spearman correlation
    """
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for input_ids, y in loader:
            input_ids, y = input_ids.to(device), y.to(device)
            out = model(input_ids).squeeze(-1)
            preds.extend(out.cpu().numpy())
            labels.extend(y.cpu().numpy())
    
    preds = np.array(preds)
    labels = np.array(labels)
    
    # Calculate metrics
    r2 = r2_score(labels, preds)
    mae = mean_absolute_error(labels, preds)
    spearman_corr, spearman_pval = spearmanr(labels, preds)
    
    return {
        'r2': r2,
        'mae': mae,
        'spearman': spearman_corr,
        'spearman_pval': spearman_pval
    }


def recalibrate_bn_stats(model, device='cuda', print_stats=False):
    """Recalculate BatchNorm statistics for generated weights"""
    model.train()
    model.to(device)
    test_loader = load_cifar10(batch_size=128)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
    
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            _ = model(inputs)
    
    if print_stats:
        for name, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                print(f"{name}: mean={m.running_mean.mean().item():.4f}, "
                      f"var={m.running_var.mean().item():.4f}, "
                      f"num_batches_tracked={m.num_batches_tracked.item()}")
    

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.1
    
    return model

def get_fewshot_loaders(dataset_name='STL10', batch_size=32, num_samples_per_class=5, num_classes=10, few_shot=False):
    """
    Load datasets (STL-10 or SVHN). If few_shot=True, returns few-shot training set.
    """
    if dataset_name.upper() == 'STL10':
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4467, 0.4398, 0.4066),
                                 (0.2241, 0.2215, 0.2239))
        ])
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4467, 0.4398, 0.4066),
                                 (0.2241, 0.2215, 0.2239))
        ])
        train_dataset = datasets.STL10(root='./data', split='train', download=True, transform=transform_train)
        test_dataset = datasets.STL10(root='./data', split='test', download=True, transform=transform_test)

    elif dataset_name.upper() == 'SVHN':
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728),
                                 (0.1980, 0.2010, 0.1970))
        ])
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728),
                                 (0.1980, 0.2010, 0.1970))
        ])
        train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
        test_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if few_shot:
        # Build few-shot subset
        class_indices = defaultdict(list)
        for idx in range(len(train_dataset)):
            _, label = train_dataset[idx]
            label = label % 10 if dataset_name.upper() == 'SVHN' else label  # fix SVHN 10->0
            class_indices[label].append(idx)

        few_shot_indices = []
        for class_id in range(num_classes):
            if class_id in class_indices:
                few_shot_indices.extend(class_indices[class_id][:num_samples_per_class])

        train_dataset = Subset(train_dataset, few_shot_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"{dataset_name} loaded: {len(train_loader.dataset)} training samples, {len(test_loader.dataset)} test samples")
    return train_loader, test_loader

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
    
    train_ds = datasets.CIFAR10(root="data", train=True, download=True, transform=transform_train)
    test_ds = datasets.CIFAR10(root="data", train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, test_loader

class MemoryMappedLatentDataset(Dataset):
    """
    Dataset that reads from memory-mapped file (for target data).
    
    Efficiently loads PCA-projected latent codes from disk without loading
    entire dataset into memory. Uses numpy's memory mapping for fast access.
    
    Args:
        mmap_file: Path to memmap file created by PCA.transform()
        shape: Tuple of (n_samples, n_features) - REQUIRED for raw memmap files
        device: Device to load tensors to ('cpu' or 'cuda')
        dtype: Data type (default: np.float32)
        preload: If True, load all data to RAM (faster but uses more memory)
    """
    
    def __init__(self, mmap_file, shape, device='cpu', dtype=np.float32, preload=False):
        if not os.path.exists(mmap_file):
            raise FileNotFoundError(f"Memmap file not found: {mmap_file}")
        
        self.mmap_file = mmap_file
        self.device = device
        self.preload = preload
        
        # Load with memory mapping (raw binary file, not .npy)
        self.data = np.memmap(
            mmap_file,
            dtype=dtype,
            mode='r',
            shape=shape
        )
        
        # Validate shape
        if self.data.ndim != 2:
            raise ValueError(
                f"Expected 2D array (n_samples, n_features), got shape {self.data.shape}"
            )
        
        self.n_samples, self.n_features = self.data.shape
        
        # Optional: preload to RAM for faster access (if dataset is small)
        if preload:
            print(f"Preloading {self.n_samples} samples to RAM...")
            self.data = np.array(self.data, dtype=dtype)
            print("✓ Preload complete")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """
        Get a single sample or batch of samples.
        
        Args:
            idx: Integer index or slice
            
        Returns:
            Torch tensor on specified device
        """
        # Handle slicing for batch loading
        if isinstance(idx, slice):
            # Load slice from memmap (efficient!)
            sample = np.array(self.data[idx], dtype=np.float32)
        else:
            # Load single sample
            sample = np.array(self.data[idx], dtype=np.float32)
        
        return torch.from_numpy(sample).to(self.device)
    
    def get_batch(self, indices):
        """
        Get multiple samples by index list (more efficient than multiple __getitem__ calls).
        
        Args:
            indices: List or array of indices
            
        Returns:
            Torch tensor of shape (len(indices), n_features)
        """
        batch = np.array(self.data[indices], dtype=np.float32)
        return torch.from_numpy(batch).to(self.device)
    
    @property
    def shape(self):
        """Return dataset shape (n_samples, n_features)"""
        return self.data.shape
    
    def __repr__(self):
        return (f"MemoryMappedLatentDataset(n_samples={self.n_samples}, "
                f"n_features={self.n_features}, device='{self.device}', "
                f"dtype={self.data.dtype}, preload={self.preload})")


class StreamingLatentDataset(Dataset):
    """
    Generate noise on-the-fly (for source data).
    
    Creates random latent codes from Gaussian distribution without storing them.
    Memory-efficient for source distribution in flow matching.
    
    Args:
        n_samples: Number of samples in epoch
        latent_dim: Dimensionality of latent space
        std: Standard deviation of Gaussian noise
        device: Device to generate tensors on
        seed: Random seed for reproducibility (None = random each time)
    """
    
    def __init__(self, n_samples, latent_dim, std=0.001, device='cpu', seed=None):
        self.n_samples = n_samples
        self.latent_dim = latent_dim
        self.std = std
        self.device = device
        self.seed = seed
        
        if seed is not None:
            self.generator = torch.Generator(device=device)
            self.generator.manual_seed(seed)
        else:
            self.generator = None
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        if self.generator is not None:
            return torch.randn(
                self.latent_dim, 
                device=self.device, 
                generator=self.generator
            ) * self.std
        else:
            return torch.randn(self.latent_dim, device=self.device) * self.std
    
    def __repr__(self):
        return (f"StreamingLatentDataset(n_samples={self.n_samples}, "
                f"latent_dim={self.latent_dim}, std={self.std}, "
                f"device='{self.device}', seed={self.seed})")