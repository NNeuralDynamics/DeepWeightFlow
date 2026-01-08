import sys
import os
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
import copy
import traceback
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import IncrementalPCA
import pandas as pd
from tabulate import tabulate

from utils import (Bunch, count_parameters, evaluate_model, recalibrate_bn_stats,
                  get_fewshot_loaders, get_cifar10_loaders, get_cifar100_loaders)
from models import get_resnet18
from weight_space_objects import WeightSpaceObjectResnet
from flow_matching import FlowMatching, WeightSpaceFlowModel
from canonicalization import get_permuted_models_data, aggressive_cleanup

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_file='constants.json'):
    """Load configuration from JSON file"""
    with open(config_file, 'r') as f:
        return json.load(f)


def replace_fc_layer(model: nn.Module, num_classes: int = 100) -> nn.Module:
    """
    Replace the final FC layer for transfer learning to different number of classes.
    
    Args:
        model: Pre-trained model
        num_classes: Number of classes in target dataset
        
    Returns:
        Model with replaced final layer
    """
    model_copy = copy.deepcopy(model)
    in_features = model_copy.fc.in_features
    model_copy.fc = nn.Linear(in_features, num_classes)
    
    # Initialize new FC layer
    nn.init.kaiming_normal_(model_copy.fc.weight)
    nn.init.constant_(model_copy.fc.bias, 0)
    
    return model_copy.to(device)


def convert_models_to_weight_space(models: List[nn.Module], exclude_bn: bool = True) -> List[WeightSpaceObjectResnet]:
    """
    Convert PyTorch models to weight space objects.
    
    Args:
        models: List of PyTorch models
        exclude_bn: Whether to exclude BatchNorm parameters
        
    Returns:
        List of WeightSpaceObjectResnet
    """
    weight_space_objects = []
    
    for model in tqdm(models, desc="Converting to weight space"):
        weights, biases = [], []
        weight_shapes, bias_shapes = [], []
        
        for name, param in model.named_parameters():
            # Skip BatchNorm if requested
            if exclude_bn and "bn" in name:
                continue
                
            param = param.detach().to(device)
            
            # Check for invalid values
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"Warning: NaN/Inf in {name}, replacing with zeros")
                param = torch.zeros_like(param)
            
            if "weight" in name:
                weights.append(param.clone())
                weight_shapes.append(param.shape)
            elif "bias" in name:
                biases.append(param.clone())
                bias_shapes.append(param.shape)
        
        wso = WeightSpaceObjectResnet(weights, biases)
        wso.weight_shapes = weight_shapes
        wso.bias_shapes = bias_shapes
        weight_space_objects.append(wso)
    
    return weight_space_objects


def train_weight_space_flow(sourceloader, targetloader, input_dim: int, 
                           hidden_dim: int, model_config: Dict) -> FlowMatching:
    """
    Train flow matching model.
    
    Args:
        sourceloader: DataLoader for source distribution
        targetloader: DataLoader for target distribution
        input_dim: Dimension of weight space (or latent space if PCA)
        hidden_dim: Hidden dimension for flow model
        model_config: Configuration dictionary
        
    Returns:
        Trained FlowMatching object
    """
    flow_model = WeightSpaceFlowModel(
        input_dim, 
        hidden_dim,
        time_embed_dim=model_config.get('time_embed_dim', 128),
        dropout=model_config.get('dropout', 0.1)
    ).to(device)
    
    print(f"Flow model parameters: {count_parameters(flow_model):,}")
    
    cfm = FlowMatching(
        sourceloader=sourceloader,
        targetloader=targetloader,
        model=flow_model,
        mode="velocity",
        t_dist=model_config.get('t_dist', 'uniform'),
        device=device
    )
    
    optimizer = torch.optim.AdamW(
        flow_model.parameters(),
        lr=model_config.get('lr', 5e-4),
        weight_decay=model_config.get('weight_decay', 1e-5),
        betas=(0.9, 0.95)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=model_config.get('n_iters', 30000), 
        eta_min=1e-6
    )
    
    train_kwargs = {
        'n_iters': model_config.get('n_iters', 30000),
        'optimizer': optimizer,
        'scheduler': scheduler,
        'sigma': model_config.get('sigma', 0.001),
        'patience': model_config.get('patience', 100),
        'log_freq': 10
    }
    
    # Gradient accumulation (if configured)
    grad_accum_steps = model_config.get('gradient_accumulation_steps')
    if grad_accum_steps is not None and grad_accum_steps > 1:
        train_kwargs['accum_steps'] = grad_accum_steps
        print(f"Using gradient accumulation: {grad_accum_steps} steps")
    
    print("\nTraining flow model...")
    cfm.train(**train_kwargs)
    
    return cfm


def generate_models_from_flow(cfm: FlowMatching, n_samples: int, actual_dim: int,
                              weight_shapes: List, bias_shapes: List,
                              source_std: float, model_config: Dict,
                              ipca: Optional[IncrementalPCA] = None,
                              exclude_bn: bool = True) -> List[nn.Module]:
    """
    Generate new models using trained flow matching.
    
    Args:
        cfm: Trained FlowMatching object
        n_samples: Number of models to generate
        actual_dim: Dimension of latent/weight space
        weight_shapes: List of weight tensor shapes
        bias_shapes: List of bias tensor shapes
        source_std: Standard deviation of source noise
        model_config: Configuration dictionary
        ipca: Optional PCA object for inverse transform
        exclude_bn: Whether BatchNorm was excluded
        
    Returns:
        List of generated PyTorch models
    """
    print(f"\nGenerating {n_samples} new models...")
    generated_models = []
    
    # Get reference loader for BN recalibration
    cifar10_train, _ = get_cifar10_loaders(batch_size=128)
    
    # Generate in chunks for memory efficiency
    chunk_size = 10
    for chunk_start in tqdm(range(0, n_samples, chunk_size), desc="Generating"):
        chunk_end = min(chunk_start + chunk_size, n_samples)
        chunk_n = chunk_end - chunk_start
        
        try:
            # Sample from source distribution
            random_flat = torch.randn(chunk_n, actual_dim, device=device) * source_std
            
            # Map through flow
            new_weights_flat = cfm.map(
                random_flat,
                n_steps=model_config.get('integration_steps', 100),
                method=model_config.get('integration_method', 'rk4')
            )
            
            # Inverse PCA if used
            if ipca is not None:
                new_weights_flat = ipca.inverse_transform(new_weights_flat.cpu().numpy())
                new_weights_flat = torch.tensor(new_weights_flat, dtype=torch.float32, device=device)
            
            # Reconstruct models
            for i in range(chunk_n):
                new_wso = WeightSpaceObjectResnet.from_flat(
                    new_weights_flat[i],
                    weight_shapes=weight_shapes,
                    bias_shapes=bias_shapes,
                    device=device
                )
                
                model = get_resnet18(num_classes=10).to(device)
                
                # Apply weights (excluding BatchNorm)
                weight_idx, bias_idx = 0, 0
                for name, param in model.named_parameters():
                    if exclude_bn and "bn" in name:
                        continue
                        
                    if "weight" in name:
                        param.data = new_wso.weights[weight_idx].clone()
                        weight_idx += 1
                    elif "bias" in name:
                        param.data = new_wso.biases[bias_idx].clone()
                        bias_idx += 1
                
                # Recalibrate BatchNorm statistics
                if model_config.get('recalibrate_bn', True):
                    model = recalibrate_bn_stats(model, device=device, print_stats=False)
                
                generated_models.append(model)
            
            del random_flat, new_weights_flat
            aggressive_cleanup()
            
        except Exception as e:
            print(f"Error generating chunk {chunk_start}: {e}")
            continue
    
    print(f"Successfully generated {len(generated_models)} models")
    return generated_models


def select_top_models(models: List[nn.Module], num_top: int = 5, 
                     metric_loader = None) -> Tuple[List[nn.Module], List[float]]:
    """
    Select top-performing models based on CIFAR-10 accuracy.
    
    Args:
        models: List of models to evaluate
        num_top: Number of top models to select
        metric_loader: DataLoader for evaluation (None = use CIFAR-10 test)
        
    Returns:
        Tuple of (top_models, top_accuracies)
    """
    print(f"\n{'='*70}")
    print(f"Selecting Top {num_top} Models Based on CIFAR-10 Performance")
    print(f"{'='*70}")
    
    if metric_loader is None:
        _, metric_loader = get_cifar10_loaders(batch_size=128)
    
    model_scores = []
    for i, model in enumerate(tqdm(models, desc="Evaluating models")):
        model_eval = copy.deepcopy(model)
        acc = evaluate_model(model_eval, metric_loader, device)
        model_scores.append((i, acc, model))
        print(f"  Model {i}: {acc:.2f}%")
        del model_eval
    
    # Sort by accuracy (descending)
    model_scores.sort(key=lambda x: x[1], reverse=True)
    
    top_models = [score[2] for score in model_scores[:num_top]]
    top_accs = [score[1] for score in model_scores[:num_top]]
    top_indices = [score[0] for score in model_scores[:num_top]]
    
    # Print summary
    all_accs = [score[1] for score in model_scores]
    print(f"\n{'='*70}")
    print(f"Selection Summary:")
    print(f"{'='*70}")
    print(f"  Total models evaluated: {len(models)}")
    print(f"  Best accuracy:          {max(all_accs):.2f}%")
    print(f"  Worst accuracy:         {min(all_accs):.2f}%")
    print(f"  Mean accuracy:          {np.mean(all_accs):.2f}%")
    print(f"  Std accuracy:           {np.std(all_accs):.2f}%")
    print(f"  Top {num_top} mean:     {np.mean(top_accs):.2f}%")
    print(f"\nTop {num_top} Models:")
    for rank, (idx, acc) in enumerate(zip(top_indices, top_accs), 1):
        print(f"  Rank {rank}: Model {idx} - {acc:.2f}%")
    print(f"{'='*70}")
    
    return top_models, top_accs


def finetune_model(model: nn.Module, train_loader, test_loader, 
                  epochs: int = 10, lr: float = 1e-4, 
                  freeze_ratio: float = 0.0, device='cuda') -> Tuple[nn.Module, float]:
    """
    Finetune model on target dataset.
    
    Args:
        model: Model to finetune
        train_loader: Training data
        test_loader: Test data
        epochs: Number of training epochs
        lr: Learning rate
        freeze_ratio: Ratio of early layers to freeze (0.0 = train all, 1.0 = freeze all)
        device: Computation device
        
    Returns:
        Tuple of (finetuned_model, best_accuracy)
    """
    model = model.to(device)

    # Freeze layers based on freeze_ratio
    all_layers = [
        model.conv1, model.bn1, model.layer1, 
        model.layer2, model.layer3, model.layer4
    ]
    num_to_freeze = int(len(all_layers) * freeze_ratio)

    for layer in all_layers[:num_to_freeze]:
        for param in layer.parameters():
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, 
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        train_acc = 100.0 * train_correct / train_total
        train_loss /= train_total
        scheduler.step()

        # Evaluation phase
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        test_acc = 100.0 * test_correct / test_total
        best_acc = max(best_acc, test_acc)

        print(f"  Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}% | Best: {best_acc:.2f}%")

    return model, best_acc


def evaluate_transfer_learning(pretrained_models: List[nn.Module], 
                              generated_models: List[nn.Module],
                              num_models_to_test: int = 5,
                              epochs_list: List[int] = [0, 1, 5, 10],
                              target_dataset: str = 'cifar100',
                              few_shot_samples: int = 50,
                              device: str = 'cuda') -> Dict:
    """
    Evaluate transfer learning performance.
    
    Compares three approaches:
    1. Random Initialization (baseline)
    2. CIFAR-10 Pretrained (original models)
    3. Flow Generated (from flow matching)
    
    Args:
        pretrained_models: Original pretrained models
        generated_models: Models generated via flow matching
        num_models_to_test: Number of models to evaluate per method
        epochs_list: List of finetuning epochs to test
        target_dataset: Target dataset name ('cifar100', 'stl10', 'svhn')
        few_shot_samples: Samples per class for few-shot learning
        device: Computation device
        
    Returns:
        Dictionary of results
    """
    results = {
        'Epochs': [],
        'Method': [],
        f'{target_dataset.upper()} Accuracy': []
    }
    
    # Load target dataset
    if target_dataset.lower() == 'cifar100':
        train_loader, test_loader = get_cifar100_loaders(
            batch_size=128, 
            few_shot=True, 
            num_samples_per_class=few_shot_samples
        )
        num_classes = 100
    elif target_dataset.lower() == 'stl10':
        train_loader, test_loader = get_fewshot_loaders(
            dataset_name='STL10',
            batch_size=128,
            few_shot=True,
            num_samples_per_class=few_shot_samples
        )
        num_classes = 10
    elif target_dataset.lower() == 'svhn':
        train_loader, test_loader = get_fewshot_loaders(
            dataset_name='SVHN',
            batch_size=128,
            few_shot=True,
            num_samples_per_class=few_shot_samples
        )
        num_classes = 10
    else:
        raise ValueError(f"Unknown target dataset: {target_dataset}")
    
    print(f"Transfer Learning Evaluation: CIFAR-10 → {target_dataset.upper()}")
    print(f"Few-shot: {few_shot_samples} samples/class, Total: {len(train_loader.dataset)} samples")
    
    # Evaluate at different finetuning epochs
    for epochs in epochs_list:
        print(f"\n{'='*70}")
        print(f"Evaluating with {epochs} epochs of finetuning")
        print(f"{'='*70}")
        
        # 1. Random Initialization Baseline
        print("\n--- Random Initialization ---")
        random_accs = []
        
        for i in range(num_models_to_test):
            model = get_resnet18(num_classes=num_classes)
            torch.manual_seed(42 + i)
            
            # Kaiming initialization
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
            
            if epochs > 0:
                _, acc = finetune_model(
                    copy.deepcopy(model), 
                    train_loader, 
                    test_loader, 
                    epochs=epochs, 
                    lr=1e-3,
                    freeze_ratio=0.0,
                    device=device
                )
            else:
                # Zero-shot evaluation
                acc = evaluate_model(model, test_loader, device)
            
            random_accs.append(acc)
            print(f"  Model {i}: {acc:.2f}%")
        
        results['Epochs'].append(epochs)
        results['Method'].append('RandomInit')
        results[f'{target_dataset.upper()} Accuracy'].append(
            f"{np.mean(random_accs):.2f} ± {np.std(random_accs):.2f}"
        )
        
        # 2. CIFAR-10 Pretrained (Original Models)
        print("\n--- CIFAR-10 Pretrained (Original) ---")
        pretrained_accs = []
        
        for i in range(min(num_models_to_test, len(pretrained_models))):
            # Replace FC layer for target number of classes
            model = replace_fc_layer(pretrained_models[i], num_classes=num_classes)
            
            if epochs > 0:
                _, acc = finetune_model(
                    model, 
                    train_loader, 
                    test_loader, 
                    epochs=epochs, 
                    lr=1e-3,
                    freeze_ratio=0.0,
                    device=device
                )
            else:
                # Zero-shot evaluation
                acc = evaluate_model(model, test_loader, device)
            
            pretrained_accs.append(acc)
            print(f"  Model {i}: {acc:.2f}%")
        
        results['Epochs'].append(epochs)
        results['Method'].append('CIFAR10-Pretrained')
        results[f'{target_dataset.upper()} Accuracy'].append(
            f"{np.mean(pretrained_accs):.2f} ± {np.std(pretrained_accs):.2f}"
        )
        
        # 3. Flow-Generated Models
        print("\n--- Flow-Generated Models ---")
        generated_accs = []
        
        for i in range(min(num_models_to_test, len(generated_models))):
            # Replace FC layer for target number of classes
            model = replace_fc_layer(generated_models[i], num_classes=num_classes)
            
            if epochs > 0:
                _, acc = finetune_model(
                    model, 
                    train_loader, 
                    test_loader, 
                    epochs=epochs, 
                    lr=1e-3,
                    freeze_ratio=0.0,
                    device=device
                )
            else:
                # Zero-shot evaluation
                acc = evaluate_model(model, test_loader, device)
            
            generated_accs.append(acc)
            print(f"  Model {i}: {acc:.2f}%")
        
        results['Epochs'].append(epochs)
        results['Method'].append('FlowGenerated')
        results[f'{target_dataset.upper()} Accuracy'].append(
            f"{np.mean(generated_accs):.2f} ± {np.std(generated_accs):.2f}"
        )
    
    return results


def print_transfer_summary(results: Dict, target_dataset: str):
    """
    Print formatted summary of transfer learning results.
    
    Args:
        results: Results dictionary from evaluate_transfer_learning
        target_dataset: Name of target dataset
    """
    df = pd.DataFrame(results)
    
    print(f"\n{'='*100}")
    print(f"Transfer Learning Results: CIFAR-10 → {target_dataset.upper()}")
    print(f"{'='*100}")
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    print(f"{'='*100}")
    
    print(f"\n{'='*100}")
    print(f"Performance Improvement Over Random Initialization")
    print(f"{'='*100}")
    
    for epoch in df['Epochs'].unique():
        epoch_data = df[df['Epochs'] == epoch]
        
        random_row = epoch_data[epoch_data['Method'] == 'RandomInit'].iloc[0]
        pretrained_row = epoch_data[epoch_data['Method'] == 'CIFAR10-Pretrained'].iloc[0]
        generated_row = epoch_data[epoch_data['Method'] == 'FlowGenerated'].iloc[0]
        
        col_name = f'{target_dataset.upper()} Accuracy'
        random_acc = float(random_row[col_name].split(' ±')[0])
        pretrained_acc = float(pretrained_row[col_name].split(' ±')[0])
        generated_acc = float(generated_row[col_name].split(' ±')[0])
        
        print(f"\nEpochs: {epoch}")
        print(f"  RandomInit:          {random_acc:.2f}%")
        print(f"  CIFAR10-Pretrained:  {pretrained_acc:.2f}% ({pretrained_acc - random_acc:+.2f}%)")
        print(f"  FlowGenerated:       {generated_acc:.2f}% ({generated_acc - random_acc:+.2f}%)")
    
    print(f"{'='*100}\n")


def train_and_evaluate_transfer(args, num_models_to_test = 25, epochs_list=[1,5,25]):
    """
    Main transfer learning training and evaluation pipeline.
    
    Pipeline:
    1. Load CIFAR-10 pretrained models
    2. Apply weight matching (optional)
    3. Train flow matching model
    4. Generate new models
    5. Evaluate transfer learning on target datasets
    """
    # Load configuration
    config = load_config(args.config)
    model_config = config['models'].get('resnet18_cifar10', {})
    model_dir = config['directories'].get('resnet18_cifar10', '../cifar10_models')
    
    print(f"Transfer Learning Evaluation: ResNet18")
    print(f"Configuration:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    
    # Step 1: Load pretrained models
    print("STEP 1: Loading CIFAR-10 Pretrained Models")
    
    pretrained_model_name = model_config.get('pretrained_model_name', 'resnet18_seed')
    org_models, permuted_models = get_permuted_models_data(
        model_name='resnet18_cifar10',
        model_dir=model_dir,
        pretrained_model_name=pretrained_model_name,
        num_models=args.num_models,
        ref_point=args.ref_point,
        device=device,
        model_config=model_config
    )
    
    print(f"Loaded {len(org_models)} models")
    
    # Get weight shapes for reconstruction
    ref_model = get_resnet18(num_classes=10)
    weight_shapes, bias_shapes = [], []
    for name, param in ref_model.named_parameters():
        if "weight" in name and "bn" not in name:
            weight_shapes.append(tuple(param.shape))
        elif "bias" in name and "bn" not in name:
            bias_shapes.append(tuple(param.shape))
    del ref_model
    
    # Process both modes (with/without weight matching)
    all_results = {}
    
    for training_mode in config.get('training_modes', ['with_gitrebasin', 'without_rebasin']):
        if args.mode and training_mode != args.mode:
            continue
        
        print(f"PROCESSING: {training_mode}")
        
        # Step 2: Convert to weight space
        print("\nSTEP 2: Converting Models to Weight Space")
        models_to_use = permuted_models if training_mode == "with_gitrebasin" else org_models
        weight_space_objects = convert_models_to_weight_space(models_to_use, exclude_bn=True)
        
        flat_target_weights = torch.stack([
            wso.flatten('cpu') for wso in weight_space_objects
        ])
        flat_dim = flat_target_weights.shape[1]
        print(f"Weight space dimension: {flat_dim:,}")
        
        # Optional PCA
        ipca = None
        if model_config.get('use_pca', False) and model_config.get('pca_components'):
            print(f"\nApplying PCA with {model_config['pca_components']} components")
            ipca = IncrementalPCA(
                n_components=model_config['pca_components'], 
                batch_size=10
            )
            flat_latent = ipca.fit_transform(flat_target_weights.cpu().numpy())
            target_tensor = torch.tensor(flat_latent, dtype=torch.float32)
            actual_dim = model_config['pca_components']
            print(f"Reduced to {actual_dim} dimensions")
        else:
            target_tensor = flat_target_weights
            actual_dim = flat_dim
        
        # Train flow models with different hidden dimensions
        for hidden_dim in model_config.get('flow_hidden_dims', [512]):
            if args.hidden_dim and hidden_dim != args.hidden_dim:
                continue
            
            print(f"\n{'='*70}")
            print(f"Training Flow Model: hidden_dim={hidden_dim}")
            print(f"{'='*70}")
            
            # Step 3: Prepare dataloaders
            source_std = model_config.get('source_std', 0.01)
            source_tensor = torch.randn_like(target_tensor) * source_std
            
            batch_size = model_config.get('batch_size', 8)
            sourceloader = DataLoader(
                TensorDataset(source_tensor), 
                batch_size=batch_size, 
                shuffle=True, 
                drop_last=True
            )
            targetloader = DataLoader(
                TensorDataset(target_tensor), 
                batch_size=batch_size, 
                shuffle=True, 
                drop_last=True
            )
            
            # Step 4: Train flow model
            cfm = train_weight_space_flow(
                sourceloader, targetloader, actual_dim, hidden_dim, model_config
            )
            
            # Step 5: Generate models
            generated_models = generate_models_from_flow(
                cfm=cfm,
                # n_samples=model_config.get('n_samples', 100),
                n_samples=5,
                actual_dim=actual_dim,
                weight_shapes=weight_shapes,
                bias_shapes=bias_shapes,
                source_std=source_std,
                model_config=model_config,
                ipca=ipca,
                exclude_bn=True
            )
            
            # Step 6: Select top models
            top_generated_models, _ = select_top_models(
                generated_models, 
                num_top=num_models_to_test
            )
            
            # Step 7: Evaluate transfer learning
            print(f"STEP 7: Evaluating Transfer Learning")
            
            results = evaluate_transfer_learning(
                pretrained_models=org_models,
                generated_models=top_generated_models,
                num_models_to_test=num_models_to_test,
                epochs_list=epochs_list,
                target_dataset=args.target_dataset,
                few_shot_samples=args.few_shot_samples,
                device=device
            )
            
            # Print and save results
            print_transfer_summary(results, args.target_dataset)
            
            # Save to file
            filename = f'transfer_{args.target_dataset}_{training_mode}_h{hidden_dim}.csv'
            pd.DataFrame(results).to_csv(filename, index=False)
            print(f"Results saved to {filename}")
            
            all_results[f"{training_mode}_h{hidden_dim}"] = results
            
            # Cleanup
            del cfm, generated_models, top_generated_models
            aggressive_cleanup()
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate transfer learning with flow-generated models'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='constants.json',
        help='Configuration file path'
    )
    parser.add_argument(
        '--num_models', 
        type=int, 
        default=100,
        help='Number of pretrained models to use'
    )
    parser.add_argument(
        '--ref_point', 
        type=int, 
        default=0,
        help='Reference model index for weight matching'
    )
    parser.add_argument(
        '--hidden_dim', 
        type=int, 
        default=None,
        help='Specific hidden dimension to test (optional)'
    )
    parser.add_argument(
        '--mode', 
        type=str, 
        default=None,
        choices=['with_gitrebasin', 'without_rebasin'],
        help='Training mode'
    )
    parser.add_argument(
        '--target_dataset',
        type=str,
        default='cifar100',
        choices=['cifar100', 'stl10', 'svhn'],
        help='Target dataset for transfer learning'
    )
    parser.add_argument(
        '--few_shot_samples',
        type=int,
        default=50,
        help='Samples per class for few-shot learning'
    )
    parser.add_argument(
        '--epochs_list',
        type=int,
        nargs='+',
        default=[0, 1, 5, 10],
        help='List of finetuning epochs to evaluate'
    )
    
    args = parser.parse_args()
    
    print(f"ResNet18 Transfer Learning Evaluation")
    print(f"Source: CIFAR-10 → Target: {args.target_dataset.upper()}")
    print(f"Device: {device}")
    
    try:
        train_and_evaluate_transfer(args)
        print("Evaluation Complete!")
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()