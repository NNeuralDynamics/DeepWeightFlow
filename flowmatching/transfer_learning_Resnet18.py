import sys
import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import traceback
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import torchvision
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import IncrementalPCA
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import *
from models import get_resnet18
from flow_matching import FlowMatching, WeightSpaceFlowModel
from canonicalization import get_permuted_models_data
import pandas as pd
from tabulate import tabulate

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_file='constants.json'):
    with open(config_file, 'r') as f:
        return json.load(f)


def train_weight_space_flow(sourceloader, targetloader, input_dim, hidden_dim, 
                           model_config=None):
    """
    Train flow model with configuration-based settings including gradient accumulation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_config is None:
        model_config = {}
    
    flow_model = WeightSpaceFlowModel(
        input_dim, 
        hidden_dim,
        time_embed_dim=model_config.get('time_embed_dim', 128),
        dropout=model_config.get('dropout', 0.1)
    ).to(device)
    
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
    
    # Build training kwargs
    train_kwargs = {
        'n_iters': model_config.get('n_iters', 30000),
        'optimizer': optimizer,
        'scheduler': scheduler,
        'sigma': model_config.get('sigma', 0.001),
        'patience': model_config.get('patience', 100),
        'log_freq': 10
    }
    
    grad_accum_steps = model_config.get('gradient_accumulation_steps')
    if grad_accum_steps is not None and grad_accum_steps > 1:
        train_kwargs['accum_steps'] = grad_accum_steps
        train_kwargs['clip_grad'] = model_config.get('clip_grad', 1.0)
        print(f"Using gradient accumulation: {grad_accum_steps} steps")
        batch_size = model_config.get('batch_size', 2)
        print(f"Effective batch size: {batch_size * grad_accum_steps}")
    
    cfm.train(**train_kwargs)
    
    return cfm



def finetune_model(model, train_loader, test_loader, epochs=10, lr=1e-4, 
                   detach_ratio=0.4, device='cuda'):
    model = model.to(device)

    all_layers = [
        model.conv1, model.bn1, model.layer1, model.layer2, model.layer3, model.layer4
    ]
    num_to_freeze = int(len(all_layers) * detach_ratio)

    for layer in all_layers[:num_to_freeze]:
        for param in layer.parameters():
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100.0 * correct / total
        train_loss /= total
        scheduler.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_acc = 100.0 * correct / total
        best_acc = max(best_acc, test_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}% | Best Acc: {best_acc:.2f}%")

    return model, best_acc

def select_top_models(generated_models, num_top=5):
    """
    """
    print("\n=== Selecting Top Models Based on CIFAR-10 Performance ===")
    
    _, cifar_test = get_cifar10_loaders(batch_size=128)
    
    model_scores = []
    for i, model in enumerate(generated_models):
        model_eval = copy.deepcopy(model)
        acc = evaluate_model(model_eval, cifar_test, device)
        model_scores.append((i, acc, model))
        print(f"Generated Model {i}: CIFAR-10 Accuracy = {acc:.2f}%")
    
    model_scores.sort(key=lambda x: x[1], reverse=True)
    
    top_models = [score[2] for score in model_scores[:num_top]]
    top_indices = [score[0] for score in model_scores[:num_top]]
    top_accs = [score[1] for score in model_scores[:num_top]]
    
    
    print(f"\nSelected Top {num_top} Models:")
    for idx, orig_idx, acc in zip(range(num_top), top_indices, top_accs):
        print(f"  Rank {idx+1}: Model {orig_idx} with {acc:.2f}% accuracy")
    
    all_accs = [score[1] for score in model_scores]
    print(f"\nModel Performance Statistics:")
    print(f"  Best: {max(all_accs):.2f}%")
    print(f"  Worst: {min(all_accs):.2f}%")
    print(f"  Mean: {np.mean(all_accs):.2f}%")
    print(f"  Std: {np.std(all_accs):.2f}%")
    print(f"  Top 5 Mean: {np.mean(top_accs):.2f}%")
    
    return top_models

def evaluate_adaptability(generated_models, num_models_to_test=5, epochs_list=[0, 1, 5, 10, 25, 50], device='cuda'):
    top_generated_models = select_top_models(
        generated_models, num_top=num_models_to_test
    )
    
    results = {
        'Epoch': [],
        'Method': [],
        'STL-10-Few-Shot': [],
        'SVHN': [],
    }
    
    tiny_train, tiny_test = get_fewshot_loaders(dataset_name='STL10')
    svhn_train, svhn_test = get_fewshot_loaders(dataset_name='SVHN')
    
    for epochs in epochs_list:
        print(f"\n=== Evaluating at {epochs} epochs ===")
        
        random_tiny_accs, random_svhn_accs = [], []
        
        for i in range(num_models_to_test):
            random_model = get_resnet18(num_classes=10)
            torch.manual_seed(42 + i)
            for m in random_model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            
            if epochs > 0:
                _, acc = finetune_model(copy.deepcopy(random_model), tiny_train, tiny_test, epochs=epochs, lr=1e-4, detach_ratio = 0.0, device=device)
                random_tiny_accs.append(acc)
                
                _, acc = finetune_model(copy.deepcopy(random_model), svhn_train, svhn_test, epochs=epochs, lr=1e-4, detach_ratio = 0.0,device=device)
                random_svhn_accs.append(acc)
            else:
                random_tiny_accs.append(evaluate_model(copy.deepcopy(random_model), tiny_test, device))
                random_svhn_accs.append(evaluate_model(copy.deepcopy(random_model), svhn_test, device))
        
        # Test Generated Models
        gen_tiny_accs, gen_svhn_accs = [], []
        
        for i in range(min(num_models_to_test, len(generated_models))):
            model = generated_models[i]
            
            if epochs > 0:
                _, acc = finetune_model(copy.deepcopy(model), tiny_train, tiny_test, epochs=epochs, lr=1e-4, device=device)
                gen_tiny_accs.append(acc)
                
                _, acc = finetune_model(copy.deepcopy(model), svhn_train, svhn_test, epochs=epochs, lr=1e-4, device=device)
                gen_svhn_accs.append(acc)
            else:
                gen_tiny_accs.append(evaluate_model(copy.deepcopy(model), tiny_test, device))
                gen_svhn_accs.append(evaluate_model(copy.deepcopy(model), svhn_test, device))
        
        # Store results (RandomInit)
        results['Epoch'].append(epochs)
        results['Method'].append('RandomInit')
        results['STL-10-Few-Shot'].append(f"{np.mean(random_tiny_accs):.2f} ± {np.std(random_tiny_accs):.2f}")
        results['SVHN'].append(f"{np.mean(random_svhn_accs):.2f} ± {np.std(random_svhn_accs):.2f}")
        
        # Store results (FlowGenerated)
        results['Epoch'].append(epochs)
        results['Method'].append('FlowGenerated')
        results['STL-10-Few-Shot'].append(f"{np.mean(gen_tiny_accs):.2f} ± {np.std(gen_tiny_accs):.2f}")
        results['SVHN'].append(f"{np.mean(gen_svhn_accs):.2f} ± {np.std(gen_svhn_accs):.2f}")
    
    return results


def train_and_evaluate_transfer(args):
    """Main transfer learning training and evaluation function"""
    config = load_config(args.config)
    model_config = config['models'].get('resnet18_cifar10', {})
    model_dir = config['directories'].get('resnet18_cifar10', '../cifar10_models')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Transfer Learning Evaluation for ResNet18")
    print(f"Config: {model_config}")
    
    model = get_resnet18()
    weight_shapes = []
    bias_shapes = []
    for name, param in model.named_parameters():
        if "weight" in name and "bn" not in name: 
            weight_shapes.append(param.shape)
        elif "bias" in name and "bn" not in name:
            bias_shapes.append(param.shape)
    
    pretrained_model_name = model_config.get('pretrained_model_name', 'resnet18_seed')
    ref_model, org_models, permuted_models = get_permuted_models_data(
        model_name='resnet18_cifar10',
        model_dir=model_dir,
        pretrained_model_name=pretrained_model_name,
        num_models=args.num_models,
        ref_point=args.ref_point,
        device=device,
        model_config=model_config
    )
    
    print("\n=== Baseline: Original Pretrained Models ===")
    baseline_results = evaluate_adaptability(
        org_models[:5],
        num_models_to_test=5,
        epochs_list=[0, 1, 5],
        device=device
    )
    
    df_baseline = pd.DataFrame(baseline_results)
    print("\nBaseline Results (Original Pretrained Models):")
    print(tabulate(df_baseline, headers='keys', tablefmt='grid', showindex=False))
    
    all_results = {}
    
    for training_mode in config.get('training_modes', ['with_gitrebasin', 'without_rebasin']):
        if args.mode and training_mode != args.mode:
            continue
            
        print(f"\n=== Processing {training_mode} ===")
        models_to_use = permuted_models if training_mode == "with_gitrebasin" else org_models
        
        weight_space_objects = []
        for model in tqdm(models_to_use, desc="Converting to weight space"):
            weights = []
            biases = []
            for name, param in model.named_parameters():
                if "weight" in name and "bn" not in name:
                    weights.append(param.data.clone())
                elif "bias" in name and "bn" not in name:
                    biases.append(param.data.clone())
            
            wso = WeightSpaceObjectResnet(weights, biases)
            wso.weight_shapes = weight_shapes
            wso.bias_shapes = bias_shapes
            weight_space_objects.append(wso)
        
        flat_target_weights = torch.stack([wso.flatten('cpu') for wso in weight_space_objects])
        flat_dim = flat_target_weights.shape[1]
        print(f"Weight space dimension: {flat_dim:,}")
        
        ipca = None
        if model_config.get('use_pca', False) and model_config.get('pca_components'):
            print(f"Applying PCA with {model_config['pca_components']} components")
            ipca = IncrementalPCA(n_components=model_config['pca_components'], batch_size=10)
            flat_latent = ipca.fit_transform(flat_target_weights.cpu().numpy())
            target_tensor = torch.tensor(flat_latent, dtype=torch.float32)
            actual_dim = model_config['pca_components']
        else:
            target_tensor = flat_target_weights
            actual_dim = flat_dim
        
        source_std = model_config.get('source_std', 0.01)
        source_tensor = torch.randn_like(target_tensor) * source_std
        
        source_dataset = TensorDataset(source_tensor)
        target_dataset = TensorDataset(target_tensor)
        
        batch_size = model_config.get('batch_size', 8)
        sourceloader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        targetloader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        for hidden_dim in model_config.get('flow_hidden_dims', [512]):
            if args.hidden_dim and hidden_dim != args.hidden_dim:
                continue
                
            print(f"\nTraining with hidden_dim={hidden_dim}")
            
            cfm = train_weight_space_flow(
                sourceloader, targetloader, actual_dim, hidden_dim,
                model_config=model_config
            )
            
            print("Generating new models...")
            n_samples = model_config.get('n_samples', 100)
            
            generated_models = []
            chunk_size = 10
            
            for chunk_start in range(0, n_samples, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_samples)
                chunk_n = chunk_end - chunk_start
                
                random_flat = torch.randn(chunk_n, actual_dim, device=device) * source_std
                
                new_weights_flat = cfm.map(
                    random_flat,
                    n_steps=model_config.get('integration_steps', 100),
                    method=model_config.get('integration_method', 'rk4')
                )
                
                if ipca is not None:
                    new_weights_flat = ipca.inverse_transform(new_weights_flat.cpu().numpy())
                    new_weights_flat = torch.tensor(new_weights_flat, dtype=torch.float32, device=device)
                
                for i in range(chunk_n):
                    ref_wso = weight_space_objects[0]
                    new_wso = WeightSpaceObjectResnet.from_flat(
                        new_weights_flat[i],
                        weight_shapes=ref_wso.weight_shapes,
                        bias_shapes=ref_wso.bias_shapes,
                        device=device
                    )
                    
                    model = get_resnet18().to(device)
                    
                    weight_idx = 0
                    bias_idx = 0
                    for name, param in model.named_parameters():
                        if "weight" in name and "bn" not in name:
                            param.data = new_wso.weights[weight_idx].clone()
                            weight_idx += 1
                        elif "bias" in name and "bn" not in name:
                            param.data = new_wso.biases[bias_idx].clone()
                            bias_idx += 1
                    
                    if model_config.get('recalibrate_bn', True):
                        model = recalibrate_bn_stats(model, device=device, print_stats=False)
                    
                    generated_models.append(model)
                
                del random_flat, new_weights_flat
                torch.cuda.empty_cache()
            
            print(f"\n=== Evaluating Transfer Learning ({training_mode}, hidden={hidden_dim}) ===")
            
            results = evaluate_adaptability(
                generated_models,
                num_models_to_test=5,
                epochs_list=[0, 1, 5],
                device=device
            )
            
            df = pd.DataFrame(results)
            
            print(f"\n{'='*100}")
            print(f"Results: {training_mode} with hidden_dim={hidden_dim}")
            print('='*100)
            print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
            
            filename = f'transfer_results_{training_mode}_h{hidden_dim}.csv'
            df.to_csv(filename, index=False)
            print(f"Results saved to {filename}")
            
            print(f"\n=== Performance Improvement Summary ===")
            for epoch in df['Epoch'].unique():
                epoch_data = df[df['Epoch'] == epoch]
                random_row = epoch_data[epoch_data['Method'] == 'RandomInit'].iloc[0]
                gen_row = epoch_data[epoch_data['Method'] == 'FlowGenerated'].iloc[0]
                
                random_stl = float(random_row['STL-10-Few-Shot'].split(' ±')[0])
                gen_stl = float(gen_row['STL-10-Few-Shot'].split(' ±')[0])
                random_svhn = float(random_row['SVHN'].split(' ±')[0])
                gen_svhn = float(gen_row['SVHN'].split(' ±')[0])
                
                print(f"\nEpoch {epoch}:")
                print(f"  STL-10 improvement: {gen_stl - random_stl:+.2f}%")
                print(f"  SVHN improvement: {gen_svhn - random_svhn:+.2f}%")
            
            all_results[f"{training_mode}_h{hidden_dim}"] = df
            
            del cfm, generated_models
            torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description='Transfer learning evaluation with flow matching')
    parser.add_argument('--config', type=str, default='constants.json',
                       help='Configuration file path')
    parser.add_argument('--num_models', type=int, default=100,
                       help='Number of pretrained models to use')
    parser.add_argument('--ref_point', type=int, default=0,
                       help='Reference model index')
    parser.add_argument('--hidden_dim', type=int, default=None,
                       help='Specific hidden dimension to test')
    parser.add_argument('--mode', type=str, default=None,
                       choices=['with_gitrebasin', 'without_rebasin'],
                       help='Training mode')
    
    args = parser.parse_args()
    
    print("ResNet18 Transfer Learning: CIFAR-10 → STL-10/SVHN")
    print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    try:
        train_and_evaluate_transfer(args)
    except Exception as e:
        print(f"Error in execution: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()