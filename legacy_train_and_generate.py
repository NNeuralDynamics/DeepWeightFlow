import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import WeightSpaceObjectMLP, WeightSpaceObjectResnet, VisionTransformerWeightSpace
from utils import *
from models import MLP_California, MC_MLP_MNIST, MLP_MNIST, MLP_Fashion_MNIST, MLP_Iris, ResNet20, get_resnet18, create_vit_small
from flow_matching import FlowMatching, WeightSpaceFlowModel
from canonicalization import get_permuted_models_data


np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_file='constants.json'):
    with open(config_file, 'r') as f:
        return json.load(f)

def get_data_loader(dataset_name: str, batch_size: int = 32, train: bool = False):
    """
    Unified function to get PyTorch DataLoader using utils' dataset loaders.
    Returns a DataLoader for the requested dataset and split.
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == "mnist":
        # MNIST loader ignores train flag, can extend later if needed
        return load_mnist(batch_size=batch_size)
    
    elif dataset_name == "fashion_mnist":
        return load_fashion_mnist(batch_size=batch_size)
    
    elif dataset_name == "cifar10":
        return load_cifar10(batch_size=batch_size)
    
    elif dataset_name == "iris":
        return load_iris_dataset(batch_size=batch_size)
    
    elif dataset_name == "california":
        return get_california_loaders(batch_size=batch_size)  
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def convert_models_to_weight_space(models_to_use, model_config):
    weight_space_objects = []

    if model_config['architecture'] == 'vit':
        for model in models_to_use:
            if hasattr(model, 'flatten'):
                weight_space_objects.append(model)
            else:
                ws_obj = VisionTransformerWeightSpace.from_vit_model(model.to(device))
                weight_space_objects.append(ws_obj)

    elif 'resnet' in model_config['architecture'].lower():
        for model in tqdm(models_to_use, desc="Converting ResNet to weight space"):
            weights, biases, weight_shapes, bias_shapes = [], [], [], []
            for name, param in model.named_parameters():
                param = param.detach().to(device)
                if torch.isnan(param).any() or torch.isinf(param).any():
                    logging.warning(f"NaN/Inf detected in {name}, replacing with zeros")
                    param = torch.zeros_like(param)

                if "weight" in name:
                    weights.append(param.clone())
                    weight_shapes.append(param.shape)
                elif "bias" in name:
                    biases.append(param.clone())
                    bias_shapes.append(param.shape)
            wso = WeightSpaceObjectResnet(weights, biases)
            wso.weight_shapes, wso.bias_shapes = weight_shapes, bias_shapes
            weight_space_objects.append(wso)

    else:  # MLP
        for model in tqdm(models_to_use, desc="Converting MLP to weight space"):
            weights, biases = [], []
            for name, param in model.named_parameters():
                param = param.detach().to(device)
                if torch.isnan(param).any() or torch.isinf(param).any():
                    logging.warning(f"NaN/Inf detected in {name}, replacing with zeros")
                    param = torch.zeros_like(param)
                if "weight" in name:
                    weights.append(param.clone())
                elif "bias" in name:
                    biases.append(param.clone())
            wso = WeightSpaceObjectMLP(weights, biases)
            weight_space_objects.append(wso)

    return weight_space_objects


def train_and_generate(args):
    config = load_config(args.config)
    model_config = config['models'][args.model]
    model_dir = config['directories'][args.model]

    print(f"Training {args.model} with config: {model_config}")
    test_loader = get_data_loader(model_config['dataset'], batch_size=32, train=False)

    pretrained_model_name = model_config.get('pretrained_model_name', args.model)
    ref_model, org_models, permuted_models = get_permuted_models_data(
        model_name=args.model,
        model_dir=model_dir,
        pretrained_model_name=pretrained_model_name,
        num_models=args.num_models,
        ref_point=args.ref_point,
        device=device,
        model_config=model_config
    )

    for hidden_dim in model_config['flow_hidden_dims']:
        if args.hidden_dim and hidden_dim != args.hidden_dim:
            continue
        print(f"Training with hidden_dim: {hidden_dim}")

        for training_mode in config['training_modes']:
            if args.mode and training_mode != args.mode:
                continue

            models_to_use = permuted_models if training_mode == "with_gitrebasin" else org_models
            weight_space_objects = convert_models_to_weight_space(models_to_use, model_config)
            flat_target_weights = torch.stack([wso.flatten(device) for wso in weight_space_objects]).to(device)
            flat_dim = flat_target_weights.shape[1]
            print(f"Weight space dimension: {flat_dim:,}")

            # PCA
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

            # print(f"  DEBUG: flat_latent.shape = {flat_latent.shape}")
            # print(f"  DEBUG: target_tensor.shape = {target_tensor.shape}")
            # print(f"  DEBUG: actual_dim = {actual_dim}")

            source_std = model_config['source_std']
            source_tensor = torch.randn_like(target_tensor) * source_std

            sourceloader = DataLoader(TensorDataset(source_tensor), batch_size=model_config['batch_size'], shuffle=True, drop_last=True)
            targetloader = DataLoader(TensorDataset(target_tensor), batch_size=model_config['batch_size'], shuffle=True, drop_last=True)

            flow_model = WeightSpaceFlowModel(
                actual_dim, hidden_dim,
                time_embed_dim=model_config['time_embed_dim'],
                dropout=model_config['dropout']
            ).to(device)
            print(f"Flow model parameters: {count_parameters(flow_model):,}")

            cfm = FlowMatching(
                sourceloader=sourceloader,
                targetloader=targetloader,
                model=flow_model,
                mode="velocity",
                t_dist=config['t_dist'],
                device=device
            )

            optimizer = torch.optim.AdamW(flow_model.parameters(), lr=model_config['lr'],
                                          weight_decay=model_config['weight_decay'], betas=(0.9, 0.95))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=model_config['n_iters'], eta_min=1e-6)

            train_kwargs = {
                'n_iters': model_config['n_iters'],
                'optimizer': optimizer,
                'scheduler': scheduler,
                'sigma': model_config['sigma'],
                'patience': model_config['patience'],
                'log_freq': 10
            }
            grad_accum_steps = model_config.get('gradient_accumulation_steps')
            if grad_accum_steps is not None:
                train_kwargs['accum_steps'] = grad_accum_steps

            cfm.train(**train_kwargs)

            # Generate new weights
            n_samples = model_config['n_samples']
            random_flat = torch.randn(n_samples, actual_dim, device=device) * source_std
            
            if "vit" not in args.model:
                new_weights_flat = cfm.map(random_flat, n_steps=model_config['integration_steps'],
                                           method=model_config['integration_method'])
                if ipca is not None:
                    new_weights_flat = ipca.inverse_transform(new_weights_flat.cpu().numpy())
                    new_weights_flat = torch.tensor(new_weights_flat, dtype=torch.float32, device=device)
           # -------------------- MLP --------------------
            generated_models = []
            if "mlp" in args.model:
                for i in range(n_samples):
                    new_wso = WeightSpaceObjectMLP.from_flat(
                        new_weights_flat[i],
                        layers=np.array(model_config['layer_layout']),
                        device=device
                    )
                    if 'fashion' and not 'mc' in args.model:
                        model = MLP_Fashion_MNIST()
                    elif 'mnist' and not 'mc' in args.model:
                        model = MLP_MNIST()
                    elif 'mnist' and 'mc' in args.model:
                        model = MC_MLP_MNIST()
                    elif 'iris' in args.model:
                        model = MLP_Iris()
                    elif 'california' in args.model:
                        model = MLP_California()
                    else:
                        raise ValueError(f"Unknown MLP model: {args.model}")
            
                    for idx in range(len(new_wso.weights)):
                        getattr(model, f'fc{idx+1}').weight.data = new_wso.weights[idx].clone()
                        getattr(model, f'fc{idx+1}').bias.data = new_wso.biases[idx].clone()
                    
                    generated_models.append(model.to(device))
            
            elif "resnet" in args.model:
                model = ResNet20() if "20" in args.model else get_resnet18()
                weight_shapes, bias_shapes = [], []
                for name, param in model.named_parameters():
                    if "weight" in name:
                        weight_shapes.append(tuple(param.shape))
                    elif "bias" in name:
                        bias_shapes.append(tuple(param.shape))
                
                for i in range(n_samples):
                    new_wso = WeightSpaceObjectResnet.from_flat(
                        torch.tensor(new_weights_flat[i], dtype=torch.float32, device=device),
                        weight_shapes,
                        bias_shapes,
                        device=device
                    )
                    
                    model = ResNet20() if "20" in args.model else get_resnet18()
                    param_dict = {}
                    weight_idx, bias_idx = 0, 0
                    for name, param in model.named_parameters():
                        if "weight" in name:
                            param_dict[name] = new_wso.weights[weight_idx]
                            weight_idx += 1
                        elif "bias" in name:
                            param_dict[name] = new_wso.biases[bias_idx]
                            bias_idx += 1
                    with torch.no_grad():
                        for name, param in model.named_parameters():
                            if name in param_dict:
                                param.copy_(param_dict[name])
                    model = recalibrate_bn_stats(model, device)
                    generated_models.append(model.to(device))
            
            # -------------------- ViT --------------------
            elif "vit" in args.model:
                reference_ws = org_models[0]
                generated_chunks = []
                chunk_size = 5
                for i in range(0, n_samples, chunk_size):
                    batch = random_flat[i:i+chunk_size].to(device)
                    gen = cfm.map(batch, n_steps=100, method=model_config['integration_method'])
                    generated_chunks.append(gen.cpu())
                generated_flat = torch.cat(generated_chunks, dim=0)
                
                if ipca is not None:
                    generated_flat = ipca.inverse_transform(generated_flat.cpu().numpy())
                    generated_flat = torch.tensor(generated_flat, dtype=torch.float32, device=device)
                
                for i in range(n_samples):
                    generated_ws = VisionTransformerWeightSpace.from_flat(
                        generated_flat[i], reference_ws, device
                    )
                    
                    new_model = create_vit_small().to(device)
                    generated_ws.apply_to_model(new_model)
                    generated_models.append(new_model)
                    del new_model
            else:
                print("Incorrect model type")
            
            if 'california' not in args.model:
                mean_acc, std_acc = print_stats(generated_models, test_loader, device)
                print(f"\nResults for {training_mode} with hidden_dim={hidden_dim}: mean={mean_acc:.4f}, std={std_acc:.4f}")
            elif 'california' in args.model:
                mse, mae = print_regression_stats(generated_models, test_loader, device)
                print(f"\nResults for {training_mode} with hidden_dim={hidden_dim}: MSE={mse:.4f}, MAE={mae:.4f}")
            else: 
                print("Evaluation not implemented for this model type.")
            del flow_model, cfm, generated_models
            torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description='Train flow matching for neural network weight generation')
    parser.add_argument('--model', type=str, required=True,
                        choices=['mlp_fashion_mnist', 'mlp_mnist', 'mlp_iris', 'resnet20_cifar10', 'mlp_california', 'resnet18_cifar10', 'vit_cifar10', 'mc_mlp_mnist_compressed'],
                        help='Model to train')
    parser.add_argument('--config', type=str, default='constants.json', help='Configuration file path')
    parser.add_argument('--num_models', type=int, default=100, help='Number of pretrained models to use')
    parser.add_argument('--ref_point', type=int, default=0, help='Reference model index')
    parser.add_argument('--hidden_dim', type=int, default=None, help='Specific hidden dimension to test')
    parser.add_argument('--mode', type=str, default=None, choices=['with_gitrebasin', 'without_rebasin'], help='Training mode')
    args = parser.parse_args()

    print(f"Starting training for {args.model}")
    train_and_generate(args)
    print("Training and generation complete!")

if __name__ == "__main__":
    main()