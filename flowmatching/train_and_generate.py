import os
import sys
import gc
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import IncrementalPCA

from weight_space_objects import (WeightSpaceObjectMLP, WeightSpaceObjectResnet, 
                                  VisionTransformerWeightSpace, BERTWeightSpace)
from utils import (count_parameters, print_stats, print_regression_stats, 
                  recalibrate_bn_stats, YelpReviewDataset,
                  load_mnist, load_fashion_mnist, load_cifar10, load_iris_dataset)
from models import (MLP_MNIST, MLP_Fashion_MNIST, MLP_Iris, ResNet20, 
                   get_resnet18, create_vit_small, create_bert_100m)
from flow_matching import FlowMatching, WeightSpaceFlowModel, VisionTransformerFlowModel
from canonicalization import get_permuted_models_data, apply_transfusion, aggressive_cleanup
from dual_pca import flatten_and_project_to_disk_maxcomponents

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_file='constants.json'):
    """Load configuration from JSON file"""
    with open(config_file, 'r') as f:
        return json.load(f)


def get_data_loader(dataset_name: str, batch_size: int = 32):
    """
    Unified function to get PyTorch DataLoader.
    Returns a DataLoader for the requested dataset.
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == "mnist":
        return load_mnist(batch_size=batch_size)
    elif dataset_name == "fashion_mnist":
        return load_fashion_mnist(batch_size=batch_size)
    elif dataset_name == "cifar10":
        return load_cifar10(batch_size=batch_size)
    elif dataset_name == "iris":
        return load_iris_dataset(batch_size=batch_size)
    elif dataset_name == "yelp":
        yelp_dataset = YelpReviewDataset('test', subset_size=10000)
        return DataLoader(yelp_dataset, batch_size=batch_size, shuffle=False)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def convert_models_to_weight_space(models_to_use, model_config):
    """Convert PyTorch models to weight space objects"""
    weight_space_objects = []

    if model_config['architecture'] == 'vit':
        for model in models_to_use:
            if hasattr(model, 'flatten'):
                weight_space_objects.append(model)
            else:
                ws_obj = VisionTransformerWeightSpace.from_vit_model(model.to(device))
                weight_space_objects.append(ws_obj)

    elif model_config['architecture'] == 'bert':
        # BERT models are already in weight space format from apply_transfusion
        return models_to_use

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
            wso.weight_shapes = weight_shapes
            wso.bias_shapes = bias_shapes
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


def create_dataloaders_for_flow_matching(pca_output_file, n_models, latent_dim, 
                                         source_std, batch_size, device='cpu'):
    """Create source and target dataloaders from PCA-projected data"""
    projected_data = np.memmap(pca_output_file, dtype=np.float32, mode='r', 
                               shape=(n_models, latent_dim))
    
    target_tensor = torch.tensor(projected_data[:], dtype=torch.float32, device=device)
    source_tensor = torch.randn_like(target_tensor) * source_std
    
    sourceloader = DataLoader(TensorDataset(source_tensor), batch_size=batch_size, 
                             shuffle=True, drop_last=True)
    targetloader = DataLoader(TensorDataset(target_tensor), batch_size=batch_size, 
                             shuffle=True, drop_last=True)
    
    return sourceloader, targetloader


def train_bert_model(args, config, model_config, model_dir, test_loader):
    """Train and generate BERT models using TransFusion and GPU-accelerated PCA"""
    
    print("BERT ARCHITECTURE - TransFusion + GPU-accelerated PCA")
    
    pretrained_model_name = model_config.get('pretrained_model_name', args.model)
    
    # Load and align BERT models using TransFusion
    ref_ws, weight_space_objects = apply_transfusion(
        model_dir=model_dir,
        num_models=args.num_models,
        pretrained_model_name=pretrained_model_name,
        ref_point=args.ref_point,
        device=device
    )
    
    reference_ws = weight_space_objects[0]
    actual_num_models = len(weight_space_objects)
    
    print(f"Loaded {actual_num_models} aligned BERT models")
    
    # Create loader function for PCA
    def load_models_batch(start_idx, end_idx):
        """Load weight spaces as numpy array (n_params, n_models)"""
        batch = []
        for i in range(start_idx, end_idx):
            ws = weight_space_objects[i]
            flat = ws.flatten(device='cpu').numpy()
            batch.append(flat)
        return np.column_stack(batch)
    
    # Use GPU-accelerated PCA with variance target
    print("\nApplying GPU-accelerated PCA...")
    output_file, pca, info = flatten_and_project_to_disk_maxcomponents(
        model_loader_func=load_models_batch,
        n_models=actual_num_models,
        target_variance=model_config.get('pca_variance', 0.98),
        batch_size=5,
        use_fp16=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"\nPCA Results:")
    print(f"  Components: {info['n_components']}")
    print(f"  Variance captured: {info['total_variance_captured']:.2%}")
    
    actual_dim = info['n_components']
    
    # Train flow models with different hidden dimensions
    for hidden_dim in model_config['flow_hidden_dims']:
        if args.hidden_dim and hidden_dim != args.hidden_dim:
            continue
            
        print(f"Training BERT flow model: hidden_dim={hidden_dim}")
        
        source_std = model_config['source_std']
        sourceloader, targetloader = create_dataloaders_for_flow_matching(
            pca_output_file=output_file,
            n_models=actual_num_models,
            latent_dim=actual_dim,
            source_std=source_std,
            batch_size=model_config['batch_size'],
            device='cpu'
        )
        
        flow_model = VisionTransformerFlowModel(actual_dim, hidden_dim).to(device)
        print(f"Flow model parameters: {count_parameters(flow_model):,}")
        
        cfm = FlowMatching(
            sourceloader=sourceloader,
            targetloader=targetloader,
            model=flow_model,
            mode="velocity",
            t_dist=config['t_dist'],
            device=device
        )
        
        cfm.input_dim = actual_dim
        
        optimizer = torch.optim.AdamW(
            flow_model.parameters(), 
            lr=model_config['lr'],
            weight_decay=model_config['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=model_config['n_iters'], 
            eta_min=1e-6
        )
        
        print("\nTraining flow model...")
        cfm.train(
            n_iters=model_config['n_iters'],
            optimizer=optimizer,
            scheduler=scheduler,
            sigma=model_config['sigma'],
            patience=model_config['patience'],
            log_freq=10
        )
        
        # Generate new models
        print(f"\nGenerating {model_config['n_samples']} new BERT models...")
        generated_models = []
        n_samples = model_config['n_samples']
        
        gen_batch_size = 5
        for i in tqdm(range(0, n_samples, gen_batch_size), desc="Generating models"):
            batch_n = min(gen_batch_size, n_samples - i)
            
            try:
                # Generate latent codes
                random_latent = torch.randn(batch_n, actual_dim, device=device) * source_std
                generated_latent = cfm.map(
                    random_latent, 
                    n_steps=model_config['integration_steps'],
                    method=model_config['integration_method']
                )
                
                # Inverse PCA transform
                generated_latent_cpu = generated_latent.cpu().numpy()
                generated_flat = pca.inverse_transform(generated_latent_cpu)
                
                # Reconstruct models
                for j in range(batch_n):
                    flat_tensor = torch.tensor(
                        generated_flat[j], 
                        dtype=torch.float32, 
                        device=device
                    )
                    
                    # Reconstruct weight space
                    generated_ws = BERTWeightSpace.from_flat(
                        flat_tensor, 
                        reference_ws, 
                        device
                    )
                    
                    # Create new model and apply weights
                    new_model = create_bert_100m(num_classes=1).to(device)
                    generated_ws.apply_to_model(new_model)
                    
                    generated_models.append(new_model)
                    
                    del generated_ws, flat_tensor
                
                del random_latent, generated_latent, generated_latent_cpu, generated_flat
                aggressive_cleanup()
                
            except Exception as e:
                print(f"Error generating batch {i}: {e}")
                aggressive_cleanup()
                continue
        
        # Evaluate using regression metrics
        print(f"\nEvaluating {len(generated_models)} generated BERT models...")
        all_metrics = []
        for idx, model in enumerate(tqdm(generated_models, desc="Evaluating")):
            try:
                metrics = print_regression_stats(model, test_loader, device)
                all_metrics.append(metrics)
            except Exception as e:
                print(f"Error evaluating model {idx}: {e}")
                continue
        
        if all_metrics:
            # Calculate statistics
            r2_scores = [m['r2'] for m in all_metrics]
            mae_scores = [m['mae'] for m in all_metrics]
            spearman_scores = [m['spearman'] for m in all_metrics]
            
            print(f"BERT Generation Results (hidden_dim={hidden_dim}):")
            print(f"  Models evaluated:  {len(all_metrics)}/{n_samples}")
            print(f"  R² Score:          {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
            print(f"  MAE:               {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
            print(f"  Spearman ρ:        {np.mean(spearman_scores):.4f} ± {np.std(spearman_scores):.4f}")
            print(f"  Best R²:           {np.max(r2_scores):.4f}")
            print(f"  Worst R²:          {np.min(r2_scores):.4f}")
        else:
            print("WARNING: No models were successfully evaluated!")
        
        del flow_model, cfm, generated_models, optimizer, scheduler
        aggressive_cleanup()
    
    # Cleanup
    try:
        os.remove(output_file)
        print(f"Cleaned up temporary file: {output_file}")
    except Exception as e:
        print(f"Could not remove temporary file: {e}")
    
    del ref_ws, weight_space_objects, reference_ws, pca
    aggressive_cleanup()


def train_standard_model(args, config, model_config, model_dir, test_loader):
    """Train and generate models using standard weight matching (MLP, ResNet, ViT)"""
    
    print(f"STANDARD ARCHITECTURE: {model_config['architecture'].upper()}")
    
    pretrained_model_name = model_config.get('pretrained_model_name', args.model)
    
    org_models, permuted_models = get_permuted_models_data(
        model_name=args.model,
        model_dir=model_dir,
        pretrained_model_name=pretrained_model_name,
        num_models=args.num_models,
        ref_point=args.ref_point,
        device=device,
        model_config=model_config
    )
    
    print(f"Loaded {len(org_models)} models")

    # Train with different hidden dimensions
    for hidden_dim in model_config['flow_hidden_dims']:
        if args.hidden_dim and hidden_dim != args.hidden_dim:
            continue

        print(f"Training with hidden_dim={hidden_dim}")

        # Train with/without weight matching
        for training_mode in config['training_modes']:
            if args.mode and training_mode != args.mode:
                continue

            print(f"\nMode: {training_mode}")
            
            models_to_use = permuted_models if training_mode == "with_gitrebasin" else org_models
            weight_space_objects = convert_models_to_weight_space(models_to_use, model_config)
            
            flat_target_weights = torch.stack([
                wso.flatten(device) for wso in weight_space_objects
            ]).to(device)
            flat_dim = flat_target_weights.shape[1]
            print(f"Weight space dimension: {flat_dim:,}")

            # Optional PCA
            ipca = None
            if model_config.get('use_pca', False) and model_config.get('pca_components'):
                print(f"Applying PCA with {model_config['pca_components']} components")
                ipca = IncrementalPCA(
                    n_components=model_config['pca_components'], 
                    batch_size=10
                )
                flat_latent = ipca.fit_transform(flat_target_weights.cpu().numpy())
                target_tensor = torch.tensor(flat_latent, dtype=torch.float32)
                actual_dim = flat_latent.shape[1] # model_config['pca_components'] 
            else:
                target_tensor = flat_target_weights
                actual_dim = flat_dim

            source_std = model_config['source_std']
            source_tensor = torch.randn_like(target_tensor) * source_std

            n_samples = target_tensor.shape[0]
            batch_size_eff = min(model_config['batch_size'], n_samples)
            drop_last = n_samples > batch_size_eff

            sourceloader = DataLoader(
                TensorDataset(source_tensor), 
                batch_size=batch_size_eff, 
                shuffle=True, 
                drop_last=drop_last
            )
            targetloader = DataLoader(
                TensorDataset(target_tensor), 
                batch_size=batch_size_eff, 
                shuffle=True, 
                drop_last=drop_last
            )

            # Create flow model
            flow_model = WeightSpaceFlowModel(
                actual_dim, 
                hidden_dim,
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

            optimizer = torch.optim.AdamW(
                flow_model.parameters(), 
                lr=model_config['lr'],
                weight_decay=model_config['weight_decay'], 
                betas=(0.9, 0.95)
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=model_config['n_iters'], 
                eta_min=1e-6
            )

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

            print("\nTraining flow model...")
            cfm.train(**train_kwargs)

            # Generate new weights
            n_samples = model_config['n_samples']
            random_flat = torch.randn(n_samples, actual_dim, device=device) * source_std
            
            # Generate models based on architecture
            generated_models = []
            
            if "mlp" in args.model:
                print(f"\nGenerating {n_samples} MLP models...")
                new_weights_flat = cfm.map(
                    random_flat, 
                    n_steps=model_config['integration_steps'],
                    method=model_config['integration_method']
                )
                
                if ipca is not None:
                    new_weights_flat = ipca.inverse_transform(new_weights_flat.cpu().numpy())
                    new_weights_flat = torch.tensor(new_weights_flat, dtype=torch.float32, device=device)
                
                for i in range(n_samples):
                    new_wso = WeightSpaceObjectMLP.from_flat(
                        new_weights_flat[i],
                        model_config['layer_layout'],  # Just pass the list
                        device=device
                    )
                    
                    if 'fashion' in args.model:
                        model = MLP_Fashion_MNIST()
                    elif 'mnist' in args.model:
                        model = MLP_MNIST()
                    elif 'iris' in args.model:
                        model = MLP_Iris()
                    else:
                        raise ValueError(f"Unknown MLP model: {args.model}")
                
                    for idx in range(len(new_wso.weights)):
                        getattr(model, f'fc{idx+1}').weight.data = new_wso.weights[idx].clone()
                        getattr(model, f'fc{idx+1}').bias.data = new_wso.biases[idx].clone()
                    
                    generated_models.append(model.to(device))
            
            elif "resnet" in args.model:
                print(f"\nGenerating {n_samples} ResNet models...")
                new_weights_flat = cfm.map(
                    random_flat, 
                    n_steps=model_config['integration_steps'],
                    method=model_config['integration_method']
                )
                
                if ipca is not None:
                    new_weights_flat = ipca.inverse_transform(new_weights_flat.cpu().numpy())
                    new_weights_flat = torch.tensor(new_weights_flat, dtype=torch.float32, device=device)
                
                # Get weight shapes from a reference model
                ref_model = ResNet20() if "20" in args.model else get_resnet18()
                weight_shapes, bias_shapes = [], []
                for name, param in ref_model.named_parameters():
                    if "weight" in name:
                        weight_shapes.append(tuple(param.shape))
                    elif "bias" in name:
                        bias_shapes.append(tuple(param.shape))
                del ref_model
                
                for i in range(n_samples):
                    new_wso = WeightSpaceObjectResnet.from_flat(
                        new_weights_flat[i],
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
            
            elif "vit" in args.model:
                print(f"\nGenerating {n_samples} ViT models...")
                reference_ws = org_models[0]
                
                # Generate in chunks for memory efficiency
                generated_chunks = []
                chunk_size = 5
                for i in range(0, n_samples, chunk_size):
                    batch = random_flat[i:i+chunk_size].to(device)
                    gen = cfm.map(
                        batch, 
                        n_steps=model_config['integration_steps'], 
                        method=model_config['integration_method']
                    )
                    generated_chunks.append(gen.cpu())
                
                generated_flat = torch.cat(generated_chunks, dim=0)
                
                if ipca is not None:
                    generated_flat = ipca.inverse_transform(generated_flat.cpu().numpy())
                    generated_flat = torch.tensor(generated_flat, dtype=torch.float32, device=device)
                
                for i in range(n_samples):
                    generated_ws = VisionTransformerWeightSpace.from_flat(
                        generated_flat[i], 
                        reference_ws, 
                        device
                    )
                    
                    new_model = create_vit_small().to(device)
                    generated_ws.apply_to_model(new_model)
                    generated_models.append(new_model)
                    del generated_ws
            
            else:
                raise ValueError(f"Unknown model type: {args.model}")
            
            # Evaluate generated models
            print(f"\nEvaluating {len(generated_models)} generated models...")
            mean_acc, std_acc = print_stats(generated_models, test_loader, device)
            
            print(f"\n{'='*60}")
            print(f"Results for {training_mode} with hidden_dim={hidden_dim}:")
            print(f"  Mean Accuracy: {mean_acc:.4f}%")
            print(f"  Std Accuracy:  {std_acc:.4f}%")
            print(f"{'='*60}\n")

            del flow_model, cfm, generated_models, optimizer, scheduler
            torch.cuda.empty_cache()


def train_and_generate(args):
    """Main training and generation function"""
    
    # Load configuration
    config = load_config(args.config)
    model_config = config['models'][args.model]
    model_dir = config['directories'][args.model]

    print(f"Training Flow Matching for {args.model}")
    print(f"Configuration:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")

    # Load test data
    test_loader = get_data_loader(model_config['dataset'], batch_size=32)
    
    # Route to appropriate training function based on architecture
    if model_config['architecture'] == 'bert':
        train_bert_model(args, config, model_config, model_dir, test_loader)
    else:
        train_standard_model(args, config, model_config, model_dir, test_loader)
    
    print("Training and generation complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Train flow matching for neural network weight generation'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        required=True,
        choices=['mlp_fashion_mnist', 'mlp_mnist', 'mlp_iris', 
                'resnet20_cifar10', 'resnet18_cifar10', 'vit_cifar10', 'bert_yelp'],
        help='Model architecture to train'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='constants.json', 
        help='Path to configuration file'
    )
    parser.add_argument(
        '--num_models', 
        type=int, 
        default=5, 
        help='Number of pretrained models to use'
    )
    parser.add_argument(
        '--ref_point', 
        type=int, 
        default=0, 
        help='Index of reference model for alignment'
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
        help='Training mode (only for non-BERT models)'
    )
    
    args = parser.parse_args()

    try:
        train_and_generate(args)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()