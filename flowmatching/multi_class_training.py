import os
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from sklearn.decomposition import IncrementalPCA

from utils import (count_parameters, 
                   recalibrate_bn_stats, print_stats, Bunch)
from weight_space_objects import (WeightSpaceObjectMLP, WeightSpaceObjectResnet)
from models import MLP_MNIST, MLP_Fashion_MNIST, MLP_Iris, ResNet20
from canonicalization import get_permuted_models_data

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiClassFlowMatching:
    def __init__(
        self,
        sourceloader,
        targetloader,
        model,
        mode="velocity",
        t_dist="uniform",
        device=None,
        normalize_pred=False,
        geometric=False,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sourceloader = sourceloader
        self.targetloader = targetloader
        self.model = model.to(self.device)
        self.mode = mode
        self.t_dist = t_dist
        self.sigma = 0.001
        self.normalize_pred = normalize_pred
        self.geometric = geometric

        self.best_loss = float('inf')
        self.best_model_state = None
        self.input_dim = None

    def sample_from_loader(self, loader):
        """Sample a batch from dataloader, returns (data, labels)"""
        try:
            if not hasattr(loader, '_iterator') or loader._iterator is None:
                loader._iterator = iter(loader)
            try:
                batch = next(loader._iterator)
            except StopIteration:
                loader._iterator = iter(loader)
                batch = next(loader._iterator)
            
            # Handle both (data,) and (data, labels) returns
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                return batch[0].to(self.device), batch[1].to(self.device)
            else:
                data = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
                return data, None
                
        except Exception as e:
            logging.info(f"Error sampling from loader: {str(e)}")
            if hasattr(loader.dataset, '__getitem__'):
                dummy = loader.dataset[0][0]
                return torch.zeros(loader.batch_size, *dummy.shape, device=self.device), None
            return torch.zeros(loader.batch_size, 1, device=self.device), None

    def sample_time_and_flow(self):
        """Sample time t and flow with class conditioning"""
        x0, _ = self.sample_from_loader(self.sourceloader)  # Source is just noise, no class
        x1, c1 = self.sample_from_loader(self.targetloader)  # Target has class labels
        
        batch_size = min(x0.size(0), x1.size(0))
        x0, x1 = x0[:batch_size], x1[:batch_size]
        
        if c1 is not None:
            c1 = c1[:batch_size]
            if c1.dim() == 1:
                c1 = c1.unsqueeze(-1)
        else:
            c1 = torch.zeros(batch_size, 1, device=self.device)

        if self.t_dist == "beta":
            alpha, beta_param = 2.0, 5.0
            t = torch.distributions.Beta(alpha, beta_param).sample((batch_size,)).to(self.device)
        else:
            t = torch.rand(batch_size, device=self.device)

        t_pad = t.view(-1, *([1] * (x0.dim() - 1)))
        
        mu_t = (1 - t_pad) * x0 + t_pad * x1
        epsilon = torch.randn_like(x0) * self.sigma
        xt = mu_t + epsilon
        ut = x1 - x0

        return Bunch(t=t.unsqueeze(-1), x0=x0, xt=xt, x1=x1, ut=ut, 
                    eps=epsilon, batch_size=batch_size, c=c1)

    def forward(self, flow):
        """Forward pass with class conditioning"""
        flow_pred = self.model(flow.xt, flow.t, flow.c)
        return None, flow_pred

    def loss_fn(self, flow_pred, flow):
        if self.mode == "target":
            l_flow = torch.mean((flow_pred.squeeze() - flow.x1) ** 2)
        else:
            l_flow = torch.mean((flow_pred.squeeze() - flow.ut) ** 2)
        return None, l_flow

    def vector_field(self, xt, t, c):
        """Compute vector field at point xt, time t, and class c"""
        _, pred = self.forward(Bunch(xt=xt, t=t, c=c, batch_size=xt.size(0)))
        return pred if self.mode == "velocity" else pred - xt

    def train(self, n_iters=10, optimizer=None, scheduler=None, sigma=0.001, patience=1e99, 
              log_freq=5, accum_steps=None):
        """Train the flow model with optional gradient accumulation"""
        self.sigma = sigma
        last_loss = 1e99
        patience_count = 0
        pbar = tqdm(range(n_iters), desc="Training steps")
        
        accum_count = 0
        accumulated_loss = 0
        
        use_grad_accum = accum_steps is not None and accum_steps > 1
        effective_accum_steps = accum_steps if use_grad_accum else 1
        
        for i in pbar:
            flow = self.sample_time_and_flow()
            _, flow_pred = self.forward(flow)
            _, loss = self.loss_fn(flow_pred, flow)
            
            if torch.isnan(loss) or torch.isinf(loss):
                logging.info(f"Skipping step {i} due to invalid loss: {loss.item()}")
                continue
            
            loss_scaled = loss / effective_accum_steps if use_grad_accum else loss
            loss_scaled.backward()
            accum_count += 1
            accumulated_loss += loss.item()
            
            should_update = (not use_grad_accum) or (accum_count == effective_accum_steps) or (i == n_iters - 1)
            
            if should_update:
                optimizer.step()
                optimizer.zero_grad()
                
                if scheduler:
                    scheduler.step()
                
                avg_loss = accumulated_loss / accum_count
                
                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    self.best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                
                if avg_loss > last_loss:
                    patience_count += 1
                    if patience_count >= patience:
                        logging.info(f"Early stopping at iteration {i}")
                        break
                else:
                    patience_count = 0
                
                last_loss = avg_loss
                accum_count = 0
                accumulated_loss = 0
            
            if i % log_freq == 0:
                desc = f"Iters [loss {loss.item():.6f}"
                if use_grad_accum:
                    desc += f", accum {accum_count}/{effective_accum_steps}"
                desc += "]"
                pbar.set_description(desc)
        
        if use_grad_accum and accum_count > 0:
            optimizer.step()
            optimizer.zero_grad()

    def map(self, x0, class_label, n_steps=50, return_traj=False, method="euler"):
        """
        Map from source to target for a specific class.
        
        Supports class interpolation: non-integer class_label values will 
        interpolate between classes (e.g., class_label=0.5 interpolates 
        between class 0 and class 1).
        Returns:
            Generated weights at target class (or trajectory if return_traj=True)
        """
        if self.best_model_state is not None:
            current_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            self.model.load_state_dict(self.best_model_state)

        self.model.eval()
        batch_size = x0.size(0)
        
        # Handle class label - supports scalars, floats, and tensors
        if isinstance(class_label, (int, float)):
            c = torch.full((batch_size, 1), float(class_label), device=self.device, dtype=torch.float32)
        else:
            c = class_label.to(self.device).float()
            if c.dim() == 1:
                c = c.unsqueeze(-1)
        
        traj = [x0.detach().clone()] if return_traj else None
        xt = x0.clone()
        times = torch.linspace(0, 1, n_steps, device=self.device)
        dt = times[1] - times[0]

        for i, t in enumerate(times[:-1]):
            with torch.no_grad():
                t_tensor = torch.ones(batch_size, 1, device=self.device) * t
                pred = self.model(xt, t_tensor, c)
                if pred.dim() > 2:
                    pred = pred.squeeze(-1)
                
                vt = pred if self.mode == "velocity" else pred - xt
                
                if method == "euler":
                    xt = xt + vt * dt
                elif method == "rk4":
                    k1 = vt
                    k2 = self.model(xt + 0.5 * dt * k1, t_tensor + 0.5 * dt, c)
                    if k2.dim() > 2:
                        k2 = k2.squeeze(-1)
                    k2 = k2 if self.mode == "velocity" else k2 - (xt + 0.5 * dt * k1)
                    
                    k3 = self.model(xt + 0.5 * dt * k2, t_tensor + 0.5 * dt, c)
                    if k3.dim() > 2:
                        k3 = k3.squeeze(-1)
                    k3 = k3 if self.mode == "velocity" else k3 - (xt + 0.5 * dt * k2)
                    
                    k4 = self.model(xt + dt * k3, t_tensor + dt, c)
                    if k4.dim() > 2:
                        k4 = k4.squeeze(-1)
                    k4 = k4 if self.mode == "velocity" else k4 - (xt + dt * k3)
                    
                    xt = xt + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

                if return_traj:
                    traj.append(xt.detach().clone())

        if self.best_model_state is not None:
            self.model.load_state_dict(current_state)
        self.model.train()
        return traj if return_traj else xt

    def generate_weights(self, n_samples=10, class_label=0, source_noise_std=0.001, **map_kwargs):
        """Generate weights for a specific class"""
        assert self.input_dim is not None, "Set `self.input_dim` before generating weights."
        source_samples = torch.randn(n_samples, self.input_dim, device=self.device) * source_noise_std
        return self.map(source_samples, class_label=class_label, **map_kwargs)


class MultiClassWeightSpaceFlowModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1, time_embed_dim=64, class_embed_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.time_embed_dim = time_embed_dim
        self.class_embed_dim = class_embed_dim

        self.class_embed = nn.Sequential(
            nn.Linear(1, class_embed_dim), 
            nn.GELU(),
            nn.Linear(class_embed_dim, class_embed_dim)
        )
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim), 
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        
        logging.info(f"hidden_dim:{hidden_dim}")
        
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_embed_dim + class_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2), 
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            
            nn.Linear(hidden_dim, input_dim)
        )
        
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, x, t, c):
        t_embed = self.time_embed(t)
        c_embed = self.class_embed(c)
        combined = torch.cat([x, t_embed, c_embed], dim=-1)
        return self.net(combined)
    

def load_config(config_file='constants.json'):
    """Load configuration file with helpful error messages."""
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, config_file)
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    
    raise FileNotFoundError(
        f"Could not find '{config_file}' in current or script directory.\n"
        f"Current dir: {os.getcwd()}\nScript dir: {script_dir}"
    )


def resolve_model_directory(model_dir_raw, config_path):
    """Resolve model directory path relative to config file or current directory."""
    if os.path.isabs(model_dir_raw):
        return model_dir_raw
    
    # Try current directory first
    if os.path.exists(model_dir_raw):
        return model_dir_raw
    
    # Try relative to config file
    config_dir = os.path.dirname(os.path.abspath(config_path))
    model_dir = os.path.join(config_dir, model_dir_raw)
    if os.path.exists(model_dir):
        return model_dir
    
    # Try parent of config directory
    parent_dir = os.path.dirname(config_dir)
    model_dir = os.path.normpath(os.path.join(parent_dir, model_dir_raw))
    if os.path.exists(model_dir):
        return model_dir
    
    raise FileNotFoundError(
        f"Could not find model directory: {model_dir_raw}\n"
        f"Tried multiple locations relative to config and current directory."
    )


def get_data_loader(dataset_name, batch_size=32, train=False):
    """Get PyTorch DataLoader for specified dataset."""
    from utils import load_mnist, load_fashion_mnist, load_iris_dataset, load_cifar10
    
    dataset_name = dataset_name.lower()
    loaders = {
        "mnist": load_mnist,
        "fashion_mnist": load_fashion_mnist,
        "iris": load_iris_dataset,
        "cifar10": load_cifar10
    }
    
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return loaders[dataset_name](batch_size=batch_size)


# ============================================================================
# Model Conversion and Reconstruction
# ============================================================================

def convert_models_to_weight_space(models, model_config):
    """Convert PyTorch models to weight space objects."""
    weight_space_objects = []
    
    for model in tqdm(models, desc="Converting to weight space"):
        weights, biases = [], []
        for name, param in model.named_parameters():
            param = param.detach().to(device)
            
            # Handle invalid values
            if torch.isnan(param).any() or torch.isinf(param).any():
                logging.warning(f"NaN/Inf in {name}, replacing with zeros")
                param = torch.zeros_like(param)
            
            if "weight" in name:
                weights.append(param.clone())
            elif "bias" in name:
                biases.append(param.clone())
        
        wso = WeightSpaceObjectMLP(weights, biases)
        weight_space_objects.append(wso)
    
    return weight_space_objects


def reconstruct_mlp_models(weights_flat, model_config, dataset):
    """Reconstruct MLP models from flattened weights."""
    generated_models = []
    n_samples = weights_flat.shape[0]
    
    # Determine model class based on dataset
    model_classes = {
        'fashion_mnist': MLP_Fashion_MNIST,
        'mnist': MLP_MNIST,
        'iris': MLP_Iris
    }
    
    dataset_key = next((k for k in model_classes if k in dataset.lower()), None)
    if dataset_key is None:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    ModelClass = model_classes[dataset_key]
    
    for i in range(n_samples):
        # Convert flat weights to weight space object
        wso = WeightSpaceObjectMLP.from_flat(
            weights_flat[i],
            layers=np.array(model_config['layer_layout']),
            device=device
        )
        
        # Create model and load weights
        model = ModelClass()
        for idx in range(len(wso.weights)):
            layer = getattr(model, f'fc{idx+1}')
            layer.weight.data = wso.weights[idx].clone()
            layer.bias.data = wso.biases[idx].clone()
        
        generated_models.append(model.to(device))
    
    return generated_models


def reconstruct_resnet_models(weights_flat, model_name):
    """Reconstruct ResNet models from flattened weights."""
    generated_models = []
    n_samples = weights_flat.shape[0]
    
    # Get template model for shapes
    model_template = ResNet20()  # Extend this for other ResNet variants
    weight_shapes, bias_shapes = [], []
    
    for name, param in model_template.named_parameters():
        if "weight" in name:
            weight_shapes.append(tuple(param.shape))
        elif "bias" in name:
            bias_shapes.append(tuple(param.shape))
    
    # Reconstruct each model
    for i in range(n_samples):
        wso = WeightSpaceObjectResnet.from_flat(
            torch.tensor(weights_flat[i], dtype=torch.float32, device=device),
            weight_shapes,
            bias_shapes,
            device=device
        )
        
        model = ResNet20()
        
        # Load weights into model
        param_dict = {}
        weight_idx, bias_idx = 0, 0
        for name, param in model.named_parameters():
            if "weight" in name:
                param_dict[name] = wso.weights[weight_idx]
                weight_idx += 1
            elif "bias" in name:
                param_dict[name] = wso.biases[bias_idx]
                bias_idx += 1
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in param_dict:
                    param.copy_(param_dict[name])
        
        # Recalibrate batch normalization statistics
        model = recalibrate_bn_stats(model, device)
        generated_models.append(model.to(device))
    
    return generated_models


# ============================================================================
# Class Data Preparation with PCA Support
# ============================================================================

class ClassDataPreparator:
    """Handles data preparation for a single class with optional PCA."""
    
    def __init__(self, models, model_config, class_label):
        self.models = models
        self.model_config = model_config
        self.class_label = class_label
        self.ipca = None
        self.flat_dim = None
        self.actual_dim = None
    
    def prepare(self):
        """
        Convert models to weight space and optionally apply PCA.
        
        Returns:
            tuple: (target_tensor, labels, ipca, flat_dim, actual_dim)
        """
        # Convert to weight space
        weight_space_objects = convert_models_to_weight_space(
            self.models, self.model_config
        )
        flat_weights = torch.stack([
            wso.flatten(device) for wso in weight_space_objects
        ]).to(device)
        
        self.flat_dim = flat_weights.shape[1]
        print(f"Class {self.class_label} weight space dimension: {self.flat_dim:,}")
        
        # Apply PCA if configured
        use_pca = self.model_config.get('use_pca', False)
        pca_components = self.model_config.get('pca_components')
        
        if use_pca and pca_components:
            print(f"Applying PCA: {self.flat_dim:,} → {pca_components} dimensions")
            
            self.ipca = IncrementalPCA(
                n_components=pca_components,
                batch_size=min(10, len(flat_weights))
            )
            
            # Fit and transform
            flat_latent = self.ipca.fit_transform(flat_weights.cpu().numpy())
            target_tensor = torch.tensor(flat_latent, dtype=torch.float32)
            self.actual_dim = pca_components
            
            variance_explained = self.ipca.explained_variance_ratio_.sum()
            print(f"  Variance explained: {variance_explained:.4f}")
        else:
            target_tensor = flat_weights
            self.actual_dim = self.flat_dim
        
        # Create class labels
        labels = torch.full(
            [target_tensor.shape[0], 1],
            self.class_label,
            dtype=torch.float32
        )
        
        return target_tensor, labels, self.ipca, self.flat_dim, self.actual_dim


def prepare_all_class_data(model_classes, config, args, training_mode):
    """
    Prepare data for all classes with validation.
    
    Args:
        model_classes: List of (model_name, class_label) tuples
        config: Configuration dictionary
        args: Command line arguments
        training_mode: 'with_gitrebasin' or 'without_rebasin'
    
    Returns:
        list: Class data dictionaries for each class
    """
    all_class_data = []
    
    for model_name, class_label in model_classes:
        print(f"\n{'-'*60}")
        print(f"Preparing Class {class_label}: {model_name}")
        print(f"{'-'*60}")
        
        model_config = config['models'][model_name]
        model_dir_raw = config['directories'][model_name]
        
        # Resolve model directory path
        model_dir = resolve_model_directory(model_dir_raw, args.config)
        print(f"Model directory: {model_dir}")
        
        pretrained_model_name = model_config.get('pretrained_model_name', 'mlp_seed')
        
        # Load models (original and permuted)
        org_models, permuted_models = get_permuted_models_data(
            model_name=model_name,
            model_dir=model_dir,
            pretrained_model_name=pretrained_model_name,
            num_models=args.num_models,
            ref_point=args.ref_point,
            device=device,
            model_config=model_config
        )
        
        # Select models based on training mode
        models_to_use = (permuted_models if training_mode == "with_gitrebasin" 
                        else org_models)
        
        # Prepare weight space data
        preparator = ClassDataPreparator(models_to_use, model_config, class_label)
        target_tensor, labels, ipca, flat_dim, actual_dim = preparator.prepare()
        
        # Store class data
        all_class_data.append({
            'model_name': model_name,
            'class_label': class_label,
            'target_tensor': target_tensor,
            'labels': labels,
            'ipca': ipca,
            'flat_dim': flat_dim,
            'actual_dim': actual_dim,
            'model_config': model_config,
            'training_mode': training_mode,
            'dataset': model_config['dataset'],
            'model_dir': model_dir,
            'pretrained_model_name': pretrained_model_name,
            'org_models': org_models,
            'permuted_models': permuted_models
        })
    
    # Validate dimensional consistency
    validate_class_dimensions(all_class_data)
    
    return all_class_data


def validate_class_dimensions(class_data_list):
    """
    Validate that all classes have the same dimensionality.
    
    Raises:
        ValueError: If dimensions don't match across classes
    """
    if len(class_data_list) < 2:
        return
    
    dims = [d['actual_dim'] for d in class_data_list]
    flat_dims = [d['flat_dim'] for d in class_data_list]
    
    if not all(d == dims[0] for d in dims):
        print("ERROR: Dimension mismatch across classes!")
        print("Common causes:")
        print("  1. Different model architectures (check layer_layout)")
        print("  2. Different PCA components across classes")
        print("  3. Different parameter counts")
        print("\nDimensions found:")
        for class_data in class_data_list:
            print(f"  Class {class_data['class_label']} ({class_data['model_name']}): "
                  f"flat={class_data['flat_dim']:,}, actual={class_data['actual_dim']:,}")
        raise ValueError(f"Dimension mismatch: {dims}")
    
    actual_dim = dims[0]
    print(f"\nAll {len(class_data_list)} classes aligned at dimension: {actual_dim:,}")
    
    if class_data_list[0]['ipca'] is not None:
        print(f"  (Original dimension: {flat_dims[0]:,}, reduced via PCA)")


# ============================================================================
# Generation and Evaluation
# ============================================================================

def generate_at_class_label(cfm, class_label, n_samples, actual_dim, source_std,
                           config, ipca=None):
    """
    Generate models at a specific class label (supports interpolation).
    
    Args:
        cfm: MultiClassFlowMatching instance
        class_label: Target class (int or float for interpolation)
        n_samples: Number of models to generate
        actual_dim: Dimension of weight space
        source_std: Standard deviation of source noise
        config: Configuration dict with integration settings
        ipca: PCA transformer to invert (optional)
    
    Returns:
        torch.Tensor: Generated weights [n_samples, flat_dim]
    """
    # Sample from source distribution
    random_flat = torch.randn(n_samples, actual_dim, device=device) * source_std
    
    # Map to target via ODE integration
    new_weights_flat = cfm.map(
        random_flat,
        class_label=class_label,
        n_steps=config['integration_steps'],
        method=config['integration_method']
    )
    
    # Inverse PCA if used
    if ipca is not None:
        new_weights_flat = ipca.inverse_transform(new_weights_flat.cpu().numpy())
        new_weights_flat = torch.tensor(
            new_weights_flat, 
            dtype=torch.float32, 
            device=device
        )
    
    return new_weights_flat


def evaluate_and_compare(generated_models, original_models, test_loader, 
                        device, class_label, model_name):
    """
    Evaluate and compare generated vs original models.
    
    Returns:
        dict: Statistics including means, stds, and differences
    """
    print(f"EVALUATION - Class {class_label} ({model_name})")
    
    # Evaluate original models
    print(f"\n[Original Models - {len(original_models)} models]")
    orig_mean, orig_std = print_stats(original_models, test_loader, device)
    
    # Evaluate generated models
    print(f"\n[Generated Models - {len(generated_models)} models]")
    gen_mean, gen_std = print_stats(generated_models, test_loader, device)
    
    # Calculate differences
    diff = abs(gen_mean - orig_mean)
    rel_diff = abs((gen_mean - orig_mean) / orig_mean * 100)
    
    # Print comparison
    print("COMPARISON:")
    print(f"Original:  {orig_mean:.4f} ± {orig_std:.4f}")
    print(f"Generated: {gen_mean:.4f} ± {gen_std:.4f}")
    print(f"Delta Accuracy: {diff:.4f} ({'+' if gen_mean > orig_mean else ''}"
          f"{rel_diff:.2f}%)")
    
    return {
        'original_mean': orig_mean,
        'original_std': orig_std,
        'generated_mean': gen_mean,
        'generated_std': gen_std,
        'difference': diff,
        'relative_diff_pct': rel_diff
    }


# ============================================================================
# Training and Generation Pipeline
# ============================================================================

def train_flow_model(class_data_list, config, args, training_mode, hidden_dim):
    """
    Train multiclass flow matching model.
    
    Args:
        class_data_list: List of class data dictionaries
        config: Configuration dictionary
        args: Command line arguments
        training_mode: Training mode string
        hidden_dim: Hidden dimension for flow model
    
    Returns:
        MultiClassFlowMatching: Trained flow matching instance
    """
    reference_config = class_data_list[0]['model_config']
    actual_dim = class_data_list[0]['actual_dim']
    
    print(f"Training Flow Model")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Mode: {training_mode}")
    print(f"  Classes: {len(class_data_list)}")
    
    # Create combined target dataset
    target_datasets = [
        TensorDataset(d['target_tensor'], d['labels'])
        for d in class_data_list
    ]
    combined_target_dataset = ConcatDataset(target_datasets)
    
    # Create source (noise) dataset
    source_std = reference_config['source_std']
    total_samples = sum(d['target_tensor'].shape[0] for d in class_data_list)
    source_tensor = torch.randn(total_samples, actual_dim) * source_std
    source_labels = torch.zeros(total_samples, 1)
    source_dataset = TensorDataset(source_tensor, source_labels)
    
    print(f"Source: N(0, {source_std}²), Total samples: {total_samples:,}")
    
    # Create dataloaders
    def collate_fn(batch):
        flats, labs = zip(*batch)
        return torch.stack(flats), torch.stack(labs)
    
    sourceloader = DataLoader(
        source_dataset,
        batch_size=reference_config['batch_size'],
        shuffle=True,
        drop_last=True
    )
    
    targetloader = DataLoader(
        combined_target_dataset,
        batch_size=reference_config['batch_size'],
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    
    # Create flow model
    flow_model = MultiClassWeightSpaceFlowModel(
        actual_dim,
        hidden_dim,
        time_embed_dim=reference_config['time_embed_dim'],
        class_embed_dim=reference_config.get('class_embed_dim', 64),
        dropout=reference_config['dropout']
    ).to(device)
    
    print(f"Flow model parameters: {count_parameters(flow_model):,}")
    
    # Create flow matcher
    cfm = MultiClassFlowMatching(
        sourceloader=sourceloader,
        targetloader=targetloader,
        model=flow_model,
        mode="velocity",
        t_dist=config['t_dist'],
        device=device
    )
    cfm.input_dim = actual_dim
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        flow_model.parameters(),
        lr=reference_config['lr'],
        weight_decay=reference_config['weight_decay'],
        betas=(0.9, 0.95)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=reference_config['n_iters'],
        eta_min=1e-6
    )
    
    # Train
    train_kwargs = {
        'n_iters': reference_config['n_iters'],
        'optimizer': optimizer,
        'scheduler': scheduler,
        'sigma': reference_config['sigma'],
        'patience': reference_config['patience'],
        'log_freq': 10
    }
    
    grad_accum_steps = reference_config.get('gradient_accumulation_steps')
    if grad_accum_steps is not None:
        train_kwargs['accum_steps'] = grad_accum_steps
    
    print("\nStarting training...")
    cfm.train(**train_kwargs)
    print(f"Training complete! Best loss: {cfm.best_loss:.6f}")
    
    # Save checkpoint if requested
    if args.save_models:
        save_flow_checkpoint(cfm, class_data_list, reference_config, 
                           args, training_mode, hidden_dim)
    
    return cfm


def save_flow_checkpoint(cfm, class_data_list, config, args, training_mode, hidden_dim):
    """Save flow model checkpoint."""
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f'flow_model_{training_mode}_hidden{hidden_dim}.pt'
    )
    
    torch.save({
        'model_state_dict': cfm.best_model_state,
        'flow_model_config': {
            'input_dim': class_data_list[0]['actual_dim'],
            'hidden_dim': hidden_dim,
            'time_embed_dim': config['time_embed_dim'],
            'class_embed_dim': config.get('class_embed_dim', 64),
            'dropout': config['dropout']
        },
        'training_config': config,
        'best_loss': cfm.best_loss,
        'num_classes': len(class_data_list),
        'class_info': [(d['model_name'], d['class_label']) for d in class_data_list]
    }, checkpoint_path)
    
    print(f"Saved checkpoint to {checkpoint_path}")


def generate_and_evaluate_all_classes(cfm, class_data_list, config, args):
    """Generate and evaluate models for all classes."""
    n_samples = config['n_samples']
    source_std = config['source_std']
    actual_dim = class_data_list[0]['actual_dim']
    evaluation_results = []
    
    for class_data in class_data_list:
        class_label = class_data['class_label']
        model_name = class_data['model_name']
        
        print(f"Generating {n_samples} models for Class {class_label} ({model_name})")
        
        # Generate weights
        new_weights_flat = generate_at_class_label(
            cfm, class_label, n_samples, actual_dim, source_std,
            config, ipca=class_data['ipca']
        )
        
        # Reconstruct models
        if "mlp" in model_name:
            generated_models = reconstruct_mlp_models(
                new_weights_flat,
                class_data['model_config'],
                class_data['dataset']
            )
        elif "resnet" in model_name.lower():
            generated_models = reconstruct_resnet_models(
                new_weights_flat,
                model_name
            )
        else:
            raise ValueError(f"Unknown model type: {model_name}")
        
        # Evaluate
        test_loader = get_data_loader(class_data['dataset'], batch_size=32)
        training_mode = class_data['training_mode']
        original_models = (class_data['permuted_models'] if training_mode == "with_gitrebasin"
                          else class_data['org_models'])
        original_models_subset = original_models[:n_samples]
        
        eval_stats = evaluate_and_compare(
            generated_models,
            original_models_subset,
            test_loader,
            device,
            class_label,
            model_name
        )
        
        eval_stats.update({
            'class_label': class_label,
            'model_name': model_name
        })
        evaluation_results.append(eval_stats)
        
        # Save class-specific stats
        save_class_stats(eval_stats, class_data, config, args)
        
        # Cleanup
        del generated_models
        torch.cuda.empty_cache()
    
    return evaluation_results


def save_class_stats(eval_stats, class_data, config, args):
    """Save generation statistics for a single class."""
    stats_file = f'generation_stats_class{class_data["class_label"]}.json'
    
    with open(stats_file, 'w') as f:
        json.dump({
            'class_label': int(class_data['class_label']),
            'model_name': class_data['model_name'],
            'n_samples': config['n_samples'],
            'original_mean_accuracy': float(eval_stats['original_mean']),
            'original_std_accuracy': float(eval_stats['original_std']),
            'generated_mean_accuracy': float(eval_stats['generated_mean']),
            'generated_std_accuracy': float(eval_stats['generated_std']),
            'accuracy_difference': float(eval_stats['difference']),
            'relative_diff_pct': float(eval_stats['relative_diff_pct']),
            'training_mode': class_data['training_mode'],
            'used_pca': class_data['ipca'] is not None,
            'pca_components': class_data['actual_dim'] if class_data['ipca'] else None
        }, f, indent=2)


def print_final_summary(evaluation_results, training_mode, hidden_dim):
    """Print final summary of all classes."""
    print(f"FINAL SUMMARY - {training_mode}, hidden_dim={hidden_dim}")
    
    for result in evaluation_results:
        print(f"\nClass {result['class_label']} ({result['model_name']}):")
        print(f"  Original:  {result['original_mean']:.4f} ± {result['original_std']:.4f}")
        print(f"  Generated: {result['generated_mean']:.4f} ± {result['generated_std']:.4f}")
        print(f"  Δ Accuracy: {result['difference']:.4f} ({result['relative_diff_pct']:.2f}%)")
    


# ============================================================================
# Main Pipeline
# ============================================================================

def train_and_generate(args):
    """Main training and generation pipeline."""
    config = load_config(args.config)
    
    # Define classes to train on
    # Format: (model_name_in_config, class_label)
    model_classes = [
        ('mlp_mnist', 0),
        ('mlp_fashion_mnist', 1),
        # Add more classes here as needed
    ]
    
    print(f"Training multiclass flow matching for {len(model_classes)} classes")
    print("Conditioning: N(0,1) + class_embed → target weights")
    
    # Process each training mode
    for training_mode in config['training_modes']:
        if args.mode and training_mode != args.mode:
            continue
        
        print(f"TRAINING MODE: {training_mode}")
        
        # Prepare data for all classes
        class_data_list = prepare_all_class_data(
            model_classes, config, args, training_mode
        )
        
        reference_config = class_data_list[0]['model_config']
        
        # Train for each hidden dimension
        for hidden_dim in reference_config['flow_hidden_dims']:
            if args.hidden_dim and hidden_dim != args.hidden_dim:
                continue
            
            # Train flow model
            cfm = train_flow_model(
                class_data_list, config, args, training_mode, hidden_dim
            )
            
            # Generate and evaluate
            evaluation_results = generate_and_evaluate_all_classes(
                cfm, class_data_list, reference_config, args
            )
            
            # Print summary
            print_final_summary(evaluation_results, training_mode, hidden_dim)
            
            # Cleanup
            del cfm
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description='Train multiclass flow matching for neural network weight generation'
    )
    parser.add_argument('--config', type=str, default='constants.json',
                       help='Configuration file path')
    parser.add_argument('--num_models', type=int, default=100,
                       help='Number of pretrained models per class')
    parser.add_argument('--ref_point', type=int, default=0,
                       help='Reference model index for canonicalization')
    parser.add_argument('--hidden_dim', type=int, default=None,
                       help='Specific hidden dimension (Preferred 2048)')
    parser.add_argument('--mode', type=str, default=None,
                       choices=['with_gitrebasin', 'without_rebasin'],
                       help='Training mode (tests all if None)')
    parser.add_argument('--save_models', action='store_true',
                       help='Save generated models and checkpoints')
    parser.add_argument('--output_dir', type=str, default='./generated_models',
                       help='Directory for saved models')
    
    args = parser.parse_args()
    
    print("MULTICLASS FLOW MATCHING - Neural Network Weight Generation")
    print(f"Config: {args.config}")
    print(f"Models per class: {args.num_models}")
    print(f"Save models: {args.save_models}")
    
    train_and_generate(args)
    print("\nTraining and generation complete!")


if __name__ == "__main__":
    main()
