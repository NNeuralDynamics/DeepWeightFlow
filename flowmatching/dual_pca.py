import gc
import numpy as np
from tqdm import tqdm
import torch
from sklearn.utils.extmath import randomized_svd
from typing import Optional, Tuple, Callable

import tempfile

class BatchedCovariancePCA:
    """
    GPU-accelerated incremental covariance-based PCA.
    
    Improvements over CPU version:
    - 10-50x faster matrix operations using GPU (cuBLAS)
    - Vectorized component computation (all components at once)
    - Same memory footprint as CPU version (true streaming, no caching)
    - Automatic fallback to CPU if GPU unavailable
    - Support for FP16 precision (2x memory reduction)
    - Micro-batching for ultra-large models (1B+ parameters)
    
    Memory footprint: O(micro_batch_size × n_params + n_models²)
    - Only 2 micro-batches in memory at once during covariance computation
    - Covariance matrix (n_models²) stays in CPU memory
    """
    
    def __init__(self, n_components: int, seed: int = 42, device: str = "cuda"):
        """
        Initialize PCA.
        
        Args:
            n_components: Number of principal components to compute
            seed: Random seed for reproducibility
            device: 'cuda' for GPU or 'cpu' for CPU computation
        """
        self.n_components = n_components
        self.seed = seed
        self.device = device if torch.cuda.is_available() else "cpu"
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.n_models_ = None
        self.n_params_ = None
        
        if self.device == "cuda":
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU Memory: {mem_gb:.1f} GB")
        else:
            print("GPU not available, using CPU")
    
    def fit(self, model_loader_func: Callable, n_models: int, batch_size: int = 20,
            use_fp16: bool = False, micro_batch_size: Optional[int] = None):
        """
        Fit PCA by building covariance matrix incrementally.
        
        Args:
            model_loader_func: Callable(start_idx, end_idx) -> array (n_params, n_models_in_batch)
                               Function that loads model parameters for the given index range
            n_models: Total number of models
            batch_size: Logical batch size (for loading from disk)
            use_fp16: Use FP16 for GPU computation (2x memory reduction, slight precision loss)
            micro_batch_size: Further split batches for ultra-large models (e.g., 1B params)
                             If None, uses batch_size directly
        
        Returns:
            self: Fitted PCA object
        """
        print(f"Building covariance matrix for {n_models} models...")
        print(f"Batch size: {batch_size} models at a time")
        if use_fp16:
            print("Using FP16 precision (2x memory reduction)")
        if micro_batch_size:
            print(f"Using micro-batching: {micro_batch_size} models per micro-batch")
        
        self.n_models_ = n_models
        compute_dtype = torch.float16 if use_fp16 else torch.float32
        
        print("Pass 1: Computing mean...")
        mean_accumulator = None
        
        for start_idx in tqdm(range(0, n_models, batch_size), desc="Mean computation"):
            end_idx = min(start_idx + batch_size, n_models)
            batch = model_loader_func(start_idx, end_idx)
            
            if mean_accumulator is None:
                self.n_params_ = batch.shape[0]
                mean_accumulator = torch.zeros(self.n_params_, dtype=torch.float32, device=self.device)
            
            batch_torch = torch.from_numpy(batch).to(self.device, dtype=compute_dtype)
            mean_accumulator += torch.sum(batch_torch, dim=1, dtype=torch.float32)
            
            del batch, batch_torch
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
        
        self.mean_ = (mean_accumulator / n_models).cpu().numpy()
        del mean_accumulator
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        print(f"Pass 2: Building covariance matrix ({n_models}×{n_models}) - streaming...")
        
        # Keep covariance in CPU memory (n_models² is typically small)
        C = np.zeros((n_models, n_models), dtype=np.float32)
        mean_torch = torch.from_numpy(self.mean_).to(self.device, dtype=compute_dtype)
        
        # Use micro-batching for ultra-large models if specified
        effective_batch_size = micro_batch_size if micro_batch_size else batch_size
        
        for i_start in tqdm(range(0, n_models, effective_batch_size), desc="Covariance rows"):
            i_end = min(i_start + effective_batch_size, n_models)
            
            # Load batch_i once per outer iteration
            batch_i = model_loader_func(i_start, i_end)
            batch_i_torch = torch.from_numpy(batch_i).to(self.device, dtype=compute_dtype)
            batch_i_centered = batch_i_torch - mean_torch.unsqueeze(1)
            
            # Inner loop: load batch_j, compute, immediately discard
            for j_start in range(0, n_models, effective_batch_size):
                j_end = min(j_start + effective_batch_size, n_models)
                
                batch_j = model_loader_func(j_start, j_end)
                batch_j_torch = torch.from_numpy(batch_j).to(self.device, dtype=compute_dtype)
                batch_j_centered = batch_j_torch - mean_torch.unsqueeze(1)
                
                # GPU-accelerated matmul: (batch_i, n_params) @ (n_params, batch_j)
                # This single operation is 10-50x faster on GPU than CPU
                cov_block = (batch_i_centered.T @ batch_j_centered).cpu().float().numpy()
                C[i_start:i_end, j_start:j_end] = cov_block
                
                # Immediately free memory
                del batch_j, batch_j_torch, batch_j_centered, cov_block
                if self.device == "cuda":
                    torch.cuda.empty_cache()
            
            del batch_i, batch_i_torch, batch_i_centered
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
        
        del mean_torch
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        # Pass 3: SVD on covariance matrix
        print(f"Pass 3: SVD on covariance matrix ({n_models}×{n_models})...")
        n_components = min(self.n_components, n_models - 1)
        
        U, S, _ = randomized_svd(
            C, 
            n_components=n_components,
            n_iter=5,
            random_state=self.seed
        )
        
        eigenvalues = (S) / (n_models - 1) #changed
        
        del C
        gc.collect()
        
        # Pass 4: Compute components in original space - VECTORIZED
        print("Computing principal components in original space...")
        
        # Components accumulator stays on GPU (small: n_components × n_params)
        components = torch.zeros((n_components, self.n_params_), 
                                 dtype=compute_dtype, device=self.device)
        U_torch = torch.from_numpy(U).to(self.device, dtype=compute_dtype)
        mean_torch = torch.from_numpy(self.mean_).to(self.device, dtype=compute_dtype)
        
        for start_idx in tqdm(range(0, n_models, effective_batch_size), desc="Computing components"):
            end_idx = min(start_idx + effective_batch_size, n_models)
            
            batch = model_loader_func(start_idx, end_idx)
            batch_torch = torch.from_numpy(batch).to(self.device, dtype=compute_dtype)
            batch_centered = batch_torch - mean_torch.unsqueeze(1)
            
            U_batch = U_torch[start_idx:end_idx, :]  # (batch_size, n_components)
            
            # VECTORIZED: Compute all components at once (10-500x faster than loop)
            # (n_params, batch_size) @ (batch_size, n_components) = (n_params, n_components)
            components += (batch_centered @ U_batch).T
            
            del batch, batch_torch, batch_centered
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
        
        # Normalize components to unit length
        components_norm = torch.norm(components, dim=1, keepdim=True)
        components = components / torch.clamp(components_norm, min=1e-10)
        
        self.components_ = components.cpu().float().numpy()
        self.explained_variance_ = eigenvalues[:n_components]
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = eigenvalues[:n_components] / total_variance
        
        del components, U_torch, mean_torch
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        print(f"Fitted PCA with {n_components} components")
        print(f"Cumulative variance: {np.sum(self.explained_variance_ratio_):.2%}")
        
        return self
    
    def transform(self, model_loader_func: Callable, n_models: int, 
                  batch_size: int = 20, output_file: Optional[str] = None,
                  use_fp16: bool = False) -> str:
        """
        Project models to PCA space.
        
        Args:
            model_loader_func: Callable(start_idx, end_idx) -> array (n_params, n_models_in_batch)
            n_models: Total number of models to transform
            batch_size: Models to load at once
            output_file: Path to save projected data (optional, creates temp file if None)
            use_fp16: Use FP16 for computation (should match fit())
        
        Returns:
            Path to memmap file with projected data (n_models, n_components)
        """
        if output_file is None:
            output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".npy").name
        
        projected = np.memmap(
            output_file, 
            dtype=np.float32, 
            mode="w+", 
            shape=(n_models, self.n_components)
        )
        
        print(f"Transforming {n_models} models to {self.n_components}-dimensional space...")
        
        compute_dtype = torch.float16 if use_fp16 else torch.float32
        mean_torch = torch.from_numpy(self.mean_).to(self.device, dtype=compute_dtype)
        components_torch = torch.from_numpy(self.components_).to(self.device, dtype=compute_dtype)
        
        idx = 0
        for start_idx in tqdm(range(0, n_models, batch_size), desc="Transforming"):
            end_idx = min(start_idx + batch_size, n_models)
            
            batch = model_loader_func(start_idx, end_idx)
            batch_torch = torch.from_numpy(batch).to(self.device, dtype=compute_dtype)
            batch_centered = batch_torch - mean_torch.unsqueeze(1)
            
            # Project: (batch_size, n_params) @ (n_components, n_params)^T
            # GPU-accelerated transformation
            batch_projected = (batch_centered.T @ components_torch.T).cpu().float().numpy()
            
            projected[idx:idx + batch_projected.shape[0]] = batch_projected
            idx += batch_projected.shape[0]
            
            del batch, batch_torch, batch_centered, batch_projected
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
        
        projected.flush()
        print(f"✓ Projected data saved to {output_file}")
        
        del mean_torch, components_torch
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        return output_file
    
    def inverse_transform(self, latent_vectors: np.ndarray) -> np.ndarray:
        """
        Reconstruct original models from PCA coordinates.
        
        Args:
            latent_vectors: Array of shape (n_samples, n_components)
        
        Returns:
            Reconstructed models of shape (n_samples, n_params)
        """
        # X_reconstructed = latent @ components + mean
        return latent_vectors @ self.components_ + self.mean_


def flatten_and_project_to_disk_maxcomponents(
    model_loader_func: Callable,
    n_models: int = 100,
    n_components: Optional[int] = None,  # None = use maximum (n_models - 1)
    target_variance: Optional[float] = None,  # e.g., 0.99 for 99% variance
    batch_size: int = 20,
    output_file: Optional[str] = None,
    use_fp16: bool = True,
    micro_batch_size: Optional[int] = None,
    device: str = "cuda"
) -> Tuple[str, BatchedCovariancePCA, dict]:
    """
    PCA with maximum possible components (up to n_models).
    
    Args:
        model_loader_func: Callable(start_idx, end_idx) -> array (n_params, n_models_in_batch)
        n_models: Total number of models (e.g., 100)
        n_components: Number of components (None = auto-determine)
        target_variance: If set, auto-determine components for this variance (e.g., 0.99)
        batch_size: Models to load at once (e.g., 20)
        use_fp16: Use FP16 precision (2x memory reduction)
        micro_batch_size: Split batches for large models (e.g., 5 for 1B params)
        device: 'cuda' or 'cpu'
        
    Returns:
        output_file: Path to projected data
        pca: Fitted PCA object
        info: Dictionary with variance info
    """
    # First fit with maximum components
    max_components = n_models - 1  # Maximum mathematically possible
    
    if n_components is None and target_variance is None:
        n_components = max_components
        print(f"Using maximum components: {n_components}")
    elif target_variance is not None:
        print("Fitting with maximum components to determine optimal number...")
        pca_temp = BatchedCovariancePCA(n_components=max_components, device=device)
        pca_temp.fit(model_loader_func, n_models, batch_size, 
                     use_fp16=use_fp16, micro_batch_size=micro_batch_size)
        
        cumulative_variance = np.cumsum(pca_temp.explained_variance_ratio_)
        n_components = np.searchsorted(cumulative_variance, target_variance) + 1
        n_components = min(n_components, max_components)
        
        print(f"Auto-selected {n_components} components for {target_variance:.1%} variance")
        print(f"Actual variance: {cumulative_variance[n_components-1]:.2%}")
    
    # Fit with determined number of components
    pca = BatchedCovariancePCA(n_components=n_components, device=device)
    pca.fit(model_loader_func, n_models, batch_size,
            use_fp16=use_fp16, micro_batch_size=micro_batch_size)
    
    # Transform
    output_file = pca.transform(model_loader_func, n_models, batch_size, 
                                output_file, use_fp16=use_fp16)
    
    # Info
    info = {
        'n_components': n_components,
        'explained_variance': pca.explained_variance_,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
        'total_variance_captured': np.sum(pca.explained_variance_ratio_)
    }
    
    print(f"COMPLETE!")
    print(f"Components: {n_components}/{n_models}")
    print(f"Variance captured: {info['total_variance_captured']:.4%}")
    print(f"Output shape: ({n_models}, {n_components})")
    print(f"Output file: {output_file}")
    
    return output_file, pca, info