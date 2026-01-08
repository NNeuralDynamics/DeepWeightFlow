"""
Unified weight matching module - SINGLE SOURCE for all weight matching functions.
This resolves conflicts between different implementations.
"""
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from permutation_specs import PermutationSpec


def get_permuted_param(ps: PermutationSpec, perm: dict, wk: str, params_b: dict, 
                       except_axis=None) -> torch.Tensor:
    """
    Get parameter with permutations applied.
    
    Args:
        ps: PermutationSpec
        perm: Dictionary of permutations
        wk: Weight key (parameter name)
        params_b: Parameter dictionary
        except_axis: Axis to skip permutation (for matching)
    
    Returns:
        Permuted parameter tensor
    """
    w = params_b[wk]
    for axis, p in enumerate(ps.axes_to_perm[wk]):
        if p is None or axis == except_axis:
            continue
        
        idx = torch.tensor(perm[p], device=w.device, dtype=torch.long)
        
        # Validate axis size
        if w.shape[axis] != idx.numel():
            raise ValueError(
                f"Axis size mismatch for {wk} axis {axis}: "
                f"weight shape {w.shape[axis]} vs permutation size {idx.numel()}"
            )
        
        w = torch.index_select(w, axis, idx)
    
    return w


def apply_permutation(ps: PermutationSpec, perm: dict, params: dict) -> dict:
    """
    Apply permutation to all parameters.
    
    Args:
        ps: PermutationSpec
        perm: Dictionary of permutations
        params: Parameter dictionary
    
    Returns:
        Dictionary of permuted parameters
    """
    return {k: get_permuted_param(ps, perm, k, params) for k in params.keys()}


def weight_matching(ps: PermutationSpec, params_a: dict, params_b: dict, 
                   max_iter: int = 100, init_perm: dict = None, 
                   silent: bool = True, device=None) -> dict:
    """
    Find permutation of params_b to maximize alignment with params_a.
    Uses Hungarian algorithm (linear sum assignment) for optimal matching.
    
    Args:
        ps: PermutationSpec
        params_a: Reference parameters (target)
        params_b: Parameters to permute (source)
        max_iter: Maximum iterations
        init_perm: Initial permutation (optional)
        silent: If True, don't print progress
        device: Device to use (auto-detect if None)
    
    Returns:
        Dictionary of optimal permutations
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move to device
    params_a = {k: v.to(device) for k, v in params_a.items()}
    params_b = {k: v.to(device) for k, v in params_b.items()}

    # Get permutation sizes
    perm_sizes = {
        p: params_a[axes[0][0]].shape[axes[0][1]] 
        for p, axes in ps.perm_to_axes.items()
    }
    
    # Initialize permutations
    if init_perm is None:
        perm = {p: torch.arange(n, device=device) for p, n in perm_sizes.items()}
    else:
        perm = {p: v.to(device) for p, v in init_perm.items()}
        
    perm_names = list(perm.keys())
    
    # Use fixed seed for reproducibility
    rng = np.random.RandomState(42)

    # Iterative refinement
    for iteration in range(max_iter):
        progress = False
        
        # Randomize order of permutations
        for p_ix in rng.permutation(len(perm_names)):
            p = perm_names[p_ix]
            n = perm_sizes[p]
            
            # Build cost matrix
            A = torch.zeros((n, n), device=device)
            
            for wk, axis in ps.perm_to_axes[p]:
                w_a = params_a[wk]
                w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)

                # Reshape for matrix multiplication
                w_a = w_a.moveaxis(axis, 0).reshape((n, -1))
                w_b = w_b.moveaxis(axis, 0).reshape((n, -1))

                # Accumulate alignment scores
                A += w_a @ w_b.T

            # Solve linear assignment problem (Hungarian algorithm)
            ri, ci = linear_sum_assignment(A.detach().cpu().numpy(), maximize=True)
            assert (ri == np.arange(len(ri))).all()

            # Check if assignment improved
            eye_old = torch.eye(n, device=device)[perm[p]]
            eye_new = torch.eye(n, device=device)[ci]

            oldL = torch.tensordot(A, eye_old, dims=([0, 1], [0, 1]))
            newL = torch.tensordot(A, eye_new, dims=([0, 1], [0, 1]))

            if not silent and newL > oldL + 1e-12:
                print(f"Iter {iteration}/{p}: improvement = {newL.item() - oldL.item():.6f}")

            progress = progress or newL > oldL + 1e-12

            # Update permutation
            perm[p] = torch.tensor(ci, device=device, dtype=torch.long)

        # Early stopping if no progress
        if not progress:
            if not silent:
                print(f"Converged at iteration {iteration}")
            break

    return perm


def validate_permutation(ps: PermutationSpec, perm: dict, params: dict) -> bool:
    """
    Validate that permutation is compatible with parameter shapes.
    
    Args:
        ps: PermutationSpec
        perm: Permutation dictionary
        params: Parameter dictionary
    
    Returns:
        True if valid, False otherwise
    """
    try:
        for wk in params.keys():
            if wk not in ps.axes_to_perm:
                continue
            
            for axis, p in enumerate(ps.axes_to_perm[wk]):
                if p is None:
                    continue
                
                if p not in perm:
                    print(f"Missing permutation {p} for {wk}")
                    return False
                
                if params[wk].shape[axis] != len(perm[p]):
                    print(f"Size mismatch: {wk} axis {axis} has size {params[wk].shape[axis]}, "
                          f"but permutation {p} has size {len(perm[p])}")
                    return False
        
        return True
    
    except Exception as e:
        print(f"Validation error: {e}")
        return False


def compute_alignment_score(ps: PermutationSpec, params_a: dict, params_b: dict, 
                           perm: dict, device=None) -> float:
    """
    Compute alignment score between two parameter sets given a permutation.
    
    Args:
        ps: PermutationSpec
        params_a: Reference parameters
        params_b: Parameters to compare
        perm: Permutation to apply to params_b
        device: Device to use
    
    Returns:
        Alignment score (higher is better)
    """
    if device is None:
        device = params_a[list(params_a.keys())[0]].device
    
    total_score = 0.0
    total_params = 0
    
    for wk in params_a.keys():
        if wk not in ps.axes_to_perm:
            continue
        
        w_a = params_a[wk].to(device)
        w_b = get_permuted_param(ps, perm, wk, params_b)
        
        # Compute dot product (alignment)
        score = torch.sum(w_a * w_b).item()
        total_score += score
        total_params += w_a.numel()
    
    return total_score / total_params if total_params > 0 else 0.0