import torch
import numpy as np
import os
import copy
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from permutation_specs import *
from models import MLP_MNIST, MLP_Fashion_MNIST, MLP_Iris, ResNet20, get_resnet18, create_vit_small
from weight_space_objects import VisionTransformerWeightSpace, BERTWeightSpace
from models import BERTAttentionWeights, BERTTransformerBlockWeights, create_bert_100m
from typing import Any, List, Tuple, Optional
import gc 

def get_permuted_param(ps, perm, wk, params_b, except_axis=None):
    w = params_b[wk]
    for axis, p in enumerate(ps.axes_to_perm[wk]):
        if p is None or axis == except_axis:
            continue
        idx = torch.tensor(perm[p], device=w.device)
        if w.shape[axis] != idx.numel():
            raise ValueError(f"Axis size mismatch for {wk} axis {axis}: {w.shape[axis]} vs {idx.numel()}")
        w = torch.index_select(w, axis, idx)
    return w

def apply_permutation(ps: PermutationSpec, perm, params):
    """Apply permutation to params"""
    return {k: get_permuted_param(ps, perm, k, params) for k in params.keys()}

def weight_matching(ps: PermutationSpec, params_a, params_b, max_iter=100, init_perm=None, silent=True, device=None):
    """Find permutation of params_b to make them match params_a."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params_a = {k: v.to(device) for k, v in params_a.items()}
    params_b = {k: v.to(device) for k, v in params_b.items()}

    perm_sizes = {p: params_a[axes[0][0]].shape[axes[0][1]] 
                  for p, axes in ps.perm_to_axes.items()}
    
    if init_perm is None:
        perm = {p: torch.arange(n, device=device) for p, n in perm_sizes.items()}
    else:
        perm = {p: v.to(device) for p, v in init_perm.items()}
        
    perm_names = list(perm.keys())
    
    rng = np.random.RandomState(42)

    for iteration in range(max_iter):
        progress = False
        
        for p_ix in rng.permutation(len(perm_names)):
            p = perm_names[p_ix]
            n = perm_sizes[p]
            
            A = torch.zeros((n, n), device=device)
            
            for wk, axis in ps.perm_to_axes[p]:
                w_a = params_a[wk]
                w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)

                w_a = w_a.moveaxis(axis, 0).reshape((n, -1))
                w_b = w_b.moveaxis(axis, 0).reshape((n, -1))

                A += w_a @ w_b.T

            ri, ci = linear_sum_assignment(A.detach().cpu().numpy(), maximize=True)

            eye_old = torch.eye(n, device=device)[perm[p]]
            eye_new = torch.eye(n, device=device)[ci]

            oldL = torch.tensordot(A, eye_old, dims=([0, 1], [0, 1]))
            newL = torch.tensordot(A, eye_new, dims=([0, 1], [0, 1]))

            if not silent and newL > oldL + 1e-12:
                print(f"{iteration}/{p}: {newL.item() - oldL.item()}")

            progress = progress or newL > oldL + 1e-12

            perm[p] = torch.tensor(ci, device=device)


        if not progress:
            break

    return perm

class TransFusionMatcher:
    """
    Weight matching using TransFusion approach for Transformers
    """
    
    def __init__(self, num_iterations: int = 5, epsilon: float = 1e-8, model_type: str = 'vit'):
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.model_type = model_type.lower()
        
        if self.model_type not in ['vit', 'bert']:
            raise ValueError(f"model_type must be 'vit' or 'bert', got '{model_type}'")
    
    def compute_spectral_distance(self, weight1: torch.Tensor, weight2: torch.Tensor) -> float:
        """Compute permutation-invariant distance using singular values"""
        try:
            _, s1, _ = torch.svd(weight1)
            _, s2, _ = torch.svd(weight2)
        except:
            _, s1, _ = np.linalg.svd(weight1.cpu().numpy())
            _, s2, _ = np.linalg.svd(weight2.cpu().numpy())
            s1 = torch.tensor(s1)
            s2 = torch.tensor(s2)
        
        max_len = max(len(s1), len(s2))
        if len(s1) < max_len:
            s1 = torch.cat([s1, torch.zeros(max_len - len(s1))])
        if len(s2) < max_len:
            s2 = torch.cat([s2, torch.zeros(max_len - len(s2))])
        
        return torch.norm(s1 - s2).item()

    def compose_attention_permutation(self, inter_head_perm: torch.Tensor,
                                    intra_head_perms: List[torch.Tensor],
                                    d_model: int, num_heads: int) -> torch.Tensor:
        """Compose inter and intra head permutations into a single block diagonal matrix"""
        head_dim = d_model // num_heads
        P_attn = torch.zeros(d_model, d_model)
        
        for i in range(num_heads):
            j = torch.argmax(inter_head_perm[:, i]).item()
            P_intra = intra_head_perms[j] if j < len(intra_head_perms) else torch.eye(head_dim)
            
            start_i = i * head_dim
            end_i = (i + 1) * head_dim
            start_j = j * head_dim
            end_j = (j + 1) * head_dim
            
            P_attn[start_i:end_i, start_j:end_j] = P_intra
        
        return P_attn

    def match_attention_heads(self, attn1, attn2) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """Two-level matching for attention heads"""
        q1, k1, v1 = attn1.split_heads()
        q2, k2, v2 = attn2.split_heads()
        
        num_heads = attn1.num_heads
        d_model = attn1.qkv_weight.shape[1]
        head_dim = d_model // num_heads
        
        distance_matrix = torch.zeros(num_heads, num_heads)
        
        for i in range(num_heads):
            for j in range(num_heads):
                dist_q = self.compute_spectral_distance(q1[i], q2[j])
                dist_k = self.compute_spectral_distance(k1[i], k2[j])
                dist_v = self.compute_spectral_distance(v1[i], v2[j])
                distance_matrix[i, j] = dist_q + dist_k + dist_v
        
        row_ind, col_ind = linear_sum_assignment(distance_matrix.detach().cpu().numpy())
        
        inter_head_perm = torch.zeros(num_heads, num_heads)
        inter_head_perm[row_ind, col_ind] = 1.0
        
        intra_head_perms = []
        for i, j in zip(row_ind, col_ind):
            cost_q = -torch.mm(q2[j], q1[i].t())
            cost_k = -torch.mm(k2[j], k1[i].t())
            cost_v = -torch.mm(v2[j], v1[i].t())
            cost_matrix = (cost_q + cost_k + cost_v) / 3.0
            
            row_ind_intra, col_ind_intra = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
            
            perm = torch.zeros(head_dim, head_dim)
            perm[row_ind_intra, col_ind_intra] = 1.0
            intra_head_perms.append(perm)
        
        composed_perm = self.compose_attention_permutation(
            inter_head_perm, intra_head_perms, d_model, num_heads
        )
        
        return inter_head_perm, intra_head_perms, composed_perm

    def match_mlp_layer(self, weight1: torch.Tensor, weight2: torch.Tensor,
                    prev_perm: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Match MLP layers using Hungarian algorithm"""
        device = weight1.device
        
        if prev_perm is not None:
            prev_perm = prev_perm.to(device)
            if prev_perm.shape[0] == weight1.shape[1]:
                weight1_permuted = torch.mm(weight1, prev_perm.t())
            else:
                weight1_permuted = weight1
        else:
            weight1_permuted = weight1
        
        cost_matrix = -torch.mm(weight2, weight1_permuted.t())
        row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
        
        n = weight1.shape[0]
        perm = torch.zeros(n, n, device=device)
        perm[row_ind, col_ind] = 1.0
        
        return perm
    
    def apply_permutation_to_weights(self, weights, perm_spec) -> Any:
        """Apply computed permutations to weight space"""
        result = copy.deepcopy(weights)
        
        prev_output_perm = None
        
        for block_idx, (source_block, result_block) in enumerate(zip(weights.blocks, result.blocks)):
            block_perm = perm_spec.block_perms[block_idx]
            d_model = source_block.attention.qkv_weight.shape[1]
            device = result_block.attention.qkv_weight.device
            
            if block_perm['attention_out'] is not None:
                P_attn = block_perm['attention_out'].to(device)
                P_attn_expanded = torch.block_diag(P_attn, P_attn, P_attn).to(device)
                
                if prev_output_perm is not None:
                    result_block.attention.qkv_weight = torch.mm(
                        torch.mm(P_attn_expanded, result_block.attention.qkv_weight),
                        prev_output_perm.t()
                    )
                else:
                    result_block.attention.qkv_weight = torch.mm(
                        P_attn_expanded, result_block.attention.qkv_weight
                    )
                
                if result_block.attention.qkv_bias is not None:
                    result_block.attention.qkv_bias = torch.mv(
                        P_attn_expanded, result_block.attention.qkv_bias
                    )
                
                result_block.attention.proj_weight = torch.mm(
                    result_block.attention.proj_weight, P_attn.t()
                )
            
            if len(result_block.mlp_weights) >= 2 and block_perm['mlp1'] is not None:
                P_mlp1 = block_perm['mlp1'].to(device)
                
                result_block.mlp_weights = list(result_block.mlp_weights)
                if prev_output_perm is not None:
                    result_block.mlp_weights[0] = torch.mm(
                        torch.mm(P_mlp1, result_block.mlp_weights[0]),
                        prev_output_perm.t()
                    )
                else:
                    result_block.mlp_weights[0] = torch.mm(P_mlp1, result_block.mlp_weights[0])
                
                if len(result_block.mlp_biases) > 0:
                    result_block.mlp_biases = list(result_block.mlp_biases)
                    result_block.mlp_biases[0] = torch.mv(P_mlp1, result_block.mlp_biases[0])
                
                result_block.mlp_weights[1] = torch.mm(result_block.mlp_weights[1], P_mlp1.t())
                
                result_block.mlp_weights = tuple(result_block.mlp_weights)
                result_block.mlp_biases = tuple(result_block.mlp_biases)
                
                prev_output_perm = torch.eye(d_model, device=device)
            else:
                prev_output_perm = torch.eye(d_model, device=device)
        
        return result
    
    def canonicalize_model(self, models: List[Any], reference_idx: int = 0) -> List[Any]:
        """Canonicalize multiple models using one as reference"""
        # Import the appropriate PermutationSpec based on model type
        if self.model_type == 'vit':
            from permutation_specs import ViTPermutationSpec
            PermutationSpec = ViTPermutationSpec
        else:  # bert
            from permutation_specs import BERTPermutationSpec
            PermutationSpec = BERTPermutationSpec
        
        reference = models[reference_idx]
        canonicalized = []
        
        for i, model in enumerate(models):
            if i == reference_idx:
                canonicalized.append(reference)
            else:
                perm_spec = PermutationSpec(len(model.blocks))
                current_model = copy.deepcopy(model)
                
                for iteration in range(self.num_iterations):
                    current_dim_perm = None
                    
                    for block_idx in range(len(current_model.blocks)):
                        current_block = current_model.blocks[block_idx]
                        reference_block = reference.blocks[block_idx]
                        
                        d_model = current_block.attention.qkv_weight.shape[1]
                        
                        inter_perm, intra_perms, composed_perm = self.match_attention_heads(
                            current_block.attention, reference_block.attention
                        )
                        perm_spec.set_block_perm(block_idx, 'attention_out', composed_perm)
                        
                        if len(current_block.mlp_weights) >= 1:
                            mlp1_perm = self.match_mlp_layer(
                                current_block.mlp_weights[0],
                                reference_block.mlp_weights[0],
                                prev_perm=current_dim_perm
                            )
                            perm_spec.set_block_perm(block_idx, 'mlp1', mlp1_perm)
                            current_dim_perm = torch.eye(d_model)
                    
                    current_model = self.apply_permutation_to_weights(current_model, perm_spec)
                
                canonicalized.append(current_model)
        
        return canonicalized

def get_model_and_spec(model_name: str, device=None):
    """Return model constructor and permutation spec based on model_name"""
    model_name = model_name.lower()
    
    if "mlp_mnist" in model_name and "fashion" not in model_name:
        return MLP_MNIST().to(device), mnist_mlp_permutation_spec_mlp()
    elif "mlp_fashion_mnist" in model_name:
        return MLP_Fashion_MNIST().to(device), fashion_mnist_mlp_permutation_spec()
    elif "mlp_iris" in model_name:
        return MLP_Iris().to(device), iris_mlp_permutation_spec_mlp()
    elif "resnet20" in model_name:
        return ResNet20().to(device), resnet20_permutation_spec()
    elif "resnet18" in model_name:
        return get_resnet18().to(device), resnet18_permutation_spec()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def aggressive_cleanup():
    """Force garbage collection and GPU memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def move_weight_space_to_cpu(ws):
    """
    Move to CPU without deep copy when possible
    """
    # Create new weight space with CPU tensors (no deepcopy)
    blocks_cpu = []
    for block in ws.blocks:
        # Create new block objects but share structure
        attn_cpu = BERTAttentionWeights(
            qkv_weight=block.attention.qkv_weight.cpu(),
            qkv_bias=block.attention.qkv_bias.cpu() if block.attention.qkv_bias is not None else None,
            proj_weight=block.attention.proj_weight.cpu(),
            proj_bias=block.attention.proj_bias.cpu() if block.attention.proj_bias is not None else None,
            num_heads=block.attention.num_heads
        )
        
        block_cpu = BERTTransformerBlockWeights(
            attention=attn_cpu,
            norm1_weight=block.norm1_weight.cpu(),
            norm1_bias=block.norm1_bias.cpu(),
            mlp_weights=tuple(w.cpu() for w in block.mlp_weights),
            mlp_biases=tuple(b.cpu() for b in block.mlp_biases),
            norm2_weight=block.norm2_weight.cpu(),
            norm2_bias=block.norm2_bias.cpu()
        )
        blocks_cpu.append(block_cpu)
    
    return BERTWeightSpace(
        token_embed=ws.token_embed.cpu(),
        pos_embed=ws.pos_embed.cpu(),
        blocks=blocks_cpu,
        norm_weight=ws.norm_weight.cpu(),
        norm_bias=ws.norm_bias.cpu(),
        head_weight=ws.head_weight.cpu(),
        head_bias=ws.head_bias.cpu()
    )


def move_weight_space_to_device(ws_cpu, device):
    # Create new blocks without deepcopy
    blocks_dev = []
    for block_cpu in ws_cpu.blocks:
        attn_dev = BERTAttentionWeights(
            qkv_weight=block_cpu.attention.qkv_weight.to(device),
            qkv_bias=block_cpu.attention.qkv_bias.to(device) if block_cpu.attention.qkv_bias is not None else None,
            proj_weight=block_cpu.attention.proj_weight.to(device),
            proj_bias=block_cpu.attention.proj_bias.to(device) if block_cpu.attention.proj_bias is not None else None,
            num_heads=block_cpu.attention.num_heads
        )
        
        block_dev = BERTTransformerBlockWeights(
            attention=attn_dev,
            norm1_weight=block_cpu.norm1_weight.to(device),
            norm1_bias=block_cpu.norm1_bias.to(device),
            mlp_weights=tuple(w.to(device) for w in block_cpu.mlp_weights),
            mlp_biases=tuple(b.to(device) for b in block_cpu.mlp_biases),
            norm2_weight=block_cpu.norm2_weight.to(device),
            norm2_bias=block_cpu.norm2_bias.to(device)
        )
        blocks_dev.append(block_dev)
    
    return BERTWeightSpace(
        token_embed=ws_cpu.token_embed.to(device),
        pos_embed=ws_cpu.pos_embed.to(device),
        blocks=blocks_dev,
        norm_weight=ws_cpu.norm_weight.to(device),
        norm_bias=ws_cpu.norm_bias.to(device),
        head_weight=ws_cpu.head_weight.to(device),
        head_bias=ws_cpu.head_bias.to(device)
    )

def flatten_weights_in_batches(weight_space_objects, batch_size=10, device='cpu'):
    """Process weights in batches to avoid OOM"""
    all_flat_weights = []
    num_batches = (len(weight_space_objects) + batch_size - 1) // batch_size
    
    print(f"Flattening {len(weight_space_objects)} weight spaces in {num_batches} batches...")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(weight_space_objects))
        batch = weight_space_objects[start_idx:end_idx]
        
        batch_flat = []
        for ws in batch:
            try:
                flat = ws.flatten(device=device)
                batch_flat.append(flat)
                del flat
            except Exception as e:
                print(f"Error flattening weight space: {e}")
                continue
        
        all_flat_weights.extend(batch_flat)
        
        # Clean up after each batch
        del batch, batch_flat
        torch.cuda.empty_cache()
        gc.collect()
        
        if (batch_idx + 1) % 5 == 0:
            print(f"Processed batch {batch_idx + 1}/{num_batches}")
    
    return all_flat_weights

def apply_transfusion(model_dir, num_models, pretrained_model_name, ref_point=0, device=None):
    """
    Load and align models using TransFusion (always applies rebasin).
    Automatically detects model type (ViT or BERT) from pretrained_model_name.
    
    Args:
        model_dir: Directory containing model checkpoints
        num_models: Number of models to process
        pretrained_model_name: Base name of model files (should contain 'vit' or 'bert')
        ref_point: Index of reference model
        device: Device to use (None = auto-detect)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Detect model type from name
    model_name_lower = pretrained_model_name.lower()
    if 'bert' in model_name_lower:
        model_type = 'bert'
    elif 'vit' in model_name_lower:
        model_type = 'vit'
    else:
        raise ValueError(f"Cannot determine model type from name '{pretrained_model_name}'. "
                        f"Name must contain 'bert' or 'vit'")
    
    print(f"Detected model type: {model_type.upper()}")
    org_weight_space_objects = []
    
    if model_type == 'bert':
        ref_model = create_bert_100m(num_classes=1).to(device)
        ref_model_path = f"{model_dir}/{pretrained_model_name}{ref_point}_best.pt"
        
        try:
            ref_model.load_state_dict(torch.load(ref_model_path, map_location=device))
            print(f"Loaded reference BERT model from {ref_model_path}")
        except Exception as e:
            print(f"Failed to load reference model: {e}")
            raise e
        
        ref_ws = BERTWeightSpace.from_bert_model(ref_model)
        ref_ws_cpu = move_weight_space_to_cpu(ref_ws)
        
        weight_space_objects = [ref_ws_cpu]
        org_weight_space_objects = [ref_ws_cpu] # This is recommended to comment this off and either fetch only canonicalized or non canonicalized models trying to fetch both for BERT can lead to memory issues
        
        # Aggressive cleanup
        del ref_model, ref_ws
        aggressive_cleanup()
        
        matcher = TransFusionMatcher(num_iterations=2, model_type='bert')
        
        for i in tqdm(range(num_models), desc="Processing BERT models"):
            if i == ref_point:
                continue
            
            model_path = f"{model_dir}/{pretrained_model_name}{i}_best.pt"
            if not os.path.exists(model_path):
                print(f"Skipping model {i} - file not found")
                continue
            
            try:
                # Load model on GPU
                model = create_bert_100m(num_classes=1).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                
                # Extract weight space
                ws = BERTWeightSpace.from_bert_model(model)
                ws_cpu = move_weight_space_to_cpu(ws)
                # This is recommended to comment this off and either fetch only canonicalized or non canonicalized models trying to fetch both for BERT can lead to memory issues, unfreeze it to fetch non canonicalized models but then comment out lines 519-529
                # org_weight_space_objects.append(ws_cpu) 
                
                del model, ws
                aggressive_cleanup()
                
                # Canonicalize (temporarily move to GPU)
                ref_ws_gpu = move_weight_space_to_device(ref_ws_cpu, device)
                ws_gpu = move_weight_space_to_device(ws_cpu, device)
                
                canonicalized_list = matcher.canonicalize_model([ref_ws_gpu, ws_gpu], reference_idx=0)
                aligned_ws = canonicalized_list[1]
                
                # Move back to CPU and delete GPU versions
                aligned_ws_cpu = move_weight_space_to_cpu(aligned_ws)
                weight_space_objects.append(aligned_ws_cpu)
                
                del ws_cpu, ref_ws_gpu, ws_gpu, canonicalized_list, aligned_ws
                aggressive_cleanup()
                
                if (i + 1) % 5 == 0:
                    print(f"Processed {i+1}/{num_models} - Memory cleared")
                    
            except Exception as e:
                print(f"Error processing model {i}: {str(e)}")
                aggressive_cleanup()
                continue
        
        print(f"Successfully processed {len(weight_space_objects)} BERT models")
        aggressive_cleanup()
        
        return org_weight_space_objects, weight_space_objects
    
    # ========== ViT Processing ==========
    else: 
        ref_model = create_vit_small().to(device)
        ref_model_path = f"{model_dir}/{pretrained_model_name}{ref_point}.pt"
        
        try:
            ref_model.load_state_dict(torch.load(ref_model_path, map_location=device))
            print(f"Loaded reference ViT model from {ref_model_path}")
        except Exception as e:
            print(f"Failed to load reference model: {e}")
            raise e
        
        ref_ws = VisionTransformerWeightSpace.from_vit_model(ref_model)
        weight_space_objects = [ref_ws]
        org_weight_space_objects = [ref_ws]
        
        matcher = TransFusionMatcher(num_iterations=10, model_type='vit')
        
        for i in tqdm(range(num_models), desc="Processing ViT models"):
            if i == ref_point:
                continue
            
            model_path = f"{model_dir}/{pretrained_model_name}{i}.pt"
            if not os.path.exists(model_path):
                print(f"Skipping model {i} - file not found")
                continue
            
            try:
                model = create_vit_small().to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                
                ws = VisionTransformerWeightSpace.from_vit_model(model)
                
                canonicalized_list = matcher.canonicalize_model([ref_ws, ws], reference_idx=0)
                org_weight_space_objects.append(ws)

                aligned_ws = canonicalized_list[1]
                
                weight_space_objects.append(aligned_ws)
                
                del model, ws, canonicalized_list
                aggressive_cleanup()
                    
            except Exception as e:
                print(f"Error processing model {i}: {str(e)}")
                aggressive_cleanup()
                continue
        
        print(f"Successfully processed {len(weight_space_objects)} ViT models")
        aggressive_cleanup()
        
        return org_weight_space_objects, weight_space_objects


def _process_mlp_iris(model_name, model_dir, num_models, ref_seed, device, model_config=None):
    """
    Process MLP models for Iris dataset with multiple initialization types.
    Uses 'mlp_xavier_seed{ref_seed}' as the reference model.

    Returns:
        ref_model: reference model
        all_org_models: list of original models
        all_permuted_models: list of models after weight matching permutations
    """
    init_types = model_config.get('mlp_init_types', ["default", "he", "xavier", "uniform", "normal"]) \
        if model_config else ["default", "he", "xavier", "uniform", "normal"]

    ref_init = "xavier"
    ref_name = f"mlp_{ref_init}_seed{ref_seed}"
    ref_model, ps = get_model_and_spec(model_name, device)
    ref_model_path = f"{model_dir}/{ref_name}.pt"

    if not os.path.exists(ref_model_path):
        raise FileNotFoundError(f"Reference model {ref_name} not found at {ref_model_path}")

    ref_model.load_state_dict(torch.load(ref_model_path, map_location=device))
    print(f"Loaded reference model from {ref_model_path}")

    params_a = {}
    for k, v in ref_model.state_dict().items():
        if k in ps.axes_to_perm:
            params_a[k] = v.clone().detach() if "weight" in k else v.clone().detach()

    all_org_models = [ref_model]
    all_permuted_models = [ref_model]

    for init_type in init_types:
        for seed in range(20):
            if init_type == ref_init and seed == ref_seed:
                continue

            model_name_curr = f"mlp_{init_type}_seed{seed}"
            model_path = f"{model_dir}/{model_name_curr}.pt"
            if not os.path.exists(model_path):
                print(f"Skipping {model_name_curr} - file not found")
                continue

            model_b, _ = get_model_and_spec(model_name, device)
            model_b.load_state_dict(torch.load(model_path, map_location=device))
            all_org_models.append(model_b)

            params_b = {}
            for k, v in model_b.state_dict().items():
                if k in ps.axes_to_perm:
                    params_b[k] = v.clone().detach() if "weight" in k else v.clone().detach()

            perm = weight_matching(ps, params_a, params_b, device=device)
            permuted_params_b = apply_permutation(ps, perm, params_b)

            reconstructed_model = copy.deepcopy(model_b)
            state_dict = reconstructed_model.state_dict()
            for k in permuted_params_b:
                state_dict[k] = permuted_params_b[k]
            reconstructed_model.load_state_dict(state_dict)

            all_permuted_models.append(reconstructed_model)
            torch.cuda.empty_cache()

    print(f"Processed {len(all_permuted_models)} models successfully (including reference)")
    return ref_model, all_org_models, all_permuted_models


def process_models(model_name, model_dir, pretrained_model_name, num_models, ref_point, device):
    """Load reference model and permute all other models to align weights."""
    ref_model, ps = get_model_and_spec(model_name, device)
    ref_model_path = f"{model_dir}/{pretrained_model_name}{ref_point}.pt"
    ref_model.load_state_dict(torch.load(ref_model_path, map_location=device))
    print(f"Loaded reference model from {ref_model_path}")

    params_a = {k: v.clone().detach() for k, v in ref_model.state_dict().items() if k in ps.axes_to_perm}

    permuted_models = [ref_model]
    org_models = [ref_model]

    for i in tqdm(range(num_models), desc="Processing models"):
        if i == ref_point:
            continue

        model_path = f"{model_dir}/{pretrained_model_name}{i}.pt"
        if not os.path.exists(model_path):
            print(f"Skipping model {i} - file not found")
            continue

        model_b, _ = get_model_and_spec(model_name, device)
        model_b.load_state_dict(torch.load(model_path, map_location=device))
        org_models.append(model_b)

        params_b = {k: v.clone().detach() for k, v in model_b.state_dict().items() if k in ps.axes_to_perm}

        perm = weight_matching(ps, params_a, params_b, device=device)
        permuted_params_b = apply_permutation(ps, perm, params_b)

        reconstructed_model = copy.deepcopy(model_b)
        state_dict = reconstructed_model.state_dict()
        for k in permuted_params_b:
            state_dict[k] = permuted_params_b[k]
        reconstructed_model.load_state_dict(state_dict)

        permuted_models.append(reconstructed_model)
        torch.cuda.empty_cache()

    print(f"Processed {len(permuted_models)} models successfully")
    return org_models, permuted_models


def get_permuted_models_data(
    model_name: str,
    model_dir: str,
    pretrained_model_name: str,
    num_models: int,
    ref_point: int = 0,
    device=None,
    model_config=None
):
    """Apply weight matching to align models with a reference model"""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "vit" in model_name.lower() or "bert" in model_name.lower():
        return apply_transfusion(model_dir, num_models, pretrained_model_name, ref_point, device)

    if "mlp" in model_name.lower():
        if "iris" in model_name.lower():
            return _process_mlp_iris(model_name, model_dir, num_models, ref_point, device, model_config)
       
    return process_models(model_name, model_dir, pretrained_model_name, num_models, ref_point, device)

def get_model_and_spec(model_name: str, device=None):
    """Return model constructor and permutation spec based on model_name"""
    from models import (MLP_MNIST, MLP_Fashion_MNIST, MLP_Iris, 
                       ResNet20, get_resnet18)
    from permutation_specs import (mnist_mlp_permutation_spec_mlp,
                                   fashion_mnist_mlp_permutation_spec,
                                   iris_mlp_permutation_spec_mlp,
                                   resnet20_permutation_spec,
                                   resnet18_permutation_spec)
    
    model_name = model_name.lower()
    
    if "mlp_mnist" in model_name and "fashion" not in model_name:
        return MLP_MNIST().to(device), mnist_mlp_permutation_spec_mlp()
    elif "mlp_fashion_mnist" in model_name:
        return MLP_Fashion_MNIST().to(device), fashion_mnist_mlp_permutation_spec()
    elif "mlp_iris" in model_name:
        return MLP_Iris().to(device), iris_mlp_permutation_spec_mlp()
    elif "resnet20" in model_name:
        return ResNet20().to(device), resnet20_permutation_spec()
    elif "resnet18" in model_name:
        return get_resnet18().to(device), resnet18_permutation_spec()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
