import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
import copy

class WeightSpaceObjectMLP:
    def __init__(self, weights, biases):
        self.weights = tuple(weights)
        self.biases = tuple(biases)
        
    def flatten(self, device=None):
        flat = torch.cat([w.flatten() for w in self.weights] + 
                        [b.flatten() for b in self.biases])
        return flat.to(device) if device else flat
    
    @classmethod
    def from_flat(cls, flat, layers, device=None):
        """Create WeightSpaceObject from flattened vector and layer sizes"""
        sizes = []
        for i in range(len(layers) - 1):
            sizes.append(layers[i] * layers[i+1]) 

        for i in range(1, len(layers)):
            sizes.append(layers[i])
            
        parts = []
        start = 0
        for size in sizes:
            parts.append(flat[start:start+size])
            start += size
            
        weights = []
        biases = []
        for i in range(len(layers) - 1):
            weights.append(parts[i].reshape(layers[i+1], layers[i]))
            biases.append(parts[i + len(layers) - 1])
            
        return cls(weights, biases)

class WeightSpaceObjectResnet:
    def __init__(self, weights, biases):
        self.weights = tuple(weights)
        self.biases = tuple(biases)
        
    def flatten(self, device=None):
        flat = torch.cat([w.flatten() for w in self.weights] + 
                        [b.flatten() for b in self.biases])
        return flat.to(device) if device else flat
    
    @classmethod
    def from_flat(cls, flat, weight_shapes, bias_shapes, device=None):
        sizes = ([np.prod(s) for s in weight_shapes + bias_shapes])
        parts = []
        start = 0
        for size in sizes:
            parts.append(flat[start:start+size])
            start += size
            
        n_weights = len(weight_shapes)
        n_biases = len(bias_shapes)
        
        weights = [parts[i].reshape(weight_shapes[i]) for i in range(n_weights)]
        biases = [parts[n_weights + i].reshape(bias_shapes[i]) for i in range(n_biases)]        
        return cls(weights, biases)

class AttentionWeights:
    """Attention layer weights for Transformers"""
    
    def __init__(self, qkv_weight: torch.Tensor, qkv_bias: Optional[torch.Tensor],
                 proj_weight: torch.Tensor, proj_bias: Optional[torch.Tensor],
                 num_heads: int):
        self.qkv_weight = qkv_weight
        self.qkv_bias = qkv_bias
        self.proj_weight = proj_weight
        self.proj_bias = proj_bias
        self.num_heads = num_heads
    
    def split_heads(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Split QKV into separate heads"""
        d_model = self.qkv_weight.shape[1]
        head_dim = d_model // self.num_heads
        
        # Split qkv_weight into Q, K, V
        qkv_split = torch.chunk(self.qkv_weight, 3, dim=0)
        
        q_heads = [qkv_split[0][i*head_dim:(i+1)*head_dim] for i in range(self.num_heads)]
        k_heads = [qkv_split[1][i*head_dim:(i+1)*head_dim] for i in range(self.num_heads)]
        v_heads = [qkv_split[2][i*head_dim:(i+1)*head_dim] for i in range(self.num_heads)]
        
        return q_heads, k_heads, v_heads


class TransformerBlockWeights:
    """Transformer block weights"""
    
    def __init__(self, attention: AttentionWeights, norm1_weight: torch.Tensor,
                 norm1_bias: torch.Tensor, mlp_weights: Tuple[torch.Tensor, ...],
                 mlp_biases: Tuple[torch.Tensor, ...], norm2_weight: torch.Tensor,
                 norm2_bias: torch.Tensor):
        self.attention = attention
        self.norm1_weight = norm1_weight
        self.norm1_bias = norm1_bias
        self.mlp_weights = mlp_weights
        self.mlp_biases = mlp_biases
        self.norm2_weight = norm2_weight
        self.norm2_bias = norm2_bias


class VisionTransformerWeightSpace:
    """Weight space object for Vision Transformers"""
    
    def __init__(self, 
                 patch_embed_weight: torch.Tensor,
                 patch_embed_bias: Optional[torch.Tensor],
                 cls_token: torch.Tensor,
                 pos_embed: torch.Tensor,
                 blocks: List[TransformerBlockWeights],
                 norm_weight: torch.Tensor,
                 norm_bias: torch.Tensor,
                 head_weight: torch.Tensor,
                 head_bias: torch.Tensor):
        
        self.patch_embed_weight = patch_embed_weight
        self.patch_embed_bias = patch_embed_bias
        self.cls_token = cls_token
        self.pos_embed = pos_embed
        self.blocks = blocks
        self.norm_weight = norm_weight
        self.norm_bias = norm_bias
        self.head_weight = head_weight
        self.head_bias = head_bias
        
    @classmethod
    def from_vit_model(cls, model: nn.Module):
        """Extract weights from a ViT model"""
        blocks = []
        
        # Extract transformer blocks
        for block in model.blocks:
            # Multi-head attention weights
            attn = block.attn
            attention_weights = AttentionWeights(
                qkv_weight=attn.qkv.weight.data.clone(),
                qkv_bias=attn.qkv.bias.data.clone() if attn.qkv.bias is not None else None,
                proj_weight=attn.proj.weight.data.clone(),
                proj_bias=attn.proj.bias.data.clone() if attn.proj.bias is not None else None,
                num_heads=attn.num_heads
            )
            
            # MLP weights - iterate through MLP's children modules
            mlp_weights = []
            mlp_biases = []
            for name, layer in block.mlp.named_children():
                if hasattr(layer, 'weight'):
                    mlp_weights.append(layer.weight.data.clone())
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        mlp_biases.append(layer.bias.data.clone())
            mlp_weights = tuple(mlp_weights)
            mlp_biases = tuple(mlp_biases)
            
            # Layer norms
            block_weights = TransformerBlockWeights(
                attention=attention_weights,
                norm1_weight=block.norm1.weight.data.clone(),
                norm1_bias=block.norm1.bias.data.clone(),
                mlp_weights=mlp_weights,
                mlp_biases=mlp_biases,
                norm2_weight=block.norm2.weight.data.clone(),
                norm2_bias=block.norm2.bias.data.clone()
            )
            blocks.append(block_weights)
        
        # Create weight space object
        return cls(
            patch_embed_weight=model.patch_embed.proj.weight.data.clone(),
            patch_embed_bias=model.patch_embed.proj.bias.data.clone() 
                            if model.patch_embed.proj.bias is not None else None,
            cls_token=model.cls_token.data.clone(),
            pos_embed=model.pos_embed.data.clone(),
            blocks=blocks,
            norm_weight=model.norm.weight.data.clone(),
            norm_bias=model.norm.bias.data.clone(),
            head_weight=model.head.weight.data.clone(),
            head_bias=model.head.bias.data.clone()
        )
    
    def apply_to_model(self, model: nn.Module):
        """Apply weights to a ViT model"""
        with torch.no_grad():
            # Patch embedding
            model.patch_embed.proj.weight.data.copy_(self.patch_embed_weight)
            if self.patch_embed_bias is not None:
                model.patch_embed.proj.bias.data.copy_(self.patch_embed_bias)
            
            # Tokens and embeddings
            model.cls_token.data.copy_(self.cls_token)
            model.pos_embed.data.copy_(self.pos_embed)
            
            # Transformer blocks
            for block, block_weights in zip(model.blocks, self.blocks):
                # Attention
                attn = block.attn
                attn.qkv.weight.data.copy_(block_weights.attention.qkv_weight)
                if block_weights.attention.qkv_bias is not None:
                    attn.qkv.bias.data.copy_(block_weights.attention.qkv_bias)
                attn.proj.weight.data.copy_(block_weights.attention.proj_weight)
                if block_weights.attention.proj_bias is not None:
                    attn.proj.bias.data.copy_(block_weights.attention.proj_bias)
                
                # Layer norms
                block.norm1.weight.data.copy_(block_weights.norm1_weight)
                block.norm1.bias.data.copy_(block_weights.norm1_bias)
                block.norm2.weight.data.copy_(block_weights.norm2_weight)
                block.norm2.bias.data.copy_(block_weights.norm2_bias)
                
                # MLP - iterate through MLP's children modules
                mlp_layers = [layer for name, layer in block.mlp.named_children() 
                             if hasattr(layer, 'weight')]
                for layer, weight in zip(mlp_layers, block_weights.mlp_weights):
                    layer.weight.data.copy_(weight)
                    
                # Handle biases separately since not all layers may have them
                mlp_bias_idx = 0
                for name, layer in block.mlp.named_children():
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        if mlp_bias_idx < len(block_weights.mlp_biases):
                            layer.bias.data.copy_(block_weights.mlp_biases[mlp_bias_idx])
                            mlp_bias_idx += 1
            
            # Final norm and head
            model.norm.weight.data.copy_(self.norm_weight)
            model.norm.bias.data.copy_(self.norm_bias)
            model.head.weight.data.copy_(self.head_weight)
            model.head.bias.data.copy_(self.head_bias)
    
    def flatten(self, device=None) -> torch.Tensor:
        """Flatten all weights into a single vector"""
        all_params = []
        
        # Patch embedding
        all_params.append(self.patch_embed_weight.flatten())
        if self.patch_embed_bias is not None:
            all_params.append(self.patch_embed_bias.flatten())
        
        # Tokens and embeddings
        all_params.append(self.cls_token.flatten())
        all_params.append(self.pos_embed.flatten())
        
        # Transformer blocks
        for block in self.blocks:
            # Attention
            all_params.append(block.attention.qkv_weight.flatten())
            if block.attention.qkv_bias is not None:
                all_params.append(block.attention.qkv_bias.flatten())
            all_params.append(block.attention.proj_weight.flatten())
            if block.attention.proj_bias is not None:
                all_params.append(block.attention.proj_bias.flatten())
            
            # Norms
            all_params.append(block.norm1_weight.flatten())
            all_params.append(block.norm1_bias.flatten())
            all_params.append(block.norm2_weight.flatten())
            all_params.append(block.norm2_bias.flatten())
            
            # MLP
            for w in block.mlp_weights:
                all_params.append(w.flatten())
            for b in block.mlp_biases:
                all_params.append(b.flatten())
        
        # Final norm and head
        all_params.append(self.norm_weight.flatten())
        all_params.append(self.norm_bias.flatten())
        all_params.append(self.head_weight.flatten())
        all_params.append(self.head_bias.flatten())
        
        flat = torch.cat(all_params)
        if device:
            flat = flat.to(device)
        return flat
    
    @classmethod
    def from_flat(cls, flat_tensor, reference_ws, device=None):
        """Reconstruct VisionTransformerWeightSpace from flattened weights"""
        if device is None:
            device = flat_tensor.device
            
        # Get all parameter shapes from reference
        param_shapes = []
        param_types = []
        
        # Patch embedding
        param_shapes.append(reference_ws.patch_embed_weight.shape)
        param_types.append('patch_embed_weight')
        
        if reference_ws.patch_embed_bias is not None:
            param_shapes.append(reference_ws.patch_embed_bias.shape)
            param_types.append('patch_embed_bias')
        
        # CLS token and pos embed
        param_shapes.append(reference_ws.cls_token.shape)
        param_types.append('cls_token')
        
        param_shapes.append(reference_ws.pos_embed.shape)
        param_types.append('pos_embed')
        
        # Transformer blocks
        for block_idx, block in enumerate(reference_ws.blocks):
            # Attention weights
            param_shapes.append(block.attention.qkv_weight.shape)
            param_types.append(f'block_{block_idx}_attn_qkv_weight')
            
            if block.attention.qkv_bias is not None:
                param_shapes.append(block.attention.qkv_bias.shape)
                param_types.append(f'block_{block_idx}_attn_qkv_bias')
            
            param_shapes.append(block.attention.proj_weight.shape)
            param_types.append(f'block_{block_idx}_attn_proj_weight')
            
            if block.attention.proj_bias is not None:
                param_shapes.append(block.attention.proj_bias.shape)
                param_types.append(f'block_{block_idx}_attn_proj_bias')
            
            # Layer norms
            param_shapes.append(block.norm1_weight.shape)
            param_types.append(f'block_{block_idx}_norm1_weight')
            
            param_shapes.append(block.norm1_bias.shape)
            param_types.append(f'block_{block_idx}_norm1_bias')
            
            param_shapes.append(block.norm2_weight.shape)
            param_types.append(f'block_{block_idx}_norm2_weight')
            
            param_shapes.append(block.norm2_bias.shape)
            param_types.append(f'block_{block_idx}_norm2_bias')
            
            # MLP weights
            for mlp_idx, mlp_weight in enumerate(block.mlp_weights):
                param_shapes.append(mlp_weight.shape)
                param_types.append(f'block_{block_idx}_mlp_weight_{mlp_idx}')
            
            # MLP biases
            for mlp_idx, mlp_bias in enumerate(block.mlp_biases):
                param_shapes.append(mlp_bias.shape)
                param_types.append(f'block_{block_idx}_mlp_bias_{mlp_idx}')
        
        # Final norm and head
        param_shapes.append(reference_ws.norm_weight.shape)
        param_types.append('norm_weight')
        
        param_shapes.append(reference_ws.norm_bias.shape)
        param_types.append('norm_bias')
        
        param_shapes.append(reference_ws.head_weight.shape)
        param_types.append('head_weight')
        
        param_shapes.append(reference_ws.head_bias.shape)
        param_types.append('head_bias')
        
        # Split flat tensor according to shapes
        sizes = [np.prod(shape) for shape in param_shapes]
        parts = []
        start = 0
        
        for size in sizes:
            parts.append(flat_tensor[start:start+size])
            start += size
        
        # Reconstruct parameters
        reconstructed_params = {}
        for i, (shape, param_type) in enumerate(zip(param_shapes, param_types)):
            reconstructed_params[param_type] = parts[i].reshape(shape).to(device)
        
        # Build the blocks
        reconstructed_blocks = []
        num_blocks = len(reference_ws.blocks)
        
        for block_idx in range(num_blocks):
            # Reconstruct attention weights
            qkv_weight = reconstructed_params[f'block_{block_idx}_attn_qkv_weight']
            qkv_bias = reconstructed_params.get(f'block_{block_idx}_attn_qkv_bias', None)
            proj_weight = reconstructed_params[f'block_{block_idx}_attn_proj_weight']  
            proj_bias = reconstructed_params.get(f'block_{block_idx}_attn_proj_bias', None)
            
            attention = AttentionWeights(
                qkv_weight=qkv_weight,
                qkv_bias=qkv_bias,
                proj_weight=proj_weight,
                proj_bias=proj_bias,
                num_heads=reference_ws.blocks[block_idx].attention.num_heads
            )
            
            # Reconstruct MLP weights
            mlp_weights = []
            mlp_biases = []
            
            mlp_weight_idx = 0
            while f'block_{block_idx}_mlp_weight_{mlp_weight_idx}' in reconstructed_params:
                mlp_weights.append(reconstructed_params[f'block_{block_idx}_mlp_weight_{mlp_weight_idx}'])
                mlp_weight_idx += 1
            
            mlp_bias_idx = 0
            while f'block_{block_idx}_mlp_bias_{mlp_bias_idx}' in reconstructed_params:
                mlp_biases.append(reconstructed_params[f'block_{block_idx}_mlp_bias_{mlp_bias_idx}'])
                mlp_bias_idx += 1
            
            # Create block
            block = TransformerBlockWeights(
                attention=attention,
                norm1_weight=reconstructed_params[f'block_{block_idx}_norm1_weight'],
                norm1_bias=reconstructed_params[f'block_{block_idx}_norm1_bias'],
                mlp_weights=tuple(mlp_weights),
                mlp_biases=tuple(mlp_biases),
                norm2_weight=reconstructed_params[f'block_{block_idx}_norm2_weight'],
                norm2_bias=reconstructed_params[f'block_{block_idx}_norm2_bias']
            )
            
            reconstructed_blocks.append(block)
        
        # Create the full weight space object
        return cls(
            patch_embed_weight=reconstructed_params['patch_embed_weight'],
            patch_embed_bias=reconstructed_params.get('patch_embed_bias', None),
            cls_token=reconstructed_params['cls_token'],
            pos_embed=reconstructed_params['pos_embed'],
            blocks=reconstructed_blocks,
            norm_weight=reconstructed_params['norm_weight'], 
            norm_bias=reconstructed_params['norm_bias'],
            head_weight=reconstructed_params['head_weight'],
            head_bias=reconstructed_params['head_bias']
        )


BERTAttentionWeights = AttentionWeights
BERTTransformerBlockWeights = TransformerBlockWeights

class BERTWeightSpace:
    """Weight space object for BERT Transformers"""
    
    def __init__(self, 
                 token_embed: torch.Tensor,
                 pos_embed: torch.Tensor,
                 blocks: List[BERTTransformerBlockWeights],
                 norm_weight: torch.Tensor,
                 norm_bias: torch.Tensor,
                 head_weight: torch.Tensor,
                 head_bias: torch.Tensor):
        
        self.token_embed = token_embed
        self.pos_embed = pos_embed
        self.blocks = blocks
        self.norm_weight = norm_weight
        self.norm_bias = norm_bias
        self.head_weight = head_weight
        self.head_bias = head_bias
        
    @classmethod
    def from_bert_model(cls, model: nn.Module):
        """Extract weights from a BERT model"""
        blocks = []
        
        for block in model.blocks:
            attn = block.attn
            attention_weights = BERTAttentionWeights(
                qkv_weight=attn.qkv.weight.data.clone(),
                qkv_bias=attn.qkv.bias.data.clone() if attn.qkv.bias is not None else None,
                proj_weight=attn.proj.weight.data.clone(),
                proj_bias=attn.proj.bias.data.clone() if attn.proj.bias is not None else None,
                num_heads=attn.num_heads
            )
            
            mlp_weights = []
            mlp_biases = []
            for name, layer in block.mlp.named_children():
                if hasattr(layer, 'weight'):
                    mlp_weights.append(layer.weight.data.clone())
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        mlp_biases.append(layer.bias.data.clone())
            mlp_weights = tuple(mlp_weights)
            mlp_biases = tuple(mlp_biases)
            
            block_weights = BERTTransformerBlockWeights(
                attention=attention_weights,
                norm1_weight=block.norm1.weight.data.clone(),
                norm1_bias=block.norm1.bias.data.clone(),
                mlp_weights=mlp_weights,
                mlp_biases=mlp_biases,
                norm2_weight=block.norm2.weight.data.clone(),
                norm2_bias=block.norm2.bias.data.clone()
            )
            blocks.append(block_weights)
        
        return cls(
            token_embed=model.token_embed.weight.data.clone(),
            pos_embed=model.pos_embed.data.clone(),
            blocks=blocks,
            norm_weight=model.norm.weight.data.clone(),
            norm_bias=model.norm.bias.data.clone(),
            head_weight=model.head.weight.data.clone(),
            head_bias=model.head.bias.data.clone()
        )
    
    def apply_to_model(self, model: nn.Module):
        """Apply weights to a BERT model"""
        with torch.no_grad():
            model.token_embed.weight.data.copy_(self.token_embed)
            model.pos_embed.data.copy_(self.pos_embed)
            
            for block, block_weights in zip(model.blocks, self.blocks):
                attn = block.attn
                attn.qkv.weight.data.copy_(block_weights.attention.qkv_weight)
                if block_weights.attention.qkv_bias is not None:
                    attn.qkv.bias.data.copy_(block_weights.attention.qkv_bias)
                attn.proj.weight.data.copy_(block_weights.attention.proj_weight)
                if block_weights.attention.proj_bias is not None:
                    attn.proj.bias.data.copy_(block_weights.attention.proj_bias)
                
                block.norm1.weight.data.copy_(block_weights.norm1_weight)
                block.norm1.bias.data.copy_(block_weights.norm1_bias)
                block.norm2.weight.data.copy_(block_weights.norm2_weight)
                block.norm2.bias.data.copy_(block_weights.norm2_bias)
                
                mlp_layers = [layer for name, layer in block.mlp.named_children() 
                             if hasattr(layer, 'weight')]
                for layer, weight in zip(mlp_layers, block_weights.mlp_weights):
                    layer.weight.data.copy_(weight)
                    
                mlp_bias_idx = 0
                for name, layer in block.mlp.named_children():
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        if mlp_bias_idx < len(block_weights.mlp_biases):
                            layer.bias.data.copy_(block_weights.mlp_biases[mlp_bias_idx])
                            mlp_bias_idx += 1
            
            model.norm.weight.data.copy_(self.norm_weight)
            model.norm.bias.data.copy_(self.norm_bias)
            model.head.weight.data.copy_(self.head_weight)
            model.head.bias.data.copy_(self.head_bias)
    
    def flatten(self, device=None) -> torch.Tensor:
        """Flatten all weights into a single vector - GPU-safe"""
        all_params = []
        
        all_params.append(self.token_embed.cpu().flatten())
        all_params.append(self.pos_embed.cpu().flatten())
        
        for block in self.blocks:
            all_params.append(block.attention.qkv_weight.cpu().flatten())
            if block.attention.qkv_bias is not None:
                all_params.append(block.attention.qkv_bias.cpu().flatten())
            all_params.append(block.attention.proj_weight.cpu().flatten())
            if block.attention.proj_bias is not None:
                all_params.append(block.attention.proj_bias.cpu().flatten())
            
            all_params.append(block.norm1_weight.cpu().flatten())
            all_params.append(block.norm1_bias.cpu().flatten())
            all_params.append(block.norm2_weight.cpu().flatten())
            all_params.append(block.norm2_bias.cpu().flatten())
            
            for w in block.mlp_weights:
                all_params.append(w.cpu().flatten())
            for b in block.mlp_biases:
                all_params.append(b.cpu().flatten())
        
        all_params.append(self.norm_weight.cpu().flatten())
        all_params.append(self.norm_bias.cpu().flatten())
        all_params.append(self.head_weight.cpu().flatten())
        all_params.append(self.head_bias.cpu().flatten())
        
        # Concatenate on CPU (safe!)
        flat = torch.cat(all_params)
        
        if device and device != 'cpu':
            flat = flat.to(device)
        
        return flat
    
    @classmethod
    def from_flat(cls, flat_tensor, reference_ws, device=None):
        """Reconstruct BERTWeightSpace from flattened weights - CORRECT VERSION"""
        
        # Determine device and move tensor FIRST
        if device is None:
            device = flat_tensor.device
        
        flat_tensor = flat_tensor.to(device)
        
        # Build parameter shapes list
        param_shapes = []
        param_types = []
        
        param_shapes.append(reference_ws.token_embed.shape)
        param_types.append('token_embed')
        
        param_shapes.append(reference_ws.pos_embed.shape)
        param_types.append('pos_embed')
        
        for block_idx, block in enumerate(reference_ws.blocks):
            param_shapes.append(block.attention.qkv_weight.shape)
            param_types.append(f'block_{block_idx}_attn_qkv_weight')
            
            if block.attention.qkv_bias is not None:
                param_shapes.append(block.attention.qkv_bias.shape)
                param_types.append(f'block_{block_idx}_attn_qkv_bias')
            
            param_shapes.append(block.attention.proj_weight.shape)
            param_types.append(f'block_{block_idx}_attn_proj_weight')
            
            if block.attention.proj_bias is not None:
                param_shapes.append(block.attention.proj_bias.shape)
                param_types.append(f'block_{block_idx}_attn_proj_bias')
            
            param_shapes.append(block.norm1_weight.shape)
            param_types.append(f'block_{block_idx}_norm1_weight')
            
            param_shapes.append(block.norm1_bias.shape)
            param_types.append(f'block_{block_idx}_norm1_bias')
            
            param_shapes.append(block.norm2_weight.shape)
            param_types.append(f'block_{block_idx}_norm2_weight')
            
            param_shapes.append(block.norm2_bias.shape)
            param_types.append(f'block_{block_idx}_norm2_bias')
            
            for mlp_idx, mlp_weight in enumerate(block.mlp_weights):
                param_shapes.append(mlp_weight.shape)
                param_types.append(f'block_{block_idx}_mlp_weight_{mlp_idx}')
            
            for mlp_idx, mlp_bias in enumerate(block.mlp_biases):
                param_shapes.append(mlp_bias.shape)
                param_types.append(f'block_{block_idx}_mlp_bias_{mlp_idx}')
        
        param_shapes.append(reference_ws.norm_weight.shape)
        param_types.append('norm_weight')
        
        param_shapes.append(reference_ws.norm_bias.shape)
        param_types.append('norm_bias')
        
        param_shapes.append(reference_ws.head_weight.shape)
        param_types.append('head_weight')
        
        param_shapes.append(reference_ws.head_bias.shape)
        param_types.append('head_bias')
        
        # Split tensor (already on correct device)
        sizes = [np.prod(shape) for shape in param_shapes]
        parts = []
        start = 0
        
        for size in sizes:
            parts.append(flat_tensor[start:start+size])
            start += size
        
        # Reshape (no .to(device) needed - already on device!)
        reconstructed_params = {}
        for i, (shape, param_type) in enumerate(zip(param_shapes, param_types)):
            reconstructed_params[param_type] = parts[i].reshape(shape)
        
        # Reconstruct blocks
        reconstructed_blocks = []
        num_blocks = len(reference_ws.blocks)
        
        for block_idx in range(num_blocks):
            qkv_weight = reconstructed_params[f'block_{block_idx}_attn_qkv_weight']
            qkv_bias = reconstructed_params.get(f'block_{block_idx}_attn_qkv_bias', None)
            proj_weight = reconstructed_params[f'block_{block_idx}_attn_proj_weight']  
            proj_bias = reconstructed_params.get(f'block_{block_idx}_attn_proj_bias', None)
            
            attention = BERTAttentionWeights(
                qkv_weight=qkv_weight,
                qkv_bias=qkv_bias,
                proj_weight=proj_weight,
                proj_bias=proj_bias,
                num_heads=reference_ws.blocks[block_idx].attention.num_heads
            )
            
            mlp_weights = []
            mlp_biases = []
            
            mlp_weight_idx = 0
            while f'block_{block_idx}_mlp_weight_{mlp_weight_idx}' in reconstructed_params:
                mlp_weights.append(reconstructed_params[f'block_{block_idx}_mlp_weight_{mlp_weight_idx}'])
                mlp_weight_idx += 1
            
            mlp_bias_idx = 0
            while f'block_{block_idx}_mlp_bias_{mlp_bias_idx}' in reconstructed_params:
                mlp_biases.append(reconstructed_params[f'block_{block_idx}_mlp_bias_{mlp_bias_idx}'])
                mlp_bias_idx += 1
            
            block = BERTTransformerBlockWeights(
                attention=attention,
                norm1_weight=reconstructed_params[f'block_{block_idx}_norm1_weight'],
                norm1_bias=reconstructed_params[f'block_{block_idx}_norm1_bias'],
                mlp_weights=tuple(mlp_weights),
                mlp_biases=tuple(mlp_biases),
                norm2_weight=reconstructed_params[f'block_{block_idx}_norm2_weight'],
                norm2_bias=reconstructed_params[f'block_{block_idx}_norm2_bias']
            )
            
            reconstructed_blocks.append(block)
        
        return cls(
            token_embed=reconstructed_params['token_embed'],
            pos_embed=reconstructed_params['pos_embed'],
            blocks=reconstructed_blocks,
            norm_weight=reconstructed_params['norm_weight'], 
            norm_bias=reconstructed_params['norm_bias'],
            head_weight=reconstructed_params['head_weight'],
            head_bias=reconstructed_params['head_bias']
        )