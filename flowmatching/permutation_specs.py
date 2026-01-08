"""
Permutation specifications for different neural network architectures.
This is the SINGLE SOURCE OF TRUTH for all permutation specs.
"""
from typing import NamedTuple
from collections import defaultdict
import torch


class PermutationSpec(NamedTuple):
    """Specification for weight permutations"""
    perm_to_axes: dict
    axes_to_perm: dict


def permutation_spec_from_axes_to_perm(axes_to_perm: dict) -> PermutationSpec:
    """Create PermutationSpec from axes_to_perm mapping"""
    perm_to_axes = defaultdict(list)
    for wk, axis_perms in axes_to_perm.items():
        for axis, perm in enumerate(axis_perms):
            if perm is not None:
                perm_to_axes[perm].append((wk, axis))
    return PermutationSpec(perm_to_axes=dict(perm_to_axes), axes_to_perm=axes_to_perm)


# ============================================================================
# MLP PERMUTATION SPECS
# ============================================================================

def iris_mlp_permutation_spec_mlp() -> PermutationSpec:
    """Permutation spec for Iris MLP (2 layers)"""
    return permutation_spec_from_axes_to_perm({
        "fc1.weight": ("P_0", None),       # (hidden, input) - permute hidden dimension
        "fc1.bias": ("P_0",),              # Bias for hidden layer
        "fc2.weight": (None, "P_0"),       # (output, hidden) - permute hidden input
        "fc2.bias": (None,),               # Bias for output (no permutation)
    })


def mnist_mlp_permutation_spec_mlp() -> PermutationSpec:
    """Permutation spec for MNIST MLP (3 layers)"""
    return permutation_spec_from_axes_to_perm({
        "fc1.weight": ("P_0", None),       # (hidden1, input)
        "fc1.bias": ("P_0",),
        "fc2.weight": ("P_1", "P_0"),      # (hidden2, hidden1)
        "fc2.bias": ("P_1",),
        "fc3.weight": (None, "P_1"),       # (output, hidden2)
        "fc3.bias": (None,),
    })


def fashion_mnist_mlp_permutation_spec() -> PermutationSpec:
    """Permutation spec for Fashion MNIST MLP (3 layers)"""
    return permutation_spec_from_axes_to_perm({
        "fc1.weight": ("P_0", None),
        "fc1.bias": ("P_0",),
        "fc2.weight": ("P_1", "P_0"),
        "fc2.bias": ("P_1",),
        "fc3.weight": (None, "P_1"),
        "fc3.bias": (None,),
    })


# ============================================================================
# RESNET PERMUTATION SPECS
# ============================================================================

def resnet20_permutation_spec() -> PermutationSpec:
    """Permutation spec for ResNet-20 with BatchNorm"""
    conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None)}

    norm = lambda name, p: {
        f"{name}.weight": (p,),
        f"{name}.bias": (p,),
        f"{name}.running_mean": (p,),
        f"{name}.running_var": (p,),
    }

    dense = lambda name, p_in, p_out: {
        f"{name}.weight": (p_out, p_in),
        f"{name}.bias": (p_out,)
    }

    easyblock = lambda name, p: {
        **conv(f"{name}.conv1", p, f"P_{name}_inner"),
        **norm(f"{name}.bn1", f"P_{name}_inner"),
        **conv(f"{name}.conv2", f"P_{name}_inner", p),
        **norm(f"{name}.bn2", p),
    }

    shortcutblock = lambda name, p_in, p_out: {
        **conv(f"{name}.conv1", p_in, f"P_{name}_inner"),
        **norm(f"{name}.bn1", f"P_{name}_inner"),
        **conv(f"{name}.conv2", f"P_{name}_inner", p_out),
        **norm(f"{name}.bn2", p_out),
        **conv(f"{name}.shortcut.0", p_in, p_out),
        **norm(f"{name}.shortcut.1", p_out),
    }

    return permutation_spec_from_axes_to_perm({
        **conv("conv1", None, "P_bg0"),
        **norm("bn1", "P_bg0"),

        **easyblock("layer1.0", "P_bg0"),
        **easyblock("layer1.1", "P_bg0"),
        **easyblock("layer1.2", "P_bg0"),

        **shortcutblock("layer2.0", "P_bg0", "P_bg1"),
        **easyblock("layer2.1", "P_bg1"),
        **easyblock("layer2.2", "P_bg1"),

        **shortcutblock("layer3.0", "P_bg1", "P_bg2"),
        **easyblock("layer3.1", "P_bg2"),
        **easyblock("layer3.2", "P_bg2"),

        **dense("linear", "P_bg2", None),
    })


def resnet18_permutation_spec() -> PermutationSpec:
    """Permutation spec for ResNet-18 with BatchNorm"""
    conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None)}

    norm = lambda name, p: {
        f"{name}.weight": (p,),
        f"{name}.bias": (p,),
        f"{name}.running_mean": (p,),
        f"{name}.running_var": (p,),
    }

    dense = lambda name, p_in, p_out: {
        f"{name}.weight": (p_out, p_in),
        f"{name}.bias": (p_out,)
    }
    
    def basicblock(name, p_in, p_out, has_shortcut=False):
        spec = {
            **conv(f"{name}.conv1", p_in, f"P_{name}_inner"),
            **norm(f"{name}.bn1", f"P_{name}_inner"),
            **conv(f"{name}.conv2", f"P_{name}_inner", p_out),
            **norm(f"{name}.bn2", p_out),
        }
        
        if has_shortcut:
            spec.update({
                **conv(f"{name}.downsample.0", p_in, p_out),
                **norm(f"{name}.downsample.1", p_out),
            })
        
        return spec
    
    return permutation_spec_from_axes_to_perm({
        **conv("conv1", None, "P_64"),
        **norm("bn1", "P_64"),
        
        **basicblock("layer1.0", "P_64", "P_64"),
        **basicblock("layer1.1", "P_64", "P_64"),
        
        **basicblock("layer2.0", "P_64", "P_128", has_shortcut=True),
        **basicblock("layer2.1", "P_128", "P_128"),
        
        **basicblock("layer3.0", "P_128", "P_256", has_shortcut=True),
        **basicblock("layer3.1", "P_256", "P_256"),
        
        **basicblock("layer4.0", "P_256", "P_512", has_shortcut=True),
        **basicblock("layer4.1", "P_512", "P_512"),
        
        **dense("fc", "P_512", None),
    })


# ============================================================================
# TRANSFORMER PERMUTATION SPECS (ViT and BERT)
# ============================================================================

class ViTPermutationSpec:
    """Specification for ViT permutations"""
    
    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.patch_embed_perm = None
        self.block_perms = []
        for _ in range(num_blocks):
            self.block_perms.append({
                'attention_in': None,
                'attention_out': None,
                'mlp1': None,
                'mlp2': None,
            })
        self.head_perm = None
    
    def set_block_perm(self, block_idx: int, perm_type: str, perm: torch.Tensor):
        """Set a specific permutation for a block"""
        self.block_perms[block_idx][perm_type] = perm


class BERTPermutationSpec:
    """Specification for BERT permutations"""
    
    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.block_perms = []
        for _ in range(num_blocks):
            self.block_perms.append({
                'attention_out': None,
                'mlp1': None,
            })
    
    def set_block_perm(self, block_idx: int, perm_type: str, perm: torch.Tensor):
        """Set a specific permutation for a block"""
        self.block_perms[block_idx][perm_type] = perm


# ============================================================================
# HELPER FUNCTION TO GET SPEC BY MODEL NAME
# ============================================================================

def get_permutation_spec(model_name: str) -> PermutationSpec:
    """
    Get permutation spec by model name.
    
    Args:
        model_name: Name of the model (e.g., 'mlp_mnist', 'resnet20_cifar10')
    
    Returns:
        PermutationSpec object
    
    Raises:
        ValueError: If model name is not recognized
    """
    model_name = model_name.lower()
    
    if "mlp_mnist" in model_name and "fashion" not in model_name:
        return mnist_mlp_permutation_spec_mlp()
    elif "mlp_fashion_mnist" in model_name:
        return fashion_mnist_mlp_permutation_spec()
    elif "mlp_iris" in model_name:
        return iris_mlp_permutation_spec_mlp()
    elif "resnet20" in model_name:
        return resnet20_permutation_spec()
    elif "resnet18" in model_name:
        return resnet18_permutation_spec()
    else:
        raise ValueError(f"No permutation spec defined for model: {model_name}")