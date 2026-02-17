# Neural Network Weight Generation via Flow Matching

This project implements a flow matching approach to generate new neural network weights by learning from distributions of pretrained models. It supports multiple architectures including MLPs, ResNets, Vision Transformers, and BERT, with optional weight alignment using Git Re-Basin and TransFusion.

## Overview

The system learns to generate functional neural network weights by:
1. Loading collections of pretrained models
2. Aligning them using architecture-specific methods:
   - **Git Re-Basin** for MLPs and ResNets
   - **TransFusion** for Vision Transformers and BERT
3. Applying dimensionality reduction:
   - **Optional PCA** for MLPs and ResNets (config-driven)
   - **Dual PCA** with automatic variance targeting for Transformers
4. Learning a flow from noise to the weight distribution
5. Generating new weight samples that achieve competitive performance

## Supported Models

| Model | Dataset | Architecture | Alignment Method | PCA |
|-------|---------|--------------|------------------|-----|
| `mlp_mnist` | MNIST | 3-layer MLP (784-32-32-10) | Git Re-Basin | Optional |
| `mlp_fashion_mnist` | Fashion-MNIST | 3-layer MLP (784-128-128-10) | Git Re-Basin | Optional |
| `mlp_iris` | Iris | 2-layer MLP (4-16-3) | Git Re-Basin | Optional |
| `resnet20_cifar10` | CIFAR-10 | ResNet-20 | Git Re-Basin | Optional |
| `resnet18_cifar10` | CIFAR-10 | ResNet-18 | Git Re-Basin | **Recommended** (99 components) |
| `vit_cifar10` | CIFAR-10 | ViT-Small | TransFusion | Optional |
| `bert_yelp` | Yelp Reviews | BERT-Base | TransFusion | **Required** (auto variance) |

## Key Architectural Differences

### Standard Models (MLP, ResNet)
- **Weight Matching**: Git Re-Basin (Hungarian algorithm)
- **PCA**: Optional, controlled by `constants.json`
- **Flow Model**: `WeightSpaceFlowModel`
- **Training**: Can work with or without weight alignment

### Transformer Models (ViT, BERT)
- **Weight Matching**: TransFusion (two-level attention head matching)
- **PCA**: 
  - **ViT**: Optional
  - **BERT**: Required with GPU-accelerated streaming PCA
- **Flow Model**: `VisionTransformerFlowModel` (deeper architecture)
- **Training**: Always uses alignment (TransFusion)

## Installation

### 1. Clone the repository
```bash
git clone git@github.com:NNeuralDynamics/equivariant-diffusion.git
cd flow-matching-weights
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate pretrained models

Generate datasets using scripts in the `generateModelsCode` directory:
```bash
# For MLPs
python generateModelsCode/generate_mnist_weights.py
python generateModelsCode/generate_fashion_mnist_weights.py
python generateModelsCode/generate_iris_weights.py

# For ResNets
python generateModelsCode/generateResnet18.py
python generateModelsCode/generateResnet20.py

# For Vision Transformer
python generateModelsCode/generateVitData.py

# For BERT
python generateModelsCode/generateDataForBert.py

# For SmallCNN
python generateModelsCode/generatecifar10_smallcnn.py
```

## Directory Structure
```
.
├── flow-matching-weights/
│   ├── train_and_generate.py        # train_and_generate training script (unified entry point)
│   ├── flow_matching.py             # Flow matching implementations
│   ├── multi_class_flow.py          # Multi-class flow matching (optional)
│   ├── models.py                    # Neural network architectures
│   ├── utils.py                     # Utility functions
│   ├── permutation_specs.py         # Permutation specifications (single source)
│   ├── canonicalization.py          # Weight matching (Git Re-Basin + TransFusion)
│   ├── weight_space_objects.py      # Weight space representations
│   ├── dual_pca.py                  # GPU-accelerated PCA
│   ├── constants.json               # Configuration file
│   ├── transfer_learning_Resnet18/  # Transfer learning experiments
|   |── transfer_learning_Cifar10_to_Cifar100  # Transfer learning experiments
│   └── README.md
├── generateModelsCode/
│   ├── generate_fashion_mnist_weights.py
│   ├── generate_iris_weights.py
│   ├── generate_mnist_weights.py
│   ├── generateResnet18.py
│   ├── generateResnet20.py
│   ├── generateVitData.py
│   └── generateBertData.py
│   └── generatecifar10_smallcnn.py
├── LICENSE
├── README.md
└── requirements.txt
```

## Usage

### Basic Training

Run training for a specific model:
```bash
python train_and_generate.py --model <model_name> [options]
```

### Examples

#### 1. Train MLP on MNIST
```bash
# Test all hidden dimensions with both modes
python flowmatching/train_and_generate.py --model mlp_mnist --hidden_dim 512 --config flowmatching/constants.json

# Test specific hidden dimension with Git Re-Basin
python train_and_generate.py --model mlp_mnist --hidden_dim 512 --mode with_gitrebasin

# Without alignment
python train_and_generate.py --model mlp_mnist --mode without_rebasin
```

#### 2. Train MLP on Fashion-MNIST
```bash
python train_and_generate.py --model mlp_fashion_mnist

# With specific configuration
python train_and_generate.py --model mlp_fashion_mnist --hidden_dim 512 --num_models 50
```

#### 3. Train MLP on Iris
```bash
# This will process all 5 initialization types
python train_and_generate.py --model mlp_iris --hidden_dim 128

# The Iris dataset tests robustness across different initializations:
# - default, he, xavier, uniform, normal
```

#### 4. Train ResNet-20 on CIFAR-10
```bash
# ResNet-20 without PCA (full weight space)
python train_and_generate.py --model resnet20_cifar10

# Test specific configuration
python train_and_generate.py --model resnet20_cifar10 --hidden_dim 256 --num_models 50
```

#### 5. Train ResNet-18 on CIFAR-10 (with PCA)
```bash
# ResNet-18 uses PCA for dimensionality reduction (recommended)
python train_and_generate.py --model resnet18_cifar10

# Ensure PCA is enabled in constants.json:
# "use_pca": true,
# "pca_components": 99
```

#### 6. Train Vision Transformer on CIFAR-10
```bash
# ViT uses TransFusion alignment
python train_and_generate.py --model vit_cifar10

# With fewer models (faster training)
python train_and_generate.py --model vit_cifar10 --num_models 50
```

#### 7. Train BERT on Yelp Reviews
```bash
# BERT uses TransFusion + GPU-accelerated PCA
python train_and_generate.py --model bert_yelp

# With specific hidden dimension
python train_and_generate.py --model bert_yelp --hidden_dim 1024 --num_models 100
```

**BERT Results** (Yelp dataset, Spearman's correlation metric):
- Hidden dim: 1024, Latent dim: 64
- With alignment: 0.7902 ± 0.0061
- Without alignment: 0.7909 ± 0.005

#### 8. Multi-Class Conditional Training
```bash
# Train multi-class flow matching model
python train_multi_class.py --model <model_name> --num_classes <num_classes>

# Example: Multi-class MNIST with 10 classes
python train_multi_class.py --model mlp_mnist --num_classes 10

# Generate models for specific class
python train_multi_class.py --model mlp_mnist --generate --class_label 5
```

#### 9. Transfer Learning: CIFAR-10 to CIFAR-100
```bash
# Train on CIFAR-10 weights, transfer to CIFAR-100
python transfer_learning_Cifar10_to_Cifar100.py --source_models 100 --hidden_dim 512

# With specific configuration
python transfer_learning_Cifar10_to_Cifar100.py --source_models 50 --hidden_dim 1024 --num_samples 20
```

#### 10. Transfer Learning with ResNet-18
```bash
cd transfer_learning_Resnet18
python transfer_learning_Resnet18.py --hidden_dim 512
```

### Command-Line Arguments

| Argument | Description | Default | Options |
|----------|-------------|---------|---------|
| `--model` | Model to train | **Required** | `mlp_mnist`, `mlp_fashion_mnist`, `mlp_iris`, `resnet20_cifar10`, `resnet18_cifar10`, `vit_cifar10`, `bert_yelp` |
| `--config` | Configuration file path | `constants.json` | Path to JSON file |
| `--num_models` | Number of pretrained models | 100 | Integer |
| `--ref_point` | Reference model index | 0 | Integer |
| `--hidden_dim` | Flow model hidden dimension | None (tests all) | Integer from config |
| `--mode` | Training mode | None (tests both) | `with_gitrebasin`, `without_rebasin` |

**Note**: The `--mode` argument only applies to MLP and ResNet models. ViT and BERT always use alignment (TransFusion).

## Configuration

The `constants.json` file contains all model-specific configurations:

### Example Configuration Structure
```json
{
  "t_dist": "uniform",
  "training_modes": ["with_gitrebasin", "without_rebasin"],
  
  "models": {
    "resnet18_cifar10": {
      "architecture": "resnet",
      "dataset": "cifar10",
      "pretrained_model_name": "resnet18_seed",
      "layer_layout": null,
      "use_pca": true,
      "pca_components": 99,
      "recalibrate_bn": true,
      "flow_hidden_dims": [512, 1024, 2048],
      "n_iters": 30000,
      "n_samples": 100,
      "batch_size": 8,
      "lr": 0.0005,
      "weight_decay": 1e-5,
      "source_std": 0.001,
      "sigma": 0.001,
      "patience": 100,
      "time_embed_dim": 64,
      "dropout": 0.1,
      "integration_steps": 100,
      "integration_method": "rk4"
    },
    
    "bert_yelp": {
      "architecture": "bert",
      "dataset": "yelp",
      "pretrained_model_name": "bert_",
      "pca_variance": 0.99,
      "flow_hidden_dims": [512, 768, 1024],
      "n_iters": 30000,
      "n_samples": 100,
      "batch_size": 2,
      "lr": 0.0001,
      "weight_decay": 1e-5,
      "source_std": 0.01,
      "sigma": 0.001,
      "patience": 100,
      "integration_steps": 100,
      "integration_method": "rk4"
    }
  },
  
  "directories": {
    "resnet18_cifar10": "./models/resnet18/",
    "bert_yelp": "./models/bert_yelp/"
  }
}
```

### Key Configuration Parameters

#### Universal Parameters
- **`flow_hidden_dims`**: List of hidden dimensions to test (e.g., `[256, 512, 1024]`)
- **`n_iters`**: Training iterations (default: 30000)
- **`n_samples`**: Number of models to generate (default: 100)
- **`batch_size`**: Training batch size
- **`lr`**: Learning rate (default: 0.0005 for ResNets, 0.0001 for Transformers)
- **`source_std`**: Noise standard deviation (default: 0.001)
- **`integration_steps`**: ODE integration steps (default: 100)
- **`integration_method`**: ODE solver (`euler` or `rk4`)

#### Architecture-Specific Parameters

##### MLPs and ResNets
- **`use_pca`**: Enable PCA dimensionality reduction (boolean)
- **`pca_components`**: Number of PCA components (e.g., 99 for ResNet-18)
- **`recalibrate_bn`**: Recalibrate BatchNorm after generation (ResNets only)
- **`gradient_accumulation_steps`**: Optional gradient accumulation

##### Vision Transformer
- **`use_pca`**: Optional PCA (usually false for ViT but true for Resnet18 and BERT)
- **`gradient_accumulation_steps`**: Recommended for memory efficiency

##### BERT
- **`pca_variance`**: Target variance for automatic PCA component selection (e.g., 0.99)
- Uses GPU-accelerated streaming PCA automatically
- No `use_pca` flag needed (always enabled)

### Model-Specific Recommendations

#### MLPs
- **Training**: Fast, works well without PCA
- **Generation**: Simple, no special handling needed
- **Iris**: Tests 5 initialization types for robustness

#### ResNets
- **ResNet-20**: Can work without PCA
- **ResNet-18**: **PCA highly recommended** (99 components)
- **BatchNorm**: Always recalibrate after generation (`recalibrate_bn: true`)

#### Vision Transformer
- **Alignment**: Always uses TransFusion
- **Memory**: Use gradient accumulation if OOM
- **PCA**: Optional, usually not needed

#### BERT
- **Alignment**: Always uses TransFusion
- **PCA**: Automatic with GPU acceleration
- **Evaluation**: Uses regression metrics (R², MAE, Spearman)
- **Memory**: Processes in batches with CPU offloading

## Algorithm Details

### Git Re-Basin (MLPs, ResNets)
Aligns models by finding optimal neuron permutations:
1. Extracts weight parameters from all models
2. Uses Hungarian algorithm to match neurons
3. Applies permutations to align weight spaces
4. Enables learning in a consistent space

### TransFusion (ViT, BERT)
Two-level permutation matching for attention heads:
1. **Inter-head matching**: Matches attention heads across models
2. **Intra-head matching**: Aligns neurons within each head
3. **Iterative refinement**: Applies permutations iteratively
4. **MLP alignment**: Matches feed-forward layers

### GPU-Accelerated PCA (BERT)
Efficient dimensionality reduction for large models:
1. **Streaming computation**: Processes models in batches
2. **Covariance matrix**: Built incrementally on GPU (10-50x faster)
3. **Automatic variance targeting**: Selects components to reach target variance
4. **Memory efficient**: Only loads 2 micro-batches at a time

## Experimental Results

### BERT-Base Results
Dataset: Yelp Reviews  
Metric: Spearman's correlation  
Architecture: BERT-Base (12 layers, 768 hidden dimension)

| Model | Hidden Dim | Latent Dim | With Alignment | Without Alignment |
|-------|------------|------------|----------------|-------------------|
| BERT-Base | 1024 | 64 | 0.7902 ± 0.0061 | 0.7909 ± 0.005 |

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**For MLPs/ResNets:**
```json
{
  "batch_size": 4,
  "gradient_accumulation_steps": 4
}
```

**For BERT:**
```json
{
  "batch_size": 1,
  "n_samples": 50
}
```

Also consider:
- Use CPU: Set `device = torch.device("cpu")` in code
- Enable PCA for ResNet-18 to reduce dimensionality
- Process BERT models in smaller batches

#### 2. Missing Pretrained Models

Ensure model files exist with correct naming:
- **MLPs**: `mlp_{init_type}_seed{N}.pt` (e.g., `mlp_xavier_seed0.pt`)
- **ResNets**: `resnet18_seed{N}.pt` or `resnet20_seed{N}.pt`
- **ViT**: `vit_seed{N}.pt`
- **BERT**: `bert_{N}_best.pt`

Check `constants.json` for correct directory paths.

#### 3. Poor Generation Quality

**Try these adjustments:**
```json
{
  "n_iters": 50000,
  "hidden_dim": 1024,
  "lr": 0.0001,
  "integration_steps": 200,
  "integration_method": "rk4"
}
```

**For ResNets:**
- Ensure Git Re-Basin alignment is enabled (`with_gitrebasin` mode)
- Enable BatchNorm recalibration (`recalibrate_bn: true`)
- Use PCA for ResNet-18

**For Transformers:**
- TransFusion alignment is automatic
- Ensure sufficient training iterations
- Check that PCA variance target is appropriate (0.99 recommended)

#### 4. Slow Training

**Speed up training:**
```json
{
  "integration_steps": 50,
  "integration_method": "euler",
  "batch_size": 16
}
```

**For BERT:**
- GPU is strongly recommended
- FP16 precision is automatically used
- Micro-batching is enabled by default

#### 5. Import Errors

Ensure all required files are present:
```bash
# Required files
train_and_generate.py
flow_matching.py
models.py
utils.py
permutation_specs.py
canonicalization.py
weight_space_objects.py
dual_pca.py
constants.json
```

If using BERT:
```bash
pip install transformers datasets
```

#### 6. Weight Matching Failures

**For Git Re-Basin (MLPs, ResNets):**
- Check that permutation specs match model architecture
- Verify all models have same architecture
- Ensure reference model (ref_point=0) loads correctly

**For TransFusion (ViT, BERT):**
- Ensure model files follow naming convention
- Check that all models have same architecture
- Verify attention head configuration matches

### Debugging Tips

#### Enable Verbose Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Check Memory Usage
```python
import torch
print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
```

## Advanced Usage

### Batch Processing

Run experiments for all models:
```bash
#!/bin/bash
models=("mlp_mnist" "mlp_fashion_mnist" "mlp_iris" "resnet20_cifar10" "resnet18_cifar10" "vit_cifar10" "bert_yelp")

for model in "${models[@]}"; do
    echo "Training $model..."
    python train_and_generate.py --model $model --num_models 100
    
    if [ $? -eq 0 ]; then
        echo "✓ $model completed successfully"
    else
        echo "✗ $model failed"
    fi
done
```

### Multi-Class Conditional Training

Train models that can generate weights for different classes, Set PCA as true in constants for these classes:
```bash
# Train multi-class flow matching model
python train_multi_class.py
```

### Transfer Learning Experiments

#### CIFAR-10 to CIFAR-100 Transfer
```bash
# Train on CIFAR-10 weights, transfer to CIFAR-100
python transfer_learning_Cifar10_to_Cifar100.py --source_models 100 --hidden_dim 512

# With specific configuration
python transfer_learning_Cifar10_to_Cifar100.py \
    --source_models 50 \
    --hidden_dim 1024 \
    --num_samples 20 \
    --batch_size 8
```

#### ResNet-18 Transfer Learning
```bash
# Transfer learning experiments with ResNet-18 to STl-10 and SVHN datasets
cd transfer_learning_Resnet18
python transfer_learning_Resnet18.py --hidden_dim 512

# Test multiple hidden dimensions
for hd in 256 512 1024; do
    python transfer_learning_Resnet18.py --hidden_dim $hd
done
```

## Citation

If you use this code in your research, please cite:
```bibtex
@article{weight-generation-flow-matching,
  title={Neural Network Weight Generation via Flow Matching},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Git Re-Basin algorithm based on [Ainsworth et al., 2022]
- TransFusion alignment for transformers
- Flow matching framework inspired on recent developments in generative modeling
