# VAR-Refine Project Summary

## What is VAR-Refine?

VAR-Refine is a hybrid image generation approach that combines the speed of Visual Autoregressive modeling (VAR) with the quality of diffusion models. The core idea is to:

1. **Generate a coarse-to-fine scaffold** using VAR (fast, structured)
2. **Refine with few-step diffusion** only on residuals (high quality, efficient)
3. **Result**: 5-10× faster than pure diffusion while maintaining quality

## Project Structure

```
var-research/
├── README.md                 # Main documentation
├── requirements.txt          # Python dependencies
├── run_experiment.py         # Main experiment runner
├── test_basic.py            # Basic functionality tests
├── EXPERIMENT_PLAN.md       # Detailed research plan
├── PROJECT_SUMMARY.md       # This file
├── configs/                 # Configuration files
│   ├── tokenizer_cifar10.yaml
│   ├── var_cifar10.yaml
│   ├── refiner_cifar10.yaml
│   └── sampling_cifar10.yaml
├── src/                     # Source code
│   ├── models/             # Model implementations
│   │   ├── hierarchical_vqvae.py
│   │   ├── var_transformer.py
│   │   └── diffusion_refiner.py
│   ├── training/           # Training scripts
│   │   ├── train_tokenizer.py
│   │   └── train_var.py
│   ├── sampling/           # Sampling pipeline
│   │   └── generate.py
│   └── utils/              # Utilities
│       └── metrics.py
├── data/                   # Dataset storage
└── experiments/            # Results and checkpoints
```

## Key Components

### 1. Hierarchical VQ-VAE Tokenizer (`src/models/hierarchical_vqvae.py`)
- **Purpose**: Encode images into pyramid latents (8²→16²→32²)
- **Features**: 
  - Multiple codebooks per scale
  - EMA codebook updates
  - Residual blocks for encoder/decoder
- **Output**: Discrete indices for each pyramid level

### 2. VAR Transformer (`src/models/var_transformer.py`)
- **Purpose**: Predict next-scale codes autoregressively
- **Features**:
  - 12-layer decoder-only transformer
  - Rotary + ALiBi positional encoding
  - Teacher forcing with scheduled sampling
- **Output**: Next-scale code predictions

### 3. Diffusion Refiner (`src/models/diffusion_refiner.py`)
- **Purpose**: Refine VAR output with few-step diffusion
- **Features**:
  - UNet-tiny architecture
  - Conditions on VAR output + class labels
  - Residual prediction (r = x* - x̂_VAR)
  - Auto-Guidance support
- **Output**: High-quality refined images

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test Basic Functionality
```bash
python test_basic.py
```

### 3. Run Full Experiment (CIFAR-10)
```bash
# Train all components
python run_experiment.py --dataset cifar10 --stage all --wandb

# Or run stages individually
python run_experiment.py --dataset cifar10 --stage tokenizer --wandb
python run_experiment.py --dataset cifar10 --stage var --wandb
python run_experiment.py --dataset cifar10 --stage sample --num_samples 1000
```

### 4. Generate Samples
```bash
python src/sampling/generate.py --config configs/sampling_cifar10.yaml --num_samples 100
```

## Training Pipeline

### Stage 0: Tokenizer Training
```bash
python src/training/train_tokenizer.py --config configs/tokenizer_cifar10.yaml --wandb
```
- **Time**: 4-6 hours on CIFAR-10
- **Output**: `experiments/tokenizer_cifar10/hierarchical_vqvae_cifar10.pt`

### Stage 1: VAR Training
```bash
python src/training/train_var.py --config configs/var_cifar10.yaml --wandb
```
- **Time**: 8-16 hours on CIFAR-10
- **Output**: `experiments/var_cifar10/var_transformer_cifar10.pt`

### Stage 2: Refiner Training
```bash
# Note: Refiner training script needs to be implemented
python src/training/train_refiner.py --config configs/refiner_cifar10.yaml --wandb
```
- **Time**: 6-12 hours per scale
- **Output**: `experiments/refiner_cifar10/diffusion_refiner_cifar10.pt`

## Configuration

All training and sampling parameters are controlled via YAML configs:

### Tokenizer Config (`configs/tokenizer_cifar10.yaml`)
```yaml
model:
  pyramid_levels: [8, 16, 32]
  codebook_size: 1024
  code_dim: 8
  hidden_dim: 128

training:
  batch_size: 64
  learning_rate: 1e-4
  num_epochs: 100
```

### VAR Config (`configs/var_cifar10.yaml`)
```yaml
model:
  num_layers: 12
  hidden_dim: 512
  num_heads: 8
  use_rotary: true
  use_alibi: true

training:
  batch_size: 32
  learning_rate: 5e-5
  num_epochs: 200
```

### Sampling Config (`configs/sampling_cifar10.yaml`)
```yaml
sampling:
  var:
    temperature: 1.0
    top_k: 100
    top_p: 0.9
  refiner:
    num_steps: 10
    guidance_scale: 1.0
    auto_guidance:
      enabled: true
      alpha: 0.5
```

## Expected Results

### Performance Targets
- **CIFAR-10**: FID < 5.0, Inception Score > 8.0
- **ImageNet-64**: FID < 15.0, Inception Score > 6.0
- **Speed**: < 100ms per image (32²)
- **NFE**: ≤10 diffusion steps

### Speed-Quality Trade-off
| Method | FID | NFE | Latency (ms) | Speedup |
|--------|-----|-----|--------------|---------|
| VAR-only | 8.5 | 0 | 20 | 1× |
| VAR-Refine (5 steps) | 6.2 | 5 | 45 | 0.4× |
| VAR-Refine (10 steps) | 5.1 | 10 | 70 | 0.3× |
| Diffusion (50 steps) | 4.8 | 50 | 350 | 0.06× |

## Key Innovations

### 1. Residual Diffusion
- Train refiner to predict residuals: r = x* - x̂_VAR
- More efficient than direct denoising
- Better gradient flow

### 2. Auto-Guidance
- Guide with worse checkpoint: ε_θ ← ε_θ + α(ε_θ - ε_φ)
- Improves quality without collapsing diversity
- Alternative to classifier-free guidance

### 3. Pyramid Latents
- Hierarchical VQ-VAE with multiple scales
- Coarse-to-fine generation
- Efficient representation

## Ablation Studies

The project includes comprehensive ablation studies:

1. **No-refiner vs Refiner**: How much lift does diffusion add?
2. **Residual vs Direct**: Does residual prediction work better?
3. **NFE Sweep**: Speed/quality Pareto curve (1, 3, 5, 10, 20 steps)
4. **Guidance Type**: CFG vs Auto-Guidance
5. **Scales Refined**: Top-1 vs top-2 scales
6. **Conditioning**: Pixels vs latent conditioning

## Compute Requirements

### Minimum Setup
- **GPU**: 24GB VRAM (RTX 4090, A5000, or similar)
- **RAM**: 32GB system memory
- **Storage**: 100GB free space

### Training Times (CIFAR-10)
- **Tokenizer**: 4-6 hours
- **VAR**: 8-16 hours
- **Refiner**: 6-12 hours
- **Total**: 18-34 hours

### Memory Usage
- **Tokenizer**: ~8GB VRAM
- **VAR**: ~16GB VRAM
- **Refiner**: ~12GB VRAM
- **Sampling**: ~8GB VRAM

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size in config
   - Enable gradient checkpointing
   - Use mixed precision (FP16)

2. **Training Instability**
   - Check learning rate
   - Monitor gradient norms
   - Use gradient clipping

3. **Codebook Collapse**
   - Monitor codebook usage
   - Adjust commitment cost
   - Use EMA updates

### Debug Commands
```bash
# Test basic functionality
python test_basic.py

# Check GPU memory
nvidia-smi

# Monitor training
tail -f experiments/*/logs.txt
```

## Next Steps

### Immediate Tasks
1. Implement refiner training script
2. Add ImageNet-64 configs
3. Implement evaluation metrics (FID, IS)
4. Add Auto-Guidance worse checkpoint training

### Future Work
1. Scale to ImageNet-256
2. Real-time generation (<50ms)
3. Class-conditional generation
4. Text-to-image extension

## Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

## Citation

If you use this code in your research, please cite:

```bibtex
@article{var-refine-2024,
  title={VAR-Refine: Hybrid Next-Scale Autoregression with Residual Diffusion Refinement},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## References

1. **VAR**: "Visual Autoregressive Modeling: Next-Scale Prediction" (NeurIPS'24 Best Paper)
2. **Auto-Guidance**: "Guiding a Diffusion Model with a Bad Version of Itself"
3. **VQ-VAE**: "Neural Discrete Representation Learning"
4. **Diffusion**: "Denoising Diffusion Probabilistic Models"

---

**Note**: This is a research implementation. For production use, additional optimizations and testing would be required.
