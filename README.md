# VAR-Refine: Hybrid Next-Scale Autoregression with Residual Diffusion Refinement

**Working Title**: "VAR-Refine: Hybrid Next-Scale Autoregression with Residual Diffusion Refinement"

## Core Idea

Generate a coarse-to-fine scaffold with Visual Autoregressive modeling (VAR), then run a tiny, few-step diffusion refiner only on the residuals at each scale. You inherit VAR's speed and structure, and diffusion's texture/faithfulness—without paying 50–100 diffusion steps.

## Method Overview

1. **Tokenizer (Pyramid Latents)**: Hierarchical VQ-VAE with codebooks per scale
2. **VAR Scaffold**: Small AR transformer predicts next-scale codes conditioned on previous scale
3. **Diffusion Refiner**: Class-conditional latent diffusion model that predicts residuals
4. **Auto-Guidance**: Guide refiner with worse checkpoint to improve sharpness without collapsing diversity

## Project Structure

```
var-research/
├── configs/                 # Configuration files
├── src/                     # Source code
│   ├── models/             # Model implementations
│   ├── training/           # Training scripts
│   ├── sampling/           # Sampling pipeline
│   └── utils/              # Utilities
├── data/                   # Dataset handling
├── experiments/            # Experiment logs and results
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train tokenizer** (Stage 0):
   ```bash
   python src/training/train_tokenizer.py --config configs/tokenizer_cifar10.yaml
   ```

3. **Train VAR scaffold** (Stage 1):
   ```bash
   python src/training/train_var.py --config configs/var_cifar10.yaml
   ```

4. **Train diffusion refiner** (Stage 2):
   ```bash
   python src/training/train_refiner.py --config configs/refiner_cifar10.yaml
   ```

5. **Generate samples**:
   ```bash
   python src/sampling/generate.py --config configs/sampling_cifar10.yaml
   ```

## Training Stages

### Stage 0: Tokenizer
- Hierarchical VQ-VAE with levels {8²→16²→32²→64²}
- Codebook size: 1024 per level, code dim: 8-16
- Datasets: CIFAR-10 (32²), ImageNet-64, ImageNet-128

### Stage 1: VAR Scaffold
- 12-16 layer decoder-only transformer, width 512, heads 8
- Rotary + ALiBi positional encoding
- Teacher forcing on next-scale codes

### Stage 2: Diffusion Refiner
- DiT-tiny or UNet-tiny in latent space (finest two scales only)
- Conditions on upsampled VAR image/features + class embedding
- Target: residual r = x* - x̂_VAR
- NFE: 5-10 steps with Heun/DPMSolver sampler

## Expected Results

- **Speed**: ≥5-10× faster than 50-step diffusion
- **Quality**: 15-30% FID improvement vs VAR-only at fixed wall-clock
- **Diversity**: Auto-Guidance maintains coverage vs CFG

## Compute Requirements

- **24GB GPU (FP16)**: Full CIFAR-10 → ImageNet-64 pipeline
- **Tokenizer**: 4-6h (CIFAR-10), 12-18h (ImageNet-64)
- **VAR**: 8-16h (CIFAR-10/ImageNet-64)
- **Refiner**: 6-12h per scale

## Citation

If you use this code, please cite:
```bibtex
@article{var-refine-2024,
  title={VAR-Refine: Hybrid Next-Scale Autoregression with Residual Diffusion Refinement},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## References

- VAR: "Visual Autoregressive Modeling: Next-Scale Prediction" (NeurIPS'24 Best Paper)
- Auto-Guidance: "Guiding a Diffusion Model with a Bad Version of Itself"
# var-residual_diffusion
