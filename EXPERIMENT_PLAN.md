# VAR-Refine Experiment Plan

## Research Overview

**Title**: "VAR-Refine: Hybrid Next-Scale Autoregression with Residual Diffusion Refinement"

**Core Idea**: Generate a coarse-to-fine scaffold with Visual Autoregressive modeling (VAR), then run a tiny, few-step diffusion refiner only on the residuals at each scale. You inherit VAR's speed and structure, and diffusion's texture/faithfulness—without paying 50–100 diffusion steps.

## Method Overview

### Stage 0: Hierarchical VQ-VAE Tokenizer
- **Architecture**: Hierarchical VQ-VAE with pyramid levels {8²→16²→32²}
- **Codebook**: 1024 codes per level, 8-dimensional embeddings
- **Training**: Reconstruction + commitment + perceptual loss
- **Expected Time**: 4-6 hours on CIFAR-10, 12-18 hours on ImageNet-64

### Stage 1: VAR Transformer Scaffold
- **Architecture**: 12-layer decoder-only transformer (512 dim, 8 heads)
- **Positional Encoding**: Rotary + ALiBi
- **Training**: Next-scale prediction with teacher forcing
- **Expected Time**: 8-16 hours on CIFAR-10/ImageNet-64

### Stage 2: Diffusion Refiner
- **Architecture**: UNet-tiny (128 channels, 2 res blocks)
- **Conditioning**: VAR output + class labels
- **Target**: Residual prediction (r = x* - x̂_VAR)
- **Sampling**: 5-10 steps with Heun/DPMSolver
- **Expected Time**: 6-12 hours per scale

### Stage 3: Auto-Guidance
- **Method**: Guide with worse checkpoint (ε_θ ← ε_θ + α(ε_θ - ε_φ))
- **Benefits**: Improves quality without collapsing diversity
- **Hyperparameter**: α ∈ [0.3, 0.7]

## Experimental Setup

### Datasets
1. **CIFAR-10** (32²): Primary development dataset
2. **ImageNet-64**: Main evaluation dataset
3. **ImageNet-256** (subset): Stretch goal

### Baselines
1. **VAR-only**: Official VAR implementation
2. **DiT-tiny/UNet-small**: 50-step diffusion
3. **Consistency-distilled**: Optional comparison

### Metrics
- **Quality**: FID, Inception Score
- **Diversity**: Precision/Recall (Kynkäänniemi), LPIPS-diversity
- **Speed**: NFE/latency (ms/img), energy/img
- **Ablations**: Reconstruction error, codebook usage

## Experiment Pipeline

### Phase 1: Development (CIFAR-10)
```bash
# Train tokenizer
python run_experiment.py --dataset cifar10 --stage tokenizer --wandb

# Train VAR scaffold
python run_experiment.py --dataset cifar10 --stage var --wandb

# Train diffusion refiner
python run_experiment.py --dataset cifar10 --stage refiner --wandb

# Generate samples
python run_experiment.py --dataset cifar10 --stage sample --num_samples 1000
```

### Phase 2: Evaluation (ImageNet-64)
```bash
# Scale up to ImageNet-64
python run_experiment.py --dataset imagenet64 --stage all --wandb
```

### Phase 3: Ablations
```bash
# Test different NFE values
python src/sampling/generate.py --config configs/sampling_cifar10.yaml --num_steps 1
python src/sampling/generate.py --config configs/sampling_cifar10.yaml --num_steps 3
python src/sampling/generate.py --config configs/sampling_cifar10.yaml --num_steps 5
python src/sampling/generate.py --config configs/sampling_cifar10.yaml --num_steps 10
python src/sampling/generate.py --config configs/sampling_cifar10.yaml --num_steps 20
```

## Expected Results

### Primary Claims
1. **Quality**: 15-30% FID improvement vs VAR-only at fixed wall-clock
2. **Speed**: ≥5-10× faster than 50-step diffusion with comparable FID
3. **Diversity**: Auto-Guidance maintains coverage vs CFG

### Quantitative Targets
- **CIFAR-10**: FID < 5.0, IS > 8.0
- **ImageNet-64**: FID < 15.0, IS > 6.0
- **Latency**: < 100ms per image (32²)
- **NFE**: ≤10 diffusion steps

### Speed-Quality Pareto
| Method | FID | NFE | Latency (ms) | Speedup |
|--------|-----|-----|--------------|---------|
| VAR-only | 8.5 | 0 | 20 | 1× |
| VAR-Refine (5 steps) | 6.2 | 5 | 45 | 0.4× |
| VAR-Refine (10 steps) | 5.1 | 10 | 70 | 0.3× |
| Diffusion (50 steps) | 4.8 | 50 | 350 | 0.06× |

## Ablation Studies

### 1. No-refiner vs Refiner
- **Question**: How much lift does diffusion add?
- **Method**: Compare VAR-only vs VAR-Refine
- **Expected**: 15-30% FID improvement

### 2. Residual vs Direct Denoising
- **Question**: Does residual prediction work better?
- **Method**: Train refiner on residuals vs direct denoising
- **Expected**: Residual wins by 10-20%

### 3. Number of Scales Refined
- **Question**: Top-1 vs top-2 scales?
- **Method**: Refine only finest scale vs finest two scales
- **Expected**: Top-1 sufficient for most gains

### 4. NFE Sweep
- **Question**: Speed/quality Pareto curve?
- **Method**: Test 1, 3, 5, 10, 20 steps
- **Expected**: Diminishing returns after 10 steps

### 5. Guidance Type
- **Question**: CFG vs Auto-Guidance?
- **Method**: Compare classifier-free guidance vs Auto-Guidance
- **Expected**: Auto-Guidance better diversity

### 6. Conditioning Location
- **Question**: Pixels vs latent conditioning?
- **Method**: Condition on VAR pixels vs VAR features
- **Expected**: Feature conditioning slightly better

## Implementation Details

### Training Configuration
```yaml
# Tokenizer (Stage 0)
model:
  pyramid_levels: [8, 16, 32]
  codebook_size: 1024
  code_dim: 8
  hidden_dim: 128

# VAR (Stage 1)
model:
  num_layers: 12
  hidden_dim: 512
  num_heads: 8
  use_rotary: true
  use_alibi: true

# Refiner (Stage 2)
model:
  model_channels: 128
  num_res_blocks: 2
  num_steps: 10
  guidance_scale: 1.0
```

### Sampling Pipeline
1. **VAR Scaffold**: Generate 8²→16²→32² autoregressively
2. **Diffusion Refine**: 5-10 steps on finest scale only
3. **Auto-Guidance**: Apply with α = 0.5
4. **Decode**: Convert to pixels with VQ-VAE

### Compute Requirements
- **24GB GPU**: Full CIFAR-10 pipeline
- **Tokenizer**: 4-6h (CIFAR-10), 12-18h (ImageNet-64)
- **VAR**: 8-16h (CIFAR-10/ImageNet-64)
- **Refiner**: 6-12h per scale

## Evaluation Protocol

### Metrics Computation
```python
# FID Score
fid = compute_fid_score(generated_images, real_images)

# Inception Score
is_score = compute_inception_score(generated_images)

# LPIPS Diversity
lpips = compute_lpips_distance(generated_images, real_images)

# Latency
latency = total_time / num_samples
```

### Statistical Significance
- **Sample Size**: 10,000 generated images
- **Confidence**: 95% confidence intervals
- **Runs**: 3 independent runs per configuration

## Risk Mitigation

### Technical Risks
1. **VQ-VAE Bottlenecks**: Use EMA codebook, per-scale commit losses
2. **Refiner Overfitting**: Early stopping + augmentations
3. **Auto-Guidance Sensitivity**: Grid search over α

### Implementation Risks
1. **Memory Issues**: Gradient checkpointing, smaller batches
2. **Training Instability**: Careful learning rate scheduling
3. **Codebook Collapse**: Monitor codebook usage

## Timeline

### Week 1-2: Development
- Implement and debug tokenizer
- Train on CIFAR-10
- Basic VAR implementation

### Week 3-4: VAR Training
- Complete VAR transformer
- Train on CIFAR-10
- Initial sampling pipeline

### Week 5-6: Refiner Development
- Implement diffusion refiner
- Train on CIFAR-10
- Auto-Guidance implementation

### Week 7-8: Evaluation
- Full pipeline evaluation
- Ablation studies
- Paper writing

## Success Criteria

### Minimum Viable Results
- [ ] VAR-Refine improves FID vs VAR-only by ≥15%
- [ ] VAR-Refine is ≥5× faster than 50-step diffusion
- [ ] Auto-Guidance maintains diversity vs CFG

### Stretch Goals
- [ ] FID improvement ≥30%
- [ ] Speed improvement ≥10×
- [ ] ImageNet-256 results
- [ ] Real-time generation (<50ms)

## References

1. **VAR**: "Visual Autoregressive Modeling: Next-Scale Prediction" (NeurIPS'24 Best Paper)
2. **Auto-Guidance**: "Guiding a Diffusion Model with a Bad Version of Itself"
3. **VQ-VAE**: "Neural Discrete Representation Learning"
4. **Diffusion**: "Denoising Diffusion Probabilistic Models"
