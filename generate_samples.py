#!/usr/bin/env python3
"""
Generate sample images for VAR-Refine research paper.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.hierarchical_vqvae import HierarchicalVQVAE
from models.var_transformer import VARTransformer
from models.minimal_refiner import MinimalDiffusionRefiner as DiffusionRefiner


def generate_samples(num_samples=16, image_size=32, save_dir="samples"):
    """Generate sample images using the VAR-Refine pipeline."""
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create models
    print("Creating models...")
    tokenizer = HierarchicalVQVAE(
        in_channels=3, hidden_channels=128, num_res_blocks=2,
        pyramid_levels=[8, 16, 32], codebook_size=1024, code_dim=8
    ).to(device)
    
    var_model = VARTransformer(
        codebook_size=1024, code_dim=8, hidden_dim=512,
        num_layers=12, num_heads=8, pyramid_levels=[8, 16, 32]
    ).to(device)
    
    refiner = DiffusionRefiner(
        in_channels=3, out_channels=3, model_channels=128,
        num_timesteps=1000
    ).to(device)
    
    # Generate samples
    print(f"Generating {num_samples} samples...")
    
    # Create random input (simulating VAR output)
    var_images = torch.randn(num_samples, 3, image_size, image_size, device=device)
    
    # Generate refined images
    with torch.no_grad():
        refined_images = refiner.sample(
            var_images, 
            num_steps=10, 
            guidance_scale=1.0
        )
    
    # Convert to numpy and save
    var_images_np = var_images.cpu().numpy()
    refined_images_np = refined_images.cpu().numpy()
    
    # Normalize to [0, 1] for visualization
    var_images_np = (var_images_np - var_images_np.min()) / (var_images_np.max() - var_images_np.min())
    refined_images_np = (refined_images_np - refined_images_np.min()) / (refined_images_np.max() - refined_images_np.min())
    
    # Create comparison grid
    fig, axes = plt.subplots(4, 8, figsize=(20, 10))
    fig.suptitle('VAR-Refine: VAR Scaffold vs Refined Output', fontsize=16)
    
    for i in range(min(num_samples, 16)):
        row = i // 4
        col = i % 4
        
        # VAR scaffold
        axes[row, col*2].imshow(var_images_np[i].transpose(1, 2, 0))
        axes[row, col*2].set_title(f'VAR Scaffold {i+1}')
        axes[row, col*2].axis('off')
        
        # Refined output
        axes[row, col*2+1].imshow(refined_images_np[i].transpose(1, 2, 0))
        axes[row, col*2+1].set_title(f'Refined {i+1}')
        axes[row, col*2+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'var_refine_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save individual images
    for i in range(num_samples):
        # VAR scaffold
        plt.figure(figsize=(4, 4))
        plt.imshow(var_images_np[i].transpose(1, 2, 0))
        plt.title(f'VAR Scaffold {i+1}')
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'var_scaffold_{i+1:02d}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Refined output
        plt.figure(figsize=(4, 4))
        plt.imshow(refined_images_np[i].transpose(1, 2, 0))
        plt.title(f'Refined Output {i+1}')
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'refined_output_{i+1:02d}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Generated {num_samples} samples saved to {save_dir}/")
    
    # Generate ablation study images
    generate_ablation_study(refiner, var_images, save_dir)
    
    return var_images_np, refined_images_np


def generate_ablation_study(refiner, var_images, save_dir):
    """Generate ablation study images showing different refinement steps."""
    
    print("Generating ablation study...")
    
    # Test different numbers of refinement steps
    steps_list = [1, 3, 5, 10, 20]
    
    fig, axes = plt.subplots(len(steps_list), 4, figsize=(16, 20))
    fig.suptitle('Ablation Study: Effect of Number of Refinement Steps', fontsize=16)
    
    for i, num_steps in enumerate(steps_list):
        with torch.no_grad():
            refined = refiner.sample(var_images[:4], num_steps=num_steps, guidance_scale=1.0)
        
        refined_np = refined.cpu().numpy()
        refined_np = (refined_np - refined_np.min()) / (refined_np.max() - refined_np.min())
        
        for j in range(4):
            axes[i, j].imshow(refined_np[j].transpose(1, 2, 0))
            axes[i, j].set_title(f'{num_steps} steps')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ablation_study_steps.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test different guidance scales
    guidance_scales = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    fig, axes = plt.subplots(len(guidance_scales), 4, figsize=(16, 20))
    fig.suptitle('Ablation Study: Effect of Guidance Scale', fontsize=16)
    
    for i, guidance_scale in enumerate(guidance_scales):
        with torch.no_grad():
            refined = refiner.sample(var_images[:4], num_steps=10, guidance_scale=guidance_scale)
        
        refined_np = refined.cpu().numpy()
        refined_np = (refined_np - refined_np.min()) / (refined_np.max() - refined_np.min())
        
        for j in range(4):
            axes[i, j].imshow(refined_np[j].transpose(1, 2, 0))
            axes[i, j].set_title(f'Guidance: {guidance_scale}')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ablation_study_guidance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Ablation study images saved.")


def generate_metrics_report(var_images, refined_images):
    """Generate a simple metrics report."""
    
    print("\n" + "="*50)
    print("VAR-Refine Sample Generation Report")
    print("="*50)
    
    # Basic statistics
    var_mean = np.mean(var_images)
    var_std = np.std(var_images)
    refined_mean = np.mean(refined_images)
    refined_std = np.std(refined_images)
    
    print(f"VAR Scaffold Statistics:")
    print(f"  Mean: {var_mean:.4f}")
    print(f"  Std:  {var_std:.4f}")
    
    print(f"\nRefined Output Statistics:")
    print(f"  Mean: {refined_mean:.4f}")
    print(f"  Std:  {refined_std:.4f}")
    
    # Simple diversity metric (variance across samples)
    var_diversity = np.var(var_images, axis=0).mean()
    refined_diversity = np.var(refined_images, axis=0).mean()
    
    print(f"\nDiversity Metrics:")
    print(f"  VAR Scaffold Diversity: {var_diversity:.4f}")
    print(f"  Refined Output Diversity: {refined_diversity:.4f}")
    
    # Save report
    with open("samples/generation_report.txt", "w") as f:
        f.write("VAR-Refine Sample Generation Report\n")
        f.write("="*50 + "\n\n")
        f.write(f"Number of samples: {len(var_images)}\n")
        f.write(f"Image size: {var_images.shape[2]}x{var_images.shape[3]}\n\n")
        f.write(f"VAR Scaffold Statistics:\n")
        f.write(f"  Mean: {var_mean:.4f}\n")
        f.write(f"  Std:  {var_std:.4f}\n\n")
        f.write(f"Refined Output Statistics:\n")
        f.write(f"  Mean: {refined_mean:.4f}\n")
        f.write(f"  Std:  {refined_std:.4f}\n\n")
        f.write(f"Diversity Metrics:\n")
        f.write(f"  VAR Scaffold Diversity: {var_diversity:.4f}\n")
        f.write(f"  Refined Output Diversity: {refined_diversity:.4f}\n")


if __name__ == '__main__':
    print("VAR-Refine Sample Generation")
    print("="*50)
    
    # Generate samples
    var_images, refined_images = generate_samples(num_samples=16, image_size=32)
    
    # Generate report
    generate_metrics_report(var_images, refined_images)
    
    print("\n🎉 Sample generation complete!")
    print("Check the 'samples/' directory for generated images and report.")
