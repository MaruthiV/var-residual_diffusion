#!/usr/bin/env python3
"""
Basic test script to verify model functionality.
"""

import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.hierarchical_vqvae import HierarchicalVQVAE
from models.var_transformer import VARTransformer
from models.minimal_refiner import MinimalDiffusionRefiner as DiffusionRefiner


def test_tokenizer():
    """Test hierarchical VQ-VAE tokenizer."""
    print("Testing Hierarchical VQ-VAE Tokenizer...")
    
    # Create model
    tokenizer = HierarchicalVQVAE(
        in_channels=3,
        hidden_channels=128,
        num_res_blocks=2,
        pyramid_levels=[8, 16, 32],
        codebook_size=1024,
        code_dim=8,
        commitment_cost=0.25,
        decay=0.99
    )
    
    # Create dummy input
    batch_size = 4
    image_size = 32
    x = torch.randn(batch_size, 3, image_size, image_size)
    
    # Forward pass
    reconstructed, indices, loss = tokenizer(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {reconstructed.shape}")
    print(f"  Number of pyramid levels: {len(indices)}")
    print(f"  Loss: {loss.item():.4f}")
    
    # Test encoding
    pyramid_indices = tokenizer.get_codebook_indices(x)
    print(f"  Encoded indices shape: {[idx.shape for idx in pyramid_indices]}")
    
    print("✓ Tokenizer test passed\n")


def test_var_transformer():
    """Test VAR transformer."""
    print("Testing VAR Transformer...")
    
    # Create model
    var_model = VARTransformer(
        codebook_size=1024,
        code_dim=8,
        hidden_dim=512,
        num_layers=12,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        max_seq_len=1024,
        use_rotary=True,
        use_alibi=True,
        pyramid_levels=[8, 16, 32]
    )
    
    # Create dummy input (pyramid indices)
    batch_size = 4
    pyramid_indices = [
        torch.randint(0, 1024, (batch_size, 8, 8)),
        torch.randint(0, 1024, (batch_size, 16, 16)),
        torch.randint(0, 1024, (batch_size, 32, 32))
    ]
    
    # Forward pass
    logits = var_model(pyramid_indices, target_scale=2)
    
    print(f"  Input pyramid shapes: {[idx.shape for idx in pyramid_indices]}")
    print(f"  Output logits shape: {logits.shape}")
    
    # Test generation
    generated_indices = var_model.generate_next_scale(
        pyramid_indices[:2],  # Only first two scales
        temperature=1.0,
        top_k=100,
        top_p=0.9
    )
    print(f"  Generated indices shape: {[idx.shape for idx in generated_indices]}")
    
    print("✓ VAR Transformer test passed\n")


def test_diffusion_refiner():
    """Test diffusion refiner."""
    print("Testing Diffusion Refiner...")
    
    # Create model
    refiner = DiffusionRefiner(
        in_channels=3,
        out_channels=3,
        model_channels=128,
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02
    )
    
    # Create dummy inputs
    batch_size = 4
    image_size = 32
    x = torch.randn(batch_size, 3, image_size, image_size)
    var_condition = torch.randn(batch_size, 3, image_size, image_size)
    class_labels = torch.randint(0, 10, (batch_size,))
    
    # Forward pass (training)
    loss = refiner.p_losses(x, var_condition, class_labels)
    print(f"  Input shape: {x.shape}")
    print(f"  VAR condition shape: {var_condition.shape}")
    print(f"  Class labels shape: {class_labels.shape}")
    print(f"  Training loss: {loss.item():.4f}")
    
    # Test sampling
    with torch.no_grad():
        sampled = refiner.sample(
            var_condition,
            class_labels=class_labels,
            num_steps=10,
            guidance_scale=1.0
        )
    print(f"  Sampled output shape: {sampled.shape}")
    
    print("✓ Diffusion Refiner test passed\n")


def test_full_pipeline():
    """Test the full pipeline end-to-end."""
    print("Testing Full Pipeline...")
    
    # Create models
    tokenizer = HierarchicalVQVAE(
        in_channels=3, hidden_channels=128, num_res_blocks=2,
        pyramid_levels=[8, 16, 32], codebook_size=1024, code_dim=8
    )
    
    var_model = VARTransformer(
        codebook_size=1024, code_dim=8, hidden_dim=512,
        num_layers=12, num_heads=8, pyramid_levels=[8, 16, 32]
    )
    
    refiner = DiffusionRefiner(
        in_channels=3, out_channels=3, model_channels=128,
        num_timesteps=1000
    )
    
    # Create dummy input
    batch_size = 2
    image_size = 32
    x = torch.randn(batch_size, 3, image_size, image_size)
    class_labels = torch.randint(0, 10, (batch_size,))
    
    print(f"  Input shape: {x.shape}")
    
    # Step 1: Encode with tokenizer
    pyramid_indices = tokenizer.get_codebook_indices(x)
    print(f"  Encoded pyramid: {[idx.shape for idx in pyramid_indices]}")
    
    # Step 2: Generate VAR scaffold (simplified - just use first two scales)
    var_indices = var_model.generate_next_scale(
        pyramid_indices[:2], temperature=1.0, top_k=100, top_p=0.9
    )
    print(f"  VAR generated: {[idx.shape for idx in var_indices]}")
    
    # Step 3: Decode VAR output (simplified)
    with torch.no_grad():
        # For now, just use the original image as VAR output
        var_image = x
    print(f"  VAR image shape: {var_image.shape}")
    
    # Step 4: Refine with diffusion
    with torch.no_grad():
        refined_image = refiner.sample(
            var_image, class_labels=class_labels, num_steps=5
        )
    print(f"  Refined image shape: {refined_image.shape}")
    
    print("✓ Full pipeline test passed\n")


def main():
    """Run all tests."""
    print("Running VAR-Refine Basic Tests")
    print("=" * 50)
    
    try:
        test_tokenizer()
        test_var_transformer()
        test_diffusion_refiner()
        test_full_pipeline()
        
        print("🎉 All tests passed! The basic functionality is working.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Train tokenizer: python run_experiment.py --dataset cifar10 --stage tokenizer")
        print("3. Train VAR: python run_experiment.py --dataset cifar10 --stage var")
        print("4. Generate samples: python run_experiment.py --dataset cifar10 --stage sample")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
