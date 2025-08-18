#!/usr/bin/env python3
"""
Debug script for diffusion refiner channel dimensions.
"""

import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.diffusion_refiner import DiffusionRefiner

def debug_refiner():
    """Debug diffusion refiner channel dimensions."""
    print("Debugging Diffusion Refiner...")
    
    # Create model
    refiner = DiffusionRefiner(
        in_channels=3,
        out_channels=3,
        model_channels=128,
        num_res_blocks=2,
        attention_resolutions=[16],
        dropout=0.1,
        channel_mult=[1, 2, 4],
        num_heads=4,
        condition_on_var=True,
        condition_on_class=True,
        num_classes=10,
        noise_schedule="linear",
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
    
    print(f"Input shape: {x.shape}")
    print(f"VAR condition shape: {var_condition.shape}")
    print(f"Class labels shape: {class_labels.shape}")
    
    # Test UNet forward pass
    timesteps = torch.randint(0, 1000, (batch_size,))
    print(f"Timesteps shape: {timesteps.shape}")
    
    try:
        # Test UNet directly
        unet_output = refiner.unet(x, timesteps, var_condition, class_labels)
        print(f"UNet output shape: {unet_output.shape}")
    except Exception as e:
        print(f"UNet failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Test step by step
        print("\nTesting UNet step by step...")
        
        # Time embedding
        time_emb = refiner.unet.time_embed(timesteps)
        print(f"Time embedding shape: {time_emb.shape}")
        
        # Class embedding
        if refiner.unet.condition_on_class and class_labels is not None:
            class_emb = refiner.unet.class_embed(class_labels)
            time_emb = time_emb + class_emb
            print(f"Time + class embedding shape: {time_emb.shape}")
        
        # Input projection
        h = refiner.unet.input_proj(x)
        print(f"After input projection: {h.shape}")
        
        # VAR condition projection
        var_cond = None
        if refiner.unet.condition_on_var and var_condition is not None:
            var_cond = refiner.unet.var_proj(var_condition)
            print(f"VAR condition projection: {var_cond.shape}")
        
        # Test first down block
        print(f"Number of down blocks: {len(refiner.unet.down_blocks)}")
        for i, layers in enumerate(refiner.unet.down_blocks):
            print(f"Down block {i}: {layers}")
            for j, layer in enumerate(layers):
                print(f"  Layer {j}: {layer}")
                if hasattr(layer, 'in_channels') and hasattr(layer, 'out_channels'):
                    print(f"    in_channels: {layer.in_channels}, out_channels: {layer.out_channels}")
        
        # Test downsampling
        print(f"Number of down samples: {len(refiner.unet.down_samples)}")
        for i, downsample in enumerate(refiner.unet.down_samples):
            print(f"Down sample {i}: {downsample}")
            if hasattr(downsample, 'in_channels') and hasattr(downsample, 'out_channels'):
                print(f"  in_channels: {downsample.in_channels}, out_channels: {downsample.out_channels}")

if __name__ == '__main__':
    debug_refiner()
