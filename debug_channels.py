#!/usr/bin/env python3
"""
Debug script to understand channel dimensions in hierarchical VQ-VAE.
"""

import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.hierarchical_vqvae import HierarchicalVQVAE

def debug_tokenizer():
    """Debug the tokenizer channel dimensions."""
    print("Debugging Hierarchical VQ-VAE Tokenizer...")
    
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
    
    print(f"Input shape: {x.shape}")
    
    # Test encoding
    pyramid_features = tokenizer.encoder(x)
    print(f"Encoder output shapes: {[f.shape for f in pyramid_features]}")
    
    # Test quantization
    quantized_features = []
    indices = []
    total_loss = 0
    
    for i, (feature, quantizer, proj) in enumerate(zip(pyramid_features, tokenizer.quantizers, tokenizer.projections)):
        print(f"Level {i}: feature shape {feature.shape}, proj output shape {proj(feature).shape}")
        feature_proj = proj(feature)
        quantized, loss, idx = quantizer(feature_proj)
        quantized_features.append(quantized)
        indices.append(idx)
        total_loss += loss
        print(f"Level {i}: quantized shape {quantized.shape}")
    
    # Test unprojection
    unprojected_features = []
    for i, (quantized, unproj) in enumerate(zip(quantized_features, tokenizer.unprojections)):
        unprojected = unproj(quantized)
        unprojected_features.append(unprojected)
        print(f"Level {i}: unprojected shape {unprojected.shape}")
    
    # Test decoder
    print(f"Decoder input shapes: {[f.shape for f in unprojected_features]}")
    
    # Start from coarsest level
    x = unprojected_features[-1]
    print(f"Starting with shape: {x.shape}")
    
    # Check decoder architecture
    print(f"Number of upsample layers: {len(tokenizer.decoder.upsample_layers)}")
    for i, upsample in enumerate(tokenizer.decoder.upsample_layers):
        print(f"Upsample layer {i}: {upsample}")
        # Check the ConvTranspose2d layer
        conv_transpose = upsample[0]
        print(f"  ConvTranspose2d: in_channels={conv_transpose.in_channels}, out_channels={conv_transpose.out_channels}")
        print(f"  Expected input channels: {conv_transpose.in_channels}, but got {x.shape[1]}")

if __name__ == '__main__':
    debug_tokenizer()
