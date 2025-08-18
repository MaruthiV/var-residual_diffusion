#!/usr/bin/env python3
"""
Debug script for VAR transformer tensor shapes.
"""

import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.var_transformer import VARTransformer

def debug_var():
    """Debug VAR transformer tensor shapes."""
    print("Debugging VAR Transformer...")
    
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
    
    print(f"Input pyramid shapes: {[idx.shape for idx in pyramid_indices]}")
    
    # Test sequence preparation
    sequences = []
    for scale_idx, indices in enumerate(pyramid_indices):
        flat_indices = indices.view(batch_size, -1)
        print(f"Scale {scale_idx}: flat shape {flat_indices.shape}")
        
        # Add scale embedding
        scale_emb = var_model.scale_embedding(torch.full_like(flat_indices, scale_idx))
        print(f"Scale {scale_idx}: scale_emb shape {scale_emb.shape}")
        
        # Add position embedding
        pos_emb = var_model.pos_embedding(torch.arange(flat_indices.shape[1], device=indices.device))
        pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1)
        print(f"Scale {scale_idx}: pos_emb shape {pos_emb.shape}")
        
        # Code embedding
        code_emb = var_model.code_embedding(flat_indices)
        print(f"Scale {scale_idx}: code_emb shape {code_emb.shape}")
        
        # Combine embeddings
        combined_emb = code_emb + scale_emb + pos_emb
        sequences.append(combined_emb)
        print(f"Scale {scale_idx}: combined shape {combined_emb.shape}")
    
    # Concatenate all scales
    full_sequence = torch.cat(sequences, dim=1)
    print(f"Full sequence shape: {full_sequence.shape}")
    
    # Test transformer layers
    seq_len = full_sequence.shape[1]
    print(f"Sequence length: {seq_len}")
    
    # Create causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len, device=full_sequence.device), diagonal=1)
    mask = mask.bool()
    print(f"Mask shape: {mask.shape}")
    
    # Test first transformer layer
    first_layer = var_model.transformer_layers[0]
    print(f"First layer: {first_layer}")
    
    # Test attention
    attn = first_layer.attn
    print(f"Attention module: {attn}")
    
    # Test with first layer
    x = full_sequence
    print(f"Input to first layer: {x.shape}")
    
    # Test norm1
    norm1_out = first_layer.norm1(x)
    print(f"After norm1: {norm1_out.shape}")
    
    # Test attention
    try:
        attn_out = first_layer.attn(norm1_out, mask)
        print(f"After attention: {attn_out.shape}")
    except Exception as e:
        print(f"Attention failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    debug_var()
