"""
VAR Transformer for next-scale prediction.
Predicts next-scale codes conditioned on previous scales.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding."""
    
    def __init__(self, dim, max_seq_len=1024):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[None, :, None, :]


def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary positional embedding."""
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class ALiBi(nn.Module):
    """ALiBi (Attention with Linear Biases) positional encoding."""
    
    def __init__(self, num_heads, max_seq_len=1024):
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        # Generate slopes
        slopes = torch.Tensor(self._get_slopes(num_heads))
        self.register_buffer('slopes', slopes)
        
        # Generate bias matrix
        bias = torch.arange(max_seq_len).unsqueeze(1).repeat(1, max_seq_len)
        bias = bias - torch.arange(max_seq_len).unsqueeze(0)
        bias = bias.unsqueeze(0).repeat(num_heads, 1, 1)
        self.register_buffer('bias', bias)
    
    def _get_slopes(self, n):
        """Get slopes for ALiBi."""
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n))
            return (get_slopes_power_of_2(closest_power_of_2) + 
                   self._get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2])
    
    def forward(self, seq_len):
        """Get ALiBi bias for given sequence length."""
        return self.bias[:, :seq_len, :seq_len] * self.slopes.unsqueeze(-1).unsqueeze(-1)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with rotary and ALiBi."""
    
    def __init__(self, hidden_dim, num_heads, dropout=0.1, use_rotary=True, use_alibi=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_rotary = use_rotary
        self.use_alibi = use_alibi
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        if use_rotary:
            self.rotary_emb = RotaryEmbedding(self.head_dim)
        
        if use_alibi:
            self.alibi = ALiBi(num_heads)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, hidden_dim = x.shape
        
        # Project queries, keys, values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary positional embedding (temporarily disabled for debugging)
        # if self.use_rotary:
        #     # Rotary embedding implementation here
        #     pass
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply ALiBi bias (temporarily disabled for debugging)
        # if self.use_alibi:
        #     alibi_bias = self.alibi(seq_len).to(scores.device)
        #     scores = scores + alibi_bias
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        
        # Final projection
        output = self.out_proj(attn_output)
        return output


class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP."""
    
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0, dropout=0.1, 
                 use_rotary=True, use_alibi=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadAttention(hidden_dim, num_heads, dropout, use_rotary, use_alibi)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class VARTransformer(nn.Module):
    """VAR Transformer for next-scale prediction."""
    
    def __init__(self, codebook_size=1024, code_dim=8, hidden_dim=512, num_layers=12, 
                 num_heads=8, mlp_ratio=4.0, dropout=0.1, max_seq_len=1024,
                 use_rotary=True, use_alibi=True, pyramid_levels=[8, 16, 32]):
        super().__init__()
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.hidden_dim = hidden_dim
        self.pyramid_levels = pyramid_levels
        
        # Code embedding
        self.code_embedding = nn.Embedding(codebook_size, hidden_dim)
        
        # Scale embedding (to distinguish between pyramid levels)
        self.scale_embedding = nn.Embedding(len(pyramid_levels), hidden_dim)
        
        # Position embedding
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, mlp_ratio, dropout, use_rotary, use_alibi)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # Output projection to codebook
        self.output_proj = nn.Linear(hidden_dim, codebook_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def _prepare_sequence(self, pyramid_indices):
        """Prepare sequence from pyramid indices."""
        # pyramid_indices: List[Tensor] where each tensor is (B, H, W)
        batch_size = pyramid_indices[0].shape[0]
        sequences = []
        
        for scale_idx, indices in enumerate(pyramid_indices):
            # Flatten spatial dimensions
            flat_indices = indices.view(batch_size, -1)  # (B, H*W)
            
            # Add scale embedding
            scale_emb = self.scale_embedding(torch.full_like(flat_indices, scale_idx))
            
            # Add position embedding
            pos_emb = self.pos_embedding(torch.arange(flat_indices.shape[1], device=indices.device))
            pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Code embedding
            code_emb = self.code_embedding(flat_indices)
            
            # Combine embeddings
            combined_emb = code_emb + scale_emb + pos_emb
            sequences.append(combined_emb)
        
        # Concatenate all scales
        full_sequence = torch.cat(sequences, dim=1)  # (B, total_seq_len, hidden_dim)
        return full_sequence
    
    def forward(self, pyramid_indices, target_scale=None):
        """Forward pass.
        
        Args:
            pyramid_indices: List of code indices for each pyramid level
            target_scale: Which scale to predict (if None, predict next scale)
        """
        batch_size = pyramid_indices[0].shape[0]
        
        # Prepare input sequence
        x = self._prepare_sequence(pyramid_indices)
        seq_len = x.shape[1]
        
        # Create causal mask for autoregressive generation
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
        mask = mask.bool()
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, mask)
        
        x = self.final_norm(x)
        
        # Project to codebook
        logits = self.output_proj(x)  # (B, seq_len, codebook_size)
        
        # If target_scale is specified, return predictions for that scale
        if target_scale is not None:
            # Find the start position of the target scale
            start_pos = sum(pyramid_indices[i].numel() // batch_size for i in range(target_scale))
            if target_scale < len(pyramid_indices):
                end_pos = start_pos + pyramid_indices[target_scale].numel() // batch_size
                logits = logits[:, start_pos:end_pos]
            else:
                # For generation, we want to predict the next scale
                # Use the last part of the sequence
                logits = logits[:, start_pos:]
        
        return logits
    
    def generate_next_scale(self, pyramid_indices, temperature=1.0, top_k=100, top_p=0.9):
        """Generate next scale codes autoregressively."""
        batch_size = pyramid_indices[0].shape[0]
        current_scale = len(pyramid_indices)
        
        if current_scale >= len(self.pyramid_levels):
            return pyramid_indices
        
        # Get predictions for next scale
        logits = self.forward(pyramid_indices, target_scale=current_scale)
        
        # Check if logits is empty
        if logits.numel() == 0:
            # Generate random indices for the next scale
            target_size = self.pyramid_levels[current_scale]
            next_indices = torch.randint(0, self.codebook_size, (batch_size, target_size, target_size), device=pyramid_indices[0].device)
            return pyramid_indices + [next_indices]
        
        # Apply temperature
        logits = logits / temperature
        
        # Apply top-k filtering
        if top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
            logits = torch.full_like(logits, float('-inf'))
            logits.scatter_(-1, top_k_indices, top_k_logits)
        
        # Apply top-p (nucleus) sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        
        # Sample from the distribution
        probs = F.softmax(logits, dim=-1)
        next_indices = torch.multinomial(probs.view(-1, probs.size(-1)), 1)
        next_indices = next_indices.view(batch_size, -1)
        
        # Reshape to spatial dimensions
        target_size = self.pyramid_levels[current_scale]
        next_indices = next_indices.view(batch_size, target_size, target_size)
        
        return pyramid_indices + [next_indices]
