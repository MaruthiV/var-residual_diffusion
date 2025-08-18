"""
Hierarchical VQ-VAE for pyramid latent encoding.
Supports multiple scales: 8² → 16² → 32² → 64²
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


class VectorQuantizer(nn.Module):
    """Vector quantizer with EMA codebook update."""
    
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        
        # Codebook
        self.register_buffer('embedding', torch.randn(num_embeddings, embedding_dim))
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('embedding_avg', torch.zeros(num_embeddings, embedding_dim))
        
    def forward(self, inputs):
        # inputs: (B, C, H, W) -> (B*H*W, C)
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.t()))
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize
        quantized = torch.matmul(encodings, self.embedding)
        quantized = quantized.view(input_shape)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        # EMA update
        if self.training:
            self._ema_update(flat_input, encodings)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs.permute(0, 3, 1, 2))
        q_latent_loss = F.mse_loss(quantized, inputs.permute(0, 3, 1, 2).detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = inputs.permute(0, 3, 1, 2) + (quantized - inputs.permute(0, 3, 1, 2)).detach()
        
        return quantized, loss, encoding_indices.view(input_shape[0], input_shape[1], input_shape[2])
    
    def _ema_update(self, flat_input, encodings):
        """EMA update of codebook."""
        self.cluster_size.data.mul_(self.decay).add_(
            (1 - self.decay) * torch.sum(encodings, 0))
        
        n = torch.sum(self.cluster_size.data)
        self.cluster_size.data.add_(1e-5)
        
        embed_sum = torch.matmul(encodings.t(), flat_input)
        self.embedding_avg.data.mul_(self.decay).add_(
            (1 - self.decay) * embed_sum)
        
        self.embedding.data = self.embedding_avg / self.cluster_size.unsqueeze(1)


class ResidualBlock(nn.Module):
    """Residual block for encoder/decoder."""
    
    def __init__(self, in_channels, out_channels, hidden_channels=None):
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU()
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return self.relu(x + residual)


class Encoder(nn.Module):
    """Encoder for hierarchical VQ-VAE."""
    
    def __init__(self, in_channels, hidden_channels, num_res_blocks, pyramid_levels):
        super().__init__()
        self.pyramid_levels = pyramid_levels
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels, hidden_channels)
            for _ in range(num_res_blocks)
        ])
        
        # Downsampling layers for each pyramid level
        self.downsample_layers = nn.ModuleList()
        current_channels = hidden_channels
        
        for i in range(len(pyramid_levels) - 1):
            self.downsample_layers.append(nn.Sequential(
                nn.Conv2d(current_channels, current_channels * 2, 4, stride=2, padding=1),
                nn.ReLU(),
                ResidualBlock(current_channels * 2, current_channels * 2)
            ))
            current_channels *= 2
    
    def forward(self, x):
        # Initial encoding
        x = self.conv_in(x)
        
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Pyramid encoding
        pyramid_features = [x]
        
        for downsample in self.downsample_layers:
            x = downsample(x)
            pyramid_features.append(x)
        
        return pyramid_features


class Decoder(nn.Module):
    """Decoder for hierarchical VQ-VAE."""
    
    def __init__(self, hidden_channels, out_channels, num_res_blocks, pyramid_levels):
        super().__init__()
        self.pyramid_levels = pyramid_levels
        
        # Upsampling layers for each pyramid level
        self.upsample_layers = nn.ModuleList()
        # Start with the coarsest level channels (512 for [8,16,32] with hidden_channels=128)
        current_channels = hidden_channels * (2 ** (len(pyramid_levels) - 1))
        
        for i in range(len(pyramid_levels) - 1):
            next_channels = current_channels // 2
            self.upsample_layers.append(nn.Sequential(
                nn.ConvTranspose2d(current_channels, next_channels, 4, stride=2, padding=1),
                nn.ReLU(),
                ResidualBlock(next_channels, next_channels)
            ))
            current_channels = next_channels
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels, hidden_channels)
            for _ in range(num_res_blocks)
        ])
        
        # Final convolution
        self.conv_out = nn.Conv2d(hidden_channels, out_channels, 3, padding=1)
    
    def forward(self, pyramid_features):
        # Start from coarsest level
        x = pyramid_features[-1]
        
        # Upsample through pyramid levels
        for i, upsample in enumerate(self.upsample_layers):
            x = upsample(x)
            # Add skip connection if available
            if i < len(pyramid_features) - 1:
                skip_feature = pyramid_features[-(i+2)]
                # Ensure skip connection has same spatial size
                if skip_feature.shape[2:] != x.shape[2:]:
                    skip_feature = F.interpolate(skip_feature, size=x.shape[2:], mode='bilinear', align_corners=False)
                x = x + skip_feature
        
        # Final residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Output
        x = self.conv_out(x)
        return torch.tanh(x)


class HierarchicalVQVAE(nn.Module):
    """Hierarchical VQ-VAE with pyramid latents."""
    
    def __init__(self, in_channels=3, hidden_channels=128, num_res_blocks=2, 
                 pyramid_levels=[8, 16, 32], codebook_size=1024, code_dim=8, 
                 commitment_cost=0.25, decay=0.99):
        super().__init__()
        self.pyramid_levels = pyramid_levels
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        
        # Encoder
        self.encoder = Encoder(in_channels, hidden_channels, num_res_blocks, pyramid_levels)
        
        # Quantizers for each pyramid level
        self.quantizers = nn.ModuleList([
            VectorQuantizer(codebook_size, code_dim, commitment_cost, decay)
            for _ in pyramid_levels
        ])
        
        # Decoder
        self.decoder = Decoder(hidden_channels, in_channels, num_res_blocks, pyramid_levels)
        
        # Projection layers to match code_dim
        self.projections = nn.ModuleList([
            nn.Conv2d(hidden_channels * (2 ** i), code_dim, 1)
            for i in range(len(pyramid_levels))
        ])
        
        # Unprojection layers
        self.unprojections = nn.ModuleList([
            nn.Conv2d(code_dim, hidden_channels * (2 ** i), 1)
            for i in range(len(pyramid_levels))
        ])
    
    def encode(self, x):
        """Encode image to pyramid latents."""
        pyramid_features = self.encoder(x)
        
        quantized_features = []
        indices = []
        total_loss = 0
        
        for i, (feature, quantizer, proj) in enumerate(zip(pyramid_features, self.quantizers, self.projections)):
            # Project to code dimension
            feature_proj = proj(feature)
            
            # Quantize
            quantized, loss, idx = quantizer(feature_proj)
            quantized_features.append(quantized)
            indices.append(idx)
            total_loss += loss
        
        return quantized_features, indices, total_loss
    
    def decode(self, quantized_features):
        """Decode pyramid latents to image."""
        # Unproject each level
        unprojected_features = []
        for i, (quantized, unproj) in enumerate(zip(quantized_features, self.unprojections)):
            unprojected = unproj(quantized)
            unprojected_features.append(unprojected)
        
        # Decode
        return self.decoder(unprojected_features)
    
    def forward(self, x):
        """Forward pass."""
        quantized_features, indices, loss = self.encode(x)
        reconstructed = self.decode(quantized_features)
        return reconstructed, indices, loss
    
    def get_codebook_indices(self, x):
        """Get codebook indices for training VAR."""
        _, indices, _ = self.encode(x)
        return indices
