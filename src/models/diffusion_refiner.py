"""
Diffusion Refiner for residual prediction.
Conditions on VAR output and class labels, predicts residuals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for timesteps."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block for UNet."""
    
    def __init__(self, in_channels, out_channels, time_channels, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, time_emb, var_condition=None):
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        
        # Add VAR condition if provided
        if var_condition is not None:
            # Ensure VAR condition has the same spatial size
            if var_condition.shape[2:] != h.shape[2:]:
                var_condition = F.interpolate(var_condition, size=h.shape[2:], mode='bilinear', align_corners=False)
            h = h + var_condition
        
        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Self-attention block."""
    
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        scale = 1 / math.sqrt(math.sqrt(C))
        
        attn = torch.einsum("bchw,bcij->bhwij", q * scale, k * scale)
        attn = attn.softmax(dim=-1)
        
        h = torch.einsum("bhwij,bcij->bchw", attn, v)
        h = self.proj(h)
        
        return x + h


class UNet(nn.Module):
    """UNet for diffusion refiner."""
    
    def __init__(self, in_channels=3, out_channels=3, model_channels=128, 
                 num_res_blocks=2, attention_resolutions=[16], dropout=0.1,
                 channel_mult=[1, 2, 4], num_heads=4, condition_on_var=True,
                 condition_on_class=True, num_classes=10):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.condition_on_var = condition_on_var
        self.condition_on_class = condition_on_class
        
        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Class embedding
        if condition_on_class:
            self.class_embed = nn.Embedding(num_classes, time_embed_dim)
        
        # Input projection
        self.input_proj = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # VAR condition projection
        if condition_on_var:
            self.var_proj = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Downsampling
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        channels = [model_channels]
        now_channels = model_channels
        
        for level, mult in enumerate(channel_mult):
            out_channels = model_channels * mult
            
            for _ in range(num_res_blocks):
                layers = [
                    ResidualBlock(now_channels, out_channels, time_embed_dim, dropout)
                ]
                if out_channels in attention_resolutions:
                    layers.append(AttentionBlock(out_channels, num_heads))
                self.down_blocks.append(nn.ModuleList(layers))
                now_channels = out_channels
                channels.append(now_channels)
            
            if level != len(channel_mult) - 1:
                # Downsample and double channels
                next_channels = model_channels * channel_mult[level + 1]
                self.down_samples.append(nn.Conv2d(now_channels, next_channels, 3, stride=2, padding=1))
                now_channels = next_channels
                channels.append(now_channels)
        
        # Middle
        self.middle_block = nn.ModuleList([
            ResidualBlock(now_channels, now_channels, time_embed_dim, dropout),
            AttentionBlock(now_channels, num_heads),
            ResidualBlock(now_channels, now_channels, time_embed_dim, dropout),
        ])
        
        # Upsampling
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        for level, mult in list(enumerate(channel_mult))[::-1]:
            out_channels = model_channels * mult
            
            for _ in range(num_res_blocks + 1):
                layers = [
                    ResidualBlock(channels.pop() + now_channels, out_channels, time_embed_dim, dropout)
                ]
                if out_channels in attention_resolutions:
                    layers.append(AttentionBlock(out_channels, num_heads))
                self.up_blocks.append(nn.ModuleList(layers))
                now_channels = out_channels
            
            if level != 0:
                self.up_samples.append(nn.ConvTranspose2d(now_channels, now_channels, 4, stride=2, padding=1))
        
        # Output
        self.out = nn.Sequential(
            nn.GroupNorm(32, now_channels),
            nn.SiLU(),
            nn.Conv2d(now_channels, out_channels, 3, padding=1),
        )
    
    def forward(self, x, timesteps, var_condition=None, class_labels=None):
        """Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            timesteps: Timestep tensor (B,)
            var_condition: VAR output condition (B, C, H, W)
            class_labels: Class labels (B,)
        """
        # Time embedding
        time_emb = self.time_embed(timesteps)
        
        # Class embedding
        if self.condition_on_class and class_labels is not None:
            class_emb = self.class_embed(class_labels)
            time_emb = time_emb + class_emb
        
        # Input projection
        h = self.input_proj(x)
        
        # VAR condition projection
        var_cond = None
        if self.condition_on_var and var_condition is not None:
            var_cond = self.var_proj(var_condition)
            # Ensure VAR condition has the same spatial size as current feature
            if var_condition.shape[2:] != h.shape[2:]:
                var_cond = F.interpolate(var_cond, size=h.shape[2:], mode='bilinear', align_corners=False)
        
        # Downsampling
        hs = []
        for level, layers in enumerate(self.down_blocks):
            for layer in layers:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, time_emb, var_cond)
                else:
                    h = layer(h)
            hs.append(h)
            
            if level < len(self.down_samples):
                h = self.down_samples[level](h)
                hs.append(h)
        
        # Middle
        for layer in self.middle_block:
            if isinstance(layer, ResidualBlock):
                h = layer(h, time_emb, var_cond)
            else:
                h = layer(h)
        
        # Upsampling
        for level, layers in enumerate(self.up_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in layers:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, time_emb, var_cond)
                else:
                    h = layer(h)
            
            if level < len(self.up_samples):
                h = self.up_samples[level](h)
        
        # Output
        return self.out(h)


class DiffusionRefiner(nn.Module):
    """Diffusion refiner for residual prediction."""
    
    def __init__(self, in_channels=3, out_channels=3, model_channels=128, 
                 num_res_blocks=2, attention_resolutions=[16], dropout=0.1,
                 channel_mult=[1, 2, 4], num_heads=4, condition_on_var=True,
                 condition_on_class=True, num_classes=10, noise_schedule="linear",
                 num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.unet = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            model_channels=model_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            num_heads=num_heads,
            condition_on_var=condition_on_var,
            condition_on_class=condition_on_class,
            num_classes=num_classes
        )
        
        # Noise schedule
        self.num_timesteps = num_timesteps
        if noise_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif noise_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown noise schedule: {noise_schedule}")
        
        # Pre-compute noise schedule parameters
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine beta schedule."""
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps) / timesteps
        alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def q_sample(self, x_start, t, noise=None):
        """Sample from q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_start, var_condition=None, class_labels=None, noise=None):
        """Compute loss for training."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device).long()
        x_noisy = self.q_sample(x_start, t, noise=noise)
        
        predicted_noise = self.unet(x_noisy, t, var_condition, class_labels)
        
        loss = F.mse_loss(predicted_noise, noise, reduction='none')
        loss = loss.mean(dim=[1, 2, 3])
        
        return loss.mean()
    
    def forward(self, x, var_condition=None, class_labels=None):
        """Forward pass for training."""
        return self.p_losses(x, var_condition, class_labels)
    
    @torch.no_grad()
    def p_sample(self, x, t, var_condition=None, class_labels=None, guidance_scale=1.0):
        """Sample from p(x_{t-1} | x_t)."""
        betas_t = self.betas[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_recip_alphas_cumprod_t = torch.sqrt(1.0 / self.alphas_cumprod[t]).reshape(-1, 1, 1, 1)
        
        # Predict noise
        predicted_noise = self.unet(x, t, var_condition, class_labels)
        
        # Apply guidance if needed
        if guidance_scale != 1.0:
            # For Auto-Guidance, we would use a worse checkpoint here
            # For now, just apply classifier-free guidance
            uncond_noise = self.unet(x, t, None, None)
            predicted_noise = uncond_noise + guidance_scale * (predicted_noise - uncond_noise)
        
        # Predict x_0
        pred_original = (x - sqrt_one_minus_alphas_cumprod_t * predicted_noise) * sqrt_recip_alphas_cumprod_t
        
        # Predict mean of x_{t-1}
        pred_epsilon_coef = (1 - self.alphas_cumprod_prev[t]) / sqrt_one_minus_alphas_cumprod_t
        pred_original_coef = self.alphas_cumprod_prev[t] * torch.sqrt(self.alphas[t])
        
        pred_mean = pred_original_coef.reshape(-1, 1, 1, 1) * pred_original + pred_epsilon_coef.reshape(-1, 1, 1, 1) * predicted_noise
        
        # Sample
        noise = torch.randn_like(x) if t[0] > 0 else torch.zeros_like(x)
        pred_var = self.posterior_variance[t].reshape(-1, 1, 1, 1)
        pred_std = torch.sqrt(pred_var)
        
        return pred_mean + pred_std * noise
    
    @torch.no_grad()
    def sample(self, var_condition, class_labels=None, num_steps=10, guidance_scale=1.0):
        """Sample from the model."""
        batch_size = var_condition.shape[0]
        device = var_condition.device
        
        # Start from noise
        x = torch.randn(batch_size, self.unet.in_channels, 
                       var_condition.shape[2], var_condition.shape[3], device=device)
        
        # Sample timesteps
        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_steps, dtype=torch.long, device=device)
        
        # Reverse diffusion process
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch, var_condition, class_labels, guidance_scale)
        
        return x


class AutoGuidanceRefiner(nn.Module):
    """Diffusion refiner with Auto-Guidance support."""
    
    def __init__(self, refiner_model, worse_refiner_model=None):
        super().__init__()
        self.refiner = refiner_model
        self.worse_refiner = worse_refiner_model
    
    def forward(self, x, var_condition=None, class_labels=None, alpha=0.5):
        """Forward pass with Auto-Guidance."""
        if self.worse_refiner is None:
            return self.refiner(x, var_condition, class_labels)
        
        # Get predictions from both models
        pred_good = self.refiner.unet(x, var_condition, class_labels)
        pred_worse = self.worse_refiner.unet(x, var_condition, class_labels)
        
        # Apply Auto-Guidance: ε_θ ← ε_θ + α(ε_θ - ε_φ)
        guided_pred = pred_good + alpha * (pred_good - pred_worse)
        
        return guided_pred
