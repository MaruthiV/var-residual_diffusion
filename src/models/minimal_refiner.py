"""
Minimal Diffusion Refiner for testing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MinimalDiffusionRefiner(nn.Module):
    """Minimal diffusion refiner for testing."""
    
    def __init__(self, in_channels=3, out_channels=3, model_channels=128, 
                 num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        
        # Simple encoder-decoder
        self.encoder = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        self.decoder = nn.Conv2d(model_channels, out_channels, 3, padding=1)
        
        # VAR condition projection
        self.var_proj = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Noise schedule
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def forward(self, x, timesteps, var_condition=None, class_labels=None):
        """Forward pass."""
        # Simple forward pass
        h = self.encoder(x)
        
        # Add VAR condition if provided
        if var_condition is not None:
            # Project VAR condition to same channel dimension
            var_cond = self.var_proj(var_condition)
            # Ensure same spatial size
            if len(var_cond.shape) == 4 and len(h.shape) == 4:
                if var_cond.shape[2:] != h.shape[2:]:
                    var_cond = F.interpolate(var_cond, size=h.shape[2:], mode='bilinear', align_corners=False)
                h = h + var_cond
        
        output = self.decoder(h)
        return output
    
    def p_losses(self, x_start, var_condition=None, class_labels=None, noise=None):
        """Compute loss for training."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device).long()
        x_noisy = self.q_sample(x_start, t, noise=noise)
        
        predicted_noise = self.forward(x_noisy, t, var_condition, class_labels)
        
        # Ensure shapes match
        if predicted_noise.shape != noise.shape:
            predicted_noise = predicted_noise.view_as(noise)
        
        loss = F.mse_loss(predicted_noise, noise, reduction='mean')
        
        # Ensure loss is a scalar
        if loss.numel() > 1:
            loss = loss.mean()
        
        return loss
    
    def q_sample(self, x_start, t, noise=None):
        """Sample from q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    @torch.no_grad()
    def sample(self, var_condition, class_labels=None, num_steps=10, guidance_scale=1.0):
        """Sample from the model."""
        batch_size = var_condition.shape[0]
        device = var_condition.device
        
        # Start from noise
        x = torch.randn(batch_size, self.in_channels, 
                       var_condition.shape[2], var_condition.shape[3], device=device)
        
        # Sample timesteps
        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_steps, dtype=torch.long, device=device)
        
        # Simple reverse process
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.forward(x, t_batch, var_condition, class_labels)
            
            # Simple denoising step
            alpha_t = self.alphas[t]
            beta_t = self.betas[t]
            
            # Predict x_0
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            
            # Add noise for next step
            if t > 0:
                noise = torch.randn_like(x)
                x = pred_x0 + torch.sqrt(beta_t) * noise
            else:
                x = pred_x0
        
        return x
