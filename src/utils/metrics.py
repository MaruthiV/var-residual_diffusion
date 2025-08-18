"""
Utility functions for computing metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def compute_reconstruction_loss(reconstructed, target, loss_type='mse'):
    """Compute reconstruction loss between reconstructed and target images."""
    if loss_type == 'mse':
        return F.mse_loss(reconstructed, target)
    elif loss_type == 'l1':
        return F.l1_loss(reconstructed, target)
    elif loss_type == 'huber':
        return F.huber_loss(reconstructed, target)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features."""
    
    def __init__(self, feature_layers=[2, 7, 12, 21, 30]):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features
        self.feature_layers = feature_layers
        self.feature_extractors = nn.ModuleList()
        
        start = 0
        for end in feature_layers:
            self.feature_extractors.append(vgg[start:end])
            start = end
    
    def forward(self, x, y):
        """Compute perceptual loss between x and y."""
        loss = 0
        for extractor in self.feature_extractors:
            x = extractor(x)
            y = extractor(y)
            loss += F.mse_loss(x, y)
        return loss


# Global perceptual loss instance
_perceptual_loss = None


def compute_perceptual_loss(reconstructed, target):
    """Compute perceptual loss using VGG features."""
    global _perceptual_loss
    if _perceptual_loss is None:
        _perceptual_loss = PerceptualLoss()
    
    return _perceptual_loss(reconstructed, target)


def compute_fid_score(generated_images, real_images, device='cuda'):
    """Compute FID score between generated and real images."""
    try:
        from pytorch_fid import fid_score
        # This is a placeholder - you'll need to implement FID computation
        # or use a library like pytorch-fid
        return 0.0
    except ImportError:
        print("pytorch-fid not installed, skipping FID computation")
        return 0.0


def compute_inception_score(generated_images, device='cuda'):
    """Compute Inception Score for generated images."""
    try:
        from pytorch_fid import inception_score
        # This is a placeholder - you'll need to implement IS computation
        return 0.0
    except ImportError:
        print("pytorch-fid not installed, skipping Inception Score computation")
        return 0.0


def compute_lpips_distance(generated_images, real_images, device='cuda'):
    """Compute LPIPS distance between generated and real images."""
    try:
        import lpips
        loss_fn = lpips.LPIPS(net='alex').to(device)
        
        distances = []
        for gen, real in zip(generated_images, real_images):
            dist = loss_fn(gen.unsqueeze(0), real.unsqueeze(0))
            distances.append(dist.item())
        
        return torch.tensor(distances).mean().item()
    except ImportError:
        print("lpips not installed, skipping LPIPS computation")
        return 0.0


def compute_psnr(reconstructed, target):
    """Compute Peak Signal-to-Noise Ratio."""
    mse = F.mse_loss(reconstructed, target)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def compute_ssim(reconstructed, target, window_size=11):
    """Compute Structural Similarity Index."""
    # This is a simplified SSIM implementation
    # For production use, consider using a library like pytorch-msssim
    
    def gaussian_window(size, sigma=1.5):
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g
    
    def ssim_per_channel(x, y, window):
        mu1 = F.conv2d(x, window.unsqueeze(0).unsqueeze(0), padding=window_size//2)
        mu2 = F.conv2d(y, window.unsqueeze(0).unsqueeze(0), padding=window_size//2)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(x * x, window.unsqueeze(0).unsqueeze(0), padding=window_size//2) - mu1_sq
        sigma2_sq = F.conv2d(y * y, window.unsqueeze(0).unsqueeze(0), padding=window_size//2) - mu2_sq
        sigma12 = F.conv2d(x * y, window.unsqueeze(0).unsqueeze(0), padding=window_size//2) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()
    
    window = gaussian_window(window_size).to(reconstructed.device)
    
    if reconstructed.dim() == 3:
        reconstructed = reconstructed.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)
    
    ssim_values = []
    for i in range(reconstructed.shape[1]):  # For each channel
        ssim_val = ssim_per_channel(
            reconstructed[:, i:i+1], 
            target[:, i:i+1], 
            window
        )
        ssim_values.append(ssim_val)
    
    return torch.stack(ssim_values).mean()


def compute_metrics_batch(reconstructed, target, metrics=['mse', 'psnr', 'ssim']):
    """Compute multiple metrics for a batch of images."""
    results = {}
    
    for metric in metrics:
        if metric == 'mse':
            results[metric] = F.mse_loss(reconstructed, target).item()
        elif metric == 'psnr':
            results[metric] = compute_psnr(reconstructed, target).item()
        elif metric == 'ssim':
            results[metric] = compute_ssim(reconstructed, target).item()
        elif metric == 'l1':
            results[metric] = F.l1_loss(reconstructed, target).item()
        else:
            print(f"Unknown metric: {metric}")
    
    return results
