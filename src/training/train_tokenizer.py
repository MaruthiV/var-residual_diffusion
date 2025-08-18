"""
Training script for hierarchical VQ-VAE tokenizer.
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import wandb
from tqdm import tqdm
import numpy as np

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hierarchical_vqvae import HierarchicalVQVAE
from utils.metrics import compute_reconstruction_loss, compute_perceptual_loss


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_dataset(config):
    """Setup dataset and dataloader."""
    if config['training']['dataset'] == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize(config['data']['image_size']),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        train_dataset = datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        val_dataset = datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {config['training']['dataset']}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=True, 
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=False, 
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    return train_loader, val_loader


def setup_model(config):
    """Setup model."""
    model = HierarchicalVQVAE(
        in_channels=3,
        hidden_channels=config['model']['hidden_dim'],
        num_res_blocks=config['model']['num_res_blocks'],
        pyramid_levels=config['model']['pyramid_levels'],
        codebook_size=config['model']['codebook_size'],
        code_dim=config['model']['code_dim'],
        commitment_cost=config['model']['commitment_cost'],
        decay=config['model']['decay']
    )
    return model


def setup_optimizer(model, config):
    """Setup optimizer and scheduler."""
    if config['training']['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    elif config['training']['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['training']['optimizer']}")
    
    if config['training']['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['training']['num_epochs']
        )
    else:
        scheduler = None
    
    return optimizer, scheduler


def train_epoch(model, train_loader, optimizer, config, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_commit_loss = 0
    total_perceptual_loss = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (data, _) in enumerate(pbar):
        data = data.to(device)
        
        # Forward pass
        reconstructed, indices, commit_loss = model(data)
        
        # Compute losses
        recon_loss = compute_reconstruction_loss(reconstructed, data)
        perceptual_loss = compute_perceptual_loss(reconstructed, data)
        
        # Total loss
        loss = (config['training']['recon_loss_weight'] * recon_loss + 
                config['training']['commitment_loss_weight'] * commit_loss +
                config['training']['perceptual_loss_weight'] * perceptual_loss)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        if config['training']['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
        
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_commit_loss += commit_loss.item()
        total_perceptual_loss += perceptual_loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Recon': f'{recon_loss.item():.4f}',
            'Commit': f'{commit_loss.item():.4f}'
        })
        
        # Log to wandb
        if batch_idx % config['training']['log_every'] == 0:
            wandb.log({
                'train/loss': loss.item(),
                'train/recon_loss': recon_loss.item(),
                'train/commit_loss': commit_loss.item(),
                'train/perceptual_loss': perceptual_loss.item(),
                'train/learning_rate': optimizer.param_groups[0]['lr'],
                'epoch': epoch,
                'batch': batch_idx
            })
    
    return {
        'loss': total_loss / len(train_loader),
        'recon_loss': total_recon_loss / len(train_loader),
        'commit_loss': total_commit_loss / len(train_loader),
        'perceptual_loss': total_perceptual_loss / len(train_loader)
    }


def validate(model, val_loader, config, device, epoch):
    """Validate model."""
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_commit_loss = 0
    
    with torch.no_grad():
        for data, _ in tqdm(val_loader, desc='Validation'):
            data = data.to(device)
            
            # Forward pass
            reconstructed, indices, commit_loss = model(data)
            
            # Compute losses
            recon_loss = compute_reconstruction_loss(reconstructed, data)
            
            # Total loss
            loss = (config['training']['recon_loss_weight'] * recon_loss + 
                    config['training']['commitment_loss_weight'] * commit_loss)
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_commit_loss += commit_loss.item()
    
    metrics = {
        'val/loss': total_loss / len(val_loader),
        'val/recon_loss': total_recon_loss / len(val_loader),
        'val/commit_loss': total_commit_loss / len(val_loader),
        'epoch': epoch
    }
    
    wandb.log(metrics)
    return metrics


def save_checkpoint(model, optimizer, epoch, config, metrics):
    """Save model checkpoint."""
    save_dir = config['output']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'metrics': metrics
    }
    
    checkpoint_path = os.path.join(save_dir, f"{config['output']['model_name']}.pt")
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description='Train hierarchical VQ-VAE tokenizer')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--wandb', action='store_true', help='Use wandb logging')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Setup device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup wandb
    if args.wandb:
        wandb.init(
            project="var-refine",
            name=f"tokenizer_{config['training']['dataset']}",
            config=config
        )
    
    # Setup dataset
    train_loader, val_loader = setup_dataset(config)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Setup model
    model = setup_model(config)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer
    optimizer, scheduler = setup_optimizer(model, config)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, config, device, epoch)
        
        # Validate
        val_metrics = validate(model, val_loader, config, device, epoch)
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Save checkpoint
        if epoch % config['training']['save_every'] == 0:
            save_checkpoint(model, optimizer, epoch, config, val_metrics)
        
        # Save best model
        if val_metrics['val/loss'] < best_val_loss:
            best_val_loss = val_metrics['val/loss']
            save_checkpoint(model, optimizer, epoch, config, val_metrics)
            print(f"New best validation loss: {best_val_loss:.4f}")
    
    print("Training completed!")
    
    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
