"""
Training script for VAR transformer.
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import wandb
from tqdm import tqdm
import numpy as np

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.var_transformer import VARTransformer
from models.hierarchical_vqvae import HierarchicalVQVAE


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_tokenizer(tokenizer_path, device):
    """Load pre-trained tokenizer."""
    checkpoint = torch.load(tokenizer_path, map_location=device)
    config = checkpoint['config']
    
    tokenizer = HierarchicalVQVAE(
        in_channels=3,
        hidden_channels=config['model']['hidden_dim'],
        num_res_blocks=config['model']['num_res_blocks'],
        pyramid_levels=config['model']['pyramid_levels'],
        codebook_size=config['model']['codebook_size'],
        code_dim=config['model']['code_dim'],
        commitment_cost=config['model']['commitment_cost'],
        decay=config['model']['decay']
    )
    tokenizer.load_state_dict(checkpoint['model_state_dict'])
    tokenizer.eval()
    return tokenizer


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
    """Setup VAR transformer model."""
    model = VARTransformer(
        codebook_size=config['model']['codebook_size'],
        code_dim=config['model']['code_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        mlp_ratio=config['model']['mlp_ratio'],
        dropout=config['model']['dropout'],
        max_seq_len=config['model']['max_seq_len'],
        use_rotary=config['model']['use_rotary'],
        use_alibi=config['model']['use_alibi'],
        pyramid_levels=config['model']['pyramid_levels']
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


def get_scheduled_sampling_ratio(config, step):
    """Get scheduled sampling ratio."""
    if not config['training']['scheduled_sampling']['enabled']:
        return 1.0
    
    start_ratio = config['training']['scheduled_sampling']['start_ratio']
    end_ratio = config['training']['scheduled_sampling']['end_ratio']
    decay_steps = config['training']['scheduled_sampling']['decay_steps']
    
    if step >= decay_steps:
        return end_ratio
    
    ratio = start_ratio + (end_ratio - start_ratio) * (step / decay_steps)
    return ratio


def train_epoch(model, tokenizer, train_loader, optimizer, config, device, epoch, global_step):
    """Train for one epoch."""
    model.train()
    tokenizer.eval()
    
    total_loss = 0
    total_accuracy = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (data, _) in enumerate(pbar):
        data = data.to(device)
        batch_size = data.shape[0]
        
        # Get codebook indices from tokenizer
        with torch.no_grad():
            pyramid_indices = tokenizer.get_codebook_indices(data)
        
        # Train on each scale (except the first one)
        for scale in range(1, len(config['model']['pyramid_levels'])):
            # Get target indices for this scale
            target_indices = pyramid_indices[scale]
            
            # Get input indices (all previous scales)
            input_indices = pyramid_indices[:scale]
            
            # Forward pass
            logits = model(input_indices, target_scale=scale)
            
            # Reshape for loss computation
            logits = logits.view(batch_size, -1, config['model']['codebook_size'])
            target_indices = target_indices.view(batch_size, -1)
            
            # Compute loss
            loss = F.cross_entropy(logits.view(-1, config['model']['codebook_size']), 
                                 target_indices.view(-1))
            
            # Compute accuracy
            pred_indices = torch.argmax(logits, dim=-1)
            accuracy = (pred_indices == target_indices).float().mean()
            
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            num_batches += 1
        
        # Average loss over scales
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        # Backward pass
        optimizer.zero_grad()
        avg_loss.backward()
        
        if config['training']['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
        
        optimizer.step()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'Acc': f'{avg_accuracy:.4f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
        
        # Log to wandb
        if batch_idx % config['training']['log_every'] == 0:
            wandb.log({
                'train/loss': avg_loss,
                'train/accuracy': avg_accuracy,
                'train/learning_rate': optimizer.param_groups[0]['lr'],
                'epoch': epoch,
                'batch': batch_idx,
                'global_step': global_step
            })
        
        global_step += 1
    
    return {
        'loss': total_loss / num_batches,
        'accuracy': total_accuracy / num_batches
    }, global_step


def validate(model, tokenizer, val_loader, config, device, epoch):
    """Validate model."""
    model.eval()
    tokenizer.eval()
    
    total_loss = 0
    total_accuracy = 0
    num_batches = 0
    
    with torch.no_grad():
        for data, _ in tqdm(val_loader, desc='Validation'):
            data = data.to(device)
            batch_size = data.shape[0]
            
            # Get codebook indices from tokenizer
            pyramid_indices = tokenizer.get_codebook_indices(data)
            
            # Validate on each scale (except the first one)
            for scale in range(1, len(config['model']['pyramid_levels'])):
                # Get target indices for this scale
                target_indices = pyramid_indices[scale]
                
                # Get input indices (all previous scales)
                input_indices = pyramid_indices[:scale]
                
                # Forward pass
                logits = model(input_indices, target_scale=scale)
                
                # Reshape for loss computation
                logits = logits.view(batch_size, -1, config['model']['codebook_size'])
                target_indices = target_indices.view(batch_size, -1)
                
                # Compute loss
                loss = F.cross_entropy(logits.view(-1, config['model']['codebook_size']), 
                                     target_indices.view(-1))
                
                # Compute accuracy
                pred_indices = torch.argmax(logits, dim=-1)
                accuracy = (pred_indices == target_indices).float().mean()
                
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                num_batches += 1
    
    metrics = {
        'val/loss': total_loss / num_batches,
        'val/accuracy': total_accuracy / num_batches,
        'epoch': epoch
    }
    
    wandb.log(metrics)
    return metrics


def save_checkpoint(model, optimizer, epoch, config, metrics, global_step):
    """Save model checkpoint."""
    save_dir = config['output']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'metrics': metrics
    }
    
    checkpoint_path = os.path.join(save_dir, f"{config['output']['model_name']}.pt")
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description='Train VAR transformer')
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
            name=f"var_{config['training']['dataset']}",
            config=config
        )
    
    # Load tokenizer
    tokenizer = load_tokenizer(config['data']['tokenizer_path'], device)
    print("Loaded tokenizer")
    
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
    global_step = 0
    
    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        
        # Train
        train_metrics, global_step = train_epoch(
            model, tokenizer, train_loader, optimizer, config, device, epoch, global_step
        )
        
        # Validate
        val_metrics = validate(model, tokenizer, val_loader, config, device, epoch)
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Save checkpoint
        if epoch % config['training']['save_every'] == 0:
            save_checkpoint(model, optimizer, epoch, config, val_metrics, global_step)
        
        # Save best model
        if val_metrics['val/loss'] < best_val_loss:
            best_val_loss = val_metrics['val/loss']
            save_checkpoint(model, optimizer, epoch, config, val_metrics, global_step)
            print(f"New best validation loss: {best_val_loss:.4f}")
    
    print("Training completed!")
    
    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
