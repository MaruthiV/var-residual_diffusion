"""
Sampling script for VAR-Refine pipeline.
Generates images using VAR scaffold + diffusion refiner.
"""

import os
import argparse
import yaml
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
import time
from tqdm import tqdm

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hierarchical_vqvae import HierarchicalVQVAE
from models.var_transformer import VARTransformer
from models.diffusion_refiner import DiffusionRefiner, AutoGuidanceRefiner
from utils.metrics import compute_fid_score, compute_inception_score, compute_lpips_distance


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_models(config, device):
    """Load all trained models."""
    models = {}
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer_checkpoint = torch.load(config['models']['tokenizer_path'], map_location=device)
    tokenizer_config = tokenizer_checkpoint['config']
    
    tokenizer = HierarchicalVQVAE(
        in_channels=3,
        hidden_channels=tokenizer_config['model']['hidden_dim'],
        num_res_blocks=tokenizer_config['model']['num_res_blocks'],
        pyramid_levels=tokenizer_config['model']['pyramid_levels'],
        codebook_size=tokenizer_config['model']['codebook_size'],
        code_dim=tokenizer_config['model']['code_dim'],
        commitment_cost=tokenizer_config['model']['commitment_cost'],
        decay=tokenizer_config['model']['decay']
    )
    tokenizer.load_state_dict(tokenizer_checkpoint['model_state_dict'])
    tokenizer.eval()
    models['tokenizer'] = tokenizer
    
    # Load VAR transformer
    print("Loading VAR transformer...")
    var_checkpoint = torch.load(config['models']['var_path'], map_location=device)
    var_config = var_checkpoint['config']
    
    var_model = VARTransformer(
        codebook_size=var_config['model']['codebook_size'],
        code_dim=var_config['model']['code_dim'],
        hidden_dim=var_config['model']['hidden_dim'],
        num_layers=var_config['model']['num_layers'],
        num_heads=var_config['model']['num_heads'],
        mlp_ratio=var_config['model']['mlp_ratio'],
        dropout=var_config['model']['dropout'],
        max_seq_len=var_config['model']['max_seq_len'],
        use_rotary=var_config['model']['use_rotary'],
        use_alibi=var_config['model']['use_alibi'],
        pyramid_levels=var_config['model']['pyramid_levels']
    )
    var_model.load_state_dict(var_checkpoint['model_state_dict'])
    var_model.eval()
    models['var'] = var_model
    
    # Load diffusion refiner
    print("Loading diffusion refiner...")
    refiner_checkpoint = torch.load(config['models']['refiner_path'], map_location=device)
    refiner_config = refiner_checkpoint['config']
    
    refiner = DiffusionRefiner(
        in_channels=refiner_config['model']['in_channels'],
        out_channels=refiner_config['model']['out_channels'],
        model_channels=refiner_config['model']['model_channels'],
        num_res_blocks=refiner_config['model']['num_res_blocks'],
        attention_resolutions=refiner_config['model']['attention_resolutions'],
        dropout=refiner_config['model']['dropout'],
        channel_mult=refiner_config['model']['channel_mult'],
        num_heads=refiner_config['model']['num_heads'],
        condition_on_var=refiner_config['model']['condition_on_var'],
        condition_on_class=refiner_config['model']['condition_on_class'],
        num_classes=refiner_config['model']['num_classes'],
        noise_schedule=refiner_config['model']['noise_schedule'],
        num_timesteps=refiner_config['model']['num_timesteps'],
        beta_start=refiner_config['model']['beta_start'],
        beta_end=refiner_config['model']['beta_end']
    )
    refiner.load_state_dict(refiner_checkpoint['model_state_dict'])
    refiner.eval()
    models['refiner'] = refiner
    
    # Load worse refiner for Auto-Guidance if available
    if config['sampling']['refiner']['auto_guidance']['use_worse_checkpoint']:
        try:
            worse_refiner_checkpoint = torch.load(config['models']['worse_refiner_path'], map_location=device)
            worse_refiner = DiffusionRefiner(
                in_channels=refiner_config['model']['in_channels'],
                out_channels=refiner_config['model']['out_channels'],
                model_channels=refiner_config['model']['model_channels'],
                num_res_blocks=refiner_config['model']['num_res_blocks'],
                attention_resolutions=refiner_config['model']['attention_resolutions'],
                dropout=refiner_config['model']['dropout'],
                channel_mult=refiner_config['model']['channel_mult'],
                num_heads=refiner_config['model']['num_heads'],
                condition_on_var=refiner_config['model']['condition_on_var'],
                condition_on_class=refiner_config['model']['condition_on_class'],
                num_classes=refiner_config['model']['num_classes'],
                noise_schedule=refiner_config['model']['noise_schedule'],
                num_timesteps=refiner_config['model']['num_timesteps'],
                beta_start=refiner_config['model']['beta_start'],
                beta_end=refiner_config['model']['beta_end']
            )
            worse_refiner.load_state_dict(worse_refiner_checkpoint['model_state_dict'])
            worse_refiner.eval()
            models['worse_refiner'] = worse_refiner
            print("Loaded worse refiner for Auto-Guidance")
        except:
            print("Warning: Could not load worse refiner, Auto-Guidance will be disabled")
            models['worse_refiner'] = None
    
    return models


def generate_var_scaffold(var_model, tokenizer, batch_size, device, config):
    """Generate VAR scaffold from noise."""
    print("Generating VAR scaffold...")
    
    # Start from Gaussian noise at coarsest scale
    coarsest_size = tokenizer.pyramid_levels[0]
    pyramid_indices = []
    
    # Generate random indices for coarsest scale
    coarsest_indices = torch.randint(
        0, tokenizer.codebook_size, 
        (batch_size, coarsest_size, coarsest_size), 
        device=device
    )
    pyramid_indices.append(coarsest_indices)
    
    # Generate next scales autoregressively
    for scale in range(1, len(tokenizer.pyramid_levels)):
        print(f"Generating scale {scale} ({tokenizer.pyramid_levels[scale]}x{tokenizer.pyramid_levels[scale]})")
        
        # Generate next scale
        pyramid_indices = var_model.generate_next_scale(
            pyramid_indices,
            temperature=config['sampling']['var']['temperature'],
            top_k=config['sampling']['var']['top_k'],
            top_p=config['sampling']['var']['top_p']
        )
    
    # Decode to image
    with torch.no_grad():
        # Convert indices to quantized features
        quantized_features = []
        for i, indices in enumerate(pyramid_indices):
            # Get codebook embeddings
            embeddings = tokenizer.quantizers[i].embedding[indices]
            quantized_features.append(embeddings)
        
        # Decode
        var_image = tokenizer.decode(quantized_features)
    
    return var_image, pyramid_indices


def refine_with_diffusion(refiner, var_image, class_labels, device, config):
    """Refine VAR output with diffusion."""
    print("Refining with diffusion...")
    
    # Apply Auto-Guidance if enabled
    if (config['sampling']['refiner']['auto_guidance']['enabled'] and 
        hasattr(refiner, 'worse_refiner') and refiner.worse_refiner is not None):
        
        # Use Auto-Guidance refiner
        auto_guidance_refiner = AutoGuidanceRefiner(refiner, refiner.worse_refiner)
        
        with torch.no_grad():
            refined_image = auto_guidance_refiner.sample(
                var_image,
                class_labels=class_labels,
                num_steps=config['sampling']['refiner']['num_steps'],
                guidance_scale=config['sampling']['refiner']['guidance_scale']
            )
    else:
        # Use standard refiner
        with torch.no_grad():
            refined_image = refiner.sample(
                var_image,
                class_labels=class_labels,
                num_steps=config['sampling']['refiner']['num_steps'],
                guidance_scale=config['sampling']['refiner']['guidance_scale']
            )
    
    return refined_image


def save_images(images, save_dir, prefix="sample"):
    """Save generated images."""
    os.makedirs(save_dir, exist_ok=True)
    
    for i, image in enumerate(images):
        # Convert from [-1, 1] to [0, 255]
        image = ((image + 1) / 2 * 255).clamp(0, 255).byte()
        image = image.permute(1, 2, 0).cpu().numpy()
        
        # Save as PIL image
        pil_image = Image.fromarray(image)
        save_path = os.path.join(save_dir, f"{prefix}_{i:04d}.png")
        pil_image.save(save_path)
    
    print(f"Saved {len(images)} images to {save_dir}")


def compute_evaluation_metrics(generated_images, real_images, device, config):
    """Compute evaluation metrics."""
    metrics = {}
    
    if config['evaluation']['compute_fid']:
        print("Computing FID score...")
        metrics['fid'] = compute_fid_score(generated_images, real_images, device)
    
    if config['evaluation']['compute_is']:
        print("Computing Inception Score...")
        metrics['inception_score'] = compute_inception_score(generated_images, device)
    
    if config['evaluation']['compute_lpips']:
        print("Computing LPIPS distance...")
        metrics['lpips'] = compute_lpips_distance(generated_images, real_images, device)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Generate images with VAR-Refine')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for generation')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save images')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.num_samples is not None:
        config['sampling']['num_samples'] = args.num_samples
    if args.batch_size is not None:
        config['sampling']['batch_size'] = args.batch_size
    if args.save_dir is not None:
        config['output']['save_dir'] = args.save_dir
    
    # Set random seed
    torch.manual_seed(config['sampling']['seed'])
    np.random.seed(config['sampling']['seed'])
    
    # Setup device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    models = load_models(config, device)
    
    # Move models to device
    for model_name, model in models.items():
        if model is not None:
            models[model_name] = model.to(device)
    
    # Generate images
    num_samples = config['sampling']['num_samples']
    batch_size = config['sampling']['batch_size']
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    all_generated_images = []
    all_var_images = []
    
    start_time = time.time()
    
    for batch_idx in tqdm(range(num_batches), desc="Generating images"):
        # Determine batch size for this iteration
        current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
        
        # Generate random class labels
        class_labels = torch.randint(0, 10, (current_batch_size,), device=device)  # CIFAR-10
        
        # Generate VAR scaffold
        var_image, pyramid_indices = generate_var_scaffold(
            models['var'], models['tokenizer'], current_batch_size, device, config
        )
        
        # Refine with diffusion
        refined_image = refine_with_diffusion(
            models['refiner'], var_image, class_labels, device, config
        )
        
        # Store results
        all_generated_images.append(refined_image.cpu())
        all_var_images.append(var_image.cpu())
    
    # Concatenate all batches
    generated_images = torch.cat(all_generated_images, dim=0)
    var_images = torch.cat(all_var_images, dim=0)
    
    total_time = time.time() - start_time
    print(f"Generated {num_samples} images in {total_time:.2f} seconds")
    print(f"Average time per image: {total_time/num_samples:.3f} seconds")
    
    # Save images
    save_dir = config['output']['save_dir']
    save_images(generated_images, save_dir, "refined")
    save_images(var_images, os.path.join(save_dir, "var_only"), "var")
    
    # Save metadata
    if config['output']['save_metadata']:
        metadata = {
            'num_samples': num_samples,
            'batch_size': batch_size,
            'total_time': total_time,
            'time_per_image': total_time / num_samples,
            'config': config
        }
        
        import json
        with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    # Compute evaluation metrics if requested
    if any([config['evaluation']['compute_fid'], 
            config['evaluation']['compute_is'], 
            config['evaluation']['compute_lpips']]):
        
        # Load reference dataset
        if config['evaluation']['reference_dataset'] == 'cifar10':
            from torchvision import datasets
            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            reference_dataset = datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transform
            )
            
            # Sample reference images
            reference_indices = torch.randperm(len(reference_dataset))[:num_samples]
            reference_images = []
            for idx in reference_indices:
                reference_images.append(reference_dataset[idx][0])
            reference_images = torch.stack(reference_images)
        
        # Compute metrics
        metrics = compute_evaluation_metrics(
            generated_images, reference_images, device, config
        )
        
        print("Evaluation metrics:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        
        # Save metrics
        if config['output']['save_metadata']:
            metadata['evaluation_metrics'] = metrics
            with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
    
    print("Generation completed!")


if __name__ == '__main__':
    main()
