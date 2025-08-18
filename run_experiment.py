#!/usr/bin/env python3
"""
Main experiment runner for VAR-Refine.
Orchestrates the complete training and evaluation pipeline.
"""

import os
import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    print(f"Completed in {end_time - start_time:.2f} seconds")
    print(f"Return code: {result.returncode}")
    
    if result.stdout:
        print("STDOUT:")
        print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    if result.returncode != 0:
        print(f"ERROR: Command failed with return code {result.returncode}")
        sys.exit(1)
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Run VAR-Refine experiment')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['cifar10', 'imagenet64'], help='Dataset to use')
    parser.add_argument('--stage', type=str, default='all',
                       choices=['tokenizer', 'var', 'refiner', 'sample', 'all'],
                       help='Which stage to run')
    parser.add_argument('--wandb', action='store_true', help='Use wandb logging')
    parser.add_argument('--num_samples', type=int, default=1000, 
                       help='Number of samples to generate')
    parser.add_argument('--skip_training', action='store_true', 
                       help='Skip training and go directly to sampling')
    args = parser.parse_args()
    
    # Set up paths
    config_dir = Path("configs")
    experiment_dir = Path("experiments")
    
    # Determine config file based on dataset
    if args.dataset == 'cifar10':
        tokenizer_config = config_dir / "tokenizer_cifar10.yaml"
        var_config = config_dir / "var_cifar10.yaml"
        refiner_config = config_dir / "refiner_cifar10.yaml"
        sampling_config = config_dir / "sampling_cifar10.yaml"
    elif args.dataset == 'imagenet64':
        # For ImageNet-64, you would need to create these configs
        print("ImageNet-64 configs not yet implemented")
        sys.exit(1)
    else:
        print(f"Unknown dataset: {args.dataset}")
        sys.exit(1)
    
    # Check if configs exist
    for config_file in [tokenizer_config, var_config, refiner_config, sampling_config]:
        if not config_file.exists():
            print(f"Config file not found: {config_file}")
            sys.exit(1)
    
    # Stage 0: Train tokenizer
    if args.stage in ['tokenizer', 'all'] and not args.skip_training:
        print("\n" + "="*80)
        print("STAGE 0: Training Hierarchical VQ-VAE Tokenizer")
        print("="*80)
        
        cmd = ["python", "src/training/train_tokenizer.py", 
               "--config", str(tokenizer_config)]
        if args.wandb:
            cmd.append("--wandb")
        
        run_command(cmd, "Training hierarchical VQ-VAE tokenizer")
    
    # Stage 1: Train VAR transformer
    if args.stage in ['var', 'all'] and not args.skip_training:
        print("\n" + "="*80)
        print("STAGE 1: Training VAR Transformer")
        print("="*80)
        
        cmd = ["python", "src/training/train_var.py", 
               "--config", str(var_config)]
        if args.wandb:
            cmd.append("--wandb")
        
        run_command(cmd, "Training VAR transformer")
    
    # Stage 2: Train diffusion refiner
    if args.stage in ['refiner', 'all'] and not args.skip_training:
        print("\n" + "="*80)
        print("STAGE 2: Training Diffusion Refiner")
        print("="*80)
        
        # Note: You'll need to implement the refiner training script
        print("Refiner training script not yet implemented")
        print("You can implement it following the pattern of the other training scripts")
    
    # Stage 3: Generate samples
    if args.stage in ['sample', 'all']:
        print("\n" + "="*80)
        print("STAGE 3: Generating Samples")
        print("="*80)
        
        cmd = ["python", "src/sampling/generate.py", 
               "--config", str(sampling_config),
               "--num_samples", str(args.num_samples)]
        
        run_command(cmd, "Generating samples with VAR-Refine")
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    # Print summary
    print(f"\nExperiment Summary:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Stages completed: {args.stage}")
    print(f"  Samples generated: {args.num_samples}")
    
    if args.stage in ['sample', 'all']:
        print(f"\nGenerated samples saved to:")
        print(f"  Refined images: experiments/samples_{args.dataset}/")
        print(f"  VAR-only images: experiments/samples_{args.dataset}/var_only/")
        print(f"  Metadata: experiments/samples_{args.dataset}/metadata.json")


if __name__ == '__main__':
    main()
