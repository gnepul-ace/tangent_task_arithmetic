#!/usr/bin/env python3
"""
Main Training Script for RLHF in Tangent Space

This script implements the full RLHF training pipeline with geometric analysis.
It integrates with verl/oat frameworks and tracks the three geometric mechanisms.

Usage:
    # Train with default config
    python experiments/train_rlhf_geometric.py

    # Train with specific config
    python experiments/train_rlhf_geometric.py --config experiments/config/math_linear.yaml

    # Train with config overrides
    python experiments/train_rlhf_geometric.py --config math_linear.yaml \
        --training.num_steps 5000 --training.learning_rate 1e-5

    # Distributed training (8 GPUs)
    torchrun --nproc_per_node=8 experiments/train_rlhf_geometric.py \
        --config math_linear.yaml
"""

import os
import sys
from pathlib import Path
import argparse
import yaml
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import wandb
from tqdm import tqdm
from typing import Dict, Any, Optional
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rlhf.kl_constrained_trainer import (
    GeometricRLHFTrainer,
    RLHFTrainingConfig,
    GeometricAnalysisConfig,
)
from src.rlhf.linearize_lm import ImprovedLinearizedLLM, LinearizationConfig


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Handle inheritance (__base__)
    if '__base__' in config:
        base_path = Path(config_path).parent / config['__base__']
        base_config = load_config(str(base_path))
        # Merge configs (current overrides base)
        def deep_merge(base, override):
            merged = base.copy()
            for key, value in override.items():
                if key == '__base__':
                    continue
                if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                    merged[key] = deep_merge(merged[key], value)
                else:
                    merged[key] = value
            return merged
        config = deep_merge(base_config, config)

    return config


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')

        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def load_dataset_for_training(config: Dict[str, Any], split: str = "train"):
    """
    Load dataset based on configuration

    Supports: MATH, GSM8K, code generation
    """
    dataset_name = config['dataset']['name']

    if dataset_name == "math":
        # Load MATH dataset
        dataset = load_dataset("lighteval/MATH", split=split)

        # Filter by difficulty if specified
        if '[' in config['dataset']['train_split']:
            # Parse difficulty levels (e.g., "train[3:]" -> levels 3-5)
            pass  # Already handled by load_dataset split syntax

    elif dataset_name == "gsm8k":
        dataset = load_dataset("gsm8k", "main", split=split)

    elif dataset_name == "code":
        # Use code contest dataset or similar
        dataset = load_dataset("deepmind/code_contests", split=split)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Limit samples if specified
    if config['dataset'].get('max_samples'):
        dataset = dataset.select(range(min(len(dataset), config['dataset']['max_samples'])))

    return dataset


def create_dataloader(dataset, tokenizer, config: Dict[str, Any], shuffle: bool = True):
    """Create DataLoader from dataset"""

    def collate_fn(batch):
        """Collate batch with tokenization"""
        # Extract prompts from batch
        prompts = [item['problem'] if 'problem' in item else item['question'] for item in batch]

        # Tokenize
        tokenized = tokenizer(
            prompts,
            max_length=config['dataset']['max_length'],
            padding=config['dataset']['padding'],
            truncation=config['dataset']['truncation'],
            return_tensors='pt'
        )

        return tokenized

    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    return dataloader


def train_step_ppo(
    model,
    ref_model,
    reward_model,
    batch,
    optimizer,
    config: RLHFTrainingConfig,
):
    """
    Single PPO training step

    This is a simplified version - in practice, use verl or oat
    for full PPO implementation with value function, etc.
    """
    input_ids = batch['input_ids'].cuda()
    attention_mask = batch['attention_mask'].cuda()

    # Generate rollouts
    with torch.no_grad():
        # Sample from current policy
        generated = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=True,
            pad_token_id=model.config.pad_token_id,
        )

        # Compute log probs from current and reference policies
        current_outputs = model(generated, attention_mask=attention_mask)
        ref_outputs = ref_model(generated, attention_mask=attention_mask)

        current_logprobs = torch.log_softmax(current_outputs.logits, dim=-1)
        ref_logprobs = torch.log_softmax(ref_outputs.logits, dim=-1)

        # Get rewards (simplified - use actual reward model)
        rewards = torch.ones(generated.shape[0]).cuda()  # Placeholder

    # PPO update (simplified)
    # In practice, implement full PPO with advantage estimation, value function, etc.
    # Or use verl/oat which handle this

    optimizer.zero_grad()

    # Forward pass
    outputs = model(generated, attention_mask=attention_mask)
    logprobs = torch.log_softmax(outputs.logits, dim=-1)

    # KL penalty
    kl_div = (logprobs - ref_logprobs).mean()
    kl_penalty = config.kl_coeff * kl_div

    # Loss (simplified - add advantage, value loss, etc.)
    loss = -rewards.mean() + kl_penalty

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
    optimizer.step()

    metrics = {
        'loss': loss.item(),
        'kl_div': kl_div.item(),
        'reward': rewards.mean().item(),
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train RLHF with Geometric Analysis")
    parser.add_argument('--config', type=str, default='experiments/config/base_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (overrides config)')
    parser.add_argument('--use_verl', action='store_true',
                        help='Use verl framework for RLHF')
    parser.add_argument('--use_oat', action='store_true',
                        help='Use oat framework for RLHF')

    args, unknown = parser.parse_known_args()

    # Load configuration
    config = load_config(args.config)

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    is_main_process = rank == 0

    if is_main_process:
        print(f"Starting RLHF training with geometric analysis")
        print(f"Config: {args.config}")
        print(f"World size: {world_size}")

    # Initialize wandb
    if is_main_process and config['wandb']['enabled']:
        wandb.init(
            project=config['wandb']['project'],
            entity=config['wandb']['entity'],
            name=config['wandb']['run_name'],
            config=config,
            tags=config['wandb']['tags'],
        )

    # Load model and tokenizer
    if is_main_process:
        print(f"\nLoading model: {config['model']['name']}")

    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        torch_dtype=getattr(torch, config['model']['dtype']),
        device_map={'': local_rank} if world_size > 1 else 'auto',
    )

    ref_model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        torch_dtype=getattr(torch, config['model']['dtype']),
        device_map={'': local_rank} if world_size > 1 else 'auto',
    )

    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Freeze reference model
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    # Create linearized model if requested
    linear_model = None
    if config['model']['use_linearized']:
        if is_main_process:
            print("\n🔄 Creating linearized model (Tangent Space training)")

        linear_config = LinearizationConfig(
            use_cache=True,
            low_rank_approx=False,
        )
        linear_model = ImprovedLinearizedLLM(model, linear_config)

        # Use linear model for training
        training_model = linear_model
    else:
        training_model = model

    # Load datasets
    if is_main_process:
        print(f"\nLoading dataset: {config['dataset']['name']}")

    train_dataset = load_dataset_for_training(config, split=config['dataset']['train_split'])
    val_dataset = load_dataset_for_training(config, split=config['dataset']['eval_split'])

    train_dataloader = create_dataloader(train_dataset, tokenizer, config, shuffle=True)
    val_dataloader = create_dataloader(val_dataset, tokenizer, config, shuffle=False)

    if is_main_process:
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")

    # Create training configs
    train_config = RLHFTrainingConfig(
        model_name=config['model']['name'],
        use_linearized=config['model']['use_linearized'],
        algorithm=config['training']['algorithm'],
        kl_coeff=config['training']['kl_coeff'],
        kl_target=config['training']['kl_target'],
        adaptive_kl=config['training']['adaptive_kl'],
        num_steps=config['training']['num_steps'],
        batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'],
        num_gpus=world_size,
    )

    analysis_config = GeometricAnalysisConfig(
        analyze_every_n_steps=config['geometric_analysis']['analyze_every_n_steps'],
        track_riemannian=config['geometric_analysis']['track_riemannian'],
        track_fisher_steering=config['geometric_analysis']['track_fisher_steering'],
        track_sequence_stability=config['geometric_analysis']['track_sequence_stability'],
        track_ntk=config['geometric_analysis']['track_ntk'],
        compute_fisher=config['geometric_analysis']['compute_fisher'],
    )

    # Create geometric trainer
    trainer = GeometricRLHFTrainer(
        model=training_model,
        ref_model=ref_model,
        reward_model=None,  # TODO: Load reward model
        tokenizer=tokenizer,
        train_config=train_config,
        analysis_config=analysis_config,
        device=f'cuda:{local_rank}',
    )

    # Initialize analyzers
    if is_main_process:
        print("\n📊 Initializing geometric analyzers...")

    trainer.initialize_analyzers(train_dataloader)

    if is_main_process:
        print("\n🚀 Starting training...\n")

    # Training loop
    optimizer = torch.optim.AdamW(
        training_model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=config['training']['weight_decay'],
    )

    global_step = 0
    for epoch in range(100):  # Large number, will break based on steps
        pbar = tqdm(train_dataloader, disable=not is_main_process, desc=f"Epoch {epoch}")

        for batch in pbar:
            if global_step >= train_config.num_steps:
                break

            # Training step (use verl/oat in practice)
            metrics = train_step_ppo(
                training_model,
                ref_model,
                None,  # reward model
                batch,
                optimizer,
                train_config,
            )

            # Logging
            if global_step % config['training']['log_every_n_steps'] == 0:
                if is_main_process:
                    pbar.set_postfix(metrics)
                    if config['wandb']['enabled']:
                        wandb.log(metrics, step=global_step)

            # Geometric analysis
            if global_step % analysis_config.analyze_every_n_steps == 0 and global_step > 0:
                if is_main_process:
                    print(f"\n📊 Running geometric analysis at step {global_step}...")

                    results = trainer.run_geometric_analysis(
                        global_step,
                        val_dataloader,
                        is_full_analysis=True,
                    )

                    # Log results
                    print(f"  KL divergence: {results.get('riemannian', {}).get('kl_divergence', 0):.6f}")
                    print(f"  Fisher norm: {results.get('riemannian', {}).get('fisher_norm', 0):.6f}")
                    if 'fisher_steering' in results:
                        print(f"  Steering score: {results['fisher_steering']['steering_score']:.3f}")
                    if 'ntk' in results:
                        print(f"  NTK lazy: {results['ntk']['is_lazy']}")

                    if config['wandb']['enabled']:
                        wandb.log({'geometric_analysis': results}, step=global_step)

            # Save checkpoint
            if global_step % config['training']['save_every_n_steps'] == 0 and global_step > 0:
                if is_main_process:
                    output_dir = args.output_dir or config['training']['output_dir']
                    trainer.save_checkpoint(global_step, output_dir)

            global_step += 1

        if global_step >= train_config.num_steps:
            break

    # Final summary
    if is_main_process:
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)

        summary = trainer.get_training_summary()
        print("\n📈 Geometric Analysis Summary:")
        print(json.dumps(summary, indent=2))

        print(f"\n✅ Hypothesis Validated: {summary['overall_hypothesis_validated']}")

        if config['wandb']['enabled']:
            wandb.log({'final_summary': summary})
            wandb.finish()

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
