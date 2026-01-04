"""
KL-Constrained RLHF Trainer with Geometric Analysis

This module provides a training wrapper that:
1. Implements KL-constrained RL (PPO/GRPO)
2. Instruments training with geometric analyzers
3. Tracks the three mechanisms (Riemannian, Fisher, Stability)
4. Supports both verl and oat frameworks
5. Provides real-time monitoring and checkpointing

Integration with external frameworks:
- verl: https://github.com/volcengine/verl
- oat: From understand-r1-zero repo
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import json
import time

# Import our modules
from .linearize_lm import ImprovedLinearizedLLM, LinearizationConfig
from .fisher_information import FisherInformationComputer, FisherConfig
from .ntk_tracker import NTKTracker, NTKConfig
from ..analysis.riemannian_confinement import RiemannianConfinementAnalyzer, RiemannianConfinementConfig
from ..analysis.fisher_steering import FisherSteeringAnalyzer, FisherSteeringConfig
from ..analysis.sequence_stability import SequenceLevelStabilityAnalyzer, SequenceStabilityConfig


@dataclass
class GeometricAnalysisConfig:
    """Configuration for geometric analysis during training"""
    # Analysis frequency
    analyze_every_n_steps: int = 100  # Run full analysis every N steps
    quick_check_every_n_steps: int = 10  # Quick checks (KL, norms) more frequently

    # Component toggles
    track_riemannian: bool = True
    track_fisher_steering: bool = True
    track_sequence_stability: bool = True
    track_ntk: bool = True
    compute_fisher: bool = True

    # Fisher computation
    fisher_num_batches: int = 50  # Batches for Fisher computation
    fisher_update_interval: int = 500  # Recompute Fisher every N steps

    # Checkpointing
    save_analysis_checkpoints: bool = True
    checkpoint_dir: str = "./checkpoints/analysis"

    # Visualization
    generate_plots: bool = True
    plot_dir: str = "./plots"


@dataclass
class RLHFTrainingConfig:
    """Configuration for RLHF training"""
    # Model
    model_name: str = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    use_linearized: bool = False  # Train in tangent space

    # KL constraint
    kl_coeff: float = 0.01  # KL penalty coefficient
    kl_target: float = 0.01  # Target KL divergence
    adaptive_kl: bool = True  # Adaptive KL coefficient

    # RL algorithm
    algorithm: str = "ppo"  # "ppo" or "grpo"
    ppo_epochs: int = 4
    ppo_clip_range: float = 0.2

    # Training
    num_steps: int = 10000
    batch_size: int = 8
    mini_batch_size: int = 4
    learning_rate: float = 1e-5
    gradient_accumulation_steps: int = 1

    # Generation
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.9

    # Logging
    log_every_n_steps: int = 10
    eval_every_n_steps: int = 100
    save_every_n_steps: int = 500

    # Distributed training
    use_deepspeed: bool = False
    num_gpus: int = 8


class GeometricRLHFTrainer:
    """
    RLHF Trainer with integrated geometric analysis

    This trainer wraps existing RLHF implementations (verl, oat) and adds
    comprehensive geometric instrumentation to validate our hypothesis.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        reward_model: nn.Module,
        tokenizer: Any,
        train_config: RLHFTrainingConfig,
        analysis_config: Optional[GeometricAnalysisConfig] = None,
        device: str = "cuda",
    ):
        self.model = model
        self.ref_model = ref_model  # Reference model (θ_0)
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.train_config = train_config
        self.analysis_config = analysis_config or GeometricAnalysisConfig()
        self.device = device

        # Freeze reference model
        for p in self.ref_model.parameters():
            p.requires_grad = False
        self.ref_model.eval()

        # Create linearized version if requested
        self.linear_model = None
        if train_config.use_linearized:
            linear_config = LinearizationConfig(
                use_cache=True,
                low_rank_approx=False,
            )
            self.linear_model = ImprovedLinearizedLLM(model, linear_config)

        # Initialize geometric analyzers
        self._init_analyzers()

        # Training state
        self.global_step = 0
        self.epoch = 0

        # Metrics history
        self.metrics_history = []

    def _init_analyzers(self):
        """Initialize all geometric analyzers"""
        # Fisher computer
        if self.analysis_config.compute_fisher:
            fisher_config = FisherConfig(
                num_samples=1000,
                block_diagonal=True,
                compute_eigenvectors=True,
                num_top_eigenvectors=100,
            )
            self.fisher_computer = FisherInformationComputer(
                self.ref_model,  # Compute Fisher at initialization
                fisher_config,
                self.device
            )
        else:
            self.fisher_computer = None

        # NTK tracker
        if self.analysis_config.track_ntk:
            ntk_config = NTKConfig(
                num_samples=100,
                diagonal_only=True,  # Use diagonal for efficiency
                lazy_threshold=0.1,
            )
            self.ntk_tracker = NTKTracker(
                self.model,
                ntk_config,
                self.device
            )
        else:
            self.ntk_tracker = None

        # Riemannian confinement analyzer
        if self.analysis_config.track_riemannian:
            riem_config = RiemannianConfinementConfig(
                kl_epsilon=self.train_config.kl_target,
                track_trajectory=False,  # Save memory
            )
            self.riem_analyzer = RiemannianConfinementAnalyzer(
                self.model,
                self.ref_model,
                self.fisher_computer,
                riem_config,
                self.device
            )
        else:
            self.riem_analyzer = None

        # Fisher steering analyzer
        if self.analysis_config.track_fisher_steering and self.fisher_computer:
            steer_config = FisherSteeringConfig(
                num_principal_components=100,
                steering_score_threshold=1.0,
            )
            self.fisher_steering = FisherSteeringAnalyzer(
                self.fisher_computer,
                steer_config,
                self.device
            )
        else:
            self.fisher_steering = None

        # Sequence stability analyzer
        if self.analysis_config.track_sequence_stability:
            stab_config = SequenceStabilityConfig(
                sequence_lengths=[64, 128, 256, 512],
                num_samples=100,
            )

            # Use linearized model if available
            if self.linear_model:
                self.stability_analyzer = SequenceLevelStabilityAnalyzer(
                    self.linear_model,
                    None,  # Will compute from task vector
                    stab_config,
                    self.device
                )
            else:
                self.stability_analyzer = None
        else:
            self.stability_analyzer = None

    def initialize_analyzers(self, dataloader):
        """
        Initialize analyzers that need data (Fisher, NTK)

        Should be called before training starts.
        """
        print("Initializing geometric analyzers...")

        # Compute initial Fisher
        if self.fisher_computer:
            print("Computing Fisher Information Matrix...")
            self.fisher_computer.compute_empirical_fisher(
                dataloader,
                num_batches=self.analysis_config.fisher_num_batches
            )
            self.fisher_computer.compute_eigendecomposition()
            print(f"  Fisher computed. Effective ranks: {self.fisher_computer.compute_effective_rank()}")

        # Initialize NTK
        if self.ntk_tracker:
            print("Computing initial NTK...")
            self.ntk_tracker.initialize_ntk(dataloader, num_batches=10)
            print(f"  NTK initialized.")

        print("Analyzers ready!")

    def compute_kl_penalty(
        self,
        logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute KL divergence penalty and adaptive coefficient

        KL = E[log π_θ(y|x) - log π_0(y|x)]

        Returns:
            (kl_penalty, metrics)
        """
        # Per-token KL
        kl_per_token = logprobs - ref_logprobs

        # Mask padding
        if masks is not None:
            kl_per_token = kl_per_token * masks

        # Mean KL
        if masks is not None:
            kl_div = (kl_per_token * masks).sum() / masks.sum()
        else:
            kl_div = kl_per_token.mean()

        # Adaptive KL coefficient
        if self.train_config.adaptive_kl:
            # If KL > target, increase coefficient
            # If KL < target, decrease coefficient
            kl_error = kl_div.item() - self.train_config.kl_target
            self.train_config.kl_coeff *= (1.0 + 0.1 * np.sign(kl_error))
            self.train_config.kl_coeff = np.clip(self.train_config.kl_coeff, 0.001, 0.1)

        penalty = self.train_config.kl_coeff * kl_div

        metrics = {
            'kl_divergence': kl_div.item(),
            'kl_penalty': penalty.item(),
            'kl_coeff': self.train_config.kl_coeff,
        }

        return penalty, metrics

    def run_geometric_analysis(
        self,
        step: int,
        dataloader: torch.utils.data.DataLoader,
        is_full_analysis: bool = False,
    ) -> Dict[str, Any]:
        """
        Run geometric analysis at current step

        Args:
            step: Current training step
            dataloader: Validation data for analysis
            is_full_analysis: Whether to run full analysis (expensive)

        Returns:
            Analysis results
        """
        results = {'step': step}

        # 1. Riemannian confinement (always quick to compute)
        if self.riem_analyzer:
            riem_results = self.riem_analyzer.analyze_at_step(
                step, dataloader, num_batches=10
            )
            results['riemannian'] = {
                'kl_divergence': riem_results['kl_divergence'],
                'l2_norm': riem_results['l2_norm'],
                'fisher_norm': riem_results['fisher_norm'],
                'bound_satisfied': riem_results['bound_validation']['is_bound_satisfied']
                    if riem_results['bound_validation'] else None,
            }

        # 2. NTK tracking
        if self.ntk_tracker and is_full_analysis:
            ntk_metrics = self.ntk_tracker.update_ntk(dataloader, step, num_batches=10)
            results['ntk'] = {
                'relative_change': ntk_metrics['relative_change'],
                'is_lazy': ntk_metrics['is_lazy'],
            }

        # 3. Fisher steering (requires parameter update)
        if self.fisher_steering and is_full_analysis:
            # Compute parameter update
            update_vector = {}
            for (name, param), (_, ref_param) in zip(
                self.model.named_parameters(),
                self.ref_model.named_parameters()
            ):
                update_vector[name] = param - ref_param

            steer_results = self.fisher_steering.full_analysis(update_vector, step)
            results['fisher_steering'] = {
                'steering_score': steer_results['decomposition']['global']['steering_score'],
                'validates_hypothesis': steer_results['overall_validates_hypothesis'],
            }

        # 4. Sequence stability
        if self.stability_analyzer and is_full_analysis:
            stab_results = self.stability_analyzer.analyze_at_step(
                step, dataloader, num_batches=10
            )
            results['sequence_stability'] = {
                'mean_error': stab_results['sample_mean_error'],
                'growth_exponent': stab_results['scaling']['power_law']['exponent']
                    if 'scaling' in stab_results and 'power_law' in stab_results['scaling'] else None,
            }

        # 5. Update Fisher periodically
        if (self.fisher_computer and is_full_analysis and
            step % self.analysis_config.fisher_update_interval == 0):
            print(f"  Recomputing Fisher at step {step}...")
            self.fisher_computer.compute_empirical_fisher(
                dataloader,
                num_batches=self.analysis_config.fisher_num_batches
            )

        return results

    def save_checkpoint(self, step: int, save_dir: str):
        """Save training checkpoint with geometric analysis state"""
        save_path = Path(save_dir) / f"checkpoint_step_{step}"
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model
        torch.save(self.model.state_dict(), save_path / "model.pt")

        # Save analyzers
        if self.riem_analyzer:
            self.riem_analyzer.save_analysis(save_path / "riemannian_analysis.pt")

        if self.fisher_steering:
            self.fisher_steering.save_analysis(save_path / "fisher_steering.pt")

        if self.stability_analyzer:
            self.stability_analyzer.save_analysis(save_path / "sequence_stability.pt")

        if self.ntk_tracker:
            self.ntk_tracker.save_tracker(save_path / "ntk_tracker.pt")

        if self.fisher_computer:
            self.fisher_computer.save_fisher(save_path / "fisher_info.pt")

        # Save metrics history
        with open(save_path / "metrics_history.json", 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

        print(f"Checkpoint saved to {save_path}")

    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of geometric analysis across training"""
        summary = {'training_steps': self.global_step}

        if self.riem_analyzer:
            summary['riemannian'] = self.riem_analyzer.get_summary_statistics()

        if self.fisher_steering:
            summary['fisher_steering'] = self.fisher_steering.get_summary_statistics()

        if self.stability_analyzer:
            summary['sequence_stability'] = self.stability_analyzer.get_summary_statistics()

        if self.ntk_tracker:
            summary['ntk'] = self.ntk_tracker.analyze_lazy_regime()

        # Overall validation
        hypothesis_checks = []
        if self.riem_analyzer:
            hypothesis_checks.append(
                summary['riemannian'].get('trust_region_compliance_rate', 0) > 0.8
            )
        if self.fisher_steering:
            hypothesis_checks.append(
                summary['fisher_steering'].get('hypothesis_validated', False)
            )
        if self.stability_analyzer:
            hypothesis_checks.append(
                summary['sequence_stability'].get('hypothesis_validated', False)
            )
        if self.ntk_tracker:
            hypothesis_checks.append(
                summary['ntk'].get('is_lazy_regime', False)
            )

        summary['overall_hypothesis_validated'] = (
            sum(hypothesis_checks) / len(hypothesis_checks) > 0.66  # 2/3 majority
        ) if hypothesis_checks else False

        return summary


# Helper function to create trainer from config
def create_geometric_trainer(
    model_name: str,
    train_config: RLHFTrainingConfig,
    analysis_config: Optional[GeometricAnalysisConfig] = None,
    device: str = "cuda",
):
    """
    Create a GeometricRLHFTrainer with all components initialized

    Args:
        model_name: HuggingFace model identifier
        train_config: Training configuration
        analysis_config: Analysis configuration
        device: Device for training

    Returns:
        Initialized trainer
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load models
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load reward model (placeholder - should be provided)
    reward_model = None  # TODO: Load actual reward model

    # Create trainer
    trainer = GeometricRLHFTrainer(
        model=model,
        ref_model=ref_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        train_config=train_config,
        analysis_config=analysis_config,
        device=device,
    )

    return trainer


# Example usage:
"""
# Create configurations
train_config = RLHFTrainingConfig(
    model_name="Qwen/Qwen2.5-Math-1.5B-Instruct",
    use_linearized=False,  # Start with full model
    kl_target=0.01,
    num_steps=10000,
    num_gpus=8,
)

analysis_config = GeometricAnalysisConfig(
    analyze_every_n_steps=100,
    track_riemannian=True,
    track_fisher_steering=True,
    track_sequence_stability=True,
    track_ntk=True,
)

# Create trainer
trainer = create_geometric_trainer(
    "Qwen/Qwen2.5-Math-1.5B-Instruct",
    train_config,
    analysis_config,
)

# Initialize analyzers
trainer.initialize_analyzers(train_dataloader)

# Training loop (integrate with verl/oat)
for step in range(train_config.num_steps):
    # ... RL training step (PPO/GRPO) ...

    # Periodic geometric analysis
    if step % analysis_config.analyze_every_n_steps == 0:
        results = trainer.run_geometric_analysis(step, val_dataloader, is_full_analysis=True)
        print(f"Step {step} Analysis:")
        print(f"  KL divergence: {results['riemannian']['kl_divergence']:.6f}")
        print(f"  Steering score: {results.get('fisher_steering', {}).get('steering_score', 0):.3f}")

    # Save checkpoints
    if step % train_config.save_every_n_steps == 0:
        trainer.save_checkpoint(step, "./checkpoints")

# Final summary
summary = trainer.get_training_summary()
print(f"\nHypothesis validated: {summary['overall_hypothesis_validated']}")
"""
