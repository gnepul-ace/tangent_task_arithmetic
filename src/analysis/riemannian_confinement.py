"""
Riemannian Confinement Analyzer

THEORETICAL CLAIM:
The KL trust region D_KL(π_θ || π_0) ≤ ε acts as a strict geometric bound
on parameter displacement in the Riemannian metric defined by the Fisher Information Matrix.

MATHEMATICAL FOUNDATION:
Second-order Taylor expansion of KL divergence:
D_KL(π_θ || π_0) ≈ 1/2 * (θ - θ_0)^T F (θ - θ_0)

Therefore:
||θ - θ_0||_F^2 ≤ 2ε

Where ||·||_F is the Fisher norm: ||v||_F^2 = v^T F v

This analyzer validates:
1. KL divergence correlates with Fisher norm displacement
2. Trust region violations lead to large parameter changes
3. Riemannian geometry confines RLHF to a small region of parameter space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class RiemannianConfinementConfig:
    """Configuration for Riemannian confinement analysis"""
    # KL computation
    num_samples_kl: int = 1000  # Samples for KL divergence estimation
    batch_size: int = 8
    seq_length: int = 128

    # Fisher norm computation
    use_fisher: bool = True  # Use Fisher norm vs. L2 norm

    # Tracking
    track_per_layer: bool = True  # Track displacement per layer
    track_trajectory: bool = True  # Track parameter trajectory over time

    # Theoretical bounds
    kl_epsilon: float = 0.01  # Trust region size
    bound_tolerance: float = 1.2  # Allow up to 20% violation


class RiemannianConfinementAnalyzer:
    """
    Analyzes the Riemannian confinement mechanism in RLHF

    Validates that KL trust region bounds parameter displacement.
    """

    def __init__(
        self,
        model: nn.Module,
        initial_model: nn.Module,
        fisher_computer: Optional[Any] = None,  # FisherInformationComputer
        config: Optional[RiemannianConfinementConfig] = None,
        device: str = "cuda"
    ):
        """
        Args:
            model: Current model (θ_t)
            initial_model: Initial model (θ_0)
            fisher_computer: Computed Fisher information for metric
            config: Configuration
            device: Device for computation
        """
        self.model = model
        self.initial_model = initial_model
        self.fisher_computer = fisher_computer
        self.config = config or RiemannianConfinementConfig()
        self.device = device

        # Freeze initial model
        for p in self.initial_model.parameters():
            p.requires_grad = False
        self.initial_model.eval()

        # History tracking
        self.kl_history = []
        self.displacement_history = []
        self.fisher_norm_history = []
        self.l2_norm_history = []
        self.trajectory = []  # [(step, params_snapshot)]

    def compute_kl_divergence(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_batches: Optional[int] = None,
    ) -> float:
        """
        Compute KL divergence: D_KL(π_θ || π_0)

        D_KL = E_{x,y~π_θ}[log π_θ(y|x) - log π_0(y|x)]

        Args:
            dataloader: Data for KL estimation
            num_batches: Number of batches to use

        Returns:
            KL divergence (scalar)
        """
        self.model.eval()
        self.initial_model.eval()

        kl_sum = 0.0
        num_tokens = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if num_batches and batch_idx >= num_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                # Get logits from both models
                current_output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                current_logits = current_output.logits

                initial_output = self.initial_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                initial_logits = initial_output.logits

                # Compute KL divergence per token
                # KL(p||q) = sum_i p_i * (log p_i - log q_i)
                current_log_probs = F.log_softmax(current_logits, dim=-1)
                initial_log_probs = F.log_softmax(initial_logits, dim=-1)
                current_probs = F.softmax(current_logits, dim=-1)

                kl_per_token = (current_probs * (current_log_probs - initial_log_probs)).sum(dim=-1)

                # Mask padding tokens
                if attention_mask is not None:
                    kl_per_token = kl_per_token * attention_mask
                    num_tokens += attention_mask.sum().item()
                else:
                    num_tokens += kl_per_token.numel()

                kl_sum += kl_per_token.sum().item()

        kl_divergence = kl_sum / num_tokens if num_tokens > 0 else 0.0

        return kl_divergence

    def compute_parameter_displacement(
        self,
        use_fisher_norm: bool = True,
    ) -> Dict[str, float]:
        """
        Compute parameter displacement from initial model

        Returns both L2 norm and Fisher norm:
        - L2: ||θ - θ_0||_2^2 = Σ (θ - θ_0)^2
        - Fisher: ||θ - θ_0||_F^2 = (θ - θ_0)^T F (θ - θ_0)

        Returns:
            Dictionary with displacement metrics
        """
        # Compute parameter difference
        param_diff = {}
        l2_norm_sq = 0.0

        for (name, param), (_, init_param) in zip(
            self.model.named_parameters(),
            self.initial_model.named_parameters()
        ):
            diff = param - init_param
            param_diff[name] = diff.detach()
            l2_norm_sq += (diff ** 2).sum().item()

        l2_norm = np.sqrt(l2_norm_sq)

        # Compute Fisher norm if available
        fisher_norm = None
        fisher_norm_sq = None

        if use_fisher_norm and self.fisher_computer is not None:
            fisher_norm_sq = self.fisher_computer.compute_fisher_norm(param_diff)
            fisher_norm = np.sqrt(fisher_norm_sq)

        # Per-layer norms
        layer_norms = {
            name: torch.norm(diff).item()
            for name, diff in param_diff.items()
        }

        return {
            'l2_norm': l2_norm,
            'l2_norm_squared': l2_norm_sq,
            'fisher_norm': fisher_norm,
            'fisher_norm_squared': fisher_norm_sq,
            'layer_norms': layer_norms,
            'param_diff': param_diff,
        }

    def validate_riemannian_bound(
        self,
        kl_divergence: float,
        fisher_norm_squared: float,
    ) -> Dict[str, Any]:
        """
        Validate the theoretical bound: ||θ - θ_0||_F^2 ≤ 2 * D_KL

        Args:
            kl_divergence: Measured KL divergence
            fisher_norm_squared: Measured Fisher norm squared

        Returns:
            Validation results
        """
        theoretical_bound = 2 * kl_divergence
        is_satisfied = fisher_norm_squared <= theoretical_bound * self.config.bound_tolerance

        relative_displacement = fisher_norm_squared / (theoretical_bound + 1e-8)

        return {
            'kl_divergence': kl_divergence,
            'fisher_norm_squared': fisher_norm_squared,
            'theoretical_bound': theoretical_bound,
            'is_bound_satisfied': is_satisfied,
            'relative_displacement': relative_displacement,
            'bound_violation': max(0, fisher_norm_squared - theoretical_bound),
        }

    def analyze_at_step(
        self,
        step: int,
        dataloader: torch.utils.data.DataLoader,
        num_batches: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Perform full Riemannian confinement analysis at a training step

        Args:
            step: Current training step
            dataloader: Data for KL estimation
            num_batches: Number of batches

        Returns:
            Complete analysis results
        """
        # 1. Compute KL divergence
        kl_div = self.compute_kl_divergence(dataloader, num_batches)

        # 2. Compute parameter displacement
        displacement = self.compute_parameter_displacement(
            use_fisher_norm=self.config.use_fisher
        )

        # 3. Validate theoretical bound
        if displacement['fisher_norm_squared'] is not None:
            bound_validation = self.validate_riemannian_bound(
                kl_div,
                displacement['fisher_norm_squared']
            )
        else:
            bound_validation = None

        # 4. Track history
        self.kl_history.append({'step': step, 'kl': kl_div})
        self.l2_norm_history.append({'step': step, 'l2_norm': displacement['l2_norm']})

        if displacement['fisher_norm'] is not None:
            self.fisher_norm_history.append({
                'step': step,
                'fisher_norm': displacement['fisher_norm']
            })

        # 5. Store trajectory if enabled
        if self.config.track_trajectory:
            param_snapshot = {
                name: param.detach().cpu().clone()
                for name, param in self.model.named_parameters()
            }
            self.trajectory.append((step, param_snapshot))

        results = {
            'step': step,
            'kl_divergence': kl_div,
            'l2_norm': displacement['l2_norm'],
            'fisher_norm': displacement['fisher_norm'],
            'bound_validation': bound_validation,
        }

        return results

    def compute_confinement_score(self) -> float:
        """
        Compute overall confinement score

        Score = correlation between KL and Fisher norm
        High score (close to 1) validates the confinement mechanism

        Returns:
            Correlation coefficient
        """
        if len(self.kl_history) < 2:
            return 0.0

        kl_values = [h['kl'] for h in self.kl_history]
        fisher_values = [h['fisher_norm'] for h in self.fisher_norm_history]

        if len(fisher_values) == 0:
            return 0.0

        # Compute correlation
        correlation = np.corrcoef(kl_values[:len(fisher_values)], fisher_values)[0, 1]

        return correlation

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics across all tracked steps

        Returns:
            Summary metrics
        """
        if not self.kl_history:
            return {'error': 'No history tracked yet'}

        kl_values = [h['kl'] for h in self.kl_history]
        l2_values = [h['l2_norm'] for h in self.l2_norm_history]

        summary = {
            'num_steps': len(self.kl_history),
            'mean_kl': np.mean(kl_values),
            'max_kl': np.max(kl_values),
            'mean_l2_displacement': np.mean(l2_values),
            'max_l2_displacement': np.max(l2_values),
        }

        if self.fisher_norm_history:
            fisher_values = [h['fisher_norm'] for h in self.fisher_norm_history]
            summary['mean_fisher_displacement'] = np.mean(fisher_values)
            summary['max_fisher_displacement'] = np.max(fisher_values)
            summary['confinement_correlation'] = self.compute_confinement_score()

        # Check if KL stayed within trust region
        kl_violations = sum(1 for kl in kl_values if kl > self.config.kl_epsilon)
        summary['trust_region_violations'] = kl_violations
        summary['trust_region_compliance_rate'] = 1 - (kl_violations / len(kl_values))

        return summary

    def save_analysis(self, path: str):
        """Save analysis results to disk"""
        save_dict = {
            'config': self.config,
            'kl_history': self.kl_history,
            'l2_norm_history': self.l2_norm_history,
            'fisher_norm_history': self.fisher_norm_history,
            'trajectory': self.trajectory if self.config.track_trajectory else None,
            'summary': self.get_summary_statistics(),
        }
        torch.save(save_dict, path)

    def load_analysis(self, path: str):
        """Load analysis results from disk"""
        save_dict = torch.load(path)
        self.kl_history = save_dict['kl_history']
        self.l2_norm_history = save_dict['l2_norm_history']
        self.fisher_norm_history = save_dict['fisher_norm_history']
        if save_dict['trajectory'] is not None:
            self.trajectory = save_dict['trajectory']


def plot_riemannian_confinement(
    analyzer: RiemannianConfinementAnalyzer,
    save_path: Optional[str] = None
):
    """
    Visualize Riemannian confinement analysis

    Plots:
    1. KL divergence over time
    2. Fisher norm vs. L2 norm over time
    3. KL vs. Fisher norm scatter (validates bound)

    Args:
        analyzer: Analyzer with history
        save_path: Path to save plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: KL divergence over time
    steps = [h['step'] for h in analyzer.kl_history]
    kl_values = [h['kl'] for h in analyzer.kl_history]

    axes[0, 0].plot(steps, kl_values, 'b-', linewidth=2, label='KL Divergence')
    axes[0, 0].axhline(
        y=analyzer.config.kl_epsilon,
        color='r',
        linestyle='--',
        linewidth=2,
        label=f'Trust Region (ε={analyzer.config.kl_epsilon})'
    )
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('D_KL(π_θ || π_0)')
    axes[0, 0].set_title('KL Divergence Evolution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Parameter displacement norms
    l2_values = [h['l2_norm'] for h in analyzer.l2_norm_history]
    axes[0, 1].plot(steps, l2_values, 'g-', linewidth=2, label='L2 Norm')

    if analyzer.fisher_norm_history:
        fisher_values = [h['fisher_norm'] for h in analyzer.fisher_norm_history]
        axes[0, 1].plot(steps, fisher_values, 'r-', linewidth=2, label='Fisher Norm')

    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('||θ - θ_0||')
    axes[0, 1].set_title('Parameter Displacement')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: KL vs. Fisher norm (validates theoretical bound)
    if analyzer.fisher_norm_history:
        fisher_values = [h['fisher_norm'] for h in analyzer.fisher_norm_history]
        fisher_sq = [f**2 for f in fisher_values]
        kl_for_fisher = kl_values[:len(fisher_sq)]

        axes[1, 0].scatter(kl_for_fisher, fisher_sq, alpha=0.6, s=50)

        # Plot theoretical bound: ||θ||_F^2 = 2 * KL
        kl_range = np.linspace(0, max(kl_for_fisher), 100)
        bound = 2 * kl_range

        axes[1, 0].plot(kl_range, bound, 'r--', linewidth=2, label='Bound: ||θ||²_F ≤ 2·D_KL')
        axes[1, 0].set_xlabel('KL Divergence')
        axes[1, 0].set_ylabel('Fisher Norm Squared')
        axes[1, 0].set_title('Riemannian Bound Validation')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Summary statistics
    summary = analyzer.get_summary_statistics()
    summary_text = "\n".join([
        "Summary Statistics:",
        f"Mean KL: {summary.get('mean_kl', 0):.6f}",
        f"Max KL: {summary.get('max_kl', 0):.6f}",
        f"Mean L2 Displacement: {summary.get('mean_l2_displacement', 0):.6f}",
        f"Mean Fisher Displacement: {summary.get('mean_fisher_displacement', 0):.6f}",
        f"Confinement Correlation: {summary.get('confinement_correlation', 0):.4f}",
        f"Trust Region Compliance: {summary.get('trust_region_compliance_rate', 0)*100:.1f}%",
    ])

    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                     family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Analysis Summary')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


# Example usage:
"""
from transformers import AutoModelForCausalLM

# Load models
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
initial_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

# Create analyzer
from src.rlhf.fisher_information import FisherInformationComputer
fisher_computer = FisherInformationComputer(model)
fisher_computer.compute_empirical_fisher(train_dataloader, num_batches=100)

config = RiemannianConfinementConfig(kl_epsilon=0.01)
analyzer = RiemannianConfinementAnalyzer(model, initial_model, fisher_computer, config)

# During training
for step in range(0, 1000, 100):
    # ... training code ...

    results = analyzer.analyze_at_step(step, val_dataloader, num_batches=10)
    print(f"Step {step}:")
    print(f"  KL divergence: {results['kl_divergence']:.6f}")
    print(f"  Fisher norm: {results['fisher_norm']:.6f}")
    if results['bound_validation']:
        print(f"  Bound satisfied: {results['bound_validation']['is_bound_satisfied']}")

# Visualize results
plot_riemannian_confinement(analyzer, 'riemannian_confinement.png')

# Get summary
summary = analyzer.get_summary_statistics()
print(f"Confinement correlation: {summary['confinement_correlation']:.4f}")
"""
