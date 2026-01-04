"""
Sequence-Level Stability Analyzer

THEORETICAL CLAIM:
The linearization provides stable predictions even for long-horizon reasoning
due to Fisher information counteracting error accumulation.

MATHEMATICAL FOUNDATION:
Linearization error at position t:
ε_t = |f(x_{1:t}; θ) - f_lin(x_{1:t}; θ)|

Accumulated error:
E_total = Σ_{t=1}^T ε_t

HYPOTHESIS: E_total grows sublinearly with sequence length T due to:
1. KL constraints limit parameter deviation
2. Fisher metric provides stability through curvature control
3. Autoregressive structure allows error cancellation

This analyzer validates:
1. Per-token linearization error stays bounded
2. Accumulated error grows sublinearly (√T or log T instead of T)
3. Stability improves with stronger KL constraints
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import numpy as np
from torch.func import functional_call


@dataclass
class SequenceStabilityConfig:
    """Configuration for sequence-level stability analysis"""
    # Sequence lengths to test
    sequence_lengths: List[int] = None  # Default: [32, 64, 128, 256, 512]

    # Error computation
    num_samples: int = 100  # Number of sequences to evaluate
    batch_size: int = 4

    # Analysis
    compute_per_position: bool = True  # Per-position error analysis
    compute_accumulated: bool = True  # Accumulated error over sequence
    compute_growth_rate: bool = True  # Fit error growth model

    # Thresholds
    sublinear_threshold: float = 0.7  # Exponent < 1.0 for sublinear growth

    def __post_init__(self):
        if self.sequence_lengths is None:
            self.sequence_lengths = [32, 64, 128, 256, 512]


class SequenceLevelStabilityAnalyzer:
    """
    Analyzes sequence-level stability of linearized RLHF

    Validates that linearization errors accumulate sublinearly.
    """

    def __init__(
        self,
        linear_model: nn.Module,  # LinearizedLLM or ImprovedLinearizedLLM
        full_model: Optional[nn.Module] = None,  # Full model for comparison
        config: Optional[SequenceStabilityConfig] = None,
        device: str = "cuda"
    ):
        self.linear_model = linear_model
        self.full_model = full_model
        self.config = config or SequenceStabilityConfig()
        self.device = device

        # History
        self.error_history = []
        self.per_length_errors = {length: [] for length in self.config.sequence_lengths}
        self.growth_rate_history = []

    def compute_token_level_error(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute per-token linearization error

        ε_t = |logits_full(t) - logits_linear(t)|

        Args:
            input_ids: Input sequence [batch_size, seq_len]
            attention_mask: Attention mask

        Returns:
            Per-token errors and statistics
        """
        with torch.no_grad():
            # Linearized model output
            linear_output = self.linear_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            linear_logits = linear_output.logits

            # Full model output
            if self.full_model is not None:
                full_output = self.full_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                full_logits = full_output.logits
            else:
                # Use the linear model's method to compute true output
                if hasattr(self.linear_model, 'compute_linearization_error'):
                    # ImprovedLinearizedLLM has this method
                    mean_error, per_pos_errors = self.linear_model.compute_linearization_error(
                        input_ids, attention_mask
                    )
                    return {
                        'mean_error': mean_error,
                        'per_position_errors': per_pos_errors,
                    }

                # Reconstruct full model from task vector
                task_vec = self.linear_model._get_task_vector_dict()
                current_params = {
                    name: self.linear_model.params0[name] + task_vec[name]
                    for name in self.linear_model.params0.keys()
                }
                full_logits = functional_call(
                    self.linear_model.base_model,
                    current_params,
                    (input_ids,),
                    kwargs={'attention_mask': attention_mask, 'return_dict': True}
                ).logits

            # Compute absolute error per token
            token_errors = torch.abs(full_logits - linear_logits).mean(dim=-1)  # [batch, seq_len]

            # Apply mask
            if attention_mask is not None:
                token_errors = token_errors * attention_mask

        return {
            'token_errors': token_errors,  # [batch, seq_len]
            'mean_error': token_errors.mean().item(),
            'max_error': token_errors.max().item(),
            'per_position_mean': token_errors.mean(dim=0),  # [seq_len]
        }

    def compute_accumulated_error(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Compute accumulated error over sequence

        E_total = Σ_{t=1}^T ε_t

        Args:
            input_ids: Input sequence
            attention_mask: Attention mask

        Returns:
            Accumulated error metrics
        """
        errors = self.compute_token_level_error(input_ids, attention_mask)
        token_errors = errors['token_errors']  # [batch, seq_len]

        # Cumulative sum over sequence
        cumulative_errors = torch.cumsum(token_errors, dim=1)  # [batch, seq_len]

        # Mean across batch
        mean_cumulative = cumulative_errors.mean(dim=0)  # [seq_len]

        return {
            'cumulative_errors': mean_cumulative.cpu().numpy(),
            'total_error': mean_cumulative[-1].item(),
            'per_token_errors': token_errors.mean(dim=0).cpu().numpy(),
        }

    def analyze_sequence_length_scaling(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_batches: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Analyze how error scales with sequence length

        Tests hypothesis: E_total ∝ T^α where α < 1.0 (sublinear)

        Args:
            dataloader: Data loader with various sequence lengths
            num_batches: Number of batches per length

        Returns:
            Scaling analysis results
        """
        results = {
            'lengths': [],
            'mean_errors': [],
            'total_errors': [],
            'per_length_details': {},
        }

        for target_length in self.config.sequence_lengths:
            length_errors = []
            length_totals = []

            for batch_idx, batch in enumerate(dataloader):
                if num_batches and batch_idx >= num_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                # Truncate or pad to target length
                seq_len = input_ids.shape[1]
                if seq_len > target_length:
                    input_ids = input_ids[:, :target_length]
                    if attention_mask is not None:
                        attention_mask = attention_mask[:, :target_length]
                elif seq_len < target_length:
                    # Skip if too short
                    continue

                # Compute errors
                error_result = self.compute_accumulated_error(input_ids, attention_mask)

                length_errors.append(error_result['per_token_errors'].mean())
                length_totals.append(error_result['total_error'])

            if length_errors:
                mean_error = np.mean(length_errors)
                mean_total = np.mean(length_totals)

                results['lengths'].append(target_length)
                results['mean_errors'].append(mean_error)
                results['total_errors'].append(mean_total)

                results['per_length_details'][target_length] = {
                    'mean_per_token_error': mean_error,
                    'mean_total_error': mean_total,
                    'std_total_error': np.std(length_totals),
                    'num_samples': len(length_errors),
                }

        # Fit power law: E_total = a * T^α
        if len(results['lengths']) >= 3:
            lengths = np.array(results['lengths'])
            totals = np.array(results['total_errors'])

            # Log-log fit
            log_lengths = np.log(lengths)
            log_totals = np.log(totals + 1e-8)

            # Linear regression in log space
            coeffs = np.polyfit(log_lengths, log_totals, 1)
            alpha = coeffs[0]  # Exponent
            log_a = coeffs[1]  # Log of coefficient

            results['power_law'] = {
                'exponent': alpha,
                'coefficient': np.exp(log_a),
                'is_sublinear': alpha < 1.0,
                'is_logarithmic': alpha < 0.5,
                'is_linear': 0.9 <= alpha <= 1.1,
            }

            # Compute R^2
            predicted = np.exp(log_a) * (lengths ** alpha)
            ss_res = np.sum((totals - predicted) ** 2)
            ss_tot = np.sum((totals - np.mean(totals)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            results['power_law']['r_squared'] = r_squared

            # Validate hypothesis
            results['validates_hypothesis'] = (
                alpha < self.config.sublinear_threshold and r_squared > 0.8
            )

        return results

    def analyze_at_step(
        self,
        step: int,
        dataloader: torch.utils.data.DataLoader,
        num_batches: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Perform full sequence stability analysis at a training step

        Args:
            step: Training step
            dataloader: Evaluation data
            num_batches: Number of batches

        Returns:
            Complete analysis results
        """
        results = {
            'step': step,
        }

        # 1. Sequence length scaling analysis
        if self.config.compute_growth_rate:
            scaling_results = self.analyze_sequence_length_scaling(dataloader, num_batches)
            results['scaling'] = scaling_results

            # Track history
            if 'power_law' in scaling_results:
                self.growth_rate_history.append({
                    'step': step,
                    'exponent': scaling_results['power_law']['exponent'],
                    'is_sublinear': scaling_results['power_law']['is_sublinear'],
                })

        # 2. Sample some sequences for detailed analysis
        sample_errors = []
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= min(10, num_batches or 10):
                break

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            error_result = self.compute_token_level_error(input_ids, attention_mask)
            sample_errors.append(error_result['mean_error'])

        results['sample_mean_error'] = np.mean(sample_errors) if sample_errors else 0.0

        # Track history
        self.error_history.append({
            'step': step,
            'mean_error': results['sample_mean_error'],
        })

        return results

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics across training"""
        if not self.error_history:
            return {'error': 'No history tracked'}

        summary = {
            'num_steps': len(self.error_history),
            'mean_error': np.mean([h['mean_error'] for h in self.error_history]),
        }

        if self.growth_rate_history:
            exponents = [h['exponent'] for h in self.growth_rate_history]
            sublinear_count = sum(1 for h in self.growth_rate_history if h['is_sublinear'])

            summary['mean_growth_exponent'] = np.mean(exponents)
            summary['median_growth_exponent'] = np.median(exponents)
            summary['sublinear_rate'] = sublinear_count / len(self.growth_rate_history)
            summary['hypothesis_validated'] = summary['sublinear_rate'] > 0.8

        return summary

    def save_analysis(self, path: str):
        """Save analysis to disk"""
        save_dict = {
            'config': self.config,
            'error_history': self.error_history,
            'per_length_errors': self.per_length_errors,
            'growth_rate_history': self.growth_rate_history,
            'summary': self.get_summary_statistics(),
        }
        torch.save(save_dict, path)


def plot_sequence_stability(
    analyzer: SequenceLevelStabilityAnalyzer,
    scaling_results: Optional[Dict] = None,
    save_path: Optional[str] = None
):
    """Visualize sequence stability analysis"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Error vs sequence length
    if scaling_results and 'lengths' in scaling_results:
        lengths = scaling_results['lengths']
        totals = scaling_results['total_errors']

        axes[0, 0].scatter(lengths, totals, s=100, alpha=0.6, label='Measured')

        # Plot power law fit
        if 'power_law' in scaling_results:
            pl = scaling_results['power_law']
            lengths_range = np.linspace(min(lengths), max(lengths), 100)
            fitted = pl['coefficient'] * (lengths_range ** pl['exponent'])

            axes[0, 0].plot(
                lengths_range, fitted, 'r--', linewidth=2,
                label=f'Fit: E ∝ T^{pl["exponent"]:.3f}'
            )

            # Plot linear baseline
            linear_baseline = pl['coefficient'] * lengths_range
            axes[0, 0].plot(
                lengths_range, linear_baseline, 'g:', linewidth=2,
                label='Linear (T^1.0)'
            )

        axes[0, 0].set_xlabel('Sequence Length (T)')
        axes[0, 0].set_ylabel('Total Accumulated Error')
        axes[0, 0].set_title('Error Scaling with Sequence Length')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xscale('log')
        axes[0, 0].set_yscale('log')

    # Plot 2: Growth exponent over training
    if analyzer.growth_rate_history:
        steps = [h['step'] for h in analyzer.growth_rate_history]
        exponents = [h['exponent'] for h in analyzer.growth_rate_history]

        axes[0, 1].plot(steps, exponents, 'b-', linewidth=2)
        axes[0, 1].axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Linear (α=1.0)')
        axes[0, 1].axhline(y=0.5, color='g', linestyle='--', linewidth=2, label='√T (α=0.5)')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Growth Exponent (α)')
        axes[0, 1].set_title('Error Growth Rate Evolution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Mean error over training
    if analyzer.error_history:
        steps = [h['step'] for h in analyzer.error_history]
        errors = [h['mean_error'] for h in analyzer.error_history]

        axes[1, 0].plot(steps, errors, 'm-', linewidth=2)
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Mean Linearization Error')
        axes[1, 0].set_title('Error Evolution During Training')
        axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Summary
    summary = analyzer.get_summary_statistics()
    summary_text = "\n".join([
        "Sequence Stability Summary:",
        f"Mean Growth Exponent: {summary.get('mean_growth_exponent', 0):.3f}",
        f"Sublinear Rate: {summary.get('sublinear_rate', 0)*100:.1f}%",
        f"Hypothesis Validated: {summary.get('hypothesis_validated', False)}",
        f"Mean Error: {summary.get('mean_error', 0):.6f}",
    ])

    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                     family='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    axes[1, 1].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# Example usage:
"""
from src.rlhf.linearize_lm import ImprovedLinearizedLLM

# Create linearized model (after RLHF training)
linear_model = ImprovedLinearizedLLM(base_model)

# Create analyzer
config = SequenceStabilityConfig(
    sequence_lengths=[64, 128, 256, 512],
    num_samples=100,
)
analyzer = SequenceLevelStabilityAnalyzer(linear_model, config=config)

# Analyze at training step
results = analyzer.analyze_at_step(step=1000, dataloader=val_dataloader, num_batches=20)

# Check if sublinear
if results['scaling']['validates_hypothesis']:
    print(f"✓ Hypothesis validated! Growth exponent: {results['scaling']['power_law']['exponent']:.3f}")

# Visualize
plot_sequence_stability(analyzer, results['scaling'], 'sequence_stability.png')
"""
