"""
Neural Tangent Kernel (NTK) Tracker for Language Models

The NTK characterizes the training dynamics in the lazy training regime.
During RLHF with KL constraints, we hypothesize that:

1. The NTK stays approximately constant (frozen)
2. This validates that training occurs in the tangent space
3. Parameter updates remain small relative to initialization

NTK Definition:
K(x, x') = ∇_θ f(x; θ)^T ∇_θ f(x'; θ)

In lazy regime: K_t ≈ K_0 (NTK doesn't change during training)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from torch.func import functional_call, jacrev


@dataclass
class NTKConfig:
    """Configuration for NTK tracking"""
    # Sampling
    num_samples: int = 100  # Number of data points for NTK estimation
    batch_size: int = 4  # Batch size for NTK computation

    # Approximation
    use_sampling: bool = True  # Use sampling-based NTK (cheaper)
    diagonal_only: bool = False  # Only compute diagonal of NTK
    random_projection: bool = False  # Use random projection for efficiency
    projection_dim: int = 1000  # Dimension for random projection

    # Tracking
    track_spectrum: bool = True  # Track eigenvalue spectrum
    track_trace: bool = True  # Track trace of NTK
    track_frobenius_norm: bool = True  # Track Frobenius norm

    # Comparison
    lazy_threshold: float = 0.1  # Threshold for considering NTK "frozen"
    # If ||K_t - K_0||_F / ||K_0||_F < threshold, we're in lazy regime


class NTKTracker:
    """
    Tracks the Neural Tangent Kernel during training

    This validates the core hypothesis: RLHF operates in the lazy training regime
    where the NTK stays approximately constant.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[NTKConfig] = None,
        device: str = "cuda"
    ):
        self.model = model
        self.config = config or NTKConfig()
        self.device = device

        # Storage
        self.ntk_diagonal_initial = None  # K_0 diagonal
        self.ntk_diagonal_current = None  # K_t diagonal
        self.ntk_trace_history = []  # Trace over training
        self.ntk_norm_history = []  # Frobenius norm over training
        self.eigenvalue_history = []  # Top eigenvalues over training

        # Lazy regime indicators
        self.is_lazy = []  # Boolean indicators over time
        self.relative_change = []  # ||K_t - K_0|| / ||K_0|| over time

    def compute_ntk_diagonal(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_batches: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute diagonal of NTK matrix: K_ii = ||∇_θ f(x_i)||^2

        This is much cheaper than full NTK and still informative about
        training dynamics.

        Args:
            dataloader: Data loader for input samples
            num_batches: Number of batches to use

        Returns:
            NTK diagonal [num_samples]
        """
        self.model.eval()
        ntk_diagonal = []

        for batch_idx, batch in enumerate(dataloader):
            if num_batches and batch_idx >= num_batches:
                break

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            batch_size = input_ids.shape[0]

            for i in range(batch_size):
                # Single input
                single_input = input_ids[i:i+1]
                single_mask = attention_mask[i:i+1] if attention_mask is not None else None

                # Compute gradient norm squared
                self.model.zero_grad()

                outputs = self.model(
                    input_ids=single_input,
                    attention_mask=single_mask,
                    return_dict=True
                )

                # Use mean of logits as scalar output for gradient
                logits = outputs.logits
                scalar_output = logits.mean()

                scalar_output.backward()

                # Compute ||∇f||^2
                grad_norm_sq = sum(
                    (param.grad ** 2).sum().item()
                    for param in self.model.parameters()
                    if param.grad is not None
                )

                ntk_diagonal.append(grad_norm_sq)

        return torch.tensor(ntk_diagonal)

    def compute_ntk_matrix_sampled(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_samples: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute sampled NTK matrix: K_ij = ∇_θ f(x_i)^T ∇_θ f(x_j)

        This computes a subset of the full NTK matrix using sampled data points.

        Args:
            dataloader: Data loader
            num_samples: Number of samples to use (default: config.num_samples)

        Returns:
            NTK matrix [num_samples, num_samples]
        """
        num_samples = num_samples or self.config.num_samples

        # Collect samples
        samples = []
        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            for i in range(input_ids.shape[0]):
                samples.append({
                    'input_ids': input_ids[i:i+1],
                    'attention_mask': attention_mask[i:i+1] if attention_mask is not None else None
                })

                if len(samples) >= num_samples:
                    break

            if len(samples) >= num_samples:
                break

        # Compute gradients for all samples
        gradients = []

        self.model.eval()
        for sample in samples:
            self.model.zero_grad()

            outputs = self.model(
                input_ids=sample['input_ids'],
                attention_mask=sample['attention_mask'],
                return_dict=True
            )

            # Scalar output
            logits = outputs.logits
            scalar_output = logits.mean()
            scalar_output.backward()

            # Collect gradient as flat vector
            grad_flat = torch.cat([
                param.grad.flatten()
                for param in self.model.parameters()
                if param.grad is not None
            ])

            gradients.append(grad_flat.detach())

        # Compute NTK: K = G G^T where G is [num_samples, num_params]
        gradients = torch.stack(gradients)  # [num_samples, num_params]

        if self.config.random_projection:
            # Random projection to reduce dimensionality
            proj_dim = min(self.config.projection_dim, gradients.shape[1])
            projection_matrix = torch.randn(
                gradients.shape[1], proj_dim,
                device=gradients.device
            ) / np.sqrt(proj_dim)
            gradients = gradients @ projection_matrix

        ntk_matrix = gradients @ gradients.T  # [num_samples, num_samples]

        return ntk_matrix

    def initialize_ntk(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_batches: Optional[int] = None,
    ):
        """
        Initialize NTK tracking by computing K_0 (NTK at initialization)

        This should be called before training starts.

        Args:
            dataloader: Training data
            num_batches: Number of batches to use
        """
        print("Computing initial NTK (K_0)...")

        if self.config.diagonal_only:
            self.ntk_diagonal_initial = self.compute_ntk_diagonal(
                dataloader, num_batches
            )
            initial_norm = self.ntk_diagonal_initial.norm().item()
        else:
            ntk_matrix = self.compute_ntk_matrix_sampled(dataloader)
            self.ntk_matrix_initial = ntk_matrix

            # Compute statistics
            initial_norm = torch.norm(ntk_matrix, p='fro').item()

            if self.config.track_spectrum:
                eigenvalues = torch.linalg.eigvalsh(ntk_matrix)
                self.eigenvalue_history.append({
                    'step': 0,
                    'eigenvalues': eigenvalues.cpu().numpy(),
                    'max_eigenvalue': eigenvalues.max().item(),
                    'condition_number': (eigenvalues.max() / (eigenvalues.min() + 1e-8)).item(),
                })

        self.ntk_norm_history.append({'step': 0, 'norm': initial_norm})

        print(f"Initial NTK norm: {initial_norm:.4f}")

    def update_ntk(
        self,
        dataloader: torch.utils.data.DataLoader,
        step: int,
        num_batches: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Update NTK tracking at current training step

        Computes K_t and checks if ||K_t - K_0||_F / ||K_0||_F < threshold

        Args:
            dataloader: Training data
            step: Current training step
            num_batches: Number of batches

        Returns:
            Dictionary with NTK metrics
        """
        if self.ntk_diagonal_initial is None and not hasattr(self, 'ntk_matrix_initial'):
            raise ValueError("Must call initialize_ntk() first")

        metrics = {'step': step}

        if self.config.diagonal_only:
            # Compute current diagonal
            ntk_diag_current = self.compute_ntk_diagonal(dataloader, num_batches)

            # Compute relative change
            diff = ntk_diag_current - self.ntk_diagonal_initial
            relative_change = diff.norm().item() / (self.ntk_diagonal_initial.norm().item() + 1e-8)

            metrics['ntk_diagonal_norm'] = ntk_diag_current.norm().item()
            metrics['relative_change'] = relative_change
            metrics['is_lazy'] = relative_change < self.config.lazy_threshold

            self.ntk_diagonal_current = ntk_diag_current

        else:
            # Compute current full NTK
            ntk_matrix_current = self.compute_ntk_matrix_sampled(dataloader)

            # Compute relative change
            diff = ntk_matrix_current - self.ntk_matrix_initial
            diff_norm = torch.norm(diff, p='fro').item()
            initial_norm = torch.norm(self.ntk_matrix_initial, p='fro').item()
            relative_change = diff_norm / (initial_norm + 1e-8)

            metrics['ntk_frobenius_norm'] = torch.norm(ntk_matrix_current, p='fro').item()
            metrics['relative_change'] = relative_change
            metrics['is_lazy'] = relative_change < self.config.lazy_threshold

            # Compute trace
            if self.config.track_trace:
                trace = torch.trace(ntk_matrix_current).item()
                metrics['ntk_trace'] = trace
                self.ntk_trace_history.append({'step': step, 'trace': trace})

            # Compute eigenvalues
            if self.config.track_spectrum:
                eigenvalues = torch.linalg.eigvalsh(ntk_matrix_current)
                metrics['max_eigenvalue'] = eigenvalues.max().item()
                metrics['condition_number'] = (eigenvalues.max() / (eigenvalues.min() + 1e-8)).item()

                self.eigenvalue_history.append({
                    'step': step,
                    'eigenvalues': eigenvalues.cpu().numpy(),
                    'max_eigenvalue': eigenvalues.max().item(),
                    'condition_number': metrics['condition_number'],
                })

        # Record history
        self.relative_change.append(relative_change)
        self.is_lazy.append(metrics['is_lazy'])
        self.ntk_norm_history.append({
            'step': step,
            'norm': metrics.get('ntk_frobenius_norm', metrics.get('ntk_diagonal_norm'))
        })

        return metrics

    def analyze_lazy_regime(self) -> Dict[str, Any]:
        """
        Analyze whether training has been in the lazy regime

        Returns:
            Summary statistics about lazy regime behavior
        """
        if not self.is_lazy:
            return {'error': 'No NTK updates tracked yet'}

        lazy_percentage = sum(self.is_lazy) / len(self.is_lazy) * 100
        mean_relative_change = np.mean(self.relative_change)
        max_relative_change = np.max(self.relative_change)

        return {
            'lazy_percentage': lazy_percentage,
            'mean_relative_change': mean_relative_change,
            'max_relative_change': max_relative_change,
            'is_lazy_regime': lazy_percentage > 80,  # 80% of time in lazy regime
            'num_updates': len(self.is_lazy),
        }

    def compute_ntk_alignment_with_update(
        self,
        update_vector: Dict[str, torch.Tensor],
        dataloader: torch.utils.data.DataLoader,
    ) -> Dict[str, float]:
        """
        Compute alignment between parameter update and NTK principal directions

        This helps understand whether updates follow NTK-predicted directions.

        Args:
            update_vector: Parameter update (θ_t - θ_0)
            dataloader: Data for NTK computation

        Returns:
            Alignment metrics
        """
        # Flatten update vector
        update_flat = torch.cat([
            v.flatten() for v in update_vector.values()
        ])

        # Compute current NTK
        if self.config.diagonal_only:
            # For diagonal NTK, we can't compute alignment
            return {'error': 'Alignment requires full NTK (diagonal_only=False)'}

        ntk_matrix = self.compute_ntk_matrix_sampled(dataloader)

        # Compute eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(ntk_matrix)

        # Sort by eigenvalue (descending)
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Top eigenvector (principal direction of NTK)
        top_eigenvector = eigenvectors[:, 0]

        # Compute alignment
        # (This is a rough approximation since update_flat is in parameter space
        # and eigenvectors are in data space)
        # Proper alignment would require Jacobian mapping

        return {
            'top_eigenvalue': eigenvalues[0].item(),
            'eigenvalue_ratio': (eigenvalues[0] / eigenvalues[-1]).item(),
            'note': 'Full alignment requires Jacobian mapping (expensive)',
        }

    def save_tracker(self, path: str):
        """Save NTK tracker state to disk"""
        save_dict = {
            'config': self.config,
            'ntk_diagonal_initial': self.ntk_diagonal_initial,
            'ntk_diagonal_current': self.ntk_diagonal_current,
            'ntk_matrix_initial': getattr(self, 'ntk_matrix_initial', None),
            'ntk_trace_history': self.ntk_trace_history,
            'ntk_norm_history': self.ntk_norm_history,
            'eigenvalue_history': self.eigenvalue_history,
            'is_lazy': self.is_lazy,
            'relative_change': self.relative_change,
        }
        torch.save(save_dict, path)

    def load_tracker(self, path: str):
        """Load NTK tracker state from disk"""
        save_dict = torch.load(path)
        self.config = save_dict['config']
        self.ntk_diagonal_initial = save_dict['ntk_diagonal_initial']
        self.ntk_diagonal_current = save_dict['ntk_diagonal_current']
        if save_dict['ntk_matrix_initial'] is not None:
            self.ntk_matrix_initial = save_dict['ntk_matrix_initial']
        self.ntk_trace_history = save_dict['ntk_trace_history']
        self.ntk_norm_history = save_dict['ntk_norm_history']
        self.eigenvalue_history = save_dict['eigenvalue_history']
        self.is_lazy = save_dict['is_lazy']
        self.relative_change = save_dict['relative_change']


def plot_ntk_evolution(tracker: NTKTracker, save_path: Optional[str] = None):
    """
    Plot NTK evolution over training

    Shows:
    1. NTK norm over time
    2. Relative change over time
    3. Lazy regime indicator

    Args:
        tracker: NTK tracker with history
        save_path: Path to save plot (optional)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1: NTK norm
    steps = [h['step'] for h in tracker.ntk_norm_history]
    norms = [h['norm'] for h in tracker.ntk_norm_history]

    axes[0].plot(steps, norms, 'b-', linewidth=2)
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('NTK Frobenius Norm')
    axes[0].set_title('NTK Norm Evolution')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Relative change and lazy threshold
    axes[1].plot(
        steps[1:],  # Skip initial step
        tracker.relative_change,
        'r-',
        linewidth=2,
        label='Relative Change'
    )
    axes[1].axhline(
        y=tracker.config.lazy_threshold,
        color='g',
        linestyle='--',
        linewidth=2,
        label='Lazy Threshold'
    )
    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('||K_t - K_0|| / ||K_0||')
    axes[1].set_title('NTK Relative Change (Lazy Training Indicator)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Shade lazy regime
    for i, is_lazy in enumerate(tracker.is_lazy):
        if is_lazy:
            axes[1].axvspan(
                steps[i+1] - 0.5 if i > 0 else 0,
                steps[i+1] + 0.5 if i < len(steps)-2 else steps[-1],
                alpha=0.2,
                color='green'
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


# Example usage:
"""
from transformers import AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

# Create NTK tracker
ntk_config = NTKConfig(
    num_samples=100,
    diagonal_only=True,  # Use diagonal for efficiency
    lazy_threshold=0.1,
)
ntk_tracker = NTKTracker(model, ntk_config)

# Initialize before training
ntk_tracker.initialize_ntk(train_dataloader, num_batches=10)

# During training (call periodically)
for step in range(0, 1000, 100):
    # ... training code ...

    # Update NTK tracking
    metrics = ntk_tracker.update_ntk(train_dataloader, step, num_batches=10)
    print(f"Step {step}: Relative change = {metrics['relative_change']:.4f}, "
          f"Is lazy = {metrics['is_lazy']}")

# Analyze lazy regime
analysis = ntk_tracker.analyze_lazy_regime()
print(f"Lazy regime percentage: {analysis['lazy_percentage']:.1f}%")
print(f"Is lazy regime: {analysis['is_lazy_regime']}")

# Plot evolution
plot_ntk_evolution(ntk_tracker, 'ntk_evolution.png')
"""
