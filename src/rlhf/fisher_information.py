"""
Fisher Information Matrix Computation for Language Models

The Fisher Information Matrix (FIM) serves as a Riemannian metric tensor
that shapes the geometry of parameter updates during RLHF.

Key theoretical roles:
1. Defines the natural gradient direction
2. Penalizes movement along high-curvature (principal) directions
3. Forms the metric for Riemannian distance in KL trust region

FIM Definition:
F = E_{x~p(x), y~π_θ(y|x)} [∇log π_θ(y|x) ∇log π_θ(y|x)^T]
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from collections import defaultdict


@dataclass
class FisherConfig:
    """Configuration for Fisher Information computation"""
    # Approximation settings
    use_empirical_fisher: bool = True  # Use empirical FIM (actual outputs) vs. true FIM (all outputs)
    block_diagonal: bool = True  # Approximate as block-diagonal (per-layer)
    low_rank: bool = False  # Use low-rank approximation
    rank: int = 512  # Rank for low-rank approximation

    # Sampling settings
    num_samples: int = 1000  # Number of samples for FIM estimation
    batch_size: int = 8  # Batch size for FIM computation
    seq_length: int = 128  # Sequence length for sampling

    # Eigendecomposition settings
    compute_eigenvectors: bool = True  # Compute eigenvectors (expensive)
    num_top_eigenvectors: int = 100  # Number of top eigenvectors to keep

    # Stability
    damping: float = 1e-5  # Damping for numerical stability


class FisherInformationComputer:
    """
    Computes and analyzes the Fisher Information Matrix for language models

    This is crucial for the Fisher Steering mechanism: the FIM acts as a
    metric tensor that penalizes updates along high-curvature principal directions.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[FisherConfig] = None,
        device: str = "cuda"
    ):
        self.model = model
        self.config = config or FisherConfig()
        self.device = device

        # Storage for FIM components
        self.fisher_diagonal = {}  # Diagonal elements (always computed)
        self.fisher_blocks = {}  # Block-diagonal matrices (if block_diagonal=True)
        self.eigenvalues = {}  # Eigenvalues per layer
        self.eigenvectors = {}  # Eigenvectors per layer (top-k only)

        # Statistics
        self.num_samples_seen = 0

    def compute_empirical_fisher(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_batches: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute empirical Fisher Information Matrix

        Empirical FIM: F = E[∇log π(y|x) ∇log π(y|x)^T]
        where y are actual sampled outputs (not all possible outputs)

        This is more practical than true FIM for large vocabulary LLMs.

        Args:
            dataloader: DataLoader providing (input_ids, attention_mask)
            num_batches: Number of batches to use (None = all)

        Returns:
            Dictionary of Fisher diagonal/blocks per parameter
        """
        self.model.eval()

        # Initialize accumulators
        fisher_diag = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        if self.config.block_diagonal:
            # For block-diagonal, we'll accumulate outer products
            # This is expensive, so we limit to specific layers
            fisher_blocks = {}

        num_samples = 0

        for batch_idx, batch in enumerate(dataloader):
            if num_batches and batch_idx >= num_batches:
                break

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            batch_size = input_ids.shape[0]

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            logits = outputs.logits

            # Sample from the policy (or use labels if available)
            if 'labels' in batch:
                # Use actual labels
                labels = batch['labels'].to(self.device)
            else:
                # Sample from the model's distribution
                probs = torch.softmax(logits, dim=-1)
                labels = torch.multinomial(
                    probs.view(-1, probs.shape[-1]),
                    num_samples=1
                ).view(input_ids.shape)

            # Compute log probabilities
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            # Gather log probs for selected tokens
            selected_log_probs = torch.gather(
                log_probs,
                dim=-1,
                index=labels.unsqueeze(-1)
            ).squeeze(-1)

            # Mask out padding
            if attention_mask is not None:
                selected_log_probs = selected_log_probs * attention_mask

            # Sum log probs across sequence (sequence-level log prob)
            sequence_log_prob = selected_log_probs.sum(dim=1)

            # Compute gradients for each sequence in batch
            for i in range(batch_size):
                self.model.zero_grad()

                # Backward pass for single sequence
                sequence_log_prob[i].backward(retain_graph=(i < batch_size - 1))

                # Accumulate squared gradients (diagonal FIM)
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        fisher_diag[name] += param.grad.detach() ** 2

            num_samples += batch_size

        # Average over samples
        for name in fisher_diag:
            fisher_diag[name] /= num_samples

        # Add damping for numerical stability
        for name in fisher_diag:
            fisher_diag[name] += self.config.damping

        self.fisher_diagonal = fisher_diag
        self.num_samples_seen = num_samples

        return fisher_diag

    def compute_fisher_vector_product(
        self,
        vector: Dict[str, torch.Tensor],
        use_cached: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher-vector product: F @ v

        This is useful for:
        1. Natural gradient: F^{-1} @ g
        2. Fisher norm: v^T @ F @ v
        3. Projections onto Fisher metric

        Args:
            vector: Dictionary of tensors (same structure as parameters)
            use_cached: Use cached Fisher diagonal

        Returns:
            F @ v as dictionary
        """
        if not use_cached or not self.fisher_diagonal:
            raise ValueError("Fisher diagonal not computed. Run compute_empirical_fisher first.")

        result = {}
        for name, v in vector.items():
            if name in self.fisher_diagonal:
                # Diagonal approximation: F @ v ≈ diag(F) * v
                result[name] = self.fisher_diagonal[name] * v
            else:
                result[name] = torch.zeros_like(v)

        return result

    def compute_fisher_norm(
        self,
        vector: Dict[str, torch.Tensor]
    ) -> float:
        """
        Compute Fisher norm: ||v||_F^2 = v^T F v

        This is the Riemannian distance metric used in KL trust region.

        Theoretical bound:
        ||θ - θ_0||_F^2 ≤ 2 * D_KL(π_θ || π_0)

        Args:
            vector: Parameter vector (as dict)

        Returns:
            Fisher norm squared
        """
        fv = self.compute_fisher_vector_product(vector)

        norm_squared = sum(
            (vector[name] * fv[name]).sum().item()
            for name in vector.keys()
            if name in fv
        )

        return norm_squared

    def compute_eigendecomposition(
        self,
        layer_names: Optional[List[str]] = None,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute eigendecomposition of Fisher blocks

        F = Q Λ Q^T

        This identifies the principal components (high-curvature directions).
        Fisher Steering hypothesis: RLHF updates avoid principal directions.

        Args:
            layer_names: Specific layers to decompose (None = all)

        Returns:
            Dictionary: {layer_name: (eigenvalues, eigenvectors)}
        """
        if not self.fisher_diagonal:
            raise ValueError("Compute Fisher first")

        results = {}

        for name, fisher_diag in self.fisher_diagonal.items():
            if layer_names and name not in layer_names:
                continue

            # For diagonal Fisher, eigenvalues are just the diagonal elements
            # and eigenvectors are standard basis vectors
            eigenvalues = fisher_diag.flatten()

            # Sort eigenvalues in descending order
            sorted_indices = torch.argsort(eigenvalues, descending=True)
            sorted_eigenvalues = eigenvalues[sorted_indices]

            # Keep top-k eigenvalues
            top_k = min(self.config.num_top_eigenvectors, len(sorted_eigenvalues))
            top_eigenvalues = sorted_eigenvalues[:top_k]

            # For diagonal Fisher, eigenvectors are one-hot
            # We'll just store indices for memory efficiency
            top_indices = sorted_indices[:top_k]

            results[name] = (top_eigenvalues.cpu(), top_indices.cpu())

        self.eigenvalues = {name: vals for name, (vals, _) in results.items()}
        self.eigenvectors = {name: vecs for name, (_, vecs) in results.items()}

        return results

    def analyze_principal_components(
        self,
        update_vector: Dict[str, torch.Tensor],
        top_k: int = 100,
    ) -> Dict[str, Any]:
        """
        Analyze update vector in Fisher principal component basis

        This is the core of Fisher Steering analysis:
        - Project update onto principal vs. off-principal subspaces
        - Measure alignment with high-curvature directions
        - Compute participation ratio

        Args:
            update_vector: Parameter update (θ_t - θ_0)
            top_k: Number of principal components to consider

        Returns:
            Analysis dictionary with metrics
        """
        if not self.eigenvalues:
            self.compute_eigendecomposition()

        results = {
            'principal_magnitude': {},
            'off_principal_magnitude': {},
            'alignment_with_top_pc': {},
            'participation_ratio': {},
        }

        for name, update in update_vector.items():
            if name not in self.eigenvalues:
                continue

            # Get eigenvalues and indices
            eigenvalues = self.eigenvalues[name]
            top_indices = self.eigenvectors[name][:top_k]

            # Flatten update
            update_flat = update.flatten()

            # Principal component magnitudes (top-k)
            principal_magnitudes = update_flat[top_indices.to(update.device)] ** 2
            principal_norm = principal_magnitudes.sum().sqrt()

            # Off-principal norm
            total_norm = update_flat.norm()
            off_principal_norm = (total_norm ** 2 - principal_norm ** 2).sqrt()

            # Alignment with top principal component
            if len(top_indices) > 0:
                top_pc_idx = top_indices[0].item()
                alignment = abs(update_flat[top_pc_idx]) / (total_norm + 1e-8)
            else:
                alignment = 0.0

            # Participation ratio: measure of how spread out the update is
            # PR = (Σ w_i)^2 / Σ w_i^2
            # High PR = update is spread across many components
            weights = update_flat ** 2
            pr = (weights.sum() ** 2) / (weights ** 2).sum()

            results['principal_magnitude'][name] = principal_norm.item()
            results['off_principal_magnitude'][name] = off_principal_norm.item()
            results['alignment_with_top_pc'][name] = alignment.item()
            results['participation_ratio'][name] = pr.item()

        # Aggregate metrics
        results['total_principal_norm'] = sum(results['principal_magnitude'].values()) ** 0.5
        results['total_off_principal_norm'] = sum(results['off_principal_magnitude'].values()) ** 0.5
        results['fisher_steering_score'] = (
            results['total_off_principal_norm'] /
            (results['total_principal_norm'] + 1e-8)
        )

        # Fisher Steering Score > 1.0 means update is primarily off-principal
        # This validates the Fisher Steering hypothesis

        return results

    def compute_effective_rank(
        self,
        threshold: float = 0.99
    ) -> Dict[str, float]:
        """
        Compute effective rank of Fisher matrix

        Effective rank measures how many dimensions have significant curvature.

        Args:
            threshold: Cumulative eigenvalue threshold (e.g., 0.99 = 99% of mass)

        Returns:
            Effective rank per layer
        """
        if not self.eigenvalues:
            self.compute_eigendecomposition()

        effective_ranks = {}

        for name, eigenvalues in self.eigenvalues.items():
            # Normalize eigenvalues
            normalized_eigs = eigenvalues / eigenvalues.sum()

            # Find how many eigenvectors needed to capture threshold of mass
            cumsum = torch.cumsum(normalized_eigs, dim=0)
            effective_rank = (cumsum < threshold).sum().item() + 1

            effective_ranks[name] = effective_rank

        return effective_ranks

    def save_fisher(self, path: str):
        """Save computed Fisher information to disk"""
        save_dict = {
            'fisher_diagonal': {k: v.cpu() for k, v in self.fisher_diagonal.items()},
            'eigenvalues': self.eigenvalues,
            'eigenvectors': self.eigenvectors,
            'num_samples': self.num_samples_seen,
            'config': self.config,
        }
        torch.save(save_dict, path)

    def load_fisher(self, path: str):
        """Load Fisher information from disk"""
        save_dict = torch.load(path)
        self.fisher_diagonal = {
            k: v.to(self.device) for k, v in save_dict['fisher_diagonal'].items()
        }
        self.eigenvalues = save_dict['eigenvalues']
        self.eigenvectors = save_dict['eigenvectors']
        self.num_samples_seen = save_dict['num_samples']


def analyze_gradient_fisher_alignment(
    gradients: Dict[str, torch.Tensor],
    fisher_computer: FisherInformationComputer,
) -> Dict[str, float]:
    """
    Analyze how gradients align with Fisher Information Matrix

    Natural gradient = F^{-1} @ g
    This analysis shows whether updates follow natural gradient direction.

    Args:
        gradients: Current gradients
        fisher_computer: Computed Fisher information

    Returns:
        Alignment metrics
    """
    # Compute Fisher-vector product with gradients
    fisher_grad = fisher_computer.compute_fisher_vector_product(gradients)

    # Compute alignment: g^T F g / (||g|| * ||Fg||)
    grad_norm = sum((g ** 2).sum().item() for g in gradients.values()) ** 0.5
    fisher_grad_norm = sum((fg ** 2).sum().item() for fg in fisher_grad.values()) ** 0.5

    dot_product = sum(
        (gradients[name] * fisher_grad[name]).sum().item()
        for name in gradients.keys()
    )

    alignment = dot_product / (grad_norm * fisher_grad_norm + 1e-8)

    # Compute natural gradient norm
    # This requires F^{-1} which we approximate with diagonal
    natural_gradient = {
        name: gradients[name] / (fisher_computer.fisher_diagonal[name] + 1e-5)
        for name in gradients.keys()
        if name in fisher_computer.fisher_diagonal
    }

    natural_grad_norm = sum((ng ** 2).sum().item() for ng in natural_gradient.values()) ** 0.5

    return {
        'gradient_fisher_alignment': alignment,
        'gradient_norm': grad_norm,
        'fisher_gradient_norm': fisher_grad_norm,
        'natural_gradient_norm': natural_grad_norm,
    }


# Example usage:
"""
from transformers import AutoModelForCausalLM
from torch.utils.data import DataLoader

# Load model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

# Create Fisher computer
fisher_config = FisherConfig(
    num_samples=1000,
    block_diagonal=True,
    compute_eigenvectors=True,
    num_top_eigenvectors=100
)
fisher_computer = FisherInformationComputer(model, fisher_config)

# Compute Fisher on training data
fisher_diag = fisher_computer.compute_empirical_fisher(train_dataloader, num_batches=100)

# Compute eigendecomposition
eigen_results = fisher_computer.compute_eigendecomposition()

# Analyze update vector (e.g., after RLHF training)
update_vector = {
    name: param - init_param
    for (name, param), init_param in zip(model.named_parameters(), initial_params)
}
pc_analysis = fisher_computer.analyze_principal_components(update_vector)

print(f"Fisher Steering Score: {pc_analysis['fisher_steering_score']:.3f}")
# Score > 1.0 indicates off-principal updates (validates hypothesis)
"""
