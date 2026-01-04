"""
Fisher Steering Analyzer

THEORETICAL CLAIM:
The Fisher Information Matrix acts as a metric tensor that penalizes movement
along high-curvature principal directions. This forces RLHF updates to be
"off-principal" - sparse in the principal component basis.

MATHEMATICAL FOUNDATION:
Natural gradient: g_nat = F^{-1} @ g

The FIM penalizes updates in directions of high curvature (large eigenvalues).
Updates should concentrate on low-curvature (off-principal) directions.

This analyzer validates:
1. Parameter updates have low alignment with top principal components
2. Updates are sparse in PC basis (high participation ratio)
3. Fisher Steering Score = ||Δθ_off|| / ||Δθ_principal|| > 1.0
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class FisherSteeringConfig:
    """Configuration for Fisher steering analysis"""
    num_principal_components: int = 100  # Number of top PCs to track
    num_bottom_components: int = 100  # Number of bottom PCs

    # Analysis settings
    compute_alignment: bool = True
    compute_participation_ratio: bool = True
    track_per_layer: bool = True

    # Thresholds
    steering_score_threshold: float = 1.0  # Score > 1.0 validates hypothesis


class FisherSteeringAnalyzer:
    """
    Analyzes the Fisher Steering mechanism in RLHF

    Validates that FIM steers updates away from principal directions.
    """

    def __init__(
        self,
        fisher_computer: Any,  # FisherInformationComputer
        config: Optional[FisherSteeringConfig] = None,
        device: str = "cuda"
    ):
        self.fisher_computer = fisher_computer
        self.config = config or FisherSteeringConfig()
        self.device = device

        # Ensure eigendecomposition is computed
        if not self.fisher_computer.eigenvalues:
            self.fisher_computer.compute_eigendecomposition()

        # History
        self.steering_score_history = []
        self.alignment_history = []
        self.participation_ratio_history = []

    def analyze_update_decomposition(
        self,
        update_vector: Dict[str, torch.Tensor],
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Decompose update vector into principal vs. off-principal components

        Args:
            update_vector: Parameter update (θ_t - θ_0)
            step: Training step (for logging)

        Returns:
            Decomposition analysis
        """
        results = {
            'step': step,
            'per_layer': {},
            'global': {},
        }

        total_principal_norm_sq = 0.0
        total_off_principal_norm_sq = 0.0
        total_update_norm_sq = 0.0

        for name, update in update_vector.items():
            if name not in self.fisher_computer.eigenvalues:
                continue

            # Get principal component indices
            top_k = min(
                self.config.num_principal_components,
                len(self.fisher_computer.eigenvalues[name])
            )
            top_indices = self.fisher_computer.eigenvectors[name][:top_k].to(update.device)

            # Flatten update
            update_flat = update.flatten()
            update_norm_sq = (update_flat ** 2).sum().item()
            total_update_norm_sq += update_norm_sq

            # Principal component magnitudes
            principal_components = update_flat[top_indices] if len(top_indices) > 0 else torch.tensor([])
            principal_norm_sq = (principal_components ** 2).sum().item()
            total_principal_norm_sq += principal_norm_sq

            # Off-principal norm
            off_principal_norm_sq = update_norm_sq - principal_norm_sq
            total_off_principal_norm_sq += off_principal_norm_sq

            # Layer-specific analysis
            results['per_layer'][name] = {
                'principal_norm': np.sqrt(principal_norm_sq),
                'off_principal_norm': np.sqrt(off_principal_norm_sq),
                'total_norm': np.sqrt(update_norm_sq),
                'principal_fraction': principal_norm_sq / (update_norm_sq + 1e-8),
            }

        # Global metrics
        results['global']['principal_norm'] = np.sqrt(total_principal_norm_sq)
        results['global']['off_principal_norm'] = np.sqrt(total_off_principal_norm_sq)
        results['global']['total_norm'] = np.sqrt(total_update_norm_sq)

        # Fisher Steering Score: ||off|| / ||principal||
        # Score > 1.0 means update is primarily off-principal
        steering_score = (
            np.sqrt(total_off_principal_norm_sq) /
            (np.sqrt(total_principal_norm_sq) + 1e-8)
        )
        results['global']['steering_score'] = steering_score
        results['global']['validates_hypothesis'] = steering_score > self.config.steering_score_threshold

        # Track history
        if step is not None:
            self.steering_score_history.append({
                'step': step,
                'score': steering_score
            })

        return results

    def compute_pc_alignment(
        self,
        update_vector: Dict[str, torch.Tensor],
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Compute alignment of update with individual principal components

        Measures: alignment[k] = |<update, pc_k>| / ||update||

        Args:
            update_vector: Parameter update
            step: Training step

        Returns:
            Alignment metrics
        """
        results = {
            'step': step,
            'per_layer': {},
            'global': {},
        }

        all_alignments = []

        for name, update in update_vector.items():
            if name not in self.fisher_computer.eigenvalues:
                continue

            eigenvalues = self.fisher_computer.eigenvalues[name]
            top_indices = self.fisher_computer.eigenvectors[name][:self.config.num_principal_components]

            update_flat = update.flatten()
            update_norm = update_flat.norm().item()

            # Compute alignment with each PC
            alignments = []
            for idx in top_indices[:20]:  # Top 20 PCs
                if idx.item() < len(update_flat):
                    component_value = abs(update_flat[idx.item()].item())
                    alignment = component_value / (update_norm + 1e-8)
                    alignments.append(alignment)

            if alignments:
                results['per_layer'][name] = {
                    'mean_alignment': np.mean(alignments),
                    'max_alignment': np.max(alignments),
                    'top_alignments': alignments[:5],
                }
                all_alignments.extend(alignments)

        if all_alignments:
            results['global']['mean_alignment'] = np.mean(all_alignments)
            results['global']['max_alignment'] = np.max(all_alignments)
            results['global']['std_alignment'] = np.std(all_alignments)

            # Low alignment with top PCs validates steering hypothesis
            results['global']['validates_hypothesis'] = results['global']['mean_alignment'] < 0.1

            if step is not None:
                self.alignment_history.append({
                    'step': step,
                    'mean_alignment': results['global']['mean_alignment'],
                    'max_alignment': results['global']['max_alignment'],
                })

        return results

    def compute_participation_ratio(
        self,
        update_vector: Dict[str, torch.Tensor],
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Compute participation ratio of update

        PR = (Σ w_i)^2 / (Σ w_i^2)

        High PR means update is spread across many components (sparse in PC basis)

        Args:
            update_vector: Parameter update
            step: Training step

        Returns:
            Participation ratio metrics
        """
        results = {
            'step': step,
            'per_layer': {},
            'global': {},
        }

        total_pr_weighted = 0.0
        total_weight = 0.0

        for name, update in update_vector.items():
            update_flat = update.flatten()
            weights = update_flat ** 2

            # Participation ratio
            sum_weights = weights.sum().item()
            sum_weights_sq = (weights ** 2).sum().item()

            pr = (sum_weights ** 2) / (sum_weights_sq + 1e-8)
            effective_dim = len(weights)
            normalized_pr = pr / effective_dim  # Normalize by dimensionality

            results['per_layer'][name] = {
                'participation_ratio': pr,
                'normalized_pr': normalized_pr,
                'effective_dimensionality': pr,
                'total_dimensionality': effective_dim,
            }

            total_pr_weighted += pr * sum_weights
            total_weight += sum_weights

        results['global']['weighted_participation_ratio'] = total_pr_weighted / (total_weight + 1e-8)

        # High PR validates hypothesis (update is spread out)
        results['global']['validates_hypothesis'] = results['global']['weighted_participation_ratio'] > 100

        if step is not None:
            self.participation_ratio_history.append({
                'step': step,
                'pr': results['global']['weighted_participation_ratio'],
            })

        return results

    def full_analysis(
        self,
        update_vector: Dict[str, torch.Tensor],
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Perform complete Fisher steering analysis

        Args:
            update_vector: Parameter update
            step: Training step

        Returns:
            Complete analysis results
        """
        results = {
            'step': step,
            'decomposition': self.analyze_update_decomposition(update_vector, step),
            'alignment': self.compute_pc_alignment(update_vector, step),
            'participation_ratio': self.compute_participation_ratio(update_vector, step),
        }

        # Overall validation
        validates = [
            results['decomposition']['global']['validates_hypothesis'],
            results['alignment']['global'].get('validates_hypothesis', False),
            results['participation_ratio']['global']['validates_hypothesis'],
        ]
        results['overall_validates_hypothesis'] = sum(validates) >= 2  # Majority

        return results

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics across training"""
        if not self.steering_score_history:
            return {'error': 'No history tracked'}

        steering_scores = [h['score'] for h in self.steering_score_history]

        summary = {
            'num_steps': len(self.steering_score_history),
            'mean_steering_score': np.mean(steering_scores),
            'median_steering_score': np.median(steering_scores),
            'min_steering_score': np.min(steering_scores),
            'max_steering_score': np.max(steering_scores),
            'hypothesis_validated': np.mean(steering_scores) > self.config.steering_score_threshold,
        }

        if self.alignment_history:
            alignments = [h['mean_alignment'] for h in self.alignment_history]
            summary['mean_pc_alignment'] = np.mean(alignments)
            summary['low_alignment_rate'] = sum(1 for a in alignments if a < 0.1) / len(alignments)

        if self.participation_ratio_history:
            prs = [h['pr'] for h in self.participation_ratio_history]
            summary['mean_participation_ratio'] = np.mean(prs)

        return summary

    def save_analysis(self, path: str):
        """Save analysis to disk"""
        save_dict = {
            'config': self.config,
            'steering_score_history': self.steering_score_history,
            'alignment_history': self.alignment_history,
            'participation_ratio_history': self.participation_ratio_history,
            'summary': self.get_summary_statistics(),
        }
        torch.save(save_dict, path)


def plot_fisher_steering(
    analyzer: FisherSteeringAnalyzer,
    save_path: Optional[str] = None
):
    """Visualize Fisher steering analysis"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Steering score over time
    if analyzer.steering_score_history:
        steps = [h['step'] for h in analyzer.steering_score_history]
        scores = [h['score'] for h in analyzer.steering_score_history]

        axes[0, 0].plot(steps, scores, 'b-', linewidth=2, label='Steering Score')
        axes[0, 0].axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Threshold (1.0)')
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('||Δθ_off|| / ||Δθ_principal||')
        axes[0, 0].set_title('Fisher Steering Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: PC alignment
    if analyzer.alignment_history:
        steps = [h['step'] for h in analyzer.alignment_history]
        mean_align = [h['mean_alignment'] for h in analyzer.alignment_history]
        max_align = [h['max_alignment'] for h in analyzer.alignment_history]

        axes[0, 1].plot(steps, mean_align, 'g-', linewidth=2, label='Mean Alignment')
        axes[0, 1].plot(steps, max_align, 'r--', linewidth=2, label='Max Alignment')
        axes[0, 1].axhline(y=0.1, color='orange', linestyle='--', label='Low Threshold')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Alignment with Top PCs')
        axes[0, 1].set_title('Principal Component Alignment')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Participation ratio
    if analyzer.participation_ratio_history:
        steps = [h['step'] for h in analyzer.participation_ratio_history]
        prs = [h['pr'] for h in analyzer.participation_ratio_history]

        axes[1, 0].plot(steps, prs, 'm-', linewidth=2)
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Participation Ratio')
        axes[1, 0].set_title('Update Sparsity (Participation Ratio)')
        axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Summary
    summary = analyzer.get_summary_statistics()
    summary_text = "\n".join([
        "Fisher Steering Summary:",
        f"Mean Steering Score: {summary.get('mean_steering_score', 0):.3f}",
        f"Hypothesis Validated: {summary.get('hypothesis_validated', False)}",
        f"Mean PC Alignment: {summary.get('mean_pc_alignment', 0):.4f}",
        f"Mean Participation Ratio: {summary.get('mean_participation_ratio', 0):.1f}",
    ])

    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                     family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    axes[1, 1].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
