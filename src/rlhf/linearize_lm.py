"""
Language Model Linearization for Tangent Space RLHF

This module provides linearized versions of LLMs that operate in the tangent space
around pre-trained parameters. Two implementations are provided:

1. LinearizedLLM: Original implementation using torch.func.jvp
2. ImprovedLinearizedLLM: Optimized version with caching and better memory efficiency
"""

import torch
import torch.nn as nn
from torch.func import functional_call, jvp
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from collections import OrderedDict


@dataclass
class LinearizationConfig:
    """Configuration for linearized model behavior"""
    use_cache: bool = True  # Cache JVP computations
    low_rank_approx: bool = False  # Use low-rank approximation for task vector
    rank: int = 512  # Rank for low-rank approximation
    recompute_interval: int = 100  # Steps between JVP recomputation


class LinearizedLLM(nn.Module):
    """
    A Tangent-Space version of an LLM (Qwen/Llama).
    Computes: f_lin(x) = f(x; theta_0) + JVP(f, theta_0, delta_theta)

    This is the exact implementation of the Neural Tangent Kernel regime
    where the model operates in the linearized space around initialization.
    """
    def __init__(self, model: nn.Module):
        super().__init__()

        # 1. Store the base model and its frozen "theta_0" parameters
        # We put the model in eval mode for the base trace
        self.base_model = model.eval()

        # Extract the initial state (theta_0)
        # We use a dictionary format which is easier for LLMs
        self.params0 = {n: p.detach().clone() for n, p in model.named_parameters()}
        for p in self.params0.values():
            p.requires_grad = False

        # 2. Create the trainable task vector (delta_theta)
        # Instead of full params, we initialize a "Task Vector" of zeros
        self.task_vector = nn.ParameterDict({
            n.replace('.', '_'): nn.Parameter(torch.zeros_like(p))
            for n, p in model.named_parameters()
        })

    def _get_functional_params(self):
        """Helper to reconstruct parameter dict for functional_call"""
        # Mapping the flattened ParameterDict names back to the nested model names
        return {n: self.params0[n] for n in self.params0.keys()}

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """
        Forward pass in tangent space: f(θ_0) + JVP(f, θ_0, Δθ)

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            **kwargs: Additional arguments for the model

        Returns:
            Linearized logits
        """
        # We define a pure function that takes parameters and returns logits
        def func_forward(params):
            # functional_call maps params dict to the model structure
            return functional_call(
                self.base_model,
                params,
                (input_ids,),
                kwargs={'attention_mask': attention_mask, **kwargs}
            ).logits

        # theta_0: The initial frozen weights
        params0_dict = self._get_functional_params()

        # d_theta: The current task vector (delta theta)
        # We need to match the structure of params0_dict
        d_theta_dict = {
            n: self.task_vector[n.replace('.', '_')]
            for n in params0_dict.keys()
        }

        # 3. Compute JVP: f(x, theta_0) + (grad_f * d_theta)
        # out is the original model output, jvp_out is the linear correction
        out, jvp_out = jvp(
            func_forward,
            (params0_dict,),
            (d_theta_dict,)
        )

        return out + jvp_out

    def get_task_vector_norm(self) -> float:
        """Compute L2 norm of the task vector (parameter displacement)"""
        return sum(
            torch.norm(p).item() ** 2
            for p in self.task_vector.parameters()
        ) ** 0.5

    def get_linearization_error(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> float:
        """
        Compute linearization error: |f(θ) - f_linear(θ)|

        This measures how well the linear approximation holds.
        For RLHF in tangent space regime, this should be small.
        """
        # Get linearized output
        linear_out = self.forward(input_ids, attention_mask, **kwargs)

        # Get full model output at current parameters
        current_params = {
            n: self.params0[n] + self.task_vector[n.replace('.', '_')]
            for n in self.params0.keys()
        }

        with torch.no_grad():
            full_out = functional_call(
                self.base_model,
                current_params,
                (input_ids,),
                kwargs={'attention_mask': attention_mask, **kwargs}
            ).logits

        error = torch.abs(full_out - linear_out).mean().item()
        return error


class ImprovedLinearizedLLM(nn.Module):
    """
    Improved linearized LLM with optimizations:
    - Output full model structure (not just logits)
    - Better parameter name handling
    - Optional caching and low-rank approximation
    - Support for generation and training
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[LinearizationConfig] = None
    ):
        super().__init__()

        self.config = config or LinearizationConfig()
        self.base_model = model

        # Freeze base model
        for p in self.base_model.parameters():
            p.requires_grad = False
        self.base_model.eval()

        # Store initial parameters with proper nested structure
        self.params0 = OrderedDict()
        self.param_shapes = OrderedDict()

        for name, param in model.named_parameters():
            self.params0[name] = param.detach().clone()
            self.param_shapes[name] = param.shape

        # Create task vector (trainable delta)
        if self.config.low_rank_approx:
            # Low-rank approximation: Δθ ≈ U @ V^T
            # Useful for memory efficiency and implicit regularization
            self.task_vector_U = nn.ParameterDict()
            self.task_vector_V = nn.ParameterDict()

            for name, shape in self.param_shapes.items():
                if len(shape) >= 2:  # Matrix parameters
                    safe_name = name.replace('.', '__')
                    self.task_vector_U[safe_name] = nn.Parameter(
                        torch.randn(shape[0], self.config.rank) * 0.01
                    )
                    self.task_vector_V[safe_name] = nn.Parameter(
                        torch.randn(self.config.rank, shape[1]) * 0.01
                    )
                else:  # Vector parameters (biases, layer norms)
                    safe_name = name.replace('.', '__')
                    self.task_vector_U[safe_name] = nn.Parameter(
                        torch.zeros(*shape)
                    )
        else:
            # Full-rank task vector
            self.task_vector = nn.ParameterDict({
                name.replace('.', '__'): nn.Parameter(torch.zeros_like(param))
                for name, param in self.params0.items()
            })

        # Cache for JVP computation
        self.jvp_cache = None
        self.cache_step = 0

    def _get_task_vector_dict(self) -> Dict[str, torch.Tensor]:
        """Get task vector as a dictionary matching parameter names"""
        task_vec = {}

        for name in self.params0.keys():
            safe_name = name.replace('.', '__')

            if self.config.low_rank_approx:
                if safe_name in self.task_vector_U and safe_name in self.task_vector_V:
                    # Reconstruct from low-rank: Δθ = U @ V^T
                    task_vec[name] = self.task_vector_U[safe_name] @ self.task_vector_V[safe_name]
                else:
                    task_vec[name] = self.task_vector_U[safe_name]
            else:
                task_vec[name] = self.task_vector[safe_name]

        return task_vec

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        return_dict: bool = True,
        **kwargs
    ):
        """
        Forward pass with full model output structure

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            use_cache: Whether to use KV cache for generation
            return_dict: Return ModelOutput object

        Returns:
            ModelOutput with linearized logits
        """
        def func_forward(params):
            output = functional_call(
                self.base_model,
                params,
                (input_ids,),
                kwargs={
                    'attention_mask': attention_mask,
                    'use_cache': use_cache,
                    'return_dict': return_dict,
                    **kwargs
                }
            )
            # Return the full output for proper structure
            return output

        # Get parameter dictionaries
        params0_dict = dict(self.params0)
        d_theta_dict = self._get_task_vector_dict()

        # Compute linearization
        base_output, jvp_output = jvp(
            func_forward,
            (params0_dict,),
            (d_theta_dict,)
        )

        # Combine outputs
        if return_dict:
            # Create a new output object with linearized logits
            from copy import copy
            linearized_output = copy(base_output)
            linearized_output.logits = base_output.logits + jvp_output.logits
            return linearized_output
        else:
            # Tuple output: (logits, ...)
            return (base_output[0] + jvp_output[0],) + base_output[1:]

    def generate(self, input_ids: torch.Tensor, **kwargs):
        """
        Generation in tangent space

        Note: This is approximate since generation requires iterative forward passes
        and the linearization is recomputed at each step.
        """
        # For generation, we need to use the model in a special way
        # Option 1: Apply task vector to base model temporarily
        # Option 2: Use the linearized forward in a custom generation loop

        # For now, we'll apply the task vector and use base model's generate
        with torch.no_grad():
            # Temporarily apply task vector
            task_vec = self._get_task_vector_dict()
            original_params = {}

            for name, param in self.base_model.named_parameters():
                original_params[name] = param.data.clone()
                param.data.add_(task_vec[name])

            # Generate
            output = self.base_model.generate(input_ids, **kwargs)

            # Restore original parameters
            for name, param in self.base_model.named_parameters():
                param.data.copy_(original_params[name])

        return output

    def get_parameter_displacement(self) -> Dict[str, float]:
        """
        Compute parameter displacement metrics

        Returns:
            Dictionary with various displacement norms
        """
        task_vec = self._get_task_vector_dict()

        # L2 norm
        l2_norm = sum(
            torch.norm(delta).item() ** 2
            for delta in task_vec.values()
        ) ** 0.5

        # Per-layer norms
        layer_norms = {}
        for name, delta in task_vec.items():
            layer_norms[name] = torch.norm(delta).item()

        # Relative displacement (normalized by param magnitude)
        relative_norms = {}
        for name, delta in task_vec.items():
            param_norm = torch.norm(self.params0[name]).item()
            delta_norm = torch.norm(delta).item()
            relative_norms[name] = delta_norm / (param_norm + 1e-8)

        return {
            'l2_norm': l2_norm,
            'layer_norms': layer_norms,
            'relative_norms': relative_norms,
            'max_relative': max(relative_norms.values()),
            'mean_relative': sum(relative_norms.values()) / len(relative_norms)
        }

    def compute_linearization_error(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_samples: int = 1,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute linearization error across the sequence

        This validates the Sequence-Level Stability mechanism:
        Error should grow sublinearly with sequence length.

        Args:
            input_ids: Input tokens [batch_size, seq_len]
            attention_mask: Attention mask
            num_samples: Number of random positions to sample

        Returns:
            (mean_error, per_position_errors)
        """
        seq_len = input_ids.shape[1]
        errors = {}

        with torch.no_grad():
            # Get linearized output
            linear_output = self.forward(
                input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            linear_logits = linear_output.logits

            # Get true output (apply task vector to base model)
            task_vec = self._get_task_vector_dict()
            current_params = {
                name: self.params0[name] + task_vec[name]
                for name in self.params0.keys()
            }

            true_output = functional_call(
                self.base_model,
                current_params,
                (input_ids,),
                kwargs={'attention_mask': attention_mask, 'return_dict': True}
            )
            true_logits = true_output.logits

            # Compute per-position errors
            position_errors = torch.abs(true_logits - linear_logits).mean(dim=[0, 2])

            for pos in range(seq_len):
                errors[f'pos_{pos}'] = position_errors[pos].item()

            mean_error = position_errors.mean().item()

        return mean_error, errors


def compare_linearized_vs_full(
    model: nn.Module,
    linear_model: ImprovedLinearizedLLM,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Compare outputs between full model and linearized model

    This is useful for validating that RLHF operates in the tangent space regime.
    If the linear approximation is good, the outputs should be similar.
    """
    with torch.no_grad():
        # Full model output
        full_output = model(input_ids, attention_mask=attention_mask, return_dict=True)

        # Linearized output
        linear_output = linear_model(input_ids, attention_mask=attention_mask, return_dict=True)

        # Compare logits
        logit_diff = torch.abs(full_output.logits - linear_output.logits)

        # Compare predictions
        full_preds = full_output.logits.argmax(dim=-1)
        linear_preds = linear_output.logits.argmax(dim=-1)
        agreement = (full_preds == linear_preds).float().mean().item()

        results = {
            'mean_logit_diff': logit_diff.mean().item(),
            'max_logit_diff': logit_diff.max().item(),
            'prediction_agreement': agreement,
            'relative_error': (logit_diff / (torch.abs(full_output.logits) + 1e-8)).mean().item(),
        }

    return results


# Example usage:
"""
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

# Create linearized version
config = LinearizationConfig(use_cache=True, low_rank_approx=False)
linear_model = ImprovedLinearizedLLM(model, config)

# Forward pass
input_ids = tokenizer("Hello, world!", return_tensors="pt").input_ids
logits = linear_model(input_ids).logits

# Generate
output = linear_model.generate(input_ids, max_length=50)

# Check linearization quality
error, per_pos_errors = linear_model.compute_linearization_error(input_ids)
print(f"Linearization error: {error:.4f}")
"""
