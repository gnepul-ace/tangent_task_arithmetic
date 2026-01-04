# RLHF in the Tangent Space Regime: Implementation Roadmap

## Research Objective

Establish that RLHF operates fundamentally within the **Tangent Space Regime** of the pre-trained manifold through three geometric mechanisms:

1. **Riemannian Confinement**: KL trust region as geometric bound on parameter displacement
2. **Fisher Steering**: FIM as metric tensor penalizing high-curvature principal directions
3. **Sequence-Level Stability**: Linearization bounds for long-horizon reasoning stability

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   RLHF Tangent Space Framework               │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐      ┌──────────────────┐            │
│  │  Language Model  │ ───▶ │  Linearized LM   │            │
│  │  (Pre-trained)   │      │  (NTK Regime)    │            │
│  └──────────────────┘      └──────────────────┘            │
│           │                         │                        │
│           ▼                         ▼                        │
│  ┌─────────────────────────────────────────┐               │
│  │     KL-Constrained RL Optimizer         │               │
│  │     (PPO/GRPO with Fisher Tracking)     │               │
│  └─────────────────────────────────────────┘               │
│           │                                                  │
│           ├─────────────┬─────────────┬─────────────┐      │
│           ▼             ▼             ▼             ▼      │
│  ┌───────────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐ │
│  │ Riemannian    │ │ Fisher   │ │ Sequence │ │   NTK   │ │
│  │ Confinement   │ │ Steering │ │ Stability│ │ Tracker │ │
│  │ Analyzer      │ │ Analyzer │ │ Analyzer │ │         │ │
│  └───────────────┘ └──────────┘ └──────────┘ └─────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Plan

### Phase 1: Core Infrastructure (Foundation)

#### 1.1 Language Model Linearization
- **File**: `src/rlhf/linearize_lm.py`
- Extend `linearize.py` to support transformer language models
- Implement `LinearizedTransformer` class
- Support for autoregressive generation in linearized space
- Cache Jacobian computations for efficiency

#### 1.2 Fisher Information Matrix Computation
- **File**: `src/rlhf/fisher_information.py`
- Efficient FIM computation for language models
- Block-diagonal approximation (per-layer FIM)
- Eigendecomposition for principal component analysis
- Low-rank approximation for memory efficiency

#### 1.3 Neural Tangent Kernel Tracking
- **File**: `src/rlhf/ntk_tracker.py`
- Compute NTK matrix during training
- Track NTK spectral properties (eigenvalues, condition number)
- Detect "lazy training" regime (frozen NTK)
- Memory-efficient sampling-based NTK estimation

### Phase 2: KL-Constrained RL Optimizer

#### 2.1 Base RL Infrastructure
- **File**: `src/rlhf/rl_trainer.py`
- Implement PPO with KL penalty
- Implement GRPO (Group Relative Policy Optimization)
- Support for reward models and value networks
- Experience buffer and trajectory sampling

#### 2.2 KL-Constrained Optimization
- **File**: `src/rlhf/kl_constrained_optimizer.py`
- Adaptive KL coefficient (dual gradient descent)
- Trust region enforcement: $D_{KL}(\pi_\theta \| \pi_0) \leq \epsilon$
- Parameter displacement tracking
- Gradient projection onto Fisher metric

### Phase 3: Geometric Mechanism Analyzers

#### 3.1 Riemannian Confinement Analyzer
- **File**: `src/analysis/riemannian_confinement.py`
- **Theory**: KL trust region bounds parameter displacement
- **Metrics**:
  - Parameter norm: $\|\theta_t - \theta_0\|_2$
  - Fisher norm: $\|\theta_t - \theta_0\|_{F}^2 = (\theta_t - \theta_0)^T F (\theta_t - \theta_0)$
  - KL divergence vs. parameter displacement correlation
  - Tangent space projection error
- **Visualization**: Parameter trajectory on Riemannian manifold

#### 3.2 Fisher Steering Analyzer
- **File**: `src/analysis/fisher_steering.py`
- **Theory**: FIM penalizes high-curvature principal directions
- **Metrics**:
  - Gradient alignment with Fisher eigenvectors
  - Update magnitude along principal vs. off-principal directions
  - Principal component participation ratio
  - Effective rank of parameter updates
- **Key Analysis**: Show updates are "off-principal" (sparse in PC basis)

#### 3.3 Sequence-Level Stability Analyzer
- **File**: `src/analysis/sequence_stability.py`
- **Theory**: Linearization bounds for long-horizon reasoning
- **Metrics**:
  - Per-token linearization error: $|f(\theta) - f_{\text{linear}}(\theta)|$
  - Accumulated error over sequence length
  - Fisher information accumulation dynamics
  - Stability coefficient vs. sequence length
- **Visualization**: Error accumulation curves for different sequence lengths

### Phase 4: Experiments and Validation

#### 4.1 Experimental Setup
- **File**: `experiments/config/rlhf_tangent_config.yaml`
- Base models: Llama-3.2-1B, Qwen2.5-Math-1.5B
- Tasks: Mathematical reasoning (MATH dataset), code generation
- Baselines: Standard RLHF, vanilla fine-tuning, LoRA

#### 4.2 Validation Experiments
- **Experiment 1**: Verify NTK freezing during RLHF
  - Track NTK spectral properties over training
  - Compare to standard fine-tuning (non-lazy regime)

- **Experiment 2**: Riemannian confinement empirical validation
  - Vary KL constraint $\epsilon$
  - Measure parameter displacement and performance

- **Experiment 3**: Fisher steering analysis
  - Compute principal components of pre-trained FIM
  - Show RLHF updates are off-principal
  - Compare to random updates and full fine-tuning

- **Experiment 4**: Sequence-level stability
  - Test on varying sequence lengths (64, 128, 256, 512 tokens)
  - Measure linearization error accumulation
  - Validate stability bound

#### 4.3 Ablation Studies
- KL coefficient ablation
- Fisher preconditioning on/off
- Linearization vs. non-linear RLHF comparison

### Phase 5: Visualization and Analysis Tools

#### 5.1 Real-time Monitoring Dashboard
- **File**: `src/visualization/training_monitor.py`
- Live tracking of:
  - KL divergence and trust region violations
  - Parameter displacement (L2 and Fisher norms)
  - NTK eigenspectrum
  - Fisher principal component alignment
  - Linearization error

#### 5.2 Geometric Visualization
- **File**: `src/visualization/manifold_viz.py`
- 2D/3D projections of parameter trajectories
- Fisher ellipsoid visualization
- Principal component heatmaps
- Tangent space vs. manifold comparison

## Directory Structure

```
tangent_task_arithmetic/
├── src/
│   ├── rlhf/                           # RLHF core modules
│   │   ├── __init__.py
│   │   ├── linearize_lm.py             # Language model linearization
│   │   ├── fisher_information.py       # FIM computation
│   │   ├── ntk_tracker.py              # NTK tracking
│   │   ├── rl_trainer.py               # Base RL trainer
│   │   └── kl_constrained_optimizer.py # KL-constrained optimization
│   │
│   ├── analysis/                       # Geometric analyzers
│   │   ├── __init__.py
│   │   ├── riemannian_confinement.py   # Mechanism 1
│   │   ├── fisher_steering.py          # Mechanism 2
│   │   └── sequence_stability.py       # Mechanism 3
│   │
│   ├── visualization/                  # Visualization tools
│   │   ├── __init__.py
│   │   ├── training_monitor.py
│   │   └── manifold_viz.py
│   │
│   └── models/                         # Model definitions
│       ├── __init__.py
│       └── language_models.py          # LM wrappers
│
├── experiments/
│   ├── config/
│   │   ├── rlhf_tangent_config.yaml
│   │   └── baseline_config.yaml
│   │
│   ├── train_rlhf_tangent.py           # Main training script
│   ├── run_analysis.py                 # Run geometric analysis
│   └── validate_mechanisms.py          # Validation experiments
│
├── notebooks/
│   ├── 01_ntk_lazy_training.ipynb
│   ├── 02_riemannian_confinement.ipynb
│   ├── 03_fisher_steering.ipynb
│   └── 04_sequence_stability.ipynb
│
├── tests/
│   ├── test_linearization.py
│   ├── test_fisher_computation.py
│   └── test_analyzers.py
│
├── RESEARCH_ROADMAP.md                 # This file
└── environment_rlhf.yml                # Extended environment
```

## Key Theoretical Components to Implement

### 1. Lazy Training Condition
```python
# Track if NTK stays constant
def check_lazy_regime(ntk_t0, ntk_t, threshold=0.1):
    """Verify NTK freezing: ||K_t - K_0||_F / ||K_0||_F < threshold"""
    return frobenius_norm(ntk_t - ntk_t0) / frobenius_norm(ntk_t0) < threshold
```

### 2. KL Trust Region Bound
```python
# Riemannian distance bound
def verify_confinement(theta_t, theta_0, fisher, kl_div, epsilon):
    """
    Verify: D_KL(π_θ || π_0) ≤ ε implies bounded displacement
    ||θ_t - θ_0||_F^2 = (θ_t - θ_0)^T F (θ_t - θ_0) ≤ 2ε
    """
    displacement = theta_t - theta_0
    fisher_norm_sq = displacement @ fisher @ displacement
    return fisher_norm_sq, 2 * epsilon, kl_div
```

### 3. Fisher Steering Score
```python
# Measure off-principal alignment
def compute_steering_score(update, fisher_eigenvecs, top_k=100):
    """
    Project update onto principal vs. off-principal subspaces
    Score = ||update_off_principal|| / ||update_principal||
    High score → sparse, off-principal updates
    """
    principal_proj = project_onto_subspace(update, fisher_eigenvecs[:top_k])
    off_principal = update - principal_proj
    return norm(off_principal) / (norm(principal_proj) + 1e-8)
```

### 4. Linearization Error Bound
```python
# Sequence-level stability
def compute_linearization_error(model, linear_model, input_ids, max_length):
    """
    Track accumulated error: Σ_t |f(θ)_t - f_linear(θ)_t|
    Should grow sublinearly due to Fisher information counteracting accumulation
    """
    errors = []
    for t in range(max_length):
        logits_full = model(input_ids[:, :t+1])
        logits_linear = linear_model(input_ids[:, :t+1])
        errors.append(torch.abs(logits_full - logits_linear).mean())
    return torch.cumsum(torch.tensor(errors), dim=0)
```

## Dependencies

### Core
- PyTorch >= 2.0
- Transformers (HuggingFace)
- TRL (Transformer Reinforcement Learning)
- functorch (for Jacobian/Hessian)

### Analysis
- scipy (eigendecomposition)
- numpy
- matplotlib, seaborn (visualization)
- wandb (experiment tracking)

### Models
- Llama, Qwen, or other open-source LLMs
- Optional: vLLM for faster inference

## Success Criteria

### Empirical Validation
✅ **Lazy Training**: NTK eigenspectrum stays constant (< 10% change)
✅ **Riemannian Confinement**: Fisher norm correlates with KL divergence
✅ **Fisher Steering**: Updates have high off-principal ratio (> 2.0)
✅ **Sequence Stability**: Linearization error grows sublinearly

### Performance
✅ Match or exceed baseline RLHF performance
✅ Achieve similar results with fewer parameter updates
✅ Demonstrate improved sample efficiency

## Timeline Estimate

- **Phase 1** (Core Infrastructure): Implementation foundation
- **Phase 2** (RL Optimizer): KL-constrained training
- **Phase 3** (Analyzers): Geometric mechanism tracking
- **Phase 4** (Experiments): Empirical validation
- **Phase 5** (Visualization): Analysis and presentation

## References

1. **Tangent Task Arithmetic**: Ortiz-Jimenez et al., NeurIPS 2023
2. **Understanding R1-Zero**: SAIL-SG, 2024
3. **Neural Tangent Kernel**: Jacot et al., NeurIPS 2018
4. **PPO**: Schulman et al., 2017
5. **Fisher Information in RL**: Thomas et al., 2013

---

**Next Steps**: Begin Phase 1 implementation with language model linearization and Fisher computation modules.
