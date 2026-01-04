# Task Arithmetic in the Tangent Space + RLHF in the Tangent Space Regime

This repository contains:
1. **Original**: Source code for "[Task arithmetic in the tangent space: Improved editing of pre-trained models](https://arxiv.org/abs/2305.12827)"
2. **NEW**: Implementation of **RLHF in the Tangent Space Regime** - validating that Reinforcement Learning from Human Feedback operates fundamentally within the tangent space of pre-trained models

![](figures/tangent.jpg)

---

## 🆕 RLHF in the Tangent Space Regime

### Research Overview

This implementation validates a novel theoretical framework demonstrating that **RLHF operates in the Neural Tangent Kernel (NTK) regime** through three geometric mechanisms:

1. **Riemannian Confinement**: The KL trust region $D_{KL}(\pi_\theta || \pi_0) \leq \epsilon$ acts as a strict geometric bound on parameter displacement:
   $$||\theta - \theta_0||_F^2 \leq 2 \epsilon$$

2. **Fisher Steering**: The Fisher Information Matrix acts as a metric tensor that penalizes updates along high-curvature principal directions, forcing **off-principal**, sparse updates.

3. **Sequence-Level Stability**: Linearization errors accumulate **sublinearly** with sequence length ($\mathcal{O}(\sqrt{T})$ or $\mathcal{O}(\log T)$ instead of $\mathcal{O}(T)$), ensuring stability for long-horizon reasoning.

### Quick Start (RLHF)

```bash
# 1. Install dependencies
conda env create -f environment_rlhf.yml
conda activate tangent-rlhf

# Or with pip
pip install -r requirements_rlhf.txt

# 2. Train RLHF model with geometric analysis
# Option A: Standard RLHF (baseline)
python experiments/train_rlhf_geometric.py --config experiments/config/math_full.yaml

# Option B: Linearized RLHF (tangent space training)
python experiments/train_rlhf_geometric.py --config experiments/config/math_linear.yaml

# 3. Distributed training (8x H100 GPUs)
torchrun --nproc_per_node=8 experiments/train_rlhf_geometric.py \
    --config experiments/config/math_linear.yaml
```

### Available Experiment Configs

| Config | Dataset | Model Type | Purpose |
|--------|---------|------------|---------|
| `math_full.yaml` | MATH | Full model | Baseline RLHF |
| `math_linear.yaml` | MATH | Linearized | Tangent space RLHF |
| `gsm8k_linear.yaml` | GSM8K | Linearized | Grade school math |
| `code_linear.yaml` | Code | Linearized | Code generation |

### Architecture

```
tangent_task_arithmetic/
├── src/
│   ├── rlhf/                           # RLHF core modules
│   │   ├── linearize_lm.py             # Linearized LLM (tangent space)
│   │   ├── fisher_information.py       # Fisher Information Matrix
│   │   ├── ntk_tracker.py              # Neural Tangent Kernel tracking
│   │   └── kl_constrained_trainer.py   # Geometric RLHF trainer
│   │
│   ├── analysis/                       # Geometric mechanism analyzers
│   │   ├── riemannian_confinement.py   # KL trust region analysis
│   │   ├── fisher_steering.py          # Principal component analysis
│   │   └── sequence_stability.py       # Linearization error analysis
│   │
│   └── [original task arithmetic code]
│
├── experiments/
│   ├── config/                         # Experiment configurations
│   │   ├── base_config.yaml
│   │   ├── math_linear.yaml
│   │   ├── gsm8k_linear.yaml
│   │   └── code_linear.yaml
│   │
│   └── train_rlhf_geometric.py         # Main training script
│
├── RESEARCH_ROADMAP.md                 # Detailed implementation plan
└── notebooks/                          # Analysis notebooks

```

### Key Components

#### 1. Linearized Language Models

Train LLMs **explicitly in the tangent space** using first-order Taylor expansion:

```python
from src.rlhf.linearize_lm import ImprovedLinearizedLLM, LinearizationConfig

# Create linearized model
config = LinearizationConfig(use_cache=True, low_rank_approx=False)
linear_model = ImprovedLinearizedLLM(base_model, config)

# Forward pass computes: f(θ₀) + JVP(f, θ₀, Δθ)
logits = linear_model(input_ids, attention_mask=mask)

# Measure linearization error
error, per_pos_errors = linear_model.compute_linearization_error(input_ids)
```

#### 2. Fisher Information Matrix

Analyze how FIM shapes the geometry of RLHF updates:

```python
from src.rlhf.fisher_information import FisherInformationComputer

# Compute Fisher at initialization
fisher = FisherInformationComputer(model)
fisher.compute_empirical_fisher(dataloader, num_batches=100)
fisher.compute_eigendecomposition()

# Analyze parameter updates
update_vector = {name: param - init_param for name, param in model.named_parameters()}
pc_analysis = fisher.analyze_principal_components(update_vector)

# Check Fisher Steering hypothesis
print(f"Steering Score: {pc_analysis['fisher_steering_score']:.3f}")
# Score > 1.0 = off-principal updates (validates hypothesis)
```

#### 3. Three Geometric Analyzers

**Riemannian Confinement** (validates KL trust region):
```python
from src.analysis.riemannian_confinement import RiemannianConfinementAnalyzer

analyzer = RiemannianConfinementAnalyzer(model, ref_model, fisher_computer)
results = analyzer.analyze_at_step(step, dataloader)

print(f"KL divergence: {results['kl_divergence']:.6f}")
print(f"Fisher norm: {results['fisher_norm']:.6f}")
print(f"Bound satisfied: {results['bound_validation']['is_bound_satisfied']}")
```

**Fisher Steering** (validates off-principal updates):
```python
from src.analysis.fisher_steering import FisherSteeringAnalyzer

steering = FisherSteeringAnalyzer(fisher_computer)
results = steering.full_analysis(update_vector, step)

# Steering score > 1.0 means primarily off-principal updates
print(f"Steering score: {results['decomposition']['global']['steering_score']:.3f}")
```

**Sequence-Level Stability** (validates sublinear error growth):
```python
from src.analysis.sequence_stability import SequenceLevelStabilityAnalyzer

stability = SequenceLevelStabilityAnalyzer(linear_model)
results = stability.analyze_sequence_length_scaling(dataloader)

# Check if error grows sublinearly: E ∝ T^α where α < 1.0
print(f"Growth exponent: {results['power_law']['exponent']:.3f}")
print(f"Sublinear: {results['power_law']['is_sublinear']}")
```

### Training with Geometric Analysis

The `GeometricRLHFTrainer` instruments RLHF training with comprehensive geometric tracking:

```python
from src.rlhf.kl_constrained_trainer import GeometricRLHFTrainer

trainer = GeometricRLHFTrainer(
    model=model,
    ref_model=ref_model,
    reward_model=reward_model,
    tokenizer=tokenizer,
    train_config=train_config,
    analysis_config=analysis_config,
)

# Initialize analyzers
trainer.initialize_analyzers(train_dataloader)

# During training
for step in training_loop:
    # ... RLHF training step ...

    # Periodic geometric analysis
    if step % 100 == 0:
        results = trainer.run_geometric_analysis(step, val_dataloader)

        # Automatically tracks:
        # - KL divergence and trust region compliance
        # - Parameter displacement (L2 and Fisher norms)
        # - NTK stability (lazy regime detection)
        # - Fisher steering score
        # - Sequence-level linearization error
```

### Expected Results

If the hypothesis is correct, you should observe:

✅ **Lazy Training**: NTK relative change < 10% throughout training
✅ **Riemannian Confinement**: High correlation (>0.8) between KL and Fisher norm
✅ **Fisher Steering**: Steering score > 1.0 (updates are off-principal)
✅ **Sequence Stability**: Error growth exponent α < 0.7 (sublinear)

### Integration with RLHF Frameworks

This code is designed to integrate with existing RLHF frameworks:

- **verl** (volcengine): High-performance RLHF library
- **oat** (from understand-r1-zero): Research-oriented RL framework

See `src/rlhf/kl_constrained_trainer.py` for integration points.

### Hardware Requirements

- **Minimum**: 1x A100 40GB (for 1.5B models)
- **Recommended**: 8x H100 80GB (for distributed training)
- **Tested on**: 8x H100 80GB with Qwen2.5-Math-1.5B

### Detailed Documentation

- **[RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md)**: Complete implementation plan, theoretical foundation, and phase-by-phase breakdown
- **Notebooks**: Coming soon (analysis and visualization notebooks)
- **Examples**: See `experiments/train_rlhf_geometric.py` for full training example

---

## Original: Task Arithmetic in the Tangent Space

This is the source code to reproduce the experiments of the paper "[Task arithmetic in the tangent space: Improved editing of pre-trained models](https://arxiv.org/abs/2305.12827)" by Guillermo Ortiz-Jimenez*, Alessandro Favero* and Pascal Frossard.

### Dependencies (Original)

To run the original code, please install all its dependencies:
```sh
conda env create
conda activate tangent-arithmetic
```
and add the `src` directory to the `PYTHONPATH`:
```sh
cd tangent_task_arithmetic
export PYTHONPATH="$PYTHONPATH:$PWD"
```

### Repository content (Original)

This repository is heavily based on the code from [Ilharco et al. (2022)](https://github.com/mlfoundations/task_vectors) and follows the same structure.

#### Task vectors

The task vector logic in [src/task_vectors.py](src/task_vectors.py) has been extended to distinguish between `NonLinearTaskVector`s and `LinearizedTaskVector`s which can be applied to non-linear `ImageEncoder`s and `LinearizedImageEncoder`s, respectively. Given a pre-trained checkpoint and a fine-tuned checkpoint, you can create a linearized/standard task vector as:

```python
from src.task_vectors import NonLinearTaskVector, LinearizedTaskVector

# Non-linear task vector.
zeroshot_checkpoint = ... # Pre-trained non-linear image encoder.
finetuned_checkpoint = ... # Non-linearly fine-tuned checkpoint.

nonlinear_task_vector = NonLinearTaskVector(zeroshot_checkpoint, finetuned_checkpoint)

# Tangent task vector.
linear_zeroshot_checkpoint = ... # Pre-trained linearized image encoder.
linear_finetuned_checkpoint = ... # Linearly fine-tuned checkpoint.

linear_task_vector = LinearizedTaskVector(linear_zeroshot_checkpoint, linear_finetuned_checkpoint)
```

Once created, we can modify and combine the task vectors through arithmetic operations in Python, e.g.,
```python
negated_task_vector = -task_vector # Negating a task vector.
multi_task_vector = 0.5 * task_vector_1 + 0.7 * task_vector_2 # Adding two vectors.
```
and apply them to a pre-trained encoder as:
```python
edited_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=0.8)
```

Sometimes, we may want to apply a non-linear task vector to a `LinearizedImageEncoder` (to obtain posthoc linearized models for example), or viceversa. Both `NonLinearTaskVector` and `LinearizedTaskVector` can be casted and applied to encoders from the complementary class as
```python
linear_edited_encoder = nonlinear_task_vector.apply_to_linear(linear_pretrained_encoder, scaling_coef=0.8)
```

#### Linearized Models

The module [src/linearize.py](src/linearize.py) provides tools to linearize any PyTorch `nn.Module`.

To linearize any `model` object of the class `nn.Module` one can simply do:
```python
from src.linearize import LinearizedModel

model = ... # An object of the class `nn.Module`.
linear_model = LinearizedModel(model) # This object can be treated as any other `nn.Module`.
```
Specifically for `ImageEncoder`s the class `LinearizedImageEncoder` provides a simple way to linearize a CLIP image encoder while retaining the same API as the original object from the `ImageEncoder` class. We can therefore create a linearized CLIP model as:
```python
from src.linearize import LinearizedImageEncoder
from src.heads import get_classification_head
from src.modeling import ImageClassifier

args = ... # Arguments used to define an `ImageEncoder`.
linear_encoder = LinearizedImageEncoder(args, keep_lang=False) # This object can be treated as any other `ImageEncoder`.

classification_head = get_classification_head(args, train_dataset)

linear_clip = ImageClassifier(image_encoder, classification_head)
```
#### Training

The script `src/finetune.py` can be used to reproduce the training protocol we used to fine-tune our models on all our downstream tasks (both linearly and non-linearly).
```sh
python src/finetune.py --finetuning-mode=standard --model=ViT-B-32 --world-size=2 # Finetune non-linearly on 2 GPUs
python src/finetune.py --finetuning-mode=linear --model=ViT-B-32 --world-size=2 # Finetune linearly on 2 GPUs
```

#### Evaluation

We provide different scripts to evaluate the different task vectors obtained using the previous scripts.

##### Single-task accuracy
Having run `src/finetune.py` for a given model, you can evaluate the performance of the fine-tuned weights on each single task by running
```sh
# Evaluate pre-trained models.
python src/eval_single_task.py --model=ViT-B-32 --finetuning-mode=none

# Evaluate non-linearly fine-tuned models.
python src/eval_single_task.py --model=ViT-B-32 --finetuning-mode=standard

# Evaluate linearly fine-tuned models.
python src/eval_single_task.py --model=ViT-B-32 --finetuning-mode=linear

# Evaluate post-hoc linearized models. Requires having run finetune.py with --finetuning=mode=standard.
python src/eval_single_task.py --model=ViT-B-32 --finetuning-mode=posthoc
```

##### Task addition
Once evaluated on the single tasks, we can evaluate the task arithmetic performance of the different strategies on the addition benchmark.
```sh
# Evaluate non-linearly fine-tuned models.
python src/eval_task_addition.py --model=ViT-B-32 --finetuning-mode=standard

# Evaluate linearly fine-tuned models.
python src/eval_task_addition.py --model=ViT-B-32 --finetuning-mode=linear

# Evaluate post-hoc linearized models.
python src/eval_task_addition.py --model=ViT-B-32 --finetuning-mode=posthoc
```

##### Task negation
We can evaluate the task arithmetic performance of the different strategies on the negation benchmark.
```sh
# Evaluate non-linearly fine-tuned models.
python src/eval_task_negation.py --model=ViT-B-32 --finetuning-mode=standard

# Evaluate linearly fine-tuned models.
python src/eval_task_negation.py --model=ViT-B-32 --finetuning-mode=linear

# Evaluate post-hoc linearized models.
python src/eval_task_negation.py --model=ViT-B-32 --finetuning-mode=posthoc
```

### Datasets
To download and prepare the datasets, please follow the instructions in [this issue](https://github.com/mlfoundations/task_vectors/issues/1).

---

## References

### Original Paper
If you use the task arithmetic code, please cite:
```bibtex
@article{ortizjimenez2023tangent,
  title   = {Task Arithmetic in the Tangent Space: Improved Editing of Pre-Trained
             Models},
  author  = {Guillermo Ortiz{-}Jim{\'{e}}nez and
             Alessandro Favero and
             Pascal Frossard},
  journal = {arXiv:2305.12827},
  year    = {2023},
  note    = {\url{https://arxiv.org/abs/2305:12827}},
}
```

### RLHF in Tangent Space Research
If you use the RLHF tangent space code, please cite:
```bibtex
@article{rlhf_tangent_space2025,
  title   = {Reinforcement Learning from Human Feedback Operates in the Tangent Space Regime},
  author  = {[Your Name]},
  journal = {In Preparation},
  year    = {2025},
}
```

### Related Work
- [Task Vectors](https://github.com/mlfoundations/task_vectors) - Ilharco et al., 2022
- [Understanding R1-Zero](https://github.com/sail-sg/understand-r1-zero) - SAIL-SG, 2024
- [Neural Tangent Kernel](https://arxiv.org/abs/1806.07572) - Jacot et al., NeurIPS 2018

---

## License

MIT License (see LICENSE file)

## Contributing

Contributions are welcome! Please open an issue or pull request.

## Contact

For questions about the RLHF tangent space implementation, please open an issue on GitHub.
