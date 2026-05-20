<div align="center">

<br/>

```
███╗   ███╗██╗███╗   ██╗██╗      ██████╗ ██╗
████╗ ████║██║████╗  ██║██║     ██╔══██╗██║
██╔████╔██║██║██╔██╗ ██║██║     ██████╔╝██║
██║╚██╔╝██║██║██║╚██╗██║██║     ██╔═══╝ ██║
██║ ╚═╝ ██║██║██║ ╚████║██║     ██║     ██║
╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═╝     ╚═╝     ╚═╝
```

### A miniature π0-style Vision-Language-Action agent with a **flow matching** action head

<br/>

[![CI](https://github.com/oliverdippel/vlm_agent/actions/workflows/ci.yml/badge.svg)](https://github.com/oliverdippel/vlm_agent/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![W&B](https://img.shields.io/badge/Tracked-W%26B-orange.svg)](https://wandb.ai)

<br/>

</div>

---

## What is this?

**mini-π0** is a from-scratch implementation of a Vision-Language-Action (VLA) agent inspired by Physical Intelligence's [π0](https://www.physicalintelligence.company/download/pi0.pdf) and [π0.5](https://www.physicalintelligence.company/blog/pi05). It combines a frozen large vision-language model backbone with a lightweight **conditional flow matching** action head, trained end-to-end on robot manipulation demonstrations.

The key idea: rather than predicting actions directly (MLP) or via a long denoising chain (diffusion), flow matching learns a straight-line optimal transport path from Gaussian noise to the action distribution. Fewer function evaluations, more stable training, cleaner theory.

```
Image + Language instruction
         │
         ▼
┌─────────────────────┐
│  PaliGemma-3B (4bit)│  ← frozen
│  Visual + Language  │
│  Encoder            │
└────────┬────────────┘
         │ conditioning embedding (4096-dim)
         ▼
┌─────────────────────┐
│  Flow Matching Head │  ← trained
│                     │
│  xₜ = (1-t)x₀+tx₁  │
│  v_θ(xₜ, t, c)      │
│                     │
│  Euler ODE → action │
└─────────────────────┘
         │
         ▼
    7-DoF action
  (Δpos, Δrot, gripper)
```

---

## Key results

> Results on **LIBERO-Spatial** benchmark (10 tasks, 20 eval episodes per task)

| Method | Success Rate | Inference (ms) |
|---|---|---|
| Random agent | 0.0% | — |
| MLP baseline | — | — |
| **Flow matching (NFE=5)** | — | ~5ms |
| **Flow matching (NFE=10)** | — | ~7.5ms |
| **Flow matching (NFE=20)** | — | ~14ms |

*Results being updated as training runs complete. See [W&B report](#) for live curves.*

---

## Architecture

### Frozen VLM backbone — `src/features/feature_engineering.py`

[PaliGemma-3B](https://huggingface.co/google/paligemma-3b-pt-224) loaded in 4-bit quantization via `bitsandbytes`, running on Apple MPS (M3). Provides rich visual and language understanding with zero gradient flow — all expressivity comes from the action head.

```python
# CLS token + mean-pooled instruction tokens → 4096-dim conditioning
cond = backbone.encode(images, instructions)  # (B, 4096)
```

### Conditional Flow Matching head — `src/models/flow_head.py`

The core contribution. A residual MLP that predicts the vector field $v_\theta(x_t, t, c)$ transporting noise → action along straight-line paths:

$$\mathcal{L}_\text{CFM} = \mathbb{E}_{t, x_0, x_1} \left[\| v_\theta((1-t)x_0 + tx_1,\ t,\ c) - (x_1 - x_0) \|^2 \right]$$

At inference, Euler integration solves the ODE in `NFE` steps:

$$x_{t+\Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t, c)$$

**Why flow matching over diffusion?**
- No noise schedule to tune
- Straight-line paths → fewer function evaluations needed
- More stable training (no score function estimation)
- Directly in line with π0.5's architecture choices

### EMA weights

Exponential moving average of the vector field network (decay=0.999) used at inference — stabilises action quality significantly, especially in early training.

---

## Project structure

```
vlm_agent/
├── src/
│   ├── config/           # Hydra configs
│   ├── data/
│   │   ├── ingestion.py       # LIBERO dataset download
│   │   ├── transformation.py  # Dataset, DataLoader, augmentation
│   │   └── validation.py      # Data quality checks
│   ├── entity/
│   │   └── config_entity.py   # Typed dataclasses (FlowConfig etc.)
│   ├── envs/
│   │   └── libero_wrapper.py  # LIBERO gym interface
│   ├── features/
│   │   └── feature_engineering.py  # Frozen PaliGemma backbone
│   ├── models/
│   │   ├── flow_head.py       # CFM loss + Euler sampler ← core
│   │   ├── train.py           # BC training loop
│   │   └── evaluate.py        # Rollout evaluator
│   └── pipelines/
│       ├── train.py           # End-to-end training pipeline
│       └── eval.py            # Evaluation pipeline
├── tests/
│   ├── unit/                  # Fast, no-env tests
│   └── integration/           # Full rollout tests
├── scripts/                   # Download, MPS check, ablations
├── data/raw/                  # LIBERO hdf5 demos
├── models/                    # Saved checkpoints
├── reports/                   # Ablation plots, figures
└── wandb/                     # Local W&B cache
```

---

## Quickstart

### 1. Install

```bash
# requires Python 3.11+, Apple Silicon (MPS) or CUDA
git clone https://github.com/oliverdippel/vlm_agent
cd vlm_agent

brew install uv
uv sync
```

### 2. Verify MPS + backbone

```bash
uv run python scripts/check_mps.py
# Expected: MPS working. Matrix mul result shape: torch.Size([1024, 1024])
```

### 3. Download LIBERO-Spatial data

```bash
uv run python scripts/download_data.py --suite libero_spatial --out data/raw/
# ~10GB download
```

### 4. Run tests

```bash
uv run pytest                  # all unit tests
uv run ruff check .            # linting
```

### 5. Train

```bash
# BC training with flow matching head
uv run python main.py

# NFE ablation sweep
uv run python main.py ablation.n_euler_steps=5
uv run python main.py ablation.n_euler_steps=10
uv run python main.py ablation.n_euler_steps=20
```

---

## Ablations

Three axes studied:

**1. Action head type** — does flow matching outperform a simple MLP?

**2. Number of function evaluations (NFE)** — speed vs quality trade-off on Apple MPS

**3. BC → RL fine-tuning** *(Phase 3, in progress)* — does PPO on top of a BC-initialised flow head improve success rate?

All ablations tracked in W&B. See `reports/` for figures.

---

## Roadmap

- [x] Phase 1 — Infrastructure, data pipeline, eval harness
- [x] Phase 2 — Flow matching action head + BC training
- [ ] Phase 3 — PPO fine-tuning on top of BC init
- [ ] Phase 4 — arXiv writeup + demo video

---

## Hardware

Developed and trained entirely on an **Apple M3 MacBook Pro** using MPS acceleration. PaliGemma-3B runs in 4-bit quantization (~6GB memory). Full BC training run: ~X hours.

---

## References

```bibtex
@article{black2024pi0,
  title={π0: A Vision-Language-Action Flow Model for General Robot Control},
  author={Black et al.},
  year={2024}
}

@article{lipman2022flow,
  title={Flow Matching for Generative Modeling},
  author={Lipman et al.},
  year={2022}
}

@article{liu2023libero,
  title={LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning},
  author={Liu et al.},
  year={2023}
}
```

---

## Citation

```bibtex
@misc{dippel2025minipizero,
  title={mini-π0: VLA Agent with Flow Matching Action Head},
  author={Oliver Dippel},
  year={2025},
  url={https://github.com/oliverdippel/vlm_agent}
}
```

---

<div align="center">
<sub>Built by <a href="https://oliverdippel.github.io">Oliver Dippel</a> · St Gallen, Switzerland</sub>
</div>