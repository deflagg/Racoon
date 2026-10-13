# Racoon

A deliberately tiny but structurally faithful Transformer baseline for mechanistic experiments.

> **Name note:** the repository is named `Racoon` on GitHub. In the research conversation we called the model **Raccoon**. The code uses `Raccoon`.

## Frozen baseline architecture

- Vocabulary size: 8
- Maximum sequence length: 8
- Model width (`d_model`): 4
- Attention heads: 1
- Transformer blocks: 1
- Attention: full Q, K, V, and output projection O
- Pre-norm residual block
- MLP hidden width: 8
- MLP activation: GELU
- Learned token embeddings
- Learned positional embeddings
- Final LayerNorm
- Language-model output weights tied to token embeddings
- Causal attention mask

Total trainable parameters: **244**

### Parameter count

| Component | Parameters |
|---|---:|
| Token embedding, 8 x 4 | 32 |
| Position embedding, 8 x 4 | 32 |
| Q/K/V/O projections, each 4 x 4 + bias | 80 |
| Two pre-norm LayerNorms | 16 |
| MLP, 4 -> 8 -> 4 with biases | 76 |
| Final LayerNorm | 8 |
| LM head | 0 additional (weight tied to token embedding) |
| **Total** | **244** |

## Context-dependent task

The vocabulary is `0..7`.

Tokens `0` and `1` are mode/toggle tokens. Tokens `2..7` form a six-token cycle.

- Mode 0: `2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 2`
- Mode 1: `2 -> 7 -> 6 -> 5 -> 4 -> 3 -> 2`

The same current token has a different correct successor depending on the earlier mode token. For example:

- `0, 4, 5, 6, 7, 2, 3`
- `1, 4, 3, 2, 7, 6, 5`

So when the current token is `4`, the correct next token is `5` in mode 0 but `3` in mode 1. The task therefore cannot be solved perfectly from the current token alone.

### Loss-mask detail

The starting cycle token is deliberately varied across all six rotations. Therefore the token immediately after the mode token cannot be inferred from the mode alone. The training harness ignores the loss at that first position and evaluates the six cycle transitions, where the target is well-defined and perfect accuracy is possible.

This is important because an unmasked fully shifted target would create contradictory training labels at the mode position.

## Files

- `raccoon.py` — frozen 244-parameter model definition and parameter-count assertion.
- `train_context.py` — deterministic data generator plus a small training/evaluation harness.
- `experiment_notes.md` — what the exploratory session established, and what remains unproven.
- `requirements.txt` — minimal dependency list.

## Run

```bash
pip install -r requirements.txt
python raccoon.py
python train_context.py --seed 0 --steps 5000
```

To sweep seeds:

```bash
for s in 0 1 2 3 4 5 6 7 8 9; do
  python train_context.py --seed $s --steps 5000
done
```

## Research status

The **architecture** above is the frozen baseline we want to preserve.

The exploratory conversation previously reported that the context task was learnable and that ablating attention damaged performance. Those historical numbers are recorded in `experiment_notes.md`, but they should be treated as session notes until independently rerun from this repository. The training harness here is intended to make that rerun explicit and reproducible.

The next mechanistic question is not merely whether Raccoon can solve the task, but **why some initializations optimize more cleanly than others**: gradients, attention allocation, representation geometry, and parameter trajectories are natural observables.
