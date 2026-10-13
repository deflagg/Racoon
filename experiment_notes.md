# Experiment notes

## Purpose

Raccoon is the smallest baseline in this project that we currently treat as a
qualitatively ordinary Transformer rather than a deliberately degenerate toy.

The immediate goal is not benchmark performance. It is to have a model small
enough that every parameter, activation, attention matrix, gradient, and
optimization trajectory can be inspected.

## Why Raccoon replaced Dolphin

An earlier width-2 model ("Dolphin") exposed a structural problem: LayerNorm at
width 2 removes almost all directional freedom. That makes the model
qualitatively unlike a normal Transformer in a way that is too strong for a
clean baseline.

Raccoon therefore uses width 4 while restoring the ordinary Transformer pieces:

- Q, K, V, and O projections
- causal self-attention
- residual connections
- pre-norm LayerNorm
- a two-layer MLP
- learned token and position embeddings
- final LayerNorm
- next-token logits from the model state

The output weights are tied to the token embedding in order to keep the full
baseline at exactly 244 trainable parameters.

## Context-dependent task

Tokens 0 and 1 are mode tokens.

Mode 0 traverses:

    2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 2

Mode 1 traverses:

    2 -> 7 -> 6 -> 5 -> 4 -> 3 -> 2

The current token alone is insufficient. For example, after token 4, the target
is 5 in mode 0 and 3 in mode 1.

The dataset contains every mode/start pair: 2 modes x 6 starting cycle tokens =
12 sequences.

### Important loss-mask detail

The starting cycle token is deliberately varied. Therefore the model cannot
infer the first cycle token from the mode token alone. The training harness
ignores the loss at sequence position 0 and evaluates only transitions among
cycle tokens, where context is well-defined and a perfect solution is possible.

This corrects an ambiguity in the original exploratory conversation, where the
spoken example described a fully shifted target sequence even though the first
target would be contradictory across different starting rotations.

## Historical exploratory observations

During the live exploratory session, we reported:

- 7 of 10 random seeds reached 100% on a context-dependent version of the task
  within 5,000 steps.
- the other three were reported around 77.8%, 90.3%, and 97.2%.
- changing only the mode token changed the learned prediction in the appropriate
  direction for the solved runs.
- ablating the attention contribution strongly reduced performance.

These are useful research notes, but they are **not yet repository-verified
measurements**. The exact original script was not preserved at the time those
numbers were spoken. They should therefore be treated as historical exploratory
claims, not as canonical results.

The code in this repository is the canonical baseline going forward. New
measurements should be generated from this code and committed with enough
metadata to reproduce them.

## Next experiment

Compare successful and stalled seeds from identical training conditions and log:

- loss and accuracy
- total and per-parameter gradient norms
- attention matrices
- Q/K/V/O projection norms
- LayerNorm scale and bias
- token and positional representations
- residual-stream activations
- pairwise representation geometry

The key question is: **what is the earliest measurable internal difference that
predicts whether a seed will converge cleanly?**
