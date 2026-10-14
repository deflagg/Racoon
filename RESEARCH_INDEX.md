# Racoon Research Index

This repository contains multiple experimental lines. Keep them isolated so architecture changes, training objectives, and conclusions do not get mixed together.

## Core model

The repository root contains the original small token-based Raccoon Transformer and its associated training code.

## Research tracks

### `experiments/`
Earlier token-sequence experiments using the frozen 244-parameter Raccoon baseline, including context-dependent learning, data-curation, and information-efficiency studies.

### `visual_prediction/`
Pure visual predictive-learning experiments.

Current question:

> If Raccoon only watches raw visual sequences and learns to predict the next frame, what reusable internal structure emerges on its own?

This track uses a visual Transformer-style predictor rather than the original 244-parameter token model.

## Organization convention for new work

Each substantially different research direction should get its own top-level folder.

Within a track, prefer:

- `README.md` — research question, motivation, and current interpretation
- `MODEL_SPEC.md` — architecture and training objective
- `*.py` — executable experiments
- `results/` or dated result files — measurements only
- `notes/` — exploratory observations that are not yet established results

Do not silently overwrite old experiments when the architecture, objective, dataset, or evaluation changes materially. Create a new run or subfolder instead.

When an experiment was run in a transient environment but its exact source is unavailable, record only preserved facts and explicitly mark missing implementation details rather than reconstructing them from memory.
