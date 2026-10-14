# Visual Prediction Research Branch

This folder isolates the pure-visual predictive-learning experiments from the repository's original 244-parameter token Raccoon experiments.

## Research question

If Raccoon only watches raw visual sequences and learns to predict the next frame, what reusable internal structure emerges on its own?

Training uses no object labels, motion labels, velocity targets, actions, or interventions.

## Task

The model observes four grayscale frames and predicts the fifth:

```
frame 1 -> frame 2 -> frame 3 -> frame 4 -> predict frame 5
```

The synthetic world contains four visually distinct object identities and four motion rules. Some object-motion combinations are withheld completely during training, so evaluation can test whether the model recombines learned factors rather than memorizing exact combinations.

## Architecture family

This experiment uses a small standard Transformer-style visual predictor with the usual ingredients:

- self-attention
- feed-forward/MLP blocks
- residual connections
- layer normalization
- a next-frame prediction head

It is not a JEPA objective: the training target is the next frame's pixels.

This is intentionally separate from the repository's original 244-parameter token Raccoon model.

### Reproducibility note

The transient source file that produced the recorded run was created in the prior research session but is not available in the current runtime. Therefore this folder records only facts that were preserved in the experiment transcript and does **not** invent missing width/head/layer/hyperparameter values. A future executable reimplementation should be committed only after those choices are explicitly fixed and rerun.

## Recorded result

The most important observation was that next-frame prediction produced a strongly transferable representation of future position, including on held-out object-motion combinations, while object identity remained weakly abstracted.

See `results_recorded_2026-10-14.json` and `MODEL_SPEC.md`.
