# Visual Predictor Model Specification

## Known and fixed from the recorded experiment

- Input: 4 grayscale frames
- Output target: frame 5 pixels
- Learning mode: passive visual observation only
- Architecture family: standard Transformer-style predictor
- Components: self-attention, MLP/feed-forward blocks, residual connections, layer normalization, prediction head
- Training supervision: next-frame pixels only
- No actions or interventions
- No semantic/object/motion labels used for training
- Four object identities
- Four motion rules
- Held-out object-motion combinations used for compositional transfer evaluation

## Unknown from the preserved transcript

The exact transient run's following values were not preserved and must not be guessed:

- parameter count
- model width / embedding dimension
- number of Transformer layers
- number of attention heads
- frame resolution
- optimizer details
- learning rate
- batch size
- exact epoch/update schedule
- exact random seeds

The recorded research results are preserved separately, but the original transient script is not currently recoverable from this repository or the present runtime.
