# Published reports

This directory contains the compact evidence behind the released models: training-data coverage, model selection on development sets, and acceptance on frozen test sets.

- `training/scene-coverage.json` counts screenshots, cards, identities, display modes, resolutions and visible field values of every full-screen set.
- `training/model-selection.json` records, for each of the seven models, its architecture, initialisation, training settings, selected checkpoint with its development metrics, deployment thresholds and weight hashes.
- `acceptance/holdout-full-screen.json` holds the stratified end-to-end results of `bdon_vision.evaluate_engine` on the two final-profile sets, scored after all models and thresholds were frozen.
- `acceptance/real-sample-regression.json` summarises the real-screenshot regression set without the images.

Per-screenshot traces, screenshots and game assets are not included. Scope limits and denominators are documented in [EVALUATION.md](../docs/EVALUATION.md), [DATA_CARD.md](../docs/DATA_CARD.md) and [TRAINING.md](../docs/TRAINING.md).
