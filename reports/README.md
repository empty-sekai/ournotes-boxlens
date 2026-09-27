# Published reports

This directory contains the compact evidence needed to understand the released training coverage, final acceptance, and CPU performance. Reports are grouped by their purpose rather than by internal experiment version.

- `training/` records card/rank coverage, level and rank sample counts, hard-field learning curves, and the selected checkpoints.
- `acceptance/` records the clear rank matrix, stratified full-screen synthetic results, data audit, open-set check, and a sanitized real-sample regression summary.
- `performance/` contains each measured CPU benchmark run.

The repository omits intermediate smoke outputs, superseded per-run reports, per-screenshot traces, screenshots, game assets, and private calibration details. The full-screen aggregate preserves counts and test strata; its raw source report SHA-256 values remain in that summary. Scope limits and denominators are documented in [EVALUATION.md](../docs/EVALUATION.md), [DATA_CARD.md](../docs/DATA_CARD.md), and [TRAINING.md](../docs/TRAINING.md).
