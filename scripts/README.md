# Training Scripts

## train.py

Main training script. Usage:

```bash
python scripts/train.py --config configs/experiment.yaml [--resume path/to/checkpoint.pt]
```

## train_phase1.py

Phase 1 training (Network G only).

## train_phase2.py

Phase 2 training (Network D only, G frozen).

## train_phase3.py

Phase 3 joint fine-tuning (G + D + R).

## interactive_training.py

Interactive training with live visualization.

## Example:

```bash
# Phase 1
python scripts/train.py --config configs/exp1.yaml --phase 1 --epochs 500

# Phase 2
python scripts/train.py --config configs/exp1.yaml --phase 2 --epochs 200

# Phase 3
python scripts/train.py --config configs/exp1.yaml --phase 3 --epochs 100
```
