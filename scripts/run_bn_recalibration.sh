#!/usr/bin/env bash
# Example: pass checkpoint paths for each method you have trained.
set -euo pipefail
cd "$(dirname "$0")/.."

python experiments/bn_recalibration.py \
  --forget_class 0 \
  --checkpoints \
    purge:checkpoints/purge_cifar10_resnet18_class_c0.pth \
  "$@"
