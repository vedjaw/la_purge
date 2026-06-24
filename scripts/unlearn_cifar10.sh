#!/usr/bin/env bash
# Recommended PURGE run: CIFAR-10 class 0, kl_retain objective.
set -euo pipefail
cd "$(dirname "$0")/.."

python run.py \
  --config configs/cifar10_kl_retain.yaml \
  --checkpoint ./checkpoints/base_cifar10_resnet18.pth \
  --data_dir ./data --save_dir ./checkpoints
