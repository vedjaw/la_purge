#!/usr/bin/env bash
# Train base ResNet-18 on CIFAR-10 (no unlearning). Run once before unlearning.
set -euo pipefail
cd "$(dirname "$0")/.."

python run.py \
  --dataset cifar10 --model resnet18 \
  --forget_type class --forget_class 0 \
  --data_dir ./data --save_dir ./checkpoints \
  --epochs 0
