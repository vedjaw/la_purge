#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python experiments/train_groupnorm_base.py "$@"
