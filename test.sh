#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-/home/anaconda3/envs/pytorch/bin/python}"
data_root="${DATA_ROOT:-/path/to/datasets/EIVR/}"
dataset_tag="$(basename -- "${data_root%/}")"
output_root="${OUTPUT_ROOT:-$project_dir/results/${dataset_tag}_eivd_test}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

cd "$project_dir"
exec "$python_bin" test.py \
  --data-root "$data_root" \
  --split test \
  --dataset-name infra_h5 \
  --dataset-module data.infra_h5 \
  --dataset-class InfraH5EventDataset \
  --checkpoint "$project_dir/weights/model_best_eivd.pth.tar" \
  --checkpoint-key state_dict \
  --model EIVD \
  --n-features 16 \
  --n-blocks 15 \
  --activation gelu \
  --frames 8 \
  --past-frames 2 \
  --future-frames 2 \
  --event-window-size 1000 \
  --target-index 2 \
  --output-root "$output_root" \
  "$@"
