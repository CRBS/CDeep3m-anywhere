#!/bin/bash
# Start (or resume) a training run in the background; log goes to <out>/train.log.
# Run the same command again after a restart and it continues from <out>/latest.pt.
#   tools/train_bg.sh <1fm|3fm|5fm> <deploy.prototxt> <train_data dir> <out dir> [extra trainer args, e.g. --weights x.caffemodel --max-iter 3000]
# Needs: python with torch/numpy/h5py (e.g. the venv), CAFFE_PROTO if you pass --weights.
[ $# -ge 4 ] || { sed -n '2,6p' "$0"; exit 1; }
model=$1; deploy=$2; data=$3; out=$4; shift 4
here=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$out"
cd "$here" || exit 1
nohup python3 -m trainer.train --model "$model" --deploy "$deploy" --data "$data" --out "$out" \
      --progress-dir "$here/progress" "$@" >> "$out/train.log" 2>&1 &
echo "training started in the background (pid $!), log: $out/train.log"
