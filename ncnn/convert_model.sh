#!/bin/bash
# Convert a CDeep3M (Caffe) 1-frame model to ncnn.
#
#   convert_model.sh <deploy.prototxt> <model.caffemodel> <out_dir>
#
# Writes <out_dir>/model.param and <out_dir>/model.bin. Put <out_dir> at
# <trained model dir>/<1fm>/ncnn/ so that `--backend ncnn` finds it.
# Needs caffe2ncnn from an ncnn build (cmake -DNCNN_BUILD_TOOLS=ON); set
# CAFFE2NCNN=/path/to/caffe2ncnn or put it on PATH.
set -e
[ $# -eq 3 ] || { sed -n '2,9p' "$0"; exit 1; }
here=$(cd "$(dirname "$0")" && pwd)
c2n=${CAFFE2NCNN:-$(command -v caffe2ncnn || true)}
[ -x "$c2n" ] || { echo "caffe2ncnn not found; set CAFFE2NCNN"; exit 2; }
mkdir -p "$3"
python3 "$here/sanitize_prototxt.py" "$1" "$3/deploy2d.prototxt"
"$c2n" "$3/deploy2d.prototxt" "$2" "$3/model.param" "$3/model.bin"
echo "wrote $3/model.param and $3/model.bin"
