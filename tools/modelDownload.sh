#!/bin/bash
# Download the pretrained SBEM mitochondria model and the CDeep3M2 sample training/validation/test images.
#
#   tools/modelDownload.sh [--model-only | --data-only] [dest dir]      (default dest: ./downloads)
#
# Result:
#   <dest>/SBEM_Mitochondria/{1fm,3fm,5fm}/...      pretrained model (DOI 10.7295/W9CDEEP3M50681, ~685 MB)
#   <dest>/mito_testsample/training/{images,labels}  20 labelled training images
#   <dest>/mito_testsample/validation/{images,labels} 10 labelled validation images
#   <dest>/mito_testsample/testset/                  11 unlabelled test images
# Already downloaded parts are skipped; an interrupted model download resumes.
set -euo pipefail

MODEL_URL=https://cildata.crbs.ucsd.edu/media/model_display/50681/SBEM_Mitochondria.tar
MODEL_SIZE=684759040
CDEEP3M2_COMMIT=67d6308          # same commit as patches/cdeep3m2-67d6308-device-backend.patch
DATA_URL=https://codeload.github.com/CRBS/cdeep3m2/tar.gz/$CDEEP3M2_COMMIT

do_model=1; do_data=1
case "${1:-}" in
  --model-only) do_data=0; shift ;;
  --data-only)  do_model=0; shift ;;
  -h|--help)    sed -n '2,12p' "$0"; exit 0 ;;
esac
dest=${1:-downloads}
mkdir -p "$dest"
dest=$(cd "$dest" && pwd)
command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }

if [ $do_model = 1 ]; then
  if [ -f "$dest/SBEM_Mitochondria/1fm/deploy.prototxt" ]; then
    echo "model: already in $dest/SBEM_Mitochondria, skipped"
  else
    tar_file=$dest/SBEM_Mitochondria.tar
    echo "model: downloading $MODEL_URL (~685 MB)"
    curl -fL --retry 5 -C - -o "$tar_file" "$MODEL_URL"
    got=$(stat -c %s "$tar_file")
    [ "$got" = "$MODEL_SIZE" ] || { echo "model: size $got, expected $MODEL_SIZE; delete $tar_file and rerun" >&2; exit 1; }
    echo "model: extracting"
    tar -xf "$tar_file" -C "$dest"
    rm -f "$tar_file"
    ls -d "$dest"/SBEM_Mitochondria/*fm
  fi
fi

if [ $do_data = 1 ]; then
  if [ -d "$dest/mito_testsample/training/images" ]; then
    echo "data: already in $dest/mito_testsample, skipped"
  else
    echo "data: downloading mito_testsample from CDeep3M2 at $CDEEP3M2_COMMIT"
    tmp=$(mktemp -d "$dest/.dl.XXXXXX"); trap 'rm -rf "$tmp"' EXIT
    curl -fL --retry 5 "$DATA_URL" | tar -xz -C "$tmp" --wildcards '*/mito_testsample/*'
    mv "$tmp"/*/mito_testsample "$dest/"
    for d in training/images training/labels validation/images validation/labels testset; do
      printf '  %-20s %s files\n' "$d" "$(find "$dest/mito_testsample/$d" -maxdepth 1 -type f | wc -l)"
    done
  fi
fi

cat <<EOF

Done. Next steps (see README.md and docs/TRAINING.md):
  # training data -> augmented HDF5 (run inside the patched cdeep3m2 checkout)
  python3 PreprocessTrainingData.py $dest/mito_testsample/training/images $dest/mito_testsample/training/labels 0 $dest/train_data
  # convert the 1fm model for ncnn
  ncnn/convert_model.sh $dest/SBEM_Mitochondria/1fm/deploy.prototxt $dest/SBEM_Mitochondria/1fm/trainedmodel/*.caffemodel $dest/SBEM_Mitochondria/1fm/ncnn
EOF
