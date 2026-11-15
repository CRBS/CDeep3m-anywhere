#!/bin/bash
# Compare a model trained with the PyTorch trainer against a model trained with Caffe/CUDA (e.g. on Expanse),
# both 1fm and trained with the same staged recipe. Both models go through the SAME prediction:
# ncnn, CPU, int8, same images, same augspeed, same threshold. Then the predictions are compared pixel by pixel.
#
#   runComparisionExpanse500Itr.sh <pytorch model> <caffe model> <images dir> <labels dir> <out dir>
#
#   <pytorch model>  a trainer checkpoint (latest.pt / ckpt_*.pt) or an already exported .caffemodel
#   <caffe model>    the .caffemodel trained with Caffe (e.g. 1fm_classifer_iter_5000.caffemodel)
#   <labels dir>     label PNGs of the images, or "none"
#
# Environment (all required unless a default is given):
#   CDEEP3M2    patched CDeep3M2 checkout (with --backend ncnn)
#   NCNN_BUILD  ncnn build dir with the tools
#   CAFFE_PROTO caffe.proto of the CDeep3M Caffe fork
#   DEPLOY      1fm deploy.prototxt
#   TEMPLATE    any original 1fm .caffemodel (layout template for the export)   [only for a .pt input]
#   TRAIN_DATA  augmented training data folder (BatchNorm re-estimation)         [only for a .pt input]
#   CALIB_IMAGES  images for the int8 calibration (default: <images dir>; use training images)
#   AUGSPEED    default 1 (8 variants, the CDeep3M default); 10 = 1 variant, fastest
#   THRESHOLD   default 127
#   NAME_A / NAME_B   default PyTorch / CaffeCUDA
set -e
[ $# -eq 5 ] || { sed -n '2,22p' "$0"; exit 1; }
A=$(readlink -f "$1"); B=$(readlink -f "$2"); IMAGES=$(readlink -f "$3"); LABELS=$4; OUT=$5
for v in CDEEP3M2 NCNN_BUILD CAFFE_PROTO DEPLOY; do [ -n "${!v}" ] || { echo "set $v" >&2; exit 2; }; done
export NCNN_BUILD CAFFE_PROTO
AUGSPEED=${AUGSPEED:-1}; THRESHOLD=${THRESHOLD:-127}; NAME_A=${NAME_A:-PyTorch}; NAME_B=${NAME_B:-CaffeCUDA}
CALIB_IMAGES=${CALIB_IMAGES:-$IMAGES}
repo=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"; OUT=$(readlink -f "$OUT")

# int8 calibration data: made once, used for both models
if [ ! -f "$OUT/calib/list.txt" ]; then
  python3 "$repo/ncnn/make_calibration.py" --repo "$CDEEP3M2" --images "$CALIB_IMAGES" --out "$OUT/calib" --model 1fm
fi

run_one() {  # name  model-file
  local name=$1 src=$2 m="$OUT/$1/model/1fm" pred="$OUT/$1/pred"
  if [ -f "$pred/ensembled/DONE" ]; then echo "[$name] prediction exists, reusing it"; return; fi
  mkdir -p "$m/trainedmodel"
  cp "$DEPLOY" "$m/deploy.prototxt"
  if [[ "$src" == *.pt ]]; then
    for v in TEMPLATE TRAIN_DATA; do [ -n "${!v}" ] || { echo "set $v (needed to export a .pt)" >&2; exit 2; }; done
    (cd "$repo" && python3 -m trainer.export_caffemodel --deploy "$DEPLOY" --checkpoint "$src" --template "$TEMPLATE" \
        --out "$m/trainedmodel/1fm_classifer_iter_1.caffemodel" --data "$TRAIN_DATA" --recalibrate 40)
  else
    cp "$src" "$m/trainedmodel/1fm_classifer_iter_1.caffemodel"
  fi
  "$repo/ncnn/convert_model.sh" "$DEPLOY" "$m/trainedmodel/1fm_classifer_iter_1.caffemodel" "$m/ncnn" "$OUT/calib/list.txt"
  [ -f "$m/ncnn/int8/model.bin" ] || { echo "[$name] int8 model was not written" >&2; exit 3; }
  rm -rf "$pred"
  (cd "$CDEEP3M2" && NCNN_PRECISION=int8 ./runprediction.sh --models 1fm --denoise 0 --augspeed "$AUGSPEED" \
        --backend ncnn --device cpu "$OUT/$name/model" "$IMAGES" "$pred" > "$OUT/$name/pred.log" 2>&1) || true
  while [ ! -f "$pred/DONE" ] && [ ! -f "$pred/ERROR" ]; do sleep 2; done
  [ -f "$pred/DONE" ] || { echo "[$name] prediction failed, see $OUT/$name/pred.log" >&2; exit 4; }
  while [ ! -f "$pred/ensembled/DONE" ]; do sleep 1; done
  echo "[$name] prediction done: $pred/ensembled"
}

run_one "$NAME_A" "$A"
run_one "$NAME_B" "$B"

lab=(); [ "$LABELS" = none ] || lab=(--labels "$(readlink -f "$LABELS")")
python3 "$repo/tools/compare_predictions.py" --a "$OUT/$NAME_A/pred/ensembled" --b "$OUT/$NAME_B/pred/ensembled" \
    --out "$OUT/comparison" --images "$IMAGES" "${lab[@]}" --name-a "$NAME_A" --name-b "$NAME_B" --threshold "$THRESHOLD"
echo "report: $OUT/comparison/summary.txt, per_image.tsv and side_by_side/"
