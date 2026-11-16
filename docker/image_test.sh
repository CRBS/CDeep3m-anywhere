#!/bin/bash
# Runs INSIDE the cdeep3m-anywhere image (as a normal user): convert (int8) -> predict -> train -> export.
# /data = the folder prepared by docker/downloadTest.sh, /sample = mito_testsample (read only).
# Argument: test folder name under /data.   DEVICE=cpu|vulkan  device for the prediction step (default cpu).
set -e
T=/data/$1; rm -rf "$T"; mkdir -p "$T/model/1fm/trainedmodel" "$T/calib_img" "$T/pred_img"
M=/data/models/SBEM_Mitochondria/1fm
echo "== environment"; id -u; python -c "import torch, ncnn, numpy; print('torch', torch.__version__, 'cuda available', torch.cuda.is_available(), '| ncnn', ncnn.__version__, '| numpy', numpy.__version__)"
cp $M/deploy.prototxt "$T/model/1fm/"; cp $M/trainedmodel/1fm_classifer_iter_80000.caffemodel "$T/model/1fm/trainedmodel/"
ls /sample/training/images | head -5 | while read f; do cp "/sample/training/images/$f" "$T/calib_img/"; done
ls /sample/validation/images | head -3 | while read f; do cp "/sample/validation/images/$f" "$T/pred_img/"; done

echo "== 1. conversion with int8"
make_calibration.py --repo "$CDEEP3M2" --images "$T/calib_img" --out "$T/calib" --model 1fm
convert_model.sh "$T/model/1fm/deploy.prototxt" "$T/model/1fm/trainedmodel/1fm_classifer_iter_80000.caffemodel" "$T/model/1fm/ncnn" "$T/calib/list.txt" > "$T/convert.log" 2>&1
ls -la "$T/model/1fm/ncnn/model.bin" "$T/model/1fm/ncnn/int8/model.bin"

DEVICE=${DEVICE:-cpu}
# int8 is for the CPU; on a GPU the full precision model is used
if [ "$DEVICE" = cpu ]; then export NCNN_PRECISION=int8; fi
echo "== 2. prediction (ncnn, $DEVICE${NCNN_PRECISION:+, $NCNN_PRECISION})"
cd "$CDEEP3M2"
SECONDS=0
./runprediction.sh --models 1fm --denoise 0 --augspeed 10 --backend ncnn --device "$DEVICE" "$T/model" "$T/pred_img" "$T/pred" > "$T/pred.log" 2>&1 || true
for i in $(seq 1 300); do [ -f "$T/pred/DONE" ] || [ -f "$T/pred/ERROR" ] && break; sleep 2; done
[ -f "$T/pred/DONE" ] || { echo "PREDICTION FAILED"; tail -20 "$T/pred.log"; exit 4; }
echo "prediction took ${SECONDS}s on $DEVICE"; unset NCNN_PRECISION
sleep 3; ls "$T/pred/ensembled/"
python - "$T" <<'PY'
import glob, sys, numpy as np
from PIL import Image
T = sys.argv[1]
p = [np.array(Image.open(f).convert("L")) > 127 for f in sorted(glob.glob(T + "/pred/ensembled/Segmented_*.png"))]
g = [np.array(Image.open(f).convert("L")) > 127 for f in sorted(glob.glob("/sample/validation/labels/*.png"))[:len(p)]]
tp = sum((a & b).sum() for a, b in zip(p, g)); s = sum(a.sum() + b.sum() for a, b in zip(p, g))
print("Dice vs labels on %d validation images: %.4f" % (len(p), 2.0 * tp / s))
PY

echo "== 3. training, 3 updates, --torch auto"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo "no NVIDIA GPU visible"
cd /data
train_bg.sh --torch auto 1fm $M/deploy.prototxt /data/train_data "$T/run" --weights $M/trainedmodel/1fm_classifer_iter_80000.caffemodel --max-iter 3 --iter-size 1 < /dev/null
for i in $(seq 1 200); do grep -q "^finished\|Traceback\|rror" "$T/run/train.log" 2>/dev/null && break; sleep 2; done
grep -v "Warning\|back_sum\|detach" "$T/run/train.log" | cut -c1-150

echo "== 4. export the trained checkpoint"
python -m trainer.export_caffemodel --deploy $M/deploy.prototxt --checkpoint "$T/run/latest.pt" \
   --template $M/trainedmodel/1fm_classifer_iter_80000.caffemodel --out "$T/trained.caffemodel" --data /data/train_data --recalibrate 5 2>&1 | grep -E "re-est|wrote|rror"
ls -la "$T/trained.caffemodel"

echo "== 5. conversion of a multi-frame model (3fm, fp32)"
M3=/data/models/SBEM_Mitochondria/3fm
convert_model.sh $M3/deploy.prototxt "$(ls $M3/trainedmodel/*.caffemodel | head -1)" "$T/ncnn3fm" > "$T/convert3fm.log" 2>&1 || { tail -5 "$T/convert3fm.log"; exit 5; }
ls -la "$T/ncnn3fm/model.bin"
echo "== ALL STEPS PASSED"
