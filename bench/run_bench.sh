#!/bin/bash
# Benchmark the 1fm model through the full CDeep3M2 pipeline with different engines.
#
# Required environment:
#   REPO         CDeep3M2 checkout with patches/cdeep3m2-*.patch applied and
#                ncnn_predict.py copied in (see README)
#   MODEL_DIR    trained model dir containing 1fm/{deploy.prototxt,trainedmodel,ncnn}
#   TEST_DIR     folder of test images (speed runs)
#   VAL_DIR      folder of labelled validation images (accuracy runs)
#   OUT          output folder (created)
# Optional:
#   CAFFE_PATH   CPU-only build of the caffe fork (needed unless SKIP_CAFFE=1)
#   SKIP_CAFFE=1 skip the Caffe runs
#   SKIP_VULKAN=1 skip the Vulkan run
# python3 on PATH must have numpy, h5py, pillow, opencv, scikit-image, joblib, ncnn.
for v in REPO MODEL_DIR TEST_DIR VAL_DIR OUT; do
  [ -n "${!v}" ] || { echo "set $v (see header of $0)"; exit 1; }
done
mkdir -p "$OUT"; RES=$OUT/timings.tsv
echo -e "run\tdataset\tbackend\tdevice\taugspeed\twall_s\tstatus" > "$RES"

run() {  # name dataset backend device augspeed [env assignments...]
  local name=$1 data=$2 backend=$3 device=$4 speed=$5; shift 5
  local out=$OUT/$name
  rm -rf "$out"
  local t0=$(date +%s.%N)
  env "$@" "$REPO"/runprediction.sh --models 1fm --denoise 0 --augspeed "$speed" \
      --backend "$backend" --device "$device" "$MODEL_DIR" "$data" "$out" > "$OUT/$name.log" 2>&1
  while [ ! -f "$out/DONE" ] && [ ! -f "$out/ERROR" ]; do sleep 2; done
  local t1=$(date +%s.%N)
  local st=ok; [ -f "$out/ERROR" ] && st=ERROR
  local ds=validation; [ "$data" == "$TEST_DIR" ] && ds=testset
  printf "%s\t%s\t%s\t%s\t%s\t%.1f\t%s\n" "$name" $ds "$backend" "$device" "$speed" \
      "$(echo "$t1 - $t0" | bc)" "$st" >> "$RES"
  while pgrep -f 'predictworker|preprocessworker|postprocessworker|predict_seg_new|ncnn_predict' > /dev/null; do sleep 2; done
}

# 1) one augmentation variant, engine comparison on the test images
[ -z "$SKIP_CAFFE" ]  && run s_caffe_cpu   "$TEST_DIR" caffe cpu    10 OPENBLAS_NUM_THREADS=8
run s_ncnn_cpu    "$TEST_DIR" ncnn  cpu    10 X=1
[ -z "$SKIP_VULKAN" ] && run s_ncnn_vulkan "$TEST_DIR" ncnn  vulkan 10 X=1
# 2) accuracy on labelled validation images, one variant
[ -z "$SKIP_CAFFE" ]  && run v_caffe_cpu   "$VAL_DIR"  caffe cpu    10 OPENBLAS_NUM_THREADS=8
run v_ncnn_cpu    "$VAL_DIR"  ncnn  cpu    10 X=1
# 3) full quality: 8 variants for 1fm at --augspeed 1
run f_ncnn_cpu    "$TEST_DIR" ncnn  cpu    1  X=1
run fv_ncnn_cpu   "$VAL_DIR"  ncnn  cpu    1  X=1
[ -z "$SKIP_CAFFE" ]  && run f_caffe_cpu   "$TEST_DIR" caffe cpu    1  OPENBLAS_NUM_THREADS=8
echo done > "$OUT/ALLDONE"
echo "now: python3 analyze.py $OUT   (VAL_LABELS=<labels dir> must be set)"
