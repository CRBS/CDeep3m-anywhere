#!/bin/bash
# Compare ncnn fp32 and ncnn int8 (CPU) through the full CDeep3M2 pipeline, 1fm model.
#
# Required environment:
#   REPO        patched CDeep3M2 checkout (see README)
#   MODEL_DIR   trained model dir whose 1fm/ncnn/ holds model.* (fp32) and int8/model.*
#   TEST_DIR    folder of test images (speed runs)
#   VAL_DIR     folder of labelled validation images (accuracy runs)
#   OUT         output folder
# python3 on PATH needs the packages from the README. NCNN_PRECISION selects the model.
for v in REPO MODEL_DIR TEST_DIR VAL_DIR OUT; do [ -n "${!v}" ] || { echo "set $v (see header)"; exit 1; }; done
mkdir -p "$OUT"; RES=$OUT/timings.tsv
echo -e "run\tprecision\tdataset\taugspeed\twall_s" > "$RES"
run() {  # name precision dataset_dir dsname augspeed
  local out=$OUT/$1; rm -rf "$out"; local t0=$(date +%s.%N)
  NCNN_PRECISION=$2 "$REPO"/runprediction.sh --models 1fm --denoise 0 --augspeed $5 --backend ncnn --device cpu \
      "$MODEL_DIR" "$3" "$out" > "$OUT/$1.log" 2>&1
  while [ ! -f "$out/DONE" ] && [ ! -f "$out/ERROR" ]; do sleep 2; done
  local t1=$(date +%s.%N)
  printf "%s\t%s\t%s\t%s\t%.1f\n" "$1" "$2" "$4" "$5" "$(echo "$t1 - $t0" | bc)" >> "$RES"
  sleep 5   # let background workers finish
}
run val1_fp32  fp32 "$VAL_DIR"  validation 10
run val1_int8  int8 "$VAL_DIR"  validation 10
run val8_fp32  fp32 "$VAL_DIR"  validation 1
run val8_int8  int8 "$VAL_DIR"  validation 1
run test8_fp32  fp32 "$TEST_DIR" testset 1
run test8_int8  int8 "$TEST_DIR" testset 1
run test8_fp32b fp32 "$TEST_DIR" testset 1
run test8_int8b int8 "$TEST_DIR" testset 1
echo done > "$OUT/ALLDONE"
echo "now: VAL_LABELS=<labels dir> python3 analyze_int8.py $OUT"
