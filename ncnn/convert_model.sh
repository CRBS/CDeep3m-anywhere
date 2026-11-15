#!/bin/bash
# Convert a CDeep3M (Caffe) 1fm/3fm/5fm model to ncnn, optionally with int8 quantization.
#
#   convert_model.sh <deploy.prototxt> <model.caffemodel> <out_dir> [calibration_list.txt]
#
# Writes <out_dir>/model.param and model.bin (fp32). With a calibration list
# (made by make_calibration.py) it also writes <out_dir>/int8/model.param and model.bin.
# Put <out_dir> at <trained model dir>/<1fm>/ncnn/ so that `--backend ncnn` finds it.
#
# 3fm/5fm also need  export CAFFE_PROTO=<caffe fork>/src/caffe/proto/caffe.proto  and python protobuf.
# Needs an ncnn build with tools: cmake -DNCNN_BUILD_TOOLS=ON, then
#   export NCNN_BUILD=/path/to/ncnn/build
# (caffe2ncnn, ncnn2table and ncnn2int8 are looked up there or on PATH).
set -e
[ $# -eq 3 ] || [ $# -eq 4 ] || { sed -n '2,12p' "$0"; exit 1; }
here=$(cd "$(dirname "$0")" && pwd)
find_tool() {  # name relative-path-in-build
  if [ -n "$NCNN_BUILD" ] && [ -x "$NCNN_BUILD/$2" ]; then echo "$NCNN_BUILD/$2"; return; fi
  command -v "$1" || { echo "$1 not found; set NCNN_BUILD" >&2; exit 2; }
}
c2n=$(find_tool caffe2ncnn tools/caffe/caffe2ncnn)
out=$3
mkdir -p "$out"
# Multi-frame models (3fm/5fm, input depth > 1) need their two 3D layers rewritten as 2D first;
# the 1-frame model only needs the depth-1 dimension dropped.
frames=$(python3 - "$1" <<'PY'
import re, sys
t = " ".join(l for l in open(sys.argv[1]).read().splitlines() if not l.lstrip().startswith("#"))
print(re.search(r"input_shape\s*\{([^}]*)\}", t).group(1).replace("dim:", " ").split()[-1])
PY
)
if [ "$frames" -gt 1 ]; then
  [ -n "$CAFFE_PROTO" ] || { echo "multi-frame model: set CAFFE_PROTO to caffe.proto of the CDeep3M Caffe fork" >&2; exit 2; }
  python3 "$here/convert_multiframe.py" "$1" "$2" "$out" --proto "$CAFFE_PROTO"
  weights="$out/model2d.caffemodel"
else
  python3 "$here/sanitize_prototxt.py" "$1" "$out/deploy2d.prototxt"
  weights="$2"
fi
"$c2n" "$out/deploy2d.prototxt" "$weights" "$out/model.param" "$out/model.bin"

# caffe2ncnn does not write an Input layer for 'input_shape'. The quantization tool needs
# one, so add it, taking the size from the sanitized prototxt.
python3 - "$out" <<'PY'
import re, sys
out = sys.argv[1]
dims = [int(x) for x in re.search(r"input_shape\s*\{([^}]*)\}", open(out + "/deploy2d.prototxt").read()).group(1).replace("dim:", " ").split()]
n, c, h, w = dims
lines = open(out + "/model.param").read().splitlines()
if not any(l.split()[:1] == ["Input"] for l in lines[2:]):
    layers, blobs = lines[1].split()
    lines[1] = "%d %s" % (int(layers) + 1, blobs)
    lines.insert(2, "Input            data             0 1 data 0=%d 1=%d 2=%d" % (w, h, c))
    open(out + "/model.param", "w").write("\n".join(lines) + "\n")
PY
echo "wrote $out/model.param and $out/model.bin (fp32)"

if [ $# -eq 4 ]; then
  t2t=$(find_tool ncnn2table tools/quantize/ncnn2table)
  t2i=$(find_tool ncnn2int8 tools/quantize/ncnn2int8)
  read w h c < <(awk 'NR==3{print $(NF-2), $(NF-1), $NF}' "$out/model.param" | sed 's/[0-9]=//g')
  "$t2t" "$out/model.param" "$out/model.bin" "$4" "$out/int8.table" \
        "shape=[$w,$h,$c]" thread=$(nproc 2>/dev/null || sysctl -n hw.ncpu) method=kl type=1
  mkdir -p "$out/int8"
  "$t2i" "$out/model.param" "$out/model.bin" "$out/int8/model.param" "$out/int8/model.bin" "$out/int8.table"
  echo "wrote $out/int8/model.param and model.bin (int8)"
fi
