#!/bin/bash
# Start (or resume) a training run in the background; log goes to <out>/train.log.
# Run the same command again after a restart and it continues from <out>/latest.pt.
#
#   tools/train_bg.sh [--torch auto|cuda|rocm|cpu] <1fm|3fm|5fm> <deploy.prototxt> <train_data dir> <out dir> [trainer args]
#
#   --torch   which PyTorch to train with (default auto):
#               cuda  NVIDIA GPU, CUDA build of PyTorch
#               rocm  AMD GPU, ROCm build of PyTorch
#               cpu   no GPU
#               auto  look at the hardware: NVIDIA -> cuda, AMD (ROCm capable) -> rocm, neither -> cpu.
#                     If it is unclear (both kinds of GPU, or a GPU that the installed PyTorch cannot use)
#                     the script asks; when it cannot ask (no terminal) it stops and tells you to pass --torch.
#   trainer args, e.g. --weights x.caffemodel --max-iter 3000 --gpu 1
#
# Python: one environment per PyTorch build can be given with CDEEP3M_PY_CUDA / CDEEP3M_PY_ROCM / CDEEP3M_PY_CPU
# (path of that environment's python). Without them `python3` on PATH is used for every choice.
# Needs CAFFE_PROTO if you pass --weights.
want=auto
if [ "$1" = "--torch" ]; then want=$2; shift 2; fi
case "$want" in auto|cuda|rocm|cpu) ;; *) echo "--torch must be auto, cuda, rocm or cpu" >&2; exit 1 ;; esac
[ $# -ge 4 ] || { sed -n '2,19p' "$0"; exit 1; }
model=$1; deploy=$2; data=$3; out=$4; shift 4
here=$(cd "$(dirname "$0")/.." && pwd)

py_for() {  # python of the environment for a PyTorch choice
  case "$1" in
    cuda) echo "${CDEEP3M_PY_CUDA:-python3}" ;;
    rocm) echo "${CDEEP3M_PY_ROCM:-python3}" ;;
    *)    echo "${CDEEP3M_PY_CPU:-python3}" ;;
  esac
}
torch_kind() {  # what the PyTorch of this python can do right now: cuda, rocm, cpu or none (no torch)
  "$1" - <<'PY' 2>/dev/null || echo none
import torch
if torch.cuda.is_available():
    print("rocm" if getattr(torch.version, "hip", None) else "cuda")
else:
    print("cpu")
PY
}
ask() {  # question, allowed answers...  -> prints the answer; fails when there is no terminal
  local q=$1 a; shift
  [ -t 0 ] || return 1
  while true; do
    read -r -p "$q [$(IFS=/; echo "$*")]: " a
    for x in "$@"; do [ "$a" = "$x" ] && { echo "$a"; return 0; }; done
  done
}

# ---- hardware
has_nvidia=no; has_amd=no
if command -v nvidia-smi >/dev/null 2>&1 && [ -n "$(nvidia-smi -L 2>/dev/null | grep -i '^GPU')" ]; then has_nvidia=yes; fi
if [ -e /dev/kfd ] && grep -qi '^0x1002' /sys/class/drm/card*/device/vendor 2>/dev/null; then has_amd=yes; fi
ncpu=$(nproc 2>/dev/null || echo "?")
echo "hardware: NVIDIA GPU $has_nvidia, AMD GPU (ROCm) $has_amd, $ncpu CPU threads"

# ---- choose
choice=$want
if [ "$want" = auto ]; then
  if [ $has_nvidia = yes ] && [ $has_amd = yes ]; then
    choice=$(ask "Both an NVIDIA and an AMD GPU were found. Which PyTorch should train?" cuda rocm cpu) \
      || { echo "unclear which GPU to use: pass --torch cuda, rocm or cpu" >&2; exit 2; }
  elif [ $has_nvidia = yes ]; then choice=cuda
  elif [ $has_amd = yes ]; then choice=rocm
  else choice=cpu
  fi
fi
py=$(py_for "$choice")
kind=$(torch_kind "$py")
[ "$kind" != none ] || { echo "no usable PyTorch in '$py' (for --torch $choice)" >&2; exit 2; }
if [ "$choice" != cpu ] && [ "$kind" != "$choice" ]; then
  msg="--torch $choice was chosen, but the PyTorch in '$py' can only use: $kind."
  if [ "$want" = auto ] && [ "$kind" = cpu ]; then
    ans=$(ask "$msg Train on the CPU instead?" cpu abort) \
      || { echo "$msg Install the matching PyTorch build or pass --torch cpu." >&2; exit 2; }
    [ "$ans" = cpu ] || exit 2
    choice=cpu; py=$(py_for cpu)
  else
    echo "$msg Install the matching PyTorch build or pass another --torch." >&2; exit 2
  fi
fi
device=cuda; [ "$choice" = cpu ] && device=cpu     # the ROCm build also calls its device "cuda"
echo "training with PyTorch: $choice (python: $py, --device $device)"

mkdir -p "$out"
# small progress files for git (tools/autopush.sh); skipped when the folder is not writable (e.g. in a container)
prog=${CDEEP3M_PROGRESS_DIR:-$here/progress}; pargs=()
if mkdir -p "$prog" 2>/dev/null && [ -w "$prog" ]; then pargs=(--progress-dir "$prog"); fi
cd "$here" || exit 1
nohup "$py" -m trainer.train --model "$model" --deploy "$deploy" --data "$data" --out "$out" \
      "${pargs[@]}" --device "$device" "$@" >> "$out/train.log" 2>&1 &
echo "training started in the background (pid $!), log: $out/train.log"
