#!/bin/bash
# Download everything the image test needs into one folder, and optionally run the test.
#
#   docker/downloadTest.sh <data folder> [--run]
#
#   IMAGE=cdeep3m-anywhere:gpu   image to use (default: cdeep3m-anywhere:gpu if it exists, else :cpu)
#   DEVICE=cpu|vulkan            device for the prediction step of the test (default cpu)
#
# What it puts into <data folder> (about 3 GB; steps that are already done are skipped):
#   models/SBEM_Mitochondria/{1fm,3fm,5fm}   pre-trained CDeep3M model (https://doi.org/10.7295/W9CDEEP3M50681)
#   mito_testsample/                         sample images and labels from the CDeep3M2 repository
#   train_data/                              augmented training data, made from the sample inside the image
#   image_test.sh                            the test itself (a copy of docker/image_test.sh)
#
# Downloads run on the host (curl or wget, git). The container needs no network.
set -e -o pipefail
[ $# -ge 1 ] || { sed -n '2,15p' "$0"; exit 1; }
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$1"; D=$(readlink -f "$1"); run=${2:-}
if [ ! -w "$D" ]; then
  echo "cannot write to $D (owner: $(stat -c %U "$D")). Docker creates a missing mount folder as root." >&2
  echo "Fix it with:  sudo chown -R $(id -un): $D   or use another folder." >&2
  exit 1
fi
MODEL_URL=https://cildata.crbs.ucsd.edu/media/model_display/50681/SBEM_Mitochondria.tar
SAMPLE_REPO=https://github.com/CRBS/cdeep3m2.git

if [ -z "$IMAGE" ]; then
  if docker image inspect cdeep3m-anywhere:gpu >/dev/null 2>&1; then IMAGE=cdeep3m-anywhere:gpu; else IMAGE=cdeep3m-anywhere:cpu; fi
fi
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "image $IMAGE not found; build it first (see docker/README.md)" >&2; exit 1; }
# GPU=auto|on|off as in docker/run.sh. "graphics" is needed for Vulkan (ncnn --device vulkan).
GPU=${GPU:-auto}; gpu=()
has_toolkit() { docker info 2>/dev/null | grep -qi 'runtimes:.*nvidia' || command -v nvidia-container-cli >/dev/null 2>&1; }
if [ "$GPU" = on ] || { [ "$GPU" = auto ] && nvidia-smi -L >/dev/null 2>&1 && has_toolkit; }; then
  gpu=(--gpus 'all,"capabilities=compute,utility,graphics"')
fi
echo "image $IMAGE, GPU: ${gpu[*]:-none (set GPU=on to force)}"
in_image() { docker run --rm "${gpu[@]}" --user "$(id -u):$(id -g)" --shm-size=2g \
                 -e DEVICE="${DEVICE:-cpu}" -v "$D":/data -v "$D/mito_testsample":/sample:ro "$IMAGE" "$@"; }

echo "== 1. model"
if [ -f "$D/models/SBEM_Mitochondria/1fm/deploy.prototxt" ]; then echo "already there"; else
  tmp="$D/.model_tmp"; rm -rf "$tmp"; mkdir -p "$tmp"
  if command -v curl >/dev/null; then curl -fL --retry 3 -o "$tmp/model.tar" "$MODEL_URL"
  else wget -O "$tmp/model.tar" "$MODEL_URL"; fi
  tar xf "$tmp/model.tar" -C "$tmp"
  # the folder that holds 1fm/ 3fm/ 5fm/, wherever it is inside the archive
  src=$(dirname "$(dirname "$(find "$tmp" -path '*/1fm/deploy.prototxt' | head -n 1)")")
  [ -d "$src/1fm" ] || { echo "no 1fm/deploy.prototxt in the archive" >&2; exit 2; }
  mkdir -p "$D/models"; rm -rf "$D/models/SBEM_Mitochondria"; mv "$src" "$D/models/SBEM_Mitochondria"; rm -rf "$tmp"
fi
ls "$D/models/SBEM_Mitochondria"

echo "== 2. sample images"
if [ -d "$D/mito_testsample/training/images" ]; then echo "already there"; else
  tmp="$D/.sample_tmp"; rm -rf "$tmp"
  git clone --quiet --depth 1 "$SAMPLE_REPO" "$tmp"
  rm -rf "$D/mito_testsample"; mv "$tmp/mito_testsample" "$D/mito_testsample"; rm -rf "$tmp"
fi
ls "$D/mito_testsample"

echo "== 3. training data (augmentation of the sample, a few minutes)"
if ls "$D"/train_data/*.h5 >/dev/null 2>&1; then echo "already there"; else
  mkdir -p "$D/train_data"
  in_image bash -c 'cd "$CDEEP3M2" && python PreprocessTrainingData.py /sample/training/images /sample/training/labels /data/train_data' | tail -n 6
fi
du -sh "$D/train_data"

cp "$here/image_test.sh" "$D/image_test.sh"
cmd="docker run --rm ${gpu[*]} --user \"\$(id -u):\$(id -g)\" --shm-size=2g -e DEVICE=${DEVICE:-cpu} -v $D:/data -v $D/mito_testsample:/sample:ro $IMAGE bash /data/image_test.sh image_test"
if [ "$run" = "--run" ]; then
  echo "== 4. test in $IMAGE"
  in_image bash /data/image_test.sh image_test
else
  echo; echo "Ready. Run the test with:"; echo "  $cmd"; echo "or: $0 $1 --run"
fi
