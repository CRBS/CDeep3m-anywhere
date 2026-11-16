#!/bin/bash
# Start the CDeep3M-anywhere container with a data folder mounted at /data.
#
#   docker/run.sh <data folder> [command ...]          (no command = interactive shell)
#
#   IMAGE=cdeep3m-anywhere:cpu   image to run (default)
#   GPU=auto|on|off              auto (default) passes the NVIDIA GPUs if the host has the NVIDIA container toolkit
#
# Files are written with your own user id, so results in the data folder belong to you.
set -e
[ $# -ge 1 ] || { sed -n '2,9p' "$0"; exit 1; }
data=$(readlink -f "$1"); shift
[ -d "$data" ] || { echo "not a folder: $data" >&2; exit 1; }
IMAGE=${IMAGE:-cdeep3m-anywhere:cpu}
GPU=${GPU:-auto}
gpu=()
if [ "$GPU" = on ] || { [ "$GPU" = auto ] && docker info 2>/dev/null | grep -qi 'runtimes:.*nvidia'; }; then
  gpu=(--gpus all)
fi
tty=(); [ -t 0 ] && tty=(-it)
exec docker run --rm "${tty[@]}" "${gpu[@]}" --user "$(id -u):$(id -g)" --shm-size=2g \
     -v "$data":/data -w /data "$IMAGE" "$@"
