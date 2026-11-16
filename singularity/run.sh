#!/bin/bash
# Start the CDeep3M-anywhere Singularity image with a data folder mounted at /data (like docker/run.sh).
#
#   singularity/run.sh <data folder> [command ...]      (no command = interactive shell)
#
#   SIF=/path/to/cdeep3m-anywhere.sif   image (default: next to this script)
#   GPU=auto|on|off                     auto (default) adds --nv when nvidia-smi works on the host
set -e
[ $# -ge 1 ] || { sed -n '2,8p' "$0"; exit 1; }
data=$(readlink -f "$1"); shift
[ -d "$data" ] || { echo "not a folder: $data" >&2; exit 1; }
SIF=${SIF:-$(cd "$(dirname "$0")" && pwd)/cdeep3m-anywhere.sif}
[ -f "$SIF" ] || { echo "image not found: $SIF (set SIF=...)" >&2; exit 1; }
cmd=$(command -v apptainer || command -v singularity) || { echo "singularity/apptainer not found (Expanse: module load singularitypro)" >&2; exit 1; }
nv=()
GPU=${GPU:-auto}
if [ "$GPU" = on ] || { [ "$GPU" = auto ] && nvidia-smi -L >/dev/null 2>&1; }; then nv=(--nv); fi
[ $# -ge 1 ] || set -- bash
exec "$cmd" exec "${nv[@]}" ${EXTRA_BINDS:+-B "$EXTRA_BINDS"} -B "$data":/data --pwd /data "$SIF" "$@"
