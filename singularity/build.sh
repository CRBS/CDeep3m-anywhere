#!/bin/bash
# Build singularity/cdeep3m-anywhere.sif from the local Docker image (no root needed, only access to Docker):
# adds singularity/Dockerfile on top of cdeep3m-anywhere:gpu, then converts it with docker-daemon://.
#
#   singularity/build.sh                 uses the Docker image cdeep3m-anywhere:gpu
#   DOCKER_BUILD=1 singularity/build.sh  builds that Docker image first (GPU variant, docs/DOCKER_INSTALL.md step 3)
#   TMPBASE=/big/disk                    where the temporary build files go (about 20 GB; default: singularity/build_tmp)
set -e
here=$(cd "$(dirname "$0")" && pwd)
repo=$(dirname "$here")
cmd=$(command -v apptainer || command -v singularity) || { echo "singularity/apptainer not found" >&2; exit 1; }
if [ "${DOCKER_BUILD:-0}" = 1 ]; then
  docker build --network host -f "$repo/docker/Dockerfile" -t cdeep3m-anywhere:gpu --build-arg JOBS="$(nproc)" \
      --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu118 --build-arg TORCH_SPEC="torch==2.7.1" "$repo"
fi
docker image inspect cdeep3m-anywhere:gpu >/dev/null || { echo "Docker image cdeep3m-anywhere:gpu missing (DOCKER_BUILD=1)" >&2; exit 1; }
tmp=${TMPBASE:-$here/build_tmp}/tmp.$$
mkdir -p "$tmp/tmp" "$tmp/cache"
trap 'rm -rf "$tmp"' EXIT
export SINGULARITY_TMPDIR=$tmp/tmp SINGULARITY_CACHEDIR=$tmp/cache APPTAINER_TMPDIR=$tmp/tmp APPTAINER_CACHEDIR=$tmp/cache
t0=$(date +%s)
docker build -f "$here/Dockerfile" -t cdeep3m-anywhere:sif "$repo"
"$cmd" build --force "$here/cdeep3m-anywhere.sif" docker-daemon://cdeep3m-anywhere:sif
echo "built $here/cdeep3m-anywhere.sif ($(du -h "$here/cdeep3m-anywhere.sif" | cut -f1)) in $(( $(date +%s)-t0 )) s"
