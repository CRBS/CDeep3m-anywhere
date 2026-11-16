#!/bin/bash
# Install Apptainer without root into a folder of your own (default ~/apptainer), to run .sif images on a machine
# whose Singularity cannot run them (for example `starter-suid not found`). Uses unprivileged user namespaces.
# Tested on Ubuntu 20.04. Afterwards: export PATH=~/apptainer/usr/bin:$PATH   (singularity/run.sh finds it)
#
#   singularity/install_apptainer_local.sh [folder]
set -e
D=$(readlink -f "${1:-$HOME/apptainer}")
V=1.3.6
FUSE=http://archive.ubuntu.com/ubuntu/pool/universe/f/fuse3/libfuse3-3_3.9.0-2_amd64.deb   # Ubuntu 20.04
mkdir -p "$D/dl"
curl -sSfL -o "$D/dl/apptainer.deb" https://github.com/apptainer/apptainer/releases/download/v$V/apptainer_${V}_amd64.deb
dpkg-deb -x "$D/dl/apptainer.deb" "$D"
# The non-setuid build finds its config and state relative to <prefix>=$D/usr.
ln -sfn ../etc "$D/usr/etc"
mkdir -p "$D/var/lib/apptainer/mnt/session"
ln -sfn ../var "$D/usr/var"
# squashfuse_ll (mounts the .sif) needs libfuse3; take it from the Ubuntu package if the host has none.
if ! ldconfig -p | grep -q libfuse3.so.3; then
  curl -sSfL -o "$D/dl/libfuse3.deb" "$FUSE"
  dpkg-deb -x "$D/dl/libfuse3.deb" "$D"
  B=$D/usr/libexec/apptainer/bin
  [ -f "$B/squashfuse_ll.bin" ] || mv "$B/squashfuse_ll" "$B/squashfuse_ll.bin"
  printf '#!/bin/sh\n# libfuse3 is not installed on this host: use the copy in %s/lib\nLD_LIBRARY_PATH=%s/lib/x86_64-linux-gnu exec %s/squashfuse_ll.bin "$@"\n' "$D" "$D" "$B" > "$B/squashfuse_ll"
  chmod +x "$B/squashfuse_ll"
fi
"$D/usr/bin/apptainer" --version
echo "done. Use: export PATH=$D/usr/bin:\$PATH"
