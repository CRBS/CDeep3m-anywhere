# Thin layer on the CDeep3M-anywhere GPU image for the Singularity conversion (singularity/build.sh).
ARG BASE=cdeep3m-anywhere:gpu
FROM ${BASE}
COPY singularity/gpu_check.sh /usr/local/bin/gpu_check.sh
# Singularity mounts your home folder: do not let packages in ~/.local shadow the ones in the image.
ENV PYTHONNOUSERSITE=1
LABEL org.opencontainers.image.source=https://github.com/CRBS/CDeep3m-anywhere \
      org.opencontainers.image.description="CDeep3M-anywhere for Singularity: ncnn CPU/Vulkan prediction, PyTorch CUDA 11.8 training"
