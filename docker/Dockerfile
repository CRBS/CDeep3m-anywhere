# CDeep3M-anywhere: ncnn prediction (CPU or Vulkan GPU), model conversion and the PyTorch trainer.
#
# Build from the repository root:
#   docker build -f docker/Dockerfile -t cdeep3m-anywhere:cpu .
#   docker build -f docker/Dockerfile -t cdeep3m-anywhere:cu118 \
#       --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu118 --build-arg TORCH_SPEC="torch==2.7.1" .
#
# The image does not depend on the host OS. A GPU needs only the host's driver and the NVIDIA container
# toolkit (see docker/README.md). No model weights or data are in the image; mount them at run time.

ARG BASE=ubuntu:24.04

# ---------- stage 1: build the ncnn conversion tools ----------
FROM ${BASE} AS builder
ARG NCNN_TAG=20261020
ARG JOBS=4
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git ca-certificates libprotobuf-dev protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*
# Only the tools are built here (caffe2ncnn, ncnn2table, ncnn2int8). They run on the CPU, so Vulkan is off;
# prediction uses the ncnn Python wheel, which has its own Vulkan support. NCNN_SIMPLEOCV avoids an OpenCV
# dependency (calibration data are .npy files, no image decoding is needed).
RUN git clone --depth 1 --branch ${NCNN_TAG} https://github.com/Tencent/ncnn.git /src/ncnn \
    && cmake -S /src/ncnn -B /src/ncnn/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DNCNN_VULKAN=OFF -DNCNN_BUILD_TOOLS=ON -DNCNN_SIMPLEOCV=ON \
        -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_TESTS=OFF -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_PYTHON=OFF \
    && cmake --build /src/ncnn/build -j ${JOBS} \
    && mkdir -p /opt/ncnn_build/tools/caffe /opt/ncnn_build/tools/quantize \
    && cp /src/ncnn/build/tools/caffe/caffe2ncnn /opt/ncnn_build/tools/caffe/ \
    && cp /src/ncnn/build/tools/quantize/ncnn2table /src/ncnn/build/tools/quantize/ncnn2int8 /opt/ncnn_build/tools/quantize/
# CDeep3M2 at the commit the patch was made for, and caffe.proto of the CDeep3M Caffe fork
RUN git clone https://github.com/CRBS/cdeep3m2.git /opt/cdeep3m2 \
    && git -C /opt/cdeep3m2 checkout 67d6308 && rm -rf /opt/cdeep3m2/.git
RUN git clone --filter=blob:none --no-checkout https://github.com/coleslaw481/caffe_nd_sense_segmentation.git /src/fork \
    && git -C /src/fork show 6566b5d:src/caffe/proto/caffe.proto > /opt/caffe.proto

# ---------- stage 2: runtime ----------
FROM ${BASE}
# PyTorch build: the CPU wheel by default. For an NVIDIA GPU pick the index that matches the host DRIVER:
#   driver 450.80 .. 524  -> https://download.pytorch.org/whl/cu118 with TORCH_SPEC="torch==2.7.1"
#   driver >= 525.60      -> https://download.pytorch.org/whl/cu126 (or newer)
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_SPEC=torch
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip ca-certificates git patch parallel bc util-linux procps \
        libgomp1 libprotobuf32t64 libvulkan1 vulkan-tools \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --no-cache-dir ncnn numpy h5py pillow opencv-python-headless scikit-image joblib imageio \
        matplotlib pandas pyyaml requests protobuf \
    && pip install --no-cache-dir --index-url ${TORCH_INDEX} ${TORCH_SPEC}

COPY --from=builder /opt/ncnn_build /opt/ncnn_build
COPY --from=builder /opt/cdeep3m2 /opt/cdeep3m2
COPY --from=builder /opt/caffe.proto /opt/caffe.proto
COPY . /opt/cdeep3m-anywhere
RUN cd /opt/cdeep3m2 \
    && patch -p1 < /opt/cdeep3m-anywhere/patches/cdeep3m2-67d6308-device-backend.patch \
    && cp /opt/cdeep3m-anywhere/ncnn/ncnn_predict.py . \
    && chmod +x *.sh /opt/cdeep3m-anywhere/ncnn/*.sh /opt/cdeep3m-anywhere/tools/*.sh \
    && chmod -R a+rX /opt

ENV CDEEP3M2=/opt/cdeep3m2 \
    NCNN_BUILD=/opt/ncnn_build \
    CAFFE_PROTO=/opt/caffe.proto \
    PYTHONPATH=/opt/cdeep3m-anywhere \
    PATH=/opt/cdeep3m2:/opt/cdeep3m-anywhere/ncnn:/opt/cdeep3m-anywhere/tools:/opt/venv/bin:$PATH \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    HOME=/tmp
WORKDIR /work
CMD ["bash"]
