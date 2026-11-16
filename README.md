# CDeep3M-anywhere

Run CDeep3M-trained EM segmentation models on any CPU or Vulkan GPU, no CUDA required.
Built on [ncnn](https://github.com/Tencent/ncnn).

**Status: prediction on ncnn for 1fm, 3fm and 5fm; training with a PyTorch trainer** (CPU, resumable after a
shutdown, see [docs/TRAINING.md](docs/TRAINING.md)). ncnn itself cannot train. Tested on one machine only
(Ubuntu 24.04, gcc 13, i9-11900K, Intel UHD 750).

## Why
CDeep3M2 predicts through a custom Caffe fork that needs an NVIDIA GPU, `nvidia-smi` and a
2018 build toolchain. This project replaces the Caffe call with ncnn, which runs on x86/ARM
CPUs and on any GPU with a Vulkan driver, and leaves the rest of the pipeline
(preprocessing, augmentation, ensembling, merging) unchanged.

## Results in short
* ncnn on the CPU is 3.2x faster than Caffe on the CPU; on a Tesla P100, ncnn Vulkan is 3.8x faster than the
  original Caffe on CUDA (93 s against 353 s for the 8-variant test).
* Output matches Caffe (Dice 1.00000); accuracy on 10 labelled validation images is Dice 0.840 with 8 variants.
* On the CPU, int8 plus skipping the padding slices make the 8-variant run 1.8x faster than fp32, with unchanged
  accuracy.

Tables and details (speed, int8, 3fm/5fm, padding skip): [docs/TECHNICAL_NOTE.md](docs/TECHNICAL_NOTE.md).

## Quick start
**With Docker** (nothing to install but Docker): follow [docs/DOCKER_INSTALL.md](docs/DOCKER_INSTALL.md), which builds the
image, downloads the model and sample images with `tools/modelDownload.sh`, converts the model and predicts.
Without Docker:

1. **Get CDeep3M2 and apply the patch**
   ```
   git clone https://github.com/CRBS/cdeep3m2.git && cd cdeep3m2 && git checkout 67d6308
   patch -p1 < ../CDeep3m-anywhere/patches/cdeep3m2-67d6308-device-backend.patch
   cp ../CDeep3m-anywhere/ncnn/ncnn_predict.py .
   chmod +x *.sh
   ```
   The patch adds `--device auto|cpu|gpu|vulkan` and `--backend caffe|ncnn`, removes the hard
   `nvidia-smi` requirement, and fixes two bugs: `configs/check_limits.py` only worked from the
   repo directory, and `Create_overlays.py` failed on palette PNGs with current OpenCV/scikit-image.
2. **Python packages**: `pip install ncnn numpy h5py pillow opencv-python-headless scikit-image joblib imageio matplotlib pandas pyyaml requests`. Also install GNU `parallel`, `bc` and GNU `time` (`/usr/bin/time`).
3. **Get a model** from the [model zoo](https://cdeep3m.crbs.ucsd.edu/home/pre_trained_models),
   for example SBEM and ssTEM Mitochondria (DOI 10.7295/W9CDEEP3M50681). Keep its folder layout.
4. **Build ncnn tools** (needs `libprotobuf-dev protobuf-compiler`; Vulkan needs only the normal GPU driver):
   ```
   cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=ON -DNCNN_BUILD_TOOLS=ON \
         -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_PYTHON=OFF -DNCNN_BUILD_TESTS=OFF ..
   ```
5. **Convert a model** (1fm shown; 3fm and 5fm work the same, but also `pip install protobuf` and `export CAFFE_PROTO=<caffe fork>/src/caffe/proto/caffe.proto`)
   ```
   export NCNN_BUILD=/path/to/ncnn/build
   ncnn/convert_model.sh <model>/1fm/deploy.prototxt <model>/1fm/trainedmodel/*.caffemodel <model>/1fm/ncnn
   ```
   Optional int8: make calibration data from labelled **training** images (not the images you evaluate on),
   then pass its list as a fourth argument. This also writes `<model>/1fm/ncnn/int8/`.
   ```
   python3 ncnn/make_calibration.py --repo <patched cdeep3m2> --images <training images> --out calib
   ncnn/convert_model.sh <deploy.prototxt> <model.caffemodel> <model>/1fm/ncnn calib/list.txt
   ```
6. **Predict**
   ```
   ./runprediction.sh --models 1fm --denoise 0 --augspeed 1 --backend ncnn --device cpu <model> <images> <out>
   ```
   `--device auto` (the default) uses the first discrete Vulkan GPU when ncnn finds one and the CPU otherwise
   (integrated GPUs stay on the CPU, which was faster); `--device vulkan` or `--device cpu` forces one. In Docker,
   Vulkan on NVIDIA needs the `graphics` driver capability, which `docker/run.sh` passes (see [docker/README.md](docker/README.md)). The default precision is int8 when `<model>/<m>/ncnn/int8/` exists and the device is the CPU (otherwise fp32); `NCNN_PRECISION=fp32` forces fp32. `--denoise 0` is needed unless `prox_tv` is installed
   and you use a denoised model. Results are in `<out>/ensembled/Segmented_*.png`.

Optional, to compare with Caffe: build the Caffe fork CPU-only with
`patches/caffe-fork-6566b5d-cpu-only-modern-toolchain.patch` and `caffe-fork/Makefile.config.cpu-only`
(needs libopenblas, protobuf, hdf5, boost, glog, gflags, OpenCV 4), then run with `--backend caffe --device cpu`.

## Reproduce the benchmark
See the header of `bench/run_bench.sh` for the environment variables; then
`VAL_LABELS=<labels dir> python3 bench/analyze.py <output dir>`.

## How the conversion works
The 1fm deploy network is plain 2D in practice: 3D-style layers with depth 1, a 5D input shape,
and a fork-specific `phase: PREDICT` field. `ncnn/sanitize_prototxt.py` strips those, then ncnn's
`caffe2ncnn` converts it. `convert_model.sh` also adds the `Input` layer that `caffe2ncnn` does not write
(the int8 quantization tool needs it). The result was checked against Caffe on real data.

## Known limitations
* 3fm/5fm: tested at `--augspeed 10` (1 variant per model); the full 16-variant run was not timed. Their conversion needs the fork's `caffe.proto` and python protobuf.
* The ncnn backend skips the 4 z-padding slices (2 at each end) whose predictions the merge step
  discards; the output is unchanged. `NCNN_SKIP_EDGE=0` turns this off. Small images are still zero-padded to 1024x1024.
* Note: for the 1fm model `--augspeed 1` produces 8 variants, not 16.
* int8 is CPU only and was calibrated on 20 images; check accuracy on your own data before relying on it.
* Only tested on Linux x86-64.

## License
See [LICENSE](LICENSE) and [NOTICE](NOTICE). The code derived from CDeep3M2 is under the UC Regents
non-commercial license, which requires the notice to be kept.
