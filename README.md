# CDeep3M-anywhere

Run CDeep3M-trained EM segmentation models on any CPU or Vulkan GPU, no CUDA required.
Built on [ncnn](https://github.com/Tencent/ncnn).

**Status: prediction only, 1fm model only.** Training is not supported (ncnn cannot train).
3fm and 5fm models have not been converted or tested. Tested on one machine only
(Ubuntu 24.04, gcc 13, i9-11900K, Intel UHD 750).

## Why
CDeep3M2 predicts through a custom Caffe fork that needs an NVIDIA GPU, `nvidia-smi` and a
2018 build toolchain. This project replaces the Caffe call with ncnn, which runs on x86/ARM
CPUs and on any GPU with a Vulkan driver, and leaves the rest of the pipeline
(preprocessing, augmentation, ensembling, merging) unchanged.

## Measured results (1fm, SBEM mitochondria model, 11 test images of 1024x1024)
| Engine | 1 variant | 8 variants (full quality) | Per slice |
|---|---|---|---|
| Caffe CPU (8 BLAS threads) | 114 s | 863 s | 7.3 s |
| **ncnn CPU** | **37 s** | **273 s** | **2.2 s** |
| ncnn Vulkan (Intel UHD 750, fp32) | 302 s | not run | 19.9 s |

* ncnn CPU is 3.2x faster than Caffe CPU on the same machine.
* Final segmentations of ncnn and Caffe differ by at most 1 gray level on 0.001% of pixels (Dice 1.00000).
* Accuracy on 10 labelled validation images: Dice 0.724 with 1 variant, **0.840 with 8 variants**
  (IoU 0.725, precision 0.769, recall 0.926). The pretrained model may have seen related data,
  so treat this as indicative, not a clean test score.
* The Intel iGPU is slower than the CPU here. Vulkan only pays off on faster GPUs, which we have not tested.
* fp16 on the Intel iGPU changed the output slightly, so the runner uses fp32.

### int8 quantization (ncnn CPU)
| Job (1fm, ncnn CPU) | fp32 | int8 | Speedup |
|---|---|---|---|
| 10 validation images, 1 variant | 33 s | 27 s | 1.22x |
| 10 validation images, 8 variants | 245 s | 181 s | 1.35x |
| 11 test images, 8 variants (mean of 2 runs) | 262 s | 198 s | 1.32x |

* The int8 model is 31.8 MB instead of 96.2 MB.
* Accuracy against ground truth is unchanged within noise: Dice 0.8455 (int8) vs 0.8403 (fp32) with 8 variants.
* int8 and fp32 final segmentations are close but not identical: Dice 0.984 between them on the
  validation images (3.3% of pixels differ by more than 8 gray levels, at mask boundaries), 0.995 on the test set.
* Calibration used 20 training images with the KL method. Other methods and image counts were not tried.
* int8 is CPU only; it has not been tested on Vulkan.

Report: [bench/INT8_REPORT.md](bench/INT8_REPORT.md).

The timings in this README were measured before the padding skip below was added; they have not been re-measured.

Full report: [bench/BENCHMARK.md](bench/BENCHMARK.md). Estimates for hardware we did not measure
(other GPUs, phone, training) are in [docs/NOTES.md](docs/NOTES.md) and are labelled as estimates.

## Quick start
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
2. **Python packages**: `pip install ncnn numpy h5py pillow opencv-python-headless scikit-image joblib imageio matplotlib pandas pyyaml requests`. Also install GNU `parallel` and `bc`.
3. **Get a model** from the [model zoo](https://cdeep3m.crbs.ucsd.edu/home/pre_trained_models),
   for example SBEM and ssTEM Mitochondria (DOI 10.7295/W9CDEEP3M50681). Keep its folder layout.
4. **Build ncnn tools** (needs `libprotobuf-dev protobuf-compiler`; Vulkan needs only the normal GPU driver):
   ```
   cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=ON -DNCNN_BUILD_TOOLS=ON \
         -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_PYTHON=OFF -DNCNN_BUILD_TESTS=OFF ..
   ```
5. **Convert the 1fm model**
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
   Use `--device vulkan` for a Vulkan GPU. Prefix with `NCNN_PRECISION=int8` to use the int8 model (CPU only). `--denoise 0` is needed unless `prox_tv` is installed
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
* 1fm only. 3fm/5fm take several input slices and are untested.
* For the 1fm model the ncnn backend skips the 4 z-padding slices (2 at each end) that the merge step
  discards; the output is unchanged. `NCNN_SKIP_EDGE=0` turns this off. Small images are still zero-padded to 1024x1024.
* Note: for the 1fm model `--augspeed 1` produces 8 variants, not 16.
* int8 is CPU only and was calibrated on 20 images; check accuracy on your own data before relying on it.
* Only tested on Linux x86-64.

## License
See [LICENSE](LICENSE) and [NOTICE](NOTICE). The code derived from CDeep3M2 is under the UC Regents
non-commercial license, which requires the notice to be kept.
