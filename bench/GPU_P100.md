# GPU: ncnn Vulkan vs the original Caffe/CUDA on a Tesla P100

1fm SBEM mitochondria model, the 11 test images of `mito_testsample/testset` (1024x1024), `--denoise 0`.
Machine: 2x Xeon Gold 5218N, Tesla P100-PCIE-16GB, NVIDIA driver 470.256.02, Ubuntu 20.04, Docker 20.10 with
nvidia-container-cli 1.5.1. One run each, strictly one after another, on a shared server (not otherwise busy).

* ncnn: image `cdeep3m-anywhere:gpu` (commit 0d67dfc) through `docker/run.sh`, `--backend ncnn --device auto`.
  auto picked Vulkan GPU 0 in every run (`Running prediction with ncnn on vulkan (GPU 0)`); fp32, padding skip on.
* Caffe: the existing CDeep3M2 2.1.5 image (`cdeep3m-2.1.5:U16C11.3`, Caffe on CUDA 11.3), unchanged pipeline.

| Job | ncnn Vulkan | Caffe CUDA | Speedup |
|---|---|---|---|
| 1 variant (augspeed 10), prediction step | 12.3 s | 44.3 s | 3.6x |
| 1 variant, start to last segmentation | 18.4 s | 64.0 s | 3.5x |
| **8 variants (augspeed 1), prediction step** | **93.3 s** | **353.4 s** | **3.8x** |
| 8 variants, start to last segmentation | 101.4 s | 369.4 s | 3.6x |

"Prediction step" is the `real` time the pipeline logs for the network runs; "start to last segmentation" runs from
the creation of the output folder to the last `ensembled/Segmented_*.png` (preprocessing, prediction, merge, ensemble).

* Part of the gap is the padding skip: ncnn does not predict the 4 z-padding slices that the merge discards,
  Caffe does. That alone cuts the per-stack work from 15 to 11 slices.
* Output: ncnn Vulkan vs Caffe CUDA Dice 1.0000; the final segmentations differ by at most 1 gray level on
  0.0015 % (1 variant) and 0.0006 % (8 variants) of the pixels.
* The first Caffe runs ended with an error in the last step, `Create_overlays.py` (OpenCV rejects the 3-channel
  test images). Rerun with that fix (`docker/Dockerfile.caffe`, image `cdeep3m-caffe:fixed`) both runs finished
  cleanly (DONE, 11 segmentations, 11 overlays) with byte-identical segmentations and the same times within 1 %:
  prediction step 44.5 s / 353.6 s, start to last segmentation 64.1 s / 371.6 s, total wall 67.5 s / 375.1 s
  (1 / 8 variants). The table shows the first runs.
* Per predicted slice (ncnn 11 per stack, Caffe 15): ncnn Vulkan 1.12 s (1 variant) and 1.06 s (8 variants),
  Caffe 2.95 s in both, so the engine alone is about 2.7x faster; the padding skip gives the rest.
* Vulkan on this host needed the Docker fix in 0d67dfc (libEGL, the NVIDIA ICD/EGL vendor files, `graphics` capability).

Raw numbers: [gpu_p100_timings.tsv](gpu_p100_timings.tsv).
