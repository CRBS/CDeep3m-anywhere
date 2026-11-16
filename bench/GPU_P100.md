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
* The Caffe runs ended with an error in the last step, `Create_overlays.py` (OpenCV rejects the 3-channel test
  images; the bug the patch in `patches/` fixes). The segmentations were complete, so the times up to the last
  segmentation are valid; the total wall time of the 1-variant Caffe run was not recorded.
* Vulkan on this host needed the Docker fix in 0d67dfc (libEGL, the NVIDIA ICD/EGL vendor files, `graphics` capability).

Raw numbers: [gpu_p100_timings.tsv](gpu_p100_timings.tsv).
