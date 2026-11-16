# Technical note: measured results

Speed and accuracy of the ncnn backend against Caffe, int8 quantization, the multi-frame (3fm, 5fm) models and the
padding skip.

Unless a section says otherwise, the machine is the one in the [README](../README.md)'s status line (Ubuntu 24.04, i9-11900K,
Intel UHD 750); the GPU numbers come from a Tesla P100 ([bench/GPU_P100.md](../bench/GPU_P100.md)).

## Results in short
* ncnn on the CPU is 3.2x faster than Caffe on the CPU; on a Tesla P100, ncnn Vulkan is 3.8x faster than the
  original Caffe on CUDA (93 s against 353 s for the 8-variant test).
* Output matches Caffe (Dice 1.00000); accuracy on 10 labelled validation images is Dice 0.840 with 8 variants.
* On the CPU, int8 plus skipping the padding slices make the 8-variant run 1.8x faster than fp32, with unchanged
  accuracy.

## Measured results (1fm, SBEM mitochondria model, 11 test images of 1024x1024; Caffe and fp32 rows predate the padding skip)
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
* The Intel iGPU is slower than the CPU here. On a discrete GPU Vulkan pays off: on a Tesla P100, ncnn Vulkan ran the
  8-variant test in 93 s against 353 s for the original Caffe on CUDA (3.8x), with identical output
  ([bench/GPU_P100.md](../bench/GPU_P100.md)).
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

Report: [bench/INT8_REPORT.md](../bench/INT8_REPORT.md).

### 3fm and 5fm
The multi-frame models start with two 3D convolutions over 3 or 5 neighbouring slices. `ncnn/convert_multiframe.py`
rewrites them as exact 2D convolutions (slices as channels), so the rest converts as for 1fm. ncnn matches Caffe on the
real slices (max diff 0.00001); on the validation images at augspeed 10 the Dice is 0.80 (3fm) and 0.79 (5fm) against 0.72 for
1fm, and 0.82 for the three-model ensemble. int8 matches fp32 within noise. Details: [bench/MULTIFRAME.md](../bench/MULTIFRAME.md).

### Padding skip (default)
The pipeline pads every stack with 2 z-slices at each end and the merge step discards their predictions,
so the ncnn backend no longer predicts them (3fm and 5fm still read the padded slices as input context). This is the default;
`NCNN_SKIP_EDGE=0` turns it off. Output is identical (max abs diff 0).

| Job (ncnn CPU, 8 variants) | fp32 + padding | int8 + padding | **int8, no padding** | Both vs fp32 + padding |
|---|---|---|---|---|
| 11 test images | 262 s | 198 s | **146 s** | **1.80x** |
| 10 validation images | 245 s | 181 s | **140 s** | **1.75x** |
| 1 image (int8) | not run | 70.6 s | **17.5 s** | padding skip alone: 4.0x |

The fp32 and with-padding numbers are the earlier measurements; only the no-padding runs were new.
Details: [bench/PADDING_SKIP.md](../bench/PADDING_SKIP.md).

Full report: [bench/BENCHMARK.md](../bench/BENCHMARK.md). Estimates for hardware we did not measure
(other GPUs, phone, training) are in [docs/NOTES.md](NOTES.md) and are labelled as estimates.
