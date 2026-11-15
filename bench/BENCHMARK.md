# CDeep3M 1fm prediction benchmark

CPU: 11th Gen Intel(R) Core(TM) i9-11900K @ 3.50GHz (16 threads), RAM 62 GB, GPU: Intel UHD 750 (Vulkan), Python 3.12.3

Model: SBEM_Mitochondria 1fm (iter 80000). Full pipeline via runprediction.sh (preprocess, predict, ensemble, overlay), denoise off.

## 1. Engine comparison, 1 augmentation variant, 11 test images (1024x1024)

| Engine | Wall time | Prediction step | Per slice (15 padded) |
|---|---|---|---|
| Caffe CPU (8 BLAS threads) | 114 s | 109 s | 7.3 s |
| ncnn CPU | 37 s | 32 s | 2.2 s |
| ncnn Vulkan (Intel UHD 750, fp32) | 302 s | 298 s | 19.9 s |

## 2. Full quality, 8 augmentation variants (1fm at --augspeed 1), 11 test images

| Engine | Wall time | Prediction step |
|---|---|---|
| Caffe CPU (8 BLAS threads) | 863 s | 857 s |
| ncnn CPU | 273 s | 268 s |

ncnn CPU speedup over Caffe CPU: **3.2x** wall, 3.2x prediction step.

## 3. Agreement with Caffe (final ensembled segmentation, 0-255 scale)

| Comparison | mean abs diff | max abs diff | % pixels differing | Dice (>127) |
|---|---|---|---|---|
| s_ncnn_cpu vs s_caffe_cpu | 0.0000 | 1 | 0.001 | 1.00000 |
| s_ncnn_vulkan vs s_caffe_cpu | 0.0000 | 1 | 0.001 | 1.00000 |
| f_ncnn_cpu vs f_caffe_cpu | 0.0000 | 1 | 0.001 | 1.00000 |

## 4. Accuracy against ground truth (10 labelled validation images)

| Run | Dice (pooled) | Dice (per-image mean) | IoU | Precision | Recall |
|---|---|---|---|---|---|
| Caffe CPU, 1 variant | 0.7241 | 0.7242 | 0.5675 | 0.7221 | 0.7260 |
| ncnn CPU, 1 variant | 0.7241 | 0.7242 | 0.5675 | 0.7221 | 0.7260 |
| ncnn CPU, 8 variants | 0.8403 | 0.8396 | 0.7246 | 0.7690 | 0.9262 |
