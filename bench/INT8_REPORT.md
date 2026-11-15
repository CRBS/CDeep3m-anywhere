# ncnn fp32 vs ncnn int8 (CPU), 1fm model

Calibration: 20 labelled training images (KL method), 20 real slices. Model size: fp32 96.2 MB, int8 31.8 MB.

## Speed (wall time of the full pipeline)

| Job | fp32 | int8 | Speedup |
|---|---|---|---|
| 10 validation images, 1 variant | 33 s | 27 s | **1.22x** |
| 10 validation images, 8 variants | 245 s | 181 s | **1.35x** |
| 11 test images, 8 variants (mean of 2 runs) | 262 s | 198 s | **1.32x** |

## Accuracy against ground truth (10 labelled validation images)

| Run | Dice | IoU | Precision | Recall |
|---|---|---|---|---|
| fp32, 1 variant | 0.7241 | 0.5675 | 0.7221 | 0.7260 |
| int8, 1 variant | 0.7198 | 0.5623 | 0.7350 | 0.7053 |
| fp32, 8 variants | 0.8403 | 0.7246 | 0.7690 | 0.9262 |
| int8, 8 variants | 0.8455 | 0.7324 | 0.7822 | 0.9200 |

## Agreement of int8 with fp32 (final ensembled segmentation, 0-255 scale)

| Job | mean abs diff | max abs diff | % pixels differing by >8 | Dice between them |
|---|---|---|---|---|
| validation, 1 variant | 1.461 | 202 | 3.589 | 0.9651 |
| validation, 8 variants | 1.038 | 110 | 3.276 | 0.9835 |
| test set, 8 variants | 0.268 | 94 | 0.292 | 0.9950 |
