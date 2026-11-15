# 3fm and 5fm on ncnn (CPU), validation images, --augspeed 10

Parity with Caffe CPU on an 8-slice stack (real slices, i.e. shifts 2-5 that the merge step keeps):
3fm and 5fm both max |diff| 0.00001, correlation 1.00000. The padding shifts (discarded by the merge)
differ where Caffe zero-fills input outside the stack and the runner clamps.

Accuracy against ground truth on the 10 labelled validation images, 1 variant per model (augspeed 10).
The 1fm rows are the earlier measurements; 3fm/5fm are new. Ensemble = mean of the three models' final
images, the same operation as EnsemblePredictions.py.

| Run | Dice | IoU | Precision | Recall |
|---|---|---|---|---|
| 1fm int8 (earlier data) | 0.7198 | 0.5623 | 0.7350 | 0.7053 |
| 1fm fp32 (earlier data) | 0.7241 | 0.5675 | 0.7221 | 0.7260 |
| 3fm int8 | 0.8023 | 0.6699 | 0.7547 | 0.8564 |
| 3fm fp32 | 0.8014 | 0.6686 | 0.7538 | 0.8553 |
| 5fm int8 | 0.7946 | 0.6592 | 0.7523 | 0.8419 |
| 5fm fp32 | 0.7956 | 0.6606 | 0.7574 | 0.8380 |
| 1fm+3fm+5fm ensemble, int8 | 0.8184 | 0.6927 | 0.7872 | 0.8523 |
| 1fm+3fm+5fm ensemble, fp32 | 0.8208 | 0.6961 | 0.7879 | 0.8566 |

* int8 matches fp32 within noise for 3fm and 5fm (2.5% and 2.0% of pixels differ by more than 8 gray levels).
* The pipeline's own ensemble step on a one-image run of all three models equals the offline mean (max diff 0).
* Wall times of the single end-to-end runs are in multiframe_e2e_times.tsv (one run each, not a benchmark):
  about 20 s int8 and 26-27 s fp32 per model for 10 images at augspeed 10.
* These are single-variant results. The default `--augspeed 1` uses 16 variants for 3fm and 5fm.
