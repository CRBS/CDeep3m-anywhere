# Padding skip (default for the 1fm model), ncnn CPU, 8 variants

Skipping the 2+2 z-padding slices that the merge step discards. Wall time of the full pipeline.
fp32 and int8 *with* padding are the earlier measurements (bench/INT8_REPORT.md, test set = mean of 2 runs);
only the no-padding runs were new. Raw new timings: padding_skip_timings.tsv.

| Job | fp32 + padding | int8 + padding | int8, no padding | int8 alone | padding skip alone (on int8) | both vs fp32 + padding |
|---|---|---|---|---|---|---|
| 11 test images | 262 s | 198 s | 146 s | 1.32x | 1.36x | 1.80x |
| 10 validation images | 245 s | 181 s | 140 s | 1.35x | 1.30x | 1.75x |
| 1 image (int8) | not run | 70.6 s | 17.5 s | not run | 4.0x | not run |

* Segmentations with the skip are identical to the with-padding int8 results (max abs diff 0) on both sets,
  so accuracy is unchanged (validation Dice 0.8455).
* The gain follows the work removed (15 -> 11 slices is 1.36x); fixed preprocessing and merge costs reduce it
  for smaller stacks. The two test-set runs both measured 145.7 s; validation and single-image cases were run once.
