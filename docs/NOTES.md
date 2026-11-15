# Notes and estimates

## Measured on the development machine (i9-11900K, 62 GB RAM, Intel UHD 750)
* ncnn CPU 2.2 s per 1024x1024 slice; Caffe CPU 7.3 s; ncnn Vulkan on UHD 750 19.9 s (fp32).
* Fusing BatchNorm/Scale/ReLU with `ncnnoptimize`: no speedup (29.7 s vs 29.6 s per variant), identical output.
* Running 2 or 4 ncnn processes in parallel: no gain (30.3 and 30.9 s per variant vs 29.6 s). The CPU is already saturated.
* Caffe CPU with 8 BLAS threads was about 15% faster than 16 threads (107 s vs 127 s per variant).
* int8 (ncnn2table KL, 20 training slices): 1.22-1.35x faster end to end, Dice vs ground truth unchanged within noise.
  Model is 3x smaller. Smaller gain than hoped because only convolutions are quantized.
* Active layers in the 1fm deploy network are all standard (Convolution, Deconvolution with bilinear weights,
  BatchNorm+Scale, ReLU, Eltwise, Concat, Pooling, Softmax). The `Unpooling` layers in the file are commented out.
* The training loss is one `SoftmaxWithLoss` with class balancing (`label_select_param`) and `ignore_label: 2`.

## Estimates (NOT measured)
These were derived from the numbers above and the hardware specs. Treat them as rough.
* Radeon Pro WX 5100 with ncnn Vulkan: about 1-5 s per slice, best guess near the CPU's 2.2 s.
* NVIDIA RTX A2000: about 1-2 min for the 11-image, 8-variant job with Caffe+CUDA+cuDNN,
  about 1-8 min with ncnn Vulkan.
* Samsung S25 (ncnn, Android): about 2-8 s per slice; peak memory not measured.
* Training in Caffe CPU mode: about 85-110 s per default update (40 patches of 320x320),
  so a 30,000 iteration run would take about a month; fine-tuning with a smaller batch for a few thousand
  iterations about 17-23 h. A PyTorch trainer might be 3-6x faster; unproven.

## Ideas, in order of expected value
1. Skip the z-padding slices for 1fm: implemented and now the default. Measured with int8: 1.30-1.36x on stacks, 4.0x on a single image (see bench/PADDING_SKIP.md); output identical.
2. Use the real image size instead of padding to 1024x1024.
3. Measure Dice against the number of augmentation variants (only 1 and 8 were measured).
4. Combined int8 + padding skip: measured 1.75-1.80x over fp32 with padding.
5. Try other int8 calibration methods (percentile, more images).

## Corrections made during development
* For the 1fm model `--augspeed 1` yields 8 variants, not 16 (see `augment_package.py`).
* The benchmark runs were marked ERROR only because of the `configs/check_limits.py` bug fixed by the patch;
  their segmentation outputs were complete. `bench/timings.tsv` keeps the raw status column.
