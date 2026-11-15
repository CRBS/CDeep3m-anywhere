# Training with the PyTorch trainer

ncnn cannot train, so `trainer/` is a PyTorch re-implementation of the CDeep3M Caffe training recipe for the
1fm, 3fm and 5fm models. A trained model is exported back to a `.caffemodel` and then converted with
`ncnn/convert_model.sh` like any other model.

## Quick start (CPU)
```
# 1. training data (CDeep3M2): images + labels -> augmented HDF5 files (0 = no secondary augmentation, no denoising)
python3 PreprocessTrainingData.py <images dir> <labels dir> 0 <train_data dir>

# 2. fine-tune a pretrained model (recommended) - or leave out --weights to train from scratch
export CAFFE_PROTO=<caffe fork>/src/caffe/proto/caffe.proto        # only needed to read/write .caffemodel files
tools/train_bg.sh 1fm <model>/1fm/deploy.prototxt <train_data dir> <run dir> \
    --weights <model>/1fm/trainedmodel/<...>.caffemodel --max-iter 3000 --iter-size 2 --base-lr 0.001

# 3. export (re-estimates the BatchNorm statistics, see below), convert, predict
python3 -m trainer.export_caffemodel --deploy <model>/1fm/deploy.prototxt --checkpoint <run dir>/latest.pt \
    --template <model>/1fm/trainedmodel/<...>.caffemodel --out <new model>/1fm/trainedmodel/<name>.caffemodel --data <train_data dir>
ncnn/convert_model.sh <model>/1fm/deploy.prototxt <new model>/1fm/trainedmodel/<name>.caffemodel <new model>/1fm/ncnn
```
Defaults follow CDeep3M2's `caffetrain.sh`: 30000 updates, batch of 5 patches of 320x320 x `--iter-size` 8
accumulation steps, SGD momentum 0.9, weight decay 0.0005, polynomial learning rate (power 0.8), base lr 0.01.

## Stopping and resuming (switching the PC off)
* Progress is saved atomically to `<run dir>/latest.pt` every 10 minutes (`--checkpoint-minutes`) and every 100
  updates (`--checkpoint-every`), plus numbered `ckpt_*.pt` copies (last 3 kept). A checkpoint holds the weights, the
  SGD momentum, the sampler state, the update counter and the loss history (about 190 MB).
* On SIGTERM, SIGINT or SIGHUP (a normal shutdown or Ctrl-C) the trainer drops the unfinished update, saves and
  exits with code 0. After a power cut you lose at most the work since the last periodic checkpoint.
* **To resume, run exactly the same command again.** Keep `<run dir>` on a disk that survives restarts.
* Verified: SIGTERM in the middle of an update saved the state at the last completed update, and the same command
  resumed from it.

## Auto-push
* `tools/autopush.sh` commits and pushes all changes of this repository to `origin/main` (never forced). It refuses to
  commit files over 20 MB or text that looks like a credential or a personal path.
  `.gitignore` excludes weights (`*.caffemodel`, `*.pt`, ...), data and `venv/`.
* `tools/autopush_loop.sh 15` runs it every 15 minutes. It survives closing the terminal but not a reboot; to
  start it at every boot add a cron line (`crontab -e`):
  `@reboot cd <repo> && nohup tools/autopush_loop.sh 15 >/dev/null 2>&1 &`
* The trainer writes small progress files into `progress/` (`1fm.csv`: one line per update, `1fm.json`: latest status),
  which the auto-push picks up. Checkpoints stay local.

## BatchNorm statistics after fine-tuning (important)
The BatchNorm running mean/variance are a slow moving average (time constant about 1000 forward passes). After a
short fine-tune they still describe the old weights. Measured on 30 updates at lr 0.001: the model classified 64% of
the pixels of a slice as mitochondria (the original: 5%) although the convolution weights had moved by only 0.3%;
restoring the original weights fixed it, restoring only the statistics did not, and re-estimating the statistics did
(6%). `trainer.export_caffemodel` therefore re-estimates them (`--recalibrate`, default 40 batches of class-balanced
training patches) before writing the model. Do not export with `--recalibrate 0` after a short run.

## What was verified
* The network matches the Caffe forward pass for 1fm, 3fm and 5fm (max abs difference <= 0.000008 on real slices).
* The patch sampler and the balanced loss follow the fork's source (`patch_sampler.cpp`, `sample_selector.cpp`,
  `softmax_loss_layer.cpp`): uniform volume and centre, background centres accepted with probability 1/4 and
  mitochondria always, gradients of rejected pixels dropped but normalised by all valid pixels. The fork's mirror option
  is commented out in its code, so there is no on-the-fly flipping.
* Training, checkpoint, resume, export, conversion and prediction work end to end: a 30-update fine-tune of 1fm on the
  20 sample training images gave Dice 0.7155 on the validation images against 0.7241 for the original (not an
  improvement, a check that the chain does not damage the model).
* 3fm and 5fm: 4-update fine-tunes, export and conversion also work (ncnn on the exported model vs the PyTorch
  checkpoint: max abs difference 0.000012 for 3fm and 0.000009 for 5fm; the checkpoint reloads from the export exactly;
  the first-layer 3D kernels stay shared).
* **Not verified:** that training dynamics match a Caffe training run (loss curves were not compared with Caffe),
  and long runs. CPU only for now (the code is plain PyTorch; GPU support was not added or tested).

## Speed (one short run on an i9-11900K, 8 threads, not a benchmark)
One forward and backward pass of 5 patches of 320x320 took about 3.2 s, so an update takes about 3.2 s x `--iter-size`
(6.3 s at 2, about 25 s at the default 8). A full default run of 30000 updates is therefore on the order of 9 days
on this CPU, a 3000-update fine-tune at `--iter-size` 2 about 5 hours. These are extrapolations from that one run.
