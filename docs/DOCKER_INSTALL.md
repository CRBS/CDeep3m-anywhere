# Install and run with Docker

From an empty machine to a segmentation: build the image, download the model and the sample images, convert the
model for ncnn, predict. Every command was run as written on a Linux server with a Tesla P100 (driver 470, Docker
20.10); the times at the end are from that machine.

Run everything from the repository root. Nothing is installed on the host except Docker (and, for a GPU, the
NVIDIA driver and container toolkit); all tools run inside the image.

## 1. Requirements
* Docker, and your user in the `docker` group (`docker info` works without `sudo`).
* `curl` and `tar` on the host (for the download script).
* About 12 GB of disk for the image, 1 GB for the model and data.
* Optional NVIDIA GPU: the driver and the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  Check with `docker run --rm --gpus all ubuntu:24.04 nvidia-smi`. No CUDA toolkit is needed on the host.
  ncnn uses the GPU through **Vulkan**, not CUDA, so the driver must include its graphics/Vulkan libraries.
  A normal driver install does; a compute-only ("headless") install may not. Check on the host:
  `ls /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0` (must exist).

### How the image makes Vulkan work on NVIDIA (nothing to do on the host)
Without these, `vulkaninfo` in the container fails with `ERROR_INCOMPATIBLE_DRIVER` and ncnn falls back to the CPU
(found on driver 470 with container toolkit 1.5.1). All of it is in the repository, so every rebuilt image has it:

| Where | What | Why |
|---|---|---|
| `docker/Dockerfile` | installs `libegl1 libxext6 libx11-6` | the NVIDIA Vulkan driver library loads them |
| `docker/Dockerfile` | writes `/usr/share/vulkan/icd.d/nvidia_icd.json` and `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` | point Vulkan and EGL to the NVIDIA driver; older container toolkits do not provide them |
| `docker/run.sh`, `docker/docker-compose.yml` | `--gpus 'all,"capabilities=compute,utility,graphics"'` | without `graphics` Docker does not mount the driver's Vulkan libraries |

The driver libraries themselves come from the host at run time (the container toolkit mounts them), so the same
image works with any driver version. If you start containers yourself instead of with `docker/run.sh`, pass the
`--gpus` option above.

## 2. Get the code
```
git clone https://github.com/CRBS/CDeep3m-anywhere.git
cd CDeep3m-anywhere
```

## 3. Build the image
For a machine with an NVIDIA GPU (the image also runs on the CPU when no GPU is visible):
```
docker build --network host -f docker/Dockerfile -t cdeep3m-anywhere:gpu \
    --build-arg JOBS=16 \
    --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu118 --build-arg TORCH_SPEC="torch==2.7.1" .
```
For a machine without a GPU (smaller, CPU PyTorch):
```
docker build --network host -f docker/Dockerfile -t cdeep3m-anywhere:cpu --build-arg JOBS=16 .
```
* `--network host` avoids a build that hangs at `apt-get update` when the build container has no network
  (seen on our server); it is harmless elsewhere.
* `JOBS` is the number of compile jobs for the ncnn tools; set it to your core count or less.
* The cu118 PyTorch build fits NVIDIA drivers 450.80 and newer. It only matters for training; prediction uses
  ncnn with Vulkan, which works with any recent driver.
* The build takes about 10 to 20 minutes and the GPU image is about 8 GB.

Check that the GPU is visible to ncnn (GPU machines only):
```
IMAGE=cdeep3m-anywhere:gpu docker/run.sh . vulkaninfo --summary          # lists your GPU
IMAGE=cdeep3m-anywhere:gpu docker/run.sh . python3 /opt/cdeep3m2/ncnn_predict.py --pick-device
```
The second command prints `vulkan 0` when prediction will run on GPU 0, or `cpu`.

## 4. Download the model and the sample images
```
tools/modelDownload.sh
```
This writes into `downloads/` (ignored by git):

| Folder | Contents |
|---|---|
| `downloads/SBEM_Mitochondria/{1fm,3fm,5fm}` | pretrained SBEM mitochondria model (DOI 10.7295/W9CDEEP3M50681, 685 MB download) |
| `downloads/mito_testsample/training/{images,labels}` | 20 labelled training images |
| `downloads/mito_testsample/validation/{images,labels}` | 10 labelled validation images |
| `downloads/mito_testsample/testset` | 11 test images |

`tools/modelDownload.sh --model-only` or `--data-only` fetches one part; `tools/modelDownload.sh <folder>`
downloads somewhere else. Parts that are already there are skipped.

## 5. Convert the model for ncnn (once per model)
`docker/run.sh <folder> <command>` runs a command in the container with `<folder>` mounted at `/data`, as your own
user, and passes the NVIDIA GPU (with the `graphics` capability Vulkan needs) when the host has one.
Set `IMAGE=cdeep3m-anywhere:cpu` instead of `:gpu` on a machine without a GPU.

```
IMAGE=cdeep3m-anywhere:gpu docker/run.sh downloads bash -c '
  make_calibration.py --repo $CDEEP3M2 --images /data/mito_testsample/training/images --out /data/calib --model 1fm &&
  convert_model.sh /data/SBEM_Mitochondria/1fm/deploy.prototxt \
      /data/SBEM_Mitochondria/1fm/trainedmodel/1fm_classifer_iter_80000.caffemodel \
      /data/SBEM_Mitochondria/1fm/ncnn /data/calib/list.txt'
```
This writes `SBEM_Mitochondria/1fm/ncnn/model.{param,bin}` (fp32, used on the GPU) and `ncnn/int8/` (int8, used on
the CPU). The calibration uses the **training** images; never calibrate on the images you evaluate on.
Without the last argument (`/data/calib/list.txt`) only the fp32 model is written, which is enough for a GPU.
For 3fm and 5fm replace `1fm` everywhere (the image already sets `CAFFE_PROTO`, which they need).

## 6. Predict
```
IMAGE=cdeep3m-anywhere:gpu docker/run.sh downloads runprediction.sh \
    --models 1fm --denoise 0 --augspeed 1 --backend ncnn --device auto \
    /data/SBEM_Mitochondria /data/mito_testsample/testset /data/out_testset
```
* Results: `downloads/out_testset/ensembled/Segmented_*.png` (mitochondria probability, 0..255) and
  `downloads/out_testset/overlay/overlay_*.png`. The run is complete when `downloads/out_testset/DONE` exists;
  on failure an `ERROR` file says why, and the logs are in `downloads/out_testset/logs/`.
* `--device auto` uses the first discrete Vulkan GPU and otherwise the CPU. To see which one it used:
  `grep "ncnn on" downloads/out_testset/logs/prediction.log` (e.g. `Running prediction with ncnn on vulkan (GPU 0)`).
  `--device vulkan` or `--device cpu` forces one.
* `--augspeed 1` predicts 8 augmented variants per model and averages them (best quality); `--augspeed 10`
  predicts one (fastest, lower accuracy).
* `--denoise 0` is required with this model (the denoising step needs `prox_tv`, which is not in the image).
* Your own images: put the folder under `downloads/` (or mount another folder) and pass its `/data/...` path.
  Several models: `--models 1fm,3fm,5fm` after converting each.

### Expected run time (11 test images of 1024x1024, 1fm model)
| | 1 variant (`--augspeed 10`) | 8 variants (`--augspeed 1`) |
|---|---|---|
| ncnn Vulkan, Tesla P100 (`--device auto`) | 21 s | 103 s |
| original Caffe on CUDA, same GPU, for reference | 68 s | 375 s |

Both engines give the same segmentation (Dice 1.0000). Details: [bench/GPU_P100.md](../bench/GPU_P100.md).
On the CPU expect several times longer; see the measurements in the main [README](../README.md).

## 7. Train (optional)
How the staged 1fm training was run on the Tesla P100: from scratch on the 20 sample training images, one polynomial
learning-rate schedule (base lr 0.01, power 0.8, `--iter-size 1`), resumed with `--max-iter` 1000, then 3000, then
5000 in the same run folder, so the stages are 1000, 2000 and 2000 updates. The trainer and its options are described
in [TRAINING.md](TRAINING.md).

### Training data (once)
```
IMAGE=cdeep3m-anywhere:gpu docker/run.sh downloads bash -c 'cd /opt/cdeep3m2 && \
  python3 PreprocessTrainingData.py /data/mito_testsample/training/images /data/mito_testsample/training/labels 0 /data/train_data'
```
`0` = no secondary augmentation and no denoising. Result: 16 augmented stacks in `downloads/train_data` (913 MB).

### Stage script
Save as `downloads/train_staged/run_stages.sh` (paths are inside the container):
```
#!/bin/bash
cd /opt/cdeep3m-anywhere
R=/data/train_staged
t_all=$(date +%s)
for m in 1000 3000 5000; do
  t0=$(date +%s); echo "STAGE max-iter $m start $(date '+%F %T')"
  python3 -m trainer.train --model 1fm --deploy /data/SBEM_Mitochondria/1fm/deploy.prototxt --data /data/train_data \
      --out $R/run --max-iter $m --iter-size 1 --base-lr 0.01 --device cuda --progress-dir $R/progress || { echo "STAGE $m FAILED"; exit 1; }
  cp $R/run/latest.pt $R/stage_${m}.pt
  echo "STAGE max-iter $m done in $(( $(date +%s)-t0 )) s"
done
echo "ALL STAGES DONE in $(( $(date +%s)-t_all )) s"
```
* Each stage resumes from `run/latest.pt`; the learning rate continues on the same curve (0.01 at update 1, 0.00723
  at 1001, 0.00480 at 3001). A copy of the weights after each stage is kept as `stage_1000.pt`, `stage_3000.pt`,
  `stage_5000.pt` (about 190 MB each).
* To fine-tune the pretrained model instead, add
  `--weights /data/SBEM_Mitochondria/1fm/trainedmodel/1fm_classifer_iter_80000.caffemodel` and a lower `--base-lr`
  (e.g. 0.001) to the first stage only (later stages resume from the checkpoint).
* No GPU: use `--device cpu` (about 3.3 s per update on an i9-11900K instead of 0.2 s).
* `--progress-dir` keeps the progress files out of the repository's `progress/` folder, which `tools/autopush.sh` pushes.

### Start it in the background
```
mkdir -p downloads/train_staged     # and save run_stages.sh there
docker run -d --name cdeep3m-train-staged --gpus 'all,"capabilities=compute,utility,graphics"' \
    --user "$(id -u):$(id -g)" --shm-size=2g -v "$PWD/downloads":/data -w /data cdeep3m-anywhere:gpu \
    bash -c 'bash /data/train_staged/run_stages.sh > /data/train_staged/train.log 2>&1'
```
`-d` detaches the container from the terminal, so the training keeps running when the SSH/PuTTY session closes or
goes idle. To restart after an interruption, `docker rm cdeep3m-train-staged` and run the same command again: every
stage resumes from `run/latest.pt` (saved every 100 updates and every 10 minutes), and finished stages end at once.

### Watch it
```
grep STAGE downloads/train_staged/train.log                   # stage start/end times
tail -2 downloads/train_staged/train.log                        # e.g. update 149/1000  lr 0.00880  loss 0.25754 (avg16 0.24580)  0.2s
cat downloads/train_staged/progress/1fm.json                    # latest update, loss, elapsed time
docker ps --filter name=cdeep3m-train-staged                    # still running?
nvidia-smi                                                      # GPU in use
```
The container clock is UTC, so the times in `train.log` are UTC.

A status line every 10 minutes, written to a file by a process that also survives a closed session: save as
`downloads/train_staged/reporter.sh` and start it with
`cd downloads/train_staged && chmod +x reporter.sh && setsid nohup ./reporter.sh >> report.log 2>&1 < /dev/null &`
```
#!/bin/bash
R=$(cd "$(dirname "$0")" && pwd)
line() {
  python3 - "$R" <<'PY'
import json, sys, os
try:
    j = json.load(open(os.path.join(sys.argv[1], "progress", "1fm.json")))
    print("update %d (stage max-iter %d of 1000/3000/5000), %.1f%% of 5000, avg loss %.4f, elapsed %.1f min"
          % (j["update"], j["max_iter"], 100.0 * j["update"] / 5000, j["avg_loss_last_16"], j["elapsed_hours"] * 60))
except Exception as e:
    print("no progress yet (%s)" % e.__class__.__name__)
PY
}
while true; do
  st=$(docker inspect -f '{{.State.Status}} {{.State.ExitCode}}' cdeep3m-train-staged 2>/dev/null)
  stages=$(grep -E "^STAGE .* done|FAILED|ALL STAGES" $R/train.log 2>/dev/null | tr '\n' ';')
  if [[ "$st" != running* ]]; then echo "$(date '+%F %T') FINAL container=$st | $(line) | $stages"; break; fi
  echo "$(date '+%F %T') REPORT | $(line) | $stages"
  for i in $(seq 1 60); do sleep 10; docker inspect -f '{{.State.Running}}' cdeep3m-train-staged 2>/dev/null | grep -q true || break; done
done
```
Read it with `cat downloads/train_staged/report.log`. Tip: start long interactive work inside `tmux` so the session
itself survives a disconnect (`tmux new -s work`, later `tmux attach -t work`).

### Measured (Tesla P100, 2026-11-15)
| Stage | Updates | Time |
|---|---|---|
| `--max-iter 1000` | 1 to 1000 | 282 s |
| `--max-iter 3000` | 1001 to 3000 | 551 s |
| `--max-iter 5000` | 3001 to 5000 | 550 s |
| **all** | 5000 | **1383 s (23 min)** |

About 0.2 s per update plus the checkpoint saves; the same recipe on the CPU (i9-11900K, 8 threads) took 3.3 s per
update, about 4.6 hours. Average loss (last 16 updates) at the end: 0.114. No errors; the learning rate at the stage
boundaries matched the earlier CPU run exactly.

### Use the trained model
Export to a `.caffemodel` (this re-estimates the BatchNorm statistics on the training data; the original model only
serves as a layout template, its weights are not used), convert, predict. The new model folder needs the same layout
as a model-zoo model (`1fm/deploy.prototxt`, `1fm/trainedmodel/*.caffemodel`, `1fm/ncnn/`):
```
IMAGE=cdeep3m-anywhere:gpu docker/run.sh downloads bash -c '
  set -e
  M=/data/trained_staged/1fm; mkdir -p $M/trainedmodel; cp /data/SBEM_Mitochondria/1fm/deploy.prototxt $M/
  cd /opt/cdeep3m-anywhere
  python3 -m trainer.export_caffemodel --deploy $M/deploy.prototxt --checkpoint /data/train_staged/stage_5000.pt \
      --template /data/SBEM_Mitochondria/1fm/trainedmodel/1fm_classifer_iter_80000.caffemodel \
      --out $M/trainedmodel/1fm_classifer_iter_5000.caffemodel --data /data/train_data
  convert_model.sh $M/deploy.prototxt $M/trainedmodel/1fm_classifer_iter_5000.caffemodel $M/ncnn'

IMAGE=cdeep3m-anywhere:gpu docker/run.sh downloads runprediction.sh \
    --models 1fm --denoise 0 --augspeed 1 --backend ncnn --device auto \
    /data/trained_staged /data/mito_testsample/validation/images /data/out_val_trained_staged
```
Use `stage_1000.pt` or `stage_3000.pt` to export an earlier stage. Export took 146 s, the prediction of the 10
validation images (8 variants, Vulkan) 107 s. To score against the labels and compare with the pretrained model
(predicted the same way into `out_val_SBEM_Mitochondria`):
```
IMAGE=cdeep3m-anywhere:gpu GPU=off docker/run.sh downloads python3 /opt/cdeep3m-anywhere/tools/compare_predictions.py \
    --a out_val_trained_staged/ensembled --b out_val_SBEM_Mitochondria/ensembled \
    --labels mito_testsample/validation/labels --images mito_testsample/validation/images \
    --out compare_val_trained --name-a Staged5000 --name-b Pretrained
```
Result on the 10 validation images: Dice **0.688** for this model (5000 updates from scratch on 20 images) and
**0.840** for the pretrained model (80000 iterations on its own, larger training set). Report:
`downloads/compare_val_trained/summary.txt`, pictures in `side_by_side/`.

## Troubleshooting
| Problem | Fix |
|---|---|
| `docker build` hangs at `apt-get update` | add `--network host` (as above) |
| `vulkaninfo`: `ERROR_INCOMPATIBLE_DRIVER`, or `--pick-device` prints `cpu` on an NVIDIA machine | the image predates the Vulkan fix (commit 0d67dfc): rebuild it. Start containers with `docker/run.sh`, which passes `--gpus 'all,"capabilities=compute,utility,graphics"'`; plain `--gpus all` may leave out the Vulkan driver |
| `docker/run.sh` does not pass the GPU | force it with `GPU=on docker/run.sh ...` |
| `ERROR ... Create_overlays.py` | only in the original CDeep3M2 Caffe images; this image has the fix. For those images see `docker/Dockerfile.caffe` |
| Permission errors writing to `/data` | the folder must be writable by your user; `docker/run.sh` runs as your user id |

## GPU compatibility
Only the **Tesla P100** (Pascal, driver 470.256.02) was tested in Docker, for both engines. "Expected" rows follow
from what the image contains and the vendors' published support; they were not run.

### ncnn on Vulkan (`cdeep3m-anywhere` image, `--device auto` or `vulkan`)
ncnn needs a Vulkan 1.0+ driver on the host. `--device auto` uses only **discrete** GPUs; an integrated GPU stays on the
CPU (pass `--device vulkan` to force it).

| GPU | Works in this image | Notes |
|---|---|---|
| NVIDIA Pascal: Tesla P100 | **yes, tested** | driver 470, 1 variant 21 s, 8 variants 103 s (11 images) |
| NVIDIA Maxwell, Pascal, Volta, Turing, Ampere, Ada, Hopper, Blackwell (GTX 900 and newer, V100, T4, A100, A40, L40, H100, RTX 20/30/40/50) | expected | any driver with Vulkan; host needs `libGLX_nvidia.so.0` (see Requirements) |
| NVIDIA Kepler (K80, GTX 600/700) | expected | Vulkan is supported up to driver 470 (the last for Kepler) |
| AMD Radeon / Instinct (GCN and newer) | not out of the box | add `mesa-vulkan-drivers` to the image and run with `--device /dev/dri`; not tested |
| Intel integrated (UHD, Iris Xe) | not out of the box | same Mesa change; tested outside Docker on a UHD 750: 9x **slower** than the CPU |
| No GPU | yes | runs on the CPU (int8 when the int8 model exists) |

### Caffe on CUDA
Two CDeep3M2 2.1.5 base images (both CUDA 11.3) cover **different GPUs**; pick the base by GPU (next section).
Driver for both: 450 or newer for datacenter GPUs (418.x and 440.x also accepted), 465 or newer for GeForce/RTX
(`nvidia-smi` shows it).

| GPU (compute capability) | `cdeep3m-2.1.5:U16C11.3` (local) | `ncmir/cdeep3m:C-11_2021_09_21` (Docker Hub) |
|---|---|---|
| Tesla P100 (6.0) | **yes, tested** (1 variant 68 s, 8 variants 375 s) | **no, tested**: `no kernel image is available for execution on the device` |
| P40, P4, GTX 10 (6.1) | expected (compiled code) | no (no code for Pascal) |
| V100 (7.0) | expected (compiled on first use) | no (no code for Volta) |
| Maxwell (5.x) | expected (compiled code) | expected (compiled code) |
| T4, RTX 20 (7.5), A100 (8.0) | expected (compiled code) | expected (compiled code) |
| A40, A10, RTX 30 (8.6), L40, RTX 40 (8.9) | expected (compiled code) | expected (compiled code) |
| H100 (9.0) | expected (compiled on first use) | expected (compiled on first use) |
| Blackwell: B200, RTX 50 (10.x, 12.x) | uncertain (newer than CUDA 11.3) | uncertain |
| Kepler: K80, K40 (3.x) | **no** | **no** |

Compiled code (SASS) in the local image: sm_50, 52, 60, 61, 75, 80, plus PTX up to compute_80. In the Docker Hub image:
sm_50, 75, 80, 86, plus PTX compute_86. Compiled code for sm_XY runs on GPUs X.Z with Z >= Y (same major version);
PTX for compute_XY is compiled by the driver on first use for any GPU of version X.Y or higher. A GPU that neither
covers fails with the error above.
Only the P100 rows were run; the others follow from this list.


On GPUs that compile on first use, keep the compiled code between runs by adding `-e CUDA_CACHE_PATH=/data/.nv_cache`
to `docker run` (otherwise `--rm` and `HOME=/tmp` discard it after every run).

## Optional: the original Caffe/CUDA pipeline for comparison
This needs a CDeep3M2 2.1.5 Caffe image as the base. Choose by GPU (table above):

| Base | Get it | Use for |
|---|---|---|
| `ncmir/cdeep3m:C-11_2021_09_21` | public: `docker pull ncmir/cdeep3m:C-11_2021_09_21` (about 4 GB download) | Maxwell, Turing, Ampere and newer (T4, A100, A40, RTX 20/30/40, L40, H100). **Not** Pascal or Volta |
| `cdeep3m-2.1.5:U16C11.3` | only on the server where it was built (2021, not in any registry); copy it (see below) | everything above **plus Pascal and Volta** (P100, P40, V100) |

Both have the same `Create_overlays.py`, which fails at the last step on 3-channel PNGs such as the test images
(`Invalid number of channels in input image`), so the build below adds the fix in
`patches/cdeep3m2-create-overlays-fix.patch`. Everything else (paths, entrypoint, commands) is the same for both.

### Build (once)
With the local base (Pascal/Volta and newer):
```
docker build --network host -f docker/Dockerfile.caffe -t cdeep3m-caffe:fixed \
    --build-arg BASE=cdeep3m-2.1.5:U16C11.3 .
```
With the public base (Turing/Ampere and newer; Docker pulls it if needed):
```
docker build --network host -f docker/Dockerfile.caffe -t cdeep3m-caffe:fixed \
    --build-arg BASE=ncmir/cdeep3m:C-11_2021_09_21 .
```
This adds one small layer (the patched `Create_overlays.py`) on top of the base image and takes a few seconds.
Check that the fix is in and the GPU is visible:
```
docker run --rm --entrypoint grep cdeep3m-caffe:fixed -c "palette/RGB" /home/cdeep3m/Create_overlays.py   # prints 1
docker run --rm --gpus all --entrypoint nvidia-smi cdeep3m-caffe:fixed -L                                 # lists the GPU
```

### Predict
```
docker run --rm --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp --shm-size=2g -v "$PWD/downloads":/data \
    --entrypoint /home/cdeep3m/runprediction.sh cdeep3m-caffe:fixed \
    --models 1fm --denoise 0 --augspeed 1 /data/SBEM_Mitochondria /data/mito_testsample/testset /data/out_caffe
```
* The base image's entrypoint is `top -b`, hence `--entrypoint`.
* It uses the original `.caffemodel` (no conversion) and all visible GPUs; to use one, add `--gpu 0` before
  `--models`.
* Results as for ncnn: `downloads/out_caffe/ensembled/`, `overlay/`, `DONE`. On the P100 the segmentation is identical
  to ncnn's (Dice 1.0000).

### Use the image on another machine
With a Turing/Ampere or newer GPU, build from the public base there (above); nothing has to be copied.
For Pascal or Volta you need the local base, which exists only where it was built. Copy it instead of rebuilding (the other machine does not need the base image):
```
docker save cdeep3m-caffe:fixed | gzip > cdeep3m-caffe-fixed.tar.gz      # on this machine (several GB)
docker load < cdeep3m-caffe-fixed.tar.gz                                # on the other machine
```
or push it to a registry your group uses. For a cluster without Docker, convert it with Singularity/Apptainer
(see [docker/README.md](../docker/README.md)) and run with `--nv`. Check the GPU against the table above first.
