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

## Troubleshooting
| Problem | Fix |
|---|---|
| `docker build` hangs at `apt-get update` | add `--network host` (as above) |
| `vulkaninfo`: `ERROR_INCOMPATIBLE_DRIVER`, or `--pick-device` prints `cpu` on an NVIDIA machine | the image predates the Vulkan fix (commit 0d67dfc): rebuild it. Start containers with `docker/run.sh`, which passes `--gpus 'all,"capabilities=compute,utility,graphics"'`; plain `--gpus all` may leave out the Vulkan driver |
| `docker/run.sh` does not pass the GPU | force it with `GPU=on docker/run.sh ...` |
| `ERROR ... Create_overlays.py` | only in the original CDeep3M2 Caffe images; this image has the fix. For those images see `docker/Dockerfile.caffe` |
| Permission errors writing to `/data` | the folder must be writable by your user; `docker/run.sh` runs as your user id |

## Optional: the original Caffe/CUDA pipeline for comparison
If you have an existing CDeep3M2 Caffe image (for example `cdeep3m-2.1.5:U16C11.3`), add the overlay fix and run it
on the same data:
```
docker build --network host -f docker/Dockerfile.caffe -t cdeep3m-caffe:fixed --build-arg BASE=cdeep3m-2.1.5:U16C11.3 .
docker run --rm --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp --shm-size=2g -v "$PWD/downloads":/data \
    --entrypoint /home/cdeep3m/runprediction.sh cdeep3m-caffe:fixed \
    --models 1fm --denoise 0 --augspeed 1 /data/SBEM_Mitochondria /data/mito_testsample/testset /data/out_caffe
```
