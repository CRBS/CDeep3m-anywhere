# Docker environment

One image with everything this repository needs: the patched CDeep3M2 pipeline, ncnn (Python wheel plus the
conversion tools), the PyTorch trainer and the comparison tools. It contains no model weights and no data.

The image is based on Ubuntu 24.04 and does not depend on the host's OS version, so the same image runs on
Ubuntu 20.04, 22.04, 24.04 or any other Linux with Docker. Only a GPU depends on the host (its driver).

## Build
From the repository root:

```
docker build -f docker/Dockerfile -t cdeep3m-anywhere:cpu .
```

This is enough for prediction on the CPU, prediction on a Vulkan GPU, model conversion and CPU training.

### Optional: PyTorch with CUDA
Pick the PyTorch build that matches the host's NVIDIA **driver** (`nvidia-smi` shows it). The CUDA libraries
come inside the PyTorch wheel; nothing else has to be installed on the host.

| Host driver | Build arguments |
|---|---|
| 450.80 to 524 (for example 470.x) | `--build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu118 --build-arg TORCH_SPEC="torch==2.7.1"` |
| 525.60 or newer | `--build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126` (the cu118 build above also works) |

```
docker build -f docker/Dockerfile -t cdeep3m-anywhere:cu118 \
    --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu118 --build-arg TORCH_SPEC="torch==2.7.1" .
```

Note: `trainer/train.py` runs on the CPU only at the moment. The CUDA image is the environment for GPU
training once the trainer has a device option; it does not make training faster by itself.

## Docker Compose (the simplest way)
`docker/docker-compose.yml` has two services: `cpu` (any machine) and `gpu` (NVIDIA; falls back to the CPU).

```
docker compose -f docker/docker-compose.yml build gpu
DATA=/path/to/data UID_GID="$(id -u):$(id -g)" docker compose -f docker/docker-compose.yml run --rm gpu
```

Inside, `tools/train_bg.sh --torch auto ...` detects the hardware and picks the PyTorch device.

## Run
```
docker/run.sh /path/to/data                       # interactive shell, the folder is mounted at /data
docker/run.sh /path/to/data runprediction.sh --models 1fm --denoise 0 --augspeed 1 \
      --backend ncnn --device cpu /data/model /data/images /data/out
```

`docker/run.sh` passes the NVIDIA GPUs automatically when the host has the NVIDIA container toolkit, and
runs as your own user so the results belong to you. Set `IMAGE=cdeep3m-anywhere:cu118` to use another image,
`GPU=off` to hide the GPUs.

Inside the container:

| What | Where |
|---|---|
| This repository | `/opt/cdeep3m-anywhere` (`ncnn/` and `tools/` are on `PATH`) |
| Patched CDeep3M2 | `/opt/cdeep3m2` (on `PATH`; also `$CDEEP3M2`) |
| ncnn tools | `$NCNN_BUILD` = `/opt/ncnn_build` |
| `caffe.proto` of the CDeep3M Caffe fork | `$CAFFE_PROTO` = `/opt/caffe.proto` |

Examples (paths under `/data` are yours):

```
# convert a model, with int8
make_calibration.py --repo $CDEEP3M2 --images /data/train/images --out /data/calib
convert_model.sh /data/model/1fm/deploy.prototxt /data/model/1fm/trainedmodel/x.caffemodel /data/model/1fm/ncnn /data/calib/list.txt

# train (resumes from /data/run/latest.pt when it exists)
python -m trainer.train --model 1fm --deploy /data/model/1fm/deploy.prototxt --data /data/augmented --out /data/run
```

## GPU on the host
* **NVIDIA**: install the driver and the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
  then check with `docker run --rm --gpus all cdeep3m-anywhere:cpu nvidia-smi`. For ncnn on the GPU check
  `docker run --rm --gpus all cdeep3m-anywhere:cpu vulkaninfo --summary` and use `--device vulkan`.
* **AMD / Intel (Mesa)**: pass the render device, `docker run --device /dev/dri ...`, and install
  `mesa-vulkan-drivers` in the image (add it to the `apt-get install` line).
* **No GPU**: nothing to do; `--device cpu` works everywhere.

## Singularity / Apptainer (HPC)
Build the Docker image on a machine where you have Docker, then convert it:

```
docker save cdeep3m-anywhere:cpu -o cdeep3m-anywhere.tar
singularity build cdeep3m-anywhere.sif docker-archive://cdeep3m-anywhere.tar
singularity exec --nv -B /path/to/data:/data cdeep3m-anywhere.sif runprediction.sh ...
```

## What is tested
Status (2026-11-15):

* Tested on a machine without a GPU, both services built with `docker compose` (`cpu` 2.6 GB with
  torch 2.15 CPU, `gpu` 10 GB with torch 2.7.1+cu118), run as a normal user (`--user 1000:1000`):
  1. 1fm conversion to ncnn with int8 calibration
  2. `runprediction.sh` with `--backend ncnn --device cpu` (int8) on 3 validation images, Dice 0.7142
  3. `tools/train_bg.sh --torch auto` for 3 updates (chose the CPU in both images, 3.2 s per update)
  4. `trainer.export_caffemodel` with BatchNorm re-estimation
  5. 3fm conversion (multi-frame path)
* Not tested: training or prediction on a GPU inside a container (CUDA, ROCm, Vulkan), the interactive
  question of `--torch auto`, 5fm, long training runs, the Singularity conversion.
