# Singularity image (SDSC Expanse, V100)

A Singularity/Apptainer image of the CDeep3M-anywhere Docker image (`docker/Dockerfile`, GPU variant with PyTorch
2.7.1 + CUDA 11.8), for clusters without Docker, such as SDSC Expanse. Same contents: the patched CDeep3M2, ncnn
(CPU and Vulkan), the PyTorch trainer. No model weights, no data.

| File | What |
|---|---|
| `build.sh` | builds `cdeep3m-anywhere.sif` from the local Docker image (no root needed) |
| `Dockerfile` | the thin layer `build.sh` adds: `gpu_check.sh` and `PYTHONNOUSERSITE=1` |
| `install_apptainer_local.sh` | installs Apptainer without root into `~/apptainer` (for gaia, see below) |
| `gpu_check.sh` | run inside the image: checks PyTorch CUDA (training) and Vulkan (ncnn prediction) |
| `run.sh` | like `docker/run.sh`: mounts a data folder at `/data`, adds `--nv` when there is a GPU |
| `expanse_predict.sbatch` | Slurm job: prediction on one V100 |
| `expanse_train.sbatch` | Slurm job: staged training (1000 + 2000 + 2000 updates) on one V100 |

## V100 support
The V100 is compute capability 7.0 (`sm_70`).
* **Training** (PyTorch 2.7.1+cu118) has `sm_70` kernels. Needs NVIDIA driver 450.80 or newer on the node.
* **Prediction** (ncnn Vulkan) does not depend on the CUDA version; it needs the node's NVIDIA driver Vulkan
  libraries, which `--nv` mounts. `gpu_check.sh` shows whether that works.
* The optional Caffe image (`docker/Dockerfile.caffe`) is not part of this; the local Caffe build has no `sm_70` code.

## What is tested (2026-11-16, on gaia, Tesla P100, driver 470)
* Built with gaia's Singularity 3.8.3 from `cdeep3m-anywhere:gpu`: 3.8 GB `.sif` in 19 minutes (temporary files on NFS).
* Run with Apptainer 1.3.6 installed without root (`install_apptainer_local.sh`, see below), through `run.sh` (`--nv`):
  * `gpu_check.sh`: `== ALL OK` (PyTorch CUDA works; Vulkan works with Apptainer's default `nvliblist.conf`;
    `--device auto` picks `vulkan 0`).
  * Prediction of the 11 test images (1fm, 8 variants): 73 s, `ncnn on vulkan (GPU 0)`, output identical to the
    Docker run (0 of 11.5 million pixels differ).
  * Training: 50 updates on CUDA, 0.2 s per update (same as Docker), checkpoints written.
* Not tested: Expanse and the V100. Run step 3 there first.

### Running a .sif on gaia
gaia's own Singularity can build images but not run them (`starter-suid not found`: its `libexec` files are missing).
`singularity/install_apptainer_local.sh` puts Apptainer into `~/apptainer` without root (it downloads the Apptainer
`.deb` from GitHub and, because gaia has no `libfuse3`, Ubuntu 20.04's `libfuse3-3`), then:
```
export PATH=~/apptainer/usr/bin:$PATH
singularity/run.sh downloads gpu_check.sh
```

## 1. Build (on a machine with Docker and Singularity, e.g. gaia)
Build the Docker image first (docs/DOCKER_INSTALL.md step 3), then:
```
singularity/build.sh                      # -> singularity/cdeep3m-anywhere.sif
DOCKER_BUILD=1 singularity/build.sh       # also (re)builds the Docker image cdeep3m-anywhere:gpu
```
Temporary files (about 20 GB) go to `singularity/build_tmp`; set `TMPBASE=/other/disk` to move them.
Expanse login nodes have no Docker and do not allow building, so build elsewhere and copy the `.sif`.

## 2. Copy to Expanse
```
ssh <user>@login.expanse.sdsc.edu mkdir -p /expanse/lustre/projects/<project>/<user>/cdeep3m
rsync -av --progress singularity/cdeep3m-anywhere.sif singularity/run.sh singularity/*.sbatch \
    <user>@login.expanse.sdsc.edu:/expanse/lustre/projects/<project>/<user>/cdeep3m/
rsync -av downloads/SBEM_Mitochondria downloads/mito_testsample \
    <user>@login.expanse.sdsc.edu:/expanse/lustre/projects/<project>/<user>/cdeep3m/data/
```
Put the converted model (`SBEM_Mitochondria/1fm/ncnn`, made in docs/DOCKER_INSTALL.md step 5) in the copy, or
convert on Expanse with `run.sh <data> convert_model.sh ...`. Use your project's Lustre space, not your home
(small quota). The paths above follow SDSC's documentation; check them with `ls /expanse/lustre/projects`.

## 3. Check the GPU (interactive)
```
srun --account=<project> --partition=gpu-debug --gpus=1 --cpus-per-task=10 --mem=92G --time=00:20:00 --pty bash
module purge; module load singularitypro       # `module spider singularity` if the name differs
cd /expanse/lustre/projects/<project>/<user>/cdeep3m
./run.sh data gpu_check.sh
```
It ends with `== ALL OK` when both training (CUDA) and prediction (Vulkan) can use the GPU.

## 4. Predict (batch job)
Set `#SBATCH --account` in the script, then:
```
sbatch expanse_predict.sbatch $PWD/data SBEM_Mitochondria mito_testsample/testset out_testset
```
The log `cdeep3m-predict.<jobid>.out` shows the GPU, the check, the time, and which device every package used
(`Running prediction with ncnn on vulkan (GPU 0)`). Results: `data/out_testset/ensembled/Segmented_*.png`.

Interactive, the same as with Docker:
```
./run.sh data runprediction.sh --models 1fm --denoise 0 --augspeed 1 --backend ncnn --device auto \
    /data/SBEM_Mitochondria /data/mito_testsample/testset /data/out_testset
```

## 5. Train (batch job)
Prepare the training stacks once (CPU only, can run in the same kind of job or interactively):
```
./run.sh data bash -c 'cd /opt/cdeep3m2 && python3 PreprocessTrainingData.py \
    /data/mito_testsample/training/images /data/mito_testsample/training/labels 0 /data/train_data'
sbatch expanse_train.sbatch $PWD/data
```
Results in `data/train_staged/` (`stage_1000.pt`, `stage_3000.pt`, `stage_5000.pt`, `progress/1fm.json`). If the job
stops early, submit it again; it resumes. `STAGES="1000 3000 5000"` (environment, passed by `sbatch`) sets the
stages. Export, convert and compare as in docs/DOCKER_INSTALL.md section 7, prefixing the commands with
`./run.sh data` instead of `docker/run.sh downloads`.

## Notes
* The image is read-only. Write everything under `/data` (the mounted folder) or `/tmp`.
* Singularity also mounts your home folder. `PYTHONNOUSERSITE=1` (set in the image) keeps packages from
  `~/.local` out of the container's Python.
* `GPU=off ./run.sh ...` runs without `--nv` (CPU only). `EXTRA_BINDS=/expanse/lustre/scratch/$USER` mounts more folders.
* If `gpu_check.sh` says Vulkan is not usable, the node's Singularity does not mount the NVIDIA Vulkan driver files
  (`libGLX_nvidia.so.0`, `libnvidia-glvkspirv.so`, ...; set in Singularity's `nvliblist.conf`). Prediction then falls
  back to the CPU (much slower); training is not affected.
