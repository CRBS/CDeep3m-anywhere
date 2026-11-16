#!/bin/bash
# Prediction on one Expanse V100 (gpu-shared). Edit the account, then:
#   sbatch expanse_predict.sbatch <data folder> <model folder> <images folder> <output folder>
# The model, images and output folders are given relative to <data folder> (it is mounted at /data).
# Example: sbatch expanse_predict.sbatch /expanse/lustre/projects/abc123/$USER/cdeep3m SBEM_Mitochondria mito_testsample/testset out_testset
#SBATCH --job-name=cdeep3m-predict
#SBATCH --account=CHANGE_ME
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --mem=92G
#SBATCH --time=01:00:00
#SBATCH --output=%x.%j.out
set -e
[ $# -eq 4 ] || { echo "usage: sbatch $0 <data folder> <model> <images> <output>" >&2; exit 1; }
module purge
module load singularitypro
here=${SLURM_SUBMIT_DIR:-$(pwd)}
export SIF=${SIF:-$here/cdeep3m-anywhere.sif}
run=$here/run.sh
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
$run "$1" gpu_check.sh || echo "WARNING: GPU check failed, see above"
t0=$(date +%s)
$run "$1" runprediction.sh --models 1fm --denoise 0 --augspeed 1 --backend ncnn --device auto \
    "/data/$2" "/data/$3" "/data/$4"
echo "prediction took $(( $(date +%s)-t0 )) s"
grep -h "Running prediction with" "$1/$4"/*/logs/prediction.log 2>/dev/null | sort | uniq -c || true
ls "$1/$4"/DONE "$1/$4"/ERROR 2>/dev/null || true
