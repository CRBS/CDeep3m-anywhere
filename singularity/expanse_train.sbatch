#!/bin/bash
# Staged 1fm training on one Expanse V100 (gpu-shared): 1000, then 2000, then 2000 updates on one learning-rate
# schedule, as in docs/DOCKER_INSTALL.md section 7. Edit the account, then:
#   sbatch expanse_train.sbatch <data folder>
# <data folder> must contain SBEM_Mitochondria/ (the model, for deploy.prototxt) and train_data/ (the augmented
# training stacks from PreprocessTrainingData.py). Results go to <data folder>/train_staged.
# If the job ends early (time limit, node failure), submit it again: every stage resumes from run/latest.pt.
#SBATCH --job-name=cdeep3m-train
#SBATCH --account=CHANGE_ME
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --mem=92G
#SBATCH --time=02:00:00
#SBATCH --output=%x.%j.out
set -e
[ $# -eq 1 ] || { echo "usage: sbatch $0 <data folder>" >&2; exit 1; }
module purge
module load singularitypro
here=${SLURM_SUBMIT_DIR:-$(pwd)}
export SIF=${SIF:-$here/cdeep3m-anywhere.sif}
STAGES=${STAGES:-"1000 3000 5000"}
$here/run.sh "$1" gpu_check.sh || echo "WARNING: GPU check failed, see above"
$here/run.sh "$1" bash -c "
  cd /opt/cdeep3m-anywhere
  R=/data/train_staged; mkdir -p \$R
  t_all=\$(date +%s)
  for m in $STAGES; do
    t0=\$(date +%s); echo \"STAGE max-iter \$m start \$(date '+%F %T')\"
    python3 -m trainer.train --model 1fm --deploy /data/SBEM_Mitochondria/1fm/deploy.prototxt --data /data/train_data \
        --out \$R/run --max-iter \$m --iter-size 1 --base-lr 0.01 --device cuda --threads ${SLURM_CPUS_PER_TASK:-8} \
        --progress-dir \$R/progress || { echo \"STAGE \$m FAILED\"; exit 1; }
    cp \$R/run/latest.pt \$R/stage_\${m}.pt
    echo \"STAGE max-iter \$m done in \$(( \$(date +%s)-t0 )) s\"
  done
  echo \"ALL STAGES DONE in \$(( \$(date +%s)-t_all )) s\"
"
