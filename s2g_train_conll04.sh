#!/bin/bash
#SBATCH --job-name=s2g_train_conll04
#SBATCH --output=outputs/finetune/conll04/s2g_run_%j.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00

echo "Targeting Physical DGX GPU Slot(s): $SLURM_JOB_GPUS"
/home/bt19d200/Ayaan/ddp-S2G/s2g_env/bin/python -u -m s2g.scripts.train --config configs/conll04.yaml

# Exit the script immediately if any internal command fails
set -e

# Automatically kill all background child processes if this script exits or crashes
trap 'kill 0' EXIT