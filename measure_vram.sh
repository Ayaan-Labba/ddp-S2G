#!/bin/bash
#SBATCH --job-name=measure_vram
#SBATCH --output=outputs/finetune/nyt/s2g_run_%j.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:10:00

echo "Targeting Physical DGX GPU Slot(s): $SLURM_JOB_GPUS"
/home/bt19d200/Ayaan/ddp-S2G/s2g_env/bin/python -u -m s2g.scripts.measure_vram --config configs/nyt.yaml