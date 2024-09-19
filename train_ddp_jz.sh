#!/bin/bash

#SBATCH --job-name=wavlm_ssl_sv
#SBATCH --output=slurm_%j
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --constraint=a100
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH --account=kdp@a100

module purge

module load cpuarch/amd
module load pytorch-gpu/py3/1.12.1

srun python -u trainSpeakerNet.py --config configs/wavlm_mhfa_dlg_lc.yaml --train_list exp/train_list_dino.txt --distributed