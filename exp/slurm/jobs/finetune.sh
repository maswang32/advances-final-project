#!/bin/bash
#SBATCH --nodelist=huang-l40s-1
#SBATCH --job-name=finetune
#SBATCH --account=hi-res
#SBATCH --partition=hi-res
#SBATCH --qos=hi-res-main
#SBATCH --time=23:59:59
#SBATCH --output=finetune.log
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G

cd /data/scratch/ycda/gen/advances_project
conda activate advances
python fine_tuning.py
