#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --job-name=qec_${CODE}_${TRICKY}
#SBATCH --output=logs/qec_${CODE}_${TRICKY}_%j.out
#SBATCH --account=REPLACE_WITH_YOUR_ACCOUNT

source venv/bin/activate
python main.py --code_type ${CODE} --tricky ${TRICKY}