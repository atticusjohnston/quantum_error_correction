#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=Test
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --job-name=qec_${CODE}_${TRICKY}
#SBATCH --output=logs/qec_${CODE}_${TRICKY}_%j.out

source venv/bin/activate
python main.py --code_type ${CODE} --tricky ${TRICKY}