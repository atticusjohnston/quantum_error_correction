#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=a100-test
#SBATCH --job-name=qec_%j
#SBATCH --output=logs/qec_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=atticus.johnston@student.uq.edu.au

source venv/bin/activate
python main.py --code_type ${CODE} --tricky ${TRICKY}