#!/bin/bash
#SBATCH --job-name=audio_xai_%j
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu

# Load modules
module purge
module load cuda/11.3
module load python/3.8

# Activate virtual environment
source /path/to/your/venv/bin/activate

# Run experiment
python experiments/objective1/run_experiment.py --config configs/experiment_config.json

# Exit
exit 0
