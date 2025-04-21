#!/bin/bash
#SBATCH --job-name=audio_audiomnist
#SBATCH --output=slurm/logs/train_audiomnist_%j.out
#SBATCH --error=slurm/logs/train_audiomnist_%j.err
#SBATCH --partition=gpu2v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00

# Print start time
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"

# Load Anaconda module
echo "Loading Anaconda module..."
module load anaconda3

# Activate your environment
echo "Activating conda environment..."
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate myvm

# Verify the environment
echo "Conda environment:"
conda env list

# Verify GPU availability
echo "GPU information:"
nvidia-smi

# Create necessary directories if they don't exist
mkdir -p slurm/logs

# Run the audioMNIST training script
echo "Running audioMNIST Training script..."
python -m src.models.train_audiomnist_model

# Print completion message
echo "Training completed successfully!"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"