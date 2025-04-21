#!/bin/bash
#SBATCH --job-name=MNISTxai
#SBATCH --output=slurm/logs/MNISTxai_%j.out
#SBATCH --error=slurm/logs/MNISTxai_%j.err
#SBATCH --partition=gpu1v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00

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


# Run the MNISTxai experiments
echo "Running MNISTxai experiments..."
python -m src.xai.objective2_experiments_z

# Print completion message
echo "Objective 2 experiments completed successfully!"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"  