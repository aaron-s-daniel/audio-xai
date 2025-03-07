import os

# Define the enhanced folder structure
folders = [
    "src/models",
    "src/xai_methods",
    "src/metrics",
    "src/utils",
    "experiments/objective1",
    "experiments/objective2",
    "notebooks",
    "results/configs",
    "results/logs",
    "results/metrics", 
    "results/visualizations",
    "results/summaries",
    "checkpoints",
    "scripts",
    # Add cluster-specific directories
    "slurm",
    "slurm/logs",
    "env_setup",
    "configs"
]

# Enhanced descriptions including cluster-specific elements
descriptions = {
    "src": "Source code for the project.",
    "src/models": "Model architectures for audio and image classification.",
    "src/xai_methods": "Implementation of XAI methods (Saliency, Integrated Gradients, Guided GradCAM).",
    "src/metrics": "Evaluation metrics for XAI methods (Faithfulness, Sensitivity, Sparseness, Randomness).",
    "src/utils": "Utility functions for data handling, visualization, etc.",
    "experiments": "Experiment scripts for Objectives 1 and 2.",
    "experiments/objective1": "Cross-modality comparison experiments (RQ1).",
    "experiments/objective2": "Audio-specific adaptation experiments (RQ2).",
    "notebooks": "Jupyter notebooks for analysis and visualization.",
    "results": "Experimental results including configurations, logs, metrics, and visualizations.",
    "results/configs": "Experiment configuration files.",
    "results/logs": "Experiment logs recording progress and outcomes.",
    "results/metrics": "Computed metrics for each experiment.",
    "results/visualizations": "Generated visualizations of explanations and metrics.",
    "results/summaries": "Summary reports of experiment findings.",
    "checkpoints": "Model checkpoints from training.",
    "scripts": "General utility scripts for the project.",
    "slurm": "Slurm job scripts for cluster computing.",
    "slurm/logs": "Output and error logs from Slurm jobs.",
    "env_setup": "Scripts and configuration for virtual environment setup.",
    "configs": "Global configuration files including cluster parameters and experiment settings."
}

# Create a more comprehensive data config file
data_config = """# Data Configuration File

# External dataset paths - modify these to match your environment
AUDIOMNIST_PATH = "/home/mqa887/tmp/audiomnist/AudioMNIST-master/data"
MNIST_PATH = "/path/to/mnist/data"  # Update this with your MNIST location

# Dataset parameters
AUDIO_SAMPLE_RATE = 16000
AUDIO_DURATION = 1.0
MEL_BINS = 128
TIME_STEPS = 128

# Data caching settings (for faster loading)
CACHE_DIR = "/tmp/audio_xai_cache"  # Consider using scratch space on cluster
USE_CACHED_SPECTROGRAMS = True
"""

# Create a sample Slurm script template
slurm_template = """#!/bin/bash
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
"""

# Create a virtualenv setup script
venv_setup = """#!/bin/bash
# Script to set up virtual environment for Audio XAI project

# Create virtual environment
python -m venv /path/to/your/venv

# Activate environment
source /path/to/your/venv/bin/activate

# Install required packages
pip install --upgrade pip
pip install -r requirements.txt

echo "Virtual environment setup complete!"
"""

# Create folders and markdown files
for folder in folders:
    # Create folder
    os.makedirs(folder, exist_ok=True)
    print(f"Created: {folder}")
    
    # Create README.md in each folder
    folder_name = folder.split("/")[-1]
    parent = "/".join(folder.split("/")[:-1])
    
    with open(f"{folder}/README.md", "w") as f:
        f.write(f"# {folder_name.capitalize()}\n\n")
        
        # Get description for this specific folder
        if folder in descriptions:
            desc = descriptions[folder]
        else:
            # Try to get parent folder description
            desc = descriptions.get(parent, "")
        
        f.write(f"{desc}\n\n")
        f.write(f"## Purpose\n\n")
        f.write(f"This directory contains files related to {folder_name} for the Audio XAI thesis project.\n\n")
        
        # Add specific content instructions based on folder
        if "models" in folder:
            f.write("## Expected Content\n\n")
            f.write("- CNN model architectures for audio spectrograms\n")
            f.write("- CNN model architectures for MNIST images\n")
        elif "xai_methods" in folder:
            f.write("## Expected Content\n\n")
            f.write("- Saliency Maps implementation\n")
            f.write("- Integrated Gradients implementation\n")
            f.write("- Guided GradCAM implementation\n")
            f.write("- Audio-specific adaptations\n")
        elif "metrics" in folder and not "results" in folder:
            f.write("## Expected Content\n\n")
            f.write("- Faithfulness correlation implementation\n")
            f.write("- Sensitivity/robustness metric implementation\n")
            f.write("- Sparseness measurement implementation\n")
            f.write("- Randomization test implementation\n")
        elif folder == "slurm":
            f.write("## Slurm Job Submission\n\n")
            f.write("This directory contains Slurm job scripts for cluster computing.\n\n")
            f.write("### Usage\n\n")
            f.write("Submit a job using:\n")
            f.write("```bash\n")
            f.write("sbatch slurm/run_experiment.sh\n")
            f.write("```\n\n")
            f.write("### Job Monitoring\n\n")
            f.write("Check job status with:\n")
            f.write("```bash\n")
            f.write("squeue -u $USER\n")
            f.write("```\n")
        elif folder == "env_setup":
            f.write("## Virtual Environment Setup\n\n")
            f.write("This directory contains scripts for setting up the Python virtual environment.\n\n")
            f.write("### Setup Instructions\n\n")
            f.write("1. Review and modify `setup_venv.sh` with your desired paths\n")
            f.write("2. Run the setup script:\n")
            f.write("```bash\n")
            f.write("bash env_setup/setup_venv.sh\n")
            f.write("```\n")
    
    print(f"Created README.md in {folder}")

# Create a specific README for a placeholder data directory
os.makedirs("data", exist_ok=True)
with open("data/README.md", "w") as f:
    f.write("# External Data\n\n")
    f.write("This project uses external datasets located outside the repository to avoid duplication.\n\n")
    f.write("## Dataset Locations\n\n")
    f.write("- **AudioMNIST**: Located at `/home/mqa887/tmp/audiomnist/AudioMNIST-master/data`\n")
    f.write("- **MNIST**: Update path in configs/data_config.py file\n\n")
    f.write("## Dataset Structure\n\n")
    f.write("### AudioMNIST\n")
    f.write("The AudioMNIST dataset should contain folders numbered 00-09 (for each digit),\n")
    f.write("with each folder containing .wav files of spoken digits.\n\n")
    f.write("### MNIST\n")
    f.write("Standard MNIST dataset with training and test sets.\n\n")
    f.write("## Configuration\n\n")
    f.write("All dataset paths are configured in `configs/data_config.py`. ")
    f.write("Update this file if your dataset locations differ.\n")

# Create the data_config.py file
os.makedirs("configs", exist_ok=True)
with open("configs/data_config.py", "w") as f:
    f.write(data_config)

# Create sample Slurm script
with open("slurm/run_experiment.sh", "w") as f:
    f.write(slurm_template)

# Create virtualenv setup script
with open("env_setup/setup_venv.sh", "w") as f:
    f.write(venv_setup)

# Create requirements.txt
requirements = """
# Core dependencies
numpy
scipy
matplotlib
pandas
scikit-learn

# Deep learning
torch>=1.10.0
torchvision>=0.11.0

# Audio processing
librosa>=0.8.1
soundfile

# Data handling
h5py
pytest
pyyaml

# Visualization
seaborn
plotly

# Jupyter
jupyter
ipywidgets
"""

with open("requirements.txt", "w") as f:
    f.write(requirements)

print("Enhanced folder structure creation complete!")
print("Added Slurm scripts, virtual environment setup, and configuration files")