#!/bin/bash
# Script to set up virtual environment for Audio XAI project

# Create virtual environment
python -m venv /path/to/your/venv

# Activate environment
source /path/to/your/venv/bin/activate

# Install required packages
pip install --upgrade pip
pip install -r requirements.txt

echo "Virtual environment setup complete!"
