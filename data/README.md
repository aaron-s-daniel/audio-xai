# External Data

This project uses external datasets located outside the repository to avoid duplication.

## Dataset Locations

- **AudioMNIST**: Located at `/home/mqa887/tmp/audiomnist/AudioMNIST-master/data`
- **MNIST**: Update path in configs/data_config.py file

## Dataset Structure

### AudioMNIST
The AudioMNIST dataset should contain folders numbered 00-09 (for each digit),
with each folder containing .wav files of spoken digits.

### MNIST
Standard MNIST dataset with training and test sets.

## Configuration

All dataset paths are configured in `configs/data_config.py`. Update this file if your dataset locations differ.
