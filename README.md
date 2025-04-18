# Audio XAI: Explainable AI Methods for Audio Classification

This repository contains code and documentation for my Master's Thesis research on explainable AI (XAI) methods for audio data classification.

## Overview

This research investigates the performance of post-hoc explanation methods across different data modalities, with a specific focus on comparing traditional image-based XAI methods with their application to audio spectrograms. The study aims to develop improvements to existing XAI methods specifically tailored for audio data analysis.

## Research Questions

1. How do state-of-the-art post-hoc evaluation methods perform on audio spectrograms compared to image data, and how does data resolution impact these performances?
2. How can existing XAI approaches (particularly Integrated Gradients) be improved for better performance on audio data?

## Repository Structure

```
.
├── data/                     # Dataset storage directory
│   ├── audio_mnist/          # AudioMNIST dataset
│   └── mnist/                # MNIST dataset
├── src/                      # Source code
│   ├── datasets/             # Dataset loading and preprocessing
│   ├── models/               # Model architectures
│   ├── xai_methods/          # Implementation of XAI methods
│   ├── metrics/              # Evaluation metrics for XAI methods
│   └── utils/                # Utility functions
├── experiments/              # Experiment scripts
│   ├── objective1/           # Cross-modality comparison experiments
│   └── objective2/           # Audio-specific adaptation experiments
├── notebooks/                # Jupyter notebooks for analysis
├── results/                  # Experimental results
│   ├── configs/              # Experiment configurations
│   ├── logs/                 # Experiment logs
│   ├── metrics/              # Computed metrics
│   ├── visualizations/       # Generated visualizations
│   └── summaries/            # Experiment summaries
├── checkpoints/              # Model checkpoints
├── scripts/                  # Utility scripts
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Datasets

This research focuses on two primary datasets:

1. **AudioMNIST**: 30,000 audio recordings of spoken digits (0-9) from 60 different speakers
2. **MNIST**: 70,000 handwritten digit images (0-9)

## XAI Methods

The following XAI methods are implemented and evaluated:

1. **Saliency Maps**: A simple gradient-based attribution method
2. **Integrated Gradients**: An axiomatic attribution method that satisfies implementation invariance and sensitivity
3. **Guided GradCAM**: A class-discriminative localization technique that combines guided backpropagation with class activation mapping

## Evaluation Metrics

The performance of XAI methods is evaluated using the following metrics:

1. **Faithfulness Correlation**: Measures correlation between feature importance and prediction impact
2. **Average Sensitivity (Robustness)**: Measures stability of explanations under input perturbations
3. **Sparseness (Complexity)**: Measures how focused/concentrated the explanation is
4. **Randomness**: Tests if explanations are model-dependent rather than random

## Setup and Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.10+
- CUDA-capable GPU (recommended)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/audio-xai.git
cd audio-xai
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download and prepare datasets:
```bash
python scripts/prepare_datasets.py
```

## Running Experiments

### Objective 1: Cross-Modality Comparison

```bash
# Generate explanations using all methods on MNIST
python experiments/objective1/mnist_explanations.py

# Generate explanations using all methods on AudioMNIST
python experiments/objective1/audiomnist_explanations.py

# Compute evaluation metrics and perform comparison
python experiments/objective1/compute_metrics.py
python experiments/objective1/comparison_analysis.py
```

### Objective 2: Audio-Specific Adaptations

```bash
# Implement and test audio-specific adaptations to Integrated Gradients
python experiments/objective2/audio_adapted_ig.py

# Evaluate adapted methods
python experiments/objective2/evaluate_adaptations.py
```

## Results Visualization

```bash
# Generate visualizations for thesis
python scripts/generate_thesis_figures.py

# Run interactive analysis notebook
jupyter notebook notebooks/results_analysis.ipynb
```

## Contributing

This repository is primarily for academic research purposes. If you'd like to contribute or have questions, please open an issue.

## License

[MIT License](LICENSE)

## Acknowledgements

- The AudioMNIST dataset from [Becker et al., 2018]
- The MNIST dataset from [LeCun et al., 1998]
- PyTorch and the open-source machine learning community