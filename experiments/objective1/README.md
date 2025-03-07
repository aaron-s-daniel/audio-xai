# Objective 1: Cross-Modality Comparison of XAI Methods

This directory contains experiments related to the first research objective: comparing the performance of post-hoc explanation methods on audio spectrograms versus image data.

## Research Question

**RQ1**: How do state-of-the-art post-hoc evaluation methods perform on audio spectrograms compared to image data, and how does data resolution impact these performances?

## Experimental Design

### 1. Model Training

Both models will be designed with similar architectures to ensure fair comparison:

- **MNIST Model**:
  - Simple CNN with 3-4 convolutional layers
  - Target accuracy: >98%
  - Script: `train_mnist_model.py`

- **AudioMNIST Model**:
  - CNN with similar architecture to MNIST model
  - Input: Mel-spectrograms (64-128 mel bins)
  - Target accuracy: >90%
  - Script: `train_audiomnist_model.py`

### 2. XAI Method Implementation

Three explanation methods will be implemented and applied to both models:

- **Saliency Maps**:
  - Simple gradient-based attribution
  - Script: `generate_saliency.py`

- **Integrated Gradients**:
  - Baseline selection: Zero (black) for images, silent for audio
  - Path steps: 50 (default)
  - Script: `generate_integrated_gradients.py`

- **Guided GradCAM**:
  - Target layer: Last convolutional layer
  - Script: `generate_guided_gradcam.py`

### 3. Resolution Study

Each method will be applied at different resolutions to study impact:

- **MNIST**:
  - Original (28×28)
  - Upsampled to match audio (e.g., 64×64, 128×128)

- **AudioMNIST**:
  - Low resolution (32 mel bins, reduced time frames)
  - Medium resolution (64 mel bins)
  - High resolution (128 mel bins)
  - Script: `resolution_impact_study.py`

### 4. Evaluation Metrics

The following metrics will be computed for all method-dataset-resolution combinations:

- **Faithfulness Correlation**:
  - Implementation: `../../src/metrics/faithfulness.py`
  - Measured by correlation between feature importance and prediction impact
  - Script: `compute_faithfulness.py`

- **Average Sensitivity (Robustness)**:
  - Implementation: `../../src/metrics/sensitivity.py`
  - Measures stability under small input perturbations
  - Script: `compute_sensitivity.py`

- **Sparseness (Complexity)**:
  - Implementation: `../../src/metrics/sparseness.py`
  - Quantifies concentration of explanations
  - Script: `compute_sparseness.py`

- **Randomness Test**:
  - Implementation: `../../src/metrics/randomness.py`
  - Tests explanation dependence on model parameters
  - Script: `compute_randomness.py`

### 5. Comparative Analysis

- **Cross-Modality Analysis**:
  - Direct comparison of metrics across MNIST and AudioMNIST
  - Script: `cross_modality_analysis.py`

- **Resolution Impact Analysis**:
  - How resolution affects explanation quality for each method
  - Script: `resolution_analysis.py`

## Expected Outputs

1. **Trained Models**:
   - Saved in `../../checkpoints/` directory
   - Naming convention: `{dataset}_{architecture}_{resolution}.pt`

2. **Generated Explanations**:
   - Saved in `../../results/visualizations/objective1/` directory
   - Format: PNG images and NumPy arrays
   - Naming convention: `{dataset}_{method}_{resolution}_{sample_id}.{ext}`

3. **Metric Results**:
   - Saved in `../../results/metrics/objective1/` directory
   - Format: CSV files with metrics per method/dataset/resolution
   - Naming convention: `{metric}_{dataset}_{method}_{resolution}.csv`

4. **Analysis Reports**:
   - Saved in `../../results/summaries/objective1/` directory
   - Format: Markdown files with tables and key findings
   - Naming convention: `{analysis_type}_summary.md`

## Experiment Execution

Run these experiments in the following sequence:

```bash
# 1. Train models
python experiments/objective1/train_mnist_model.py
python experiments/objective1/train_audiomnist_model.py

# 2. Generate explanations for all methods
python experiments/objective1/generate_explanations.py --dataset mnist --method saliency
python experiments/objective1/generate_explanations.py --dataset mnist --method integrated_gradients
python experiments/objective1/generate_explanations.py --dataset mnist --method guided_gradcam
python experiments/objective1/generate_explanations.py --dataset audiomnist --method saliency
python experiments/objective1/generate_explanations.py --dataset audiomnist --method integrated_gradients
python experiments/objective1/generate_explanations.py --dataset audiomnist --method guided_gradcam

# 3. Run resolution study
python experiments/objective1/resolution_impact_study.py

# 4. Compute metrics
python experiments/objective1/compute_metrics.py

# 5. Run analysis
python experiments/objective1/cross_modality_analysis.py
python experiments/objective1/resolution_analysis.py
```

For GPU cluster execution, use:

```bash
sbatch slurm/objective1_train.sh
sbatch slurm/objective1_explain.sh
sbatch slurm/objective1_analyze.sh
```

## Expected Findings

This objective is expected to reveal:

1. Which XAI methods perform best for audio spectrograms vs. images
2. How explanation quality varies with resolution in both modalities
3. Whether certain explanation artifacts are modality-specific
4. Insights to guide audio-specific adaptations in Objective 2

## References

- Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks.
- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks.
- Becker, S., et al. (2018). AudioMNIST: Exploring XAI for audio analysis on a simple benchmark.