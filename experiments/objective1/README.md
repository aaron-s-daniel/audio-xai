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
  - Script: `src/models/train_audiomnist_model.py`

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

# XAI Experiments for AudioMNIST Spectrograms

This document details the implementation and usage of the XAI (eXplainable AI) experiments for analyzing the AudioMNIST model's decision-making process.

## Overview

The `objective1_experiments.py` script implements various XAI methods to generate and evaluate explanations for a trained AudioMNIST model's predictions. The experiments focus on three key XAI methods and four evaluation metrics to assess the quality of the explanations.

## Implementation Details

### XAI Methods

The script implements three popular XAI methods:

1. **Saliency Maps**
   - Basic gradient-based approach
   - Shows which input features most influence the model's prediction
   - Implemented using Captum's `Saliency` class

2. **Integrated Gradients**
   - Path-based attribution method
   - Considers the gradients along a path from a baseline to the input
   - Better at capturing feature importance compared to basic saliency
   - Implemented using Captum's `IntegratedGradients` class

3. **Guided GradCAM**
   - Combines Guided Backpropagation with Class Activation Mapping
   - Particularly effective for CNN architectures
   - Uses the last convolutional layer (`model.features[-1]`) for activation maps
   - Implemented using Captum's `GuidedGradCam` class

### Evaluation Metrics

The script uses four quantitative metrics from the Quantus library:

1. **Faithfulness Correlation**
   - Measures how well the explanation correlates with model predictions
   - Uses baseline replacement perturbation
   - Higher scores indicate better alignment between explanations and model behavior
   ```python
   quantus.FaithfulnessCorrelation(
       nr_runs=10,
       perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
       similarity_func=quantus.similarity_func.correlation_pearson
   )
   ```

2. **Average Sensitivity**
   - Evaluates robustness of explanations to input perturbations
   - Uses uniform noise perturbation
   - Lower scores indicate more stable explanations
   ```python
   quantus.AvgSensitivity(
       nr_samples=10,
       perturb_func=quantus.perturb_func.uniform_noise,
       similarity_func=quantus.similarity_func.difference
   )
   ```

3. **Sparseness**
   - Measures how focused/concentrated the explanations are
   - Higher scores indicate more precise attribution
   ```python
   quantus.Sparseness()
   ```

4. **Random Logit**
   - Tests if explanations are specific to the predicted class
   - Uses SSIM (Structural Similarity Index) for comparison
   - Lower scores indicate better class specificity
   ```python
   quantus.RandomLogit(
       num_classes=10,
       similarity_func=quantus.similarity_func.ssim
   )
   ```

## Visualizations

### Explanation Visualizations

For each XAI method, the script generates visualizations comparing the original spectrogram with its explanation:

```python
def visualize_explanation(self, spectrogram, explanation, method, sample_idx):
    plt.figure(figsize=(10, 5))
    
    # Original spectrogram
    plt.subplot(1, 2, 1)
    plt.imshow(spectrogram.squeeze(), cmap='viridis')
    plt.title("Original Spectrogram")
    plt.axis('off')
    
    # Explanation
    plt.subplot(1, 2, 2)
    plt.imshow(explanation.squeeze(), cmap='seismic', clim=(-1, 1))
    plt.title(f"{method} Explanation")
    plt.axis('off')
```

**How to Interpret:**
- Left plot: Original mel-spectrogram (frequency vs. time)
  - Brighter colors indicate higher energy at that frequency/time
- Right plot: Attribution map
  - Red regions: Positive attribution (features supporting the prediction)
  - Blue regions: Negative attribution (features opposing the prediction)
  - White regions: Neutral attribution
  - Intensity indicates attribution strength

### Method Ranking Visualization

The script generates a heatmap comparing the performance of different XAI methods across metrics:

```python
def generate_summary_visualizations(self, results):
    df = pd.DataFrame.from_dict(results, orient='index')
    df_normalized = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    df_normalized_rank = df_normalized.rank()
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_normalized_rank, annot=True, cmap='YlGnBu', fmt='.0f')
    plt.title('Ranking of XAI Methods across Metrics')
```

**How to Interpret:**
- Rows: XAI methods
- Columns: Evaluation metrics
- Values: Relative ranking (1 = best, 3 = worst)
- Color intensity: Darker colors indicate better performance

## Results Storage

Results are saved in the following structure:
```
results/audiomnist/xai_results/
├── metrics/
│   ├── xai_metrics.csv       # Quantitative metrics for each method
│   ├── xai_results.json      # Detailed results including all metrics
│   └── method_ranking_heatmap.png
└── visualizations/
    ├── saliency_sample_*.png
    ├── integratedgradients_sample_*.png
    └── guidedgradcam_sample_*.png
```

## Usage

To run the experiments:
```bash
python experiments/objective1/src/xai/objective1_experiments.py
```

## Future Improvements

1. **Visualization Standardization**
   - Current spectrogram visualizations in XAI results differ from those in preprocessing
   - Need to align visualization parameters (colormap, scale, orientation) with `audio_preprocessing.py`
   - Consider adding frequency and time axes labels for better interpretability

2. **Additional Metrics**
   - Consider adding localization metrics
   - Implement temporal coherence measures specific to audio data

3. **Batch Processing**
   - Currently processes one batch of test data
   - Could be extended to analyze more samples for robust results

## Dependencies

- PyTorch
- Captum
- Quantus
- NumPy
- Matplotlib
- Seaborn
- Pandas