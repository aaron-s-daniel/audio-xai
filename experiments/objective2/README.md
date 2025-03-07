# Objective 2: Audio-Specific Adaptations for XAI Methods

This directory contains experiments related to the second research objective: developing and evaluating improvements to existing XAI methods specifically for audio data classification.

## Research Question

**RQ2**: How can existing XAI approaches be improved for better performance on audio data?

## Motivation

Findings from Objective 1 are expected to reveal limitations in how traditional XAI methods handle audio spectrograms. This objective aims to address these limitations by developing audio-specific adaptations that leverage the unique characteristics of audio data.

## Experimental Design

### 1. Adaptation Design & Implementation

Based on insights from Objective 1, we will develop several adaptations to Integrated Gradients (our primary focus) and potentially other methods:

#### 1.1 Audio-Specific Baseline Selection

- **Traditional Approach**: Zero (black) baseline for images
- **Proposed Adaptations**:
  - Noise-based baselines (white noise, pink noise)
  - Average audio baseline from training data
  - Frequency-specific baselines
  - Script: `audio_baselines.py`

#### 1.2 Frequency-Band Aggregation

- **Traditional Approach**: Pixel-wise attribution
- **Proposed Adaptations**:
  - Mel-band aggregation following psychoacoustic principles
  - Critical band aggregation 
  - Harmonic grouping for improved interpretability
  - Script: `frequency_aggregation.py`

#### 1.3 Temporal Dynamics Incorporation

- **Traditional Approach**: Static attribution maps
- **Proposed Adaptations**:
  - Temporal smoothing across frames
  - Dynamic weighting that respects audio evolution
  - Attack/decay-aware attribution
  - Script: `temporal_dynamics.py`

#### 1.4 Audible Explanations

- **Traditional Approach**: Visual heatmaps only
- **Proposed Adaptations**:
  - Feature-based audio filtering
  - Selective audio reconstruction
  - Importance-weighted audio synthesis
  - Script: `audible_explanations.py`

### 2. Implementation & Testing

Each adaptation will be systematically implemented and tested:

- **Implementation**:
  - Extend base XAI methods in `../../src/xai_methods/`
  - Create adaptation-specific parameters
  - Ensure compatibility with existing metrics
  - Script: `implement_adaptations.py`

- **Initial Testing**:
  - Verification on simple test cases
  - Qualitative assessment of explanation quality
  - Script: `test_adaptations.py`

### 3. Comprehensive Evaluation

The adaptations will be rigorously evaluated using:

- **Standard Metrics** (same as Objective 1):
  - Faithfulness Correlation
  - Average Sensitivity (Robustness)
  - Sparseness (Complexity)
  - Randomness Test
  - Script: `evaluate_adaptations.py`

- **Audio-Specific Evaluation**:
  - Perceptual alignment with audio features
  - Harmonic consistency
  - Temporal coherence
  - Script: `audio_specific_evaluation.py`

- **Comparative Analysis**:
  - Direct comparison with original methods
  - Analysis of improvement patterns
  - Script: `comparative_analysis.py`

### 4. Case Studies

Select specific audio examples for in-depth analysis:

- **Correctly Classified Examples**:
  - Analyze how adaptations improve explanation quality
  - Script: `case_study_correct.py`

- **Misclassified Examples**:
  - Evaluate if adaptations provide better insights into errors
  - Script: `case_study_errors.py`

- **Edge Cases**:
  - Analyze examples near decision boundaries
  - Script: `case_study_edge.py`

## Expected Outputs

1. **Adaptation Implementations**:
   - Source code in `../../src/xai_methods/audio_adaptations/`
   - Documentation of design choices and parameters

2. **Generated Explanations**:
   - Saved in `../../results/visualizations/objective2/`
   - Visual formats: PNG images
   - Audio formats: WAV files for audible explanations
   - Naming convention: `{adaptation}_{method}_{sample_id}.{ext}`

3. **Evaluation Results**:
   - Saved in `../../results/metrics/objective2/`
   - Format: CSV files with metrics per adaptation
   - Naming convention: `{metric}_{adaptation}.csv`

4. **Analysis Reports**:
   - Saved in `../../results/summaries/objective2/`
   - Format: Markdown files with tables, figures, and key findings
   - Naming convention: `{analysis_type}_summary.md`

5. **Audio Examples**:
   - Saved in `../../results/audio_explanations/`
   - Format: WAV files of audible explanations
   - Naming convention: `{adaptation}_{sample_id}.wav`

## Experiment Execution

Run these experiments in the following sequence:

```bash
# 1. Implement adaptations
python experiments/objective2/implement_adaptations.py

# 2. Test adaptations on sample data
python experiments/objective2/test_adaptations.py

# 3. Generate explanations with adapted methods
python experiments/objective2/generate_adapted_explanations.py --adaptation baseline
python experiments/objective2/generate_adapted_explanations.py --adaptation frequency
python experiments/objective2/generate_adapted_explanations.py --adaptation temporal
python experiments/objective2/generate_adapted_explanations.py --adaptation audible

# 4. Evaluate adaptations
python experiments/objective2/evaluate_adaptations.py
python experiments/objective2/audio_specific_evaluation.py

# 5. Run comparative analysis
python experiments/objective2/comparative_analysis.py

# 6. Generate case studies
python experiments/objective2/case_studies.py
```

For GPU cluster execution, use:

```bash
sbatch slurm/objective2_implement.sh
sbatch slurm/objective2_evaluate.sh
sbatch slurm/objective2_analyze.sh
```

## Expected Findings

This objective is expected to yield:

1. Concrete improvements to XAI methods for audio data
2. Quantitative measurements of these improvements
3. Insights into which adaptation strategies are most effective
4. Novel approaches for creating audible explanations
5. Guidelines for applying XAI to audio classification systems

## Limitations & Future Work

Document potential limitations that could be addressed in future research:

- Transferability to other audio tasks beyond digit classification
- Scalability to larger audio datasets
- Extension to other XAI methods
- Integration with human evaluation

## References

- Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks.
- Schuller, B. W., et al. (2022). Towards sonification in multimodal and user-friendly explainable AI.
- Parekh, J., et al. (2022). Listen to interpret: Post-hoc interpretability for audio networks with NMF.