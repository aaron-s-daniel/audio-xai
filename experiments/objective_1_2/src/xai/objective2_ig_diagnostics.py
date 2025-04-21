# ig_audio_diagnostics.py

import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import entropy
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def compute_energy(spectrogram):
    return torch.sum(spectrogram ** 2).item()


def compute_entropy(spectrogram):
    flat = spectrogram.flatten().cpu().numpy()
    flat = flat - flat.min() + 1e-6  # shift to positive domain
    flat /= flat.sum()  # normalize to probability
    return entropy(flat)


def compute_dominant_freq_band(spectrogram):
    band_energies = spectrogram.sum(dim=-1)
    dominant_band = torch.argmax(band_energies).item()
    return dominant_band / spectrogram.shape[-2]  # normalize by number of bands


def load_faithfulness_scores(json_path):
    import json
    with open(json_path, 'r') as f:
        results = json.load(f)
    # Assume IG is the only method in this experiment
    return results['IntegratedGradients']['Faithfulness_per_sample']


def compute_input_features(spectrograms):
    features = []
    for s in tqdm(spectrograms, desc="Computing spectrogram features"):
        energy = compute_energy(s)
        ent = compute_entropy(s)
        dom_freq = compute_dominant_freq_band(s)
        features.append({
            "energy": energy,
            "entropy": ent,
            "dominant_freq": dom_freq
        })
    return pd.DataFrame(features)


def correlate_features_with_faithfulness(features_df, faithfulness):
    features_df["faithfulness"] = faithfulness[:len(features_df)]
    corr = features_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation Between Input Features and IG Faithfulness")
    plt.tight_layout()
    plt.savefig("diagnostics_feature_correlation.png")
    plt.close()
    return features_df, corr


if __name__ == "__main__":
    spectrograms = torch.load("results/audiomnist/audiomnist_test_spectrograms.pt")
    spectrograms = spectrograms[:1000]  # subset for speed

    # Load per-sample faithfulness (assume you've already saved it)
    # If not available yet, we can mock this temporarily
    faithfulness_scores = load_faithfulness_scores("experiments/objective_1_2/obj2/zero_per_sample/metrics/xai_results_zero_per_sample.json")
    # faithfulness_scores = [-0.45] * len(spectrograms)  # TEMP stub for development

    features_df = compute_input_features(spectrograms)
    features_df, corr_matrix = correlate_features_with_faithfulness(features_df, faithfulness_scores)

    print("Top correlations:")
    print(corr_matrix["faithfulness"].sort_values(ascending=False))
