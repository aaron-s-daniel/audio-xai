# objective2_mnist_ig_baselines.py

import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from torch.utils.data import DataLoader, TensorDataset
from captum.attr import IntegratedGradients
from src.models.architectures import AlexNet
import quantus


def get_baseline(inputs, mode="zero"):
    if mode == "zero":
        return torch.zeros_like(inputs)
    elif mode == "median":
        return torch.median(inputs, dim=0).values.unsqueeze(0).expand_as(inputs)
    elif mode == "low_energy_mask":
        threshold = torch.quantile(inputs, 0.2)
        masked = inputs.clone()
        masked[masked > threshold] = 0
        return masked
    elif mode == "noisy_silence":
        return torch.randn_like(inputs) * 0.01
    else:
        raise ValueError(f"Unknown baseline mode: {mode}")


def evaluate_ig(model, inputs, labels, baseline_mode):
    ig = IntegratedGradients(model)
    baseline = get_baseline(inputs, mode=baseline_mode)
    attributions = ig.attribute(inputs, baselines=baseline, target=labels)

    faithfulness = quantus.FaithfulnessCorrelation(
        nr_runs=10,
        perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
        similarity_func=quantus.similarity_func.correlation_pearson,
        disable_warnings=True,
    )
    complexity = quantus.Sparseness(disable_warnings=True)

    faith = faithfulness(
        model=model,
        x_batch=inputs.cpu().numpy(),
        y_batch=labels.cpu().numpy(),
        a_batch=attributions.cpu().detach().numpy(),
        device="cuda" if torch.cuda.is_available() else "cpu",
        explain_func=None,
    )
    comp = complexity(
        model=model,
        x_batch=inputs.cpu().numpy(),
        y_batch=labels.cpu().numpy(),
        a_batch=attributions.cpu().detach().numpy(),
        device="cuda" if torch.cuda.is_available() else "cpu",
        explain_func=None,
    )

    return float(np.mean(faith)), float(np.mean(comp))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNet().to(device)
    model.load_state_dict(torch.load("results/mnist/best_mnist_alexnet.pth", map_location=device))
    model.eval()

    inputs = torch.load("results/mnist/mnist_test_images.pt").to(device)
    labels = torch.load("results/mnist/mnist_test_labels.pt").to(device)

    inputs = inputs[:1000]  # Limit for speed
    labels = labels[:1000]

    results = {}
    for mode in ["zero", "median", "low_energy_mask", "noisy_silence"]:
        print(f"Running IG for baseline: {mode}")
        faith, comp = evaluate_ig(model, inputs, labels, mode)
        results[mode] = {"faithfulness": faith, "complexity": comp}

    with open("results/mnist/ig_baseline_results.json", "w") as f:
        json.dump(results, f, indent=4)

    df = pd.DataFrame.from_dict(results, orient="index")
    df.to_csv("results/mnist/ig_baseline_summary.csv")
    print("\nResults:")
    print(df)


if __name__ == "__main__":
    main()
