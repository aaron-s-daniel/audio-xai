import os
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from captum.attr import (IntegratedGradients, Saliency, GuidedGradCam,
                          DeepLift, InputXGradient, Occlusion)
import quantus
import seaborn as sns
import pandas as pd
from datetime import datetime
import json
from tqdm import tqdm
from src.models.architectures import AlexNet

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class XAIExperiment:
    def __init__(self, model_path, test_data_path, test_labels_path, results_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'metrics'), exist_ok=True)

        self.model = self.load_model(model_path)
        self.test_data, self.test_labels = self.load_test_data(test_data_path, test_labels_path)
        self.test_loader = self.create_data_loader()

        self.xai_methods = {
            "Saliency": Saliency(self.model),
            "IntegratedGradients": IntegratedGradients(self.model),
            "GuidedGradCam": GuidedGradCam(self.model, self.model.features[-1]),
            "DeepLift": DeepLift(self.model),
            "InputXGradient": InputXGradient(self.model),
            "Occlusion": Occlusion(self.model)
        }

        self.metrics = {
            "Faithfulness": quantus.FaithfulnessCorrelation(
                nr_runs=10,
                perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
                similarity_func=quantus.similarity_func.correlation_pearson,
                disable_warnings=True,
            ),
            "Robustness": quantus.AvgSensitivity(
                nr_samples=10,
                perturb_func=quantus.perturb_func.uniform_noise,
                similarity_func=quantus.similarity_func.difference,
                disable_warnings=True,
            ),
            "Complexity": quantus.Sparseness(disable_warnings=True),
            "Randomisation": quantus.RandomLogit(
                num_classes=10,
                similarity_func=quantus.similarity_func.ssim,
                disable_warnings=True,
            ),
        }

    def load_model(self, model_path):
        model = AlexNet().to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def load_test_data(self, test_data_path, test_labels_path):
        test_data = torch.load(test_data_path)
        test_labels = torch.load(test_labels_path)
        return test_data, test_labels

    def create_data_loader(self):
        test_dataset = TensorDataset(self.test_data, self.test_labels)
        return DataLoader(test_dataset, batch_size=32, shuffle=False)

    def generate_explanations(self, method, inputs, targets):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        if method == "Occlusion":
            return self.xai_methods[method].attribute(
                inputs,
                target=targets,
                sliding_window_shapes=(1, 5, 5),
                strides=(1, 2, 2),
                baselines=torch.zeros_like(inputs)
            ).cpu().detach().numpy()
        else:
            return self.xai_methods[method].attribute(inputs, target=targets).cpu().detach().numpy()

    def evaluate_explanations(self, method, explanations, x_batch, y_batch):
        results = {}
        for metric_name, metric_func in self.metrics.items():
            print(f"Evaluating {metric_name} for {method}...")
            try:
                scores = metric_func(
                    model=self.model,
                    x_batch=x_batch.cpu().numpy(),
                    y_batch=y_batch.cpu().numpy(),
                    a_batch=explanations,
                    device=self.device.type,
                    explain_func=self.generate_explanations,
                    explain_func_kwargs={"method": method}
                )
                results[metric_name] = float(np.mean(scores))
            except Exception as e:
                print(f"Error evaluating {metric_name} for {method}: {str(e)}")
                results[metric_name] = None
        return results

    def visualize_explanation(self, image, explanation, method, sample_idx):
        attr_min, attr_max = np.min(explanation), np.max(explanation)
        abs_max = max(abs(attr_min), abs(attr_max))

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(image.squeeze(), aspect='auto', origin='lower', cmap='viridis')
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(explanation.squeeze(), cmap='seismic', clim=(-abs_max, abs_max), origin='lower')
        plt.title(f"{method} Attribution")
        plt.axis('off')

        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'visualizations', f'{method.lower()}_sample_{sample_idx}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def run_experiments(self):
        results = {}
        try:
            x_batch, y_batch = next(iter(self.test_loader))
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

            for method in tqdm(self.xai_methods.keys(), desc="Processing XAI methods"):
                print(f"\nGenerating explanations for {method}...")
                try:
                    explanations = self.generate_explanations(method, x_batch, y_batch)
                    method_results = self.evaluate_explanations(method, explanations, x_batch, y_batch)
                    results[method] = method_results

                    for i in range(min(5, len(x_batch))):
                        self.visualize_explanation(
                            x_batch[i].cpu().numpy(),
                            explanations[i],
                            method,
                            i
                        )
                except Exception as e:
                    print(f"Error processing method {method}: {str(e)}")
                    results[method] = {"error": str(e)}

            self.save_results(results)
            self.generate_summary_visualizations(results)

        except Exception as e:
            print(f"Error in run_experiments: {str(e)}")
            if results:
                self.save_results(results)

        return results

    def save_results(self, results):
        metrics_df = pd.DataFrame.from_dict(results, orient='index')
        metrics_df.to_csv(os.path.join(self.results_dir, 'metrics', 'xai_metrics.csv'))

        with open(os.path.join(self.results_dir, 'metrics', 'xai_results.json'), 'w') as f:
            json.dump(results, f, indent=4)

    def generate_summary_visualizations(self, results):
        df = pd.DataFrame.from_dict(results, orient='index')
        df_normalized = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        df_normalized_rank = df_normalized.rank()

        plt.figure(figsize=(10, 6))
        sns.heatmap(df_normalized_rank, annot=True, cmap='YlGnBu', fmt='.0f')
        plt.title('Ranking of XAI Methods across Metrics')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'metrics', 'method_ranking_heatmap.png'))
        plt.close()


def main():
    model_path = 'results/audiomnist/best_audiomnist_alexnet.pth'
    test_data_path = 'results/audiomnist/audiomnist_test_spectrograms.pt'
    test_labels_path = 'results/audiomnist/audiomnist_test_labels.pt'
    results_dir = 'results/audiomnist/xai_results2'

    experiment = XAIExperiment(
        model_path=model_path,
        test_data_path=test_data_path,
        test_labels_path=test_labels_path,
        results_dir=results_dir
    )

    results = experiment.run_experiments()

    print("\nExperiment completed successfully!")
    print(f"Results saved to: {results_dir}")
    print("\nSummary of results:")
    for method, metrics in results.items():
        print(f"\n{method}:")
        for metric, value in metrics.items():
            try:
                print(f"  {metric}: {float(value):.4f}")
            except (ValueError, TypeError):
                print(f"  {metric}: {value}")


if __name__ == "__main__":
    main()
