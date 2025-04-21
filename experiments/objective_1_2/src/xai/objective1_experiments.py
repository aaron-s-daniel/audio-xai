import os
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients, Saliency, GuidedGradCam
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
        
        # Create results directory
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'metrics'), exist_ok=True)
        
        # Load model and data
        self.model = self.load_model(model_path)
        self.test_data, self.test_labels = self.load_test_data(test_data_path, test_labels_path)
        self.test_loader = self.create_data_loader()
        
        # Initialize XAI methods
        self.xai_methods = {
            "Saliency": Saliency(self.model),
            "IntegratedGradients": IntegratedGradients(self.model),
            "GuidedGradCam": GuidedGradCam(self.model, self.model.features[-1])
        }
        
        # Initialize metrics
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
            "Complexity": quantus.Sparseness(
                disable_warnings=True,
            ),
            "Randomisation": quantus.RandomLogit(
                num_classes=10,
                similarity_func=quantus.similarity_func.ssim,
                disable_warnings=True,
            ),
        }

    def load_model(self, model_path):
        """Load the trained model."""
        model = AlexNet().to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def load_test_data(self, test_data_path, test_labels_path):
        """Load test data and labels."""
        test_data = torch.load(test_data_path)
        test_labels = torch.load(test_labels_path)
        return test_data, test_labels

    def create_data_loader(self):
        """Create DataLoader for test data."""
        test_dataset = TensorDataset(self.test_data, self.test_labels)
        return DataLoader(test_dataset, batch_size=32, shuffle=False)

    def generate_explanations(self, method, inputs, targets, model=None, device=None):
        """Generate explanations using specified method."""
        # Use provided model or self.model
        model = model if model is not None else self.model
        # Use provided device or self.device
        device = device if device is not None else self.device
        
        if isinstance(inputs, np.ndarray):
            inputs = torch.FloatTensor(inputs).to(device)
        if isinstance(targets, np.ndarray):
            targets = torch.LongTensor(targets).to(device)
        
        return self.xai_methods[method].attribute(inputs, target=targets).cpu().detach().numpy()

    def evaluate_explanations(self, method, explanations, x_batch, y_batch):
        """Evaluate explanations using all metrics."""
        results = {}
        for metric_name, metric_func in self.metrics.items():
            print(f"Evaluating {metric_name} for {method}...")
            try:
                scores = metric_func(
                    model=self.model,
                    x_batch=x_batch.cpu().numpy(),
                    y_batch=y_batch.cpu().numpy(),
                    a_batch=explanations,
                    device=self.device.type,  # Pass 'cuda' or 'cpu' as string
                    explain_func=self.generate_explanations,
                    explain_func_kwargs={"method": method}
                )
                results[metric_name] = float(np.mean(scores))  # Convert to float for JSON serialization
            except Exception as e:
                print(f"Error evaluating {metric_name} for {method}: {str(e)}")
                results[metric_name] = None
        return results

    def visualize_explanation(self, spectrogram, explanation, method, sample_idx, sr=16000, duration=1.0):
        """Visualize original spectrogram and explanation with proper axes and scaling."""
        # Print attribution statistics
        attr_min = np.min(explanation)
        attr_max = np.max(explanation)
        attr_mean = np.mean(explanation)
        attr_std = np.std(explanation)
        print(f"\nAttribution statistics for {method}:")
        print(f"Min: {attr_min:.6f}, Max: {attr_max:.6f}")
        print(f"Mean: {attr_mean:.6f}, Std: {attr_std:.6f}")
        
        plt.figure(figsize=(15, 6))
        
        # Plot original spectrogram
        plt.subplot(1, 2, 1)
        im1 = plt.imshow(spectrogram.squeeze(), aspect='auto', origin='lower', cmap='viridis')
        plt.title("Original Spectrogram")
        
        # Create proper time ticks
        time_steps = spectrogram.shape[-1]
        time_ticks = np.linspace(0, time_steps-1, 5)
        time_labels = [f"{t*duration/time_steps:.1f}" for t in time_ticks]
        plt.xticks(time_ticks, time_labels)
        plt.xlabel('Time (seconds)')
        
        # Create proper frequency ticks
        n_mels = spectrogram.shape[-2]
        freq_ticks = np.linspace(0, n_mels-1, 5)
        max_freq = sr/2  # Nyquist frequency
        freq_labels = [f"{f*max_freq/n_mels/1000:.1f}" for f in freq_ticks]
        plt.yticks(freq_ticks, freq_labels)
        plt.ylabel('Frequency (kHz)')
        
        # Add colorbar
        plt.colorbar(im1, format='%+2.0f dB')
        
        # Plot explanation with adjusted visualization
        plt.subplot(1, 2, 2)
        
        # Normalize attribution if values are very small
        if attr_max - attr_min < 1e-6:
            print(f"Warning: Very small attribution range detected for {method}")
            if np.abs(attr_mean) < 1e-6:
                print("Warning: Near-zero attributions detected")
            
            # Apply stronger normalization for small values
            explanation_norm = (explanation - attr_min) / (attr_max - attr_min + 1e-10)
            im2 = plt.imshow(explanation_norm.squeeze(), aspect='auto', origin='lower',
                           cmap='seismic', clim=(0, 1))
            plt.title(f"{method} Attribution\n(Normalized)")
        else:
            # Use symmetric scaling around zero for normal ranges
            abs_max = max(abs(attr_min), abs(attr_max))
            im2 = plt.imshow(explanation.squeeze(), aspect='auto', origin='lower',
                           cmap='seismic', clim=(-abs_max, abs_max))
            plt.title(f"{method} Attribution")
        
        # Match axes with original spectrogram
        plt.xticks(time_ticks, time_labels)
        plt.xlabel('Time (seconds)')
        plt.yticks(freq_ticks, freq_labels)
        plt.ylabel('Frequency (kHz)')
        
        # Add colorbar for attribution
        if attr_max - attr_min < 1e-6:
            plt.colorbar(im2, format='%.2e')  # Scientific notation for small values
        else:
            plt.colorbar(im2, format='%.2f')
        
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'visualizations', 
                               f'{method.lower()}_sample_{sample_idx}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def run_experiments(self):
        """Run all XAI experiments and evaluations."""
        results = {}
        
        try:
            # Get a batch of test data
            x_batch, y_batch = next(iter(self.test_loader))
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            
            # Load parameters from model
            sr = 16000  # Default sampling rate
            duration = 1.0  # Default duration
            
            # Generate and evaluate explanations for each method
            for method in tqdm(self.xai_methods.keys(), desc="Processing XAI methods"):
                print(f"\nGenerating explanations for {method}...")
                try:
                    explanations = self.generate_explanations(method, x_batch, y_batch)
                    
                    # Evaluate explanations
                    method_results = self.evaluate_explanations(method, explanations, x_batch, y_batch)
                    results[method] = method_results
                    
                    # Visualize explanations for first few samples
                    for i in range(min(5, len(x_batch))):
                        self.visualize_explanation(
                            x_batch[i].cpu().numpy(),
                            explanations[i],
                            method,
                            i,
                            sr=sr,
                            duration=duration
                        )
                except Exception as e:
                    print(f"Error processing method {method}: {str(e)}")
                    results[method] = {"error": str(e)}
                    continue
            
            # Save intermediate results after each method
            self.save_results(results)
            
            # Generate and save visualizations
            self.generate_summary_visualizations(results)
            
        except Exception as e:
            print(f"Error in run_experiments: {str(e)}")
            # Save whatever results we have
            if results:
                self.save_results(results)
        
        return results

    def save_results(self, results):
        """Save experiment results."""
        # Save metrics
        metrics_df = pd.DataFrame.from_dict(results, orient='index')
        metrics_df.to_csv(os.path.join(self.results_dir, 'metrics', 'xai_metrics.csv'))
        
        # Save detailed results
        with open(os.path.join(self.results_dir, 'metrics', 'xai_results.json'), 'w') as f:
            json.dump(results, f, indent=4)

    def generate_summary_visualizations(self, results):
        """Generate summary visualizations of results."""
        # Create normalized DataFrame for heatmap
        df = pd.DataFrame.from_dict(results, orient='index')
        df_normalized = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        df_normalized_rank = df_normalized.rank()
        
        # Plot heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(df_normalized_rank, annot=True, cmap='YlGnBu', fmt='.0f')
        plt.title('Ranking of XAI Methods across Metrics')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'metrics', 'method_ranking_heatmap.png'))
        plt.close()

def main():
    # Define paths
    model_path = 'results/audiomnist/best_audiomnist_alexnet.pth'
    test_data_path = 'results/audiomnist/audiomnist_test_spectrograms.pt'
    test_labels_path = 'results/audiomnist/audiomnist_test_labels.pt'
    results_dir = 'results/audiomnist/xai_results'
    
    # Run experiments
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
            print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main() 