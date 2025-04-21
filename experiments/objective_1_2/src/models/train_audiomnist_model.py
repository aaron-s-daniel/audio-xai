import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import json
import pickle
from datetime import datetime
from src.models.architectures import AlexNet

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create results directory
results_dir = 'results/audiomnist'
os.makedirs(results_dir, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_preprocessed_data(data_path):
    """Load preprocessed AudioMNIST data."""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Convert numpy arrays to PyTorch tensors
    train_data = {
        'high_res': torch.FloatTensor(data['high_res']['train']),
        'low_res': torch.FloatTensor(data['low_res']['train']),
        'labels': torch.LongTensor(data['labels']['train'])
    }
    
    val_data = {
        'high_res': torch.FloatTensor(data['high_res']['val']),
        'low_res': torch.FloatTensor(data['low_res']['val']),
        'labels': torch.LongTensor(data['labels']['val'])
    }
    
    test_data = {
        'high_res': torch.FloatTensor(data['high_res']['test']),
        'low_res': torch.FloatTensor(data['low_res']['test']),
        'labels': torch.LongTensor(data['labels']['test'])
    }
    
    return train_data, val_data, test_data, data['params']

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / len(train_loader), accuracy

def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / len(val_loader), accuracy

def plot_training_history(history, save_path):
    """Plot training and validation metrics."""
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Load preprocessed data
    data_path = 'data/audio_mnist/preprocessed/audiomnist_preprocessed.pkl'
    train_data, val_data, test_data, params = load_preprocessed_data(data_path)
    
    # Create data loaders
    train_dataset = TensorDataset(train_data['high_res'], train_data['labels'])
    val_dataset = TensorDataset(val_data['high_res'], val_data['labels'])
    test_dataset = TensorDataset(test_data['high_res'], test_data['labels'])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model, criterion, and optimizer
    model = AlexNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training parameters
    num_epochs = 50
    best_val_acc = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(results_dir, 'best_audiomnist_alexnet.pth'))
    
    # Plot training history
    plot_training_history(history, os.path.join(results_dir, 'training_history.png'))
    
    # Evaluate on test set
    model.load_state_dict(torch.load(os.path.join(results_dir, 'best_audiomnist_alexnet.pth')))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    
    # Generate classification report
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    report = classification_report(all_labels, all_preds)
    
    # Save results
    results = {
        'training_history': history,
        'test_metrics': {
            'loss': test_loss,
            'accuracy': test_acc,
            'classification_report': report
        },
        'model_params': {
            'architecture': 'AlexNet',
            'input_channels': 1,
            'num_classes': 10,
            'optimizer': 'Adam',
            'learning_rate': 0.001,
            'batch_size': 32,
            'num_epochs': num_epochs
        },
        'preprocessing_params': params,
        'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    }
    
    with open(os.path.join(results_dir, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Save test data for XAI experiments
    torch.save(test_data['high_res'], os.path.join(results_dir, 'audiomnist_test_spectrograms.pt'))
    torch.save(test_data['labels'], os.path.join(results_dir, 'audiomnist_test_labels.pt'))
    
    print(f"\nTraining completed. Results saved to {results_dir}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(report)

if __name__ == "__main__":
    main() 