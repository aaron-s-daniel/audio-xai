import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle

def load_audiomnist(data_path, sr=16000, duration=1.0):
    """Load AudioMNIST dataset with specified sampling rate and duration."""
    audio_path = os.path.join(data_path, 'AudioMNIST-master', 'data')
    data = []
    labels = []
    target_length = int(sr * duration)
    
    # Iterate through folders 00 to 09 (for digits 0-9)
    for folder in range(10):
        folder_name = f"{folder:02d}"  # This will give '00', '01', ..., '09'
        folder_path = os.path.join(audio_path, folder_name)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist.")
            continue
            
        for filename in os.listdir(folder_path):
            if filename.endswith('.wav'):
                file_path = os.path.join(folder_path, filename)
                audio, _ = librosa.load(file_path, sr=sr, duration=duration)
                
                # Pad or truncate the audio to the target length
                if len(audio) > target_length:
                    audio = audio[:target_length]
                else:
                    audio = np.pad(audio, (0, max(0, target_length - len(audio))))           
                data.append(audio)
                labels.append(folder)  # Use folder number as label (0-9)
    
    return np.array(data), np.array(labels)

def create_spectrograms(audio_data, sr=16000, n_mels_high=128, n_mels_low=32, 
                        time_steps_high=128, time_steps_low=32):
    """Create high and low resolution spectrograms from audio data."""
    
    high_res_specs = []
    low_res_specs = []
    
    for audio in audio_data:
        # High resolution spectrogram
        mel_spec_high = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr, 
            n_mels=n_mels_high,
            hop_length=int(len(audio)/time_steps_high)
        )
        mel_spec_db_high = librosa.power_to_db(mel_spec_high, ref=np.max)
        
        # Ensure consistent shape
        if mel_spec_db_high.shape[1] < time_steps_high:
            pad_width = time_steps_high - mel_spec_db_high.shape[1]
            mel_spec_db_high = np.pad(mel_spec_db_high, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec_db_high = mel_spec_db_high[:, :time_steps_high]
        
        # Low resolution spectrogram
        mel_spec_low = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr, 
            n_mels=n_mels_low,
            hop_length=int(len(audio)/time_steps_low)
        )
        mel_spec_db_low = librosa.power_to_db(mel_spec_low, ref=np.max)
        
        # Ensure consistent shape
        if mel_spec_db_low.shape[1] < time_steps_low:
            pad_width = time_steps_low - mel_spec_db_low.shape[1]
            mel_spec_db_low = np.pad(mel_spec_db_low, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec_db_low = mel_spec_db_low[:, :time_steps_low]
        
        high_res_specs.append(mel_spec_db_high)
        low_res_specs.append(mel_spec_db_low)
    
    # Convert to arrays and reshape for CNN input (batch, channels, height, width)
    high_res_specs = np.array(high_res_specs).reshape((-1, 1, n_mels_high, time_steps_high))
    low_res_specs = np.array(low_res_specs).reshape((-1, 1, n_mels_low, time_steps_low))
    
    return high_res_specs, low_res_specs

def visualize_spectrograms(waveform, high_res_spec, low_res_spec, sample_idx=0, sr=16000, duration=1.0):
    """Visualize waveform and spectrograms for a sample with proper axis labels."""
    plt.figure(figsize=(15, 10))
    
    # Plot waveform
    plt.subplot(3, 1, 1)
    time_axis = np.linspace(0, duration, len(waveform))
    plt.plot(time_axis, waveform)
    plt.title('Waveform')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.xlim(0, duration)
    
    # Plot high-res spectrogram
    plt.subplot(3, 1, 2)
    plt.imshow(high_res_spec[sample_idx, 0], aspect='auto', origin='lower', cmap='viridis')
    plt.title(f'High Resolution Spectrogram ({high_res_spec.shape[2]}x{high_res_spec.shape[3]})')
    
    # Create proper time ticks for high-res
    time_ticks = np.linspace(0, high_res_spec.shape[3]-1, 5)
    time_labels = [f"{t*duration/high_res_spec.shape[3]:.1f}" for t in time_ticks]
    plt.xticks(time_ticks, time_labels)
    plt.xlabel('Time (seconds)')
    
    # Create proper frequency ticks for high-res
    freq_ticks = np.linspace(0, high_res_spec.shape[2]-1, 5)
    max_freq = sr/2  # Nyquist frequency
    freq_labels = [f"{f*max_freq/high_res_spec.shape[2]/1000:.1f}" for f in freq_ticks]
    plt.yticks(freq_ticks, freq_labels)
    plt.ylabel('Frequency (kHz)')
    
    plt.colorbar(format='%+2.0f dB')
    
    # Plot low-res spectrogram
    plt.subplot(3, 1, 3)
    plt.imshow(low_res_spec[sample_idx, 0], aspect='auto', origin='lower', cmap='viridis')
    plt.title(f'Low Resolution Spectrogram ({low_res_spec.shape[2]}x{low_res_spec.shape[3]})')
    
    # Create proper time ticks for low-res
    time_ticks = np.linspace(0, low_res_spec.shape[3]-1, 5)
    time_labels = [f"{t*duration/low_res_spec.shape[3]:.1f}" for t in time_ticks]
    plt.xticks(time_ticks, time_labels)
    plt.xlabel('Time (seconds)')
    
    # Create proper frequency ticks for low-res
    freq_ticks = np.linspace(0, low_res_spec.shape[2]-1, 5)
    freq_labels = [f"{f*max_freq/low_res_spec.shape[2]/1000:.1f}" for f in freq_ticks]
    plt.yticks(freq_ticks, freq_labels)
    plt.ylabel('Frequency (kHz)')
    
    plt.colorbar(format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig(f'spectrogram_comparison_{sample_idx}.png', dpi=300)
    plt.close()

    
def preprocess_audiomnist(data_path, output_dir, sr=16000, duration=1.0, 
                         n_mels_high=128, n_mels_low=32, 
                         time_steps_high=128, time_steps_low=32,
                         test_size=0.2, valid_size=0.2):
    """Preprocess AudioMNIST data and save high/low resolution versions."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load raw audio data
    print("Loading AudioMNIST dataset...")
    X_audio, y = load_audiomnist(data_path, sr=sr, duration=duration)
    
    # Create high and low resolution spectrograms
    print("Creating spectrograms...")
    X_high_res, X_low_res = create_spectrograms(
        X_audio, sr=sr, 
        n_mels_high=n_mels_high, n_mels_low=n_mels_low,
        time_steps_high=time_steps_high, time_steps_low=time_steps_low
    )
    
    # Split data into train, validation, and test sets
    # First split off test set
    X_audio_train, X_audio_test, X_high_train, X_high_test, X_low_train, X_low_test, y_train, y_test = train_test_split(
        X_audio, X_high_res, X_low_res, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Then split train into train and validation
    valid_ratio = valid_size / (1 - test_size)
    X_audio_train, X_audio_val, X_high_train, X_high_val, X_low_train, X_low_val, y_train, y_val = train_test_split(
        X_audio_train, X_high_train, X_low_train, y_train, 
        test_size=valid_ratio, random_state=42, stratify=y_train
    )
    
    # Save preprocessed data
    data = {
        'audio': {
            'train': X_audio_train,
            'val': X_audio_val,
            'test': X_audio_test
        },
        'high_res': {
            'train': X_high_train,
            'val': X_high_val,
            'test': X_high_test
        },
        'low_res': {
            'train': X_low_train,
            'val': X_low_val,
            'test': X_low_test
        },
        'labels': {
            'train': y_train,
            'val': y_val,
            'test': y_test
        },
        'params': {
            'sr': sr,
            'duration': duration,
            'n_mels_high': n_mels_high,
            'n_mels_low': n_mels_low,
            'time_steps_high': time_steps_high,
            'time_steps_low': time_steps_low
        }
    }
    
    # Save to pickle file
    with open(os.path.join(output_dir, 'audiomnist_preprocessed.pkl'), 'wb') as f:
        pickle.dump(data, f)
    
    # Visualize a few examples
    for i in range(min(5, len(X_audio_test))):
        visualize_spectrograms(X_audio_test[i], X_high_test, X_low_test, sample_idx=i)
    
    print(f"Preprocessing complete. Data saved to {output_dir}")
    print(f"High resolution spectrograms shape: {X_high_train.shape}")
    print(f"Low resolution spectrograms shape: {X_low_train.shape}")
    
    return data

if __name__ == "__main__":
    data_path = '/home/mqa887/tmp/audiomnist'
    output_dir = 'data/audio_mnist/preprocessed'
    
    data = preprocess_audiomnist(
        data_path=data_path,
        output_dir=output_dir,
        n_mels_high=128,
        n_mels_low=32,
        time_steps_high=128,
        time_steps_low=32
    )
    
    print("Dataset preprocessed and saved successfully!")