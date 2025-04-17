# Audio Preprocessing Documentation

This document explains the audio preprocessing pipeline used for the AudioMNIST dataset. The preprocessing involves loading audio files, creating spectrograms at different resolutions, and preparing the data for model training.

## Overview

The preprocessing pipeline consists of several key functions that work together to:
1. Load raw audio data from the AudioMNIST dataset
2. Create high and low resolution spectrograms
3. Visualize the spectrograms for verification
4. Split the data into train, validation, and test sets
5. Save the preprocessed data for model training

## Functions

### `load_audiomnist(data_path, sr=16000, duration=1.0)`

This function loads the AudioMNIST dataset and performs initial preprocessing.

**Parameters:**
- `data_path`: Path to the AudioMNIST dataset
- `sr`: Sampling rate (default: 16000 Hz)
- `duration`: Target duration for each audio clip (default: 1.0 seconds)

**Process:**
1. Iterates through folders 00-09 (representing digits 0-9)
2. Loads each WAV file using librosa
3. Normalizes audio length by padding or truncating to target duration
4. Returns numpy arrays of audio data and corresponding labels

### `create_spectrograms(audio_data, sr=16000, n_mels_high=128, n_mels_low=32, time_steps_high=128, time_steps_low=32)`

Creates high and low resolution mel spectrograms from the audio data.

**Parameters:**
- `audio_data`: Array of audio waveforms
- `sr`: Sampling rate
- `n_mels_high`: Number of mel bands for high resolution (default: 128)
- `n_mels_low`: Number of mel bands for low resolution (default: 32)
- `time_steps_high`: Number of time steps for high resolution (default: 128)
- `time_steps_low`: Number of time steps for low resolution (default: 32)

**Process:**
1. Creates high-resolution mel spectrogram
2. Creates low-resolution mel spectrogram
3. Converts power spectrograms to decibel scale
4. Ensures consistent shapes through padding
5. Reshapes spectrograms for CNN input format (batch, channels, height, width)

### `visualize_spectrograms(waveform, high_res_spec, low_res_spec, sample_idx=0, sr=16000, duration=1.0)`

Creates visualization of the audio waveform and spectrograms for verification.

**Parameters:**
- `waveform`: Audio waveform
- `high_res_spec`: High resolution spectrogram
- `low_res_spec`: Low resolution spectrogram
- `sample_idx`: Index of the sample to visualize
- `sr`: Sampling rate
- `duration`: Audio duration

**Output:**
- Saves a figure with three subplots:
  1. Waveform plot
  2. High-resolution spectrogram
  3. Low-resolution spectrogram
- Includes proper axis labels and colorbars
- Saves as PNG file with high DPI (300)

**More Info on Visualization:**
- Properly labels the waveform with time in seconds
- Adds accurate time labels to both spectrograms in seconds
- Adds frequency labels in kHz to both spectrograms
- Maintains the different resolutions but shows the same physical range (same time duration and frequency range)

Both spectrograms represent the same 1-second audio clip at different resolutions.
For the frequency axis we use the Nyquist frequency (half the sampling rate) as the maximum frequency.

### `preprocess_audiomnist(data_path, output_dir, sr=16000, duration=1.0, n_mels_high=128, n_mels_low=32, time_steps_high=128, time_steps_low=32, test_size=0.2, valid_size=0.2)`

Main preprocessing function that orchestrates the entire pipeline.

**Parameters:**
- `data_path`: Path to AudioMNIST dataset
- `output_dir`: Directory to save preprocessed data
- `sr`: Sampling rate
- `duration`: Audio duration
- `n_mels_high`: High resolution mel bands
- `n_mels_low`: Low resolution mel bands
- `time_steps_high`: High resolution time steps
- `time_steps_low`: Low resolution time steps
- `test_size`: Proportion of data for test set (default: 0.2)
- `valid_size`: Proportion of data for validation set (default: 0.2)

**Process:**
1. Creates output directory
2. Loads raw audio data
3. Creates spectrograms
4. Splits data into train/validation/test sets
5. Saves preprocessed data as pickle file
6. Generates visualization examples
7. Returns preprocessed data dictionary

## Data Structure

The preprocessed data is saved as a dictionary with the following structure:
```python
{
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
```

## Usage Example

```python
data_path = '/path/to/audiomnist'
output_dir = 'data/audio_mnist/preprocessed'

data = preprocess_audiomnist(
    data_path=data_path,
    output_dir=output_dir,
    n_mels_high=128,
    n_mels_low=32,
    time_steps_high=128,
    time_steps_low=32
)
```

## Notes

- The preprocessing pipeline ensures consistent audio lengths and spectrogram dimensions
- High and low resolution spectrograms are created to support different model architectures
- Data is split into train/validation/test sets while maintaining class balance
- Visualizations are generated to verify the preprocessing quality
- All parameters are saved with the preprocessed data for reproducibility 