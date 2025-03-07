# Data Configuration File

# External dataset paths - modify these to match your environment
AUDIOMNIST_PATH = "/home/mqa887/tmp/audiomnist/AudioMNIST-master/data"
MNIST_PATH = "/path/to/mnist/data"  # Update this with your MNIST location

# Dataset parameters
AUDIO_SAMPLE_RATE = 16000
AUDIO_DURATION = 1.0
MEL_BINS = 128
TIME_STEPS = 128

# Data caching settings (for faster loading)
CACHE_DIR = "/tmp/audio_xai_cache"  # Consider using scratch space on cluster
USE_CACHED_SPECTROGRAMS = True
