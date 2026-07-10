import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

# Mapping mHealth labels (1-12) to (0-11), and 0 (null) to 12
def map_label(label):
    if label == 0:
        return 12
    elif 1 <= label <= 12:
        return label - 1
    return -1

def load_and_window_subject_data(file_path, window_size=100, step_size=50):
    """
    Loads a single subject's log file for mHealth, extracts right-lower-arm 
    sensor data (columns 15-23), and maps the label (column 24).
    
    Args:
        file_path: pathlib.Path to the subject's log file
        window_size: number of samples per window (default: 100 for 2s at 50Hz)
        step_size: number of samples to slide the window (default: 50 for 50% overlap)
        
    Returns:
        signals: numpy array of shape (num_windows, 9, window_size)
        labels: numpy array of shape (num_windows,)
    """
    data = []
    mapped_labels = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 24:
                    continue
                
                # columns 15 to 23 are indices 14 to 22
                sensor_data = [float(parts[i]) for i in range(14, 23)]
                label = int(parts[23])
                
                data.append(sensor_data)
                mapped_labels.append(map_label(label))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return np.array([]), np.array([])

    data = np.array(data, dtype=np.float32)
    mapped_labels = np.array(mapped_labels, dtype=np.int64)
    
    num_samples = len(data)
    
    windows_signals = []
    windows_labels = []
    
    # Sliding window
    for start in range(0, num_samples - window_size + 1, step_size):
        end = start + window_size
        
        # Extracted window
        window_signal = data[start:end]
        window_label_seq = mapped_labels[start:end]
        
        # Most frequent label in the window using offset to support -1
        counts = np.bincount(window_label_seq + 1)
        mode_idx = counts.argmax()
        mode_label = mode_idx - 1
        
        # Discard window if mode is invalid (-1)
        if mode_label == -1:
            continue
        
        # Append signal transposed to shape (9, window_size)
        windows_signals.append(window_signal.T)
        windows_labels.append(mode_label)
        
    return np.array(windows_signals, dtype=np.float32), np.array(windows_labels, dtype=np.int64)

class MHealthDataset(Dataset):
    def __init__(self, root_path, subject_ids):
        """
        Load mHealth data for specific subjects.
        
        Args:
            root_path: Path object to the mHealth dataset root directory
            subject_ids: List of integers specifying which subjects to load
        """
        all_signals = []
        all_labels = []
        all_subjects = []
        
        root_path = Path(root_path)
        for sbj_id in subject_ids:
            file_path = root_path / f"mHealth_subject{sbj_id}.log"
            if not file_path.exists():
                print(f"Warning: {file_path} not found. Skipping.")
                continue
                
            signals, labels = load_and_window_subject_data(file_path)
            
            all_signals.append(signals)
            all_labels.append(labels)
            all_subjects.extend([sbj_id] * len(labels))
            
        if len(all_signals) > 0:
            self.signals = np.concatenate(all_signals, axis=0)
            self.labels = np.concatenate(all_labels, axis=0)
            self.subjects = np.array(all_subjects)
        else:
            self.signals = np.array([])
            self.labels = np.array([])
            self.subjects = np.array([])

    def compute_statistics(self):
        """
        Compute mean and std over all windows and time steps for each channel.
        Returns:
            mean: array of shape (1, num_channels, 1)
            std: array of shape (1, num_channels, 1)
        """
        if len(self.signals) == 0:
            return 0.0, 1.0
        # self.signals shape: (num_windows, num_channels, window_size)
        mean = self.signals.mean(axis=(0, 2), keepdims=True)
        std = self.signals.std(axis=(0, 2), keepdims=True)
        return mean, std

    def apply_normalization(self, mean, std):
        """
        Apply z-normalization using provided mean and std.
        """
        if len(self.signals) == 0:
            return
        std = np.where(std < 1e-6, 1.0, std)
        self.signals = (self.signals - mean) / std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.signals[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

    @staticmethod
    def load_activity_labels(root_path=None):
        return [
            'Standing still',
            'Sitting and relaxing',
            'Lying down',
            'Walking',
            'Climbing stairs',
            'Waist bends forward',
            'Frontal elevation of arms',
            'Knees bending (crouching)',
            'Cycling',
            'Jogging',
            'Running',
            'Jump front & back',
            'null'
        ]
