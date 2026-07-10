import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path

LABEL_MAP = {
    'Adductor': 0,
    'ArmCurl': 1,
    'BenchPress': 2,
    'LegCurl': 3,
    'LegPress': 4,
    'Null': 5,
    'Riding': 6,
    'RopeSkipping': 7,
    'Running': 8,
    'Squat': 9,
    'StairClimber': 10,
    'Walking': 11
}

def load_and_window_recgym_data(file_path, subject_ids, window_size=100, step_size=50):
    """
    Loads RecGym data for given subjects, wrist position, 
    extracts A_x, A_y, A_z, G_x, G_y, G_z and Workout,
    and applies sliding window.
    """
    df = pd.read_csv(file_path)
    df = df[df['Position'] == 'wrist']
    df = df[df['Subject'].isin(subject_ids)]
    
    windows_signals = []
    windows_labels = []
    
    for sub in subject_ids:
        sub_df = df[df['Subject'] == sub]
        if len(sub_df) < window_size:
            continue
            
        data = sub_df[['A_x', 'A_y', 'A_z', 'G_x', 'G_y', 'G_z']].values.astype(np.float32)
        labels = sub_df['Workout'].map(LABEL_MAP).fillna(-1).values.astype(np.int64)
        
        # # Per-channel z-normalization (zero-mean, unit-variance)
        # # Each of the 6 channels has different units/scales,
        # # so we normalize each independently per subject.
        # channel_mean = data.mean(axis=0, keepdims=True)  # (1, 6)
        # channel_std = data.std(axis=0, keepdims=True)    # (1, 6)
        # channel_std = np.where(channel_std < 1e-6, 1.0, channel_std)  # avoid division by zero for constant channels
        # data = (data - channel_mean) / channel_std
        
        num_samples = len(data)
        for start in range(0, num_samples - window_size + 1, step_size):
            end = start + window_size
            window_signal = data[start:end]
            window_label_seq = labels[start:end]
            
            counts = np.bincount(window_label_seq + 1)
            mode_idx = counts.argmax()
            mode_label = mode_idx - 1
            
            if mode_label == -1:
                continue
                
            windows_signals.append(window_signal.T)
            windows_labels.append(mode_label)
            
    if len(windows_signals) == 0:
        return np.array([]), np.array([])
        
    return np.array(windows_signals, dtype=np.float32), np.array(windows_labels, dtype=np.int64)


class RecGymDataset(Dataset):
    def __init__(self, root_path, subject_ids):
        """
        Load RecGym data for specific subjects.
        
        Args:
            root_path: Path object to the RecGym dataset root directory
            subject_ids: List of integers specifying which subjects to load
        """
        file_path = Path(root_path) / "RecGym.csv"
        if not file_path.exists():
            print(f"Warning: {file_path} not found.")
            self.signals = np.array([])
            self.labels = np.array([])
            self.subjects = np.array([])
            return

        self.signals, self.labels = load_and_window_recgym_data(file_path, subject_ids)
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
        # Return (input, target) as tensors
        x = torch.tensor(self.signals[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

    @staticmethod
    def load_activity_labels(root_path=None):
        return [
            'Adductor', 'ArmCurl', 'BenchPress', 'LegCurl', 'LegPress',
            'Null', 'Riding', 'RopeSkipping', 'Running', 'Squat',
            'StairClimber', 'Walking'
        ]
