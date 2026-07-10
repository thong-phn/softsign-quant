import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import zipfile
import pandas as pd

# Mapping of activities to integer labels [0, 7]
ACTIVITY_MAP = {
    'walking': 0,
    'running': 1,
    'sitting': 2,
    'standing': 3,
    'lying': 4,
    'climbingup': 5,
    'climbingdown': 6,
    'jumping': 7,
}

def load_sensor(root_path, proband, sensor_prefix, zip_prefix, activity):
    """
    Load data for a specific sensor, subject and activity from zip file.
    Args:
        root_path: Path object to the realworld dataset root (datasets/realworld)
        proband: integer subject ID
        sensor_prefix: e.g. 'acc', 'Gyroscope', 'MagneticField'
        zip_prefix: e.g. 'acc', 'gyr', 'mag'
        activity: string, e.g. 'walking'
    Returns:
        DataFrame containing time, x, y, z
    """
    proband_dir = root_path / f"proband{proband}" / "data"
    zip_path = proband_dir / f"{zip_prefix}_{activity}_csv.zip"
    
    if not zip_path.exists():
        return None
        
    try:
        with zipfile.ZipFile(zip_path) as z:
            target_name = f"{sensor_prefix}_{activity}_waist.csv".lower()
            actual_name = next((n for n in z.namelist() if n.lower() == target_name), None)
            if not actual_name:
                return None
                
            with z.open(actual_name) as f:
                df = pd.read_csv(f)
                
                # Check for columns. Typically 'attr_time', 'attr_x', 'attr_y', 'attr_z'
                if 'attr_time' in df.columns:
                    time_col = 'attr_time'
                    x_col = 'attr_x'
                    y_col = 'attr_y'
                    z_col = 'attr_z'
                elif 'time' in df.columns:
                    time_col = 'time'
                    x_col = 'x'
                    y_col = 'y'
                    z_col = 'z'
                else:
                    print(f"Unknown columns in {actual_name}: {df.columns}")
                    return None
                    
                df = df[[time_col, x_col, y_col, z_col]].copy()
                df.columns = ['time', f'{zip_prefix}_x', f'{zip_prefix}_y', f'{zip_prefix}_z']
                
                # Sort by time to ensure it is strictly monotonic for merge_asof
                df.sort_values('time', inplace=True)
                
                # Remove duplicates by time if any
                df.drop_duplicates(subset=['time'], keep='first', inplace=True)
                return df
    except Exception as e:
        print(f"Error loading {zip_path}: {e}")
        return None

def load_and_window_subject_data(root_path, proband, window_size=100, step_size=50):
    """
    Loads a single subject's data for all activities, aligns acc, gyr, mag, 
    and applies a sliding window.
    """
    windows_signals = []
    windows_labels = []
    
    for activity, label in ACTIVITY_MAP.items():
        df_acc = load_sensor(root_path, proband, 'acc', 'acc', activity)
        df_gyr = load_sensor(root_path, proband, 'Gyroscope', 'gyr', activity)
        df_mag = load_sensor(root_path, proband, 'MagneticField', 'mag', activity)
        
        # We need all three sensors to be available
        if df_acc is None or df_gyr is None or df_mag is None or len(df_acc) == 0:
            continue
            
        # Merge using pandas merge_asof
        df = pd.merge_asof(df_acc, df_gyr, on='time', direction='nearest')
        df = pd.merge_asof(df, df_mag, on='time', direction='nearest')
        
        # Drop rows with NaN (if any due to merging issues, though merge_asof fills them)
        df.dropna(inplace=True)
        
        if len(df) < window_size:
            continue
            
        data = df[['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'mag_x', 'mag_y', 'mag_z']].values.astype(np.float32)
        
        num_samples = len(data)
        
        # Sliding window
        for start in range(0, num_samples - window_size + 1, step_size):
            end = start + window_size
            window_signal = data[start:end]
            
            # window_signal shape is (window_size, 9)
            # transpose to (9, window_size)
            windows_signals.append(window_signal.T)
            windows_labels.append(label)
            
    return np.array(windows_signals, dtype=np.float32), np.array(windows_labels, dtype=np.int64)

class RealworldDataset(Dataset):
    def __init__(self, root_path, subject_ids, window_size=100, step_size=50):
        """
        Load RealWorld data for specific subjects.
        
        Args:
            root_path: Path object to the realworld dataset root directory
            subject_ids: List of integers specifying which subjects to load (e.g. 1 to 15)
        """
        all_signals = []
        all_labels = []
        all_subjects = []
        
        root_path = Path(root_path)
        for proband in subject_ids:
            signals, labels = load_and_window_subject_data(root_path, proband, window_size, step_size)
            
            if len(signals) > 0:
                all_signals.append(signals)
                all_labels.append(labels)
                all_subjects.extend([proband] * len(labels))
            
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
            'walking',
            'running',
            'sitting',
            'standing',
            'lying',
            'climbingup',
            'climbingdown',
            'jumping'
        ]
