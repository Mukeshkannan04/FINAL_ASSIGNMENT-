import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class AirDrawDataset(Dataset):
    def __init__(self, data_dir, max_len=200):
        self.data_dir = data_dir
        self.max_len = max_len
        self.file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
        self.scaler = StandardScaler()
        self._fit_scaler()

    def _fit_scaler(self):
        # Fit scaler on all data to normalize features
        all_data = []
        for f in self.file_paths: 
            df = pd.read_csv(f)
            all_data.append(df.values)
        if all_data:
            self.scaler.fit(np.vstack(all_data))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        
        # Extract label from filename (e.g., user2_digit5_trial03.csv -> 5)
        filename = os.path.basename(path)
        try:
            label = int(filename.split('digit')[1].split('_')[0])
        except:
            label = 0 # Fallback if naming fails
        
        df = pd.read_csv(path)
        features = df.values
        
        # Normalize
        features = self.scaler.transform(features)
        
        # Pad or Truncate to fixed length (200)
        if len(features) < self.max_len:
            padding = np.zeros((self.max_len - len(features), 6))
            features = np.vstack((features, padding))
        else:
            features = features[:self.max_len, :]
            
        # Transpose for PyTorch CNN [Channels, Length] -> [6, 200]
        features = torch.tensor(features, dtype=torch.float32).transpose(0, 1)
        
        return features, torch.tensor(label, dtype=torch.long)

def get_dataloaders(data_dir, batch_size=8):
    dataset = AirDrawDataset(data_dir)
    # Split: 80% Train, 20% Test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    return (DataLoader(train_data, batch_size=batch_size, shuffle=True),
            DataLoader(test_data, batch_size=batch_size, shuffle=False))