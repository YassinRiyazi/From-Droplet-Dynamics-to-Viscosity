import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class TimeSeriesDataset(Dataset):
    def __init__(self, root_dir: str,
                 transform: bool = None):
        """
        Args:
            root_dir (string): Directory with all the folders containing .npy files
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Get all .npy files from all subdirectories
        self.file_paths = []
        for folder in self.root_dir.iterdir():
            if folder.is_dir():
                self.file_paths.extend(sorted(folder.glob('*.npy')))
        
        # Load all data into memory
        self.data = []
        for file_path in self.file_paths:
            data = np.load(file_path)
            self.data.append(data)
        
        # Concatenate all time series
        self.data = np.concatenate(self.data, axis=0)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        # Convert to torch tensor
        sample = torch.from_numpy(sample).float()
        
        return sample

# Example usage:
if __name__ == "__main__":
    # Create dataset
    dataset = TimeSeriesDataset(root_dir='emb/320')
    
    # Print dataset size
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"Sample shape: {sample.shape}") 