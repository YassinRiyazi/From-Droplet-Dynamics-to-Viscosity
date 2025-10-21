import  os
import  torch
import  numpy               as      np
from    torch.utils.data    import  Dataset


class TimeSeriesFolderDataset(Dataset):
    def __init__(self, folder_path,
                 transform=None,
                 seq_len=None,
                 stride=15):
        
        self.folder_path = folder_path
        self.transform = transform
        self.seq_len = seq_len
        self.stride = stride
        self.files = []
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if file.endswith('.npy'):
                self.files.append(file_path)

        # Sort the files by their numerical value
        self.files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    def __len__(self):
        return (len(self.files) - self.seq_len) // self.stride + 1

    def __getitem__(self, idx):
        seq = []
        for i in range(idx * self.stride, idx * self.stride + self.seq_len):
            file_path = self.files[i]
            data = np.load(file_path)
            # Assuming the numpy file contains a single time series
            time_series = data[()]
            # Convert to PyTorch tensor
            time_series = torch.tensor(time_series, dtype=torch.float32)
            # Append to the sequence
            seq.append(time_series)
        # print(file_path)
        # Stack the sequence into a single tensor
        seq = torch.stack(seq)
        # Apply transformation if specified
        if self.transform:
            seq = self.transform(seq)
        return seq


class TimeSeriesDataset(Dataset):
    def __init__(self, root_dir,
                 transform=None,
                 seq_len=100,
                 stride=15):
        self.root_dir = root_dir
        self.transform = transform
        self.datasets = {}
        for folder in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                self.datasets[folder] = TimeSeriesFolderDataset(folder_path, transform, seq_len=seq_len, stride=stride)

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets.values())

    def __getitem__(self, idx):
        for folder, dataset in self.datasets.items():
            if idx < len(dataset):
                return dataset[idx], torch.tensor(float(folder.split("_")[0]), dtype=torch.float32)
            idx -= len(dataset)
        raise IndexError("Index out of range")
    

if __name__ == "__main__":
    vv = TimeSeriesDataset("Data", stride=15)
    vv[10][0].shape