"""
    Author: Yassin Riyazi
    Date: 08-05-2025
    Description: This module contains classes for reading and processing CSV files
    
    I assumed the tilt, fluid, repetition structure is consistent in df and images.
"""

import glob
from torch.utils.data import Dataset
import os,torch
import pandas as pd
import numpy as np

class CSVReader:
    def replace_non_values_with_zero(self):
        """
        Replace non-values in the DataFrame with zero.

        Returns:
            None
        """
        self.df = self.df.replace({np.nan: 0, None: 0})

    def __init__(self,
                 file_path:str,
                 delete_columns:bool=None):
        """
        Initialize the CSVReader class.

        Args:
            file_path (str): The path to the CSV file.
            delete_columns (list[str], optional): A list of column names to delete. Defaults to None.

        Raises:
            FileNotFoundError: If the file does not exist.
        
        raises:
            ValueError: If the CSV file is empty or invalid.
            FileNotFoundError: If the file does not exist.

        """
        try:
            self.df = pd.read_csv(file_path, usecols=["time (s)",
                                                      "x_center (cm)",
                                                      "adv (degree)",
                                                      "rec (degree)",
                                                      "contact_line_length (cm)",
                                                      "y_center (cm)",
                                                      "middle_angle_degree (degree)",
                                                      "velocity (cm/s)"])  # Ignore the first column
        except FileNotFoundError:
            print(f"File {file_path} not found.")
            raise FileNotFoundError
        except pd.errors.EmptyDataError:
            print(f"File {file_path} is empty.")
            raise ValueError
        except pd.errors.ParserError:
            print(f"File {file_path} is invalid.")
            raise ValueError

        if delete_columns is not None:
            self.df = self.df.drop(columns=delete_columns)

        # self.replace_non_values_with_zero()

    def get_values(self,
                   row_index:int,
                   row_end:int) -> np.ndarray:
        """
        Get the values for every column at the specified row index.

        Args:
            row_index (int): The index of the row to retrieve values from.
            row_end (int): The end index of the row to retrieve values from.

        Returns:
            numpy.ndarray: A NumPy array containing the values for every column at the specified row index.
        """
        if row_index < 0 or row_index >= self.df.shape[0]:
            raise IndexError("Row index out of range.")

        values = self.df.iloc[row_index:row_end].values
        return np.array(values)
    
class CSVDataset(Dataset):
    def __init__(self,
                 data:list[str,float],
                 seq_len=10,
                 stride=2) -> None:
        """
        Initialize the CSVDataset class.

        Args:
            folder_path (str): The path to the folder containing the CSV files.
            seq_len (int, optional): The length of the input sequences. Defaults to 100.
            stride (int, optional): The stride for moving the window. Defaults to 15.
        """
        self.stride         = stride
        self.seq_len        = seq_len
        # self.folder_path    = data

        self.data           = CSVReader(os.path.join(data[0],'SR_result','result.csv'))
        self.viscosity      = torch.tensor(data[1])

        self.viscosity_data = pd.read_csv('/media/roboprocessing/Data/frame_Extracted_Vids_DFs/DATA_Sheet.csv')
        self.fluids         = self.viscosity_data["Bottle number"]

    def __len__(self):
        return (self.data.df.shape[0] - self.seq_len) // self.stride + 1

    def __getitem__(self,
                    idx:int) -> torch.Tensor:
        """
        Get a sequence of data from the dataset.

        Args:
            idx (int): The index of the sequence to retrieve.

        Returns:
            torch.Tensor: A tensor containing the sequence of data.
        """
        _data   = self.data.get_values(idx * self.stride, idx * self.stride + self.seq_len)
        tensor  = torch.tensor(_data, dtype=torch.float32)

        # _temp = self.folder_path.split(os.sep)[-2]  # Second last directory name
        # label = [ii for ii in self.fluids if ii in _temp]
        # if not label:
        #     raise ValueError(f"No label found for image {self.folder_path}")
        # dfIndex = self.viscosity_data.index[self.viscosity_data["Bottle number"] == label[0]]
        # label = self.viscosity_data.iloc[dfIndex]['Viscosity 25C']

        # return tensor, torch.tensor(label.values, dtype=torch.float32)
        return tensor,self.viscosity
        
class TimeSeriesDataset_dataframe(Dataset):
    def __init__(self,
                 root_dirs:list[str],
                 seq_len:int=100,
                 stride:int=2):
        self.root_dir = root_dirs
        self.datasets = {}
        for dfs in sorted(root_dirs):
            self.datasets[dfs] = CSVDataset(dfs, seq_len=seq_len, stride=stride)

    def __len__(self):
        vv = 0
        for dataset in list(self.datasets.keys()):
            try:
                vv += len(self.datasets[dataset])
            except Exception as e:
                print(f"Error processing dataset {dataset}: {e}")
        return vv

    def __getitem__(self, idx):
        for dataset in self.datasets.keys():
        #     try:
        #         if idx < len(self.datasets[dataset]):
        #             return self.datasets[dataset][idx]
        #         idx -= len(self.datasets[dataset])
        #     except Exception as e:
        #         print(f"Error processing dataset {dataset}: {e}")
            if idx < len(self.datasets[dataset]):
                return self.datasets[dataset][idx]
            idx -= len(self.datasets[dataset])

    
def get_subdirectories(root_dir, max_depth=2):
    directories = []
    for root, dirs, _ in sorted(os.walk(root_dir)):
        if root == root_dir:
            continue  # Skip the root directory itself
        depth = root[len(root_dir):].count(os.sep)
        if depth < max_depth:
            directories.append(root)
        else:
            del dirs[:]  # Stop descending further
    return directories

if __name__ == "__main__":
    # dirs = []
    # root_directory = "/media/roboprocessing/Data/frame_Extracted_Vids_DFs"
    # for tilt in sorted(glob.glob(os.path.join(root_directory, "*"))):
    #     for fluid in sorted(glob.glob(os.path.join(tilt, "*"))):
    #         for idx, repetition in enumerate(sorted(glob.glob(os.path.join(fluid, "*")))):
    #             if idx < 5:
    #                 dirs.append(repetition)
    dirs = ['/media/roboprocessing/Data/frame_Extracted_Vids_DFs/280/S3-SNr2.6_D/frame_Extracted20250622_212730_DropNumber_01', '/media/roboprocessing/Data/frame_Extracted_Vids_DFs/280/S3-SNr2.6_D/frame_Extracted20250622_212730_DropNumber_02', '/media/roboprocessing/Data/frame_Extracted_Vids_DFs/280/S3-SNr2.6_D/frame_Extracted20250622_212730_DropNumber_03', '/media/roboprocessing/Data/frame_Extracted_Vids_DFs/280/S3-SNr2.6_D/frame_Extracted20250622_212730_DropNumber_04', '/media/roboprocessing/Data/frame_Extracted_Vids_DFs/280/S3-SNr2.6_D/frame_Extracted20250622_212730_DropNumber_05', '/media/roboprocessing/Data/frame_Extracted_Vids_DFs/280/S3-SNr2.6_D/frame_Extracted20250622_212730_DropNumber_06', '/media/roboprocessing/Data/frame_Extracted_Vids_DFs/285/S3-SNr2.6_D/frame_Extracted20250622_211755_DropNumber_01', '/media/roboprocessing/Data/frame_Extracted_Vids_DFs/285/S3-SNr2.6_D/frame_Extracted20250622_211755_DropNumber_02', '/media/roboprocessing/Data/frame_Extracted_Vids_DFs/285/S3-SNr2.6_D/frame_Extracted20250622_211755_DropNumber_03', '/media/roboprocessing/Data/frame_Extracted_Vids_DFs/285/S3-SNr2.6_D/frame_Extracted20250622_211755_DropNumber_04']
    dataset = TimeSeriesDataset_dataframe(dirs, seq_len=100, stride=2)

    print(f"Total number of sequences in the dataset: {len(dataset)} {dataset[1][0].shape}")