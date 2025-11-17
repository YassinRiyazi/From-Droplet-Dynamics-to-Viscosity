import os
from sklearn import utils
import yaml
import torch
import random
import numpy as np
from    torch.utils.data            import  Dataset
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import dataset

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

def set_randomness(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class data_set:
    def __init__(self,):

        self.cache_dir = "/home/d25u2/Desktop/From-Droplet-Dynamics-to-Viscosity/Output"
        if not os.path.exists(self.cache_dir):
            raise FileNotFoundError(f"Cache directory {self.cache_dir} does not exist.")
        
        

        # cache_test = os.path.join(cache_dir, f"dataset_cache_test.pkl")
        self._case      = config['Dataset']['embedding']['positional_encoding']
        self.Ref        = config['Dataset']['reflection_removal']
        self.AElayers = config['Training']['Constant_feature_AE']['AutoEncoder_layers']
    
    def load_addresses(self) -> None:
        self.dicAddressesTrain, self.dicAddressesValidation, self.dicAddressesTest = dataset.dicLoader(root = config['Dataset']['Dataset_Root'],)
    
    def load_datasets(self, 
                      embedding_dim: int,
                        stride=config['Training']['Constant_feature_AE']['Stride'],
                        sequence_length=config['Training']['Constant_feature_AE']['window_Lenght']
                      ) -> tuple:
        
        self.ID = f"{config['Dataset']['embedding']['positional_encoding']}_s{stride}_w{sequence_length}"
        self.cache_train    = os.path.join(self.cache_dir, f"dataset_cache_train_{self.ID}.pkl")
        self.cache_val      = os.path.join(self.cache_dir, f"dataset_cache_val_{self.ID}.pkl")
        self.model_name = f"CNN_AE_{self.AElayers}_{config['Training']['Constant_feature_AE']['Architecture']}_{self._case}_{embedding_dim}_{self.Ref=}_{self.ID}"
        # ===== TRAINING DATASET =====
        if os.path.exists(self.cache_train):
            print("Loading TRAINING dataset from cache...")
            self.train_dataset = dataset.MotherFolderDataset.load_cache(self.cache_train)
        else:
            print("Creating TRAINING dataset from scratch (this will take time)...")
            self.train_dataset = dataset.MotherFolderDataset(
                dicAddresses=self.dicAddressesTrain,
                stride=stride,
                sequence_length=sequence_length
            )
            print("Saving training dataset cache...")
            self.train_dataset.save_cache(self.cache_train)
        
        # ===== VALIDATION DATASET =====
        if os.path.exists(self.cache_val):
            print("Loading VALIDATION dataset from cache...")
            self.val_dataset = dataset.MotherFolderDataset.load_cache(self.cache_val)
        else:
            print("Creating VALIDATION dataset from scratch...")
            self.val_dataset = dataset.MotherFolderDataset(
                dicAddresses=self.dicAddressesValidation,
                stride=stride,
                sequence_length=sequence_length
            )
            print("Saving validation dataset cache...")
            self.val_dataset.save_cache(self.cache_val)
        return self.train_dataset, self.val_dataset
    
    def reflectionReturn_Setter(self, flag: bool = True) -> None:
        self.train_dataset.reflectionReturn_Setter(flag)
        self.val_dataset.reflectionReturn_Setter(flag)
        return self.train_dataset, self.val_dataset

    def ablation(self,):
        # TODO: Implement ablation study functionality
        raise NotImplementedError
