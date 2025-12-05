import os
import yaml
import torch
import random
import numpy as np
from typing import Any, Mapping, Optional, Tuple, cast

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import dataset
from dataset.header import FeatureSelection



with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

def set_randomness(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) #type: ignore
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class data_set:
    def __init__(self, _data_config: str|None = None) -> None:
        if _data_config is not None:
            with open(_data_config, "r") as file:
                data_config = yaml.safe_load(file)
        else:
            with open("data_config.yaml", "r") as file:
                data_config = yaml.safe_load(file)

        self.cache_dir = "/home/roboprocessing/Desktop/From-Droplet-Dynamics-to-Viscosity/Output"
        if not os.path.exists(self.cache_dir):
            raise FileNotFoundError(f"Cache directory {self.cache_dir} does not exist.")
        
    
        # cache_test = os.path.join(cache_dir, f"dataset_cache_test.pkl")
        self._case      = config['Dataset']['embedding']['positional_encoding']
        self.Ref        = config['Dataset'].get('use_reflection_removal', False)
        self.AElayers = config['Training']['Constant_feature_AE']['AutoEncoder_layers']
        self.SuperResolution = config['Dataset']['super_resolution']
        self.data_config = data_config
        self.train_dataset: Optional[dataset.MotherFolderDataset] = None
        self.val_dataset: Optional[dataset.MotherFolderDataset] = None
        self.feature_selection: Optional[FeatureSelection] = None
        self.feature_dim: Optional[int] = None

    def _feature_signature(self) -> str:
        features_cfg_raw = self.data_config.get('features', {})
        if isinstance(features_cfg_raw, Mapping):
            features_cfg: Mapping[str, Any] = cast(Mapping[str, Any], features_cfg_raw)
        else:
            features_cfg = cast(Mapping[str, Any], {})

        selection = FeatureSelection.from_config(features_cfg)
        self.feature_selection = selection

        parts: list[str] = []
        parts.append(f"tilt{int(selection.use_tilt)}")
        parts.append(f"count{int(selection.use_count)}")

        drop_pairs = (
            (selection.drop_flags[0], "dropx"),
            (selection.drop_flags[1], "dropy"),
        )
        for enabled, short in drop_pairs:
            parts.append(f"{short}{int(enabled)}")

        srof_pairs = [
            (selection.srof_flags[0], "t"),
            (selection.srof_flags[1], "xc"),
            (selection.srof_flags[2], "yc"),
            (selection.srof_flags[3], "adv"),
            (selection.srof_flags[4], "rec"),
            (selection.srof_flags[5], "mid"),
            (selection.srof_flags[6], "cll"),
            (selection.srof_flags[7], "vel"),
        ]
        for enabled, short in srof_pairs:
            parts.append(f"{short}{int(enabled)}")

        return "_".join(parts)
    
    def load_addresses(self) -> None:
        self.dicAddressesTrain, self.dicAddressesValidation, self.dicAddressesTest = dataset.dicLoader(root = config['Dataset']['Dataset_Root'],)
    
    def load_datasets(self, 
                    stride:int=config['Training']['Constant_feature_AE']['Stride'],
                    sequence_length:int=config['Training']['Constant_feature_AE']['window_Lenght']
                        ) -> Tuple[dataset.MotherFolderDataset, dataset.MotherFolderDataset]:
        
        feature_sig = self._feature_signature()
        if self.feature_selection is None:
            raise RuntimeError("Feature selection failed to initialize")
        self.feature_dim = self.feature_selection.final_size
        embedding_dim = self.feature_dim
        canonical_features = self.feature_selection.to_config()

        if self.SuperResolution== True:
            self.id = f"{config['Dataset']['embedding']['positional_encoding']}_s{stride}_w{sequence_length}_SR{self.SuperResolution}_{feature_sig}"
        else:
            self.id = f"{config['Dataset']['embedding']['positional_encoding']}_s{stride}_w{sequence_length}_{feature_sig}"

        self.cache_train    = os.path.join(self.cache_dir, f"dataset_cache_train_{self.id}.pkl")
        self.cache_val      = os.path.join(self.cache_dir, f"dataset_cache_val_{self.id}.pkl")
        self.model_name = f"CNN_AE_{self.AElayers}_{config['Training']['Constant_feature_AE']['Architecture']}_{self._case}_{embedding_dim}_{self.Ref=}_{self.id}"
        # ===== TRAINING DATASET =====
        if os.path.exists(self.cache_train):
            print("Loading TRAINING dataset from cache...")
            self.train_dataset = dataset.MotherFolderDataset.load_cache(
                self.cache_train,
                feature_config=canonical_features
            )
        else:
            print(f"Creating TRAINING dataset from scratch (this will take time)... {self.cache_train}")
            self.train_dataset = dataset.MotherFolderDataset(
                dicAddresses=self.dicAddressesTrain,
                stride=stride,
                sequence_length=sequence_length,
                feature_config=canonical_features
            )
            print("Saving training dataset cache...")
            self.train_dataset.save_cache(self.cache_train)
        
        # ===== VALIDATION DATASET =====
        if os.path.exists(self.cache_val):
            print("Loading VALIDATION dataset from cache...")
            self.val_dataset = dataset.MotherFolderDataset.load_cache(
                self.cache_val,
                feature_config=canonical_features
            )
        else:
            print("Creating VALIDATION dataset from scratch...")
            self.val_dataset = dataset.MotherFolderDataset(
                dicAddresses=self.dicAddressesValidation,
                stride=stride,
                sequence_length=sequence_length,
                feature_config=canonical_features
            )
            print("Saving validation dataset cache...")
            self.val_dataset.save_cache(self.cache_val)
        train_ds = cast(Optional[dataset.MotherFolderDataset], self.train_dataset)
        if train_ds is None:
            raise RuntimeError("Training dataset failed to load")

        val_ds = cast(Optional[dataset.MotherFolderDataset], self.val_dataset)
        if val_ds is None:
            raise RuntimeError("Validation dataset failed to load")

        return train_ds, val_ds
    
    # def reflectionReturn_Setter(self, flag: bool = True) -> Tuple[dataset.MotherFolderDataset, dataset.MotherFolderDataset]:
    #     if self.train_dataset is None or self.val_dataset is None:
    #         raise RuntimeError("Datasets not loaded")
    #     self.train_dataset.reflectionReturn_Setter(flag)
    #     self.val_dataset.reflectionReturn_Setter(flag)
    #     return self.train_dataset, self.val_dataset

    # def ablation(self,) -> None:
    #     # TODO: Implement ablation study functionality
    #     raise NotImplementedError
