"""
    Author: Yassin Riyazi
    Date: 06-08-2025

    Description:
        Custom types and data structures for dataset management in viscosity prediction tasks.
"""

import os
import random
import numpy as np
import torch
import pandas as pd
from numpy.typing import NDArray
from typing import Dict, List, TypeAlias, NamedTuple

def setSeed(seed:int) -> None:
    """
    Set the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # type: ignore
    torch.cuda.manual_seed_all(seed)
    return None

# DaughterFolderDataset 
""" 
A dictionary mapping string keys to `BatchAddress` objects.

This type is useful for storing and organizing lists of dataset 
image addresses, where each key identifies a batch or category.
"""
BatchAddress : TypeAlias = List[str| os.PathLike[str]]
StringListDict: TypeAlias = Dict[str, BatchAddress]

class DataSetData(NamedTuple):
    """
    A mapping from a string identifier to detailed batch information.

    Each value contains:
        - length (int): Number of dataset images in the batch.
        - addresses (BatchAddress): The addresses pointing to dataset images.
        - viscosity (float): The viscosity value associated with this batch.
    """
    length: int
    addresses: BatchAddress
    viscosity: float
    dropLocation: pd.DataFrame | None = None
    SROF: pd.DataFrame | None = None

DaughterSet_internal_: TypeAlias = tuple[float, # viscosity
                                         list[str | os.PathLike[str]], # addresses
                                         NDArray[np.int8], # drop location
                                         NDArray[np.float16], # 4S-SROF
                                         int, # tilt
                                         int, # count of images
                                         ]


DataSetShitInfo: TypeAlias = dict[str, DataSetData]

# DaughterFolderDataset
DaughterSetInput_getitem_: TypeAlias =list[tuple[float, list[str | os.PathLike[str]]]]

# MotherFolderDataset and DaughterFolderDataset __getitem__ return type
DaughterSet_getitem_: TypeAlias = tuple[torch.Tensor, torch.Tensor]