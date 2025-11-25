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
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple, TypeAlias, NamedTuple, cast

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
DaughterSet_getitem_: TypeAlias = tuple[torch.Tensor, torch.Tensor, torch.Tensor]

DROP_POSITION_ORDER: Tuple[str, str] = ("x_center", "y_center")
SROF_ORDER: Tuple[str, ...] = (
    "time",
    "x_center",
    "y_center",
    "adv",
    "rec",
    "middle_angle_degree",
    "contact_line_length",
    "velocity",
)


@dataclass(frozen=True)
class FeatureSelection:
    """Encapsulates feature toggle choices used for ablation studies."""

    use_tilt: bool
    use_count: bool
    drop_flags: Tuple[bool, bool]
    srof_flags: Tuple[bool, bool, bool, bool, bool, bool, bool, bool]

    @staticmethod
    def _to_bool(value: Any, default: bool = True) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return default

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> "FeatureSelection":
        cfg = dict(cfg or {})

        drop_cfg_raw = cfg.get("drop_position", {})
        drop_cfg: Mapping[str, Any]
        if isinstance(drop_cfg_raw, Mapping):
            drop_cfg = cast(Mapping[str, Any], drop_cfg_raw)
        else:
            drop_cfg = {}

        srof_cfg_raw = cfg.get("srof", {})
        srof_cfg: Mapping[str, Any]
        if isinstance(srof_cfg_raw, Mapping):
            srof_cfg = cast(Mapping[str, Any], srof_cfg_raw)
        else:
            srof_cfg = {}

        drop_flags: list[bool] = []
        for key in DROP_POSITION_ORDER:
            value = drop_cfg.get(key, True)
            drop_flags.append(cls._to_bool(value, True))

        srof_flags: list[bool] = []
        for key in SROF_ORDER:
            value = srof_cfg.get(key, True)
            srof_flags.append(cls._to_bool(value, True))

        drop_tuple = cast(Tuple[bool, bool], tuple(drop_flags))
        srof_tuple = cast(Tuple[bool, bool, bool, bool, bool, bool, bool, bool], tuple(srof_flags))

        return cls(
            use_tilt=cls._to_bool(cfg.get("tilt_angle", True), True),
            use_count=cls._to_bool(cfg.get("frame_count", True), True),
            drop_flags=drop_tuple,
            srof_flags=srof_tuple,
        )

    def to_config(self) -> Dict[str, Any]:
        return {
            "tilt_angle": self.use_tilt,
            "frame_count": self.use_count,
            "drop_position": {
                key: self.drop_flags[i]
                for i, key in enumerate(DROP_POSITION_ORDER)
            },
            "srof": {
                key: self.srof_flags[i]
                for i, key in enumerate(SROF_ORDER)
            },
        }

    @property
    def drop_indices(self) -> Tuple[int, ...]:
        return tuple(i for i, flag in enumerate(self.drop_flags) if flag)

    @property
    def srof_indices(self) -> Tuple[int, ...]:
        return tuple(i for i, flag in enumerate(self.srof_flags) if flag)

    @property
    def combined_feature_names(self) -> Tuple[str, ...]:
        names: list[str] = []
        for idx in self.drop_indices:
            names.append(f"drop_position.{DROP_POSITION_ORDER[idx]}")
        for idx in self.srof_indices:
            names.append(f"srof.{SROF_ORDER[idx]}")
        return tuple(names)

    @property
    def final_feature_names(self) -> Tuple[str, ...]:
        names: list[str] = []
        if self.use_tilt:
            names.append("tilt_angle")
        if self.use_count:
            names.append("frame_count")
        names.extend(self.combined_feature_names)
        return tuple(names)

    @property
    def combined_size(self) -> int:
        return len(self.drop_indices) + len(self.srof_indices)

    @property
    def final_size(self) -> int:
        size = self.combined_size
        if self.use_tilt:
            size += 1
        if self.use_count:
            size += 1
        return size