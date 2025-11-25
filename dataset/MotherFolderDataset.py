"""
    Author: Yassin Riyazi

    Date:
        - 06-08-2025
        - 08-08-2025: 
            separated DaughterFolderDataset and MotherFolderDataset

    Learned:
        - Make dataset at other branch Markov. 
        Keep all necessary data if each folder should be loaded separately.

"""

import  os
# from sympy import root
import glob
import  tqdm
import  torch
import  pickle
import  numpy                       as      np
import  pandas                      as      pd # type: ignore
from    sklearn.model_selection     import  train_test_split # type: ignore
from    torch.utils.data            import  Dataset
from    typing                      import  List, Dict, Union, Any
from    PIL                         import  Image
from    torchvision                 import  transforms # type: ignore

import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(__file__))
sys.path.append(__file__)
import utils

if __name__ == "__main__":
    from    header                  import  StringListDict, setSeed, DaughterSet_getitem_, FeatureSelection
    from    DaughterFolderDataset   import  DaughterFolderDataset
    from    light_source.LightSourceReflectionRemoving import LightSourceReflectionRemover

else:
    from    .header                 import  StringListDict, setSeed, DaughterSet_getitem_, FeatureSelection
    from    .DaughterFolderDataset  import  DaughterFolderDataset
    from    .light_source.LightSourceReflectionRemoving import LightSourceReflectionRemover

class MotherFolderDataset(Dataset[DaughterSet_getitem_]):
    """
        Description:
            Custom Dataset for Image Loading and Preprocessing 
            Supports caching of {index: path} dictionaries using YAML or pickle.
            Supports loading sequences of consecutive images for sequence-based models.

        Caution:
            MotherFolderDataset assumes:
                - Transformation are applied in DaughterFolderDataset

        TODO:
            - Add support for excluding certain fluids based on criteria.
    """

    def __init__(self,
                 dicAddresses: StringListDict ,
                 stride: int = 1,
                 sequence_length: int = 1,  # New parameter for sequence length
                 feature_config: Dict[str, Any] | None = None,
                 ) -> None:
        """
        Args:
            dicAddresses (StringListDict): Dictionary containing image addresses.
            extension (str): File extension of images.
            stride (int): Stride factor for selecting images.
            sequence_length (int): Number of consecutive images to load per sample.

        """
        setSeed(42)
        self.feature_selection = FeatureSelection.from_config(feature_config)
        self.feature_config = self.feature_selection.to_config()
        self.feature_names = self.feature_selection.final_feature_names
        self.feature_signature = self._feature_signature_from_selection(self.feature_selection)

        # self.data_dir = data_dir
        self.stride = stride
        self.sequence_length = sequence_length
        self.dicAddresses = dicAddresses

        # Fallback: regenerate from scratch
        self.DaughterSetLoader(self.dicAddresses, feature_selection=self.feature_selection)
        self.splits = None

        self._MaximumViscosityGetter = self.MaximumViscosityGetter()
        print(f"Maximum viscosity in dataset: {self._MaximumViscosityGetter}")
        
        # Compute normalization statistics across all daughters
        self.compute_normalization_stats()
        print(f"Computed normalization stats: SROF shape {self.srof_mean.shape}")
        
        # Propagate normalization stats back to all daughter datasets
        for daughter in self.DaughterSets.values():
            daughter.srof_mean = self.srof_mean
            daughter.srof_std = self.srof_std
            daughter.feature_selection = self.feature_selection
            daughter.feature_names = self.feature_names
            daughter.combined_feature_dim = self.feature_selection.combined_size
            daughter.final_feature_dim = self.feature_selection.final_size

    @property
    def feature_dim(self) -> int:
        return self.feature_selection.final_size

    @property
    def combined_feature_dim(self) -> int:
        return self.feature_selection.combined_size

    @staticmethod
    def _feature_signature_from_selection(selection: FeatureSelection) -> str:
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

    def save_cache(self, filepath: str) -> None:
        """
        Save a lightweight cache of this MotherFolderDataset to avoid slow re-initialization.
        
        The cache contains:
        - Configuration (dicAddresses, stride, sequence_length)
        - Precomputed DataAddress lists from each DaughterFolderDataset
        - Dataset length and maximum viscosity
        - Precomputed embedding_file dictionaries to avoid regenerating embeddings
        - Normalization statistics (SROF mean/std)
        
        This allows fast loading without re-scanning directories and CSVs.
        
        Args:
            filepath (str): Path where the cache file will be saved (e.g., 'dataset_cache.pkl')
        """
        cache: Dict[str, Any] = {
            'dicAddresses': self.dicAddresses,
            'len': self._len,
            'stride': self.stride,
            'sequence_length': self.sequence_length,
            'maximum_viscosity': self._MaximumViscosityGetter,
            'daughter_data': {},
            'embedding_files': {},
            'srof_mean': self.srof_mean,
            'srof_std': self.srof_std,
            'feature_selection': self.feature_config,
        }
        
        # Extract DataAddress lists and embedding_file dictionaries from each DaughterFolderDataset
        for fluid, ds in self.DaughterSets.items():
            cache['daughter_data'][fluid] = ds.DataAddress
            cache['embedding_files'][fluid] = ds.embedding_file
        
        with open(filepath, 'wb') as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Dataset cache saved to: {filepath}")

    @classmethod
    def load_cache(cls, filepath: str, feature_config: Dict[str, Any] | None = None) -> 'MotherFolderDataset':
        """
        Quickly reconstruct a MotherFolderDataset from a previously saved cache.
        
        This bypasses directory scanning and CSV parsing by reusing saved DataAddress lists.
        Images are still loaded lazily during __getitem__ calls.
        Embedding files are also restored from cache to avoid regeneration.
        
        Args:
            filepath (str): Path to the cache file (e.g., 'dataset_cache.pkl')
            
        Returns:
            MotherFolderDataset: Reconstructed dataset ready for use
        """
        with open(filepath, 'rb') as f:
            cache = pickle.load(f)
        
        # Create instance without calling __init__
        self = object.__new__(cls)
        
        # Restore attributes
        self.dicAddresses           = cache['dicAddresses']
        self.stride                 = cache['stride']
        self.sequence_length        = cache['sequence_length']
        self.splits                 = None
        self._len                   = cache['len']
        self._MaximumViscosityGetter = cache['maximum_viscosity']

        cached_selection = FeatureSelection.from_config(cache.get('feature_selection'))
        requested_selection = FeatureSelection.from_config(feature_config)
        if feature_config is None:
            self.feature_selection = cached_selection
        else:
            if cached_selection != requested_selection:
                raise ValueError(
                    "Cached dataset feature configuration does not match requested configuration. "
                    "Please regenerate the cache or specify matching settings."
                )
            self.feature_selection = requested_selection

        self.feature_config = self.feature_selection.to_config()
        self.feature_names = self.feature_selection.final_feature_names
        self.feature_signature = self._feature_signature_from_selection(self.feature_selection)

        # Restore normalization statistics
        default_mean = np.zeros(self.feature_selection.combined_size, dtype=np.float32)
        default_std = np.ones(self.feature_selection.combined_size, dtype=np.float32)
        self.srof_mean = cache.get('srof_mean', default_mean)
        self.srof_std = cache.get('srof_std', default_std)
        
        # Reconstruct DaughterSets using cached DataAddress lists
        # from .DaughterFolderDataset import DaughterFolderDataset
        
        # Create a prototype to get config-dependent attributes
        try:
            proto = DaughterFolderDataset(
                dirs=[],
                seq_len=self.sequence_length,
                stride=self.stride,
                srof_mean=self.srof_mean,
                srof_std=self.srof_std,
                feature_selection=self.feature_selection,
            )
        except:
            proto = None
        
        class _CachedDaughter(DaughterFolderDataset):
            """Lightweight replacement for DaughterFolderDataset that reuses cached data"""
            def __init__(inner_self, dataaddress: list, embedding_file_cache: dict, proto_ref: Any, 
                        srof_mean: np.ndarray | None, srof_std: np.ndarray | None, selection: FeatureSelection):
                # Inherit attributes from prototype to ensure full compatibility
                # This copies all attributes initialized from config in DaughterFolderDataset
                if proto_ref is not None:
                    inner_self.__dict__.update(proto_ref.__dict__)
                else:
                    super().__init__(
                        dirs=[],
                        seq_len=self.sequence_length,
                        stride=self.stride,
                        srof_mean=srof_mean,
                        srof_std=srof_std,
                        feature_selection=selection,
                    )
                
                # Override DataAddress with cached data
                inner_self.DataAddress = dataaddress
                
                # Restore embedding_file dictionary from cache instead of regenerating
                inner_self.embedding_file = embedding_file_cache
                
                # Set normalization statistics
                inner_self.srof_mean = srof_mean
                inner_self.srof_std = srof_std
                inner_self.feature_selection = selection
                inner_self.feature_names = selection.final_feature_names
                inner_self.combined_feature_dim = selection.combined_size
                inner_self.final_feature_dim = selection.final_size

            
        # Rebuild DaughterSets from cached data
        self.DaughterSets = {}
        for fluid, dataaddr in cache['daughter_data'].items():
            embedding_cache = cache.get('embedding_files', {}).get(fluid, {})
            self.DaughterSets[fluid] = _CachedDaughter(dataaddr, embedding_cache, proto, 
                                                      self.srof_mean, self.srof_std, self.feature_selection)
        
        print(f"Dataset loaded from cache: {filepath}")
        print(f"Maximum viscosity in dataset: {self._MaximumViscosityGetter}")
        print(f"Normalization stats: SROF mean shape {self.srof_mean.shape}, std shape {self.srof_std.shape}")
        
        return self

    def MaximumViscosityGetter(self) -> float:
        maxViscosity = 0.0
        for DaughterSet in self.DaughterSets.values():
            viscosity = DaughterSet[0][1].item()
            if viscosity > maxViscosity:
                maxViscosity = viscosity
        return maxViscosity

    def compute_normalization_stats(self) -> None:
        """
        Compute global mean and std for SROF features across all daughter datasets.
        This ensures consistent normalization across train/val/test splits.
        
        Statistics are computed column-wise and stored as:
        - self.srof_mean: mean values for each feature (drop_position + SROF)
        - self.srof_std: standard deviation for each feature (with epsilon for stability)
        
        Features normalized follow the configured subset of drop_position and SROF features.
        """
        feature_dim = self.feature_selection.combined_size
        if feature_dim == 0:
            self.srof_mean = np.zeros(0, dtype=np.float32)
            self.srof_std = np.ones(0, dtype=np.float32)
            print("Feature selection excludes drop_position and SROF; skipping normalization stats computation.")
            return

        collected: list[np.ndarray] = []

        drop_idx = self.feature_selection.drop_indices
        srof_idx = self.feature_selection.srof_indices

        for daughter in self.DaughterSets.values():
            for data_item in daughter.DataAddress:
                parts: list[np.ndarray] = []
                if drop_idx:
                    parts.append(data_item[2][:, drop_idx])
                if srof_idx:
                    parts.append(data_item[3][:, srof_idx])
                if parts:
                    combined = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
                    collected.append(combined)

        if not collected:
            print("Warning: No feature data found for normalization")
            self.srof_mean = np.zeros(feature_dim, dtype=np.float32)
            self.srof_std = np.ones(feature_dim, dtype=np.float32)
            return

        stacked = np.vstack(collected)
        self.srof_mean = np.mean(stacked, axis=0, dtype=np.float32)
        self.srof_std = np.std(stacked, axis=0, dtype=np.float32)

        epsilon = 1e-8
        self.srof_std = np.where(self.srof_std < epsilon, 1.0, self.srof_std)

        print(f"Normalization stats computed from {stacked.shape[0]} frames")
        print(f"  Selected feature count: {feature_dim}")
        print(f"  Mean: {self.srof_mean}")
        print(f"  Std: {self.srof_std}")

    def DaughterSetLoader(self,
                          dicAddresses: StringListDict,
                          verbose: bool = True,
                          feature_selection: FeatureSelection | None = None) -> None:
        """
        Provide a dictionary of fluid name as key and experiment repetition in a list
        Caution:
            If a fluid in all tilts angles doesn't exist, it will be ignored and removed from fluids.
        """
        
        minLength:int = int(1e12)
        self.DaughterSets: Dict[str, DaughterFolderDataset] = {}

        kk = tqdm.tqdm(dicAddresses.items())
        for fluid, dirs in kk:
            
            kk.set_postfix({"Loading DaughterSet for fluid": fluid}) #type:ignore Updated to show fluid name
            # Pass normalization stats if available (will be None on first load)
            self.DaughterSets[fluid] = DaughterFolderDataset(
                dirs,
                seq_len=self.sequence_length,
                stride=self.stride,
                srof_mean=getattr(self, 'srof_mean', None),
                srof_std=getattr(self, 'srof_std', None),
                feature_selection=feature_selection,
            )
            if self.DaughterSets[fluid].__len__() == 0:
                if verbose:
                    print(f"Fluid {fluid} with viscosity {fluid} has no images and therefore has no effect on training.")
                else:
                    pass
            else:
                minLength = min(minLength, self.DaughterSets[fluid].__len__())
        if verbose:
            print(f"Minimum length across fluids: {minLength}")

        for DaughterSet in self.DaughterSets.values():
            DaughterSet.dataNormalizer(minLength)

        self._len:int = 0
        for DaughterSet in self.DaughterSets.values():
            self._len += DaughterSet.__len__()

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> DaughterSet_getitem_:
        """
        Returns a sequence of images and the label for the last image in the sequence.

        Args:
            idx (int): Index of the starting image in the sequence.

        Returns:
            tuple: (sequence, label)
                - sequence: Tensor of shape (sequence_length, channels, height, width)
                - label: Tensor with the viscosity label for the last image
        """
        # if idx < 0 or idx >= self._len:
        #     raise IndexError(f"Index {idx} out of range for dataset with length {self._len}")

        for viscosityInDic, dataset in self.DaughterSets.items():
            del viscosityInDic
            if idx < len(dataset):
                Data = list(dataset[idx]) # type: ignore
                # convert tuple to list to allow modification
                Data[1] = Data[1]/self._MaximumViscosityGetter
                return tuple(Data)
            idx -= len(dataset)
        raise IndexError(f"Index {idx} out of range after traversing all DaughterSets.")
    
    def reflectionReturn_Setter(self,state: bool,) -> None:
        for dataset in self.DaughterSets.values():
            dataset.ReturnReflection = state


def split_dataset(dataset: MotherFolderDataset,
                  train_size: float = 0.7,
                  val_size: float = 0.15,
                  test_size: float = 0.15,
                  seed: int = 42) -> tuple[Dataset[DaughterSet_getitem_], Dataset[DaughterSet_getitem_], Dataset[DaughterSet_getitem_]]:
    """
    Split a MotherFolderDataset into train, validation, and test sets.

    Args:
        dataset (MotherFolderDataset): The dataset to split.
        train_size (float): Proportion of the dataset to include in the train split.
        val_size (float): Proportion of the dataset to include in the validation split.
        test_size (float): Proportion of the dataset to include in the test split.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
        "train_size + val_size + test_size must equal 1"

    setSeed(seed)
    indices = list(range(len(dataset)))
    train_idx, temp_idx = train_test_split(indices, train_size=train_size, random_state=seed) # type: ignore
    val_ratio = val_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(temp_idx, train_size=val_ratio, random_state=seed) # type: ignore

    train_subset = torch.utils.data.Subset(dataset, train_idx) # type: ignore
    val_subset   = torch.utils.data.Subset(dataset, val_idx)  # type: ignore
    test_subset  = torch.utils.data.Subset(dataset, test_idx) # type: ignore

    return train_subset, val_subset, test_subset

def pathCompleterList(paths:List[Union[str, os.PathLike[str]]],
                      root:Union[str, os.PathLike[str]]) -> List[str]:
    ValidPaths = [os.path.join(root, os.path.relpath(path, '/media/d2u25/Dont/frames_Process_30')) for path in paths]
    return [path for path in ValidPaths if os.path.exists(path)] # type: ignore

def dicLoader(root:Union[None,str, os.PathLike[str]]=None,
              rootAddress:Union[None,str, os.PathLike[str]]=os.path.join(os.path.dirname(__file__), "dataset_splits")
              ) -> tuple[StringListDict, StringListDict, StringListDict]:
    if rootAddress is None:
        rootAddress = os.path.dirname(__file__)

    dicAddressesTrain       = pickle.load(open(os.path.join(rootAddress, "dicAddressesTrain.pkl"),       "rb"))
    dicAddressesValidation  = pickle.load(open(os.path.join(rootAddress, "dicAddressesValidation.pkl"),  "rb"))
    dicAddressesTest        = pickle.load(open(os.path.join(rootAddress, "dicAddressesTest.pkl"),        "rb"))
            
    if root is None:
        return dicAddressesTrain, dicAddressesValidation, dicAddressesTest
    

    for dic in [dicAddressesTrain, dicAddressesValidation, dicAddressesTest]:
        for key in list(dic.keys()):
            dic[key] = pathCompleterList(dic[key], root)
            if dic[key] is None  or len(dic[key]) == 0:
                del dic[key]

    for dic in [dicAddressesTrain, dicAddressesValidation, dicAddressesTest]:
        for key in list(dic.keys()):
            dic[key] = [os.path.normpath(path) for path in dic[key] if os.path.exists(path)]
    return dicAddressesTrain, dicAddressesValidation, dicAddressesTest

def save_dataset_with_splits(root: str = os.path.join(os.path.dirname(__file__), "dataset_splits"),
                             DataAddress: str = "/media/d2u25/Dont/frames_Process_30",
                                 ) -> None:
    fluidNames: set[str] = set() # type: ignore
    for tilt in glob.glob(os.path.join(DataAddress, "*")):
        for fluid in glob.glob(os.path.join(tilt, "*")):
            fluidNames.add(os.path.basename(fluid.split('_')[0])) # type: ignore
    # breakpoint()
    fluidNames:List[str] = sorted(list(fluidNames))

    dicAddressesTrain:Dict[str, List[str]]        = {}
    dicAddressesValidation:Dict[str, List[str]]   = {}
    dicAddressesTest:Dict[str, List[str]]         = {}
    for fluid in fluidNames:
        dicAddressesTrain[fluid] = []
        dicAddressesValidation[fluid] = []
        dicAddressesTest[fluid] = []
        for SubFluid in glob.glob(os.path.join(DataAddress, "*", f"{fluid}_*")):

            Reps = glob.glob(os.path.join(SubFluid, "*"))
            Reps = [folder for folder in Reps if os.path.isdir(folder)]
            for i, rep in enumerate(Reps):
                if len(Reps) >= 6 and i <= 5:
                    dicAddressesTrain[fluid].append(rep)
                elif i > 5 and i < 7:
                    dicAddressesValidation[fluid].append(rep)
                else:
                    dicAddressesTest[fluid].append(rep)

    os.makedirs(root, exist_ok=True)

    with open(os.path.join(root, "dicAddressesTrain.pkl"), "wb") as f:
        pickle.dump(dicAddressesTrain, f)
    with open(os.path.join(root, "dicAddressesValidation.pkl"), "wb") as f:
        pickle.dump(dicAddressesValidation, f)
    with open(os.path.join(root, "dicAddressesTest.pkl"), "wb") as f:
        pickle.dump(dicAddressesTest, f)

StringListDict = Dict[str, List[str]]

def DS_limiter(dataset: StringListDict,
               fluid_Constraints: List[str],
               tilt_Exclusion: List[str]) -> StringListDict:
    # raise NotImplementedError("DS_limiter is not yet finished.")
    _dataset: StringListDict = {}
    for fluid in dataset.keys():
        if fluid in fluid_Constraints:
            continue
        _dataset[fluid] = dataset[fluid]

    if len(fluid_Constraints) == 0:
        _dataset = dataset

    for fluid in _dataset.keys():
        dirs = _dataset[fluid]
        _dataset[fluid] = [dir for dir in dirs if all(excl not in dir for excl in tilt_Exclusion)] # type: ignore

    return _dataset

def DS_limiter_inv(dataset: StringListDict,
                   fluid_Constraints: List[str],
                   tilt_Exclusion: List[str]) -> StringListDict:
    _dataset: StringListDict = {}
    for fluid in dataset.keys():
        if not fluid in fluid_Constraints:
            continue
        _dataset[fluid] = dataset[fluid]
    
    if len(fluid_Constraints) == 0:
        _dataset = dataset

    if len(tilt_Exclusion) == 0:
        return _dataset
    for fluid, dirs in _dataset.items():
        invDirs = [dir for dir in dirs if any(excl in dir for excl in tilt_Exclusion)] # type: ignore
        if len(invDirs) != 0:
            _dataset[fluid] = invDirs

    return _dataset

if __name__ == "__main__":
    
    # # phase 1: create dataset splits and save them
    # save_dataset_with_splits(DataAddress="/media/d25u2/Dont/Viscosity")

    # Phase 2: Load dataset splits and create MotherFolderDataset instances
    # Load dataset splits
    dicAddressesTrain, dicAddressesValidation, dicAddressesTest = dicLoader(
        root="/media/d25u2/Dont/Viscosity"
    )

    cache_dir = "/home/d25u2/Desktop/From-Droplet-Dynamics-to-Viscosity/Output"
    os.makedirs(cache_dir, exist_ok=True)
    ID = f"{utils.config['Dataset']['embedding']['positional_encoding']}_s{utils.config['Training']['Constant_feature_AE']['Stride']}_w{utils.config['Training']['Constant_feature_AE']['window_Lenght']}"
    cache_train = os.path.join(cache_dir, f"dataset_cache_train_{ID}.pkl")
    cache_val = os.path.join(cache_dir, f"dataset_cache_val_{ID}.pkl")
    # cache_test = os.path.join(cache_dir, f"dataset_cache_test.pkl")
    
    
    import matplotlib.pyplot as plt
    # ===== TRAINING DATASET =====
    if os.path.exists(cache_train):
        print("Loading TRAINING dataset from cache...")
        train_dataset = MotherFolderDataset.load_cache(
            cache_train,
            feature_config=utils.data_config.get('features', {})
        )

        item = train_dataset[100]
        train_dataset.reflectionReturn_Setter(state=False)
        img = item[0][0].squeeze().numpy()  # Remove channel dimension
        print(f"Image shape: {img.shape}")
        plt.figure()
        plt.imshow(img, cmap='gray')
        plt.axis('off')  # Hide axis
        plt.show()  


        for dataset in train_dataset.DaughterSets.values():
            dataset.reflect_remover = True
        item = train_dataset[100]
        img = item[0][0].squeeze().numpy()  # Remove channel dimension
        print(f"Image shape: {img.shape}")
        plt.figure()
        plt.imshow(img, cmap='gray')
        plt.axis('off')  # Hide axis
        plt.show() 

        plt.figure()
        train_dataset.reflectionReturn_Setter(state=True)
        item = train_dataset[100]
        img = item[0][0].squeeze().numpy()  # Remove channel dimension
        plt.imshow(img, cmap='gray')
        plt.axis('off')  # Hide axis
        plt.show()
        img = item[1][0].squeeze().numpy()  # Remove channel dimension
        plt.imshow(img, cmap='gray')
        plt.axis('off')  # Hide axis
        plt.show()  

    else:
        print("Creating TRAINING dataset from scratch (this will take time)...")
        train_dataset = MotherFolderDataset(
            dicAddresses=dicAddressesTrain,
            stride=utils.config['Training']['Constant_feature_AE']['Stride'],
            sequence_length=utils.config['Training']['Constant_feature_AE']['window_Lenght']
        )
        print("Saving training dataset cache...")
        train_dataset.save_cache(cache_train)
    
    # ===== VALIDATION DATASET =====

    if os.path.exists(cache_val):
        print("Loading VALIDATION dataset from cache...")
        val_dataset = MotherFolderDataset.load_cache(
            cache_val,
            feature_config=utils.data_config.get('features', {})
        )
    else:
        print("Creating VALIDATION dataset from scratch...")
        val_dataset = MotherFolderDataset(
            dicAddresses=dicAddressesValidation,
            stride=utils.config['Training']['Constant_feature_AE']['Stride'],
            sequence_length=utils.config['Training']['Constant_feature_AE']['window_Lenght']
        )
        print("Saving validation dataset cache...")
        val_dataset.save_cache(cache_val)
    
    # # ===== TEST DATASET =====
    # if os.path.exists(cache_test):
    #     print("Loading TEST dataset from cache...")
    #     test_dataset = MotherFolderDataset.load_cache(cache_test)
    # else:
    #     print("Creating TEST dataset from scratch...")
    #     test_dataset = MotherFolderDataset(
    #         dicAddresses=dicAddressesTest,
    #         stride=1,
    #         sequence_length=5
    #     )
    #     print("Saving test dataset cache...")
    #     test_dataset.save_cache(cache_test)
    
    # Example: Test loading a sample
    # sample, label = dataset[0]
    # print(f"Sample shape: {sample.shape}, Label: {label}")


    # print(f"Total samples in training dataset: {len(dataset)}")
    # breakpoint()
    # dicAddressesTrain = DS_limiter_inv(dicAddressesTrain,['S3-SDS01_D', 'S3-SDS10_D','S3-SDS99_D'],['/280/','/285/','/290/','/295/','/300/',])
    # print("Debugging MotherFolderDataset.py")
    # 
    # sample, label = dataset[0]
    # print(f"Sample shape: {sample.shape}, Label: {label}")

    # train_set, val_set, test_set = split_dataset(dataset)
    # print(f"Train set size: {len(train_set.indices)}, Validation set size: {len(val_set)}, Test set size: {len(test_set)}") # type: ignore

    
    # Saving the datasets with splits
    

    # save_dataset_with_splits(mm, train_dataset, val_dataset, test_dataset)


    #%% Debugging DataSetShit
    # dicAddressesTrain, dicAddressesValidation, dicAddressesTest = dicLoader(rootAddress="Projects/Viscosity/",
    #                                                                         root = "/media/d2u25/Dont/frames_Process_30")
