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
import  pandas                      as      pd # type: ignore
from    sklearn.model_selection     import  train_test_split # type: ignore
from    torch.utils.data            import  Dataset
from    typing                      import  List, Dict, Union, Tuple

if __name__ == "__main__":
    from    header                  import  StringListDict, setSeed, DaughterSet_getitem_
    from    DaughterFolderDataset   import  DaughterFolderDataset
else:
    from    .header                 import  StringListDict, setSeed, DaughterSet_getitem_
    from    .DaughterFolderDataset  import  DaughterFolderDataset

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
                 resize: Tuple[int, int],
                 dicAddresses: StringListDict ,
                 extension: str,
                 stride: int = 1,
                 sequence_length: int = 1,  # New parameter for sequence length
                 ) -> None:
        """
        Args:
            dicAddresses (StringListDict): Dictionary containing image addresses.
            extension (str): File extension of images.
            stride (int): Stride factor for selecting images.
            sequence_length (int): Number of consecutive images to load per sample.

        """
        setSeed(42)
        self.resize = resize
        # self.data_dir = data_dir
        self.extension = extension
        self.stride = stride
        self.sequence_length = sequence_length
        self.dicAddresses = dicAddresses


        self.viscosity_data = pd.read_csv("/home/d2u25/Desktop/Main/Projects/Viscosity/DATA_Sheet.csv") # type: ignore
        self.fluids = self.viscosity_data["Bottle number"]

        # Fallback: regenerate from scratch
        self.DaughterSetLoader(self.dicAddresses)
        self.splits = None

        self._MaximumViscosityGetter = self.MaximumViscosityGetter()
        print(f"Maximum viscosity in dataset: {self._MaximumViscosityGetter}")

    def MaximumViscosityGetter(self) -> float:
        maxViscosity = 0.0
        for DaughterSet in self.DaughterSets.values():
            viscosity = DaughterSet[0][1].item()
            if viscosity > maxViscosity:
                maxViscosity = viscosity
        return maxViscosity

    def DaughterSetLoader(self,
                          dicAddresses: StringListDict,
                          verbose: bool = True) -> None:
        """
        Provide a dictionary of fluid name as key and experiment repetition in a list
        Caution:
            If a fluid in all tilts angles doesn't exist, it will be ignored and removed from fluids.
        """
        
        minLength:int = int(1e12)
        self.DaughterSets: Dict[str, DaughterFolderDataset] = {}

        kk = tqdm.tqdm(dicAddresses.items())
        for fluid, dirs in kk:
            kk.set_postfix({f"Loading DaughterSet for fluid: ": ""}) # type: ignore
            self.DaughterSets[fluid] = DaughterFolderDataset(dirs,
                                                            seq_len=self.sequence_length,
                                                            stride=self.stride,
                                                            extension=self.extension,
                                                            resize=self.resize)
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
                Data = dataset[idx] # type: ignore
                return Data[0], Data[1]/self._MaximumViscosityGetter 
            idx -= len(dataset)
        raise IndexError(f"Index {idx} out of range after traversing all DaughterSets.")

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

    return dicAddressesTrain, dicAddressesValidation, dicAddressesTest

def save_dataset_with_splits(root: str = os.path.join(os.path.dirname(__file__), "dataset_splits"),
                             DataAddress: str = "/media/d2u25/Dont/frames_Process_30",
                                 ) -> None:
    fluidNames: set[str] = set() # type: ignore
    for tilt in glob.glob(os.path.join(DataAddress, "*")):
        for fluid in glob.glob(os.path.join(tilt, "*")):
            fluidNames.add(os.path.basename(fluid)) # type: ignore
    # breakpoint()
    fluidNames:List[str] = sorted(list(fluidNames))

    dicAddressesTrain:Dict[str, List[str]]        = {}
    dicAddressesValidation:Dict[str, List[str]]   = {}
    dicAddressesTest:Dict[str, List[str]]         = {}
    for fluid in fluidNames:
        dicAddressesTrain[fluid] = []
        dicAddressesValidation[fluid] = []
        dicAddressesTest[fluid] = []
        for SubFluid in glob.glob(os.path.join(DataAddress, "*", fluid)):

            Reps = glob.glob(os.path.join(SubFluid, "*"))
            for i, rep in enumerate(Reps):
                if len(Reps) > 5 and i < 4:
                    dicAddressesTrain[fluid].append(rep)
                elif i > 4 and i < 7:
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
    # Example usage
    # dicAddressesTrain, dicAddressesValidation, dicAddressesTest = dicLoader(root="/media/d2u25/Dont/frames_Process_30",
    #                                                                         rootAddress="Projects/Viscosity/")
    
    # breakpoint()
    # dicAddressesTrain = DS_limiter_inv(dicAddressesTrain,['S3-SDS01_D', 'S3-SDS10_D','S3-SDS99_D'],['/280/','/285/','/290/','/295/','/300/',])
    # print("Debugging MotherFolderDataset.py")
    # dataset = MotherFolderDataset(dicAddresses=dicAddressesTrain,
    #                               extension=".png",
    #                               stride=1,
    #                               sequence_length=5)
    # print(f"Total samples in training dataset: {len(dataset)}")
    # sample, label = dataset[0]
    # print(f"Sample shape: {sample.shape}, Label: {label}")

    # train_set, val_set, test_set = split_dataset(dataset)
    # print(f"Train set size: {len(train_set.indices)}, Validation set size: {len(val_set)}, Test set size: {len(test_set)}") # type: ignore

    
    # Saving the datasets with splits
    

    # save_dataset_with_splits(mm, train_dataset, val_dataset, test_dataset)


    #%% Debugging DataSetShit
    # dicAddressesTrain, dicAddressesValidation, dicAddressesTest = dicLoader(rootAddress="Projects/Viscosity/",
    #                                                                         root = "/media/d2u25/Dont/frames_Process_30")


    save_dataset_with_splits(root="./dataset_splits",
                             DataAddress="/media/d2u25/Dont/frames_Process_30")