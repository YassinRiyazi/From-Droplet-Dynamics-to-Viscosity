"""
Author:         Yassin Riyazi
Date:           04.08.2025
Description:    General purpose self balancing dataset loader.
License:        GNU General Public License v3.0

TODO:
    - Load csv file aside images
    - Generate a random sequence of images from a folder. and test different dataloader settings.

Changes:
    - 2024-08-06:   Initial version.
        - [V] Balancing data with number of images in each repetition.
        - [V] Saving and loading dataset splits.
        - [V] Splitting test, validation and train sets.

Learned:
    The type annotation
        1. Callable[[str | os.PathLike], Union[np.ndarray, torch.Tensor]]:

            Callable[[...], ...] → describes a function type.
            Input: str | os.PathLike → the function takes one argument that is either a string (like "file.png") or an os.PathLike (e.g. pathlib.Path).
            Output: Union[np.ndarray, torch.Tensor] → the function must return either a NumPy array or a PyTorch tensor.

        2.dict
            The type annotation dict[str, tuple[int, list[str | os.PathLike]]] in Python describes a dictionary with a specific structure. 
            dict: The data structure is a dictionary, which maps keys to values.
            str: The keys of the dictionary are strings.
            tuple[int, list[str | os.PathLike]]: The values are tuples, where each tuple contains:
                An int as the first element.
                A list as the second element, where the list contains elements that are either str or os.PathLike.
    PDB:
        Similar to GDB, can be very useful for debugging. For example, you can set breakpoints inside Python syntax with if and if it reaches certain conditions, you can inspect variables and the call stack.

        It will automatically integrate points if you run the debugger if it hits breakpoints and in normal run mode it will opens the PDB console.
    
    GLOB:
        Can finds pattern in *<string>* mimicking the ```str in ``` syntax of Python.

    Data structures:
        Type safety and providing aliases helps a lot on code readability and maintainability.
        Basically its a cheat sheet of what is data shape and types.
"""

import  os
import  glob
import  torch
import  random
import  numpy                   as      np
from    PIL                     import  Image
from    torch.utils.data        import  Dataset
from    torchvision             import  transforms # type: ignore
from    typing                  import  Callable, Tuple, List
import pandas as pd
from numpy.typing import NDArray

if __name__ == "__main__" or __package__ is not None:
    from    header                  import  BatchAddress, DataSetData, DaughterSetInput_getitem_, setSeed, DaughterSet_getitem_ # type: ignore
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
else:
    from    .header                   import  BatchAddress, DataSetData, DaughterSetInput_getitem_, setSeed, DaughterSet_getitem_ # type: ignore

import  utils
from scipy.interpolate import interp1d, CubicSpline # type: ignore

def interpolate_motion(x: np.ndarray, y: np.ndarray, length: int):
    """
    Interpolate (x, y) points representing a continuous motion into
    a smooth trajectory with consistent length.

    Args:
        x (np.ndarray): x-coordinates (may contain duplicates).
        y (np.ndarray): y-coordinates.
        length (int): Desired number of resampled points.

    Returns:
        tuple[np.ndarray, np.ndarray]: Interpolated x and y arrays.

    Example:
        x = np.array([0, 1, 1, 2, 3, 5])
        y = np.array([0, 1, 2, 2, 3, 10])
        x_new, y_new = interpolate_motion(x, y, 50)
    """
    # Remove duplicates in (x, y) pairs to avoid zero-length steps
    coords = np.column_stack((x, y))
    _, unique_idx = np.unique(coords, axis=0, return_index=True)
    coords = coords[np.sort(unique_idx)]
    x, y = coords[:, 0], coords[:, 1]

    # Parameterize curve by cumulative arc length
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    t = np.concatenate(([0], np.cumsum(distances)))

    # Normalize parameter to [0, 1]
    t /= t[-1]

    # Interpolate x(t), y(t)
    fx = interp1d(t, x, kind="linear")
    fy = interp1d(t, y, kind="linear")

    # Resample at uniform parameter values
    t_new = np.linspace(0, 1, length)
    x_new = fx(t_new)
    y_new = fy(t_new)

    return x_new, y_new

def interpolate_motion_extended(
    x: np.ndarray,
    y: np.ndarray,
    length: int,
    kind: str = "linear",
    arc_length: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate (x, y) points representing a continuous motion into a smooth trajectory
    with a specified number of points.

    Args:
        x (np.ndarray): 1D array of x-coordinates.
        y (np.ndarray): 1D array of y-coordinates (must match x in length).
        length (int): Desired number of resampled points (must be positive).
        kind (str, optional): Interpolation method ('linear' or 'cubic'). Defaults to 'linear'.
        arc_length (bool, optional): If True, parameterize by arc length; otherwise, use uniform
                                    parameterization. Defaults to True.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Interpolated x and y arrays of length `length`.

    Raises:
        ValueError: If inputs are invalid (e.g., mismatched lengths, insufficient points, invalid kind).
        TypeError: If x or y are not NumPy arrays or length is not an integer.

    Example:
        >>> import numpy as np
        >>> x = np.array([0, 1, 1, 2, 3, 5])
        >>> y = np.array([0, 1, 2, 2, 3, 10])
        >>> x_new, y_new = interpolate_motion(x, y, 50, kind="cubic")
    """
    # Input validation
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("x and y must be NumPy arrays")
    if not isinstance(length, int):
        raise TypeError("length must be an integer")
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1D arrays of the same length")
    if length < 1:
        raise ValueError("length must be a positive integer")
    if kind not in ["linear", "cubic"]:
        raise ValueError("kind must be 'linear' or 'cubic'")

    # Remove duplicates in (x, y) pairs
    coords = np.column_stack((x, y))
    _, unique_idx = np.unique(coords, axis=0, return_index=True)
    coords = coords[np.sort(unique_idx)]
    x_unique, y_unique = coords[:, 0], coords[:, 1]

    # Check for sufficient points
    n_points = len(x_unique)
    if n_points < 2:
        raise ValueError("At least two unique points are required for interpolation")
    if kind == "cubic" and n_points < 3:
        raise ValueError("Cubic interpolation requires at least three unique points")

    # Parameterization
    if arc_length:
        # Arc-length parameterization
        distances = np.sqrt(np.diff(x_unique) ** 2 + np.diff(y_unique) ** 2)
        if np.all(distances == 0):
            raise ValueError("All points are identical; cannot interpolate")
        t = np.concatenate(([0], np.cumsum(distances)))
        t = t / t[-1]  # Normalize to [0, 1]
    else:
        # Uniform parameterization
        t = np.linspace(0, 1, n_points)

    # Interpolation
    t_new = np.linspace(0, 1, length)
    if kind == "linear":
        fx = interp1d(t, x_unique, kind="linear", bounds_error=False, fill_value="extrapolate")
        fy = interp1d(t, y_unique, kind="linear", bounds_error=False, fill_value="extrapolate")
    else:  # cubic
        fx = CubicSpline(t, x_unique, bc_type="natural")
        fy = CubicSpline(t, y_unique, bc_type="natural")

    # Resample
    x_new = fx(t_new)
    y_new = fy(t_new)

    return x_new, y_new

class DaughterFolderDataset(Dataset[DaughterSet_getitem_]):
    def dataNormalizer(self,
                       MaxLength:int) -> None:
        """
            Shuffling data and removing the excess

            Version 1:
                rng = np.random.default_rng(42) # independent RNG, no global state
                rng.shuffle(self.DataAddress)   # deterministic shuffle
                Changed because rng expects numpy array, and converting list to array and back is costly.
        """
        random.seed(42)  # deterministic shuffle
        random.shuffle(self.DataAddress)
        self.DataAddress = self.DataAddress[:MaxLength]

    def __init__(self,
                 dirs: BatchAddress,
                 seq_len: int,
                 stride: int,
                 resize: Tuple[int, int]=(201,201),
                 ):
        """
        Initialize the dataset by loading all files from the specified folder.
        Args:
            folder_path (str | os.PathLike): Path to the folder containing the time series files.
            transform (callable, optional): A function/transform to apply to the data.
            seq_len (int, optional): Length of the sequence to be extracted from the time series.
            stride (int, optional): Step size for extracting sequences.
            extension (str, optional): File extension of the time series files to be loaded.

        Caution:
            Ensure that the DataHandler function is properly defined and can handle the loading of your specific data format.
            data = DataHandler(file_path)

        TODO:
            - Add a holder for data, then fill/rewrite the tensor instead of the list.
        """
        super().__init__()
        assert seq_len is not None, "Sequence length must be specified."
        assert stride is not None, "Stride must be specified."

        self.seq_len        = seq_len
        self.stride         = stride
        # self.DataHandler    = DataHandler(extension=self.extension, resize=resize)

        foldersDic          = self.loadAddresses(dirs,utils.config['image_extension'])

        self.DataAddress    = self.loadOrderedImages(foldersDic,
                                                        seqLength=seq_len,
                                                        _Stride=stride)
        
        self.transform: Callable[[Image.Image], torch.Tensor] = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(resize),
            transforms.ToTensor(), #Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] 
            # transforms.Normalize((0.5,), (0.5,)),
        ])

    def __len__(self):
        return len(self.DataAddress)

    def __getitem__(self, idx:int) -> DaughterSet_getitem_:
        seq: list[torch.Tensor] = []
        for file_path in self.DataAddress[idx][1]:
            data = Image.open(file_path)
            data = self.transform(data)
            seq.append(data)
        # Wrap NumPy array into Torch tensor without copying
        seq_tensor = torch.stack(seq)

        return seq_tensor, torch.tensor(self.DataAddress[idx][0], dtype=torch.float32), torch.tensor(self.DataAddress[idx][2], dtype=torch.int16), torch.tensor(self.DataAddress[idx][3], dtype=torch.float32), torch.tensor(self.DataAddress[idx][4], dtype=torch.int16)
    
    def checkingFilesExist(self, 
                           files:List[str],
                           dropLocation: pd.DataFrame,
                           SROF: pd.DataFrame
                           ) -> bool:
        """
        Check if all files in the provided list of addresses exist.

        Args:
            files (List[str | os.PathLike]): A list of file paths to check.

        Returns:
            bool: True if all files exist, False otherwise.
        """
        filename_set = {os.path.basename(file) for file in files}
        detection_set = set(dropLocation['image'])
        SROF_set = set(SROF['file number'])

        if len(filename_set ^ detection_set)!=0:
            raise FileNotFoundError("Some files in dropLocation CSV do not match the image files.")
        
        if len(filename_set ^ SROF_set)!=0:
            raise FileNotFoundError("Some files in 4S-SROF CSV do not match the image files.")
        
        return True

    def loadAddresses(self,
                      data_address:BatchAddress,
                            extension: str,
                        ) -> dict[str, DataSetData]:
        """
        load image addresses from a directory.
        Args:
            data_address (str | os.PathLike): The path to the directory containing image folders.
            extension (str): The file extension of the images to load.
        Returns:
            dict[str, tuple[int, list[str | os.PathLike]]]: A dictionary mapping folder names to a tuple containing the number of images and a list of image paths.
        """
        assert extension.startswith('.'), "Extension should start with a dot (e.g., '.png')"

        foldersDic: dict[str, DataSetData] = {}
        for folder in data_address:
            viscosity = float(os.path.basename(folder).split("_")[-1])
            files =  sorted(glob.glob(os.path.join(folder, utils.config['full_size_image_folder'], f"*{extension}")))

            dropLocation = pd.read_csv(os.path.join(folder, utils.config['cropped_image_folder'], "detections.csv")) # type: ignore
            SROF = pd.read_csv(os.path.join(folder, utils.config['SROF'])) # type: ignore

            _ = self.checkingFilesExist(files, dropLocation, SROF)

            foldersDic[folder] = DataSetData(len(files), files, viscosity, dropLocation, SROF) # type: ignore
        return foldersDic
    
    @staticmethod
    def loadOrderedImages(foldersDic:dict[str, DataSetData],
                        seqLength: int = 2,
                        _Stride: int = 5
                        ) -> list[tuple[float, list[str | os.PathLike[str]], NDArray[np.int8],  NDArray[np.float16], int]]:
        """
        Load ordered images from the provided dictionary of folders.
        Caution:
            Dictionary is passed by reference.
        Args:
            foldersDic (dict[str, tuple[int, list[str | os.PathLike]]]): A dictionary mapping folder names to a tuple containing the number of images and a list of image paths.
            seqLength (int): The length of the sequence of images to load.
            _Stride (int): The stride to use when loading images.
        Returns:
            list[list[str | os.PathLike]]: A list of lists, where each inner list contains the paths of the ordered images.
        """
        DataAddress:list[tuple[float, list[str | os.PathLike[str]], NDArray[np.int8],  NDArray[np.float16], int]] = []
        index = 0
        _go = True
        while _go:
            _failedCases = 0
            for folder, (count, files, viscosity, dropLocation, SROF) in foldersDic.items():
                # del folder
                tilt = 360 - int(folder.split(os.sep)[4])

                start = index * _Stride
                end = start + seqLength
                if end > (count):
                    _failedCases += 1
                else:
                    DataAddress.append((viscosity, files[start:end],
                                        dropLocation.iloc[start:end,1:].to_numpy(dtype=np.int16),   # type: ignore
                                        SROF.iloc[start:end,1:].to_numpy(dtype=np.float16),         # type: ignore
                                        tilt))      
            index += 1
            if _failedCases >= foldersDic.keys().__len__():
                _go = False
        return DataAddress

if __name__ == "__main__":
    vv = DaughterFolderDataset(dirs=['/media/Dont/Teflon-AVP/280/S2-SNr2.1_D/T528_01_4.460000000000'],
                         seq_len=1,
                         stride=1,)
    
    print(vv[10])
