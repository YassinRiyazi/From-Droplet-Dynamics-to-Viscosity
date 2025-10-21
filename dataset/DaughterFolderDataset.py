"""
    Author: Yassin Riyazi
    Date: 04-08-2025
    Description:
        - General purpose self balancing dataset loader.

    TODO:
        - [V] Balancing data with number of images in each repetition.
        - [V] Saving and loading dataset splits.
        - [V] Splitting test, validation and train sets.
        - Generate a random sequence of images from a folder. and test different dataloader settings.


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
import  pickle
import  numpy                   as      np
from    PIL                     import  Image
from    torch.utils.data        import  Dataset
from    torchvision             import  transforms # type: ignore
from    typing                  import  Callable, Tuple, List

try:
    from    .header                  import  BatchAddress, DataSetShit, DaughterSetInput_getitem_, setSeed, DaughterSet_getitem_ # type: ignore
except ImportError:
    from    header                   import  BatchAddress, DataSetShit, DaughterSetInput_getitem_, setSeed, DaughterSet_getitem_ # type: ignore

from scipy.interpolate import interp1d, CubicSpline # type: ignore

import sys
ModuleDetection = "/home/d2u25/Desktop/Main/src/PyThon/ContactAngle/DropDetection"
sys.path.append(ModuleDetection)
from DropDetection_Sum import detectionV2


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

class DataHandler():
    """
        TODO:
            - Add support for more pickles
                - Normalize length of points for each drop
            - Add support for more csv 4S-FROS
    """
    def __init__(self,
                 extension: str,
                 resize: Tuple[int, int]=(201,201)
                ) -> None:
        self.extension = extension
        self.resize = resize

        self.transform: Callable[[Image.Image], torch.Tensor] = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(self.resize),
            transforms.ToTensor(), #Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] 
            # transforms.Normalize((0.5,), (0.5,)),
        ])
        if self.extension == ".png":
            self.forward: Callable[[list[str | os.PathLike[str]]], torch.Tensor] = self.DataHandlerPNG
            self.loaddata = self.loadOrderedImages
            
        elif self.extension == ".pkl":
            self.forward: Callable[[np.ndarray], torch.Tensor] = self.DataHandlerPickle
            self.loaddata = self.loadOrderedPickles

    def DataHandlerPNG(self, DataAddress:List[str | os.PathLike[str]]) -> torch.Tensor:
        """
        Load a PNG image as a grayscale torch.Tensor efficiently.

        Args:
            file_path (str | os.PathLike): Path to the PNG image.

        Returns:
            torch.Tensor: Grayscale image as a float32 tensor.

        Caution:
                Apply transformation here 
                Images should be Grayscale
        """
        seq: list[torch.Tensor] = []
        for file_path in DataAddress:
            data = Image.open(file_path)
            data = self.transform(data)
            seq.append(data)
        # Wrap NumPy array into Torch tensor without copying
        return torch.stack(seq)

    @staticmethod
    def loadOrderedImages(foldersDic:dict[str, DataSetShit],
                        seqLength: int = 2,
                        _Stride: int = 5
                        ) -> list[tuple[float, list[str | os.PathLike[str]]]]:
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
        DataAddress:list[tuple[float, list[str | os.PathLike[str]]]] = []
        index = 0
        _go = True
        while _go:
            _failedCases = 0
            for folder, (count, files, viscosity) in foldersDic.items():
                del folder
                start = index * _Stride
                end = start + seqLength
                if end > (count):
                    _failedCases += 1
                else:
                    DataAddress.append((viscosity, files[start:end]))
            index += 1
            if _failedCases >= foldersDic.keys().__len__():
                _go = False
        return DataAddress
    
    def DataHandlerPickle(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32)
    
    @staticmethod
    def loadOrderedPickles(foldersDic:dict[str, DataSetShit],
                           pLength = 300,
                            seqLength: int = 2,
                            _Stride: int = 5
                            ) -> DaughterSetInput_getitem_:
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
        for folder, (count, files, viscosity) in foldersDic.items():
            with open(files[0], 'rb') as f:
                
                sample = pickle.load(f)
            data = np.zeros((len(sample), pLength, 3))

            for adress, array in sample.items():
                dataIndex = int(adress.split(os.sep)[-1].split('.')[0].split('_')[-1])-1
                time_index = float(dataIndex)/4000
                x,y = interpolate_motion_extended(array[:,0], array[:,1], pLength)

                data[dataIndex, :, 0] = x
                data[dataIndex, :, 1] = y
                data[dataIndex, :, 2] = time_index  # Use the calculated time_index

            foldersDic[folder] = DataSetShit(len(sample), data, viscosity) # type

        DataAddress:DaughterSetInput_getitem_ = []
        index = 0
        _go = True
        while _go:
            _failedCases = 0
            for folder, (count, datas, viscosity) in foldersDic.items():
                del folder
                start = index * _Stride
                end = start + seqLength
                if end > (count):
                    _failedCases += 1
                else:
                    DataAddress.append((viscosity, datas[start:end]))
            index += 1
            if _failedCases >= foldersDic.keys().__len__():
                _go = False
        return DataAddress

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
                 extension: str,
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
        self.extension      = extension
        self.DataHandler    = DataHandler(extension=self.extension, resize=resize)

        foldersDic          = self.loadAddresses(dirs,self.extension)

        
        self.DataAddress    = self.DataHandler.loaddata(foldersDic,
                                                        seqLength=seq_len,
                                                        _Stride=stride)


    def __len__(self):
        return len(self.DataAddress)

    def __getitem__(self, idx:int) -> DaughterSet_getitem_:
        seq_tensor = self.DataHandler.forward(self.DataAddress[idx][1])

        return seq_tensor, torch.tensor(self.DataAddress[idx][0], dtype=torch.float32)
    
    @staticmethod
    def loadAddresses(data_address:BatchAddress,
                            extension: str,
                        ) -> dict[str, DataSetShit]:
        """
        load image addresses from a directory.
        Args:
            data_address (str | os.PathLike): The path to the directory containing image folders.
            extension (str): The file extension of the images to load.
        Returns:
            dict[str, tuple[int, list[str | os.PathLike]]]: A dictionary mapping folder names to a tuple containing the number of images and a list of image paths.
        """
        assert extension.startswith('.'), "Extension should start with a dot (e.g., '.png')"

        foldersDic: dict[str, DataSetShit] = {}
        for folder in data_address:
            viscosity = float(os.path.basename(folder).split("_")[-1])
            files =  sorted(glob.glob(os.path.join(folder, f"*{extension}")))

            foldersDic[folder] = DataSetShit(len(files), files, viscosity) # type: ignore
        return foldersDic

if __name__ == "__main__":
    vv = DaughterFolderDataset(dirs=['/media/d2u25/Dont/frames_Process_30/285/S2-SNr2.5_D/T317_02_20.670000000000'],
                         seq_len=1,
                         stride=15,
                         extension=".png")
    # # vv.dataNormalizer(500)
    # print(vv[10][0].shape, vv.__len__())
    # # import glob
    # # print()
    # # loadOrderedPickles()
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(16,9))
    # colors = plt.cm.RdBu(np.linspace(0, 1, len(vv[10][0][0])))  # Array of RGBA colors from RdBu colormap

    # plt.scatter(vv[10][0][0][:,0], vv[10][0][0][:,1], c=colors, s=6, alpha=0.8)
    # plt.axis('equal')
    # plt.ylim(0, 130)
    # plt.show()
    vv[0]
    print("Done")

    
    np_img = vv[0][0].mul(255).numpy()
    np_img = np_img.squeeze(0).astype(np.int32)   # shape: (H, W)


    import matplotlib.pyplot as plt
    plt.imshow(np_img, cmap='gray')
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()