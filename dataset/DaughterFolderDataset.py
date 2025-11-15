"""
Author:         Yassin Riyazi
Date:           04.08.2025
Description:    General purpose self balancing dataset loader.
License:        GNU General Public License v3.0

TODO:
    - Load csv file aside images
    - Generate a random sequence of images from a folder. and test different dataloader settings.

Changes:
    - 11.11.2025:
        - [X] Wrong dimention on emebdding
        - [] Add a dictionary for different embedding types/sizes.
        - [] Add super resolution option.

    - 08.06.2025:   Initial version.
        - [X] Balancing data with number of images in each repetition.
        - [X] Saving and loading dataset splits.
        - [X] Splitting test, validation and train sets.

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
import  cv2                    
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

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from positional_encoding.PositionalImageGenerator import PE_Generator
    from light_source.LightSourceReflectionRemoving import LightSourceReflectionRemover
    from    header                  import  BatchAddress, DataSetData, DaughterSet_internal_, DaughterSet_getitem_ # type: ignore
else:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(os.path.abspath(__file__))
    from positional_encoding.PositionalImageGenerator import PE_Generator
    from light_source.LightSourceReflectionRemoving import LightSourceReflectionRemover
    from header                   import  BatchAddress, DataSetData, DaughterSet_internal_, DaughterSet_getitem_ # type: ignore

import  utils
from scipy.interpolate import interp1d, CubicSpline # type: ignore

def interpolate_motion(x: NDArray[np.int16], y: NDArray[np.int16], length: int
                       ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
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
    x_new: NDArray[np.float64] = fx(t_new).astype(np.float64)
    y_new: NDArray[np.float64] = fy(t_new).astype(np.float64)

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

        self.wide               = utils.config['Dataset']['wide']
        self.super_res          = utils.config['Dataset'].get('super_resolution', False)
        self.super_res_factor   = utils.config['Dataset']['embedding'].get('super_resolution_factor', 1)
        self.reflect_remover    = utils.config['Dataset']['reflection_removal']
        self.embed_bool = utils.config['Dataset']['embedding']['positional_encoding'] != 'False'
        self.embedID = f"{utils.config['Dataset']['embedding']['positional_encoding']}_PE_height_{utils.config['Dataset']['embedding']['PE_height']}_default_size_"
        if self.super_res:
            self.embedID = f"{utils.config['Dataset']['embedding']['positional_encoding']}_PE_height_{utils.config['Dataset']['embedding']['PE_height']}_default_size_"

        self.embedding_file:dict[int, NDArray[np.uint8] | None] = {}

        self.seq_len        = seq_len
        self.stride         = stride
        # self.DataHandler    = DataHandler(extension=self.extension, resize=resize)

        foldersDic          = self.loadAddresses(dirs,utils.config['image_extension'])

        self.DataAddress    = self.loadOrderedImages(foldersDic,
                                                        seqLength=seq_len,
                                                        _Stride=stride)
        resize = utils.config['Dataset']['resize'][self.wide]
        if self.super_res:
            resize = (resize[0]*self.super_res_factor, resize[1]*self.super_res_factor)
        self.transform: Callable[[Image.Image], torch.Tensor] = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(resize),
            transforms.ToTensor(), #Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] 
            # transforms.Normalize((0.5,), (0.5,)),
        ])
        self.ReturnReflection = False

    def PE_embedding(self,
                     size_x: int,) -> NDArray[np.uint8] | None:       
        positional_encoding : bool = False
        velocity_encoding   : bool = False
        image_width = utils.config['Dataset']['embedding']['default_image_size'][0]
        match utils.config['Dataset']['embedding']['positional_encoding']:
            case 'False':
                return None
            case 'Position':
                positional_encoding = True
                velocity_encoding   = False
                size_x = utils.config['Dataset']['embedding']['default_image_size'][0]
            case 'Velocity':
                positional_encoding = False
                velocity_encoding   = True
            case _:
                raise ValueError(f"Invalid positional encoding option: {utils.config['Dataset']['positional_encoding']}")
        
        PE_height = utils.config['Dataset']['embedding']['PE_height']
        # TODO adjust the height based on config file
        if self.super_res:
            size_x      *= self.super_res_factor
            PE_height   *= self.super_res_factor
            image_width *= self.super_res_factor



        pe_norm = PE_Generator( size_x,
                                Resize              = True,
                                velocity_encoding   = velocity_encoding,
                                positional_encoding = positional_encoding,
                                PE_height           = PE_height,
                                default_image_size  = (image_width,PE_height),
                                )
        return pe_norm

    def image_embedding(self, source_img: NDArray[np.int8], drop_position:NDArray[np.int8], count:int) -> NDArray[np.int8]:
        """
            TODO:
                - Correct the size of the output, right now its scale and crop is wrong but it show some thing
        """

        PE_height = utils.config['Dataset']['embedding']['PE_height']
        drop_height = utils.config['Dataset']['embedding']['drop_height']
        tolerance   = utils.config['Dataset']['cropped_tolerance']
        
        # TODO: Embedding is now made in loadAddresses function.
        endpoint, beginning = drop_position

        if self.super_res:
            PE_height   = PE_height     * self.super_res_factor
            drop_height = drop_height   * self.super_res_factor
            endpoint    = endpoint      * self.super_res_factor
            beginning   = beginning     * self.super_res_factor
            tolerance   = tolerance     * self.super_res_factor

        # pe_norm     = cv2.resize(self.embedding_file, (1245, PE_height), interpolation=cv2.INTER_LINEAR)
        pe_norm     = self.embedding_file[count].copy()
        pe_norm     = pe_norm[:,endpoint-tolerance:beginning+tolerance]

        # TODO: Check if bitwise not is required.
        # cv2.bitwise_not(source_img, source_img)

        # TODO: Morphological operations can be added as an option in config file.
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # source_img     = cv2.morphologyEx(source_img, cv2.MORPH_CLOSE, kernel)

        _, binary_mask  = cv2.threshold(source_img, 20, 255, cv2.THRESH_BINARY_INV)
        contours, _     = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            raise ValueError("No dark regions found in the source image.")
        
        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        # Create a mask for the largest contour
        # https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
        contour_mask = np.zeros(source_img.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        contour_mask = cv2.erode(contour_mask,np.ones((5,5),np.uint8),iterations = 3)
        threshold_activation = 1

        inside = contour_mask <= threshold_activation
        # pe_norm[PE_height-utils.config['Dataset']['embedding']['drop_height']:,
        #         endpoint-utils.config['Dataset']['cropped_tolerance']:beginning+utils.config['Dataset']['cropped_tolerance']][inside] = source_img[inside]
        pe_norm[PE_height-drop_height:,:][inside] = source_img[inside]
        return pe_norm

    def __len__(self):
        return len(self.DataAddress)

    def __getitem__(self, idx:int) -> DaughterSet_getitem_:
        if self.ReturnReflection:
            return self.getitem_Reflection(idx)
        else:
            return self.__getitem__Normal(idx)

    def __getitem__Normal(self, idx:int) -> DaughterSet_getitem_:
        seq: list[torch.Tensor] = []
        
        viscosity       = self.DataAddress[idx][0]
        SROF            = self.DataAddress[idx][3]
        tilt            = self.DataAddress[idx][4]
        count           = self.DataAddress[idx][5]

        for file_path, drop_position in zip(self.DataAddress[idx][1],self.DataAddress[idx][2]):
            pil = Image.open(file_path).convert("L")

            if isinstance(pil, np.ndarray)==False:
                pil = np.array(pil)

            if self.reflect_remover:
                pil = LightSourceReflectionRemover(pil)

            if self.embed_bool:
                pil = self.image_embedding(pil, drop_position, count)

            if isinstance(pil, np.ndarray):
                pil = Image.fromarray(pil.astype(np.uint8))

            data = self.transform(pil)
            seq.append(data)

        seq_tensor = torch.stack(seq)
        return seq_tensor, torch.tensor(viscosity, dtype=torch.float32), torch.tensor(drop_position, dtype=torch.int16), torch.tensor(SROF, dtype=torch.float32), torch.tensor(tilt, dtype=torch.int16)
    
    def getitem_Reflection(self, idx:int) -> tuple[torch.Tensor, torch.Tensor]:
            seq: list[torch.Tensor] = []
            seq_reflection: list[torch.Tensor] = []
            
            # viscosity       = self.DataAddress[idx][0]
            # SROF            = self.DataAddress[idx][3]
            # tilt            = self.DataAddress[idx][4]
            count           = self.DataAddress[idx][5]

            for file_path, drop_position in zip(self.DataAddress[idx][1],self.DataAddress[idx][2]):
                pil = Image.open(file_path).convert("L")
                pil_temp = pil.copy()

                if isinstance(pil, np.ndarray)==False:
                    pil = np.array(pil)
                    pil_temp = np.array(pil_temp)

                pil_temp = LightSourceReflectionRemover(pil_temp)

                if self.embed_bool:
                    pil = self.image_embedding(pil, drop_position, count)
                    pil_temp = self.image_embedding(pil_temp, drop_position, count)

                if isinstance(pil, np.ndarray):
                    pil = Image.fromarray(pil.astype(np.uint8))
                    pil_temp = Image.fromarray(pil_temp.astype(np.uint8))

                data = self.transform(pil)
                seq.append(data)

                pil_temp = self.transform(pil_temp)
                _temp = data - pil_temp
                _temp = _temp.squeeze(0).numpy()
                _temp = self.reflection_detection(_temp)
                _temp = torch.tensor(_temp, dtype=torch.float32).unsqueeze(0)
                seq_reflection.append(_temp)

            # Wrap NumPy array into Torch tensor without copying
            seq_tensor = torch.stack(seq)
            seq_reflection_tensor = torch.stack(seq_reflection)
            return seq_tensor, seq_reflection_tensor
    

    def reflection_detection(self, img_u8:NDArray[np.float32]) -> NDArray[np.int8]:
        img_u8 = cv2.normalize(img_u8, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # --- 1. Threshold to extract dark spot ---
        _, thresh = cv2.threshold(img_u8, 80, 255, cv2.THRESH_BINARY_INV)

        # --- 2. Find contours ---
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # --- 3. Select smallest dark contour ---
        min_area = float("inf")
        smallest_cnt = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 5 < area < min_area:   # skip tiny noise
                min_area = area
                smallest_cnt = cnt

        # --- 4. Create mask ---
        mask = np.zeros_like(img_u8)
        if smallest_cnt is not None:
            cv2.drawContours(mask, [smallest_cnt], -1, 255, -1)

        spot = cv2.bitwise_and(img_u8, img_u8, mask=mask)
        return spot
    

    def checkingFilesExist(self, 
                           files:List[str],
                           dropLocation: pd.DataFrame,
                           SROF: pd.DataFrame,
                           folder: str
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

        # TODO : Fix 4s-SROF processor
        _ignore = files.pop()  # removing last image because its not being processed in the csv files.
        _ignore = {os.path.basename(_ignore)}

        if (filename_set ^ detection_set) - _ignore:
            print(filename_set ^ detection_set)
            raise FileNotFoundError(f"Some files in dropLocation CSV do not match the image files in folder {folder}.")
        
        if (filename_set ^ SROF_set) - _ignore:
            print(filename_set ^ SROF_set)
            raise FileNotFoundError(f"Some files in 4S-SROF CSV do not match the image files in folder {folder}.")
        
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

            
            if self.wide==True:
                files =  sorted(glob.glob(os.path.join(folder, utils.config['full_size_image_folder'],  f"*{extension}")))
            elif self.wide == False and self.super_res == False:
                files =  sorted(glob.glob(os.path.join(folder, utils.config['cropped_image_folder'],    f"*{extension}")))
            elif self.super_res == True:
                files =  sorted(glob.glob(os.path.join(folder, utils.config['super_resolution_cropped'],    f"*{extension}")))
            else:
                raise ValueError("Invalid value for 'wide' configuration.")

            dropLocation = pd.read_csv(os.path.join(folder, utils.config['cropped_image_folder'], "detections.csv")) # type: ignore
            
            SROF = pd.read_csv(os.path.join(folder, utils.config['SROF'])) # type: ignore

            _ = self.checkingFilesExist(files, dropLocation, SROF,folder)
            _lenght = len(files)
            # Embedding generation and saving
            self.embedding_file[_lenght] = self.PE_embedding(size_x=len(files))
            if self.embedding_file[_lenght] is not None:
                size_x = self.embedding_file[_lenght].shape[1]
                size_y = self.embedding_file[_lenght].shape[0]
                cv2.imwrite(os.path.join(folder, f'{self.embedID}{size_x}x{size_y}.png'), self.embedding_file[_lenght])

            foldersDic[folder] = DataSetData(_lenght, files, viscosity, dropLocation, SROF) # type: ignore
        return foldersDic
    
    @staticmethod
    def loadOrderedImages(foldersDic:dict[str, DataSetData],
                        seqLength: int = 2,
                        _Stride: int = 5
                        ) -> list[DaughterSet_internal_]:
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
        DataAddress:list[DaughterSet_internal_] = []
        index = 0
        _go = True
        while _go:
            _failedCases = 0
            for folder, (count, files, viscosity, dropLocation, SROF) in foldersDic.items():
                # del folder
                tilt = 360 - int(folder.split(os.sep)[5])

                start = index * _Stride
                end = start + seqLength
                if end > (count):
                    _failedCases += 1
                else:
                    DataAddress.append((viscosity,
                                        files[start:end],
                                        dropLocation.iloc[start:end,1:].to_numpy(dtype=np.int16),   # type: ignore
                                        SROF.iloc[start:end,1:].to_numpy(dtype=np.float16),         # type: ignore
                                        tilt,
                                        count,
                                        ))      
            index += 1
            if _failedCases >= foldersDic.keys().__len__():
                _go = False
        return DataAddress

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    vv = DaughterFolderDataset(dirs=['/media/d25u2/Dont/Viscosity/280/S5-S2.01_S20/D175220_01_4.46',
                                    #  '/media/d25u2/Dont/Viscosity/280/S5-S2.01_S20/D175220_02_4.46',
                                     '/media/d25u2/Dont/Viscosity/280/S5-S90per_S8/D165644_16_142.40'],
                         seq_len=1,
                         stride=1,)
    
    cc = vv[317]
    print(cc[0].shape,vv.__len__())
    # plotting a tensor image
    img = cc[0][0].squeeze().numpy()  # Remove channel dimension
    plt.imshow(img, cmap='gray')
    plt.axis('off')  # Hide axis
    plt.show()  

    vv.ReturnReflection = True
    cc = vv[317]
    print(cc[0].shape,vv.__len__())
    # plotting a tensor image
    img = cc[0][0].squeeze().numpy()  # Remove channel dimension
    plt.imshow(img, cmap='gray')
    plt.axis('off')  # Hide axis
    plt.show()  
    img = cc[1][0].squeeze().numpy()  # Remove channel dimension
    plt.imshow(img, cmap='gray')
    plt.axis('off')  # Hide axis
    plt.show()
