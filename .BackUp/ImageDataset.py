import  os
import  glob
import torch
import  tqdm
import  yaml
import  pickle
import  pandas                  as      pd
import  torchvision.transforms  as      transforms
from    PIL                     import  Image
from    torch.utils.data        import  Dataset

class loc_ImageDataset(Dataset):
    """
    Description:
        Custom Dataset for Image Loading and Preprocessing 
        Supports caching of {index: path} dictionaries using YAML or pickle.
        Supports loading sequences of consecutive images for sequence-based models.

    Author: Yassin Riyazi

    Date:
        - 06-08-2025
    """

    def __init__(self,
                 data_dir: str = "/media/d2u25/Dont/frames_Process_15_Patch",
                 extension: str = ".png",
                 skip: int = 1,
                 sequence_length: int = 1,  # New parameter for sequence length
                 index_file: str = "image_index.pkl",
                 load_from_file: bool = True,
                 use_yaml: bool = False):
        """
        Args:
            data_dir (str): Directory containing the images.
            extension (str): File extension of images.
            skip (int): Skip factor for selecting images.
            sequence_length (int): Number of consecutive images to load per sample.
            index_file (str): Path to save/load cached index.       
            load_from_file (bool): Whether to load cached index instead of re-scanning.
            use_yaml (bool): Save index as YAML (otherwise uses pickle).
        """
        self.data_dir = data_dir
        self.extension = extension
        self.skip = skip
        self.sequence_length = sequence_length
        self.index_file = os.path.join(data_dir, index_file)
        self.use_yaml = use_yaml

        if load_from_file and os.path.exists(self.index_file):
            print(f"Loading cached index from {self.index_file}...")
            if self.use_yaml:
                with open(self.index_file, "r") as f:
                    self.image_dict = yaml.safe_load(f)
            else:
                with open(self.index_file, "rb") as f:
                    self.image_dict = pickle.load(f)
        else:
            print("Scanning directories for image files (this may take a while)...")
            image_files = glob.glob(os.path.join(data_dir, "*", "*", "*", "*" + extension))
            image_files = sorted(image_files)[::skip]  # Ensure sorted order for consistency
            self.image_dict = {i: path for i, path in enumerate(image_files)}

            print(f"Saving cached index to {self.index_file}...")
            if self.use_yaml:
                with open(self.index_file, "w") as f:
                    yaml.dump(self.image_dict, f)
            else:
                with open(self.index_file, "wb") as f:
                    pickle.dump(self.image_dict, f)

        # Define transformations
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((201, 201)),
            transforms.ToTensor(),
        ])

        self.viscosity_data = pd.read_csv(os.path.join(data_dir, 'DATA_Sheet.csv'))
        self.fluids = self.viscosity_data["Bottle number"]

        # Adjust length to account for sequence length
        self._len = max(0, len(self.image_dict) - (self.sequence_length - 1))

    def __len__(self):
        """
        Returns the number of sequences available, accounting for sequence_length.
        """
        return self._len

    def __getitem__(self, idx):
        """
        Returns a sequence of images and the label for the last image in the sequence.

        Args:
            idx (int): Index of the starting image in the sequence.

        Returns:
            tuple: (sequence, label)
                - sequence: Tensor of shape (sequence_length, channels, height, width)
                - label: Tensor with the viscosity label for the last image
        """
        if idx < 0 or idx >= self._len:
            raise IndexError(f"Index {idx} out of range for dataset with length {self._len}")

        # Load sequence of images
        sequence = []
        for i in range(idx, idx + self.sequence_length):
            if i not in self.image_dict:
                raise ValueError(f"Image index {i} not found in image_dict")
            img_path = self.image_dict[i]
            image = Image.open(img_path)
            image = self.transform(image)
            sequence.append(image)

        # Stack images into a single tensor: (sequence_length, channels, height, width)
        sequence = torch.stack(sequence, dim=0)

        # Get label for the last image in the sequence
        last_img_path = self.image_dict[idx + self.sequence_length - 1]
        _temp = last_img_path.split(os.sep)[-3]  # Third last directory name
        label = [ii for ii in self.fluids if ii in _temp]
        if not label:
            raise ValueError(f"No label found for image {last_img_path}")
        dfIndex = self.viscosity_data.index[self.viscosity_data["Bottle number"] == label[0]]
        label = self.viscosity_data.iloc[dfIndex]['Viscosity 25C']

        return sequence, torch.tensor(label.values, dtype=torch.float32)

    def dims(self,) -> tuple[tuple[int, int], tuple[int, int]]:

        min_width   = float( 'inf')
        min_height  = float( 'inf')
        max_width   = float('-inf')
        max_height  = float('-inf')

        for path in tqdm.tqdm(self.image_dict.values(), desc="Calculating dimensions"):
            try:
                with Image.open(path) as img:
                    width, height   = img.size
                    min_width       = min(min_width, width)
                    min_height      = min(min_height, height)
                    max_width       = max(max_width, width)
                    max_height      = max(max_height, height)
            except Exception as e:
                print(f"Error processing {path}: {e}")
        return (min_width, min_height), (max_width, max_height)



if __name__ == "__main__":
    # First run: scans and saves index
    dataset = loc_ImageDataset(
        data_dir="/media/d2u25/Dont/frames_Process_15_Patch",
        skip=4,
        load_from_file=False,
        use_yaml=False  # Change to True for YAML
    )

    # Subsequent run: loads from pickle or YAML instantly
    fast_dataset = loc_ImageDataset(
        data_dir="/media/d2u25/Dont/frames_Process_15_Patch",
        skip=4,
        load_from_file=True,
        use_yaml=False
    )

    print(f"Number of images: {len(fast_dataset)}")
    sample_image, path = fast_dataset[0]
    print(f"Sample image shape: {sample_image.shape}")
    print(f"Sample image path: {path}")
    print(f"Image dimensions (min, max): {fast_dataset.dims()}")

