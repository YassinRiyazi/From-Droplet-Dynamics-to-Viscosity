"""
    Author: Yassin Riyazi
    Date: 11-08-2023
    Description: Plot the final results of the train and validation process after training finished.
    TODO:
        - make Y scale logarithmic
"""

import os
import numpy                as np
import pandas               as pd
import matplotlib.pyplot    as plt
from typing import Optional, List
from tqdm import tqdm
from numpy.typing import NDArray


def annotating(ax: plt.Axes,
               plot_desc: str,
               data_1: NDArray[np.float64],
               data_2: NDArray[np.float64],
               data_3: Optional[NDArray[np.float64]] = None) -> None:
    """
    Annotate the plot with the lowest point information for each dataset, avoiding overlap.

    Args:
        ax (plt.Axes): The axes to annotate.
        plot_desc (str): Description of the plot (e.g., "Loss", "Accuracy").
        data_1 (np.ndarray): Training data.
        data_2 (np.ndarray): Validation data.
        data_3 (Optional[np.ndarray]): Test data (if available).

    Returns:
        None: no return value
    """
    datasets: list[NDArray[np.float64]] = [data_1, data_2]
    dataset_names = ['Train', 'Validation']
    if data_3 is not None:
        datasets.append(data_3)
        dataset_names.append('Test')

    # dtaset sanitazation
    for i, d in enumerate(datasets):
        if not np.all(np.isfinite(d)):
            print(f"Dataset {dataset_names[i]} contains invalid values.")
            # drop invalid values for annotation
            valid_indices = np.isfinite(d)
            datasets[i] = d[valid_indices]

    # Filter out empty datasets for range calculation
    valid_datasets = [d for d in datasets if d.size > 0]
    
    if not valid_datasets:
        print("All datasets are empty or contain only invalid values. Skipping annotation.")
        return

    # Calculate data range for dynamic offset
    data_max = max(np.max(data) for data in valid_datasets)
    data_min = min(np.min(data) for data in valid_datasets)
    data_range = data_max - data_min if data_max != data_min else 1.0
    # offset_step = 0.1 * data_range  # Vertical offset for each annotation

    for idx, (data, name) in enumerate(zip(datasets, dataset_names)):
        if data.size == 0:
            continue
            
        min_value = np.min(data)
        min_epoch = np.argmin(data)

        # Calculate vertical position for annotation to avoid overlap
        # Stack annotations upwards starting from a base offset
        xytext_y = min_value + (0.1 + idx * 0.15) * data_range

        if not (np.isfinite(min_epoch) and np.isfinite(min_value)):
            continue
        if not (np.isfinite(xytext_y)):
            print(min_value, (0.1 + idx * 0.15) , data_range)
            continue
        # Add annotation for the minimum point
        ax.annotate(f'{name} min {plot_desc}: {min_value:.4E}\n(Epoch {min_epoch})',
                    xy=(min_epoch, min_value),
                    xytext=(min_epoch, xytext_y),
                    fontsize=10,
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.8))

def result_plot(model_name:str,
             save_address: str | os.PathLike,
             plot_desc:str,
             data_1:np.array,
             data_2:np.array,
             data_3:np.array = None,
             DPI:int=100,
             axis_label_size:int=15,
             x_grid:Optional[float]=0.1,
             y_grid:Optional[float]=0.5,
             axis_size:Optional[int]=12,
             y_lim:Optional[list[int,int]]=None,
             ShowPlot: bool = True) -> None:

    """
    In this function, by getting the two lists of data, the result will be plotted as the desired title

    Args:
        model_name (str): doesn't do anything in this version, but can be implemented to save an image file with the given name
        plot_desc (str): define the identity of the data, i.e., Accuracy, loss,...
        data_1 (np.array)     : array of the first data set .
        data_2 (np.array)     : array of the second data set .
        data_3 (np.array)     : array of third data set (optional)
        DPI (int)        :   define the quality of the plot
        axis_label_size (int)  :   define the label size of the axes
        x_grid (float)      :   x_grid capacity
        y_grid (float)      :   y_grid capacity
        axis_size (int)    :   axis number's size

    Returns:
        none:  by now

    See Also:
        - The size of the two lists must be the same 

    Notes:
        - The size of the two lists must be the same

    Example:
        >>> result_plot("perceptron", "loss", data_1, data_2)
    """
    # Set Times New Roman as the font for all text
    # plt.rcParams['font.family'] = 'Times New Roman'
    
    assert(len(data_1)==len(data_2))

    
    fig , ax = plt.subplots(1,figsize=(10,5) , dpi=DPI)

    if data_3 is not None:
        fig.suptitle(f"Train , validation and Test "+plot_desc,y=0.95 , fontsize=20)
    else:
        fig.suptitle(f"Train and validation "+plot_desc,y=0.95 , fontsize=20)

    styles = {"train": "-b", "val": "--r", "test": ":g"}

    epochs = range(len(data_1))
    ax.plot(epochs, data_1, styles["train"], linewidth=3, label=' train '+ plot_desc)
    ax.plot(epochs, data_2, styles["val"], linewidth=3, label=' validation '+plot_desc)
    if data_3:
        ax.plot(epochs, data_3, styles["test"], linewidth=3, label=' test '+plot_desc)

    ax.set_xlabel("epoch"       ,fontsize=axis_label_size)
    ax.set_ylabel(plot_desc     ,fontsize=axis_label_size)
    if x_grid:
        ax.grid(axis="x",alpha=0.1)
    if y_grid:
        ax.grid(axis="y",alpha=0.5)

    annotating(ax, plot_desc, data_1, data_2, data_3)

    ax.legend(loc=0,prop={"size":9})

    ax.tick_params(axis="x",labelsize=axis_size)
    ax.tick_params(axis="y",labelsize=axis_size)

    #spine are borde line of plot
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if y_lim:
        ax.set_ylim(y_lim)

    plt.grid(alpha=0.3)
    plt.yscale('log')
    plt.savefig(os.path.join(save_address, model_name+'.png'), bbox_inches='tight')
    if ShowPlot:
        plt.show()
    plt.close()

def ResultSavorMain(df_address:str,
                    save_dir:str,
                    lossPlot:bool=True,
                    AccuPlot:bool=False,
                    DPI:int=400,
                    ShowPlot: bool = True) -> None:
    """
    Save a batch of original and reconstructed images from the dataloader and save target/predicted values to a text file.
    
    Args:
        DFAddress (str): Path to the DataFrame containing model results.
        save_dir (str): Directory to save the images and text file.
        epoch (int): Current epoch number for naming.
        num_samples (int): Number of samples to save from the batch.
    
    Returns:
        None: Saves images and text file to the specified directory.
    """
    df = pd.read_csv(df_address) # type: ignore

    # Automatically determine the latest batch index for each mode
    latest_batches = df.groupby("mode")["batch_index"].max().to_dict() # type: ignore


    train       = df.query('mode == "train"').query(f'batch_index == {latest_batches["train"]}') # type: ignore

    validation  = df.query('mode == "val"').query(f'batch_index == {latest_batches["val"]}') # type: ignore

    try:
        test    = df.query('mode == "test"').query(f'batch_index == {latest_batches["test"]}') # type: ignore

    except KeyError:
        test    = None

    Model_name = os.path.basename(df_address).replace("_report.csv", "")

    if lossPlot:
        result_plot(Model_name+"_loss",
                    save_address=save_dir,
                    plot_desc="loss",
                    data_1 = np.array(train['avg_train_loss_till_current_batch']),
                    data_2 = np.array(validation['avg_val_loss_till_current_batch']),
                    data_3 = None if test is None else np.array(test['avg_val_loss_till_current_batch']),
                    DPI=DPI,
                    ShowPlot=ShowPlot)
    if AccuPlot:
        raise NotImplementedError("Accuracy plot is not implemented yet, check out the dataframe structure and fill it accordingly.")

def find_csv_files(directory: str) -> None:
    """
    Find all .csv files in the specified directory and its subdirectories.

    Args:
        directory (str): Path to the directory to search.

    Returns:
        None: This function does not return a value.

    Example:
        >>> find_csv_files("/home/d2u25/Desktop/Main/Projects/Viscosity/P2NeuralNetwork/Nphase4_AutoEncoder/checkpoints")
    """
    # csv_files = []
    
    # Walk through directory and subdirectories
    for root, _, files in tqdm(os.walk(directory)):
        # Filter for .csv files and add their full paths
        for file in (files):
            if file.endswith('_report.csv'):
                df_address = os.path.join(root, file)
                # csv_files.append(df_address)
    
                ResultSavorMain(df_address =df_address,
                                save_dir=root,
                                lossPlot=True,
                                AccuPlot=False,
                                DPI=400,
                                ShowPlot=False)
    # return csv_files
    return None

if __name__ == "__main__":
    # df_address = 'Projects/Viscosity/P2NeuralNetwork/Nphase4_AutoEncoder/checkpoints/Encoder_8192_LSTM_HD256_SL1_20250814_021059/Encoder_8192_LSTM_HD256_SL1_report.csv'
    # save_dir = os.path.join(*df_address.split(os.path.sep)[:-1])
    

    # ResultSavorMain(df_address =df_address,
    #                 save_dir=save_dir,
    #                 lossPlot=True,
    #                 AccuPlot=False,
    #                 DPI=400,
    #                 ShowPlot=True)
    # find_csv_files("/home/d2u25/Desktop/Main/Projects/Viscosity/P2NeuralNetwork/Nphase4_AutoEncoder/checkpoints")
    # find_csv_files("/media/roboprocessing/Data/checkpoints")
    find_csv_files("/home/roboprocessing/Desktop/From-Droplet-Dynamics-to-Viscosity/Output/checkpoints")