"""
    Author: Yassin Riyazi
    Date: 06-09-2025

    Description:
        This script computes the Smoothness Index (SI) for a dataset using a pre-trained AutoEncoder and LSTM model.
        It loads the dataset, processes it through the model, and calculates the SI for both training and validation sets.
        The results are printed out at the end.

    TODO:
        - Save results to a file for further analysis with different model configurations.
        - Save displayed command line for reproducibility.

    default i % 100 != 0 For bare dataset the Mean SI: 0.681539536877112
    default i % 100 != 0 With AutoEncoder the Mean SI: 0.6898800047961149        
"""

# %%
import  re
import  os
import  tqdm
import  pickle
import  glob
import  torch
import  networks
import  numpy               as      np
import  dataset             as      DSS
import  pandas              as      pd
import  seaborn             as      sns
import  matplotlib.pyplot   as      plt
from    time                import  strftime
from    Utils.SmoothnessIndex     import  Kalhor_SmoothnessIndex
from    torch.utils.data    import  DataLoader
from    typing              import  TypeAlias

DsL: TypeAlias = DataLoader[tuple[torch.Tensor, torch.Tensor]]

def generateSample():
    from    sklearn.model_selection import train_test_split # type: ignore
    Batch = 1000
    x = torch.rand(size=[Batch,9000], dtype=torch.float32)  # generates floats

    # generating random gaussian float between 0 and 1
    y = torch.rand(size=[Batch,1], dtype=torch.float32).view(-1,1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)  # type: ignore
    x_train: torch.Tensor
    x_test: torch.Tensor
    y_train: torch.Tensor
    y_test: torch.Tensor

    # Compute Smoothness Index
    SI = Kalhor_SmoothnessIndex(x_train.cuda(), y_train.cuda())

    print("Linear SmI:", SI.smi_linear())   # one variant
    print("Nonlinear SmI:", SI.cross_smi_linear(x_test.cuda(), y_test.cuda()))  # another variant

def DfLoader(sequence_length: int,
             stride: int,
             batch_size: int) -> DsL:
    dicAddressesTrain, dicAddressesValidation, dicAddressesTest = DSS.dicLoader(rootAddress="Projects/Viscosity/",
                                                                                root = '/media/d2u25/Dont/frame_Extracted_Vids_DFs/')
    del dicAddressesTest
    
    _dirs = []
    for fluid, addresses in dicAddressesTrain.items():
        _dirs.extend(addresses)
    dirs = []
    for dir in _dirs:
        vv = os.path.split(dir)
        try:
            path = glob.glob(f"{vv[0]}/*{vv[1][1:4]}*{vv[1][5:7]}")[0]
            viscosity = float(vv[1].split("_")[-1])
            if os.path.exists(path):
                if os.path.isfile(os.path.join(path,'SR_result','result.csv')):
                    dirs.append((path, viscosity))
        except Exception as e:
            # print(f"Error processing {dir}: {e}")
            pass
    # dirs = dirs[::4]
    # Load dataset
    train_set = DSS.TimeSeriesDataset_dataframe(
                                        root_dirs = dirs,
                                        stride=stride,
                                        seq_len=sequence_length,
                                    )
    _dirs = []
    for fluid, addresses in dicAddressesTrain.items():
        _dirs.extend(addresses)
    dirs = []
    for dir in _dirs:
        vv = os.path.split(dir)
        try:
            path = glob.glob(f"{vv[0]}/*{vv[1][1:4]}*{vv[1][5:7]}")[0]
            viscosity = float(vv[1].split("_")[-1])
            if os.path.exists(path):
                if os.path.isfile(os.path.join(path,'SR_result','result.csv')):
                    dirs.append((path, viscosity))
        except Exception as e:
            print(f"Error processing {dir}: {e}")
            pass
    dirs = dirs[::4]
    val_set = DSS.TimeSeriesDataset_dataframe(
                                        root_dirs = dirs,
                                        stride=stride,
                                        seq_len=sequence_length,
                                    )
    

    dataloader      = DataLoader(train_set, batch_size=64, num_workers=16, shuffle=True, pin_memory=True, drop_last=True)
    val_loader      = DataLoader(val_set, batch_size=batch_size, num_workers=16, shuffle=True, pin_memory=True, drop_last=True)
    return dataloader, val_loader

def loadModelDatasets(
        UseModels: bool,
        hidden_dim: int,
        CnnAutoEncoderEmbdSize: int,
        sequence_length: int,
        batch_size: int,
        stride: int,
        _case: str
    ) -> tuple[networks.AutoEncoder_CNN_LSTM.Encoder_LSTM,DsL,DsL]:

    reduced = False
    extension = ".png"
     # Data directory based on case
    if _case == 'default':
        data_dir            = "/media/d2u25/Dont/frames_Process_30"
    elif _case == 'Position':
        data_dir            = "/media/d2u25/Dont/frames_Process_30_Position"
    elif _case == 'Velocity':
        data_dir            = "/media/d2u25/Dont/frames_Process_30_Velocity"
    elif _case == 'NoRef':
        data_dir            = "/media/d2u25/Dont/frames_Process_30_LightSource"
    elif _case == 'DropCoordinate':
        reduced             = True
        data_dir            = "/media/d2u25/Dont/frames_Process_30_PINN"
        extension           = ".pkl"
    elif _case == 'Dataframe':
        data_dir            = "/media/d2u25/Dont/frame_Extracted_Vids_DFs"
        extension           = ".csv"
        return None, *DfLoader(sequence_length, stride, batch_size)
    else:
        raise ValueError(f"Unknown case: {_case}")

    
    dicAddressesTrain, dicAddressesValidation, dicAddressesTest = DSS.dicLoader(rootAddress="Projects/Viscosity/", root = data_dir)
    del dicAddressesTest
    if reduced:
        for name, item in dicAddressesTrain.items():
            dicAddressesTrain[name] = item[::4]

        for name, item in dicAddressesValidation.items():
            dicAddressesValidation[name] = item[::4]
    # Load dataset
    train_set = DSS.MotherFolderDataset(
                                        dicAddresses = dicAddressesTrain,
                                        stride=stride,
                                        sequence_length=sequence_length,
                                        extension=extension
                                    )

    val_set = DSS.MotherFolderDataset(
                                        dicAddresses = dicAddressesValidation,
                                        stride=stride,
                                        sequence_length=sequence_length,
                                        extension=extension
                                    )

    train_loader    = DataLoader(train_set, batch_size=batch_size, num_workers=16, shuffle=True, pin_memory=True)
    val_loader      = DataLoader(val_set, batch_size=batch_size, num_workers=16, shuffle=True, pin_memory=True, drop_last=True)

    if UseModels:
        model = networks.AutoEncoder_CNN_LSTM.Encoder_LSTM(
                address_autoencoder= glob.glob(f'/home/d2u25/Desktop/Main/Projects/Viscosity/P2NeuralNetwork/Nphase4_AutoEncoder/checkpoints/CNN_AE_{CnnAutoEncoderEmbdSize}_{_case}_*/*.pt')[0],
                input_dim=CnnAutoEncoderEmbdSize,  # Adjust based on your data
                hidden_dim=hidden_dim,  # Adjust based on your model architecture
                num_layers=2,  # Number of LSTM layers
                dropout=0.1,  # Dropout rate
                sequence_length=sequence_length,
            )
        model.eval()
        model.cuda()
        return model, train_loader, val_loader
    else:
        return None, train_loader, val_loader

def parse_results(filename: str) -> pd.DataFrame:
    """
    Parse experiment results from a log-like text file.
    Supports both formats:
      - 'Processing case: XYZ with sequence length: N'
      - 'Parameters: ... sequence_length=N ...' followed by 'Processing case: XYZ'

    Args:
        filename (str): Path to the text file.

    Returns:
        pd.DataFrame: DataFrame with columns [Case, SequenceLength, MeanSI].
    """
    data = []
    with open(filename, "r") as f:
        lines = f.readlines()

    case, seq_len = None, None
    for line in lines:
        # Format 1: "Processing case: XYZ with sequence length: N"
        match_case_seq = re.search(r"Processing case:\s*(\w+)\s*with sequence length:\s*(\d+)", line)
        if match_case_seq:
            case = match_case_seq.group(1)
            seq_len = int(match_case_seq.group(2))
            continue

        # Format 2: "Parameters: ... sequence_length=N ..."
        match_param = re.search(r"sequence_length=(\d+)", line)
        if match_param:
            seq_len = int(match_param.group(1))
            continue

        # Case without explicit sequence length
        match_case = re.search(r"Processing case:\s*(\w+)", line)
        if match_case:
            case = match_case.group(1)
            continue

        # Mean SI
        match_si = re.search(r"Mean SI:\s*([\d.]+)", line)
        if match_si and case is not None and seq_len is not None:
            si = float(match_si.group(1))
            data.append([case, seq_len, si])
            case, seq_len = None, None  # reset for next entry

    return pd.DataFrame(data, columns=["Case", "SequenceLength", "MeanSI"])


def plot_heatmap(df: pd.DataFrame,name: str = "Mean_SI_Heatmap.png"):
    """
    Plot a heatmap of MeanSI by case and sequence length.

    Args:
        df (pd.DataFrame): DataFrame with [Case, SequenceLength, MeanSI].
    """
    pivot = df.pivot(index="Case", columns="SequenceLength", values="MeanSI")

    # Sort rows (cases) by mean MeanSI
    row_order = pivot.mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[row_order]

    plt.figure(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", vmin=df["MeanSI"].min(), vmax=df["MeanSI"].max())
    plt.title("Mean SmI Heatmap")
    plt.ylabel("Case")
    plt.xlabel("Sequence Length")
    plt.tight_layout()
    plt.savefig(name, dpi=300)
    plt.show()

#%%
if __name__ == "__main__":
    import sys

    baseDir = os.path.dirname(os.path.abspath(__file__))
    current_fileName = os.path.split(__file__)[1].split('.py')[0]
    sys.stdout = open(f"{baseDir}/log/{current_fileName}_{strftime('%Y%m%d_%H%M%S')}.txt", "w")

    UseModels               = False
    stride                  = 4
    batch_size              = 64
    hidden_dim              = 256
    CnnAutoEncoderEmbdSize  = 1024
    _cases                  = ['Dataframe']#['default', 'Position', 'Velocity', 'NoRef']
    skipIndex               = 100

    for _case in _cases:
        for sequence_length in [1, 10, 100]:
            print(f"Parameters: stride={stride}, sequence_length={sequence_length}, batch_size={batch_size}, hidden_dim={hidden_dim}, CnnAutoEncoderEmbdSize={CnnAutoEncoderEmbdSize}, skipIndex={skipIndex}")
            model, train_loader, val_loader = loadModelDatasets(UseModels, hidden_dim, CnnAutoEncoderEmbdSize, sequence_length, batch_size, stride, _case)
            listSIDATA  = []
            datas = {'UseModels':UseModels,'stride': stride, 'sequence_length': sequence_length, 'batch_size': batch_size,                                        # type: ignore
                    'hidden_dim': hidden_dim, 'CnnAutoEncoderEmbdSize': CnnAutoEncoderEmbdSize, '_case': _case, 'skipIndex': skipIndex}     # type: ignore
            
            print(f"Processing case: {_case}")
            print(f"Parameters: stride={stride}, sequence_length={sequence_length}, batch_size={batch_size}, hidden_dim={hidden_dim}, CnnAutoEncoderEmbdSize={CnnAutoEncoderEmbdSize}, skipIndex={skipIndex}")
            result = {}
            for i, (data, target) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
                if i % skipIndex != 0:
                    continue
                bb = data.size(0)
                data = data.cuda()
                target = target.cuda()
                if model:
                    with torch.no_grad():
                        data = model._encoder(data)
                target = target.view(bb, 1)
                data = data.view(bb, -1)
                SI = Kalhor_SmoothnessIndex(data, target)

                for i, (data, target) in enumerate(val_loader):
                    if i % skipIndex != 0:
                        continue
                    bb = data.size(0)
                    data = data.cuda()
                    target = target.cuda()
                    if model:
                        with torch.no_grad():
                            data = model._encoder(data)

                    x_test = data.view(bb, -1)
                    y_test = target.view(bb, 1)
                    listSIDATA.append(SI.cross_smi_linear(x_test,y_test).cpu().item())

            SIData = np.array(listSIDATA)
            print("Mean SI:", SIData.mean())
            result[_case] = [SIData.mean(), SIData.std(), datas, SIData]


            with open(f"{baseDir}/log/Data_SI_{_case}_{sequence_length}_{SIData.mean()}_{strftime('%Y%m%d_%H%M%S')}.pkl", "wb") as f:
                pickle.dump(result, f)



    df = parse_results("/home/d2u25/Desktop/Main/Projects/Viscosity/P2NeuralNetwork/Nphase4_AutoEncoder/log/SIData_20250908_112200.txt")
    plot_heatmap(df, name=f"{baseDir}/log/Mean_SI_BareData.png")
    
    
    df = parse_results("/home/d2u25/Desktop/Main/Projects/Viscosity/P2NeuralNetwork/Nphase4_AutoEncoder/log/SIData_20250908_144016.txt")
    plot_heatmap(df, name=f"{baseDir}/log/Mean_SI_Compressed.png")

