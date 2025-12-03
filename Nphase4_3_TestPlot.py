"""
    Author:         Yassin Riyazi
    Date:           04-08-2025
    Description:    Train an embedding-based LSTM for time series data.

    Learned:
        - Python uses pickle to send data between multiprocessing workers. And it can not handle namedtuple.
        Still namedtuple help me in writing the code logic but then I had to removed it before sending to workers.
    
    TODO:
        - 

"""
#%%
import  os
os.chdir("/home/d2u25/Desktop/Main/Projects/Viscosity/P2NeuralNetwork/Nphase4_AutoEncoder")
import  glob 
import  torch
import  networks
import  pickle
import  tqdm
import  numpy               as      np
import  dataset             as      DSS
import  torch.nn            as      nn
from    torch.utils.data    import  DataLoader
from    typing              import  Callable, Optional, Union, Dict, Dict, List, TypeAlias, NamedTuple # type: ignore


# Define the type alias
LossReps: TypeAlias                 = Dict[str, List[float]|float]
Tilt_LossReps: TypeAlias            = Dict[int, LossReps]
Fluid_Tilt_LossReps: TypeAlias      = Dict[str, Tilt_LossReps]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Set float32 matrix multiplication precision to medium
torch.set_float32_matmul_precision('high')

def handler_supervised(Args:tuple[torch.Tensor, torch.Tensor],
                       criterion: nn.Module,
                       model: nn.Module):
    """
    This function is a placeholder for handling supervised training.
    It can be extended to include specific logic for supervised learning tasks.
    """
    Args = [arg.contiguous().to(device) for arg in Args]
    model.lstm.reset_states(Args[0])  # Reset LSTM states before processing a new batch
    output = model(Args[0])
    loss = criterion(output, Args[1].view(-1))
    return output, loss

def loadModel(CnnAutoEncoderEmbdSize:int, HiddenSizeLSTM:int, SEQUENCE_LENGTH:int, Skip:int,
              _case:str) -> nn.Module:
    baseDir = "/home/d2u25/Desktop/Main/Projects/Viscosity/P2NeuralNetwork/Nphase4_AutoEncoder/checkpoints"
               
    model = networks.AutoEncoder_CNN_LSTM.Encoder_LSTM(
        address_autoencoder= glob.glob(f'{baseDir}/CNN_AE_{CnnAutoEncoderEmbdSize}_{_case}_*/*.pt')[0],
        input_dim=CnnAutoEncoderEmbdSize,  # Adjust based on your data
        hidden_dim=HiddenSizeLSTM,  # Adjust based on your model architecture
        num_layers=2,  # Number of LSTM layers
        dropout=0.1,  # Dropout rate
        sequence_length=SEQUENCE_LENGTH,
    )
    # Load weights
    state_dict = torch.load(glob.glob(f"{baseDir}/AE_CNN_{CnnAutoEncoderEmbdSize}_LSTM_HD{HiddenSizeLSTM}_SL{SEQUENCE_LENGTH}_Skip{Skip}_case='{_case}'*/*.pt")[0], map_location=device)  # or "cuda"
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()  # Set the model to evaluation mode

    return model


import  multiprocessing as      mp
from    functools       import  partial
from    collections     import  namedtuple
_D_TestLoopIterate = namedtuple('LocalDSs', ['fluid', 'tilt', 'RepName', 'loader'])

def _build_localds(rep:str, SEQUENCE_LENGTH:int, skip:int, drop_last:bool= True) -> tuple[str, int, str, DataLoader]:
    """Worker function for parallel dataset building."""
    _temp   = rep.split(os.sep)
    tilt    = int(_temp[-3])
    fluid   = _temp[-2]
    RepName = _temp[-1]

    LocalDS = DSS.DaughterFolderDataset([rep], seq_len=SEQUENCE_LENGTH, stride=skip)
    loader  = DataLoader(LocalDS,
                         batch_size=1,
                         num_workers=22,   # still use dataloader workers
                         pin_memory=True,
                         shuffle=False,
                         drop_last=drop_last)
    return (fluid, tilt, RepName, loader)

def BaseGen(dicAddressesTest:DSS.DaughterFolderDataset.StringListDict,
            SEQUENCE_LENGTH:int,
            skip:int,
            num_workers:int|None=None,
            drop_last:bool=False) -> list[_D_TestLoopIterate]:
    """
    Parallelized version of BaseGen using multiprocessing and tqdm.
    """
    reps = []
    for fluid in dicAddressesTest.keys():
        for Rep in dicAddressesTest[fluid]:
            reps.append(Rep)

    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 2)

    print(f"Building datasets using {num_workers} workers...")
    with mp.Pool(processes=num_workers) as pool:
        LocalDSs = list(
            tqdm.tqdm(
                pool.imap(partial(_build_localds, SEQUENCE_LENGTH=SEQUENCE_LENGTH, skip=skip, drop_last=drop_last), reps),
                total=len(reps),
                desc="Building datasets"
            )
        )
    return LocalDSs

def TestLoopIterateUnordered(LocalDSs:tuple[str, int, str, DataLoader],
                    ErrorsOverRepetition:Fluid_Tilt_LossReps,
                    skip:int,
                    SEQUENCE_LENGTH:int,
                    CnnAutoEncoderEmbdSize:int,
                    criterion:nn.Module,
                    model:nn.Module,
                    _case='default') -> None:
    for fluid, tilt, RepName, loader in tqdm.tqdm(LocalDSs):
        ErrorsOverRepetition[fluid][tilt][RepName] = []
        dim = 0
        for batch in loader:
            with torch.no_grad():
                dim += batch[0].shape[0]
                output, loss = handler_supervised(batch, criterion, model)
            ErrorsOverRepetition[fluid][tilt][RepName].append(loss.item()*output.shape[0])

        ErrorsOverRepetition[fluid][tilt][RepName] = np.array(ErrorsOverRepetition[fluid][tilt][RepName]).sum()/dim

    baseDir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(baseDir, f"ErrORep_Case{_case}_skip{skip}_SL{SEQUENCE_LENGTH}_CnnAEEmbdSize{CnnAutoEncoderEmbdSize}.pkl"), "wb") as f:
        pickle.dump(ErrorsOverRepetition, f)

    return  None

def AssembeledUnordered(dicAddressesTest:DSS.DaughterFolderDataset.StringListDict,
               SEQUENCE_LENGTH:int,
               skip:int,
               ErrorsOverRepetition:Fluid_Tilt_LossReps,
               CnnAutoEncoderEmbdSize:int,
               criterion:nn.Module,
               model:nn.Module,
               _case='default',
               drop_last:bool=False) -> None:

    LocalDSs = BaseGen(dicAddressesTest,
             SEQUENCE_LENGTH,
             skip,
             drop_last=drop_last)

    TestLoopIterateUnordered(LocalDSs, ErrorsOverRepetition,
                    skip=skip,
                    SEQUENCE_LENGTH=SEQUENCE_LENGTH,
                    CnnAutoEncoderEmbdSize=CnnAutoEncoderEmbdSize,
                    criterion=criterion,
                    model=model,
                    _case=_case)
    
def TestLoopIterateOrdered(LocalDSs:tuple[str, int, str, DataLoader],
                    ErrorsOverRepetition:Fluid_Tilt_LossReps,
                    skip:int,
                    SEQUENCE_LENGTH:int,
                    CnnAutoEncoderEmbdSize:int,
                    criterion:nn.Module,
                    model:nn.Module,
                    _case='default') -> None:
    for fluid, tilt, RepName, loader in tqdm.tqdm(LocalDSs):
        ErrorsOverRepetition[fluid][tilt][RepName] = []
        for batch in loader:
            with torch.no_grad():
                output, loss = handler_supervised(batch, criterion, model)
            ErrorsOverRepetition[fluid][tilt][RepName].append(loss.item())

    baseDir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(baseDir, f"ErrORep_Case{_case}_skip{skip}_SL{SEQUENCE_LENGTH}_CnnAEEmbdSize{CnnAutoEncoderEmbdSize}_AllCases.pkl"), "wb") as f:
        pickle.dump(ErrorsOverRepetition, f)

    return  None

def AssembeledOrdered(dicAddressesTest:DSS.DaughterFolderDataset.StringListDict,
               SEQUENCE_LENGTH:int,
               skip:int,
               ErrorsOverRepetition:Fluid_Tilt_LossReps,
               CnnAutoEncoderEmbdSize:int,
               criterion:nn.Module,
               model:nn.Module,
               _case='default',
               drop_last:bool=False) -> None:

    LocalDSs = BaseGen(dicAddressesTest,
             SEQUENCE_LENGTH,
             skip,
             drop_last=drop_last,
             )

    TestLoopIterateOrdered(LocalDSs, ErrorsOverRepetition,
                    skip=skip,
                    SEQUENCE_LENGTH=SEQUENCE_LENGTH,
                    CnnAutoEncoderEmbdSize=CnnAutoEncoderEmbdSize,
                    criterion=criterion,
                    model=model,
                    _case=_case)

if __name__ == "__main__":

    _case                  = 'default'
    skip                   = 4
    SEQUENCE_LENGTH        = 1
    HiddenSizeLSTM         = 256
    CnnAutoEncoderEmbdSize = 1024

    criterion = nn.MSELoss()
    model = loadModel(CnnAutoEncoderEmbdSize,
                      HiddenSizeLSTM,
                      SEQUENCE_LENGTH,
                      Skip=skip,
                      _case=_case,)
    
    if _case == 'Velocity':
        root = '/media/roboprocessing/Data/frames_Process_30_Velocity'
    elif _case == 'default':
        root = '/media/roboprocessing/Data/frames_Process_30'

    dicAddressesTrain, dicAddressesValidation, dicAddressesTest = DSS.dicLoader(root = root)
    del dicAddressesTrain, dicAddressesValidation

    #%%
    ErrorsOverRepetition: Fluid_Tilt_LossReps = {}
    for fluid in dicAddressesTest.keys():
        ErrorsOverRepetition[fluid] = {}
        for tilt in range(280,340+5,5):
            ErrorsOverRepetition[fluid][tilt] = {}

    
    AssembeledOrdered(dicAddressesTest,
               SEQUENCE_LENGTH,
               skip,
               ErrorsOverRepetition,
               CnnAutoEncoderEmbdSize,
               criterion,
               model,
               _case=_case)
    # ErrorsOverRepetition = pickle.load(open(f"ErrORep_Case{_case}_skip{skip}_SL{SEQUENCE_LENGTH}_CnnAEEmbdSize{CnnAutoEncoderEmbdSize}.pkl", "rb"))


