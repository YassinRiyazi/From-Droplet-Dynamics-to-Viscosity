"""
    Author:         Yassin Riyazi
    Date:           04-08-2025
    Description:    Train an embedding-based LSTM for time series data.

    TODO:
        - 

"""
import  os
import  glob 
import  torch
import  networks
import  dataset             as      DSS
import  torch.nn            as      nn
import  torch.optim         as      optim
from    torch.utils.data    import  DataLoader
from    typing              import  Callable, Optional, Union, Tuple # type: ignore

import  sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../', 'src/PyThon/NeuralNetwork/trainer')))
from Base import train # type: ignore

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the random seed for reproducibility
torch.manual_seed(42) # type: ignore
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

def save_reconstructions(
                         model: nn.Module,
                         dataloader: torch.utils.data.DataLoader,
                         device: torch.device,
                         save_dir: str,
                         epoch: int,
                         num_samples: int = 64) -> None:
    """Save a batch of original and reconstructed images from the dataloader and save target/predicted values to a text file.
    Args:
        model (nn.Module): The trained autoencoder model
        dataloader (torch.utils.data.DataLoader): DataLoader for validation or test set
        device (torch.device): Device to run the model on
        save_dir (str): Directory to save the images and text file
        epoch (int): Current epoch number for naming
        num_samples (int): Number of samples to save from the batch
    Returns:
        None: Saves images and text file to the specified directory.
    """
    num_samples = max(num_samples, 64)  # Ensure at least one sample is saved
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for i, Args in enumerate(dataloader):
            Args = [arg.contiguous().to(device) for arg in Args]
            model.lstm.reset_states(Args[0])  # Reset LSTM states before processing a new batch
            output = model(Args[0])

            target = Args[1].view(-1)
            predicted = output.view(-1)

            total_samples = Args[1].size(0)
            rand_indices = torch.randperm(total_samples)[:num_samples]
            target = target[rand_indices]
            predicted = predicted[rand_indices]

            # Save target and predicted values to a text file
            text_file_path = os.path.join(save_dir, f"reconstructions_epoch_{epoch}_batch_{i}.txt")
            with open(text_file_path, 'w') as f:
                f.write("Sample Index\tTarget Value\tPredicted Value\n")
                for j in range(min(num_samples, len(target))):
                    f.write(f"{j}\t{target[j].item():.6f}\t{predicted[j].item():.6f}\n")

            if i >= 5:  # Limit to first 5 batches
                break  # Only process the 5 batch

def train_lstm_model(SEQUENCE_LENGTH:int,
                     skip: int,
                     _case:str,
                     epochs:int,
                     ImageSize:Tuple[int, int],
                     proj_dim:int,
                     LSTMEmbdSize:int,

                     lr:float = 1e-2,
                     hidden_dim:int = 256,
                     batch_size:int = 16,
                     GPU_temperature:int = 70,
                     Autoencoder_CNN: torch.nn.modules = None) -> None:
    
    if  _case == "default":
        data_dir    = '/media/d2u25/Dont/frames_Process_30'

    elif    _case == "NoRef":
        data_dir    = '/media/d2u25/Dont/frames_Process_30_LightSource'

    elif    _case == "Position":
        data_dir    = '/media/d2u25/Dont/frames_Process_30_Position'

    elif    _case == "Velocity":
        data_dir    = '/media/d2u25/Dont/frames_Process_30_Velocity_P540'

    elif    _case == "Velocity_wide":
        data_dir    = '/media/d2u25/Dont/frames_Process_30_Velocity_wide'
        
    elif    _case == "Position_wide":
        data_dir    = '/media/d2u25/Dont/frames_Process_30_Position_wide'

    else:
        raise NotImplementedError("case not implemented")
    
    if 'wide' in str.lower(_case):
        if ImageSize[1] != 1024 and ImageSize[0] != 256:
            raise ValueError("For wide cases, ImageSize width must be 1024")
        if LSTMEmbdSize != 16384:
            raise ValueError("For wide cases, LSTMEmbdSize must be 16384")
    
    model = networks.AutoEncoder_CNN_LSTM.Encoder_LSTM(
        address_autoencoder= glob.glob(f'/home/d2u25/Desktop/Main/Projects/Viscosity/P2NeuralNetwork/Nphase4_AutoEncoder/checkpoints/*{_case}*/*.pt')[0],
        proj_dim=proj_dim,  # Adjust based on your data
        LSTMEmbdSize=LSTMEmbdSize,
        hidden_dim=hidden_dim,  # Adjust based on your model architecture
        num_layers=2,  # Number of LSTM layers
        dropout=0.3,  # Dropout rate
        sequence_length=SEQUENCE_LENGTH,
        Autoencoder_CNN=Autoencoder_CNN,
    )
    
    dicAddressesTrain, dicAddressesValidation, dicAddressesTest = DSS.dicLoader(root = data_dir)
    del dicAddressesTest
    ###################################
    ###################################
    # fluid_Constraints: list[str]= []
    # tilt_Exclusion: list[str]   = []# ,'/285/','/290/','/295/','/300/'
    # Sorted_fluid = ['S3-SDS10_D+0.8797 mPa.s', 'S3-SDS01_D+0.8797 mPa.s', 'S3-Water_D+0.8797 mPa.s', 'S3-Water_nD+0.8797 mPa.s', 'S2-SNr2.1_D+2.5426 mPa.s', 'S2-SNr2.14_D+2.5426 mPa.s', 'S3-SNr3.01_D+2.5681 mPa.s', 'S3-50Per_D+5.7695 mPa.s', 'S3-SNr3.02_D+13.5901 mPa.s', 'S3-70Per_D+17.4963 mPa.s', 'S2-SNr2.5_D+19.9383 mPa.s', 'S3-SNr3.03_D+24.0283 mPa.s', 'S3-SNr2.6_D+24.0722 mPa.s', 'S3-SNr2.7_D+28.1570 mPa.s', 'S2-SNr2.9_D+36.3760 mPa.s', 'S3-80Per_D+38.1382 mPa.s', 'S3-SNr3.04_D+44.5447 mPa.s', 'S3-SNr3.05_D+54.9317 mPa.s', 'S3-SNr3.06_D+65.1178 mPa.s', 'S3-SNr3.07_D+75.2828 mPa.s', 'S3-SNr3.08_D+84.6743 mPa.s']
    # # 'S3-SDS99_D+0.8797 mPa.s',, 'S3-SNr3.12_D+124.1283 mPa.s', 'S3-90Per_D+142.4221 mPa.s'
    # for fluid in Sorted_fluid[::2]: #Sorted_fluid:
    #     fluid = fluid.split('+')[0]
    #     fluid_Constraints.append(fluid)


    # dicAddressesTrain           = DSS.DS_limiter(dicAddressesTrain,fluid_Constraints,tilt_Exclusion)
    # dicAddressesValidation      = DSS.DS_limiter_inv(dicAddressesValidation,fluid_Constraints,tilt_Exclusion)
    ###################################
    ###################################
    # Load dataset
    train_set = DSS.MotherFolderDataset(
                                        resize=ImageSize,
                                        dicAddresses = dicAddressesTrain,
                                        stride=skip,
                                        sequence_length=SEQUENCE_LENGTH,
                                        extension=".png"
                                    )

    val_set = DSS.MotherFolderDataset(
                                        resize=ImageSize,
                                        dicAddresses = dicAddressesValidation,
                                        stride=skip,
                                        sequence_length=SEQUENCE_LENGTH,
                                        extension=".png"
                                    )

    # Optimize DataLoader
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=16, shuffle=True, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, num_workers=16, shuffle=False, pin_memory=True)
  

    # model = LSTMModel(input_dim, hidden_dim, num_layers, dropout)
    # Define the loss function and optimizer
    # optimizer = optim.SGD(model.parameters(), lr=1e-2,)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=lr,
                                  weight_decay=1e-4
                                  )
    criterion = nn.MSELoss()
    
    if isinstance(optimizer, optim.AdamW):
        lr_scheduler = None
    else:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train(
        model = model,
        train_loader = train_loader,
        val_loader = val_loader,
        criterion = criterion,
        optimizer = optimizer,
        epochs = epochs,
        device = device,
        Plateaued = None,
        model_name = f"AE_CNN_LSTM_HD{hidden_dim}_SL{SEQUENCE_LENGTH}_Skip{skip}_{case=}",

        handler = handler_supervised,
        handler_postfix=save_reconstructions,

        ckpt_save_freq=30,
        ckpt_save_path=os.path.join(os.path.dirname(__file__), 'checkpoints'),
        ckpt_path=None,
        report_path=os.path.join(os.path.dirname(__file__), 'training_report.csv'),
        use_hard_negative_mining=False,

        lr_scheduler = lr_scheduler,
        GPU_temperature = GPU_temperature
    )


if __name__ == "__main__":
    Ref = True
    SEQUENCE_LENGTH = 10
    skip = 4
    ImageSize: Tuple[int, int] = (201,201)
    LSTMEmbdSize = 512
    proj_dim = LSTMEmbdSize
    Autoencoder_CNN = networks.AutoEncoder_CNNV1_0.Autoencoder_CNN
    
    
    
    for case in ['Velocity']:
        for hidden_dim in [256]:
            train_lstm_model(SEQUENCE_LENGTH=SEQUENCE_LENGTH,
                             hidden_dim=hidden_dim,
                             epochs=30,
                             skip=skip,
                             _case=case,
                             lr=0.001,
                             ImageSize=ImageSize,
                             LSTMEmbdSize=LSTMEmbdSize,
                             proj_dim=proj_dim,
                             Autoencoder_CNN=Autoencoder_CNN,)
