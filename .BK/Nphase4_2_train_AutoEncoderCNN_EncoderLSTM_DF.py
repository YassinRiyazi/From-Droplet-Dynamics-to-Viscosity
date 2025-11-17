"""
    Author:         Yassin Riyazi
    Date:           04-08-2025
    Description:    Train an embedding-based LSTM for time series data.

    TODO:
        - 

"""
import  os
import  time
import  glob 
import  torch
import  networks
import  numpy               as      np
import  dataset             as      DSS
import  torch.nn            as      nn
import  torch.optim         as      optim
from    torch.utils.data    import  DataLoader
from    typing              import  Callable, Optional, Union

import utils

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


def train_lstm_model(CnnAutoEncoderEmbdSize = 256,
                     SEQUENCE_LENGTH = 20,
                     hidden_dim = utils.config['Training']['Constant_feature_LSTM']['Hidden_size'],
                     stride=4,
                     epochs = 5,
                     GPU_temperature =  utils.config['Training']['GPU_temperature'],
                     data_dir    = '/media/d2u25/Dont/frames_Process_15_Patch') -> None:
    """
    This function is a placeholder for training the LSTM model.
    It can be extended to include specific logic for training tasks.
    """
    batch_size  = 800
    
    dirs = []
    root_directory = "/media/d2u25/Dont/frame_Extracted_Vids_DFs"
    for tilt in sorted(glob.glob(os.path.join(root_directory, "*"))):
        for fluid in sorted(glob.glob(os.path.join(tilt, "*"))):
            for idx, repetition in enumerate(sorted(glob.glob(os.path.join(fluid, "*")))):
                if os.path.isfile(os.path.join(repetition, 'SR_result', 'result.csv')) and idx < 5:
                    dirs.append(repetition)

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
                                        seq_len=SEQUENCE_LENGTH,
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
            # print(f"Error processing {dir}: {e}")
            pass
    val_set = DSS.TimeSeriesDataset_dataframe(
                                        root_dirs = dirs,
                                        stride=stride,
                                        seq_len=SEQUENCE_LENGTH,
                                    )
    

    train_loader      = DataLoader(train_set, batch_size=64, num_workers=16, shuffle=True, pin_memory=True, drop_last=True)
    val_loader      = DataLoader(val_set, batch_size=batch_size, num_workers=16, shuffle=True, pin_memory=True, drop_last=True)

    
    model = networks.AutoEncoder_CNN_LSTM.Encoder_LSTM(
        address_autoencoder=None,
        input_dim=CnnAutoEncoderEmbdSize,  # Adjust based on your data
        hidden_dim=hidden_dim,  # Adjust based on your model architecture
        num_layers=2,  # Number of LSTM layers
        dropout=0.1,  # Dropout rate
        sequence_length=SEQUENCE_LENGTH,
    )

    # model = LSTMModel(input_dim, hidden_dim, num_layers, dropout)
    # Define the loss function and optimizer
    # optimizer = optim.SGD(model.parameters(), lr=1e-2,)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.MSELoss()

    lr_scheduler = None# optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    train(
        model = model,
        train_loader = train_loader,
        val_loader = val_loader,
        criterion = criterion,
        optimizer = optimizer,
        epochs = epochs,
        device = device,
        model_name = f"DFs_Encoder_{CnnAutoEncoderEmbdSize}_LSTM_HD{hidden_dim}_SL{SEQUENCE_LENGTH}",

        handler = handler_supervised,
        handler_postfix=save_reconstructions,

        ckpt_save_freq=3,
        ckpt_save_path=os.path.join(os.path.dirname(__file__), 'checkpoints'),
        ckpt_path=None,
        report_path=os.path.join(os.path.dirname(__file__), 'training_report.csv'),
        use_hard_negative_mining=False,

        lr_scheduler = lr_scheduler,
        GPU_temperature = GPU_temperature
    )


if __name__ == "__main__":
    ##### data_dir    = '/media/d2u25/Dont/frames_Process_15_Patch'
    CnnAutoEncoderEmbdSize = 8
    for SEQUENCE_LENGTH in [1, 10, 100]:
        for hidden_dim in ([256]):
            print(f"Training with CnnAutoEncoderEmbdSize={CnnAutoEncoderEmbdSize}, SEQUENCE_LENGTH={SEQUENCE_LENGTH}, hidden_dim={hidden_dim}")
            train_lstm_model(CnnAutoEncoderEmbdSize=CnnAutoEncoderEmbdSize,
                                SEQUENCE_LENGTH=SEQUENCE_LENGTH,
                                hidden_dim=hidden_dim,
                                epochs=30,
                                GPU_temperature=70,
                                data_dir    = '/media/d2u25/Dont/frames_Process_15_PVelocity')