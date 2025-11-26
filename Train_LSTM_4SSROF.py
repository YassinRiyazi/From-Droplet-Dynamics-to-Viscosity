"""
    Author:         Yassin Riyazi
    Date:           04-08-2025
    Description:    Train an embedding-based LSTM for time series data.

    Changelog:
        - 04-08-2025: Initial version. 
        - 17-11-2025: Update to new be adaptible with the congig file structure.
                        Added dataset in utils to load datasets.
        

    TODO:
        - imtegrate the df to the LSTM training pipeline.

"""
import  os
import  glob
import sys 
import  torch
import  networks
# import  dataset             as      DSS
import  torch.nn            as      nn
import  torch.optim         as      optim
from    torch.utils.data    import  DataLoader
from    typing              import  Callable, Optional, Union, Tuple # type: ignore
import  utils

import deeplearning

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import  torch
utils.set_randomness(42)

# Set float32 matrix multiplication precision to medium
# torch.set_float32_matmul_precision('high')

def handler_supervised(Args:tuple[torch.Tensor, torch.Tensor],
                       criterion: nn.Module,
                       model: nn.Module,
                       **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """
    This function is a placeholder for handling supervised training.
    It can be extended to include specific logic for supervised learning tasks.
    """
    Args[1] = Args[1].contiguous().to(device)
    Args[2] = Args[2].contiguous().to(device)

    model.lstm.reset_states(Args[2])  # Reset LSTM states before processing a new batch
    output = model(Args[2])
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
    return 0 
    num_samples = max(num_samples, 64)  # Ensure at least one sample is saved
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for i, Args in enumerate(dataloader):
            Args = [arg.contiguous().to(device) for arg in Args]
            model.lstm.reset_states(Args[2])  # Reset LSTM states before processing a new batch
            output = model(Args[2])

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

            if i >= 30:  # Limit to first 30     batches
                break  # Only process the 5 batch

def train_lstm_model(config: str ,
                     proj_dim:int|None = None,
                     LSTMEmbdSize:int|None = None,

                     skip: int = utils.config['Training']['Constant_feature_LSTM']['Stride'],
                     SEQUENCE_LENGTH:int = utils.config['Training']['Constant_feature_LSTM']['window_Lenght'],
                     hidden_dim:int = 256,
                     Autoencoder_CNN: torch.nn.Module | None = None) -> None:
    utils.config['Training']['Constant_feature_LSTM']['learning_rate'] = 1e-3


    _Ds = utils.data_set(_data_config=config)
    _Ds.load_addresses()
    train_set, val_set = _Ds.load_datasets(
                                           stride=utils.config['Training']['Constant_feature_LSTM']['Stride'],
                                           sequence_length=utils.config['Training']['Constant_feature_LSTM']['window_Lenght'],)
    
    for ds in train_set.DaughterSets.values():
        ds.S4ORF_only = True
    for ds in val_set.DaughterSets.values():
        ds.S4ORF_only = True
    
    proj_dim = train_set[0][2].shape[1]
    LSTMEmbdSize = proj_dim
    utils.config['Dataset']['embedding']['positional_encoding'] = 'False'
    _case = utils.config['Dataset']['embedding']['positional_encoding']
    case = _case

    print("=" * 60)
    print(proj_dim)
    print("=" * 60)
    
    print(f"Training LSTM model with embedding case: {_case}, hidden_dim: {hidden_dim}, sequence_length: {SEQUENCE_LENGTH}, stride: {skip}")

    _case   = utils.config['Dataset']['embedding']['positional_encoding']
    Ref     = utils.config['Dataset']['reflection_removal']
    ID      = f"{utils.config['Dataset']['embedding']['positional_encoding']}_s{utils.config['Training']['Constant_feature_LSTM']['Stride']}_w{utils.config['Training']['Constant_feature_LSTM']['window_Lenght']}"
    model_name_AE = f"CNN_AE_{utils.config['Training']['Constant_feature_AE']['AutoEncoder_layers']}_{utils.config['Training']['Constant_feature_AE']['Architecture']}_{_case}_{proj_dim}_{Ref=}_{ID}"
    
    model_name_AE = model_name_AE.replace('_Ref','_self.Ref')

    if Autoencoder_CNN is not None:
        model_addresse = sorted(glob.glob(f'Output/checkpoints/AE_CNN/*{model_name_AE}*/*.pt'))
        if len(model_addresse) == 1 :
            AE_Address = model_addresse[0]
        else:
            raise ValueError(f"Expected exactly one checkpoint for case '{model_name_AE}', but found {len(model_addresse)}.")
    else:
        AE_Address = None
    
    model = networks.AutoEncoder_CNN_LSTM.Encoder_LSTM(
        address_autoencoder = AE_Address,
        proj_dim            = proj_dim,  # Adjust based on your data
        LSTMEmbdSize        = LSTMEmbdSize,
        hidden_dim          = 128,# hidden_dim,  # Adjust based on your model architecture
        num_layers          = 1, #utils.config['Training']['Constant_feature_LSTM']['Num_layers'],  # Number of LSTM layers
        dropout             = 0, #utils.config['Training']['Constant_feature_LSTM']['DropOut'],  # Dropout rate
        Autoencoder_CNN     = None if Autoencoder_CNN is None else Autoencoder_CNN,
        # use_attention=True  # NEW PARAMETER
    )


    # Optimize DataLoader
    batch_size = utils.config['Training']['batch_size']
    if sys.platform == 'linux':
        num_workers = utils.config['Training']['num_workers']
    elif sys.platform == 'win32':
        num_workers = 1
        
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers,
                               shuffle=True, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers,
                               shuffle=True, pin_memory=True)
  

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr            = float(utils.config['Training']['Constant_feature_LSTM']['learning_rate']),
                                  weight_decay  = float(utils.config['Training']['weight_decay'])
                                  )
    criterion = nn.MSELoss()
    
    if isinstance(optimizer, optim.AdamW): #type: ignore
        lr_scheduler = None
    else:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    deeplearning.train(
        model = model,
        train_loader = train_loader,
        val_loader = val_loader,

        criterion = criterion,
        optimizer = optimizer,
        device = device,

        Plateaued = None,
        model_name = f"AE_CNN_LSTM_HD{hidden_dim}_SL{SEQUENCE_LENGTH}_s{skip}_w{utils.config['Training']['Constant_feature_LSTM']['window_Lenght']}_{case=}",

        handler = handler_supervised,
        handler_postfix=save_reconstructions,

        ckpt_save_path=os.path.join(os.path.dirname(__file__),'Output', 'checkpoints','LSTM'),
        ckpt_path=None,
        report_path=os.path.join(os.path.dirname(__file__),'Output','LSTM', 'training_report.csv'),

        lr_scheduler = lr_scheduler,
        epochs                  = utils.config['Training']['num_epochs'],
        ckpt_save_freq          = utils.config['Training']['checkpoint_save_freq'],
        use_hard_negative_mining= utils.config['Training']['hard_negative_mining'],
        GPU_temperature         = utils.config['Training']['GPU_temperature'],
    )


if __name__ == "__main__":

    proj_dim = None
    LSTMEmbdSize = proj_dim
    
    config = 'Configs/narmal+tilt.yaml'

    train_lstm_model(
                    hidden_dim=utils.config['Training']['Constant_feature_LSTM']['Hidden_size'],
                    LSTMEmbdSize=LSTMEmbdSize,
                    proj_dim=proj_dim,
                    config = config
                    )
