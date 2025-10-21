"""
    Author: Yassin Riyazi
    Date: 04-08-2025
    Description: Train a CNN-based autoencoder for image data.

    Tested:
        - L1 Loss doesn't result in any thing good.
        - Normalizing images transforms.Normalize((0.5,), (0.5,)), doesn't help.
    
    Note:
        In previous successful tests: the sigmoid activation at end of Decoder was off.
        GAP is not working well with AutoEncoder, because it removes spatial information.
    

"""
import  os
import  glob
import  torch
import  networks
import  torch.nn            as      nn
import  dataset             as      DSS
from    torch.utils.data    import  DataLoader
from    torch.optim         import  Adam, lr_scheduler, AdamW
from    torchvision.utils   import  save_image
from    typing              import  Callable, Optional, Tuple, Union
import  torch.nn.functional as F
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../', 'src/PyThon/NeuralNetwork/trainer')))
from Base import train

# Set the random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Set float32 matrix multiplication precision to medium
torch.set_float32_matmul_precision('high')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def Plateaued_Closed(save_model:Callable[..., None],
              model:torch.nn.Module,
              optimizer:torch.optim.Optimizer,
              plotter,
              save_dir:str,
              model_name:str) -> int:
    save_model(file_path=save_dir, file_name=f"early_stop_{model_name}.ckpt", model=model, optimizer=optimizer)
    torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}_early_stop.pt"))
    if plotter is not None:
        plotter.close()
    return 404

def Plateaued_LrDivider(model:torch.nn.Module,
                        optimizer:torch.optim.Optimizer,
                        plotter,
                        save_dir:str,
                        model_name:str) -> int:
    lr = optimizer.param_groups[0]["lr"]
    new_lr = lr / 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    print(Fore.MAGENTA + f"Learning rate reduced from {lr} to {new_lr}" + Style.RESET_ALL)
    raise NotImplementedError # it fails some where
    return 0

def handler_selfSupervised_dataHandler(Args: tuple[torch.Tensor, torch.Tensor],
                                       model: nn.Module,
                                       device: torch.device) -> torch.Tensor:
    # model.DropOut = False
    data = Args[0].squeeze(1).to(device)
    # raise NotImplementedError("AutoEncoderCNN doesn't support dropOut")
    return data, model(data)

def handler_selfSupervised_loss(criterion, output, data):
    # breakpoint()
    return  criterion(output, data) #+

def handler_selfSupervised(Args:tuple[torch.Tensor, torch.Tensor],
                           criterion: nn.Module,
                           model: nn.Module,
                           device: torch.device = 'cuda') -> tuple[torch.Tensor, torch.Tensor]:
    data, output    = handler_selfSupervised_dataHandler(Args, model, device)
    loss            = handler_selfSupervised_loss(criterion, output, data)
    return output, loss

def save_reconstructions(
                         model: nn.Module,
                         dataloader: torch.utils.data.DataLoader,
                         device: torch.device,
                         save_dir: str,
                         epoch: int,
                         dataHandler: Callable = handler_selfSupervised_dataHandler,
                         num_samples: int = 8,
                         _shuffle: bool = False) -> None:
    """
    Save a batch of original and reconstructed images from the dataloader.
    Args:
        model (nn.Module): The trained autoencoder model
        dataloader (torch.utils.data.DataLoader): DataLoader for validation or test set
        device (torch.device): Device to run the model on
        save_dir (str): Directory to save the images
        epoch (int): Current epoch number for naming
        num_samples (int): Number of samples to save from the batch
    Returns:
        None: Saves images to the specified directory.

    FIXME:
        - 
    Fixed:
        - randperm disturb the order of data,
    """
    """Save batches of originals and their corresponding reconstructions (order preserved)."""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for i, Args in enumerate(dataloader):
            _, recon = dataHandler(Args, model, device)
            # Take only the first num_samples
            originals = Args[0][:num_samples]

            total_samples = originals.size(0)
            if _shuffle:
                rand_indices = torch.randperm(total_samples)[:num_samples]
            else:
                rand_indices = torch.arange(min(num_samples, total_samples))

            originals = originals[rand_indices]
            reconstructions = recon[rand_indices]
            originals = originals.squeeze(1)  # Remove channel dimension if present
            # Save originals and reconstructions
            save_image(originals,       os.path.join(save_dir, f'originals_epoch{epoch}_batch{i}.png'),         nrow=num_samples)
            save_image(reconstructions, os.path.join(save_dir, f'reconstructions_epoch{epoch}_batch{i}.png'),   nrow=num_samples)
            if i >= 5:  # Limit to first 5 batches
                break  # Only process the 5 batch

def trainer(
    _case: str,
    model_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    stride: int,
    embedding_dim: int = 1024,
    ckpt_save_path: str = os.path.join(os.path.dirname(__file__), 'checkpoints'),
    report_path: str = os.path.join(os.path.dirname(__file__), 'training_report.csv'),
    ckpt_path: str|None = None,
    use_hard_negative_mining: bool = False,
    AElayers: int = 9,
    hard_mining_freq: int = 2,
    num_hard_samples: int = 1000,
    sequence_length: int = 1,
    ckpt_save_freq: int = 3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    DropOut: bool = True,
):
    
    Ref = True
    if  _case == "default":
        data_dir    = '/media/d2u25/Dont/frames_Process_30'
        
    elif    _case == "NoRef":
        data_dir    = '/media/d2u25/Dont/frames_Process_30_LightSource'
        Ref         = False

    elif    _case == "Position":
        data_dir    = '/media/d2u25/Dont/frames_Process_30_Position'

    elif    _case == "Velocity":
        data_dir    = '/media/d2u25/Dont/frames_Process_30_Velocity'
        # data_dir    = '/media/d2u25/Dont/frames_Process_30_Velocity_P540'

    elif    _case == "Velocity_wide":
        data_dir    = '/media/d2u25/Dont/frames_Process_30_Velocity_wide'

    elif    _case == "Position_wide":
        data_dir    = '/media/d2u25/Dont/frames_Process_30_Position_wide'

    elif    _case == "default_cropped":
        data_dir    = '/media/d2u25/Dont/frames_Process_30_cropped'

    else:   raise ValueError(f"Unknown case: {_case}")

    if model_name == 'Autoencoder_CNNV1_0':
        if 'wide' in str.lower(_case):
            raise NotImplementedError("Autoencoder_CNNV1_0 doesn't support wide images.")
        ImageSize: Tuple[int, int] = (201,201)
        model = networks.Autoencoder_CNNV1_0(embedding_dim = embedding_dim).to(device)
        model_name = f"CNN_AE_{AElayers}_{model_name}_{_case}_{embedding_dim}_{Ref=}"
    elif model_name == 'Autoencoder_CNNV2_0':
        ImageSize: Tuple[int, int] = (256,1024)
        model = networks.Autoencoder_CNNV2_0(num_blocks=AElayers,#num_blocks=8,
                                                    Image=ImageSize).to(device)
        model_name = f"CNN_AE_{AElayers}_{model_name}_{_case}_{Ref=}"
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    model.DropOut = DropOut
    

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
                                        stride=stride,
                                        sequence_length=sequence_length,
                                        extension=".png",
                                    )

    val_set = DSS.MotherFolderDataset(  
                                        resize=ImageSize,
                                        dicAddresses = dicAddressesValidation,
                                        stride=stride,
                                        sequence_length=sequence_length,
                                        extension=".png",
                                    )


    # Optimize DataLoader
    train_loader    = DataLoader(train_set, batch_size=batch_size, num_workers=16, shuffle=True, pin_memory=True)
    val_loader      = DataLoader(val_set, batch_size=batch_size, num_workers=16, shuffle=False, pin_memory=True)

    optimizer       = AdamW(model.parameters(), lr=learning_rate, 
                            weight_decay=1e-4
                            )
    criterion       = nn.MSELoss()

    # Learning rate scheduler, If optimizer is AdamW skipping scheduler
    if isinstance(optimizer, AdamW):
        scheduler   = None
        # scheduler   = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)  # Divide by 5 every epoch 0.2
    else:
        scheduler   = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)  # Divide by 5 every epoch 0.2

    # Train the model
    model, optimizer, report = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,

        ckpt_path= ckpt_path,

        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs,
        device=device,
        model_name=model_name,
        handler=handler_selfSupervised,
        handler_postfix=save_reconstructions,
        Plateaued=Plateaued_LrDivider,
        ckpt_save_freq=ckpt_save_freq,
        ckpt_save_path=ckpt_save_path,
        report_path=report_path,
        lr_scheduler=scheduler,
        use_hard_negative_mining=use_hard_negative_mining,
        hard_mining_freq=hard_mining_freq,
        num_hard_samples=num_hard_samples,
        new_lr=learning_rate,
    )

    
if __name__ == '__main__':
    AElayers = 9
    stride   = 4#32#4
    DropOut = True

   

    for _case in ["default_cropped",]:
        for embedding_dim in [128,]: #, 1024*4, 1024*8

            trainer(
                _case=_case,
                model_name=f'Autoencoder_CNNV1_0',
                epochs=40,
                batch_size=8,
                learning_rate=0.001, #1e-5,#0.001,#0.01,
                embedding_dim=embedding_dim,
                ckpt_save_freq=30,
                ckpt_save_path=os.path.join(os.path.dirname(__file__), 'checkpoints'),
                ckpt_path=None,
                report_path=os.path.join(os.path.dirname(__file__), 'training_report.csv'),
                use_hard_negative_mining=False,
                stride=stride,
                DropOut = DropOut,
            )