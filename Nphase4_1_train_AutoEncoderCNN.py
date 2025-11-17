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
# import  glob
# import  dataset
# import  dataset             as      DSS
import  os
import  sys
import  torch
import  utils
import  networks
import  torch.nn            as      nn
from    torch.utils.data    import  DataLoader
from    torch.optim         import  Adam, lr_scheduler, AdamW
from    torchvision.utils   import  save_image
from    typing              import  Callable, Optional, Tuple, Union
import  torch.nn.functional as      F
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../../../../',
                                             'src/PyThon/NeuralNetwork/trainer')))
import deeplearning
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
                           device: torch.device = 'cuda',
                           additional: Optional[bool] = False) -> tuple[torch.Tensor, torch.Tensor]:
    Scale = 1
    Args = [arg.to(device) if arg.dim() <= 4 else arg.squeeze(1).to(device) for arg in Args]

    output = model(Args[0])
    loss   = criterion(output, Args[0])
    if additional:
        mean_error = (output - Args[0]).abs()[Args[1] > 0.01].mean()
        mean_error = Scale * mean_error.item() / (Args[1] > 0.001).sum()

        loss  += mean_error
    return output, loss

from contextlib import contextmanager
import types

@contextmanager
def temporary_method(obj, method_name, new_method):
    """
    Temporarily replace obj.method_name with new_method.

    Works for class or instance methods.
    """
    old_method = getattr(obj, method_name)

    # bind method correctly (so `self` works)
    bound_new_method = types.MethodType(new_method, obj)
    setattr(obj, method_name, bound_new_method)

    try:
        yield   # run code under patch
    finally:
        # restore original
        setattr(obj, method_name, old_method)

def save_reconstructions(
                         model: nn.Module,
                         dataloader: torch.utils.data.DataLoader,
                         device: torch.device,
                         save_dir: str,
                         epoch: int,
                         dataHandler: Callable = handler_selfSupervised_dataHandler,
                         num_samples: int = 8,
                         _shuffle: bool = False,) -> None:
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
    Scale = 1
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for i, Args in enumerate(dataloader):
            
            # _, recon = dataHandler(Args, model, device)
            Args = [arg.to(device) if arg.dim() <= 4 else arg.squeeze(1).to(device) for arg in Args]
            recon = model(Args[0])

            mean_error = (recon - Args[0]).abs()[Args[1] > 0.01].mean()
            mean_error = Scale * mean_error.item() / (Args[1] > 0.001).sum()

            if mean_error > 0.50:
                p = 0.5  # Cap the mean error to avoid extreme adjustments
            elif mean_error < 0:
                p = 0.05
            else:
                p = mean_error
            
            model.dropout.p = p
            print(f"Mean reconstruction error (reflection areas): {mean_error:.6f}, Adjusted Dropout p: {model.dropout.p:.2f}")
            
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
            if originals.size(1) != 1:
                originals = originals.unsqueeze(1)
            if reconstructions.size(1) != 1:
                reconstructions = reconstructions.unsqueeze(1)
            save_image(originals,       os.path.join(save_dir, f'originals_epoch{epoch}_batch{i}.png'),         nrow=num_samples)
            save_image(reconstructions, os.path.join(save_dir, f'reconstructions_epoch{epoch}_batch{i}.png'),   nrow=num_samples)
                
            if i >= 5:  # Limit to first 5 batches
                break  # Only process the 5 batch

def trainer(
    embedding_dim: int = 1024,
    ckpt_save_path: str = os.path.join(os.path.dirname(__file__), 'checkpoints'),
    report_path: str = os.path.join(os.path.dirname(__file__),'Output', 'training_report.csv'),
    ckpt_path: str|None = None,
    AElayers: int = 9,

    use_hard_negative_mining:   bool    = utils.config['Training']['hard_negative_mining'],
    hard_mining_freq:           int     = utils.config['Training']['hard_mining_freq'],
    num_hard_samples:           int     = 1000,
    ckpt_save_freq:             int     = utils.config['Training']['checkpoint_save_freq'],
    device: str                         = 'cuda' if torch.cuda.is_available() else 'cpu',
    DropOut:                    bool    = utils.config['Training']['Constant_feature_AE'].get('DropOut', False)
):
    
    if utils.config['Training']['Constant_feature_AE']['Architecture'] == 'Autoencoder_CNNV1_0':
        ImageSize: Tuple[int, int] = (201,201)
        model = networks.Autoencoder_CNNV1_0(DropOut = utils.config['Training']['Constant_feature_AE']["DropOut"],#.get('DropOut', True),
                                             embedding_dim = embedding_dim).to(device)

    elif utils.config['Training']['Constant_feature_AE']['Architecture'] == 'Autoencoder_CNNV2_0':
        ImageSize: Tuple[int, int] = (256,1024)
        model = networks.Autoencoder_CNNV2_0(num_blocks=AElayers,#num_blocks=8,
                                                    Image=ImageSize).to(device)
    else:
        raise ValueError(f"Unknown model name: {utils.config['Training']['Constant_feature_AE']['Architecture']}")
    
    model.DropOut = DropOut
   
    _Ds = utils.data_set()
    _Ds.load_addresses()
    train_dataset, val_dataset = _Ds.load_datasets(embedding_dim=embedding_dim,
                                                   stride=utils.config['Training']['Constant_feature_AE']['Stride'],
                                                  sequence_length=utils.config['Training']['Constant_feature_AE']['window_Lenght']
                                                  )
    train_dataset, val_dataset = _Ds.reflectionReturn_Setter(flag=True)
    model_name = _Ds.model_name

    # Optimize DataLoader
    train_loader    = DataLoader(train_dataset, batch_size=utils.config['Training']['batch_size'], 
                                 num_workers=utils.config['Training']['num_workers'], shuffle=True, pin_memory=True)
    val_loader      = DataLoader(val_dataset, batch_size=utils.config['Training']['batch_size'], 
                                 num_workers=utils.config['Training']['num_workers'], shuffle=False, pin_memory=True)

    optimizer       = AdamW(model.parameters(), 
                            lr=float(utils.config['Training']['learning_rate']), 
                            weight_decay=float(utils.config['Training']['weight_decay'])
                            )

    criterion       = nn.MSELoss()

    # Learning rate scheduler, If optimizer is AdamW skipping scheduler
    scheduler   = None
    if isinstance(optimizer, AdamW):
        pass
        # scheduler   = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)  # Divide by 5 every epoch 0.2
    else:
        scheduler   = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)  # Divide by 5 every epoch 0.2



    

    # Train the model
    model, optimizer, report = deeplearning.train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,

        ckpt_path= ckpt_path,

        criterion=criterion,
        optimizer=optimizer,
        epochs=utils.config['Training']['num_epochs'],
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
        new_lr=float(utils.config['Training']['learning_rate']),
    )

    
if __name__ == '__main__':
    AElayers = 9
    
    for _case in utils.config['Dataset']['embedding']['Valid_encoding']:
        utils.config['Dataset']['embedding']['positional_encoding'] = _case

        for embedding_dim in ([128,1024]): #, 1024*4, ,1024*8, 128
            trainer(
                embedding_dim=embedding_dim,
                ckpt_save_path=os.path.join(os.path.dirname(__file__),'Output', 'checkpoints','AE_CNN'),
                ckpt_path=None,
                report_path=os.path.join(os.path.dirname(__file__),'Output','AE_CNN', 'training_report.csv'),
            )