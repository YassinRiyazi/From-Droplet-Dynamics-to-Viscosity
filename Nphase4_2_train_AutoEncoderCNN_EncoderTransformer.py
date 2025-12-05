"""
    Author:         Yassin Riyazi
    Date:           04-08-2025
    Description:    Train an embedding-based Transformer for time series data.

    Changelog:
        - 04-08-2025: Initial version. 
        - 17-11-2025: Update to new be adaptible with the congig file structure.
                        Added dataset in utils to load datasets.
        
"""
import  os
import  sys 
import  glob
import  torch
import  networks
import  torch.nn            as      nn
import  torch.optim         as      optim
from    torch.utils.data    import  DataLoader
from    typing              import  Callable, Optional, Union, Tuple # type: ignore
import utils
from    torchvision.utils   import  save_image

import deeplearning

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import  torch
import  random
import  numpy               as      np
utils.set_randomness(42)

# Set float32 matrix multiplication precision to medium
# torch.set_float32_matmul_precision('high')

def handler_supervised(Args:tuple[torch.Tensor, torch.Tensor],
                       criterion: nn.Module,
                       model: nn.Module,
                       b_augmer: bool = utils.config['Training']['Constant_feature_LSTM']['augmented'],
                       **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """
    This function is a placeholder for handling supervised training.
    It can be extended to include specific logic for supervised learning tasks.
    """
    Args = [arg.contiguous().to(device) for arg in Args]
    # model.lstm.reset_states(Args[0])  # Reset LSTM states before processing a new batch


    if b_augmer:
        output = model(Args[0],
                    Args[2])  # Forward pass with additional input
    else:
        output = model(Args[0])  # Forward pass without additional input
        
    loss = criterion(output, Args[1].view(-1))
    return output, loss


import matplotlib.pyplot as plt
import seaborn as sns
def save_reconstructions(model: nn.Module,
                        dataloader: torch.utils.data.DataLoader,
                        device: torch.device,
                        save_dir: str,
                        epoch: int,
                        num_samples: int = 8) -> None:
    """Save predictions and visualize attention weights."""
    return 0
    num_samples = max(num_samples, 16)
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for i, Args in enumerate(dataloader):
            Args = [arg.contiguous().to(device) for arg in Args]
            if hasattr(model, 'transformer') and hasattr(model.transformer, 'reset_states'):
                model.transformer.reset_states(Args[0])
            
            output = model(Args[0], Args[2])
            
            target = Args[1].view(-1)
            predicted = output.view(-1)
            
            total_samples = Args[1].size(0)
            rand_indices = torch.randperm(total_samples)[:num_samples]
            target = target[rand_indices]
            predicted = predicted[rand_indices]
            
            # Save predictions
            text_file_path = os.path.join(save_dir, f"predictions_epoch_{epoch}_batch_{i}.txt")
            with open(text_file_path, 'w') as f:
                f.write("Sample Index\tTarget Value\tPredicted Value\n")
                for j in range(min(num_samples, len(target))):
                    f.write(f"{j}\t{target[j].item():.6f}\t{predicted[j].item():.6f}\n")
            
            # Visualize attention weights
            # This helps trace back important features (time steps)
            attention_weights = model.AttentionWeights()
            if attention_weights is not None and len(attention_weights) > 0:
                # Visualize first layer attention for first sample
                # Shape: (batch, heads, seq_len, seq_len)
                
                # Create a figure with subplots for each layer (up to 4)
                num_layers_to_plot = min(4, len(attention_weights))
                fig, axes = plt.subplots(1, num_layers_to_plot, figsize=(5 * num_layers_to_plot, 5))
                if num_layers_to_plot == 1:
                    axes = [axes]
                
                for layer_idx in range(num_layers_to_plot):
                    attn = attention_weights[layer_idx]
                    if attn is not None:
                        # Get attention for the first sample in batch
                        # attn shape can be (batch, seq_len, seq_len) or (batch, heads, seq_len, seq_len)
                        attn_sample = attn[0].cpu().detach().numpy()
                        
                        # If we have heads dimension, average over it
                        if attn_sample.ndim == 3:
                            attn_avg = attn_sample.mean(axis=0)
                        else:
                            attn_avg = attn_sample
                        
                        ax = axes[layer_idx]
                        sns.heatmap(attn_avg, cmap='viridis', ax=ax, square=True)
                        ax.set_title(f'Layer {layer_idx + 1} Attention')
                        ax.set_xlabel('Key Position (Input Frame)')
                        ax.set_ylabel('Query Position')
            
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"attention_epoch_{epoch}_batch_{i}.png"))
                plt.close()

            # Reconstruct important part of input based on attention weights
            if attention_weights is not None and len(attention_weights) > 0:
                # Use first layer attention weights
                attn = attention_weights[0]
                if attn is not None:
                    # Handle dimensions: (batch, heads, seq_len, seq_len)
                    if attn.dim() == 4:
                        attn_avg = attn.mean(dim=1)  # Average over heads -> (batch, seq_len, seq_len)
                    else:
                        attn_avg = attn
                        
                    # Calculate frame importance: sum over queries (rows) to get column sums
                    # How much is frame j attended to by all other frames?
                    frame_importance = attn_avg.sum(dim=1)  # (batch, seq_len)
                    
                    # Normalize to sum to 1 per batch item
                    frame_importance = frame_importance / (frame_importance.sum(dim=1, keepdim=True) + 1e-9)
                    
                    # Find the most important frame index for each sample
                    best_frame_indices = frame_importance.argmax(dim=1)
                    
                    # Reshape for broadcasting: (batch, seq_len, 1, 1, 1)
                    weights = frame_importance.view(frame_importance.size(0), frame_importance.size(1), 1, 1, 1)
                    
                    # Weighted sum of input frames
                    # Args[0] shape: (batch, seq_len, C, H, W)
                    input_images = Args[0]
                    reconstructed = (input_images * weights).sum(dim=1)  # (batch, C, H, W)
                    
                    # Save a few samples
                    for j in range(min(num_samples, 5)): # Save top 5 samples per batch
                        # Prepare Reconstructed Image (Right)
                        rec_img = reconstructed[j].cpu().detach().numpy()
                        if rec_img.shape[0] == 1:
                            rec_img = rec_img[0]
                        else:
                            rec_img = rec_img.transpose(1, 2, 0)
                            if rec_img.shape[2] == 3:
                                rec_img = rec_img.mean(axis=2)
                        
                        # Normalize and Invert for Visualization
                        # Droplet is dark (low), Background is bright (high)
                        # We want Droplet -> Red (High intensity in colormap), Background -> White/Transparent (Low intensity)
                        # So we invert: Droplet -> High, Background -> Low
                        rec_img_inv = rec_img.max() - rec_img
                        rec_img_norm = (rec_img_inv - rec_img_inv.min()) / (rec_img_inv.max() - rec_img_inv.min() + 1e-9)
                        
                        # Prepare Best Original Frame (Left)
                        best_idx = best_frame_indices[j].item()
                        orig_img = input_images[j, best_idx].cpu().detach().numpy()
                        if orig_img.shape[0] == 1:
                            orig_img = orig_img[0]
                        else:
                            orig_img = orig_img.transpose(1, 2, 0)
                            if orig_img.shape[2] == 3:
                                orig_img = orig_img.mean(axis=2)

                        # Plotting Side-by-Side
                        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
                        
                        # Left: Original (Best Frame)
                        axs[0].imshow(orig_img, cmap='gray')
                        axs[0].set_title(f"Most Attended Frame (t={best_idx})")
                        axs[0].axis('off')

                        # Right: Weighted Reconstruction (Attention Visualization)
                        # cmap='Reds': 0.0 (Background) -> White, 1.0 (Droplet) -> Red
                        im = axs[1].imshow(rec_img_norm, cmap='Reds') 
                        axs[1].set_title("Weighted Reconstruction (Attention)")
                        axs[1].axis('off')
                        
                        # Add colorbar
                        plt.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
                        
                        plt.suptitle(f"Viscosity: {target[j].item():.2f}")
                        plt.tight_layout()
                        plt.savefig(os.path.join(save_dir, f"reconstructed_epoch_{epoch}_batch_{i}_sample_{j}.png"))
                        plt.close()
            if i >= 30:  # Limit to first 30 batches
                break

def train_transformer_model(
                     case:str,
                     proj_dim:int,
                     EmbdSize:int,

                     skip: int              = utils.config['Training']['Constant_features_Transformer']['Stride'],
                     SEQUENCE_LENGTH:int    = utils.config['Training']['Constant_features_Transformer']['window_Lenght'],
                     Autoencoder_CNN: torch.nn.Module| None = None,
                     d_model: int            = utils.config['Training']['Constant_features_Transformer']['d_model'],

            nhead: int              = utils.config['Training']['Constant_features_Transformer']['nhead'],
            num_layers: int         = utils.config['Training']['Constant_features_Transformer']['num_layers'],
            dropout: float          = float(utils.config['Training']['Constant_features_Transformer']['DropOut']),
    ) -> None:
  
    _Ds = utils.data_set()
    _Ds.load_addresses()
    train_set, val_set = _Ds.load_datasets(
                                           stride=utils.config['Training']['Constant_features_Transformer']['Stride'],
                                           sequence_length=utils.config['Training']['Constant_features_Transformer']['window_Lenght'],)
    
    for ds in _Ds.train_dataset.DaughterSets.values():
        ds.Status = utils.config['Training']['Constant_features_Transformer']['Dataset_status']
    for ds in _Ds.val_dataset.DaughterSets.values():
        ds.Status = utils.config['Training']['Constant_features_Transformer']['Dataset_status']
        
    SROF_size = train_set[0][2].shape[1]

    # plto train_set[0][2] with cv2

    import cv2
    cv2.imwrite('test.png', train_set[0][0][0,:,:,:].numpy()[0]*255)

    _case   = utils.config['Dataset']['embedding']['positional_encoding']
    if utils.config['Training']['Constant_features_Transformer']['Dataset_status'] == 'No_reflection':
        Ref = True
    else:
        Ref = False

    ID = f"{utils.config['Dataset']['embedding']['positional_encoding']}_s{utils.config['Training']['Constant_feature_AE']['Stride']}_w{utils.config['Training']['Constant_feature_AE']['window_Lenght']}"
    # ID = _Ds.id
    model_name_AE = f"CNN_AE_{utils.config['Training']['Constant_feature_AE']['AutoEncoder_layers']}_{utils.config['Training']['Constant_feature_AE']['Architecture']}_{_case}_{proj_dim}_{Ref=}_{ID}"
    
    model_name_AE = model_name_AE.replace('_Ref','_self.Ref')
    model_addresse = sorted(glob.glob(f'Output/checkpoints/AE_CNN/{case}/*.pt'))
    if len(model_addresse) == 1 :
        AE_Address = model_addresse[0]
    else:
        raise ValueError(f"Expected exactly one checkpoint for case '{model_name_AE}', but found {len(model_addresse)}.")
    

    model = networks.Encoder_Transformer(
        address_autoencoder=AE_Address,
        proj_dim=proj_dim,
        input_dim=proj_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=d_model * 4,
        S4_size=SROF_size,
        Autoencoder_CNN=Autoencoder_CNN,
        dropout=dropout,
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
                               shuffle=False, pin_memory=True)
  

    # Define the loss function and optimizer
    # optimizer = optim.SGD(model.parameters(), lr=1e-2,)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr            = float(utils.config['Training']['Constant_features_Transformer']['learning_rate']),
                                #   weight_decay  = float(utils.config['Training']['weight_decay'])
                                  )
    criterion = nn.MSELoss()
    
    if isinstance(optimizer, optim.AdamW):
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
        model_name = f"AE_CNN_Transformer_DM{d_model}_NH{nhead}_NL{num_layers}_SL{SEQUENCE_LENGTH}_s{skip}_w{utils.config['Training']['Constant_features_Transformer']['window_Lenght']}_{case=}",

        handler = handler_supervised,
        handler_postfix=save_reconstructions,

        ckpt_save_path=os.path.join(os.path.dirname(__file__),'Output', 'checkpoints','Transformer'),
        ckpt_path=None,
        report_path=os.path.join(os.path.dirname(__file__),'Output','Transformer', 'training_report.csv'),

        lr_scheduler            = lr_scheduler,
        epochs                  = utils.config['Training']['num_epochs'],
        ckpt_save_freq          = utils.config['Training']['checkpoint_save_freq'],
        use_hard_negative_mining= utils.config['Training']['hard_negative_mining'],
        GPU_temperature         = utils.config['Training']['GPU_temperature'],

        enable_live_plot        = utils.config['Training']['Show_plot'],
    )


if __name__ == "__main__":
    if utils.config['Training']['Constant_features_Transformer']['Dataset_status'] == 'No_reflection':
        Ref = True
    else:
        Ref = False
        
    Autoencoder_CNN = networks.Autoencoder_CNN
      

    neural_cases = [
                    # f'CNNV1_0_128_Velocity_Ref={Ref}_s2_w1', 
                    # f'CNNV1_0_1024_Velocity_Ref={Ref}_s2_w1',
                    # f'CNNV1_0_128_Position_Ref={Ref}_s2_w1', 
                    # f'CNNV1_0_1024_Position_Ref={Ref}_s2_w1',
                    # f'CNNV1_0_128_False_Ref={Ref}_s2_w1',
                    f'CNNV1_0_1024_False_Ref={Ref}_s2_w1', # With reflection
                    ]
        
    for case in neural_cases:
        _data = case.split('_')
        proj_dim =  int(_data[2])
        EmbdSize = proj_dim

        utils.config['Dataset']['embedding']['positional_encoding'] = _data[3]
        train_transformer_model(
                        case=case,
                        EmbdSize=EmbdSize,
                        proj_dim=proj_dim,
                        Autoencoder_CNN=Autoencoder_CNN,
                        )