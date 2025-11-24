"""
    Author: 
        - Yassin Riyazi
        - Farshad Sangari

    Date: 08-08-2023

    Description: Base trainer class for neural network training.

    TODO:
        - [ ] make the base function for training or validation and add some hook to it for training.
        - [ ] save the train and loss plots
        - [ ] Implement SIGINT and SIGTERM handler, save and clean up before termination
            
            In case of Ctrl + D: save the model and exit
            In case of Ctrl + C: Default behavior (terminate immediately)
            In case of Ctrl + X: turn off GPU temperature checking
            
            In case of User defined SIGNAL:
                - SIGUSR1: reduce learning rate by a factor of 5
                - SIGUSR2: turn off dropout during training

        - [ ] Load and save general information to a YAML file
        - [14-08-2025] Change color of Val to yellow, and if loss of val increased with comparison to a global minimum change the color to red.
        - [14-08-2025] Plot training loss over epochs real time in the terminal or a window with preferably openGL.
        - [11-08-2025] Added a GPU temperature monitor and sleep.
        - [ ] Save the result of the shell in a log file
        - [ ] Before terminating because of no meaningful change in loss, ask a user for confirmation and wait for 30 seconds.
        In case of no response, save and exit. Same as the case of learning rate becoming too small or Ctrl + D.


    Help:
        Find PID: nvidia-smi
        To divide Lr: kill -USR1 <pid> 

"""

import  os
import  time
import  torch
import  signal
import  subprocess
import  torch.nn        as      nn
import  pandas          as      pd
import  numpy           as      np
import  numpy.typing    as      npt

from    tqdm            import  tqdm
from    datetime        import  datetime

from    torch.optim     import  lr_scheduler # type: ignore
from    colorama        import  Fore, Style, init as colorama_init
from    typing          import  Any, TypeAlias, Tuple, Callable, Optional, Union

if __name__ == "__main__":
    from FinalresultPlotter import ResultSavorMain
else:
    from .FinalresultPlotter import ResultSavorMain

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

colorama_init(autoreset=True)
DataGetItemType: TypeAlias = Tuple[torch.Tensor, ...]

if __name__ == "__main__":
    from RealTimePlotter import RealTimePlotter
else:
    from .RealTimePlotter import RealTimePlotter


def set_divider(optimizer: torch.optim.Optimizer,
                new_lr: float|None = None,
                Divider: float = 5.0) -> None:
    if new_lr is None:
        new_lr = optimizer.param_groups[0]["lr"]/Divider

    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
    print(Fore.MAGENTA + f"Learning rate reduced from {new_lr*Divider} to {new_lr}" + Style.RESET_ALL)
    return None

def monitor_gpu_temperature(threshold: int = 70,
                            sleep_seconds: float = 5.0,
                            gpu_id: int = 0,
                            verbose: bool = False) -> None:
    """
    Checks the GPU temperature and sleeps if it exceeds a threshold.

    Args:
        threshold (int): Temperature in Celsius above which the function sleeps.
        sleep_seconds (int): Number of seconds to sleep when the threshold is exceeded.
        gpu_id (int): ID of the GPU to monitor.

    returns:
        None: The function will print a warning and sleep if the temperature exceeds the threshold.
    """
    try:
        # Query GPU temperature using nvidia-smi
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu=temperature.gpu", "--format=csv,noheader,nounits", f"-i={gpu_id}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        temp = int(result.stdout.strip())

        if temp > threshold:
            if verbose:
                print(f"[WARNING] GPU {gpu_id} temperature {temp}°C exceeds {threshold}°C. Sleeping for {sleep_seconds}s...")
            time.sleep(sleep_seconds)
        else:
            if verbose:
                print(f"[INFO] GPU {gpu_id} temperature {temp}°C is within safe limits.")

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to get GPU temperature: {e.stderr}")
    except ValueError:
        print("[ERROR] Could not parse GPU temperature.")

class AverageMeter(object):
    """
    computes and stores the average and current value

    Author: 
        - Farshad Sangari
        
    Date: 08-08-2023
    """
    def __init__(self, start_val: float = 0, start_count: int = 0, start_avg: float = 0, start_sum: float = 0):
        self.reset()
        self.val = start_val
        self.avg = start_avg
        self.sum = start_sum
        self.count = start_count

    def reset(self):
        """
        Initialize 'value', 'sum', 'count', and 'avg' with 0.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, num: int = 1) -> None:
        """
        Update 'value', 'sum', 'count', and 'avg'.
        """
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count

def create_save_dir(base_path: Union[str, os.PathLike[str]], model_name: str) -> str:
    """
    Create a timestamped directory for saving model checkpoints and reports
    """
    import shutil
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(base_path, f"{model_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    shutil.copyfile('/home/d25u2/Desktop/From-Droplet-Dynamics-to-Viscosity/config.yaml',
                    os.path.join(save_dir, f'config.yaml'))
    
    return save_dir

def save_model(file_path: str,
               file_name: str,
               model: nn.Module,
               optimizer: Optional[torch.optim.Optimizer] = None) -> None:
    """
    Save model and optimizer state

    Args:
        file_path (str): Directory to save the model
        file_name (str): Name of the file to save the model
        model (nn.Module): PyTorch model to save
        optimizer (Optional[nn.Module]): Optimizer to save (if available)
    Returns:
        None: Saves the model state to the specified file

    Authors: 
        - Yassin Riyazi
        - Farshad Sangari

    Date: 08-08-2025
    """
    
    state_dict: dict[str, Any] = dict()
    state_dict["model"] = model.state_dict()

    if optimizer is not None:
        state_dict["optimizer"] = optimizer.state_dict()
    torch.save(state_dict, os.path.join(file_path, file_name))

def load_model(ckpt_path: Union[str, os.PathLike[str]],
               model: nn.Module,
               optimizer: Optional[torch.optim.Optimizer] = None) -> tuple[nn.Module, Optional[torch.optim.Optimizer]]:
    """
    Load model and optimizer state from checkpoint
    Args:
        ckpt_path (Union[str, os.PathLike[str]]): Path to the checkpoint file
        model (nn.Module): PyTorch model to load state into
        optimizer (Optional[nn.Module]): Optimizer to load state into (if available)
    Returns:
        model (nn.Module): Model with loaded state
        optimizer (Optional[nn.Module]): Optimizer with loaded state (if provided)
    """
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint.keys():
        optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer

def normal_accuracy(pred: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Calculate the accuracy of predictions against true labels.
    Args:
        pred (torch.Tensor): Predictions from the model
        labels (torch.Tensor): True labels
    Returns:
        float: Accuracy as a percentage
    """
    return (((pred.argmax(dim=1) == labels).sum() / len(labels)) * 100).item()

def teacher_forcing_decay(epoch: int, num_epochs: int) -> float:
    """
    Calculate the teacher forcing ratio for a given epoch.
    Args:
        epoch (int): Current epoch number
        num_epochs (int): Total number of epochs
    Returns:
        float: Teacher forcing ratio for the current epoch"""
    initial_tf_ratio = 1.0
    final_tf_ratio = 0.01
    decay_rate = (final_tf_ratio / initial_tf_ratio) ** (1 / (num_epochs - 1))

    tf_ratio = max(0.01, initial_tf_ratio * (decay_rate ** epoch))
    return tf_ratio

def HardNegativeMiningPostHandler(args: tuple[torch.Tensor, ...]) -> npt.NDArray[np.float32]:
    """
    Post-processing handler for hard negative mining.
    This function can be customized to save or visualize hard negative samples.
    Currently, it does nothing but can be extended as needed.
    Args:
        args (tuple[torch.Tensor, ...]): Tuple containing the data and possibly other tensors
    Returns:
        np.ndarray: Processed data, currently just returns the first tensor in args as a numpy array
    """
    return args[0].numpy()  # Assuming args is a tuple with the first element being the data

def hard_negative_mining(model: nn.Module,
                         dataloader: torch.utils.data.DataLoader[DataGetItemType],
                         handler: Callable, #TODO: Make this more flexible for different model types
                         HardNegativeMiningPostHandler: Callable,
                         criterion: nn.Module,
                         device: Union[str, torch.device] = device,
                         num_hard_samples: int = 2000) -> torch.utils.data.DataLoader[DataGetItemType]:
    """
    Select the hardest examples (highest loss) from the dataset
    Returns a new DataLoader containing only the hard examples

    Args:
        model (nn.Module): The trained model to evaluate
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset
        criterion (nn.Module): Loss function to compute the loss
        device (Union[str, torch.device]): Device to run the model on ('cuda' or 'cpu')
        num_hard_samples (int): Number of hard examples to select
    Returns:
        torch.utils.data.DataLoader: DataLoader containing only the hard examples

    TODO:
        - Add handler for different model types (e.g., CNN, LSTM)
    """
    model.eval()
    losses = []
    all_data = []
    
    with torch.no_grad():
        for args in dataloader:
            output, loss = handler(args, criterion, model)
            # If loss is a scalar, reshape it to match batch size
            if loss.dim() == 0:
                loss = loss.unsqueeze(0)
            # Calculate per-sample loss
            per_sample_loss = loss.view(-1)
            losses.extend(per_sample_loss.cpu().numpy())
            all_data.append(HardNegativeMiningPostHandler(args))
    
    # Convert to numpy arrays
    losses = np.array(losses)
    all_data = np.concatenate(all_data, axis=0)
    
    # Get indices of hardest examples
    hard_indices = np.argsort(losses)[-num_hard_samples:]
    
    # Create new dataset with hard examples
    hard_data = all_data[hard_indices]
    hard_dataset = torch.utils.data.TensorDataset(torch.from_numpy(hard_data), torch.zeros(len(hard_data)))
    
    # Create new dataloader
    hard_loader = torch.utils.data.DataLoader(
        hard_dataset,
        batch_size=min(128, num_hard_samples),  # Use smaller batch size for hard examples
        shuffle=True,
        num_workers=4
    )
    
    return hard_loader



def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader[DataGetItemType],
    val_loader: torch.utils.data.DataLoader[DataGetItemType],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: Union[str, torch.device],
    model_name: str,
    ckpt_save_freq: int,
    ckpt_save_path: Union[str, os.PathLike[str]],
    handler: Callable[[tuple[torch.Tensor, torch.Tensor], nn.Module, nn.Module, torch.device, bool], None],
    handler_postfix: Union[Callable, None],
    Plateaued: Callable[[Callable, nn.Module, torch.optim.Optimizer, Optional[RealTimePlotter], str, str], int],
    additional_flag: bool = False,
    ckpt_path: Union[str, os.PathLike[str], None] = None,
    report_path: Union[str, os.PathLike[str], None] = None,
    lr_scheduler: Union[torch.optim.lr_scheduler.LRScheduler, None] = None,
    Validation_save_threshold: float = 0.0,
    use_hard_negative_mining: bool = True,
    hard_mining_freq: int = 1,
    num_hard_samples: int = 1000,
    GPU_temperature: int = 70,
    GPU_overheat_sleep: float = 5.0,
    # --- NEW toggles ---
    enable_live_plot: bool = True,
    prefer_opengl_plot: bool = True,
    new_lr: float|None = None,
) -> tuple[nn.Module, torch.optim.Optimizer, pd.DataFrame]:
    
    def handle_signal_SIGUSR1(signum, frame):
        print("\n SIGUSR1 received! Updating learning rate...\n")
        set_divider(optimizer)

    signal.signal(signal.SIGUSR1, handle_signal_SIGUSR1)

    def handle_signal_SIGUSR2(signum, frame):
        print("\n SIGUSR2 received! Turning off the dropout...\n")
        try:
            model.DropOut = not model.DropOut
            print(f"DropOut is now set to: {model.DropOut}")
        except AttributeError:
            print("Model does not have a DropOut attribute.")

    
    signal.signal(signal.SIGUSR2, handle_signal_SIGUSR2)

    save_dir = create_save_dir(ckpt_save_path, model_name)
    print(f"Saving checkpoints and reports to: {save_dir}")

    model = model.to(device)
    if ckpt_path is not None:
        model, optimizer = load_model(ckpt_path=ckpt_path, model=model, optimizer=optimizer)

    if new_lr is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

    # Live plotter
    plotter = RealTimePlotter(title=f"{model_name} – Real-Time Loss", prefer_opengl=prefer_opengl_plot) if enable_live_plot else None

    report = pd.DataFrame(columns=[
        "model_name", "mode", "epoch", "learning_rate", "batch_size", "batch_index",
        "loss_batch", "avg_train_loss_till_current_batch", "avg_val_loss_till_current_batch"
    ])
    numeric_columns = ["epoch", "learning_rate", "batch_size", "batch_index",
                       "loss_batch", "avg_train_loss_till_current_batch",
                       "avg_val_loss_till_current_batch"]
    for col in numeric_columns:
        report[col] = report[col].astype(float)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    try:
        print(f"{Fore.YELLOW}Model has {model.DropOut} as dropout state{Style.RESET_ALL}")
    except AttributeError:
        print("Model does not have a DropOut attribute.")


    for epoch in tqdm(range(1, epochs + 1)):
        # --- Hard negative mining section unchanged ---
        if (use_hard_negative_mining and epoch % hard_mining_freq == 0) and epoch > 4:
            print(f"Performing hard negative mining at epoch {epoch}")
            current_train_loader = hard_negative_mining(
                model, train_loader, criterion, device, num_hard_samples
            )
        else:
            current_train_loader = train_loader


        # if learning rate become smalle than 1e-6, stop training
        if optimizer.param_groups[0]["lr"] < 1e-7:
            print(Fore.RED + f"Learning rate has become too small ({optimizer.param_groups[0]['lr']}). Stopping training." + Style.RESET_ALL)
            break
        # ----------------- TRAIN -----------------
        model.train()
        loss_avg_train = AverageMeter()
        prev_train_loss = None
        train_loop = tqdm(current_train_loader, desc=f"{Fore.YELLOW}Epoch {epoch}/{epochs} [Train]{Style.RESET_ALL}")

        for batch_idx, Args in enumerate(train_loop):
            optimizer.zero_grad()
            output, loss = handler(Args, criterion, model,
                                   device=device,   
                                   additional=additional_flag,)
            del output  # Free up memory
            loss.backward()
            optimizer.step()

            loss_avg_train.update(loss.item(), Args[0].size(0))

            # Dynamic color based on current batch vs previous batch
            train_loop.set_description(f"{Fore.GREEN}Epoch {epoch}/{epochs} [Train]{Style.RESET_ALL}")
            # if prev_train_loss is not None and loss.item() > prev_train_loss:
            #     train_loop.set_description(f"{Fore.RED}Epoch {epoch}/{epochs} [Train]{Style.RESET_ALL}")
            # else:
            #     train_loop.set_description(f"{Fore.YELLOW}Epoch {epoch}/{epochs} [Train]{Style.RESET_ALL}")
            # prev_train_loss = loss.item()

            train_loop.set_postfix(loss=loss_avg_train.avg, lr=optimizer.param_groups[0]["lr"])


            # report logging ...
            new_row = {
                "model_name": model_name, "mode": "train", "epoch": float(epoch),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "batch_size": float(Args[0].size(0)), "batch_index": float(batch_idx),
                "loss_batch": float(loss.item()),
                "avg_train_loss_till_current_batch": float(loss_avg_train.avg),
                "avg_val_loss_till_current_batch": np.nan,
            }
            report = pd.concat([report, pd.DataFrame([new_row])], ignore_index=True)

            # if batch_idx % 10 == 0:
            #     monitor_gpu_temperature(threshold=GPU_temperature, sleep_seconds=GPU_overheat_sleep)

            # OPTIONAL: per-BATCH plotting (commented). Enable for finer granularity.
            if plotter is not None and batch_idx % 10 == 0:
                plotter.update(epoch - 1 + batch_idx / max(1, len(current_train_loader)),
                               train_loss=loss_avg_train.avg, val_loss=0)

        # ----------------- VALIDATION -----------------
        model.eval()
        loss_avg_val = AverageMeter()
        prev_val_loss = None

        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"{Fore.YELLOW}Epoch {epoch}/{epochs} [Val]{Style.RESET_ALL}")
            for batch_idx, Args in enumerate(val_loop):
                output, loss = handler(Args, criterion, model)
                del output  # Free up memory
                loss_avg_val.update(loss.item(), Args[0].size(0))

                if prev_val_loss is not None and loss.item() > prev_val_loss:
                    val_loop.set_description(f"{Fore.RED}Epoch {epoch}/{epochs} [Val]{Style.RESET_ALL}")
                else:
                    val_loop.set_description(f"{Fore.YELLOW}Epoch {epoch}/{epochs} [Val]{Style.RESET_ALL}")
                prev_val_loss = loss.item()

                val_loop.set_postfix(loss=loss_avg_val.avg)

                new_row = {
                    "model_name": model_name, "mode": "val", "epoch": float(epoch),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "batch_size": float(Args[0].size(0)), "batch_index": float(batch_idx),
                    "loss_batch": float(loss.item()),
                    "avg_train_loss_till_current_batch": np.nan,
                    "avg_val_loss_till_current_batch": float(loss_avg_val.avg),
                }
                report = pd.concat([report, pd.DataFrame([new_row])], ignore_index=True)

                # if batch_idx % 10 == 0:
                #     monitor_gpu_temperature(threshold=GPU_temperature, sleep_seconds=GPU_overheat_sleep)

        # ----------------- PLOT one point per epoch -----------------
        if plotter is not None:
            plotter.update(epoch=epoch, train_loss=loss_avg_train.avg, val_loss=loss_avg_val.avg)

        # ----------------- CHECKPOINTS & EARLY STOP -----------------
        if loss_avg_val.avg < best_val_loss :
            best_val_loss = loss_avg_val.avg
            epochs_no_improve = 0
            save_model(file_path=save_dir, file_name=f"best_{model_name}.ckpt",
                       model=model, optimizer=optimizer)
        else:
            epochs_no_improve += 1
            print(Fore.RED + f"{Fore.RED} No improvement for {epochs_no_improve} epochs." + Style.RESET_ALL)
            if epochs_no_improve >= 2:
                print(Fore.RED + f"Early stopping at epoch {epoch} due to no improvement for 2 epochs." + Style.RESET_ALL)
                # res = Plateaued(save_model=save_model,
                #                 model=model,
                #                 optimizer=optimizer,
                #                 plotter=plotter,
                #                 save_dir=save_dir,
                #                 model_name=model_name )
                # if res==404:
                #     break
                set_divider(optimizer)
                
                

        if epoch % ckpt_save_freq == 0:
            save_model(file_path=save_dir, file_name=f"ckpt_{model_name}_epoch{epoch}.ckpt",
                       model=model, optimizer=optimizer)

        if handler_postfix is not None:
            handler_postfix(
                model=model, dataloader=val_loader, device=device,
                save_dir=os.path.join(save_dir, f"reconstructions_epoch{epoch}"),
                epoch=epoch, num_samples=8
            )

        if lr_scheduler is not None:
            lr_scheduler.step()

        if report_path is not None:
            report.to_csv(os.path.join(save_dir, f"{model_name}_report.csv"), index=False)

    # Final save & close plot
    torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}_final.pt"))
    if plotter is not None:
        plotter.close()

    ResultSavorMain(df_address =os.path.join(save_dir, f"{model_name}_report.csv"),
                    save_dir=save_dir,
                    lossPlot=True,
                    AccuPlot=False,
                    DPI=400,
                    ShowPlot=False)
    
    return model, optimizer, report
