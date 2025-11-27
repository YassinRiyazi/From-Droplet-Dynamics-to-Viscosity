"""
    Author: 
        - Yassin Riyazi
        - Farshad Sangari

    Date: 08-08-2023

    Description: Base trainer class for neural network training.

    TODO:
        - [11-08-2025] Added a GPU temperature monitor and sleep.
        - [14-08-2025] Change color of Val to yellow, and if loss of val increased with comparison to a global minimum change the color to red.
        real time plot
            - [14-08-2025] Plot training loss over epochs real time in the terminal or a window with preferably openGL.
            - [ ] save the train and loss plots

        - [ ] Load and save general information to a YAML file and save it beside the model checkpoints and reports.
        - [ ] Implement SIGINT and SIGTERM handler, save and clean up before termination
            In case of Ctrl + D: break the training and validation loops; finally, save the model and exit
            In case of Ctrl + C: Default behavior (terminate immediately)    
            In case of User defined SIGNAL:
                - SIGUSR1: reduce learning rate by a factor of 5
                - SIGUSR2: turn off dropout during training

        
        - [] add a dummy test model, optimizer and loss in the end of this file to visualize how to use it/ how it looks like
        - [ ] Save the result of the shell in a log file, not redirecting but saving it in a file and showing it in the terminal as well.
        - [ ] Before terminating because of no meaningful change in loss, ask a user for confirmation and wait for 30 seconds.
            In case of no response, save and exit. Same as the case of learning rate becoming too small or Ctrl + D.

        - color the validation loss to easily be finds, mabube undeline it or make it bold




    Help:
        Find PID: nvidia-smi
        To divide Lr: kill -USR1 <pid> 

"""

import  os
import  sys
import  time
import  signal
import  queue
import  torch
import  yaml
import  subprocess
import  threading
import  torch.nn        as      nn
import  pandas          as      pd
import  numpy           as      np
import  numpy.typing    as      npt

from    tqdm            import  tqdm
from    datetime        import  datetime
from    pathlib         import  Path
from    contextlib      import  contextmanager
from    types           import  FrameType

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

ROOT_DIR: Path = Path(__file__).resolve().parent.parent


class TeeStream:
    def __init__(self, original: Any, log_handle: Any) -> None:
        self._original = original
        self._log_handle = log_handle

    def write(self, data: str) -> None:  # pragma: no cover - passthrough
        self._original.write(data)
        self._original.flush()
        self._log_handle.write(data)
        self._log_handle.flush()

    def flush(self) -> None:  # pragma: no cover - passthrough
        self._original.flush()
        self._log_handle.flush()


@contextmanager
def console_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    tee_out = TeeStream(original_stdout, log_file)
    tee_err = TeeStream(original_stderr, log_file)
    sys.stdout = tee_out
    sys.stderr = tee_err
    try:
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


def read_yaml_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def write_yaml(path: Path, content: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(content, handle, sort_keys=False)


def timed_input(
    prompt: str,
    timeout: float,
    input_queue: Optional[queue.Queue[Optional[str]]] = None,
) -> Optional[str]:
    print(prompt, end="", flush=True)
    if input_queue is not None:
        try:
            return input_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    if not sys.stdin or not sys.stdin.isatty():
        return None

    temp_queue: queue.Queue[Optional[str]] = queue.Queue(maxsize=1)

    def _reader() -> None:
        try:
            line = sys.stdin.readline()
            if line == "":
                temp_queue.put(None)
            else:
                temp_queue.put(line.rstrip("\n"))
        except Exception:
            temp_queue.put(None)

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()
    try:
        return temp_queue.get(timeout=timeout)
    except queue.Empty:
        return None


def request_user_confirmation(
    message: str,
    timeout: float = 30.0,
    input_queue: Optional[queue.Queue[Optional[str]]] = None,
) -> bool:
    response = timed_input(f"{message} [Y/n]: ", timeout, input_queue)
    if response is None:
        print("\n[INFO] No response received; proceeding with safe shutdown.")
        return True
    normalized = response.strip().lower()
    return normalized in ("", "y", "yes")

if __name__ == "__main__":
    from deeplearning.RealTimePlotterV2 import RealTimePlotter
else:
    from .RealTimePlotterV2 import RealTimePlotter


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
    shutil.copyfile('/home/d25u2/Desktop/From-Droplet-Dynamics-to-Viscosity/data_config.yaml',
                    os.path.join(save_dir, f'data_config.yaml'))
    
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
    handler: Callable[[tuple[torch.Tensor, torch.Tensor], nn.Module, nn.Module, torch.device, bool], tuple[torch.Tensor, torch.Tensor]],
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
    enable_live_plot: bool = True,
    prefer_opengl_plot: bool = True,
    new_lr: float | None = None,
    use_amp: bool = False,
) -> tuple[nn.Module, torch.optim.Optimizer, pd.DataFrame]:
    start_time = datetime.now()
    save_dir = create_save_dir(ckpt_save_path, model_name)
    print(f"Saving checkpoints and reports to: {save_dir}")
    save_dir_path = Path(save_dir)

    report_output_path = (
        Path(report_path) if report_path is not None else save_dir_path / f"{model_name}_report.csv"
    )
    report_output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_path = save_dir_path / "training_run.yaml"
    existing_metadata = read_yaml_if_exists(metadata_path)
    metadata_entries: list[dict[str, Any]]
    if isinstance(existing_metadata, list):
        metadata_entries = [dict(entry) for entry in existing_metadata]
    elif isinstance(existing_metadata, dict):
        metadata_entries = [dict(existing_metadata)]
    else:
        metadata_entries = []

    plateau_handler_name = getattr(Plateaued, "__name__", repr(Plateaued))  # type: ignore[arg-type]
    metadata_entry: dict[str, Any] = {
        "model_name": model_name,
        "start_time": start_time.isoformat(),
        "epochs_planned": int(epochs),
        "device": str(device),
        "pid": os.getpid(),
        "optimizer": optimizer.__class__.__name__,
        "criterion": criterion.__class__.__name__,
        "report": str(report_output_path),
        "loss_plot": None,
        "compiled": False,
        "best_checkpoint": None,
        "plateau_handler": plateau_handler_name,
    }

    try:
        metadata_entry["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            text=True,
        ).strip()
    except Exception:
        metadata_entry["git_commit"] = None

    config_snapshot = read_yaml_if_exists(ROOT_DIR / "config.yaml")
    if config_snapshot:
        metadata_entry["config_snapshot"] = config_snapshot
    data_config_snapshot = read_yaml_if_exists(ROOT_DIR / "data_config.yaml")
    if data_config_snapshot:
        metadata_entry["data_config_snapshot"] = data_config_snapshot

    metadata_entry["train_dataset_size"] = (
        len(train_loader.dataset) if hasattr(train_loader, "dataset") else None
    )
    metadata_entry["val_dataset_size"] = (
        len(val_loader.dataset) if hasattr(val_loader, "dataset") else None
    )
    metadata_entry["batch_size"] = getattr(train_loader, "batch_size", None)
    metadata_entry["num_workers"] = getattr(train_loader, "num_workers", None)

    metadata_entries.append(metadata_entry)
    write_yaml(metadata_path, metadata_entries)

    input_queue: Optional[queue.Queue[Optional[str]]] = None
    if sys.stdin and hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
        input_queue = queue.Queue()

    shutdown_event = threading.Event()
    shutdown_reason: dict[str, Optional[str]] = {"reason": None}
    epochs_completed = 0

    def request_shutdown(reason: str) -> None:
        if not shutdown_event.is_set():
            shutdown_reason["reason"] = reason
            shutdown_event.set()

    original_sigusr1 = signal.getsignal(signal.SIGUSR1)
    original_sigusr2 = signal.getsignal(signal.SIGUSR2)
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)

    def handle_signal_SIGUSR1(signum: int, frame: Optional[FrameType]) -> None:
        print("\n SIGUSR1 received! Updating learning rate...\n")
        set_divider(optimizer)

    def handle_signal_SIGUSR2(signum: int, frame: Optional[FrameType]) -> None:
        print("\n SIGUSR2 received! Toggling dropout...\n")
        try:
            model.DropOut = not model.DropOut  # type: ignore[attr-defined]
            print(f"DropOut is now set to: {model.DropOut}")
        except AttributeError:
            print("Model does not have a DropOut attribute.")

    def handle_sigterm(signum: int, frame: Optional[FrameType]) -> None:
        print("\n SIGTERM received! Initiating graceful shutdown...\n")
        request_shutdown("sigterm")

    def handle_sigint(signum: int, frame: Optional[FrameType]) -> None:
        print("\n SIGINT received! Terminating after cleanup...\n")
        request_shutdown("sigint")
        if callable(original_sigint):
            original_sigint(signum, frame)  # type: ignore[misc]
        else:
            raise KeyboardInterrupt

    signal.signal(signal.SIGUSR1, handle_signal_SIGUSR1)
    signal.signal(signal.SIGUSR2, handle_signal_SIGUSR2)
    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigint)

    listener_thread: Optional[threading.Thread] = None
    if input_queue is not None:
        def _stdin_listener() -> None:
            while True:
                try:
                    line = sys.stdin.readline()
                except Exception:
                    request_shutdown("stdin_error")
                    input_queue.put(None)
                    break
                if line == "":
                    request_shutdown("stdin_eof")
                    input_queue.put(None)
                    break
                input_queue.put(line.rstrip("\n"))

        listener_thread = threading.Thread(target=_stdin_listener, daemon=True)
        listener_thread.start()

    def confirm_and_maybe_exit(reason: str, message: str) -> bool:
        if request_user_confirmation(message, timeout=30.0, input_queue=input_queue):
            request_shutdown(reason)
            return True
        print("[INFO] Confirmation declined; continuing training.")
        return False

    def shutdown_requested() -> bool:
        if not shutdown_event.is_set():
            return False
        reason = shutdown_reason["reason"]
        if reason == "stdin_eof":
            if request_user_confirmation("Ctrl+D detected. Exit training?", 30.0, input_queue):
                return True
            print("[INFO] Ctrl+D ignored; continuing training.")
            shutdown_event.clear()
            shutdown_reason["reason"] = None
            return False
        return True

    plotter: Optional[RealTimePlotter] = None
    report = pd.DataFrame(
        columns=[
            "model_name",
            "mode",
            "epoch",
            "learning_rate",
            "batch_size",
            "batch_index",
            "loss_batch",
            "avg_train_loss_till_current_batch",
            "avg_val_loss_till_current_batch",
        ]
    )
    numeric_columns = [
        "epoch",
        "learning_rate",
        "batch_size",
        "batch_index",
        "loss_batch",
        "avg_train_loss_till_current_batch",
        "avg_val_loss_till_current_batch",
    ]
    for col in numeric_columns:
        report[col] = report[col].astype(float)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    training_stopped = False

    with console_logger(save_dir_path / "console.log"):
        try:
            model = model.to(device)
            if ckpt_path is not None:
                model, optimizer = load_model(ckpt_path=ckpt_path, model=model, optimizer=optimizer)

            if new_lr is not None:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = new_lr

            try:
                model = torch.compile(model)
                metadata_entry["compiled"] = True
                print("Model compiled with torch.compile()")
            except Exception as exc:
                print(f"Could not compile model: {exc}")

            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            plotter = (
                RealTimePlotter(title=f"{model_name} – Real-Time Loss", prefer_opengl=prefer_opengl_plot)
                if enable_live_plot
                else None
            )

            try:
                print(f"{Fore.YELLOW}Model has {model.DropOut} as dropout state{Style.RESET_ALL}")  # type: ignore[attr-defined]
            except AttributeError:
                print("Model does not have a DropOut attribute.")

            for epoch in tqdm(range(1, epochs + 1)):
                if shutdown_requested():
                    training_stopped = True
                    break

                epochs_completed = epoch

                if (use_hard_negative_mining and epoch % hard_mining_freq == 0) and epoch > 4:
                    print(f"Performing hard negative mining at epoch {epoch}")
                    current_train_loader = hard_negative_mining(
                        model,
                        train_loader,
                        handler,
                        HardNegativeMiningPostHandler,
                        criterion,
                        device,
                        num_hard_samples,
                    )
                else:
                    current_train_loader = train_loader

                lr_value = optimizer.param_groups[0]["lr"]
                if lr_value < 1e-7 and confirm_and_maybe_exit(
                    "low_learning_rate",
                    f"Learning rate has become too small ({lr_value:.2e}). Exit training?",
                ):
                    training_stopped = True
                    break

                model.train()
                loss_avg_train = AverageMeter()
                train_loop = tqdm(
                    current_train_loader,
                    desc=f"{Fore.GREEN}Epoch {epoch}/{epochs} [Train]{Style.RESET_ALL}",
                    leave=False,
                )

                for batch_idx, Args in enumerate(train_loop):
                    if shutdown_requested():
                        training_stopped = True
                        break

                    if GPU_temperature > 0 and batch_idx % 25 == 0:
                        monitor_gpu_temperature(
                            threshold=GPU_temperature,
                            sleep_seconds=GPU_overheat_sleep,
                            verbose=False,
                        )

                    optimizer.zero_grad()

                    with torch.autocast(device_type="cuda", enabled=use_amp):
                        output, loss = handler(
                            Args,
                            criterion,
                            model,
                            device=device,
                            additional=additional_flag,
                        )

                    del output

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    loss_avg_train.update(loss.item(), Args[0].size(0))

                    train_loop.set_postfix(
                        loss=f"{Style.BRIGHT}{Fore.GREEN}{loss_avg_train.avg:.6f}{Style.RESET_ALL}",
                        lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    )

                    new_row = {
                        "model_name": model_name,
                        "mode": "train",
                        "epoch": float(epoch),
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "batch_size": float(Args[0].size(0)),
                        "batch_index": float(batch_idx),
                        "loss_batch": float(loss.item()),
                        "avg_train_loss_till_current_batch": float(loss_avg_train.avg),
                        "avg_val_loss_till_current_batch": np.nan,
                    }
                    report = pd.concat([report, pd.DataFrame([new_row])], ignore_index=True)

                    if plotter is not None and batch_idx % 10 == 0:
                        plotter.update(
                            epoch - 1 + batch_idx / max(1, len(current_train_loader)),
                            train_loss=loss_avg_train.avg,
                            val_loss=None,
                        )

                if training_stopped:
                    break

                model.eval()
                loss_avg_val = AverageMeter()
                with torch.no_grad():
                    val_loop = tqdm(
                        val_loader,
                        desc=f"{Style.BRIGHT}{Fore.YELLOW}Epoch {epoch}/{epochs} [Val]{Style.RESET_ALL}",
                        leave=False,
                    )
                    prev_val_loss = None
                    for batch_idx, Args in enumerate(val_loop):
                        if shutdown_requested():
                            training_stopped = True
                            break

                        if GPU_temperature > 0 and batch_idx % 25 == 0:
                            monitor_gpu_temperature(
                                threshold=GPU_temperature,
                                sleep_seconds=GPU_overheat_sleep,
                                verbose=False,
                            )

                        output, loss = handler(Args, criterion, model)
                        del output
                        loss_avg_val.update(loss.item(), Args[0].size(0))

                        if prev_val_loss is not None and loss.item() > prev_val_loss:
                            val_loop.set_description(
                                f"{Style.BRIGHT}{Fore.RED}Epoch {epoch}/{epochs} [Val]{Style.RESET_ALL}"
                            )
                        else:
                            val_loop.set_description(
                                f"{Style.BRIGHT}{Fore.YELLOW}Epoch {epoch}/{epochs} [Val]{Style.RESET_ALL}"
                            )
                        prev_val_loss = loss.item()

                        val_loop.set_postfix(
                            loss=f"{Style.BRIGHT}{Fore.CYAN}{loss_avg_val.avg:.6f}{Style.RESET_ALL}"
                        )

                        new_row = {
                            "model_name": model_name,
                            "mode": "val",
                            "epoch": float(epoch),
                            "learning_rate": float(optimizer.param_groups[0]["lr"]),
                            "batch_size": float(Args[0].size(0)),
                            "batch_index": float(batch_idx),
                            "loss_batch": float(loss.item()),
                            "avg_train_loss_till_current_batch": np.nan,
                            "avg_val_loss_till_current_batch": float(loss_avg_val.avg),
                        }
                        report = pd.concat([report, pd.DataFrame([new_row])], ignore_index=True)

                    if training_stopped:
                        break

                val_loss_value = float(loss_avg_val.avg) if loss_avg_val.count else float("nan")
                if plotter is not None:
                    plotter.update(epoch=epoch, train_loss=loss_avg_train.avg, val_loss=val_loss_value)

                if loss_avg_val.avg < best_val_loss:
                    best_val_loss = loss_avg_val.avg
                    epochs_no_improve = 0
                    save_model(
                        file_path=save_dir,
                        file_name=f"best_{model_name}.ckpt",
                        model=model,
                        optimizer=optimizer,
                    )
                    metadata_entry["best_checkpoint"] = os.path.join(save_dir, f"best_{model_name}.ckpt")
                else:
                    epochs_no_improve += 1
                    print(Fore.RED + f" No improvement for {epochs_no_improve} epochs." + Style.RESET_ALL)
                    if epochs_no_improve >= 2:
                        if confirm_and_maybe_exit(
                            "no_improvement",
                            f"No improvement for {epochs_no_improve} epochs. Exit training?",
                        ):
                            training_stopped = True
                            break
                        epochs_no_improve = 0
                        set_divider(optimizer)

                if shutdown_requested():
                    training_stopped = True
                    break

                if epoch % ckpt_save_freq == 0:
                    save_model(
                        file_path=save_dir,
                        file_name=f"ckpt_{model_name}_epoch{epoch}.ckpt",
                        model=model,
                        optimizer=optimizer,
                    )

                if handler_postfix is not None:
                    handler_postfix(
                        model=model,
                        dataloader=val_loader,
                        device=device,
                        save_dir=os.path.join(save_dir, f"reconstructions_epoch{epoch}"),
                        epoch=epoch,
                        num_samples=8,
                    )

                if lr_scheduler is not None:
                    lr_scheduler.step()

            final_model_path = save_dir_path / f"{model_name}_final.pt"
            torch.save(model.state_dict(), final_model_path)
            report.to_csv(report_output_path, index=False)
            metadata_entry["final_model"] = str(final_model_path)
            metadata_entry["best_val_loss"] = float(best_val_loss) if best_val_loss != float("inf") else None
            metadata_entry["training_stopped"] = training_stopped

        finally:
            plot_path = save_dir_path / "loss_plot.png"
            if plotter is not None:
                try:
                    plotter.save(str(plot_path))
                    metadata_entry["loss_plot"] = str(plot_path)
                except Exception as exc:
                    print(f"[WARNING] Failed to save plot: {exc}")
                try:
                    plotter.close()
                except Exception as exc:
                    print(f"[WARNING] Failed to close plotter: {exc}")

            if report_output_path.exists():
                display_env = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
                qt_platform = os.environ.get("QT_QPA_PLATFORM", "")
                headless = not display_env and qt_platform not in {"offscreen", "minimal", "minimalegl"}
                if headless:
                    print(
                        "[INFO] Headless environment detected; skipping ResultSavorMain visualization step."
                    )
                else:
                    try:
                        ResultSavorMain(
                            df_address=str(report_output_path),
                            save_dir=str(save_dir_path),
                            lossPlot=True,
                            AccuPlot=False,
                            DPI=400,
                            ShowPlot=False,
                        )
                    except Exception as exc:
                        print(f"[WARNING] Result plotting failed: {exc}")
            else:
                print(
                    f"[WARNING] Report file not found at '{report_output_path}'. "
                    "Skipping ResultSavorMain."
                )

            signal.signal(signal.SIGUSR1, original_sigusr1)
            signal.signal(signal.SIGUSR2, original_sigusr2)
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)

            end_time = datetime.now()
            metadata_entry["end_time"] = end_time.isoformat()
            metadata_entry["duration_seconds"] = (end_time - start_time).total_seconds()
            metadata_entry["shutdown_reason"] = shutdown_reason["reason"]
            metadata_entry["final_learning_rate"] = float(optimizer.param_groups[0]["lr"])
            metadata_entry["epochs_completed"] = int(epochs_completed)
            metadata_entry["status"] = "completed" if not shutdown_reason["reason"] else "stopped"
            metadata_entries[-1] = metadata_entry
            write_yaml(metadata_path, metadata_entries)

    return model, optimizer, report



# Dummy training example to demonstrate the trainer wiring
class _DummyNet(nn.Module):
    """Minimal network used to demonstrate the trainer wiring."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        self.DropOut = True

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = inputs
        if getattr(self, "DropOut", False):
            features = nn.functional.dropout(features, p=0.1, training=self.training)
        return self.backbone(features)


def _dummy_handler(
    args: tuple[torch.Tensor, torch.Tensor],
    criterion: nn.Module,
    model: nn.Module,
    device: torch.device = device,
    additional: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs, targets = args
    inputs = inputs.to(device)
    targets = targets.to(device).float()
    predictions = model(inputs).squeeze(-1)
    loss = criterion(predictions, targets)
    return predictions, loss


def _noop_plateau_handler(
    save_model_fn: Callable[..., Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    plotter: Optional[RealTimePlotter],
    save_dir: str,
    model_name: str,
) -> int:
    return 0


def run_dummy_training_example(epochs: int = 1) -> None:
    torch.manual_seed(42)
    sample_count = 512
    inputs = torch.randn(sample_count, 1, 4, 4)
    targets = torch.rand(sample_count)
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=8)

    model = _DummyNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    demo_output_dir = ROOT_DIR / "Output" / "dummy_runs"
    demo_output_dir.mkdir(parents=True, exist_ok=True)

    train(
        model=model,
        train_loader=loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs,
        device=device,
        model_name="DummyNet",
        ckpt_save_freq=1,
        ckpt_save_path=str(demo_output_dir),
        handler=_dummy_handler,
        handler_postfix=None,
        Plateaued=_noop_plateau_handler,
        additional_flag=False,
        report_path=None,
        enable_live_plot=False,
        prefer_opengl_plot=False,
        use_amp=False,
    )


if __name__ == "__main__":
    run_dummy_training_example(epochs=1)




