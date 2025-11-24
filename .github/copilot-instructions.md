# Copilot Instructions: Droplet Dynamics to Viscosity Prediction

## Project Overview
Deep learning pipeline for predicting viscosity from high speed video footage of droplet dynamics;Extensivly studing the different features types:
1. **AutoEncoder (CNN)**: Compress droplet images into embeddings
2. **LSTM**: Predict viscosity from temporal sequences of embeddings

The pipeline processes experimental droplet videos with metadata (4S-SROF: time, position, contact angles, velocity) to predict fluid viscosity.

## Architecture & Data Flow

### Dataset Structure
```
/media/d25u2/Dont/Viscosity/
├── {tilt_angle}/              # e.g., 280, 285, 290 degrees
│   └── {fluid_name}/          # e.g., S3-SNr2.6_D
│       └── {experiment}/      # e.g., T528_01_0.017002
│           ├── frames_rotated/ or databases/ (cropped)
│           ├── detections.csv  # Drop location tracking
│           └── result.csv      # 4S-SROF features
```

**4S-SROF** = Four-Stage Surface Roughness Optimization Features (from `result.csv`):
- `time (s)`, `x_center (cm)`, `y_center (cm)`
- `adv (degree)`, `rec (degree)`, `middle_angle_degree (degree)`
- `contact_line_length (cm)`, `velocity (cm/s)`

### Two-Tier Dataset Loading
1. **MotherFolderDataset**: Manages multiple fluids, handles train/val/test splits
   - Uses `dicLoader()` to load pre-split datasets from `dataset/dataset_splits/*.pkl`
   - Internally creates a `DaughterFolderDataset` per fluid
   - **CRITICAL**: Use caching to avoid slow re-initialization (see `dataset/QUICKSTART_CACHING.md`)

2. **DaughterFolderDataset**: Loads sequences from a single fluid's experiments
   - Returns `(images, viscosity, SROF_features)` tuples
   - Handles stride and sequence_length for temporal batching
   - Supports optional positional encoding and light reflection removal

### Training Pipeline

**Phase 1 - AutoEncoder** (`Nphase4_1_train_AutoEncoderCNN.py`):
```bash
python Nphase4_1_train_AutoEncoderCNN.py
```
- Trains CNN autoencoder on single frames (self-supervised)
- Architecture: `networks/AutoEncoder_CNNV1_0.py`
- Saves to `Output/checkpoints/AE_CNN/{W|WO} Reflection/`
- Config: `config.yaml` → `Training.Constant_feature_AE`

**Phase 2 - LSTM** (`Train_LSTM_4SSROF.py`):
```bash
python Train_LSTM_4SSROF.py
```
- Loads pre-trained AutoEncoder embeddings
- Trains LSTM on sequences to predict viscosity
- Architecture: `networks/AutoEncoder_CNN_LSTM.py`
- Saves to `Output/checkpoints/LSTM/AE_CNN_LSTM_HD{hidden}_SL{seq_len}_*`
- Config: `config.yaml` → `Training.Constant_feature_LSTM`

## Configuration (`config.yaml`)

**Key settings:**
- `reflection`: Enable/disable light reflection removal preprocessing
- `Dataset.Dataset_Root`: Path to experimental data
- `Dataset.embedding.positional_encoding`: `'False'`, `'Position'`, or `'Velocity'`
- `Training.Constant_feature_AE.valid_latent_dim`: Embedding dimensions [128, 1024]
- `Training.Constant_feature_LSTM.Stride` & `window_Lenght`: Temporal sampling params

**GPU thermal management**: Training pauses at `GPU_temperature` threshold (default 67°C)

## Development Patterns

### Dataset Caching (CRITICAL for productivity)
```python
from dataset import MotherFolderDataset, dicLoader

cache_path = "Output/dataset_cache_train.pkl"
if os.path.exists(cache_path):
    dataset = MotherFolderDataset.load_cache(cache_path)  # ~2-5 seconds
else:
    dicAddresses, _, _ = dicLoader(root=config['Dataset']['Dataset_Root'])
    dataset = MotherFolderDataset(dicAddresses, stride=10, sequence_length=10)
    dataset.save_cache(cache_path)  # First run: 2-5 minutes
```

### Custom Handler Pattern
Training scripts use handler functions to abstract data processing:
- `handler_selfSupervised()`: AutoEncoder training (input = output)
- `handler_supervised()`: LSTM training with LSTM state reset per batch
- Defined in training scripts, passed to `deeplearning.Base.Trainer`

### Signal-Based Training Control
The `deeplearning.Base` trainer responds to Unix signals:
- `SIGUSR1`: Reduce learning rate by factor of 5
- `SIGUSR2`: Toggle dropout during training
- `Ctrl+D`: Save model and exit gracefully

Find PID: `nvidia-smi`, then: `kill -USR1 <pid>`

### Type Aliases & Data Structures
Heavy use of `TypeAlias` for documentation (see `dataset/header.py`):
- `StringListDict`: `Dict[str, List[str]]` for dataset address mappings
- `DaughterSet_getitem_`: Return type for dataset `__getitem__`
- `DaughterSet_internal_`: Internal tuple format with viscosity, addresses, drop location, SROF

## Common Workflows

### Running Full Training Pipeline
```bash
# 1. Train AutoEncoder
python Nphase4_1_train_AutoEncoderCNN.py

# 2. Train LSTM (automatically loads AutoEncoder from checkpoints)
python Train_LSTM_4SSROF.py

# 3. Evaluate/plot results
python Nphase4_3_TestPlot.py
```

### Adding New Preprocessing
Integrate into `DaughterFolderDataset.__init__()`:
- Light reflection removal: `dataset/light_source/LightSourceReflectionRemoving.py`
- Positional encoding: `dataset/positional_encoding/PositionalImageGenerator.py`
- Both triggered via `config.yaml` flags

### Debugging Dataset Issues
```python
# Check dataset splits exist
from dataset import dicLoader
dicTrain, dicVal, dicTest = dicLoader(root="/media/d25u2/Dont/Viscosity")

# Inspect single fluid's data
from dataset import DaughterFolderDataset
ds = DaughterFolderDataset(["/path/to/experiment"], seq_len=10, stride=5)
images, viscosity, srof = ds[0]
print(f"Images: {images.shape}, Viscosity: {viscosity}, SROF: {srof.shape}")
```

### Checkpoint Locations
- AutoEncoder: `Output/checkpoints/AE_CNN/W Reflection/*.pt`
- LSTM: `Output/checkpoints/LSTM/AE_CNN_LSTM_*/*.ckpt`
- Naming: `{model_name}_epoch_{epoch}.ckpt` or `early_stop_*.ckpt`

## Project-Specific Conventions

1. **Random seed**: Always 42 (`utils.set_randomness(42)`)
2. **File extensions**: Images default to `.png` (configurable in `config.yaml`)
3. **Dataset splits**: Pre-generated pickles in `dataset/dataset_splits/`, not dynamic
4. **Viscosity labels**: Extracted from directory names (last underscore-separated value)
5. **Sequence padding**: Not used; datasets pre-filter to valid sequence lengths
6. **Real-time plotting**: `deeplearning.RealTimePlotter` for loss curves during training
7. **PDB integration**: Breakpoints work in both debug mode and normal execution

## Critical Files to Understand

- `config.yaml`: Single source of truth for all hyperparameters
- `dataset/header.py`: All custom types and data structures
- `dataset/MotherFolderDataset.py`: Main dataset orchestrator, caching logic
- `deeplearning/Base.py`: Training loop with signal handling and GPU monitoring
- `networks/AutoEncoder_CNN_LSTM.py`: LSTM with frozen encoder integration

## Pitfalls & Gotchas

- **Always call `dataset.MotherFolderDataset.load_cache()` if cache exists** - regenerating from scratch is 60x slower
- **LSTM state reset**: Must call `model.lstm.reset_states(batch)` before each batch
- **AutoEncoder loading**: In LSTM training, encoder is frozen (`eval()` mode, `torch.no_grad()`)
- **CSV alignment**: Files in `frames_rotated/` must align with `detections.csv` and `result.csv` indices
- **Reflection removal**: Only apply if `config['reflection'] = 'on'` - affects model input distribution
- **Positional encoding**: Mutually exclusive modes - can't enable both position and velocity encoding
