# Deep Learning Pipeline for Viscosity Prediction from Droplet Dynamics

**Author:** Yassin Riyazi  
**Date:** November 2025  
**Repository:** [From-Droplet-Dynamics-to-Viscosity](https://github.com/YassinRiyazi/From-Droplet-Dynamics-to-Viscosity)

---

## Abstract

This report presents a comprehensive deep learning pipeline for predicting fluid viscosity from high-speed video footage of droplet dynamics. The proposed methodology employs a two-phase training approach: (1) a Convolutional Neural Network (CNN)-based Autoencoder for extracting compact latent representations from droplet images, and (2) temporal sequence models (LSTM and Transformer architectures) for regression-based viscosity prediction. The pipeline integrates experimental metadata features (4S-SROF: time, position, contact angles, velocity) with learned visual embeddings to achieve robust viscosity estimation. This work demonstrates the efficacy of combining self-supervised representation learning with supervised temporal modeling for rheological property inference from visual data.

---

## 1. Introduction

### 1.1 Problem Statement

Determining fluid viscosity through traditional rheometric methods requires specialized equipment and controlled laboratory conditions. Visual analysis of droplet behavior on inclined surfaces offers a non-invasive alternative, where the dynamic characteristics of sliding droplets correlate with the underlying fluid properties. However, extracting meaningful features from high-dimensional video data presents significant computational and methodological challenges.

### 1.2 Objectives

This project aims to:

1. Develop an end-to-end deep learning pipeline for viscosity prediction from droplet imagery
2. Learn compact, discriminative embeddings from raw image frames using self-supervised learning
3. Model temporal dependencies in droplet motion sequences using recurrent and attention-based architectures
4. Integrate physics-informed features (4S-SROF) with learned visual representations

---

## 2. Dataset Description

### 2.1 Data Acquisition

The experimental dataset comprises high-speed video recordings of droplet dynamics on tilted surfaces. Data is organized hierarchically:

```
/media/roboprocessing/Data/Viscosity/
├── {tilt_angle}/              # e.g., 280°, 285°, 290°
│   └── {fluid_name}/          # e.g., S3-SNr2.6_D
│       └── {experiment}/      # e.g., T528_01_0.017002
│           ├── frames_rotated/    # Full-frame images
│           ├── databases/         # Cropped droplet regions
│           ├── detections.csv     # Droplet location tracking
│           └── result.csv         # 4S-SROF features
```

### 2.2 Feature Descriptions

#### 2.2.1 Visual Features

- **Input Images:** Grayscale frames of dimensions $201 \times 201$ pixels (cropped) or $1280 \times 512$ (full-frame)
- **Preprocessing Options:**
  - Light reflection removal
  - Positional encoding (position-based or velocity-based)
  - Super-resolution enhancement (factor of 3×)

#### 2.2.2 4S-SROF Features (Four-Stage Surface Roughness Optimization Features)

| Feature | Description | Unit |
|---------|-------------|------|
| `time` | Timestamp of frame | seconds |
| `x_center`, `y_center` | Droplet centroid position | cm |
| `adv`, `rec` | Advancing and receding contact angles | degrees |
| `middle_angle_degree` | Middle contact angle | degrees |
| `contact_line_length` | Contact line width | cm |
| `velocity` | Instantaneous droplet velocity | cm/s |

### 2.3 Dataset Splitting and Loading

The dataset employs a two-tier loading architecture:

1. **MotherFolderDataset:** Orchestrates multiple fluid experiments, manages train/validation/test splits
2. **DaughterFolderDataset:** Handles individual fluid experiments, returns `(images, viscosity, SROF_features)` tuples

**Key Parameters:**
- `stride`: Step size for sequence extraction (default: 1-10)
- `sequence_length`: Number of consecutive frames per sample (default: 1-10)
- Dataset caching enabled for rapid loading (~2-5 seconds vs. 2-5 minutes)

---

## 3. Network Architecture

### 3.1 Phase 1: CNN Autoencoder

The autoencoder follows an encoder-decoder paradigm for self-supervised representation learning.

#### 3.1.1 Encoder Architecture

```
Input: (batch, 1, 201, 201)
    ↓
Conv2d(1→16, k=3, s=2, p=1) + ReLU    → (batch, 16, 100, 100)
    ↓
Conv2d(16→32, k=3, s=2, p=1) + ReLU   → (batch, 32, 50, 50)
    ↓
Conv2d(32→64, k=3, s=2, p=1) + ReLU   → (batch, 64, 25, 25)
    ↓
Conv2d(64→128, k=3, s=2, p=1) + ReLU  → (batch, 128, 13, 13)
    ↓
Flatten + Linear(128×13×13 → d_emb)   → (batch, d_emb)
```

Where $d_{emb} \in \{128, 1024\}$ represents the embedding dimension.

#### 3.1.2 Decoder Architecture

```
Input: (batch, d_emb)
    ↓
Linear(d_emb → 128×13×13) + Reshape   → (batch, 128, 13, 13)
    ↓
ConvTranspose2d(128→64, k=3, s=2) + ReLU  → (batch, 64, 25, 25)
    ↓
ConvTranspose2d(64→32, k=3, s=2) + ReLU   → (batch, 32, 50, 50)
    ↓
ConvTranspose2d(32→16, k=3, s=2) + ReLU   → (batch, 16, 100, 100)
    ↓
ConvTranspose2d(16→1, k=3, s=2) + Sigmoid → (batch, 1, 201, 201)
```

#### 3.1.3 Training Configuration

| Parameter | Value |
|-----------|-------|
| Loss Function | MSE Loss |
| Optimizer | AdamW |
| Learning Rate | $1 \times 10^{-4}$ |
| Weight Decay | $1 \times 10^{-5}$ |
| Batch Size | 16 |
| Dropout | 0.45 (optional) |
| Epochs | 50 |

### 3.2 Phase 2: LSTM Temporal Model

The LSTM model processes sequences of autoencoder embeddings to predict viscosity.

#### 3.2.1 Architecture

```
Input: (batch, seq_len, d_emb)
    ↓
LayerNorm(d_emb)                      → (batch, seq_len, d_emb)
    ↓
LSTM(d_emb → hidden_dim, num_layers)  → (batch, seq_len, hidden_dim)
    ↓
Mean Pooling (temporal axis)          → (batch, hidden_dim)
    ↓
Linear(hidden_dim → 1) + Sigmoid      → (batch, 1)
```

#### 3.2.2 Training Configuration

| Parameter | Value |
|-----------|-------|
| Hidden Dimension | 256 |
| Number of Layers | 2 |
| Dropout | 0.1 |
| Loss Function | MSE Loss |
| Optimizer | AdamW |
| Learning Rate | $1 \times 10^{-4}$ |

**Note:** The CNN encoder is frozen during LSTM training to prevent representation drift.

### 3.3 Phase 2 (Alternative): Transformer Temporal Model

An attention-based alternative to LSTM for capturing long-range temporal dependencies.

#### 3.3.1 Architecture

```
Input: (batch, seq_len, input_dim)
    ↓
Linear Projection(input_dim → d_model)    → (batch, seq_len, d_model)
    ↓
Sinusoidal Positional Encoding            → (batch, seq_len, d_model)
    ↓
N × TransformerEncoderBlock:
    ├─ Multi-Head Self-Attention (nhead=8)
    ├─ Layer Normalization
    ├─ Feed-Forward Network (GELU activation)
    └─ Residual Connections
    ↓
Global Average Pooling                    → (batch, d_model)
    ↓
MLP Head: Linear → GELU → Dropout → Linear → Sigmoid
    ↓
Output: (batch, 1)
```

#### 3.3.2 Training Configuration

| Parameter | Value |
|-----------|-------|
| Model Dimension ($d_{model}$) | 256 |
| Number of Heads | 8 |
| Number of Layers | 8 |
| Feed-Forward Dimension | $4 \times d_{model}$ |
| Dropout | 0.1 |
| Loss Function | MSE Loss |
| Optimizer | AdamW |
| Learning Rate | $1 \times 10^{-4}$ |

---

## 4. Training Methodology

### 4.1 Self-Supervised Pretraining (Autoencoder)

The autoencoder is trained using reconstruction loss:

$$\mathcal{L}_{AE} = \frac{1}{N} \sum_{i=1}^{N} \|x_i - \hat{x}_i\|_2^2$$

where $x_i$ is the input image and $\hat{x}_i$ is the reconstruction.

**Training Script:** `Nphase4_1_train_AutoEncoderCNN.py`

### 4.2 Supervised Regression (LSTM/Transformer)

The temporal models are trained to minimize viscosity prediction error:

$$\mathcal{L}_{reg} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

where $y_i \in [0, 1]$ is the normalized viscosity label.

**Training Scripts:**
- LSTM: `Train_LSTM_4SSROF.py`, `Nphase4_2_train_AutoEncoderCNN_EncoderLSTM.py`
- Transformer: `Train_Transformer_4SSROF.py`

### 4.3 Training Control Mechanisms

#### 4.3.1 Signal-Based Control

The training loop responds to Unix signals for dynamic adjustment:

| Signal | Action |
|--------|--------|
| `SIGUSR1` | Reduce learning rate by factor of 5 |
| `SIGUSR2` | Toggle dropout on/off |
| `Ctrl+D` | Save model and exit gracefully |

**Usage:** `kill -USR1 <pid>` (obtain PID from `nvidia-smi`)

#### 4.3.2 GPU Thermal Management

Training automatically pauses when GPU temperature exceeds 67°C, resuming after a configurable cooldown period.

#### 4.3.3 Early Stopping

Training terminates if:
- Validation loss fails to improve for 2 consecutive epochs (triggers learning rate reduction)
- Learning rate falls below $1 \times 10^{-7}$

### 4.4 Data Augmentation and Preprocessing

| Technique | Description |
|-----------|-------------|
| Light Reflection Removal | Morphological filtering to remove specular highlights |
| Positional Encoding | Sinusoidal encoding of droplet position or velocity |
| Normalization | Per-feature z-score normalization using training set statistics |
| Grayscale Conversion | All images converted to single-channel |

---

## 5. Feature Engineering

### 5.1 Combined Feature Representation

The final input to temporal models combines:

1. **Visual Embeddings:** $\mathbf{e}_t \in \mathbb{R}^{d_{emb}}$ from frozen autoencoder
2. **4S-SROF Features:** $\mathbf{s}_t \in \mathbb{R}^{8}$ (normalized)
3. **Experimental Metadata:** Tilt angle (normalized by 90°), frame count (normalized by 5000)

$$\mathbf{x}_t = [\mathbf{e}_t \| \mathbf{s}_t \| \theta_{tilt} \| n_{frame}]$$

### 5.2 Normalization Strategy

SROF features are globally normalized using training set statistics:

$$\hat{s}_{t,j} = \frac{s_{t,j} - \mu_j}{\sigma_j}$$

where $\mu_j$ and $\sigma_j$ are computed across all training sequences.

---

## 6. Implementation Details

### 6.1 Software Environment

- **Framework:** PyTorch
- **Python Version:** 3.11+
- **Key Dependencies:** torchvision, pandas, numpy, tqdm, colorama, OpenCV

### 6.2 Hardware Requirements

- **GPU:** NVIDIA GPU with CUDA support (recommended: ≥8GB VRAM)
- **RAM:** ≥16GB for dataset caching
- **Storage:** SSD recommended for fast data loading

### 6.3 Reproducibility

- **Random Seed:** 42 (applied to PyTorch, NumPy, and Python random)
- **CUDA Determinism:** `torch.cuda.manual_seed_all(42)`
- **Matrix Precision:** `torch.set_float32_matmul_precision('medium')`

### 6.4 Checkpoint Management

| Model | Save Location |
|-------|---------------|
| Autoencoder | `Output/checkpoints/AE_CNN/{encoding}_Ref={bool}_s{stride}_w{window}/` |
| LSTM | `Output/checkpoints/LSTM/LSTM_HD{hidden}_s{stride}_w{window}_{features}/` |
| Transformer | `Output/checkpoints/Transformer/Transformer_DM{d_model}_NH{nhead}_NL{layers}_*` |

---

## 7. Experimental Configurations

### 7.1 Embedding Variants

The pipeline supports multiple embedding configurations:

| Configuration | Embedding Dim | Positional Encoding | Reflection Removal |
|---------------|---------------|---------------------|-------------------|
| `CNNV1_0_128_False_Ref=False` | 128 | None | No |
| `CNNV1_0_1024_False_Ref=False` | 1024 | None | No |
| `CNNV1_0_128_Position_Ref=False` | 128 | Position-based | No |
| `CNNV1_0_1024_Velocity_Ref=False` | 1024 | Velocity-based | No |

### 7.2 Feature Selection Configurations

Multiple configurations available via YAML files in `Configs/`:

| Config File | Features Included |
|-------------|-------------------|
| `SROF+tilt.yaml` | 4S-SROF + Tilt angle |
| `SROF+tilt+count.yaml` | 4S-SROF + Tilt angle + Frame count |

---

## 8. Usage Instructions

### 8.1 Training Pipeline

```bash
# Step 1: Train Autoencoder (self-supervised)
python Nphase4_1_train_AutoEncoderCNN.py

# Step 2a: Train LSTM with frozen encoder
python Train_LSTM_4SSROF.py

# Step 2b: Alternative - Train Transformer
python Train_Transformer_4SSROF.py

# Step 3: Evaluate and visualize results
python Nphase4_3_TestPlot.py
```

### 8.2 Configuration

All hyperparameters are centralized in `config.yaml`:

```yaml
Dataset:
  Dataset_Root: '/media/roboprocessing/Data/Viscosity'
  reflection_removal: False
  embedding:
    positional_encoding: 'False'  # 'Position' or 'Velocity'

Training:
  batch_size: 16
  num_epochs: 50
  Constant_feature_AE:
    valid_latent_dim: [128, 1024]
  Constant_feature_LSTM:
    Hidden_size: 256
    Num_layers: 2
```

### 8.3 Dataset Caching

```python
from dataset import MotherFolderDataset, dicLoader

cache_path = "Output/dataset_cache_train.pkl"
if os.path.exists(cache_path):
    dataset = MotherFolderDataset.load_cache(cache_path)  # Fast: ~2-5s
else:
    dicAddresses, _, _ = dicLoader(root=config['Dataset']['Dataset_Root'])
    dataset = MotherFolderDataset(dicAddresses, stride=10, sequence_length=10)
    dataset.save_cache(cache_path)  # Slow initial run: 2-5min
```

---

## 9. Conclusion

This report documents a comprehensive deep learning pipeline for inferring fluid viscosity from droplet dynamics videos. The two-phase approach—self-supervised visual embedding followed by supervised temporal regression—provides a principled framework for combining learned representations with domain-specific features. The modular architecture supports flexible experimentation with different embedding dimensions, temporal models (LSTM vs. Transformer), and feature configurations.

---

## References

1. PyTorch Documentation: https://pytorch.org/docs/stable/
2. Autoencoder architectures for visual representation learning
3. LSTM networks for sequence modeling
4. Transformer architecture: "Attention Is All You Need" (Vaswani et al., 2017)

---

## Appendix A: Model Parameter Counts

| Model | Parameters |
|-------|------------|
| Autoencoder (128-dim) | ~2.1M |
| Autoencoder (1024-dim) | ~17.6M |
| LSTM (256 hidden, 2 layers) | ~1.3M |
| Transformer (256 d_model, 8 layers) | ~6.7M |

## Appendix B: Directory Structure

```
From-Droplet-Dynamics-to-Viscosity/
├── config.yaml                          # Main configuration
├── data_config.yaml                     # Data-specific settings
├── Nphase4_1_train_AutoEncoderCNN.py    # AE training
├── Train_LSTM_4SSROF.py                 # LSTM training
├── Train_Transformer_4SSROF.py          # Transformer training
├── dataset/                             # Dataset utilities
├── networks/                            # Model architectures
├── deeplearning/                        # Training infrastructure
├── utils/                               # Utility functions
└── Output/                              # Checkpoints and logs
```
