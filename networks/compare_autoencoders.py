"""
Comparison between CNN and Transformer Autoencoders

This script compares the two autoencoder architectures on:
- Model size (parameters)
- Memory usage
- Speed (throughput and latency)
- Reconstruction quality (untrained baseline)
- Embedding dimensions
"""

import torch
import torch.nn as nn
import time
from typing import Dict, Tuple

# Import both autoencoders
from networks.AutoEncoder_CNNV1_0 import Autoencoder_CNN
from networks.AutoEncoder_TransformerV2_0 import create_autoencoder, PRESET_CONFIGS


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_memory(model: nn.Module, x: torch.Tensor, device: torch.device) -> float:
    """Measure peak GPU memory usage in GB"""
    if not torch.cuda.is_available():
        return 0.0
    
    model = model.to(device)
    x = x.to(device)
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        _ = model(x)
    
    max_memory = torch.cuda.max_memory_allocated() / 1e9
    return max_memory


def measure_speed(model: nn.Module, x: torch.Tensor, device: torch.device, 
                  num_iterations: int = 100) -> Tuple[float, float]:
    """Measure throughput (images/sec) and latency (ms/batch)"""
    model = model.to(device)
    x = x.to(device)
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    
    # Benchmark
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(x)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    
    throughput = (num_iterations * x.size(0)) / elapsed
    latency = elapsed / num_iterations * 1000
    
    return throughput, latency


def measure_reconstruction_quality(model: nn.Module, x: torch.Tensor, 
                                   device: torch.device) -> Dict[str, float]:
    """Measure reconstruction quality metrics (MSE, MAE)"""
    model = model.to(device)
    x = x.to(device)
    model.eval()
    
    with torch.no_grad():
        # Handle different return signatures
        output = model(x)
        if isinstance(output, tuple):
            recon = output[0]
        else:
            recon = output
    
    mse = nn.MSELoss()(recon, x).item()
    mae = nn.L1Loss()(recon, x).item()
    
    return {'mse': mse, 'mae': mae}


def compare_models():
    """Compare CNN and Transformer autoencoders"""
    print("\n" + "="*100)
    print("AUTOENCODER COMPARISON: CNN vs Transformer")
    print("="*100)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Test configuration
    input_size = (201, 201)
    batch_size = 16
    embedding_dim = 512
    
    print(f"\nTest Configuration:")
    print(f"  Input size: {input_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Target embedding dimension: {embedding_dim}")
    
    # Create test input
    x = torch.randn(batch_size, 1, *input_size)
    
    # Initialize models
    print(f"\n{'-'*100}")
    print("INITIALIZING MODELS")
    print(f"{'-'*100}")
    
    # CNN Autoencoder
    cnn_model = Autoencoder_CNN(DropOut=False, embedding_dim=embedding_dim)
    print(f"✓ CNN Autoencoder created")
    
    # Transformer Autoencoder (test multiple presets)
    transformer_presets = ['small', 'medium']
    transformer_models = {}
    
    for preset in transformer_presets:
        config = PRESET_CONFIGS[preset].copy()
        config['latent_dim'] = embedding_dim  # Override to match CNN
        
        transformer_models[preset] = create_autoencoder(
            input_size=input_size,
            preset=preset,
            latent_dim=embedding_dim,
            use_flash_attention=True,
            use_gradient_checkpointing=False
        )
        print(f"✓ Transformer Autoencoder ({preset}) created")
    
    # Comparison table
    results = []
    
    # Evaluate CNN
    print(f"\n{'-'*100}")
    print("EVALUATING CNN AUTOENCODER")
    print(f"{'-'*100}")
    
    cnn_params_total, cnn_params_train = count_parameters(cnn_model)
    cnn_memory = measure_memory(cnn_model, x, device)
    cnn_throughput, cnn_latency = measure_speed(cnn_model, x, device)
    cnn_quality = measure_reconstruction_quality(cnn_model, x, device)
    
    # Test embedding extraction
    cnn_model_device = cnn_model.to(device)
    with torch.no_grad():
        cnn_embedding = cnn_model_device.Embedding(x.to(device))
    
    results.append({
        'Model': 'CNN',
        'Preset': 'N/A',
        'Parameters': cnn_params_total,
        'Memory (GB)': cnn_memory,
        'Throughput (img/s)': cnn_throughput,
        'Latency (ms)': cnn_latency,
        'MSE': cnn_quality['mse'],
        'MAE': cnn_quality['mae'],
        'Embedding Shape': str(cnn_embedding.shape),
    })
    
    print(f"  Parameters: {cnn_params_total:,}")
    print(f"  Memory: {cnn_memory:.3f} GB")
    print(f"  Throughput: {cnn_throughput:.2f} images/sec")
    print(f"  Latency: {cnn_latency:.2f} ms/batch")
    print(f"  MSE: {cnn_quality['mse']:.6f}")
    print(f"  Embedding: {cnn_embedding.shape}")
    
    # Evaluate Transformers
    for preset, transformer_model in transformer_models.items():
        print(f"\n{'-'*100}")
        print(f"EVALUATING TRANSFORMER AUTOENCODER ({preset.upper()})")
        print(f"{'-'*100}")
        
        trans_params_total, trans_params_train = count_parameters(transformer_model)
        trans_memory = measure_memory(transformer_model, x, device)
        trans_throughput, trans_latency = measure_speed(transformer_model, x, device)
        trans_quality = measure_reconstruction_quality(transformer_model, x, device)
        
        # Test embedding extraction
        transformer_model_device = transformer_model.to(device)
        with torch.no_grad():
            trans_embedding = transformer_model_device.Embedding(x.to(device))
        
        results.append({
            'Model': 'Transformer',
            'Preset': preset,
            'Parameters': trans_params_total,
            'Memory (GB)': trans_memory,
            'Throughput (img/s)': trans_throughput,
            'Latency (ms)': trans_latency,
            'MSE': trans_quality['mse'],
            'MAE': trans_quality['mae'],
            'Embedding Shape': str(trans_embedding.shape),
        })
        
        print(f"  Parameters: {trans_params_total:,}")
        print(f"  Memory: {trans_memory:.3f} GB")
        print(f"  Throughput: {trans_throughput:.2f} images/sec")
        print(f"  Latency: {trans_latency:.2f} ms/batch")
        print(f"  MSE: {trans_quality['mse']:.6f}")
        print(f"  Embedding: {trans_embedding.shape}")
    
    # Print summary table
    print(f"\n{'='*100}")
    print("COMPARISON SUMMARY")
    print(f"{'='*100}\n")
    
    # Header
    header = f"{'Model':<12} {'Preset':<8} {'Params':<12} {'Memory':<12} {'Throughput':<15} {'Latency':<12} {'MSE':<12}"
    print(header)
    print("-" * len(header))
    
    # Rows
    for result in results:
        row = (f"{result['Model']:<12} "
               f"{result['Preset']:<8} "
               f"{result['Parameters']:>10,}  "
               f"{result['Memory (GB)']:>8.3f} GB  "
               f"{result['Throughput (img/s)']:>11.2f} /s  "
               f"{result['Latency (ms)']:>8.2f} ms  "
               f"{result['MSE']:>10.6f}")
        print(row)
    
    # Speedup comparison
    print(f"\n{'-'*100}")
    print("RELATIVE PERFORMANCE (vs CNN)")
    print(f"{'-'*100}\n")
    
    cnn_result = results[0]
    
    for i, result in enumerate(results):
        if i == 0:
            continue  # Skip CNN (baseline)
        
        param_ratio = result['Parameters'] / cnn_result['Parameters']
        memory_ratio = result['Memory (GB)'] / cnn_result['Memory (GB)'] if cnn_result['Memory (GB)'] > 0 else 0
        speed_ratio = result['Throughput (img/s)'] / cnn_result['Throughput (img/s)']
        
        print(f"{result['Model']} ({result['Preset']}):")
        print(f"  Parameters: {param_ratio:.2f}x {'more' if param_ratio > 1 else 'fewer'}")
        if memory_ratio > 0:
            print(f"  Memory: {memory_ratio:.2f}x {'more' if memory_ratio > 1 else 'less'}")
        print(f"  Speed: {speed_ratio:.2f}x {'faster' if speed_ratio > 1 else 'slower'}")
        print()
    
    # Recommendations
    print(f"{'='*100}")
    print("RECOMMENDATIONS")
    print(f"{'='*100}\n")
    
    print("Choose CNN Autoencoder if:")
    print("  • You need maximum speed and efficiency")
    print("  • Working with limited GPU memory (< 4GB)")
    print("  • Dataset is small (< 10k images)")
    print("  • Fixed input size (201x201)")
    print("  • Simple feature extraction is sufficient")
    
    print("\nChoose Transformer Autoencoder if:")
    print("  • You need attention visualization")
    print("  • Working with large datasets (> 50k images)")
    print("  • Variable input sizes (1280x152, 201x201, etc.)")
    print("  • Capturing long-range dependencies is important")
    print("  • Have sufficient GPU memory (>= 6GB)")
    print("  • Configurable latent dimension is needed (128-8192)")
    
    print("\nPreset Recommendations:")
    print("  • tiny: Quick prototyping, small datasets")
    print("  • small: Good balance for most tasks, 4-6GB VRAM")
    print("  • medium: Better performance, 6-8GB VRAM")
    print("  • large: Research/production, >8GB VRAM")
    
    print(f"\n{'='*100}\n")


def compare_different_input_sizes():
    """Compare performance on different input sizes"""
    print("\n" + "="*100)
    print("INPUT SIZE COMPARISON")
    print("="*100)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_configs = [
        ((201, 201), 16, "Square (201x201)"),
        ((152, 640), 8, "Wide (640x152)"),
        ((152, 1280), 4, "Very Wide (1280x152)"),
    ]
    
    print(f"\nComparing Transformer performance on different input sizes:")
    print(f"Using 'small' preset with latent_dim=256\n")
    
    print(f"{'Input Size':<20} {'Batch':<8} {'Params':<12} {'Memory':<12} {'Throughput':<15} {'Latency':<12}")
    print("-" * 90)
    
    for input_size, batch_size, description in test_configs:
        model = create_autoencoder(
            input_size=input_size,
            preset='small',
            latent_dim=256,
            use_flash_attention=True
        )
        
        x = torch.randn(batch_size, 1, *input_size)
        
        params, _ = count_parameters(model)
        memory = measure_memory(model, x, device)
        throughput, latency = measure_speed(model, x, device, num_iterations=50)
        
        print(f"{description:<20} {batch_size:<8} {params:>10,}  {memory:>8.3f} GB  "
              f"{throughput:>11.2f} /s  {latency:>8.2f} ms")
    
    print("\nNote: Input size affects computation but not number of parameters.")
    print("Larger images require more memory and are slower to process.\n")


if __name__ == "__main__":
    import torch
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Run comparisons
    compare_models()
    compare_different_input_sizes()
    
    print("Comparison completed!")
