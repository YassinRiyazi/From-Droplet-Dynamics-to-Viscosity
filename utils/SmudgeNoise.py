"""
Utilities for generating smudge-like noise tensors and adding them to
normalized droplet images. Images are assumed to use the standard
(C, H, W) layout and contain pixel values in [0, 1].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class SmudgeConfig:
    """Configuration container for smudge noise generation."""

    image_size: Tuple[int, int] = (201, 201)
    num_smudges: int = 4
    size_range: Tuple[float, float] = (6.0, 32.0)
    intensity_range: Tuple[float, float] = (-0.2, 0.2)
    softness: float = 0.35
    per_sample: bool = True
    clamp_range: Tuple[float, float] = (0.0, 1.0)

    def to_kwargs(self) -> dict[str, object]:
        return {
            "image_size": self.image_size,
            "num_smudges": self.num_smudges,
            "size_range": self.size_range,
            "intensity_range": self.intensity_range,
            "softness": self.softness,
        }


def _normalize_shape(image_size: int | Sequence[int]) -> Tuple[int, int]:
    """Return (H, W) given a scalar or iterable specification."""

    if isinstance(image_size, int):
        return image_size, image_size
    if len(image_size) != 2:
        raise ValueError("image_size must be an int or length-2 sequence")
    height = int(image_size[0])
    width = int(image_size[1])
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    return height, width


def generate_smudge_mask(
    image_size: int | Sequence[int] = (201, 201),
    *,
    num_smudges: int = 4,
    size_range: Sequence[float] = (6.0, 32.0),
    intensity_range: Sequence[float] = (-0.2, 0.2),
    softness: float = 0.35,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Create a single-channel smudge mask that can be added to an image."""

    height, width = _normalize_shape(image_size)
    if num_smudges <= 0:
        return torch.zeros(1, height, width, device=device, dtype=dtype or torch.float32)

    low_size, high_size = size_range
    low_intensity, high_intensity = intensity_range
    if low_size <= 0 or high_size <= 0:
        raise ValueError("size_range entries must be positive")
    if high_size < low_size:
        raise ValueError("size_range must be non-decreasing")
    if high_intensity == 0 and low_intensity == 0:
        raise ValueError("intensity_range cannot be identically zero")
    softness = max(1e-3, softness)

    device = device or torch.device("cpu")
    dtype = dtype or torch.float32
    yy = torch.linspace(0, height - 1, height, device=device, dtype=dtype).unsqueeze(1)
    xx = torch.linspace(0, width - 1, width, device=device, dtype=dtype).unsqueeze(0)
    mask = torch.zeros((height, width), device=device, dtype=dtype)

    for _ in range(num_smudges):
        radius_y = torch.empty((), device=device, dtype=dtype).uniform_(low_size, high_size)
        radius_x = torch.empty((), device=device, dtype=dtype).uniform_(low_size, high_size)
        center_y = torch.empty((), device=device, dtype=dtype).uniform_(0, height - 1)
        center_x = torch.empty((), device=device, dtype=dtype).uniform_(0, width - 1)
        intensity = torch.empty((), device=device, dtype=dtype).uniform_(low_intensity, high_intensity)
        if torch.isclose(intensity, torch.tensor(0.0, device=device, dtype=dtype)):
            continue

        gaussian = torch.exp(
            -(((yy - center_y) / radius_y) ** 2 + ((xx - center_x) / radius_x) ** 2)
            / (2 * softness)
        )
        mask.add_(intensity * gaussian)

    return mask.clamp(-1.0, 1.0).unsqueeze(0)


def _ensure_channel_dim(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    if tensor.ndim == 3:
        return tensor
    raise ValueError("Expected tensor with 2 or 3 dims (C, H, W)")


def _sanitize_generator_kwargs(generator_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Convert free-form kwargs into strongly typed parameters."""

    sanitized: dict[str, Any] = {}
    if "num_smudges" in generator_kwargs:
        sanitized["num_smudges"] = int(generator_kwargs["num_smudges"])
    if "size_range" in generator_kwargs:
        size_range = generator_kwargs["size_range"]
        try:
            left, right = size_range  # type: ignore[misc]
        except (TypeError, ValueError):
            pass
        else:
            sanitized["size_range"] = (float(left), float(right))
    if "intensity_range" in generator_kwargs:
        int_range = generator_kwargs["intensity_range"]
        try:
            left, right = int_range  # type: ignore[misc]
        except (TypeError, ValueError):
            pass
        else:
            sanitized["intensity_range"] = (float(left), float(right))
    if "softness" in generator_kwargs:
        sanitized["softness"] = float(generator_kwargs["softness"])
    return sanitized


def add_smudge_noise(
    images: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    clamp_range: Optional[Tuple[float, float]] = (0.0, 1.0),
    per_sample: bool = True,
    generator_kwargs: Optional[dict[str, Any]] = None,
) -> torch.Tensor:
    """Add smudge noise to image tensors (B, C, H, W) or (C, H, W)."""

    generator_kwargs = generator_kwargs or {}
    sanitized_kwargs = _sanitize_generator_kwargs(generator_kwargs)

    if images.ndim == 3:
        mask = mask or generate_smudge_mask(
            image_size=images.shape[-2:],
            device=images.device,
            dtype=images.dtype,
            **sanitized_kwargs,
        )
        result = images + mask
    elif images.ndim == 4:
        batch, _, height, width = images.shape
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(0)
            if mask.shape[1:] != (1, height, width):
                raise ValueError("mask shape must match (1, H, W)")
            mask = mask.expand(batch, -1, -1, -1)
        else:
            masks: list[torch.Tensor] = []
            for _ in range(batch if per_sample else 1):
                masks.append(
                    generate_smudge_mask(
                        image_size=(height, width),
                        device=images.device,
                        dtype=images.dtype,
                        **sanitized_kwargs,
                    )
                )
            mask = torch.stack(masks, dim=0)
            if not per_sample:
                mask = mask.expand(batch, -1, -1, -1)
        result = images + mask
    else:
        raise ValueError("images must be 3D or 4D tensor")

    if clamp_range is not None:
        low, high = clamp_range
        result = result.clamp(low, high)
    return result


def visualize_smudge(mask: torch.Tensor, *, title: str | None = None) -> None:
    """Display a smudge mask using matplotlib."""

    import matplotlib.pyplot as _plt  # type: ignore

    plt: Any = _plt

    mask = _ensure_channel_dim(mask).squeeze(0).detach().cpu()
    plt.imshow(mask, cmap="magma")
    plt.colorbar(label="Intensity")
    plt.title(title or "Smudge mask")
    plt.axis("off")
    plt.show()


def visualize_smudge_gallery(
    num_examples: int = 4,
    *,
    image_size: int | Sequence[int] = (201, 201),
    cols: int = 2,
    **generator_kwargs: object,
) -> None:
    """Render multiple smudge masks for visual inspection."""

    import math
    import numpy as np
    import matplotlib.pyplot as _plt  # type: ignore

    plt: Any = _plt
    sanitized_kwargs = _sanitize_generator_kwargs(dict(generator_kwargs))

    rows = math.ceil(num_examples / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    fig.tight_layout(rect=(0, 0, 1, 1))

    axes_array = np.array(axes, dtype=object).reshape(-1)
    for idx, ax in enumerate(axes_array):
        if idx < num_examples:
            mask = generate_smudge_mask(image_size=image_size, **sanitized_kwargs)
            ax.imshow(mask.squeeze(0).cpu(), cmap="magma")
            ax.axis("off")
        else:
            ax.axis("off")

    plt.show()
