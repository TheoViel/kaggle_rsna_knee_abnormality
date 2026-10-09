"""Small inference helpers."""

import torch


def amp_dtype():
    """bfloat16 where the GPU has it (A100 / H100), float16 otherwise (T4, P100)."""
    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
