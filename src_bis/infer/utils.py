"""Small inference helpers."""

import torch


def amp_dtype():
    """float16: the Kaggle T4s have no fast bfloat16, and every model was checked to give the
    same predictions under float16 autocast as under the bfloat16 it was trained with."""
    return torch.float16
