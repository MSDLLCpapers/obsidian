"""Shared pytest fixtures for obsidian tests"""

import pytest
import torch
from obsidian.config import TORCH_DTYPE


@pytest.fixture(autouse=True)
def set_default_dtype():
    torch.set_default_dtype(TORCH_DTYPE)
    yield
